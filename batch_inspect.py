"""
Shared batch-video inspection logic (used by the CLI scripts/video_inference.py
and the REST endpoint /api/upload_video). Turns a stream of per-frame detection
results into a localized, per-zone batch report plus a defect-location map.
"""
import numpy as np
import cv2


def frame_record(idx, score, is_defect, boxes, area_frac, fps, total, meters, segments):
    """Build one per-frame record with position (metres, constant-speed
    assumption) and zone index."""
    pos_frac = idx / max(1, (total - 1)) if total > 1 else 0.0
    zone = min(segments - 1, int(pos_frac * segments))
    return {
        'frame': idx, 'time_s': round(idx / fps, 2) if fps else 0.0,
        'position_m': round(pos_frac * meters, 3), 'zone': zone + 1,
        'score': round(float(score), 2), 'is_defect': bool(is_defect),
        'defect_area_frac': round(float(area_frac), 5), 'boxes': boxes or [],
    }


def build_report(frames_info, batch, meters, segments, fps, total, n_proc,
                 threshold, min_area_frac, source, mode, device):
    """Aggregate per-frame records into a per-zone + per-event batch report."""
    seg_defect = [0] * segments
    seg_maxscore = [0.0] * segments
    seg_area = [0.0] * segments
    for fi in frames_info:
        if fi['is_defect']:
            z = fi['zone'] - 1
            seg_defect[z] += 1
            seg_maxscore[z] = max(seg_maxscore[z], fi['score'])
            seg_area[z] = max(seg_area[z], fi['defect_area_frac'])

    # Contiguous defect runs -> events with a position range.
    events, run = [], None
    for fi in frames_info:
        if fi['is_defect']:
            if run is None:
                run = {'start_frame': fi['frame'], 'start_m': fi['position_m'],
                       'end_frame': fi['frame'], 'end_m': fi['position_m'],
                       'max_score': fi['score'], 'zones': {fi['zone']}}
            else:
                run['end_frame'] = fi['frame']; run['end_m'] = fi['position_m']
                run['max_score'] = max(run['max_score'], fi['score']); run['zones'].add(fi['zone'])
        elif run is not None:
            run['zones'] = sorted(run['zones']); events.append(run); run = None
    if run is not None:
        run['zones'] = sorted(run['zones']); events.append(run)

    defect_frames = sum(1 for f in frames_info if f['is_defect'])
    zones_with_defects = [i + 1 for i, c in enumerate(seg_defect) if c > 0]
    return {
        'batch': batch, 'source': source, 'mode': mode, 'device': str(device),
        'total_frames': total, 'processed_frames': n_proc, 'fps': round(fps, 2),
        'batch_length_m': meters, 'segments': segments,
        'min_defect_area_frac': min_area_frac, 'threshold': round(threshold, 3),
        'defect_frames': defect_frames,
        'defect_rate': round(defect_frames / max(1, n_proc), 3),
        'zones_with_defects': zones_with_defects,
        'segment_summary': [
            {'zone': i + 1,
             'position_m': f"{round(i/segments*meters,2)}-{round((i+1)/segments*meters,2)}",
             'defect_frames': seg_defect[i], 'max_score': round(seg_maxscore[i], 2),
             'max_defect_area_frac': round(seg_area[i], 4),
             'status': 'DEFECT' if seg_defect[i] > 0 else 'clear'}
            for i in range(segments)],
        'defect_events': [
            {'start_m': e['start_m'], 'end_m': e['end_m'],
             'frames': f"{e['start_frame']}-{e['end_frame']}",
             'zones': e['zones'], 'max_score': round(e['max_score'], 2)}
            for e in events],
    }


def annotate_frame(frame, rec, batch, segments, src_size=224):
    """Draw defect boxes, a batch/position banner, and a red border (if defect)
    onto ``frame`` in place. ``rec`` is a frame_record; boxes are in src_size
    (heatmap) coords and are scaled to the frame."""
    H, W = frame.shape[:2]
    sx, sy = W / float(src_size), H / float(src_size)
    for b in (rec.get('boxes') or []):
        x, y = int(b['x'] * sx), int(b['y'] * sy)
        w, h = int(b['w'] * sx), int(b['h'] * sy)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, f"DEFECT {b['area_frac']*100:.1f}%", (x, max(12, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    banner = (f"Batch {batch} | {rec['position_m']:.2f}m | zone {rec['zone']}/{segments} | "
              f"score {rec['score']:.1f} | {'DEFECT' if rec['is_defect'] else 'OK'}")
    cv2.rectangle(frame, (0, 0), (W, 18), (0, 0, 0), -1)
    cv2.putText(frame, banner, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                (0, 0, 255) if rec['is_defect'] else (0, 220, 0), 1)
    if rec['is_defect']:
        cv2.rectangle(frame, (0, 0), (W - 1, H - 1), (0, 0, 255), 3)
    return frame


def build_defect_map(frames_info, meters, segments, batch, threshold, defect_frames, n_proc):
    """Render a horizontal strip along the batch length, red where defective."""
    W, H = 1000, 140
    strip = np.full((H, W, 3), 245, np.uint8)
    for fi in frames_info:
        x = int(fi['position_m'] / max(1e-6, meters) * (W - 1))
        if fi['is_defect']:
            inten = min(1.0, fi['score'] / (threshold * 3 + 1e-6))
            cv2.line(strip, (x, 24), (x, H - 24), (0, 0, int(120 + 135 * inten)), 1)
    cv2.rectangle(strip, (0, 24), (W - 1, H - 24), (60, 60, 60), 1)
    for s in range(segments + 1):
        x = int(s / segments * (W - 1))
        cv2.line(strip, (x, 18), (x, H - 18), (150, 150, 150), 1)
        cv2.putText(strip, f"{round(s/segments*meters,1)}m", (max(0, x - 12), H - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (90, 90, 90), 1)
    cv2.putText(strip, f"Batch {batch} - defect map ({defect_frames}/{n_proc} frames flagged)",
                (8, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)
    return strip
