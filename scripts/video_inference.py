#!/usr/bin/env python3
"""
Batch-video defect inference with localization.
================================================
Runs the calibrated detector on every frame of a fabric-batch video (e.g. a
conveyor recording), and reports WHERE along the batch defects occur -- by
position (metres, assuming constant conveyor speed) and by segment/zone -- with
a minimum defect-size filter. Produces:

  * <out>/<batch>_annotated.mp4  - the video with defect boxes + batch overlay
  * <out>/<batch>_report.json    - per-frame + per-zone defect report
  * <out>/<batch>_defectmap.png  - a "where are the defects" strip along the batch

Usage:
    python scripts/video_inference.py --input batch.mp4 --batch B001 \
        --meters 5 --segments 10 --mode accurate --out demo_batches/out
"""
import os
import sys
import json
import argparse

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
from fabric_engine import InferenceEngine
from app_utils.config import config
import batch_inspect


def main():
    ap = argparse.ArgumentParser(description="Batch-video defect inference with localization")
    ap.add_argument('--input', required=True, help='batch video file (.mp4/.avi/...)')
    ap.add_argument('--batch', default='B001', help='batch number/label')
    ap.add_argument('--out', default='batch_out', help='output directory')
    ap.add_argument('--mode', choices=['accurate', 'fast'], default='accurate')
    ap.add_argument('--device', default=None)
    ap.add_argument('--stride', type=int, default=1, help='process every Nth frame')
    ap.add_argument('--meters', type=float, default=5.0, help='physical batch length (m)')
    ap.add_argument('--segments', type=int, default=10, help='number of report zones')
    ap.add_argument('--min-defect-area', type=float, default=None,
                    help='override min defect area fraction (default from config)')
    args = ap.parse_args()

    if args.min_defect_area is not None:
        config.min_defect_area_frac = args.min_defect_area

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"ERROR: cannot open {args.input}")
        sys.exit(1)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 224
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 224

    os.makedirs(args.out, exist_ok=True)
    engine = InferenceEngine(device=args.device, mode=args.mode, capture_attention=False)
    print(f"Batch {args.batch}: {total} frames @ {fps:.0f} fps on {engine.device} "
          f"[{args.mode}], min defect area {config.min_defect_area_frac*100:.2f}% of frame")

    ann_path = os.path.join(args.out, f"{args.batch}_annotated.mp4")
    vw = cv2.VideoWriter(ann_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))

    seg_defect = [0] * args.segments          # defect frame count per zone
    seg_maxscore = [0.0] * args.segments
    seg_area = [0.0] * args.segments
    frames_info = []                          # per processed frame
    defect_events = []                        # contiguous defect runs
    n_proc = 0
    idx = -1
    sx, sy = W / 224.0, H / 224.0             # scale boxes (heatmap 224) -> display

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        idx += 1
        if idx % args.stride != 0:
            vw.write(frame)
            continue
        n_proc += 1
        r = engine.predict_bgr(frame, mode=args.mode)
        pos_frac = idx / max(1, (total - 1)) if total > 1 else 0.0
        position_m = round(pos_frac * args.meters, 3)
        zone = min(args.segments - 1, int(pos_frac * args.segments))

        if r.is_defect:
            seg_defect[zone] += 1
            seg_maxscore[zone] = max(seg_maxscore[zone], r.score)
            seg_area[zone] = max(seg_area[zone], r.defect_area_frac)

        # Draw boxes (scale from 224 space) + overlay
        for b in (r.boxes or []):
            x, y, w, h = int(b['x'] * sx), int(b['y'] * sy), int(b['w'] * sx), int(b['h'] * sy)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, f"DEFECT {b['area_frac']*100:.1f}%", (x, max(12, y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        banner = f"Batch {args.batch} | {position_m:.2f}m | zone {zone+1}/{args.segments} | " \
                 f"score {r.score:.1f} | {'DEFECT' if r.is_defect else 'OK'}"
        cv2.rectangle(frame, (0, 0), (W, 18), (0, 0, 0), -1)
        cv2.putText(frame, banner, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (0, 0, 255) if r.is_defect else (0, 220, 0), 1)
        if r.is_defect:
            cv2.rectangle(frame, (0, 0), (W - 1, H - 1), (0, 0, 255), 3)
        vw.write(frame)

        frames_info.append({'frame': idx, 'time_s': round(idx / fps, 2),
                            'position_m': position_m, 'zone': zone + 1,
                            'score': round(r.score, 2), 'is_defect': r.is_defect,
                            'defect_area_frac': r.defect_area_frac, 'boxes': r.boxes or []})
    cap.release()
    vw.release()
    batch_inspect.to_browser_h264(ann_path)  # H.264 so it plays in a browser

    # Contiguous defect runs -> defect events with a position range
    run = None
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
            run['zones'] = sorted(run['zones']); defect_events.append(run); run = None
    if run is not None:
        run['zones'] = sorted(run['zones']); defect_events.append(run)

    defect_frames = sum(1 for f in frames_info if f['is_defect'])
    zones_with_defects = [i + 1 for i, c in enumerate(seg_defect) if c > 0]

    report = {
        'batch': args.batch, 'source': os.path.basename(args.input),
        'mode': args.mode, 'device': str(engine.device),
        'total_frames': total, 'processed_frames': n_proc, 'fps': round(fps, 2),
        'batch_length_m': args.meters, 'segments': args.segments,
        'min_defect_area_frac': config.min_defect_area_frac,
        'threshold': round(engine.threshold, 3),
        'defect_frames': defect_frames,
        'defect_rate': round(defect_frames / max(1, n_proc), 3),
        'zones_with_defects': zones_with_defects,
        'segment_summary': [
            {'zone': i + 1,
             'position_m': f"{round(i/args.segments*args.meters,2)}-{round((i+1)/args.segments*args.meters,2)}",
             'defect_frames': seg_defect[i], 'max_score': round(seg_maxscore[i], 2),
             'max_defect_area_frac': round(seg_area[i], 4),
             'status': 'DEFECT' if seg_defect[i] > 0 else 'clear'}
            for i in range(args.segments)],
        'defect_events': [
            {'start_m': e['start_m'], 'end_m': e['end_m'],
             'frames': f"{e['start_frame']}-{e['end_frame']}",
             'zones': e['zones'], 'max_score': round(e['max_score'], 2)}
            for e in defect_events],
    }
    rep_path = os.path.join(args.out, f"{args.batch}_report.json")
    with open(rep_path, 'w') as f:
        json.dump(report, f, indent=2)

    # Defect-map strip: horizontal bar along the batch, red where defective.
    stripW, stripH = 1000, 140
    strip = np.full((stripH, stripW, 3), 245, np.uint8)
    for fi in frames_info:
        x = int(fi['position_m'] / args.meters * (stripW - 1))
        if fi['is_defect']:
            inten = min(1.0, fi['score'] / (engine.threshold * 3 + 1e-6))
            cv2.line(strip, (x, 24), (x, stripH - 24), (0, 0, int(120 + 135 * inten)), 1)
    cv2.rectangle(strip, (0, 24), (stripW - 1, stripH - 24), (60, 60, 60), 1)
    for s in range(args.segments + 1):
        x = int(s / args.segments * (stripW - 1))
        cv2.line(strip, (x, 18), (x, stripH - 18), (150, 150, 150), 1)
        cv2.putText(strip, f"{round(s/args.segments*args.meters,1)}m", (max(0, x - 12), stripH - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (90, 90, 90), 1)
    cv2.putText(strip, f"Batch {args.batch} - defect map ({defect_frames}/{n_proc} frames flagged)",
                (8, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)
    map_path = os.path.join(args.out, f"{args.batch}_defectmap.png")
    cv2.imwrite(map_path, strip)

    # Console summary
    print(f"\n=== Batch {args.batch} inspection summary ===")
    print(f"  Defect frames: {defect_frames}/{n_proc}  ({report['defect_rate']*100:.1f}%)")
    if zones_with_defects:
        print(f"  Defective zones: {zones_with_defects} of {args.segments}")
        for e in report['defect_events']:
            print(f"    - defect at {e['start_m']}-{e['end_m']} m (zones {e['zones']}, score {e['max_score']})")
    else:
        print("  No defects above the size threshold.")
    print(f"\n  Annotated video: {ann_path}")
    print(f"  Report JSON:     {rep_path}")
    print(f"  Defect map:      {map_path}")


if __name__ == '__main__':
    main()
