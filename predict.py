#!/usr/bin/env python3
"""
UltraFabric-Vision — batch/offline inference CLI.
=================================================
Score a single image or a folder of images and emit a CSV/JSON report suitable
for MES / QA dashboards. Runs the same calibrated ensemble as the live server.

Examples
--------
  python predict.py --input sample.jpg
  python predict.py --input ./roll_2024_06_09/ --out report.csv
  python predict.py --input ./frames/ --out report.json --format json --save-overlays ./out_vis
"""
import os
import sys
import csv
import json
import glob
import argparse

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import cv2
from fabric_engine import InferenceEngine

IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


def gather(path):
    if os.path.isfile(path):
        return [path]
    # Deduplicate case-insensitively (Windows globs are case-insensitive, so
    # *.png and *.PNG would otherwise return the same file twice).
    seen, files = set(), []
    for name in sorted(os.listdir(path)) if os.path.isdir(path) else []:
        if name.lower().endswith(IMG_EXTS):
            key = name.lower()
            if key not in seen:
                seen.add(key)
                files.append(os.path.join(path, name))
    return files


def main():
    ap = argparse.ArgumentParser(description="UltraFabric-Vision batch defect inference")
    ap.add_argument('--input', required=True, help='image file or folder of images')
    ap.add_argument('--out', default=None, help='report path (.csv or .json). Default: stdout only')
    ap.add_argument('--format', choices=['csv', 'json'], default=None, help='override report format')
    ap.add_argument('--save-overlays', default=None, help='dir to write heatmap overlays for defects')
    ap.add_argument('--device', default=None, help='cuda | cpu | auto (default: config/env)')
    ap.add_argument('--mode', choices=['accurate', 'fast'], default='accurate',
                    help='accurate = full ensemble (default); fast = single detector, low latency')
    args = ap.parse_args()

    files = gather(args.input)
    if not files:
        print(f"No images found at: {args.input}")
        sys.exit(1)

    engine = InferenceEngine(device=args.device, capture_attention=False, mode=args.mode)
    if not engine.calibrated:
        print("WARNING: no weights/calibration.json found — using fallback threshold. "
              "Run `python calibrate.py` for a calibrated decision boundary.")
    mode_desc = f"fast ({engine.fast_model})" if args.mode == 'fast' else "accurate (ensemble)"
    print(f"Loaded engine on {engine.device}. Mode={mode_desc}. "
          f"Scoring {len(files)} image(s)...\n")

    if args.save_overlays:
        os.makedirs(args.save_overlays, exist_ok=True)

    rows, n_defect, lat_sum = [], 0, 0.0
    for f in files:
        img = cv2.imread(f)
        if img is None:
            rows.append({'file': f, 'error': 'unreadable'})
            print(f"  {os.path.basename(f):<32} ERROR unreadable")
            continue
        r = engine.predict_bgr(img)
        lat_sum += r.latency_ms
        n_defect += int(r.is_defect)
        rows.append({
            'file': os.path.basename(f),
            'path': f,
            'score': round(r.score, 4),
            'is_defect': r.is_defect,
            'threshold': round(r.threshold, 4),
            'latency_ms': round(r.latency_ms, 2),
        })
        print(f"  {os.path.basename(f):<32} score={r.score:8.2f}  "
              f"{'DEFECT' if r.is_defect else 'OK':<6}  {r.latency_ms:6.1f} ms")
        if args.save_overlays and r.is_defect:
            cv2.imwrite(os.path.join(args.save_overlays, f'defect_{os.path.basename(f)}.jpg'),
                        engine.overlay(img, r))

    scored = [r for r in rows if 'error' not in r]
    print(f"\nSummary: {len(scored)} scored, {n_defect} defect, {len(scored)-n_defect} ok. "
          f"Avg latency {lat_sum/max(1,len(scored)):.1f} ms/img on {engine.device}.")

    if args.out:
        fmt = args.format or ('json' if args.out.lower().endswith('.json') else 'csv')
        if fmt == 'json':
            with open(args.out, 'w') as fh:
                json.dump({'threshold': engine.threshold, 'device': str(engine.device),
                           'results': rows}, fh, indent=2)
        else:
            cols = ['file', 'path', 'score', 'is_defect', 'threshold', 'latency_ms', 'error']
            with open(args.out, 'w', newline='') as fh:
                w = csv.DictWriter(fh, fieldnames=cols)
                w.writeheader()
                for r in rows:
                    w.writerow({k: r.get(k, '') for k in cols})
        print(f"Report written to {args.out}")


if __name__ == '__main__':
    main()
