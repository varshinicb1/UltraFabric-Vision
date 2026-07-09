#!/usr/bin/env python3
"""
Generate a synthetic 'fabric batch on a conveyor' video for the batch-inference
demo. A long strip of woven fabric (same texture the models were calibrated on)
scrolls past a fixed camera; defects are baked in at known positions so the
detector's localization can be validated against ground truth.

    python scripts/generate_batch_video.py --batch B001 --out demo_batches/B001.mp4

Writes the .mp4 plus a <name>_truth.json listing the true defect positions.
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import cv2


def weave_strip(width, height, seed=0):
    """A tall strip of synthetic woven fabric (matches generate_synthetic_data)."""
    rng = np.random.RandomState(seed)
    base = np.zeros((height, width, 3), dtype=np.uint8)
    base[:] = (200, 220, 230)
    for x in range(0, width, 4):
        cv2.line(base, (x, 0), (x, height), (180, 200, 210), 1)
    for y in range(0, height, 4):
        cv2.line(base, (0, y), (width, y), (180, 200, 210), 1)
    noise = rng.normal(0, 10, (height, width, 3)).astype(np.int16)
    return np.clip(base + noise, 0, 255).astype(np.uint8)


def stamp_defect(strip, yc, xc, kind, rng):
    """Bake one defect into the strip centred at (yc, xc). Returns its bbox."""
    if kind == 'stain':
        r = int(rng.randint(12, 26))
        patch = strip[max(0, yc - r - 8):yc + r + 8, max(0, xc - r - 8):xc + r + 8].copy()
        cv2.circle(strip, (xc, yc), r, (50, 50, 70), -1)
        # blur locally so the stain looks embedded
        y0, y1 = max(0, yc - r - 8), yc + r + 8
        x0, x1 = max(0, xc - r - 8), xc + r + 8
        strip[y0:y1, x0:x1] = cv2.GaussianBlur(strip[y0:y1, x0:x1], (15, 15), 0)
        return (xc - r, yc - r, 2 * r, 2 * r), 'stain'
    else:  # tear
        L = int(rng.randint(30, 60))
        x2, y2 = xc + L, yc + int(rng.randint(-8, 8))
        cv2.line(strip, (xc, yc), (x2, y2), (255, 255, 255), 3)
        return (xc, min(yc, y2) - 2, L, abs(y2 - yc) + 6), 'tear'


def main():
    ap = argparse.ArgumentParser(description="Synthetic fabric batch conveyor video")
    ap.add_argument('--out', required=True, help='output .mp4 path')
    ap.add_argument('--batch', default='B001', help='batch number/label')
    ap.add_argument('--frames', type=int, default=300)
    ap.add_argument('--fps', type=int, default=15)
    ap.add_argument('--size', type=int, default=224, help='square camera frame size')
    ap.add_argument('--speed', type=int, default=6, help='conveyor scroll px/frame')
    ap.add_argument('--defects', default='0.18,0.47,0.80',
                    help='comma fractions (0-1) of batch length where defects sit')
    ap.add_argument('--meters', type=float, default=5.0, help='physical batch length (m)')
    ap.add_argument('--seed', type=int, default=7)
    args = ap.parse_args()

    S = args.size
    scroll = (args.frames - 1) * args.speed
    strip_h = S + scroll
    rng = np.random.RandomState(args.seed)
    strip = weave_strip(S, strip_h, seed=args.seed)

    # Bake defects at requested fractions of the scrollable length.
    truth = []
    for i, frac in enumerate([float(x) for x in args.defects.split(',') if x.strip()]):
        yc = int(S / 2 + frac * scroll)
        xc = int(rng.randint(int(S * 0.25), int(S * 0.75)))
        kind = 'stain' if i % 2 == 0 else 'tear'
        bbox, k = stamp_defect(strip, yc, xc, kind, rng)
        pos_frac = yc / float(strip_h)
        truth.append({'kind': k, 'strip_y': yc, 'pos_frac': round(pos_frac, 4),
                      'position_m': round(pos_frac * args.meters, 3),
                      'appears_frame': int((yc - S / 2) / max(1, args.speed))})

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw = cv2.VideoWriter(args.out, fourcc, args.fps, (S, S))
    for i in range(args.frames):
        y0 = i * args.speed
        frame = strip[y0:y0 + S].copy()
        # small batch/position banner (does not affect inference; cropped region
        # is the top strip which the model still sees as fabric-ish, so keep it faint)
        vw.write(frame)
    vw.release()

    truth_path = os.path.splitext(args.out)[0] + '_truth.json'
    with open(truth_path, 'w') as f:
        json.dump({'batch': args.batch, 'frames': args.frames, 'fps': args.fps,
                   'meters': args.meters, 'defects': truth}, f, indent=2)

    print(f"Wrote {args.out}  ({args.frames} frames @ {args.fps} fps, {args.meters} m batch)")
    print(f"Ground-truth defects at: " +
          ", ".join(f"{d['position_m']}m ({d['kind']})" for d in truth))
    print(f"Truth saved to {truth_path}")


if __name__ == '__main__':
    main()
