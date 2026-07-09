#!/usr/bin/env python3
"""
UltraFabric-Vision - GPU / CPU inference benchmark.
====================================================
Measures real per-frame latency of the calibrated pipeline with correct GPU
timing (warm-up + torch.cuda.synchronize + percentiles), broken down by:

  * Mode        : Accurate (full ensemble) vs Fast (single detector)
  * Component   : preprocessing, PatchCore, DINO, ViT-AE, ensemble fusion
  * Precision   : fp16 AMP on vs off (CUDA only)

It reports mean / p50 / p95 / min / max latency, FPS, and peak VRAM, and can
write a Markdown table + JSON you can paste into the report.

Run (on your RTX 4050 laptop):
    python scripts/gpu_benchmark.py                       # auto GPU, 200 iters
    python scripts/gpu_benchmark.py --iters 300 --amp both --out bench.md
    python scripts/gpu_benchmark.py --device cpu          # CPU baseline

Notes:
  * First run downloads the ViT/DINO backbones if not cached.
  * "Accurate" mode is the one to quote for a spec sheet on GPU: it is both the
    most accurate and, on a modern GPU, real-time.
"""
import os
import sys
import json
import time
import argparse
import glob

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
import torch

from models import base as model_base
from models.patchcore import PatchCore
from models.dino import DINOFeatureExtractor
from models.vit_autoencoder import ViTAutoencoder
from fusion.ensemble import EnsembleFusion
from app_utils.helpers import preprocess_frame, load_calibration
from app_utils.config import config

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sync(dev):
    if dev.type == 'cuda':
        torch.cuda.synchronize()


def measure(fn, iters, warmup, dev):
    """Return latency stats (ms) for a callable, with warm-up and GPU sync."""
    for _ in range(warmup):
        fn()
    sync(dev)
    ts = np.empty(iters, dtype=np.float64)
    for i in range(iters):
        t0 = time.perf_counter()
        fn()
        sync(dev)
        ts[i] = (time.perf_counter() - t0) * 1000.0
    return {
        'mean': float(ts.mean()), 'p50': float(np.percentile(ts, 50)),
        'p95': float(np.percentile(ts, 95)), 'min': float(ts.min()),
        'max': float(ts.max()), 'fps': float(1000.0 / ts.mean()),
    }


def frame_source(n=16):
    """Real test frames if present, else synthetic; returns a list of BGR arrays."""
    files = sorted(glob.glob(os.path.join(BASE, 'data', 'test', 'good', '*')) +
                   glob.glob(os.path.join(BASE, 'data', 'test', 'defect', '*')))[:n]
    frames = [cv2.imread(f) for f in files]
    frames = [f for f in frames if f is not None]
    if not frames:
        rng = np.random.RandomState(0)
        frames = [(rng.rand(480, 640, 3) * 255).astype(np.uint8) for _ in range(n)]
    return frames


def env_info(dev):
    info = {'torch': torch.__version__, 'device': str(dev)}
    if dev.type == 'cuda':
        info['gpu'] = torch.cuda.get_device_name(0)
        info['cuda'] = torch.version.cuda
        props = torch.cuda.get_device_properties(0)
        info['vram_total_mb'] = round(props.total_memory / 1e6)
        info['capability'] = f"{props.major}.{props.minor}"
    return info


def run_suite(dev, frames, iters, warmup, amp, tensors, models, ensemble, fast_key):
    model_base.USE_AMP = amp and dev.type == 'cuda'
    pc, dino, vae = models
    ntf = len(tensors)
    idx = {'i': 0}

    def nxt():
        t = tensors[idx['i'] % ntf]
        idx['i'] += 1
        return t

    fast_obj = {'patchcore': pc, 'dino': dino, 'vit_ae': vae}.get(fast_key, dino)

    # End-to-end (preprocess + score) per mode
    def e2e_accurate():
        f = frames[idx['i'] % len(frames)]
        t = preprocess_frame(f, (224, 224), dev, config.imagenet_mean, config.imagenet_std)
        with torch.no_grad():
            ensemble.predict(t)
        idx['i'] += 1

    def e2e_fast():
        f = frames[idx['i'] % len(frames)]
        t = preprocess_frame(f, (224, 224), dev, config.imagenet_mean, config.imagenet_std)
        with torch.no_grad():
            fast_obj.predict(t)
        idx['i'] += 1

    # Component timings (exclude preprocessing; use pre-made tensors)
    def c_pre():
        f = frames[idx['i'] % len(frames)]
        preprocess_frame(f, (224, 224), dev, config.imagenet_mean, config.imagenet_std)
        idx['i'] += 1

    def c_pc():
        with torch.no_grad():
            pc.predict(nxt())

    def c_dino():
        with torch.no_grad():
            dino.predict(nxt())

    def c_vae():
        with torch.no_grad():
            vae.predict(nxt())

    if dev.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()

    results = {
        'end_to_end': {
            'accurate': measure(e2e_accurate, iters, warmup, dev),
            'fast': measure(e2e_fast, iters, warmup, dev),
        },
        'components': {
            'preprocess': measure(c_pre, iters, warmup, dev),
            'patchcore': measure(c_pc, iters, warmup, dev),
            'dino': measure(c_dino, iters, warmup, dev),
            'vit_ae': measure(c_vae, iters, warmup, dev),
        },
        'amp': model_base.USE_AMP,
    }
    if dev.type == 'cuda':
        results['peak_vram_mb'] = round(torch.cuda.max_memory_allocated() / 1e6)
    return results


def fmt_stats(s):
    return f"{s['mean']:7.2f} | {s['p50']:7.2f} | {s['p95']:7.2f} | {s['min']:7.2f} | {s['max']:7.2f} | {s['fps']:6.1f}"


def to_markdown(env, runs, fast_key):
    lines = []
    lines.append("# UltraFabric-Vision - Inference Benchmark\n")
    lines.append(f"- Device: **{env.get('gpu', env['device'])}**")
    if 'vram_total_mb' in env:
        lines.append(f"- VRAM: {env['vram_total_mb']} MB | CUDA {env['cuda']} | compute {env['capability']}")
    lines.append(f"- torch {env['torch']} | Fast-mode detector: `{fast_key}`\n")
    for label, r in runs:
        lines.append(f"## {label}{' (peak VRAM ' + str(r['peak_vram_mb']) + ' MB)' if 'peak_vram_mb' in r else ''}\n")
        lines.append("| Path | mean | p50 | p95 | min | max | FPS |")
        lines.append("|------|-----:|----:|----:|----:|----:|----:|")
        lines.append(f"| **Accurate (ensemble)** | {fmt_stats(r['end_to_end']['accurate']).replace(' | ', ' | ')} |")
        lines.append(f"| **Fast ({fast_key})** | {fmt_stats(r['end_to_end']['fast'])} |")
        for k in ('preprocess', 'patchcore', 'dino', 'vit_ae'):
            lines.append(f"| _{k}_ | {fmt_stats(r['components'][k])} |")
        lines.append("")
    lines.append("_Latencies in ms; each includes preprocessing for end-to-end rows and "
                 "excludes it for component rows. Measured with warm-up and CUDA sync._")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="UltraFabric-Vision GPU/CPU benchmark")
    ap.add_argument('--iters', type=int, default=200, help='timed iterations (default 200)')
    ap.add_argument('--warmup', type=int, default=30, help='warm-up iterations (default 30)')
    ap.add_argument('--device', default=None, help='cuda | cpu | auto (default: config/env)')
    ap.add_argument('--amp', choices=['on', 'off', 'both'], default='on',
                    help='fp16 AMP on GPU (default on; "both" runs a comparison)')
    ap.add_argument('--out', default=None, help='write a Markdown report to this path (and .json)')
    args = ap.parse_args()

    dev = torch.device(args.device or config.resolved_device())
    env = env_info(dev)
    print("Environment:", json.dumps(env, indent=2))
    if dev.type == 'cpu':
        print("\nNOTE: running on CPU. GPU AMP and the GPU k-NN speedups do not apply here.\n")

    wd = os.path.join(BASE, 'weights')
    print("Loading models...")
    pc = PatchCore().to(dev); pc.load_memory_bank(os.path.join(wd, 'patchcore_memory_bank.pkl'))
    dino = DINOFeatureExtractor().to(dev); dino.load_memory_bank(os.path.join(wd, 'dino_memory_bank.pkl'))
    vae = ViTAutoencoder().to(dev); vae.load_weights(os.path.join(wd, 'vit_ae_weights.pth'))
    ensemble = EnsembleFusion([pc, dino, vae])
    fast_key = load_calibration(config.calibration_path).get('fast_model', 'dino')

    frames = frame_source()
    tensors = [preprocess_frame(f, (224, 224), dev, config.imagenet_mean, config.imagenet_std) for f in frames]

    amp_settings = {'on': [True], 'off': [False], 'both': [True, False]}[args.amp]
    runs = []
    for amp in amp_settings:
        label = f"AMP {'ON (fp16)' if amp else 'OFF (fp32)'}" if dev.type == 'cuda' else "CPU"
        print(f"\nBenchmarking [{label}] - {args.iters} iters (+{args.warmup} warm-up)...")
        r = run_suite(dev, frames, args.iters, args.warmup, amp, tensors, (pc, dino, vae), ensemble, fast_key)
        runs.append((label, r))
        a, f = r['end_to_end']['accurate'], r['end_to_end']['fast']
        print(f"  Accurate: {a['mean']:.2f} ms  ({a['fps']:.1f} FPS)   "
              f"Fast[{fast_key}]: {f['mean']:.2f} ms  ({f['fps']:.1f} FPS)")
        if 'peak_vram_mb' in r:
            print(f"  Peak VRAM: {r['peak_vram_mb']} MB")

    md = to_markdown(env, runs, fast_key)
    print("\n" + "=" * 70 + "\n" + md)
    if args.out:
        with open(args.out, 'w', encoding='utf-8') as fh:
            fh.write(md)
        with open(os.path.splitext(args.out)[0] + '.json', 'w', encoding='utf-8') as fh:
            json.dump({'env': env, 'fast_model': fast_key,
                       'runs': [{'label': l, **r} for l, r in runs]}, fh, indent=2)
        print(f"\nWrote {args.out} and {os.path.splitext(args.out)[0]}.json")


if __name__ == '__main__':
    main()
