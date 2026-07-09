#!/usr/bin/env python3
"""End-to-end inference smoke test using the real deployment path:
preprocess_frame -> ensemble.predict (calibrated z-scores) -> shared threshold.
Reports per-image score / decision / latency on real good vs defect images."""
import os, sys, time, glob
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
import cv2, numpy as np, torch

from models.patchcore import PatchCore
from models.dino import DINOFeatureExtractor
from models.vit_autoencoder import ViTAutoencoder
from fusion.ensemble import EnsembleFusion
from app_utils.helpers import preprocess_frame, load_threshold
from app_utils.config import config

base = os.path.dirname(os.path.abspath(__file__))
wd = os.path.join(base, 'weights')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

pc = PatchCore().to(device);  pc.load_memory_bank(os.path.join(wd, 'patchcore_memory_bank.pkl'))
dino = DINOFeatureExtractor().to(device); dino.load_memory_bank(os.path.join(wd, 'dino_memory_bank.pkl'))
dino.capture_attention = True
vae = ViTAutoencoder().to(device); vae.load_weights(os.path.join(wd, 'vit_ae_weights.pth'))
ens = EnsembleFusion([pc, dino, vae])
thr = load_threshold(config.calibration_path, config.default_threshold)
print(f"Calibrated stats -> PatchCore z0={pc.score_mean:.2f}/{pc.score_std:.2f}  "
      f"DINO {dino.score_mean:.2f}/{dino.score_std:.2f}  ViT-AE {vae.score_mean:.3f}/{vae.score_std:.3f}")
print(f"Threshold: {thr:.3f}\n")

def run(path):
    img = cv2.imread(path)
    t = preprocess_frame(img, (224, 224), device, config.imagenet_mean, config.imagenet_std)
    t0 = time.time()
    with torch.no_grad():
        score, hmap = ens.predict(t)
    dt = (time.time() - t0) * 1000
    return score, score > thr, dt

# Warm-up (first frame pays lazy-init cost)
gs = sorted(glob.glob(os.path.join(base, 'data', 'test', 'good', '*')))
ds = sorted(glob.glob(os.path.join(base, 'data', 'test', 'defect', '*')))
run(gs[0])

def report(name, files, expected_anom):
    lat, correct = [], 0
    print(f"--- {name} (expect {'DEFECT' if expected_anom else 'OK'}) ---")
    for f in files[:5]:
        s, a, dt = run(f); lat.append(dt)
        correct += (a == expected_anom)
        print(f"  {os.path.basename(f):<28} score={s:7.2f}  -> {'DEFECT' if a else 'OK':<6}  {dt:6.1f} ms")
    return lat, correct

lg, cg = report("GOOD", gs, False)
ld, cd = report("DEFECT", ds, True)
alllat = lg + ld
print(f"\nAccuracy on sample: {cg + cd}/{len(lg)+len(ld)} correct")
print(f"Latency: mean={np.mean(alllat):.1f} ms  min={np.min(alllat):.1f}  max={np.max(alllat):.1f}  "
      f"(~{1000/np.mean(alllat):.1f} FPS on {device})")
