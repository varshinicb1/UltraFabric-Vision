#!/usr/bin/env python3
"""
Fast recalibration - no retraining required.
=============================================
Loads the EXISTING fitted memory banks + trained autoencoder and:
  1. Records each model's raw-score mean/std over normal data (for z-scoring),
  2. Re-saves the weights with those stats bundled in,
  3. Computes the optimal fused-score threshold (Youden's J) on the test set,
  4. Writes weights/calibration.json (shared by the web backend and PyQt UI).

Run this after the preprocessing fix if you don't want to rebuild memory banks:

    python calibrate.py

It only does forward passes, so it finishes in seconds-to-minutes. If you have
changed the training data or want the best possible memory banks, run the full
`python train.py` instead (it now calibrates automatically).
"""

import os
import sys
import json

# Force UTF-8 stdout so status glyphs don't crash on Windows cp1252 consoles.
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.patchcore import PatchCore
from models.dino import DINOFeatureExtractor
from models.vit_autoencoder import ViTAutoencoder
from fusion.ensemble import EnsembleFusion
from train import FabricDataset, get_transform, evaluate_model_ensemble, evaluate_model


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_dir = os.path.join(base_dir, 'data', 'train', 'good')
    test_good_dir = os.path.join(base_dir, 'data', 'test', 'good')
    test_defect_dir = os.path.join(base_dir, 'data', 'test', 'defect')
    weights_dir = os.path.join(base_dir, 'weights')

    for d, label in [(train_dir, "train/good"), (test_good_dir, "test/good"),
                     (test_defect_dir, "test/defect")]:
        if not os.path.isdir(d) or len(os.listdir(d)) == 0:
            print(f"ERROR: Dataset directory '{label}' is missing or empty.")
            return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    transform = get_transform()
    fit_loader = DataLoader(FabricDataset(train_dir, transform=transform), batch_size=1, shuffle=False)
    test_good_loader = DataLoader(FabricDataset(test_good_dir, transform=transform), batch_size=1, shuffle=False)
    test_defect_loader = DataLoader(FabricDataset(test_defect_dir, transform=transform), batch_size=1, shuffle=False)

    # --- Load existing weights ---
    print("Loading existing weights...")
    pc = PatchCore().to(device)
    if not pc.load_memory_bank(os.path.join(weights_dir, 'patchcore_memory_bank.pkl')):
        print("ERROR: patchcore_memory_bank.pkl not found - run train.py first.")
        return

    dino = DINOFeatureExtractor().to(device)
    if not dino.load_memory_bank(os.path.join(weights_dir, 'dino_memory_bank.pkl')):
        print("ERROR: dino_memory_bank.pkl not found - run train.py first.")
        return

    vae = ViTAutoencoder().to(device)
    if not vae.load_weights(os.path.join(weights_dir, 'vit_ae_weights.pth')):
        print("ERROR: vit_ae_weights.pth not found - run train.py first.")
        return

    # --- Calibrate each model's score distribution on NORMAL data ---
    print("Calibrating score distributions on normal data...")
    for model, name in [(pc, "PatchCore"), (dino, "DINO"), (vae, "ViT-AE")]:
        model.calibrate(fit_loader)
        print(f"  {name}: mean={model.score_mean:.4f} std={model.score_std:.4f}")

    # --- Re-save with calibration bundled in ---
    pc.save_memory_bank(os.path.join(weights_dir, 'patchcore_memory_bank.pkl'))
    dino.save_memory_bank(os.path.join(weights_dir, 'dino_memory_bank.pkl'))
    vae.save_weights(os.path.join(weights_dir, 'vit_ae_weights.pth'))
    print("  [OK] Re-saved weights with calibration stats")

    # --- Compute optimal fused-score threshold on the test set ---
    print("Computing optimal detection threshold on test set...")
    ensemble = EnsembleFusion([pc, dino, vae])
    r = evaluate_model_ensemble(ensemble, test_good_loader, test_defect_loader)

    # --- Per-model thresholds (used by Fast mode: single-detector inference) ---
    print("Computing per-model thresholds for Fast mode...")
    per_model = {}
    for model, key in [(pc, 'patchcore'), (dino, 'dino'), (vae, 'vit_ae')]:
        rm = evaluate_model(model, test_good_loader, test_defect_loader, key)
        per_model[key] = {'threshold': rm['threshold'], 'auroc': rm['auroc'],
                          'f1': rm['f1_score'], 'infer_ms': rm['avg_inference_time']}
        print(f"  {key}: AUROC={rm['auroc']:.4f}  thr={rm['threshold']:.4f}  {rm['avg_inference_time']:.1f} ms")

    # Fast mode wants the FASTEST detector that is still accurate: among models
    # within 2% AUROC of the best, choose the one with lowest inference time.
    best_auroc = max(m['auroc'] for m in per_model.values())
    eligible = {k: v for k, v in per_model.items() if v['auroc'] >= best_auroc - 0.02}
    fast_model = min(eligible, key=lambda k: eligible[k]['infer_ms'])

    calib_path = os.path.join(weights_dir, 'calibration.json')
    with open(calib_path, 'w') as f:
        json.dump({
            'threshold': r['threshold'],
            'ensemble_auroc': r['auroc'],
            'ensemble_f1': r['f1_score'],
            'per_model': per_model,
            'fast_model': fast_model,
        }, f, indent=2)

    print(f"\n  Ensemble AUROC={r['auroc']:.4f}  F1={r['f1_score']:.4f}")
    print(f"  Recommended Fast-mode detector: {fast_model}")
    print(f"  [OK] Saved calibration.json (ensemble threshold={r['threshold']:.4f})")
    print("\nCalibration complete. Restart the backend / UI to pick up the new thresholds.")


if __name__ == "__main__":
    main()
