#!/usr/bin/env python3
"""
FabricAI Pro Suite - Training & Evaluation Pipeline
=====================================================
1. Fits PatchCore memory bank on normal training images.
2. Fits DINO k-NN memory bank on normal training images.
3. Trains the ViT Autoencoder to reconstruct normal fabric.
4. Evaluates all three models + Ensemble on the held-out test set.
5. Prints a full results table and saves an IEEE-style PDF report.
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

# Local imports
from models.patchcore import PatchCore
from models.vit_autoencoder import ViTAutoencoder
from models.dino import DINOFeatureExtractor
from fusion.ensemble import EnsembleFusion
from app_utils.config import config


# ======================================================================
# Dataset
# ======================================================================
class FabricDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = sorted([
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = Image.open(self.image_files[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image


# ======================================================================
# Helpers
# ======================================================================
def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])


def evaluate_model(model, good_loader, defect_loader, model_name):
    """Evaluate a single model and return metrics dict."""
    y_true, y_scores, times = [], [], []

    # Good samples  -> label 0
    for batch in good_loader:
        batch = batch.to(model.device)
        t0 = time.time()
        score, _ = model.predict(batch)
        times.append(time.time() - t0)
        y_true.append(0)
        y_scores.append(float(score))

    # Defect samples -> label 1
    for batch in defect_loader:
        batch = batch.to(model.device)
        t0 = time.time()
        score, _ = model.predict(batch)
        times.append(time.time() - t0)
        y_true.append(1)
        y_scores.append(float(score))

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    # AUROC
    try:
        auroc = roc_auc_score(y_true, y_scores)
    except ValueError:
        auroc = 0.0

    # Optimal threshold via Youden's J
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_thresh = float(thresholds[best_idx])

    y_pred = (y_scores >= best_thresh).astype(int)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)

    avg_time_ms = np.mean(times) * 1000

    return {
        "model_name": model_name,
        "auroc": auroc,
        "f1_score": f1,
        "precision": prec,
        "recall": rec,
        "avg_inference_time": avg_time_ms,
        "threshold": best_thresh,
    }


def print_results_table(results_list):
    """Pretty-print evaluation results."""
    header = f"{'Model':<20} {'AUROC':>8} {'F1':>8} {'Prec':>8} {'Recall':>8} {'Infer(ms)':>10} {'Thresh':>8}"
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for r in results_list:
        print(f"{r['model_name']:<20} "
              f"{r['auroc']:>8.4f} "
              f"{r['f1_score']:>8.4f} "
              f"{r['precision']:>8.4f} "
              f"{r['recall']:>8.4f} "
              f"{r['avg_inference_time']:>10.2f} "
              f"{r['threshold']:>8.4f}")
    print(sep)


# ======================================================================
# Main pipeline
# ======================================================================
def train_system():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_dir = os.path.join(base_dir, 'data', 'train', 'good')
    test_good_dir = os.path.join(base_dir, 'data', 'test', 'good')
    test_defect_dir = os.path.join(base_dir, 'data', 'test', 'defect')
    weights_dir = os.path.join(base_dir, 'weights')
    os.makedirs(weights_dir, exist_ok=True)

    # Validate data exists
    for d, label in [(train_dir, "train/good"), (test_good_dir, "test/good"),
                     (test_defect_dir, "test/defect")]:
        if not os.path.isdir(d) or len(os.listdir(d)) == 0:
            print(f"ERROR: Dataset directory '{label}' is missing or empty.")
            print("  -> Run:  python scripts/generate_synthetic_data.py")
            return

    transform = get_transform()
    train_dataset = FabricDataset(train_dir, transform=transform)
    test_good_dataset = FabricDataset(test_good_dir, transform=transform)
    test_defect_dataset = FabricDataset(test_defect_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
    fit_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    test_good_loader = DataLoader(test_good_dataset, batch_size=1, shuffle=False)
    test_defect_loader = DataLoader(test_defect_dataset, batch_size=1, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device : {device}")
    print(f"Train  : {len(train_dataset)} images")
    print(f"Test   : {len(test_good_dataset)} good + {len(test_defect_dataset)} defect")

    # ==================================================================
    # 1. PatchCore - fit memory bank
    # ==================================================================
    print("\n" + "=" * 60)
    print("  PHASE 1 - Fitting PatchCore")
    print("=" * 60)
    patchcore = PatchCore().to(device)
    patchcore.fit(fit_loader)
    print(f"  Memory bank shape: {patchcore.memory_bank.shape}")
    patchcore.save_memory_bank(os.path.join(weights_dir, 'patchcore_memory_bank.pkl'))
    print("  [OK] Saved patchcore_memory_bank.pkl")

    # ==================================================================
    # 2. DINO - fit memory bank
    # ==================================================================
    print("\n" + "=" * 60)
    print("  PHASE 2 - Fitting DINO Feature Extractor")
    print("=" * 60)
    dino = DINOFeatureExtractor().to(device)
    dino.fit(fit_loader)
    print(f"  Memory bank shape: {dino.memory_bank.shape}")
    dino.save_memory_bank(os.path.join(weights_dir, 'dino_memory_bank.pkl'))
    print("  [OK] Saved dino_memory_bank.pkl")

    # ==================================================================
    # 3. ViT Autoencoder - train reconstruction
    # ==================================================================
    print("\n" + "=" * 60)
    print("  PHASE 3 - Training ViT Autoencoder")
    print("=" * 60)
    vit_ae = ViTAutoencoder().to(device)
    optimizer = torch.optim.AdamW(vit_ae.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)
    criterion = nn.MSELoss()

    epochs = 15
    vit_ae.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            reconstructed = vit_ae(batch)
            loss = criterion(reconstructed, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        avg = epoch_loss / len(train_loader)
        print(f"  Epoch {epoch + 1:2d}/{epochs} | Loss: {avg:.6f} | LR: {scheduler.get_last_lr()[0]:.2e}")

    # Calibrate reconstruction-error score distribution on normal data so the
    # autoencoder emits comparable z-scores (PatchCore/DINO self-calibrate in fit).
    vit_ae.calibrate(fit_loader)
    vit_ae.save_weights(os.path.join(weights_dir, 'vit_ae_weights.pth'))
    print("  [OK] Saved vit_ae_weights.pth")

    # ==================================================================
    # 4. Evaluation
    # ==================================================================
    print("\n" + "=" * 60)
    print("  PHASE 4 - Evaluation on Test Set")
    print("=" * 60)

    patchcore.eval()
    dino.eval()
    vit_ae.eval()

    results = []

    # Individual models
    for model, name in [(patchcore, "PatchCore"),
                        (dino, "DINO_ViT"),
                        (vit_ae, "ViT_Autoencoder")]:
        r = evaluate_model(model, test_good_loader, test_defect_loader, name)
        results.append(r)
        print(f"  {name}: AUROC={r['auroc']:.4f}  F1={r['f1_score']:.4f}")

    # Ensemble
    ensemble = EnsembleFusion([patchcore, dino, vit_ae])
    r_ens = evaluate_model_ensemble(ensemble, test_good_loader, test_defect_loader)
    results.append(r_ens)
    print(f"  Ensemble: AUROC={r_ens['auroc']:.4f}  F1={r_ens['f1_score']:.4f}")

    # Persist the calibrated ensemble threshold so ALL entry points (web backend
    # and PyQt UI) share one source of truth instead of hardcoded magic numbers.
    import json
    calib_path = os.path.join(weights_dir, 'calibration.json')
    with open(calib_path, 'w') as f:
        json.dump({
            'threshold': r_ens['threshold'],
            'ensemble_auroc': r_ens['auroc'],
            'ensemble_f1': r_ens['f1_score'],
        }, f, indent=2)
    print(f"  [OK] Saved calibration.json (threshold={r_ens['threshold']:.4f})")

    print_results_table(results)

    # ==================================================================
    # 5. Generate IEEE Report
    # ==================================================================
    print("\n  Generating IEEE-style PDF report...")
    from reports.generator import ReportGenerator
    from experiments.tracker import ExperimentTracker

    tracker = ExperimentTracker()
    metrics_dict = {}
    for r in results:
        metrics_dict[r["model_name"]] = {
            "auroc": r["auroc"],
            "f1_score": r["f1_score"],
            "avg_inference_time": r["avg_inference_time"],
        }

    gen = ReportGenerator(tracker.experiment_id)
    pdf_path = gen.generate_report(metrics_dict)
    tracker.save()
    print(f"  [OK] Report saved to: {pdf_path}")

    print("\n" + "=" * 60)
    print("  TRAINING & EVALUATION COMPLETE")
    print("=" * 60)


def evaluate_model_ensemble(ensemble, good_loader, defect_loader):
    """Evaluate the ensemble fusion as if it were a single model."""
    y_true, y_scores, times = [], [], []

    device = ensemble.models[0].device

    for batch in good_loader:
        batch = batch.to(device)
        t0 = time.time()
        score, _ = ensemble.predict(batch)
        times.append(time.time() - t0)
        y_true.append(0)
        y_scores.append(float(score))

    for batch in defect_loader:
        batch = batch.to(device)
        t0 = time.time()
        score, _ = ensemble.predict(batch)
        times.append(time.time() - t0)
        y_true.append(1)
        y_scores.append(float(score))

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    try:
        auroc = roc_auc_score(y_true, y_scores)
    except ValueError:
        auroc = 0.0

    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_thresh = float(thresholds[best_idx])

    y_pred = (y_scores >= best_thresh).astype(int)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)

    return {
        "model_name": "Ensemble",
        "auroc": auroc,
        "f1_score": f1,
        "precision": prec,
        "recall": rec,
        "avg_inference_time": np.mean(times) * 1000,
        "threshold": best_thresh,
    }


if __name__ == "__main__":
    train_system()
