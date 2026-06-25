"""Fix architecture diagram, ROC/PR curves with bootstrap CI bands."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Shadow
import numpy as np
import os, sys, torch
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.utils import resample
from collections import OrderedDict
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

base = r'C:\Users\varsh\OneDrive\Documents\6THSEM\IDP final\UltraFabric-Vision'
sys.path.insert(0, base)
fig_dir = os.path.join(base, 'report', 'Figures')

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times'],
    'font.size': 8,
    'savefig.dpi': 300, 'savefig.bbox': 'standard', 'savefig.pad_inches': 0.1,
})

# ── Load models & scores ──────────────────────────────────────────────
class FabricDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.files = sorted([os.path.join(root_dir, f) for f in os.listdir(root_dir)
                           if f.lower().endswith(('.png','.jpg','.jpeg'))])
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert('RGB')
        if self.transform: img = self.transform(img)
        return img, self.files[idx]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
test_good = FabricDataset(os.path.join(base, 'data/test/good'), transform=transform)
test_defect = FabricDataset(os.path.join(base, 'data/test/defect'), transform=transform)
good_loader = DataLoader(test_good, batch_size=1, shuffle=False)
defect_loader = DataLoader(test_defect, batch_size=1, shuffle=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from models.patchcore import PatchCore
from models.dino import DINOFeatureExtractor
from models.vit_autoencoder import ViTAutoencoder
from fusion.ensemble import EnsembleFusion

models_dict = OrderedDict()
for name, cls, wf in [
    ('PatchCore', PatchCore, 'patchcore_memory_bank.pkl'),
    ('DINO-v2', DINOFeatureExtractor, 'dino_memory_bank.pkl'),
    ('ViT-AE', ViTAutoencoder, 'vit_ae_weights.pth'),
]:
    m = cls().to_device()
    if wf.endswith('.pkl'): m.load_memory_bank(os.path.join(base, 'weights', wf))
    else: m.load_weights(os.path.join(base, 'weights', wf))
    models_dict[name] = m
models_dict['Ensemble'] = EnsembleFusion(list(models_dict.values()))

all_scores = {}
for name, model in models_dict.items():
    gs, ds = [], []
    for (batch, _) in good_loader:
        batch = batch.to(device)
        with torch.no_grad():
            s, _ = model.predict(batch) if not isinstance(model, EnsembleFusion) else model.predict(batch)
        gs.append(float(s))
    for (batch, _) in defect_loader:
        batch = batch.to(device)
        with torch.no_grad():
            s, _ = model.predict(batch) if not isinstance(model, EnsembleFusion) else model.predict(batch)
        ds.append(float(s))
    all_scores[name] = (np.array(gs), np.array(ds))
    print(f'  {name}: good {gs[0]:.2f}..{gs[-1]:.2f}, defect {ds[0]:.2f}..{ds[-1]:.2f}')

# ── [1] ARCHITECTURE DIAGRAM — clean, minimal IEEE-style ──────────────
print('\n[1] Architecture diagram (clean IEEE design)...')

NAVY = '#1B2A4A'; LIGHT = '#E8ECF1'; GRAY = '#8B95A5'; WHITE = '#FFFFFF'

fig = plt.figure(figsize=(7.0, 4.5), facecolor='white')
ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
ax.set_xlim(0, 14); ax.set_ylim(0, 7); ax.axis('off')

def rbox(ax, xy, w, h, text, fc=LIGHT, tc=NAVY, fs=9, sub=None, sc=GRAY):
    b = FancyBboxPatch(xy, w, h, boxstyle='round,pad=0.08',
                       facecolor=fc, edgecolor=NAVY, linewidth=1.2, zorder=2)
    ax.add_patch(b)
    cx, cy = xy[0]+w/2, xy[1]+h/2
    ax.text(cx, cy+0.08, text, ha='center', va='center', fontsize=fs,
            fontweight='bold', color=tc, zorder=3)
    if sub:
        ax.text(cx, cy-0.28, sub, ha='center', va='center', fontsize=fs-2,
                color=sc, zorder=3, style='italic')

def arr(ax, x1, y1, x2, y2, lw=1.2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=NAVY, lw=lw))

# Title
ax.text(0.3, 6.7, 'UltraFabric-Vision: End-to-End System Architecture',
        fontsize=11, fontweight='bold', color=NAVY)
ax.text(0.3, 6.4, 'Three complementary anomaly detection pathways fused through weighted ensemble averaging.',
        fontsize=7, color=GRAY, style='italic')

# A. Input Pipeline
ax.text(0.3, 5.9, 'A. Input Pipeline', fontsize=8, fontweight='bold', color=NAVY)
rbox(ax, (0.3, 2.8), 2.2, 0.8, 'Input Frame', NAVY, WHITE, fs=9, sub='224x224 RGB')
arr(ax, 1.4, 2.8, 1.4, 2.0)
rbox(ax, (0.3, 1.4), 2.2, 0.6, 'Preprocess', LIGHT, NAVY, fs=8, sub='CLAHE + Normalize')
arr(ax, 1.4, 1.4, 1.4, 0.8)
rbox(ax, (0.3, 0.3), 2.2, 0.5, 'Feature Extract', LIGHT, NAVY, fs=8, sub='ViT-B/16 Backbone')

# B. Three Pathways
ax.text(3.5, 6.2, 'B. Three Parallel Detection Pathways', fontsize=8, fontweight='bold', color=NAVY)

# PatchCore
rbox(ax, (3.0, 3.8), 2.8, 2.2, 'PatchCore', '#1A5276', WHITE, fs=10, sub='Density Estimation')
for y, t in [(3.3, 'ViT-B/16 blocks 8+11'), (3.0, 'Coreset memory bank K=10K'),
             (2.7, 'KNN scoring via torch.cdist'), (2.3, '→ Anomaly Score + Heatmap')]:
    ax.text(4.4, y, t, fontsize=6, color=GRAY if '→' not in t else '#1A5276',
            ha='center', fontweight='bold' if '→' in t else 'normal')

# DINO-v2
rbox(ax, (6.2, 3.8), 2.8, 2.2, 'DINO-v2', '#922B21', WHITE, fs=10, sub='Self-Supervised')
for y, t in [(3.3, 'ViT-S/8 patch size 8'), (3.0, '6-head self-attention'),
             (2.7, '784 tokens x 384 dimensions'), (2.3, '→ Anomaly Score + Heatmap')]:
    ax.text(7.6, y, t, fontsize=6, color=GRAY if '→' not in t else '#922B21',
            ha='center', fontweight='bold' if '→' in t else 'normal')

# ViT-AE
rbox(ax, (9.4, 3.8), 2.8, 2.2, 'ViT Autoencoder', '#1E8449', WHITE, fs=10, sub='Reconstruction')
for y, t in [(3.3, '4-layer encoder + 4-layer decoder'), (3.0, '256-dim bottleneck'),
             (2.7, 'MSE loss: 1.95 → 0.034'), (2.3, '→ Anomaly Score + Heatmap')]:
    ax.text(10.8, y, t, fontsize=6, color=GRAY if '→' not in t else '#1E8449',
            ha='center', fontweight='bold' if '→' in t else 'normal')

# Arrows from feature extract to pathways
for xd in [4.4, 7.6, 10.8]:
    ax.annotate('', xy=(xd, 3.8), xytext=(2.5, 0.5),
                arrowprops=dict(arrowstyle='->', color=GRAY, lw=0.8, connectionstyle='arc3,rad=0.15'))

# Dotted grouping box
g = FancyBboxPatch((2.6, 1.8), 10.0, 4.5, boxstyle='round,pad=0.15',
                    facecolor='none', edgecolor=GRAY, linewidth=0.8, linestyle='--')
ax.add_patch(g)

# C. Fusion & Output
ax.text(12.2, 5.9, 'C. Fusion', fontsize=8, fontweight='bold', color=NAVY)
for xs in [4.4, 7.6, 10.8]:
    ax.annotate('', xy=(13.0, 3.2), xytext=(xs+1.4, 3.2),
                arrowprops=dict(arrowstyle='->', color=GRAY, lw=1.0))

rbox(ax, (12.6, 2.7), 1.2, 1.0, 'Fusion', NAVY, WHITE, fs=10, sub='w = 1/3 each')
arr(ax, 13.2, 2.7, 13.2, 1.8)
rbox(ax, (12.6, 1.0), 1.2, 0.8, 'Decision', '#6C3483', WHITE, fs=9, sub='threshold = 31.87')
arr(ax, 13.2, 1.0, 13.2, 0.4)
rbox(ax, (12.6, 0.0), 1.2, 0.4, 'Output', NAVY, WHITE, fs=7)

plt.savefig(os.path.join(fig_dir, 'system_architecture.png'), dpi=300)
plt.close()
print('  Saved system_architecture.png')

# ── [2] ROC CURVES with bootstrap confidence bands ────────────────────
print('[2] ROC curves (bootstrap CI)...')
fig, ax = plt.subplots(figsize=(4.5, 4.0))
n_boot = 2000
rng = np.random.RandomState(42)

for name, (gs, ds) in all_scores.items():
    y_true = np.concatenate([np.zeros_like(gs), np.ones_like(ds)])
    y_score = np.concatenate([gs, ds])
    n = len(y_true)

    # Bootstrap ROC curves
    boot_tprs = []
    boot_aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    for _ in range(n_boot):
        idx = resample(np.arange(n), replace=True, random_state=rng)
        yr, ys = y_true[idx], y_score[idx]
        if len(np.unique(yr)) < 2:
            continue
        fpr, tpr, _ = roc_curve(yr, ys)
        boot_aucs.append(auc(fpr, tpr))
        boot_tprs.append(np.interp(mean_fpr, fpr, tpr))

    boot_tprs = np.array(boot_tprs)
    mean_tpr = boot_tprs.mean(axis=0)
    lower = np.percentile(boot_tprs, 2.5, axis=0)
    upper = np.percentile(boot_tprs, 97.5, axis=0)
    mean_auc = np.mean(boot_aucs)
    auc_ci = (np.percentile(boot_aucs, 2.5), np.percentile(boot_aucs, 97.5))

    color = {'PatchCore': '#2166AC', 'DINO-v2': '#D6604D', 'ViT-AE': '#4DAF4A', 'Ensemble': '#000000'}[name]
    ls = '-' if name == 'Ensemble' else '--'
    lw = 2.0 if name == 'Ensemble' else 1.0
    alpha_band = 0.15 if name == 'Ensemble' else 0.08

    ax.fill_between(mean_fpr, lower, upper, alpha=alpha_band, color=color)
    ax.plot(mean_fpr, mean_tpr, linestyle=ls, linewidth=lw, color=color,
            label=f'{name} (AUC={mean_auc:.4f} [{auc_ci[0]:.4f}–{auc_ci[1]:.4f}])')

ax.plot([0, 1], [0, 1], 'k:', linewidth=0.8, alpha=0.3, label='Random')
ax.set_xlim([-0.01, 1.01]); ax.set_ylim([-0.01, 1.01])
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_aspect('equal')
ax.legend(loc='lower right', fontsize=6, framealpha=0.9, edgecolor='black',
          labelspacing=0.3)

plt.savefig(os.path.join(fig_dir, 'fig_roc_curves.png'), dpi=300)
plt.close()
print('  Saved fig_roc_curves.png')

# ── [3] PR CURVES with bootstrap confidence bands ─────────────────────
print('[3] PR curves (bootstrap CI)...')
fig, ax = plt.subplots(figsize=(4.5, 4.0))

for name, (gs, ds) in all_scores.items():
    y_true = np.concatenate([np.zeros_like(gs), np.ones_like(ds)])
    y_score = np.concatenate([gs, ds])
    n = len(y_true)

    boot_precisions = []
    boot_aps = []
    mean_recall = np.linspace(0, 1, 100)

    for _ in range(n_boot):
        idx = resample(np.arange(n), replace=True, random_state=rng)
        yr, ys = y_true[idx], y_score[idx]
        if len(np.unique(yr)) < 2:
            continue
        prec, rec, _ = precision_recall_curve(yr, ys)
        boot_aps.append(average_precision_score(yr, ys))
        boot_precisions.append(np.interp(mean_recall, rec[::-1], prec[::-1]))

    boot_precisions = np.array(boot_precisions)
    mean_prec = boot_precisions.mean(axis=0)
    lower = np.percentile(boot_precisions, 2.5, axis=0)
    upper = np.percentile(boot_precisions, 97.5, axis=0)
    mean_ap = np.mean(boot_aps)
    ap_ci = (np.percentile(boot_aps, 2.5), np.percentile(boot_aps, 97.5))

    color = {'PatchCore': '#2166AC', 'DINO-v2': '#D6604D', 'ViT-AE': '#4DAF4A', 'Ensemble': '#000000'}[name]
    ls = '-' if name == 'Ensemble' else '--'
    lw = 2.0 if name == 'Ensemble' else 1.0
    alpha_band = 0.15 if name == 'Ensemble' else 0.08

    ax.fill_between(mean_recall, lower, upper, alpha=alpha_band, color=color)
    ax.plot(mean_recall, mean_prec, linestyle=ls, linewidth=lw, color=color,
            label=f'{name} (AP={mean_ap:.4f} [{ap_ci[0]:.4f}–{ap_ci[1]:.4f}])')

ax.set_xlim([-0.01, 1.01]); ax.set_ylim([-0.01, 1.08])
ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
ax.legend(loc='lower left', fontsize=6, framealpha=0.9, edgecolor='black',
          labelspacing=0.3)

plt.savefig(os.path.join(fig_dir, 'fig_pr_curves.png'), dpi=300)
plt.close()
print('  Saved fig_pr_curves.png')

# ── Summary ────────────────────────────────────────────────────────────
print('\nDone. Updated 3 figures:')
for f in ['system_architecture.png', 'fig_roc_curves.png', 'fig_pr_curves.png']:
    fp = os.path.join(fig_dir, f)
    sz = os.path.getsize(fp)
    print(f'  {f}: {sz//1024} KB')
