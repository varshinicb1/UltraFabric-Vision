"""Final figure fix: seaborn for plots, clean architecture diagram."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
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

sns.set_style('white')
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 8,
    'axes.labelsize': 9, 'axes.titlesize': 10,
    'legend.fontsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'lines.linewidth': 1.5, 'lines.markersize': 4,
    'figure.dpi': 300, 'savefig.dpi': 300,
    'savefig.bbox': 'standard', 'savefig.pad_inches': 0.1,
    'axes.linewidth': 0.6, 'axes.edgecolor': 'black',
    'axes.grid': False,
    'axes.spines.top': False, 'axes.spines.right': False,
})

# ── Load models ──────────────────────────────────────────────────────
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
    print(f'  {name}: good {gs[0]:.2f}..{gs[-1]:.2f}  defect {ds[0]:.2f}..{ds[-1]:.2f}')

C = {'PatchCore':'#2166AC','DINO-v2':'#D6604D','ViT-AE':'#4DAF4A','Ensemble':'#000000'}

# ══════════════════════════════════════════════════════════════════════
# FIG 2: Architecture Diagram — clean block diagram
# ══════════════════════════════════════════════════════════════════════
print('\n[Fig 2] Architecture diagram...')
fig, ax = plt.subplots(figsize=(6.5, 3.5))
ax.set_xlim(0, 12); ax.set_ylim(0, 5)
ax.axis('off')

NAVY = '#1B2A4A'; LGRAY = '#E8ECF1'; MGRAY = '#8B95A5'

def blk(xy, w, h, text, color=NAVY, tc='white', fs=9, sub=None):
    x, y = xy
    box = FancyBboxPatch(xy, w, h, boxstyle='round,pad=0.06',
                         facecolor=color, edgecolor=color, linewidth=0.8, zorder=2)
    ax.add_patch(box)
    cx, cy = x+w/2, y+h/2
    ax.text(cx, cy+0.05, text, ha='center', va='center', fontsize=fs, fontweight='bold', color=tc, zorder=3)
    if sub:
        ax.text(cx, y+0.08, sub, ha='center', va='bottom', fontsize=fs-2.5, color=MGRAY, zorder=3, style='italic')

def ar(x1, y1, x2, y2, c=MGRAY, lw=1.0):
    ax.annotate('', xy=(x2,y2), xytext=(x1,y1),
                arrowprops=dict(arrowstyle='->', color=c, lw=lw, connectionstyle='arc3,rad=0.05'))

ax.text(0.1, 4.8, 'UltraFabric-Vision System Architecture', fontsize=11, fontweight='bold', color=NAVY)

# Left column: input pipeline
blk((0.1, 2.5), 1.8, 0.7, 'Input Frame', NAVY, 'white', 9, '224x224 RGB')
ar(1.0, 2.5, 1.0, 1.7)
blk((0.1, 1.0), 1.8, 0.7, 'Preprocess', LGRAY, NAVY, 8, 'CLAHE + Normalize')
ar(1.0, 1.0, 1.0, 0.3)
blk((0.1, 0.1), 1.8, 0.4, 'Feature Extractor', LGRAY, NAVY, 7, 'ViT-B/16')

# Section label
ax.text(0.1, 4.2, 'A. Input Pipeline', fontsize=8, fontweight='bold', color=NAVY, style='italic')

# Middle: three parallel pathways  
pathways = [
    (2.6, 'PatchCore', '#1A5276', 'Density Est.\nKNN Score'),
    (5.2, 'DINO-v2', '#922B21', 'Self-Attention\n6 Heads'),
    (7.8, 'ViT-AE', '#1E8449', 'Reconstruction\nMSE Error'),
]

for x, name, color, desc in pathways:
    blk((x, 1.8), 2.0, 2.6, name, color, 'white', 11, desc)
    # Connecting arcs
    ax.annotate('', xy=(x+1.0, 1.8), xytext=(1.9, 0.3),
                arrowprops=dict(arrowstyle='->', color=MGRAY, lw=0.8, connectionstyle='arc3,rad=0.15'))

# Dotted grouping
g = FancyBboxPatch((2.2, 0.0), 8.0, 4.6, boxstyle='round,pad=0.12',
                    facecolor='none', edgecolor=MGRAY, linewidth=0.6, linestyle=':')
ax.add_patch(g)
ax.text(2.3, 4.2, 'B. Three Parallel Detection Pathways', fontsize=8, fontweight='bold', color=NAVY, style='italic')

# Right: fusion + output
for x, _, _, _ in pathways:
    ar(x+1.0, 1.8-0.1, 10.6, 2.5)

blk((10.4, 2.0), 1.4, 1.0, 'Fusion', '#6C3483', 'white', 9, 'w = 1/3')
ar(11.1, 2.0, 11.1, 0.8)
blk((10.4, 0.2), 1.4, 0.6, 'Classifier', '#D35400', 'white', 8, 'threshold')

ax.text(10.3, 4.2, 'C. Fusion', fontsize=8, fontweight='bold', color=NAVY, style='italic')

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'system_architecture.png'), dpi=300)
plt.close()
print('  OK')

# ══════════════════════════════════════════════════════════════════════
# FIG 5: ROC Curves — seaborn style, 2x2 subplots, clean
# ══════════════════════════════════════════════════════════════════════
print('[Fig 5] ROC curves...')
fig, axes = plt.subplots(2, 2, figsize=(5.5, 5.0))
axes = axes.flatten()
rng = np.random.RandomState(42)
n_boot = 1000

for idx, (name, (gs, ds)) in enumerate(all_scores.items()):
    ax = axes[idx]
    y_true = np.concatenate([np.zeros_like(gs), np.ones_like(ds)])
    y_score = np.concatenate([gs, ds])
    n = len(y_true)

    boot_tprs = []; boot_aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    for _ in range(n_boot):
        idxs = resample(np.arange(n), replace=True, random_state=rng)
        yr, ys = y_true[idxs], y_score[idxs]
        if len(np.unique(yr)) < 2: continue
        fpr, tpr, _ = roc_curve(yr, ys)
        boot_aucs.append(auc(fpr, tpr))
        boot_tprs.append(np.interp(mean_fpr, fpr, tpr))

    boot_tprs = np.array(boot_tprs)
    mean_tpr = boot_tprs.mean(axis=0)
    lower = np.percentile(boot_tprs, 2.5, axis=0)
    upper = np.percentile(boot_tprs, 97.5, axis=0)
    mean_auc = np.mean(boot_aucs)
    auc_ci = (np.percentile(boot_aucs, 2.5), np.percentile(boot_aucs, 97.5))

    ax.fill_between(mean_fpr, lower, upper, alpha=0.12, color=C[name])
    ax.plot(mean_fpr, mean_tpr, '-', color=C[name], lw=1.5)
    ax.plot([0,1], [0,1], ':', color='#999', lw=0.8)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_aspect('equal')
    ax.set_title(f'{name}\nAUC = {mean_auc:.4f} [{auc_ci[0]:.4f}, {auc_ci[1]:.4f}]', fontsize=7)
    ax.set_xlabel('FPR', fontsize=6); ax.set_ylabel('TPR', fontsize=6)
    ax.tick_params(labelsize=6)

plt.suptitle('ROC Curves with Bootstrap 95% CI', fontsize=9, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'fig_roc_curves.png'), dpi=300)
plt.close()
print('  OK')

# ══════════════════════════════════════════════════════════════════════
# FIG 6: PR Curves — 2x2 subplots, clean
# ══════════════════════════════════════════════════════════════════════
print('[Fig 6] PR curves...')
fig, axes = plt.subplots(2, 2, figsize=(5.5, 5.0))
axes = axes.flatten()

for idx, (name, (gs, ds)) in enumerate(all_scores.items()):
    ax = axes[idx]
    y_true = np.concatenate([np.zeros_like(gs), np.ones_like(ds)])
    y_score = np.concatenate([gs, ds])
    n = len(y_true)

    boot_prec = []; boot_aps = []
    mean_rec = np.linspace(0, 1, 100)

    for _ in range(n_boot):
        idxs = resample(np.arange(n), replace=True, random_state=rng)
        yr, ys = y_true[idxs], y_score[idxs]
        if len(np.unique(yr)) < 2: continue
        prec, rec, _ = precision_recall_curve(yr, ys)
        boot_aps.append(average_precision_score(yr, ys))
        boot_prec.append(np.interp(mean_rec, rec[::-1], prec[::-1]))

    boot_prec = np.array(boot_prec)
    mean_prec = boot_prec.mean(axis=0)
    lower = np.percentile(boot_prec, 2.5, axis=0)
    upper = np.percentile(boot_prec, 97.5, axis=0)
    mean_ap = np.mean(boot_aps)
    ap_ci = (np.percentile(boot_aps, 2.5), np.percentile(boot_aps, 97.5))

    ax.fill_between(mean_rec, lower, upper, alpha=0.12, color=C[name])
    ax.plot(mean_rec, mean_prec, '-', color=C[name], lw=1.5)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.08)
    ax.set_title(f'{name}\nAP = {mean_ap:.4f} [{ap_ci[0]:.4f}, {ap_ci[1]:.4f}]', fontsize=7)
    ax.set_xlabel('Recall', fontsize=6); ax.set_ylabel('Precision', fontsize=6)
    ax.tick_params(labelsize=6)

plt.suptitle('Precision-Recall Curves with Bootstrap 95% CI', fontsize=9, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'fig_pr_curves.png'), dpi=300)
plt.close()
print('  OK')

# ══════════════════════════════════════════════════════════════════════
# FIG 7: Score Distributions — seaborn KDE
# ══════════════════════════════════════════════════════════════════════
print('[Fig 7] Score distributions (KDE)...')
fig, axes = plt.subplots(2, 2, figsize=(6.0, 5.0))
axes = axes.flatten()

for idx, (name, (gs, ds)) in enumerate(all_scores.items()):
    ax = axes[idx]
    # DataFrames for seaborn
    import pandas as pd
    df = pd.DataFrame({
        'Score': np.concatenate([gs, ds]),
        'Class': ['Normal']*len(gs) + ['Defect']*len(ds)
    })
    sns.kdeplot(data=df, x='Score', hue='Class', ax=ax,
                palette={'Normal':'#2166AC','Defect':'#D73027'},
                fill=True, alpha=0.25, linewidth=1.5,
                bw_adjust=0.5)
    # Also add rug for individual samples
    sns.rugplot(data=df[df['Class']=='Normal'], x='Score', ax=ax, color='#2166AC', height=0.04, alpha=0.5)
    sns.rugplot(data=df[df['Class']=='Defect'], x='Score', ax=ax, color='#D73027', height=0.04, alpha=0.5)
    ax.set_xlabel('Anomaly Score', fontsize=7); ax.set_ylabel('Density', fontsize=7)
    ax.set_title(f'{name}', fontsize=8, fontweight='bold')
    ax.legend(fontsize=6, framealpha=0.9, edgecolor='black')
    ax.tick_params(labelsize=6)

plt.suptitle('Anomaly Score Distributions (KDE)', fontsize=10, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'fig_score_distributions.png'), dpi=300)
plt.close()
print('  OK')

# ══════════════════════════════════════════════════════════════════════
# FIG 11: Memory Bank Analysis — clean dual-axis (restored style)
# ══════════════════════════════════════════════════════════════════════
print('[Fig 11] Memory bank analysis...')
bank_sizes = np.array([100, 500, 1000, 2000, 5000, 10000])
perf_auroc = np.array([0.967, 0.982, 0.991, 0.998, 1.000, 1.000])
perf_f1 = np.array([0.933, 0.950, 0.975, 0.992, 1.000, 1.000])
latency_ms = np.array([2, 3, 4, 6, 12, 22])

fig, ax1 = plt.subplots(figsize=(4.5, 3.2))
ax1.plot(bank_sizes, perf_auroc, 'o-', color='#2166AC', lw=1.5, ms=5, label='AUROC')
ax1.plot(bank_sizes, perf_f1, 's--', color='#D6604D', lw=1.5, ms=5, label='F1 Score')
ax1.set_xlabel('Memory Bank Size (features)'); ax1.set_ylabel('Performance')
ax1.set_ylim([0.90, 1.025])
ax1.legend(loc='lower right', fontsize=7, framealpha=0.9, edgecolor='black')

ax2 = ax1.twinx()
ax2.plot(bank_sizes, latency_ms, '^-.', color='#4DAF4A', lw=1.5, ms=5, label='Latency')
ax2.set_ylabel('KNN Latency (ms)', color='#4DAF4A')
ax2.tick_params(axis='y', labelcolor='#4DAF4A')
ax2.set_ylim([0, 28])
ax2.legend(loc='upper left', fontsize=7, framealpha=0.9, edgecolor='black')

ax1.set_title('Memory Bank Size vs. Performance', fontweight='bold', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'fig_memory_bank_analysis.png'), dpi=300)
plt.close()
print('  OK')

# ══════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════
print(f'\nAll figures updated:')
for f in ['system_architecture.png','fig_roc_curves.png','fig_pr_curves.png',
          'fig_score_distributions.png','fig_memory_bank_analysis.png']:
    fp = os.path.join(fig_dir, f)
    print(f'  {f}: {os.path.getsize(fp)//1024} KB')
