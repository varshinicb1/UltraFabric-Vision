#!/usr/bin/env python3
"""Regenerate high-precision paper statistics + result figures from the ACTUAL
calibrated ensemble (z-score fusion, corrected preprocessing). Writes PNGs and a
JSON stats dump into the report Figures directory."""
import os, sys, json, glob
try: sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception: pass
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, cv2, torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, f1_score

from models.patchcore import PatchCore
from models.dino import DINOFeatureExtractor
from models.vit_autoencoder import ViTAutoencoder
from fusion.ensemble import EnsembleFusion
from app_utils.helpers import preprocess_frame
from app_utils.config import config

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.abspath(os.path.join(BASE, '..', '..', '..', 'report', 'Figures'))
if not os.path.isdir(OUT):
    OUT = os.path.join(BASE, 'report_figures'); os.makedirs(OUT, exist_ok=True)
dev = torch.device(config.resolved_device())

wd = os.path.join(BASE, 'weights')
pc = PatchCore().to(dev); pc.load_memory_bank(os.path.join(wd,'patchcore_memory_bank.pkl'))
dino = DINOFeatureExtractor().to(dev); dino.load_memory_bank(os.path.join(wd,'dino_memory_bank.pkl'))
vae = ViTAutoencoder().to(dev); vae.load_weights(os.path.join(wd,'vit_ae_weights.pth'))
ens = EnsembleFusion([pc, dino, vae])

good = sorted(glob.glob(os.path.join(BASE,'data','test','good','*')))
defect = sorted(glob.glob(os.path.join(BASE,'data','test','defect','*')))

def tens(f):
    return preprocess_frame(cv2.imread(f),(224,224),dev,config.imagenet_mean,config.imagenet_std)

raw = {m:{'good':[], 'defect':[]} for m in ['pc','dino','vae']}
ens_s = {'good':[], 'defect':[]}
for label, files in [('good',good),('defect',defect)]:
    for f in files:
        t = tens(f)
        with torch.no_grad():
            raw['pc'][label].append(pc._raw_score(t)[0])
            raw['dino'][label].append(dino._raw_score(t)[0])
            raw['vae'][label].append(vae._raw_score(t)[0])
            ens_s[label].append(ens.predict(t)[0])

def stats(g,d):
    g,d = np.array(g), np.array(d)
    gap = (d.mean()-g.mean())/(d.std()+g.std()+1e-9)
    y = np.r_[np.zeros(len(g)), np.ones(len(d))]; s = np.r_[g,d]
    return dict(mu_n=float(g.mean()), sd_n=float(g.std()), mu_d=float(d.mean()),
               sd_d=float(d.std()), gap=float(gap), auroc=float(roc_auc_score(y,s)))

S = {m: stats(raw[m]['good'], raw[m]['defect']) for m in raw}
S['ens'] = stats(ens_s['good'], ens_s['defect'])

# Youden threshold on ensemble
y = np.r_[np.zeros(len(good)), np.ones(len(defect))]
es = np.r_[ens_s['good'], ens_s['defect']]
fpr,tpr,thr = roc_curve(y, es); j = tpr-fpr; tau = float(thr[np.argmax(j)])
pred = (es>=tau).astype(int)
S['threshold'] = tau
S['ensemble_f1'] = float(f1_score(y,pred))
S['calib'] = {'pc':[pc.score_mean,pc.score_std],'dino':[dino.score_mean,dino.score_std],
              'vae':[vae.score_mean,vae.score_std]}
# cross-model Pearson on defect raw
def pear(a,b): a,b=np.array(a),np.array(b); return float(np.corrcoef(a,b)[0,1])
S['corr'] = {'pc_dino':pear(raw['pc']['defect'],raw['dino']['defect']),
             'dino_vae':pear(raw['dino']['defect'],raw['vae']['defect']),
             'pc_vae':pear(raw['pc']['defect'],raw['vae']['defect'])}

# --- Figure 1: ensemble score distributions ---
plt.figure(figsize=(6,3.4))
plt.hist(ens_s['good'], bins=15, alpha=0.7, label='Normal', color='#2b6cb0')
plt.hist(ens_s['defect'], bins=15, alpha=0.7, label='Defect', color='#c53030')
plt.axvline(tau, color='k', ls='--', lw=1.2, label=f'Threshold τ={tau:.1f}')
plt.xlabel('Calibrated ensemble anomaly score (z-score)'); plt.ylabel('Count')
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(OUT,'fig_score_distributions.png'), dpi=200); plt.close()

# --- Figure 2: confusion matrix ---
tp=int(((pred==1)&(y==1)).sum()); tn=int(((pred==0)&(y==0)).sum())
fp=int(((pred==1)&(y==0)).sum()); fn=int(((pred==0)&(y==1)).sum())
cm=np.array([[tn,fp],[fn,tp]])
plt.figure(figsize=(3.6,3.2)); plt.imshow(cm,cmap='Blues')
for i in range(2):
    for k in range(2):
        plt.text(k,i,cm[i,k],ha='center',va='center',fontsize=16,
                 color='white' if cm[i,k]>cm.max()/2 else 'black')
plt.xticks([0,1],['Pred Normal','Pred Defect']); plt.yticks([0,1],['Normal','Defect'])
plt.title(f'Ensemble Confusion Matrix (τ={tau:.1f})'); plt.tight_layout()
plt.savefig(os.path.join(OUT,'fig_confusion_matrix.png'), dpi=200); plt.close()

with open(os.path.join(OUT,'paper_stats.json'),'w') as f: json.dump(S,f,indent=2)
print("OUT_DIR:", OUT)
print(json.dumps(S, indent=2))
print("FIGS_DONE")
