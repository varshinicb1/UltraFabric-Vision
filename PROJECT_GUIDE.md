# UltraFabric-Vision — Complete Step-by-Step Guide

Real-time, unsupervised fabric-defect detection using a calibrated ensemble of three
Vision-Transformer detectors (PatchCore + DINO + ViT-Autoencoder). This guide takes you
from a fresh machine all the way to a deployed, tested inspection service, and also covers
compiling the research paper and IDP report.

---

## Table of Contents
1. [What this project is](#1-what-this-project-is)
2. [Prerequisites](#2-prerequisites)
3. [Repository layout](#3-repository-layout)
4. [One-time setup](#4-one-time-setup)
5. [Dataset](#5-dataset)
6. [Train and calibrate the models](#6-train-and-calibrate-the-models)
7. [Run inference — four ways](#7-run-inference--four-ways)
8. [GPU deployment with Docker](#8-gpu-deployment-with-docker)
9. [Configuration reference](#9-configuration-reference)
10. [Testing](#10-testing)
11. [Recalibrating for a new fabric (factory onboarding)](#11-recalibrating-for-a-new-fabric-factory-onboarding)
12. [Compiling the reports (LaTeX)](#12-compiling-the-reports-latex)
13. [Git workflow](#13-git-workflow)
14. [Troubleshooting](#14-troubleshooting)
15. [Known limitations](#15-known-limitations)

---

## 1. What this project is

Three complementary anomaly detectors are fused into one decision:

| Detector | Mechanism | Catches |
|----------|-----------|---------|
| **PatchCore** (ViT-B/16) | memory-bank density, k-NN distance | small local texture defects, stains |
| **DINO** (ViT-S/8) | self-supervised patch features, k-NN | global structural defects, tears |
| **ViT-Autoencoder** | reconstruction error | reconstruction failures, holes |

Each detector is trained/fitted **only on defect-free fabric**. Their raw scores live on
very different scales, so each is standardized to a **z-score** and the three are averaged
into a single decision variable. A frame is flagged when the fused z-score exceeds a
calibrated threshold.

**Two principles that make it work:**
- **Preprocessing parity** — inference preprocessing must be byte-identical to fit-time
  preprocessing, or distance-based detectors silently fail.
- **Score calibration** — standardize each detector against its own normal-data statistics
  before fusing.

---

## 2. Prerequisites

- **OS:** Windows 10/11, Linux, or macOS.
- **Python:** 3.10–3.12 recommended.
- **GPU (for real-time):** NVIDIA GPU with CUDA 12.x driver, ≥6 GB VRAM. CPU works for
  offline/batch use but is ~10–20× slower.
- **Disk:** ~4 GB free (model backbones + weights + caches). Downloads fail on a full disk.
- **Git**, and for the reports a **LaTeX** distribution (MiKTeX or TeX Live).
- Internet access on first run (downloads the DINO and ViT backbones once, then cached).

---

## 3. Repository layout

```
UltraFabric-Vision/
├── models/                 # base.py, patchcore.py, dino.py, vit_autoencoder.py
├── fusion/ensemble.py      # z-score fusion
├── temporal/smoothing.py   # temporal score smoothing
├── app_utils/              # config.py (env-driven), helpers.py (preprocessing, threshold)
├── api/input_stream.py     # webcam / RTSP / file source
├── ui/                     # PyQt desktop app (video_thread.py, main_window.py)
├── backend_api.py          # FastAPI REST + WebSocket server
├── fabric_engine.py        # embeddable InferenceEngine SDK
├── predict.py              # batch CLI (folder -> CSV/JSON)
├── train.py                # full train + calibrate + evaluate
├── train_universal.py      # train on the "universal" augmented dataset
├── calibrate.py            # fast recalibration from existing weights (no retrain)
├── tests/test_e2e.py       # end-to-end tests (SDK + HTTP)
├── deploy/                 # Dockerfile.gpu, docker-compose.gpu.yml
├── requirements.txt        # desktop (PyQt) deps
├── requirements-api.txt    # headless server deps
├── DEPLOYMENT.md           # deployment & integration reference
├── weights/                # memory banks, AE weights, calibration.json  (gitignored)
├── data/                   # train/good, test/good, test/defect          (gitignored)
└── report/                 # LaTeX research paper + IDP report
```

---

## 4. One-time setup

> Windows note: commands below use a POSIX shell (Git Bash). In PowerShell, replace
> `export VAR=x` with `$env:VAR="x"` and activate the venv with `.\venv\Scripts\Activate.ps1`.

```bash
# 1. Clone
git clone https://github.com/varshinicb1/UltraFabric-Vision.git
cd UltraFabric-Vision

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate            # Windows PowerShell: .\venv\Scripts\Activate.ps1

# 3a. For the API / inference server (headless, recommended for deployment):
#     Install a CUDA-matched torch FIRST, then the rest.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-api.txt

# 3b. For the PyQt desktop app instead, use:
# pip install -r requirements.txt
```

CPU-only machine? Skip the CUDA index and just `pip install torch torchvision`.

---

## 5. Dataset

Expected structure (defect-free images for training; a labelled split for evaluation):

```
data/
├── train/good/      # 100+ defect-free images
├── test/good/       # held-out defect-free images
└── test/defect/     # defective images (for threshold calibration + evaluation)
```

No data yet? Generate the synthetic benchmark:

```bash
python scripts/generate_synthetic_data.py
```

Supported image types: `.png .jpg .jpeg .bmp .tif .tiff`.

---

## 6. Train and calibrate the models

You have two options.

### Option A — Full training (best accuracy; rebuilds everything)
Fits both memory banks, trains the autoencoder, calibrates every detector, fixes the
threshold, and writes an evaluation report.

```bash
python train.py
```

Outputs (into `weights/`):
- `patchcore_memory_bank.pkl`, `dino_memory_bank.pkl` — memory banks **with calibration stats bundled**
- `vit_ae_weights.pth` — autoencoder weights **with calibration stats**
- `calibration.json` — the shared, Youden-optimal detection threshold

### Option B — Fast recalibration (no retraining)
If the memory banks / AE already exist and you only need to (re)compute score statistics
and the threshold — e.g., after changing preprocessing or moving to new fabric — run:

```bash
python calibrate.py
```

This only does forward passes, so it finishes in seconds-to-minutes and re-saves the
weights with fresh calibration stats plus `calibration.json`.

> Windows console tip: prefix with `PYTHONIOENCODING=utf-8` if you see a `charmap` /
> `cp1252` encode error on status glyphs.

---

## 7. Run inference — four ways

All four share the exact same calibrated pipeline.

**Accurate vs Fast mode.** Every entry point supports two modes:
- **Accurate** (default) — fuses all three detectors. Best accuracy; real-time on GPU.
- **Fast** — runs the single fastest calibrated detector (chosen by `calibrate.py`).
  Much lower latency (measured on CPU: ~75 ms/frame vs ~1060 ms for the ensemble) but
  lower robustness, since it relies on one detector. Use it for real-time inspection on
  CPU/edge, or when the GPU is unavailable; prefer Accurate on GPU where the ensemble is
  already real-time. Select via `UFV_MODE=fast`, `predict.py --mode fast`,
  `InferenceEngine(mode='fast')`, or the API `?mode=fast`.

### 7.1 Batch CLI (offline QA → CSV/JSON)
```bash
# Single image
python predict.py --input sample.jpg

# A folder, writing a CSV report and saving defect overlays
python predict.py --input ./roll_2024_06_09/ --out report.csv --save-overlays ./defect_snaps

# JSON output, forced device
python predict.py --input ./frames/ --out report.json --format json --device cuda
```
CSV columns: `file, path, score, is_defect, threshold, latency_ms, error`.

### 7.2 Embeddable SDK (edge / PLC / line control)
```python
import cv2
from fabric_engine import InferenceEngine

engine = InferenceEngine()                 # loads models + calibrated threshold once
frame  = cv2.imread("frame.jpg")           # BGR numpy array (or a grabbed camera frame)
r = engine.predict_bgr(frame)
print(r.score, r.is_defect, r.latency_ms)
if r.is_defect:
    overlay = engine.overlay(frame, r)     # heatmap-overlaid visualization
    # plc.reject(...)                       # your actuator hook
```

### 7.3 REST / WebSocket server
```bash
# Development (auto-reload):
UFV_DEVICE=cuda UFV_RELOAD=1 python backend_api.py

# Production:
UFV_DEVICE=cuda uvicorn backend_api:app --host 0.0.0.0 --port 8000
```
Then:
```bash
curl http://localhost:8000/api/health                 # {"status":"ok","engine":"loaded"}
curl http://localhost:8000/api/version                 # device, threshold, model list
curl -X POST http://localhost:8000/api/upload_image -F "file=@frame.jpg"
```
Live streaming: connect a WebSocket client to `ws://HOST:8000/ws/stream` and send
base64 JPEG frames; you receive scored results + overlay per frame. The `web_app/`
React dashboard is a reference client.

### 7.4 PyQt desktop app
```bash
python main.py
```

---

## 8. GPU deployment with Docker

Requires the host NVIDIA driver + the **NVIDIA Container Toolkit**.

```bash
# 1. Ensure weights/ contains trained + calibrated files (Section 6).

# 2. Build and run
docker compose -f deploy/docker-compose.gpu.yml up --build

# 3. Verify
curl http://localhost:8000/api/health
```

`weights/` and `data/` are bind-mounted, so you can swap in factory-calibrated weights
and restart without rebuilding. Bare-metal alternative:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-api.txt
UFV_DEVICE=cuda python backend_api.py
```

---

## 9. Configuration reference

Every setting is an environment variable (prefix `UFV_`); one image runs everywhere.

| Variable | Default | Meaning |
|----------|---------|---------|
| `UFV_DEVICE` | `auto` | `cuda` / `cpu` / `auto` |
| `UFV_MODE` | `accurate` | `accurate` = full ensemble; `fast` = single detector (low latency) |
| `UFV_USE_AMP` | `1` | fp16 mixed precision on GPU (≈2× faster) |
| `UFV_USE_COMPILE` | `0` | `torch.compile` backbones (extra speed, Linux/GPU) |
| `UFV_PORT` | `8000` | API port |
| `UFV_HOST` | `0.0.0.0` | API bind address |
| `UFV_API_KEY` | *(empty)* | if set, every request needs header `X-API-Key: <key>` |
| `UFV_DEFAULT_THRESHOLD` | `3.0` | fallback when `calibration.json` is absent |
| `UFV_TEMPORAL_WINDOW` | `3` | frames averaged in the live PyQt view |
| `UFV_RELOAD` | `0` | `1` enables uvicorn dev auto-reload |

---

## 10. Testing

```bash
python -m pytest tests/test_e2e.py -v -s
```
Validates both the SDK and the full HTTP path (model load → preprocess → fuse → threshold
→ response) on real good/defect images, plus malformed-input handling. Allow ~1 minute on
CPU (the first run downloads backbones).

---

## 11. Recalibrating for a new fabric (factory onboarding)

The shipped threshold is tuned on synthetic data. **Every new fabric family needs its own
calibration**, or you get false alarms / misses.

1. Collect ≥100 images of **known-good** production fabric → `data/train/good/`.
2. Collect a held-out `data/test/good/` and `data/test/defect/`.
3. Recalibrate (fast, no retrain):
   ```bash
   python calibrate.py
   ```
   or retrain fully for best accuracy: `python train.py`.
4. Restart the server / SDK to load the new `weights/calibration.json`.

Tuning the operating point: raise the threshold to reduce false alarms (fewer good rolls
rejected); lower it to catch subtler defects (fewer escapes). `train.py` prints AUROC/F1
to guide the choice.

---

## 12. Compiling the reports (LaTeX)

Reports live in `report/`. A LaTeX distribution is required. On this machine MiKTeX is at
`C:\Users\varsh\AppData\Local\Programs\MiKTeX\miktex\bin\x64` — add it to `PATH` if
`pdflatex` isn't found.

### Research paper (IEEE, self-contained bibliography)
```bash
cd report
pdflatex -interaction=nonstopmode research_paper.tex
pdflatex -interaction=nonstopmode research_paper.tex     # 2nd pass resolves references
```
Output: `report/research_paper.pdf`.

### IDP report (uses biblatex + glossaries)
```bash
cd report
pdflatex -interaction=nonstopmode IDP_Report.tex
bibtex   IDP_Report
makeglossaries IDP_Report
pdflatex -interaction=nonstopmode IDP_Report.tex
pdflatex -interaction=nonstopmode IDP_Report.tex
```
Output: `report/IDP_Report.pdf`.

Regenerate the paper's result figures/stats from the actual calibrated model:
```bash
python scripts/gen_paper_stats.py    # writes fig_score_distributions.png,
                                     # fig_confusion_matrix.png, paper_stats.json
```

Key report metadata lives at the top of `report/IDP_Report.tex` (title, student names +
USNs, guide, department). Team-member slots E/F only render when set, so leaving them
unset shows just the real members.

---

## 13. Git workflow

```bash
# Feature work happens on a branch:
git checkout -b my-change
# ... edit ...
git add -A
git commit -m "describe the change"
git push -u origin my-change          # open a PR from the printed link

# Reports are versioned on master under report/.
```
Weights and data are gitignored (too large / machine-specific); regenerate them with
`train.py` or `calibrate.py`.

---

## 14. Troubleshooting

| Symptom | Cause / Fix |
|---------|-------------|
| `OSError: not enough space on the disk` | Backbone download needs a few hundred MB + weight re-saves ~110 MB. Free ≥2 GB. |
| Everything scores as defective | Preprocessing mismatch or uncalibrated. Ensure inference and fit use the same preprocessing; run `calibrate.py`. |
| `charmap`/`cp1252` UnicodeEncodeError | Windows console. Prefix commands with `PYTHONIOENCODING=utf-8`. |
| `pdflatex: command not found` | Add the MiKTeX/TeX Live `bin` dir to `PATH`. |
| `File 'Figures/RVlogoVecW' not found` | College logos missing from `report/Figures/`; copy them from the report template. |
| CUDA not used (runs on CPU) | Driver/toolkit missing, or CPU-only torch installed. Reinstall torch from the CUDA index; set `UFV_DEVICE=cuda`. |
| First frame slow | Expected — lazy CUDA init + autotune. A warm-up frame runs at startup; subsequent frames are fast. |
| DINO download fails | Needs internet on first run to fetch `dino_vits8` (~80 MB), then cached under `~/.cache/torch/hub`. |

---

## 15. Known limitations

- **Synthetic-data results.** Reported AUROC/F1 are on a controlled synthetic set. Validate
  on real production fabric before field use.
- **GPU latency is projected**, not benchmarked on this machine (CPU-only dev box). Measure
  on your target GPU with `UFV_DEVICE=cuda python inference_test.py`.
- **One calibration per fabric family.** Very different weaves/materials need their own
  calibration (Section 11).
- **Fixed imaging assumptions.** Camera geometry and lighting must be stable between
  calibration and inference; large changes require recalibration.
- **Decision support, not a certified safety system.** Keep human QA in the loop.
