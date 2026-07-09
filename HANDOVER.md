# UltraFabric-Vision — Handover

Real-time textile defect detection (PatchCore + DINO + ViT-Autoencoder ensemble),
validated on an NVIDIA RTX 4050. This is the top-level handover; see
`PROJECT_GUIDE.md` for the full step-by-step runbook and `DEPLOYMENT.md` for
production/integration details.

---

## 1. Status at handover

| Area | State |
|------|-------|
| Accuracy (synthetic benchmark) | AUROC 1.000, F1 1.000; calibrated z-score fusion |
| Real-time latency (RTX 4050, fp16) | **Accurate 37.3 ms / 26.8 FPS**, Fast 9.0 ms / 110 FPS, 641 MB VRAM |
| Modes | Accurate (ensemble) + Fast (single detector) across SDK/CLI/API/UI |
| Batch video inspection | Defect localization (metres + zones), min-defect-size filter |
| QC records | Per-batch PDF report, annotated video, defect map, batch history |
| Apps | React web dashboard + FastAPI backend; PyQt desktop app |
| Tests | 8/8 end-to-end pass (SDK + HTTP) |
| Docs | Research paper (13 pp) + IDP report (38 pp), both compiled |
| Deployment | GPU Docker image + compose, env-var config |

**Not done (needs physical inputs I can't produce):** validation on *real* factory
fabric, and per-fabric calibration. Everything is wired so both are one command.

---

## 2. How to run (Python 3.12 = GPU)

> Use `py -3.12` — it has CUDA torch (2.5.1+cu121) and sees the RTX 4050.
> Python 3.14 on this machine is CPU-only.

```bash
# Web demo (dashboard + backend on GPU)
$env:UFV_DEVICE="cuda"; py -3.12 backend_api.py         # terminal 1  -> :8000
cd web_app; npm run dev                                  # terminal 2  -> :5173
# open http://localhost:5173

# GPU benchmark
py -3.12 scripts/gpu_benchmark.py --amp both --out bench.md

# Batch video inspection (CLI)
py -3.12 scripts/generate_batch_video.py --batch B001 --defects "0.2,0.5,0.8" --out demo_batches/B001.mp4
$env:UFV_DEVICE="cuda"; py -3.12 scripts/video_inference.py --input demo_batches/B001.mp4 --batch B001 --meters 5 --out demo_batches/out

# Recalibrate to new fabric (required before real deployment)
py -3.12 calibrate.py
```

Dashboard tabs: **Live Stream Monitoring** (webcam), **Offline Batch Analysis**
(images), **Batch Video Inspection** (conveyor videos → defect map + zones +
annotated video + QC PDF + batch history). Fast/Accurate toggle in the header.

---

## 3. Key files

| File | Purpose |
|------|---------|
| `fabric_engine.py` | Embeddable `InferenceEngine` SDK (modes, size-gated boxes) |
| `backend_api.py` | FastAPI REST/WebSocket server (+ `/api/upload_video`, `/api/batch_report`) |
| `batch_inspect.py` | Shared batch report + defect-map + annotation logic |
| `qc_report.py` | Per-batch QC PDF generator |
| `predict.py` | Batch image CLI (CSV/JSON) |
| `calibrate.py` | Fast recalibration (per-model thresholds, fast-model pick) |
| `scripts/gpu_benchmark.py` | GPU/CPU latency benchmark |
| `scripts/video_inference.py` | Batch-video CLI |
| `scripts/generate_batch_video.py` | Synthetic conveyor demo video |
| `models/`, `fusion/`, `app_utils/` | Detectors, z-score fusion, config/helpers |
| `web_app/` | React dashboard |
| `deploy/` | GPU Dockerfile + compose |
| `report/` | LaTeX research paper + IDP report |

Config is env-var driven (prefix `UFV_`): `UFV_DEVICE`, `UFV_MODE`, `UFV_USE_AMP`,
`UFV_MIN_DEFECT_AREA`, `UFV_API_KEY`, … (full table in `PROJECT_GUIDE.md` §9).

---

## 4. What each fix was (for the viva)

1. **Preprocessing parity** — inference-only CLAHE/blur/letterbox put frames off
   the fitted manifold; removed → accuracy restored.
2. **Z-score calibration fusion** — three detectors on incomparable scales
   (8 / 46 / 0.25) standardized to a common z-score before averaging.
3. **GPU k-NN + AMP + single-pass attention** — removed the CPU sklearn
   bottleneck; ensemble now 26.8 FPS on RTX 4050.
4. **Minimum defect size + localization** — connected-region size filter;
   position/zone mapping for batch video.
5. **Two modes** — Accurate (ensemble) vs Fast (single detector).

---

## 5. Before selling to a mill (honest checklist)

- [ ] Record real fabric (good + defective) and run `calibrate.py` on it.
- [ ] Re-run `gpu_benchmark.py` on the deployment machine.
- [ ] Fix camera geometry + lighting; recalibrate if they change.
- [ ] Tune `UFV_MIN_DEFECT_AREA` and threshold to the customer's defect spec.
- [ ] Keep a human QC in the loop (decision support, not a certified safety system).

---

## 6. Repo

GitHub: `varshinicb1/UltraFabric-Vision` (branch `master`). All code, reports,
and deployment assets are committed.
