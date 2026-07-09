# UltraFabric-Vision — Deployment & Integration Guide

Real-time fabric-defect detection for textile production lines. Three-model
ensemble (PatchCore ViT-B/16 + DINO ViT-S/8 + ViT-Autoencoder) with calibrated
z-score fusion and a single tuned decision threshold.

---

## 1. What you get

| Surface | File | Use it for |
|---|---|---|
| REST API server | `backend_api.py` | Live line integration, web dashboard, remote cameras |
| Embeddable SDK | `fabric_engine.py` | Direct integration into edge/PLC/line-control Python |
| Batch CLI | `predict.py` | Offline QA of image folders → CSV/JSON for MES |
| Calibration | `calibrate.py` | Tune the model to *your* fabric (required, see §5) |

All four share one calibrated pipeline, so a decision is identical across them.

---

## 2. Hardware requirements

- **Production:** NVIDIA GPU (≥6 GB VRAM; T4 / RTX 3060 or better), CUDA 12.x
  driver. This is what delivers line-speed latency.
- **CPU-only:** works for evaluation/batch QA but is ~10–20× slower — not for
  real-time line use.
- ~4 GB disk for model weights + backbone caches.

> Latency note: on CPU a full frame is ~0.5–1.2 s. On a modern GPU the same
> ensemble runs an order of magnitude faster (the GPU k-NN + fp16 AMP + single
> forward pass are GPU-only wins). **Benchmark on your target GPU before quoting
> a line speed** — see §6.

---

## 3. Quick start (Docker + GPU) — recommended

Prerequisites: NVIDIA driver + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

```bash
# 1. Put trained + calibrated weights in ./weights (see §5)
#    weights/patchcore_memory_bank.pkl, dino_memory_bank.pkl,
#    vit_ae_weights.pth, calibration.json

# 2. Build & run
docker compose -f deploy/docker-compose.gpu.yml up --build

# 3. Verify
curl http://localhost:8000/api/health      # {"status":"ok","engine":"loaded"}
curl http://localhost:8000/api/version      # model + threshold metadata
```

The server binds `0.0.0.0:8000`. Weights and data are bind-mounted, so you can
swap in factory-calibrated weights and restart without rebuilding.

### Bare-metal (no Docker)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-api.txt
UFV_DEVICE=cuda python backend_api.py
```

---

## 4. Configuration (environment variables)

The same image runs everywhere; configure via env vars:

| Var | Default | Meaning |
|---|---|---|
| `UFV_DEVICE` | `auto` | `cuda` / `cpu` / `auto` |
| `UFV_USE_AMP` | `1` | fp16 autocast on GPU (≈2× faster) |
| `UFV_USE_COMPILE` | `0` | `torch.compile` backbones (extra speed, Linux/GPU) |
| `UFV_PORT` | `8000` | API port |
| `UFV_HOST` | `0.0.0.0` | API bind address |
| `UFV_API_KEY` | *(empty)* | if set, every call needs header `X-API-Key: <key>` |
| `UFV_DEFAULT_THRESHOLD` | `3.0` | fallback if `calibration.json` missing |
| `UFV_TEMPORAL_WINDOW` | `3` | frames averaged in the PyQt live view |

---

## 5. Calibrate to YOUR fabric (required before production)

The shipped threshold (`weights/calibration.json`) is tuned on synthetic data.
Real fabric has its own "normal" texture, so **you must recalibrate** on samples
from the actual line, or you'll get false alarms / misses.

1. Collect images of **known-good** fabric → `data/train/good/` (100+).
2. Collect a held-out set → `data/test/good/` and `data/test/defect/`.
3. Choose one:
   - **Recalibrate only** (reuses existing memory banks, fast):
     `python calibrate.py`
   - **Full retrain** (rebuilds memory banks on your fabric — best accuracy):
     `python train.py`

Both write per-model score stats into the weight files and a tuned threshold to
`weights/calibration.json`. Restart the server to pick it up.

> Tune the operating point to your business cost: raising the threshold reduces
> false alarms (fewer good rolls rejected); lowering it catches more subtle
> defects (fewer escapes). `train.py` reports AUROC/F1 to guide this.

---

## 6. Benchmark latency on your GPU

```bash
# Full benchmark: Accurate vs Fast, per-component, AMP on/off, + Markdown/JSON export
python scripts/gpu_benchmark.py --amp both --out bench.md

# Quick check
UFV_DEVICE=cuda python inference_test.py
```
`gpu_benchmark.py` reports mean/p50/p95 latency, FPS, and peak VRAM with correct
GPU timing (warm-up + `cuda.synchronize`). Quote the **Accurate**-mode mean/FPS on
your GPU in a spec sheet — not the CPU dev number. The generated `bench.md` table
can be pasted straight into the report.

---

## 7. Integration patterns

### A. REST (any language)
```bash
curl -X POST http://LINE_HOST:8000/api/upload_image \
     -F "file=@frame.jpg"
# -> { "score": 24.1, "is_anomalous": true, "latency_ms": 38.2, "boxes": [...] }
```
Live camera streaming uses the WebSocket `ws://LINE_HOST:8000/ws/stream`
(send base64 JPEG frames, receive scored results + overlay). See `web_app/` for
a reference dashboard client.

### B. Embed the SDK (Python edge / PLC gateway)
```python
import cv2
from fabric_engine import InferenceEngine

engine = InferenceEngine()                 # loads once
frame = cv2.imread("frame.jpg")            # or a grabbed camera frame (BGR)
r = engine.predict_bgr(frame)
if r.is_defect:
    plc.reject(reason=f"anomaly {r.score:.1f}")   # your actuator hook
```

### C. Batch QA → MES/CSV
```bash
python predict.py --input ./roll_2024_06_09/ --out report.csv \
                  --save-overlays ./defect_snaps
```

---

## 8. Health, monitoring, scaling

- `GET /api/health` — liveness (public even when API key is set).
- `GET /api/version` — device, threshold, model list (for dashboards).
- One GPU is saturated by one worker; **scale with replicas/GPUs**, not uvicorn
  workers (each worker loads a full copy of the models into VRAM).
- The Docker image ships a `HEALTHCHECK` for orchestrators.

---

## 9. Testing

```bash
python -m pytest tests/test_e2e.py -v -s   # SDK + full HTTP path, real inference
```

---

## 10. Limitations & honest disclaimers (read before selling)

- **Not validated on real factory fabric yet** — only synthetic data. Validate
  accuracy on real rolls per fabric type before any guarantee.
- **One model per fabric family.** Very different textures/patterns need their
  own calibration (§5). Plan a per-SKU calibration step.
- **Lighting/camera consistency matters.** Anomaly scoring assumes inference
  conditions match the calibration images. Fix camera, lighting, and web/roll
  geometry, and recalibrate if they change.
- **Decision support, not a safety system.** Keep human QA in the loop; this is
  not certified for safety-critical rejection without oversight.
- Throughput/latency claims must come from *your* GPU benchmark (§6).
```
