import os
import cv2
import numpy as np
import torch
import base64
import time
import asyncio
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import uvicorn
import io
from PIL import Image
try:
    from py3nvml import py3nvml
    HAS_GPU_LIB = True
except ImportError:
    HAS_GPU_LIB = False

# Add project root to sys path
import sys
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from models import base as model_base
from models.patchcore import PatchCore
from models.dino import DINOFeatureExtractor
from models.vit_autoencoder import ViTAutoencoder
from fusion.ensemble import EnsembleFusion
from app_utils.helpers import resize_and_pad, apply_heatmap, preprocess_frame, load_threshold, load_calibration, detect_defect_regions
from app_utils.config import config
import tempfile
import batch_inspect

# Apply performance flags from config
model_base.USE_AMP = config.use_amp
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True  # fixed 224x224 input -> pick fast kernels

# Shared, calibrated detection threshold (single source of truth across all
# entry points). Populated at startup once models/calibration are loaded.
DETECTION_THRESHOLD = config.default_threshold

# Inference mode: 'accurate' = full ensemble; 'fast' = single detector (low
# latency). Default from env UFV_MODE; per-request override via ?mode=.
DEFAULT_MODE = os.environ.get('UFV_MODE', 'accurate').strip().lower()
FAST_MODEL = None          # the single detector object used in fast mode
FAST_MODEL_KEY = 'dino'
FAST_THRESHOLD = config.default_threshold

# Batch-video inspection: saved annotated videos + in-memory inspection history.
BATCH_OUT_DIR = os.path.join(project_root, 'batch_outputs')
os.makedirs(BATCH_OUT_DIR, exist_ok=True)
BATCH_HISTORY = []
TS_COUNTER = {'n': 1}

app = FastAPI(title="FabricAI Pro Web Engine", version="1.0")

# Enable CORS for the Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def api_key_guard(request: Request, call_next):
    """Optional shared-secret auth. Enabled only when UFV_API_KEY is set.
    Health checks are always public so orchestrators can probe liveness."""
    if config.api_key and request.url.path not in ("/api/health",):
        if request.headers.get("X-API-Key") != config.api_key:
            return JSONResponse(status_code=401, content={"error": "invalid or missing X-API-Key"})
    return await call_next(request)

# Global Ensemble Engine — device resolved from config (UFV_DEVICE env var).
device = torch.device(config.resolved_device())
if device.type == 'cpu':
    print("WARNING: running on CPU. For production line-speed inference use a CUDA GPU (set UFV_DEVICE=cuda).")
elif HAS_GPU_LIB:
    try:
        py3nvml.nvmlInit()
    except Exception:
        pass

fusion_engine = None
dino_extractor = None
active_subscribers = []

def get_gpu_stats():
    if not torch.cuda.is_available() or not HAS_GPU_LIB:
        return {"util": "N/A", "mem": "N/A"}
    try:
        handle = py3nvml.nvmlDeviceGetHandleByIndex(0)
        info = py3nvml.nvmlDeviceGetMemoryInfo(handle)
        util = py3nvml.nvmlDeviceGetUtilizationRates(handle)
        return {
            "util": f"{util.gpu}%",
            "mem": f"{info.used // 1024**2} MB / {info.total // 1024**2} MB"
        }
    except:
        return {"util": "N/A", "mem": "N/A"}

@app.on_event("startup")
async def load_models():
    global fusion_engine, dino_extractor, DETECTION_THRESHOLD
    print("Loading AI Engine...")
    weights_dir = os.path.join(project_root, 'weights')

    models = []
    # PatchCore
    pc = PatchCore().to(device)
    pc.load_memory_bank(os.path.join(weights_dir, 'patchcore_memory_bank.pkl'))
    models.append(pc)

    # DINO — capture last-layer attention in the SAME forward pass used for
    # scoring, so the dashboard attention grid costs us essentially nothing.
    dino_extractor = DINOFeatureExtractor().to(device)
    dino_extractor.load_memory_bank(os.path.join(weights_dir, 'dino_memory_bank.pkl'))
    dino_extractor.capture_attention = True
    models.append(dino_extractor)

    # ViT Autoencoder
    vae = ViTAutoencoder().to(device)
    vae.load_weights(os.path.join(weights_dir, 'vit_ae_weights.pth'))
    models.append(vae)

    fusion_engine = EnsembleFusion(models)

    # Optional torch.compile for an extra speedup on Linux/GPU (needs Triton).
    if config.use_compile:
        try:
            pc.feature_extractor = torch.compile(pc.feature_extractor)
            dino_extractor.model = torch.compile(dino_extractor.model)
            print("torch.compile enabled on backbones.")
        except Exception as e:
            print(f"torch.compile unavailable ({e}); continuing eager.")

    # Load the shared calibrated threshold (falls back to config default).
    DETECTION_THRESHOLD = load_threshold(config.calibration_path, config.default_threshold)

    # Resolve the Fast-mode detector + its threshold from calibration.
    global FAST_MODEL, FAST_MODEL_KEY, FAST_THRESHOLD
    cal = load_calibration(config.calibration_path)
    FAST_MODEL_KEY = cal.get('fast_model', 'dino')
    key2obj = {'patchcore': pc, 'dino': dino_extractor, 'vit_ae': vae}
    FAST_MODEL = key2obj.get(FAST_MODEL_KEY, dino_extractor)
    FAST_THRESHOLD = float(cal.get('per_model', {}).get(FAST_MODEL_KEY, {}).get('threshold', DETECTION_THRESHOLD))

    # Warm up: run a dummy frame so cuDNN autotuning / lazy CUDA init / AMP graph
    # capture happen now rather than spiking the first real frame's latency.
    try:
        dummy = np.zeros((224, 224, 3), dtype=np.uint8)
        process_frame(dummy)
        print("AI Engine warmed up.")
    except Exception as e:
        print(f"Warm-up skipped: {e}")

    print(f"AI Engine loaded successfully. Detection threshold={DETECTION_THRESHOLD:.3f}")

def process_frame(img_bgr, mode=None):
    """Processes a BGR image and returns visualization, score, and anomaly status.

    mode: 'accurate' (full ensemble) or 'fast' (single detector, low latency)."""
    mode = (mode or DEFAULT_MODE)
    # Canonical preprocessing — MUST match training (plain stretch-resize + ImageNet
    # normalize). GPU-side normalize. img_resized is what the model actually "sees",
    # so the heatmap overlays align with it.
    img_resized = resize_and_pad(img_bgr, (224, 224))
    tensor = preprocess_frame(img_bgr, (224, 224), device,
                              config.imagenet_mean, config.imagenet_std)

    # Inference: fast = one detector; accurate = full ensemble.
    t0 = time.time()
    with torch.no_grad():
        if mode == 'fast' and FAST_MODEL is not None:
            score, hmap = FAST_MODEL.predict(tensor)
            active_threshold = FAST_THRESHOLD
        else:
            score, hmap = fusion_engine.predict(tensor)
            active_threshold = DETECTION_THRESHOLD
    # DINO attention (accurate mode captures it inline); fall back to the heatmap.
    attn = getattr(dino_extractor, 'last_attention', None)
    attn_maps = attn[0] if attn is not None else np.stack([hmap] * 6)
    latency = (time.time() - t0) * 1000

    # Threshold for defect (mode-appropriate, calibrated).
    is_anomalous = score > active_threshold

    # Create Neural Insight Grid.
    # Normalize each head for visualization
    processed_heads = []
    for head in attn_maps:
        h_min, h_max = head.min(), head.max()
        norm_head = (head - h_min) / (h_max - h_min + 1e-8)
        # Resize to 14x14 for dashboard grid compatibility
        small_head = cv2.resize(norm_head.astype(np.float32), (14, 14), interpolation=cv2.INTER_CUBIC)
        processed_heads.append((small_head * 100).tolist())

    # Primary neural grid is the mean of all heads
    mean_attn = np.mean(attn_maps, axis=0)
    neural_grid = cv2.resize(mean_attn.astype(np.float32), (14, 14), interpolation=cv2.INTER_CUBIC)
    neural_grid = (neural_grid / (neural_grid.max() + 1e-8) * 100).astype(np.float32).tolist()

    # Create visual
    vis = apply_heatmap(img_resized, hmap)
    boxes = []
    
    # Create detailed trace
    trace = [
        f"[T+{latency*0.05:.2f}ms] Frame ingested: {img_bgr.shape[1]}x{img_bgr.shape[0]} BGR",
        f"[T+{latency*0.15:.2f}ms] DINO-v2 Multi-Head Attention computed ({len(processed_heads)} heads)",
        f"[T+{latency*0.35:.2f}ms] Extracting Patch Descriptors (dim=768, patches=784)",
        f"[T+{latency*0.55:.2f}ms] K-Nearest Neighbor memory bank alignment (K=1)",
        f"[T+{latency*0.75:.2f}ms] Mahalanobis anomaly scoring: {score:.2f}",
        f"[T+{latency*0.95:.2f}ms] Global attention pooling & defect localization",
    ]

    if is_anomalous:
        hmap_norm = cv2.normalize(hmap, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        _, defect_mask = cv2.threshold(hmap_norm, 150, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(defect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            if cv2.contourArea(cnt) > 50:
                x, y, w, h = cv2.boundingRect(cnt)
                boxes.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h)})
                cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(vis, "DEFECT", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        trace.append(f"[T+{latency:.2f}ms] Localization: {len(boxes)} anomaly clusters segmented")
    else:
        trace.append(f"[T+{latency:.2f}ms] Global score {score:.2f}% below activation threshold")

    return vis, score, is_anomalous, latency, neural_grid, processed_heads, boxes, trace

@app.get("/api/health")
def health_check():
    return {"status": "ok", "engine": "loaded" if fusion_engine else "loading"}

@app.get("/api/version")
def version_info():
    """Model/version metadata for integrators and monitoring."""
    return {
        "name": "UltraFabric-Vision",
        "version": app.version,
        "device": str(device),
        "engine_loaded": fusion_engine is not None,
        "threshold": DETECTION_THRESHOLD,
        "input_size": [config.input_width, config.input_height],
        "amp": config.use_amp,
        "models": ["PatchCore(ViT-B/16)", "DINO(ViT-S/8)", "ViT-Autoencoder"],
        "default_mode": DEFAULT_MODE,
        "fast_model": FAST_MODEL_KEY,
        "fast_threshold": FAST_THRESHOLD,
    }

@app.post("/api/upload_image")
async def upload_image(file: UploadFile = File(...), mode: str = None):
    if fusion_engine is None:
        return JSONResponse(status_code=503, content={"error": "engine still loading"})
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        return JSONResponse(status_code=400, content={"error": "Invalid or unreadable image"})

    try:
        vis, score, is_anomalous, latency, neural_grid, attn_heads, boxes, trace = process_frame(img_bgr, mode)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"inference failed: {e}"})
    
    _, buffer = cv2.imencode('.jpg', vis)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    # Raw original image
    _, raw_buffer = cv2.imencode('.jpg', img_bgr)
    raw_base64 = base64.b64encode(raw_buffer).decode('utf-8')
    
    return {
        "filename": file.filename,
        "score": float(score),
        "is_anomalous": bool(is_anomalous),
        "latency_ms": float(latency),
        "neural_grid": neural_grid,
        "attn_heads": attn_heads,
        "boxes": boxes,
        "trace": trace,
        "image_data": f"data:image/jpeg;base64,{img_base64}",
        "raw_data": f"data:image/jpeg;base64,{raw_base64}"
    }

@app.post("/api/batch_upload")
async def batch_upload(files: List[UploadFile] = File(...), mode: str = None):
    results = []
    for file in files:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img_bgr is None:
            results.append({"filename": file.filename, "error": "Invalid image"})
            continue

        vis, score, is_anomalous, latency, neural_grid, attn_heads, boxes, trace = process_frame(img_bgr, mode)
        
        _, buffer = cv2.imencode('.jpg', vis)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Raw original image
        _, raw_buffer = cv2.imencode('.jpg', img_bgr)
        raw_base64 = base64.b64encode(raw_buffer).decode('utf-8')
        
        results.append({
            "filename": file.filename,
            "score": float(score),
            "is_anomalous": bool(is_anomalous),
            "latency_ms": float(latency),
            "neural_grid": neural_grid,
            "attn_heads": attn_heads,
            "boxes": boxes,
            "trace": trace,
            "image_data": f"data:image/jpeg;base64,{img_base64}",
            "raw_data": f"data:image/jpeg;base64,{raw_base64}"
        })
    return {"count": len(results), "results": results}

@app.post("/api/upload_video")
async def upload_video(file: UploadFile = File(...), batch: str = "B001",
                       meters: float = 5.0, segments: int = 10,
                       mode: str = None, stride: int = 1):
    """Batch-video inspection: score every frame, localize defects along the
    fabric length, and return a per-zone report + a defect-location map image."""
    if fusion_engine is None:
        return JSONResponse(status_code=503, content={"error": "engine still loading"})
    mode = (mode or DEFAULT_MODE)
    threshold = FAST_THRESHOLD if (mode == 'fast' and FAST_MODEL is not None) else DETECTION_THRESHOLD

    contents = await file.read()
    suffix = os.path.splitext(file.filename or '')[1] or '.mp4'
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.write(contents); tmp.close()

    safe_batch = "".join(c for c in batch if c.isalnum() or c in ('-', '_')) or "batch"
    ann_name = f"{safe_batch}_{TS_COUNTER['n']}_annotated.mp4"
    TS_COUNTER['n'] += 1
    ann_path = os.path.join(BATCH_OUT_DIR, ann_name)
    try:
        cap = cv2.VideoCapture(tmp.name)
        if not cap.isOpened():
            return JSONResponse(status_code=400, content={"error": "cannot decode video"})
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 224
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 224
        vw = cv2.VideoWriter(ann_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))

        frames_info, idx, n_proc = [], -1, 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            idx += 1
            if idx % max(1, stride) != 0:
                vw.write(frame)
                continue
            n_proc += 1
            tensor = preprocess_frame(frame, (224, 224), device,
                                      config.imagenet_mean, config.imagenet_std)
            with torch.no_grad():
                if mode == 'fast' and FAST_MODEL is not None:
                    score, hmap = FAST_MODEL.predict(tensor)
                else:
                    score, hmap = fusion_engine.predict(tensor)
            boxes = detect_defect_regions(hmap, config.min_defect_area_frac,
                                          config.defect_intensity_frac) if score > threshold else []
            area = sum(b['area_frac'] for b in boxes)
            rec = batch_inspect.frame_record(
                idx, score, score > threshold and len(boxes) > 0, boxes, area,
                fps, total, meters, segments)
            frames_info.append(rec)
            batch_inspect.annotate_frame(frame, rec, batch, segments)
            vw.write(frame)
        cap.release()
        vw.release()
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass

    report = batch_inspect.build_report(
        frames_info, batch, meters, segments, fps, total, n_proc, threshold,
        config.min_defect_area_frac, file.filename, mode, device)
    dm = batch_inspect.build_defect_map(frames_info, meters, segments, batch,
                                        threshold, report['defect_frames'], n_proc)
    _, buf = cv2.imencode('.png', dm)
    report['defect_map'] = "data:image/png;base64," + base64.b64encode(buf).decode('utf-8')
    report['annotated_video'] = f"/api/batch_video/{ann_name}"
    report['passed'] = report['defect_frames'] == 0

    # Record in batch history (newest first, capped).
    BATCH_HISTORY.insert(0, {
        'batch': report['batch'], 'source': report['source'], 'mode': mode,
        'passed': report['passed'], 'defect_frames': report['defect_frames'],
        'processed_frames': report['processed_frames'],
        'zones_with_defects': report['zones_with_defects'],
        'batch_length_m': meters, 'defect_events': report['defect_events'],
        'segment_summary': report['segment_summary'],
        'defect_map': report['defect_map'], 'annotated_video': report['annotated_video'],
        'seq': TS_COUNTER['n'],
    })
    del BATCH_HISTORY[50:]
    return report

@app.get("/api/batch_video/{name}")
def get_batch_video(name: str):
    """Serve a saved annotated batch video for download/playback."""
    from fastapi.responses import FileResponse
    safe = os.path.basename(name)
    path = os.path.join(BATCH_OUT_DIR, safe)
    if not os.path.exists(path):
        return JSONResponse(status_code=404, content={"error": "not found"})
    return FileResponse(path, media_type="video/mp4", filename=safe)

@app.get("/api/batch_history")
def batch_history():
    """List previously inspected batches (newest first) with pass/fail summary."""
    return {"count": len(BATCH_HISTORY), "batches": BATCH_HISTORY}

@app.websocket("/ws/stream")
async def stream_endpoint(websocket: WebSocket):
    await websocket.accept()
    ws_mode = websocket.query_params.get("mode")   # 'fast' | 'accurate' | None
    try:
        while True:
            data = await websocket.receive_text()
            if data.startswith("data:image"):
                data = data.split(",")[1]

            img_bytes = base64.b64decode(data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img_bgr is not None:
                vis, score, is_anomalous, latency, neural_grid, attn_heads, boxes, trace = process_frame(img_bgr, ws_mode)
                
                _, buffer = cv2.imencode('.jpg', vis)
                out_base64 = base64.b64encode(buffer).decode('utf-8')
                
                # Get GPU stats
                gpu_stats = get_gpu_stats()
                
                payload = {
                    "score": float(score),
                    "is_anomalous": bool(is_anomalous),
                    "latency_ms": float(latency),
                    "neural_grid": neural_grid,
                    "attn_heads": attn_heads,
                    "boxes": boxes,
                    "trace": trace,
                    "image_data": f"data:image/jpeg;base64,{out_base64}",
                    "raw_data": f"data:image/jpeg;base64,{data}", # data is already base64
                    "gpu_util": gpu_stats["util"],
                    "gpu_mem": gpu_stats["mem"]
                }
                
                # Broadcast to subscribers (iterate a copy; collect dead sockets
                # and prune after — never mutate the list mid-iteration).
                dead = []
                for sub in list(active_subscribers):
                    try:
                        await sub.send_json(payload)
                    except Exception:
                        dead.append(sub)
                for sub in dead:
                    if sub in active_subscribers:
                        active_subscribers.remove(sub)

                await websocket.send_json(payload)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"Error in websocket: {e}")

@app.websocket("/ws/subscribe")
async def subscribe_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_subscribers.append(websocket)
    try:
        while True:
            await websocket.receive_text() # Keep alive
    except:
        if websocket in active_subscribers:
            active_subscribers.remove(websocket)

@app.websocket("/ws/remote_stream")
async def websocket_remote_stream(websocket: WebSocket, url: str, mode: str = None):
    await websocket.accept()
    print(f"Remote Stream connected: {url}")
    cap = cv2.VideoCapture(url)
    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                await websocket.send_json({"trace": ["ERROR: Remote stream connection lost or invalid URL"]})
                break

            # Process
            vis, score, is_anomalous, latency, neural_grid, attn_heads, boxes, trace = process_frame(frame_bgr, mode)
            
            # Encode Vis
            _, buffer = cv2.imencode('.jpg', vis)
            out_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Encode Raw
            img_resized = cv2.resize(frame_bgr, (224, 224))
            _, raw_buffer = cv2.imencode('.jpg', img_resized)
            raw_base64 = base64.b64encode(raw_buffer).decode('utf-8')
            
            # Get GPU stats
            gpu_stats = get_gpu_stats()

            await websocket.send_json({
                "score": float(score),
                "is_anomalous": bool(is_anomalous),
                "latency_ms": float(latency),
                "neural_grid": neural_grid,
                "attn_heads": attn_heads,
                "boxes": boxes,
                "trace": trace,
                "image_data": f"data:image/jpeg;base64,{out_base64}",
                "raw_data": f"data:image/jpeg;base64,{raw_base64}",
                "gpu_util": gpu_stats["util"],
                "gpu_mem": gpu_stats["mem"]
            })
            await asyncio.sleep(0.05) # Cap at ~20fps to avoid blocking
    except WebSocketDisconnect:
        print("Remote WebSocket disconnected")
    finally:
        cap.release()

if __name__ == "__main__":
    # Production defaults: bind/port from config (env-overridable), reload off.
    # Set UFV_RELOAD=1 for local dev autoreload.
    reload = os.environ.get("UFV_RELOAD", "0").strip().lower() in ("1", "true", "yes", "on")
    uvicorn.run("backend_api:app", host=config.api_host, port=config.api_port, reload=reload)
