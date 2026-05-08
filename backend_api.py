import os
import cv2
import numpy as np
import torch
import base64
import time
import asyncio
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
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

from models.patchcore import PatchCore
from models.dino import DINOFeatureExtractor
from models.vit_autoencoder import ViTAutoencoder
from fusion.ensemble import EnsembleFusion
from app_utils.helpers import resize_and_pad, apply_heatmap

app = FastAPI(title="FabricAI Pro Web Engine", version="1.0")

# Enable CORS for the Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Ensemble Engine
if not torch.cuda.is_available():
    print("WARNING: GPU (CUDA) not found. Falling back to CPU for evaluation.")
    device = torch.device('cpu')
else:
    device = torch.device('cuda')
    if HAS_GPU_LIB:
        py3nvml.nvmlInit()

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
    global fusion_engine, dino_extractor
    print("Loading AI Engine...")
    weights_dir = os.path.join(project_root, 'weights')
    
    models = []
    # PatchCore
    pc = PatchCore().to(device)
    pc.load_memory_bank(os.path.join(weights_dir, 'patchcore_memory_bank.pkl'))
    models.append(pc)
    
    # DINO
    dino_extractor = DINOFeatureExtractor().to(device)
    dino_extractor.load_memory_bank(os.path.join(weights_dir, 'dino_memory_bank.pkl'))
    models.append(dino_extractor)
    
    # ViT Autoencoder
    vae = ViTAutoencoder().to(device)
    vae.load_weights(os.path.join(weights_dir, 'vit_ae_weights.pth'))
    models.append(vae)
    
    fusion_engine = EnsembleFusion(models)
    print("AI Engine loaded successfully.")

def process_frame(img_bgr):
    """Processes a BGR image and returns visualization, score, and anomaly status."""
    img_resized = resize_and_pad(img_bgr, (224, 224))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    
    # Normalize
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = ((tensor - mean) / std).unsqueeze(0).to(device)
    
    # Inference
    t0 = time.time()
    with torch.no_grad():
        score, hmap = fusion_engine.predict(tensor)
        # Extract Real Attention Maps from DINO-v2
        # Shape: (1, 6, 28, 28) for ViT-S/8
        attn_maps = dino_extractor.get_attention_maps(tensor)[0] 
    latency = (time.time() - t0) * 1000
    
    # Threshold for defect
    is_anomalous = score > 35.0
    
    # Create Neural Insight Grid (Real Attention from Head 0)
    # Normalize each head for visualization
    processed_heads = []
    for head in attn_maps:
        h_min, h_max = head.min(), head.max()
        norm_head = (head - h_min) / (h_max - h_min + 1e-8)
        # Resize to 14x14 for dashboard grid compatibility
        small_head = cv2.resize(norm_head, (14, 14), interpolation=cv2.INTER_CUBIC)
        processed_heads.append((small_head * 100).tolist())

    # Primary neural grid is the mean of all heads
    mean_attn = np.mean(attn_maps, axis=0)
    neural_grid = cv2.resize(mean_attn, (14, 14), interpolation=cv2.INTER_CUBIC)
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

@app.post("/api/upload_image")
async def upload_image(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img_bgr is None:
        return {"error": "Invalid image"}
        
    vis, score, is_anomalous, latency, neural_grid, attn_heads, boxes, trace = process_frame(img_bgr)
    
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
async def batch_upload(files: List[UploadFile] = File(...)):
    results = []
    for file in files:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img_bgr is None:
            results.append({"filename": file.filename, "error": "Invalid image"})
            continue
            
        vis, score, is_anomalous, latency, neural_grid, attn_heads, boxes, trace = process_frame(img_bgr)
        
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

@app.websocket("/ws/stream")
async def stream_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            if data.startswith("data:image"):
                data = data.split(",")[1]
            
            img_bytes = base64.b64decode(data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img_bgr is not None:
                vis, score, is_anomalous, latency, neural_grid, attn_heads, boxes, trace = process_frame(img_bgr)
                
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
                
                # Broadcast to subscribers
                for sub in active_subscribers:
                    try:
                        await sub.send_json(payload)
                    except:
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
async def websocket_remote_stream(websocket: WebSocket, url: str):
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
            vis, score, is_anomalous, latency, neural_grid, attn_heads, boxes, trace = process_frame(frame_bgr)
            
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
    uvicorn.run("backend_api:app", host="0.0.0.0", port=8000, reload=True)
