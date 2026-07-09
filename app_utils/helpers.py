import os
import json
import cv2
import numpy as np
import torch

# Cache of (mean, std) normalization tensors per device so we don't rebuild them
# every frame.
_NORM_CACHE = {}


def _get_norm_tensors(device, mean, std):
    key = (str(device), mean, std)
    cached = _NORM_CACHE.get(key)
    if cached is None:
        m = torch.tensor(mean, device=device).view(1, 3, 1, 1)
        s = torch.tensor(std, device=device).view(1, 3, 1, 1)
        cached = (m, s)
        _NORM_CACHE[key] = cached
    return cached


def resize_and_pad(image, target_size=(224, 224)):
    """Resize a BGR frame to the model input size.

    IMPORTANT: this MUST match the training transform (see train.py::get_transform
    and train_universal.py), which is a plain ``transforms.Resize((224, 224))`` —
    i.e. a straight stretch to 224x224 with NO aspect-ratio padding, NO CLAHE and
    NO blur. The previous CLAHE+blur+letterbox version put inference images in a
    completely different pixel distribution than the fitted memory banks / trained
    autoencoder, which was the primary cause of poor accuracy. Keep this identical
    to training preprocessing.
    """
    tw, th = target_size
    return cv2.resize(image, (tw, th), interpolation=cv2.INTER_LINEAR)


def preprocess_frame(image_bgr, target_size=(224, 224), device=None,
                     mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """BGR frame -> normalized model-ready tensor of shape (1, 3, H, W).

    Matches the training pipeline exactly (stretch-resize, RGB, /255, ImageNet
    normalize). Normalization runs on ``device`` to avoid per-frame CPU float math.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    img = resize_and_pad(image_bgr, target_size)                 # stretch to size
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)                    # BGR -> RGB
    t = torch.from_numpy(np.ascontiguousarray(img)).to(device, non_blocking=True)
    t = t.permute(2, 0, 1).unsqueeze(0).float().div_(255.0)      # (1,3,H,W) in [0,1]
    m, s = _get_norm_tensors(device, tuple(mean), tuple(std))
    t = (t - m) / s
    return t


def load_threshold(calibration_path, default=3.0):
    """Load the shared, calibrated detection threshold. Falls back to ``default``
    when no calibration file exists so the app still runs before calibration."""
    try:
        if calibration_path and os.path.exists(calibration_path):
            with open(calibration_path, 'r') as f:
                data = json.load(f)
            return float(data.get('threshold', default))
    except Exception:
        pass
    return float(default)


def load_calibration(calibration_path):
    """Load the full calibration record (ensemble threshold, per-model thresholds,
    recommended fast model). Returns an empty dict if unavailable."""
    try:
        if calibration_path and os.path.exists(calibration_path):
            with open(calibration_path, 'r') as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def apply_heatmap(image, anomaly_map, alpha=0.5):
    """Apply anomaly heatmap over the original image."""
    h, w = image.shape[:2]
    if anomaly_map.shape[:2] != (h, w):
        anomaly_map = cv2.resize(anomaly_map.astype(np.float32), (w, h))
        
    anomaly_map_norm = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-5)
    heatmap = cv2.applyColorMap(np.uint8(255 * anomaly_map_norm), cv2.COLORMAP_JET)
    output = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
    return output

def get_defect_boxes(anomaly_map, threshold=0.7):
    """Extract industrial bounding boxes from the anomaly heatmap."""
    # Normalize
    hmap_norm = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-5)
    
    # Thresholding to create binary mask
    _, mask = cv2.threshold((hmap_norm * 255).astype(np.uint8), int(threshold * 255), 255, cv2.THRESH_BINARY)
    
    # Morphological cleaning (remove tiny specs)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Find Contours (YOLO-style localization)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 200: # Industrial noise filter
            x, y, w, h = cv2.boundingRect(cnt)
            boxes.append((x, y, w, h))
    return boxes
