import cv2
import numpy as np

def resize_and_pad(image, target_size=(224, 224)):
    """Resize image and pad to keep aspect ratio with CLAHE enhancement."""
    # 1. Industrial Preprocessing (Inspired by YOLOv3 Fabric Repo)
    # CLAHE for contrast enhancement
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    image_enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    # Gaussian Blur to remove sensor noise
    image_enhanced = cv2.GaussianBlur(image_enhanced, (3,3), 0)

    # 2. Resize and Pad logic
    h, w = image_enhanced.shape[:2]
    tw, th = target_size
    scale = min(tw/w, th/h)
    nw, nh = int(w * scale), int(h * scale)
    image_resized = cv2.resize(image_enhanced, (nw, nh))
    
    pad_w = (tw - nw) // 2
    pad_h = (th - nh) // 2
    
    padded_image = cv2.copyMakeBorder(image_resized, pad_h, th - nh - pad_h, pad_w, tw - nw - pad_w, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return padded_image

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
