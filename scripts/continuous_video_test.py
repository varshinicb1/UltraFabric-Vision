import cv2
import torch
import numpy as np
import os
import time
from tqdm import tqdm
from models.dino import DINOFeatureExtractor
from models.patchcore import PatchCore
from models.vit_autoencoder import ViTAutoencoder
from fusion.ensemble import EnsembleFusion
from app_utils.helpers import resize_and_pad, apply_heatmap, get_defect_boxes

def process_video(input_path, output_path, weights_dir='weights'):
    print(f"🎬 Initializing Continuous Video Processor for: {input_path}")
    
    # 1. Load Models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models = []
    
    # PatchCore
    pc = PatchCore().to(device)
    pc.load_memory_bank(os.path.join(weights_dir, 'patchcore_memory_bank.pkl'))
    models.append(pc)
    
    # DINO
    dino = DINOFeatureExtractor().to(device)
    dino.load_memory_bank(os.path.join(weights_dir, 'dino_memory_bank.pkl'))
    models.append(dino)
    
    # ViT-AE
    vae = ViTAutoencoder().to(device)
    vae.load_weights(os.path.join(weights_dir, 'vit_ae_weights.pth'))
    models.append(vae)
    
    fusion = EnsembleFusion(models)
    
    # 2. Video Capture
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"⚙️ Processing {total_frames} frames...")
    
    for _ in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret:
            break
            
        # AI Processing
        img_pre = resize_and_pad(frame, (224, 224))
        img_rgb = cv2.cvtColor(img_pre, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = ((tensor - mean) / std).unsqueeze(0).to(device)
        
        score, hmap = fusion.predict(tensor)
        
        # Visualize
        res_frame = frame.copy()
        if score > 22.0: # Universal Threshold
            # Heatmap Overlay
            res_frame = apply_heatmap(res_frame, hmap, alpha=0.5)
            # YOLO-style boxes
            boxes = get_defect_boxes(hmap, threshold=0.6)
            for (bx, by, bw, bh) in boxes:
                rx, ry = width/224, height/224
                cv2.rectangle(res_frame, (int(bx*rx), int(by*ry)), (int((bx+bw)*rx), int((by+bh)*ry)), (0, 0, 255), 4)
                cv2.putText(res_frame, f"STAIN: {score:.1f}", (int(bx*rx), int(by*ry)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Telemetry Overlay
        cv2.putText(res_frame, f"FabricAI Pro | Score: {score:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        out.write(res_frame)
        
    cap.release()
    out.release()
    print(f"✅ Continuous video saved to: {output_path}")

if __name__ == "__main__":
    input_video = "data/sample.mp4"
    output_video = "test_results/continuous_defect_analysis.mp4"
    os.makedirs('test_results', exist_ok=True)
    
    if os.path.exists(input_video):
        process_video(input_video, output_video)
    else:
        print(f"Error: {input_video} not found. Run generate_synthetic_data.py first.")
