import os, sys
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.patchcore import PatchCore
from models.dino import DINOFeatureExtractor
from models.vit_autoencoder import ViTAutoencoder
from fusion.ensemble import EnsembleFusion
from app_utils.helpers import apply_heatmap, resize_and_pad

def visualize_ensemble(image_path, weights_dir='weights'):
    print(f"Analyzing {image_path} with full ensemble stack...")
    
    # 1. Load Image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not load image.")
        return
        
    img_resized = resize_and_pad(img, (224, 224))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    # Normalize
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = (tensor - mean) / std
    tensor = tensor.unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu')

    # 2. Load Models
    models = []
    # PatchCore
    pc = PatchCore().to_device()
    pc.load_memory_bank(os.path.join(weights_dir, 'patchcore_memory_bank.pkl'))
    models.append(pc)
    
    # DINO
    dino = DINOFeatureExtractor().to_device()
    dino.load_memory_bank(os.path.join(weights_dir, 'dino_memory_bank.pkl'))
    models.append(dino)
    
    # ViT-AE
    vae = ViTAutoencoder().to_device()
    vae.load_weights(os.path.join(weights_dir, 'vit_ae_weights.pth'))
    models.append(vae)
    
    fusion = EnsembleFusion(models)

    # 3. Get Individual Predictions
    results = []
    with torch.no_grad():
        for model in models:
            score, hmap = model.predict(tensor)
            results.append(hmap)
        
        fused_score, fused_hmap = fusion.predict(tensor)

    # 4. Plotting
    plt.figure(figsize=(18, 10))
    plt.suptitle("FabricAI Pro --- Ensemble Model Visualization (Exploded View)", fontsize=16, fontweight='bold')
    
    titles = ["Original Image", "PatchCore (CNN)", "DINO (Transformer)", "ViT-AE (Recon)", "Fused Ensemble"]
    images = [img_resized] + results + [fused_hmap]
    
    for i in range(5):
        plt.subplot(1, 5, i+1)
        if i == 0:
            plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        else:
            # Normalize and apply jet color map
            m = images[i]
            m_norm = (m - m.min()) / (m.max() - m.min() + 1e-5)
            plt.imshow(m_norm, cmap='jet')
        plt.title(titles[i])
        plt.axis('off')
        
    plt.tight_layout()
    save_path = "model_visualization.png"
    plt.savefig(save_path)
    print(f"Visualization saved to {save_path}")

if __name__ == "__main__":
    # Use one of our synthetic defects for visualization
    sample_img = "data/test/defect/test_defect_000.png"
    if os.path.exists(sample_img):
        visualize_ensemble(sample_img)
    else:
        print(f"Sample image {sample_img} not found. Run generate_synthetic_data.py first.")
