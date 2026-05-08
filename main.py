import sys
import os
import torch
from PyQt5.QtWidgets import QApplication
from ui.main_window import MainWindow

# Import models
from models.dino import DINOFeatureExtractor
from models.patchcore import PatchCore
from models.vit_autoencoder import ViTAutoencoder

def main():
    print("""
    ========================================================
    |             FABRIC AI PRO SUITE — GEN 2              |
    |          Industrial Defect Detection System          |
    ========================================================
    [STATUS] Initializing High-Performance AI Engine...
    """)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    from app_utils.config import config
    base_dir = config.base_dir
    weights_dir = os.path.join(base_dir, 'weights')
    
    print("Loading AI Models and weights...")
    models = []
    
    try:
        dino = DINOFeatureExtractor().to_device()
        dino_path = os.path.join(weights_dir, 'dino_memory_bank.pkl')
        if dino.load_memory_bank(dino_path):
            print("DINO loaded with memory bank.")
        else:
            print("DINO loaded (no memory bank found).")
        models.append(dino)
        
        patchcore = PatchCore().to_device()
        patchcore_path = os.path.join(weights_dir, 'patchcore_memory_bank.pkl')
        if patchcore.load_memory_bank(patchcore_path):
            print("PatchCore loaded with memory bank.")
        else:
            print("PatchCore loaded (no memory bank found).")
        models.append(patchcore)
        
        vit_ae = ViTAutoencoder().to_device()
        vit_ae_path = os.path.join(weights_dir, 'vit_ae_weights.pth')
        if vit_ae.load_weights(vit_ae_path):
            print("ViT Autoencoder loaded with weights.")
        else:
            print("ViT Autoencoder loaded (no weights found).")
        models.append(vit_ae)
        
    except Exception as e:
        print(f"Warning: Failed to load some models: {e}")

    app = QApplication(sys.argv)
    window = MainWindow(models)
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
