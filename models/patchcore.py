import torch
import torch.nn as nn
import timm
from sklearn.neighbors import NearestNeighbors
import numpy as np
import cv2
import joblib
import os
from models.base import BaseModel


class PatchCore(BaseModel):
    """
    PatchCore anomaly detection using a Vision Transformer (ViT) Backbone.
    Extracts intermediate ViT patch features, builds a coreset memory bank
    of normal patches, and scores new images by k-NN distance.
    """

    def __init__(self):
        super().__init__()
        # Use Vision Transformer (ViT) instead of ResNet
        self.feature_extractor = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.feature_extractor.eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # Hooks to extract intermediate features from Transformer Blocks
        self.features = []

        def hook(module, input, output):
            self.features.append(output)

        # Hooking into late transformer blocks for rich semantic features
        self.feature_extractor.blocks[8].register_forward_hook(hook)
        self.feature_extractor.blocks[11].register_forward_hook(hook)

        self.memory_bank = None
        self.memory_bank_tensor = None
        self.knn = NearestNeighbors(n_neighbors=1, algorithm='auto', n_jobs=-1)

    # ------------------------------------------------------------------
    def forward(self, x):
        self.features = []
        with torch.no_grad():
            self.feature_extractor(x)

        pooled_features = []
        for feat in self.features:
            # feat shape: (B, 197, 768) where 197 = 1 class token + 196 patch tokens
            # Remove class token
            patches = feat[:, 1:, :] 
            B, N, C = patches.shape
            H = W = int(np.sqrt(N)) # For 224x224 and patch_size=16, H=W=14
            
            # Reshape sequence back into a 2D spatial grid
            spatial_feat = patches.transpose(1, 2).reshape(B, C, H, W)
            
            pooled = torch.nn.functional.avg_pool2d(spatial_feat, 3, 1, 1)
            pooled_features.append(pooled)
            
        return pooled_features

    # ------------------------------------------------------------------
    def fit(self, dataloader):
        """Build memory bank from normal training data."""
        features = []
        for x in dataloader:
            x = x.to(self.device)
            feats = self.forward(x)
            f2, f3 = feats
            f3 = torch.nn.functional.interpolate(f3, size=f2.shape[2:], mode='bilinear', align_corners=False)
            f_combined = torch.cat([f2, f3], dim=1)
            f_combined = f_combined.permute(0, 2, 3, 1).reshape(-1, f_combined.shape[1])
            features.append(f_combined.cpu().numpy())

        self.memory_bank = np.concatenate(features, axis=0)
        # Coreset subsampling
        idx = np.random.choice(len(self.memory_bank), min(10000, len(self.memory_bank)), replace=False)
        self.memory_bank = self.memory_bank[idx]
        self.knn.fit(self.memory_bank)
        if hasattr(self, 'device'):
            self.memory_bank_tensor = torch.tensor(self.memory_bank, device=self.device)

    # ------------------------------------------------------------------
    def save_memory_bank(self, path):
        """Persist the memory bank to disk."""
        joblib.dump(self.memory_bank, path)

    def load_memory_bank(self, path):
        """Load a previously saved memory bank."""
        if os.path.exists(path):
            self.memory_bank = joblib.load(path)
            self.knn.fit(self.memory_bank)
            
            # Pre-load to GPU if device is available
            device = self.device if hasattr(self, 'device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.memory_bank_tensor = torch.tensor(self.memory_bank, device=device)
            return True
        return False

    # ------------------------------------------------------------------
    def predict(self, x):
        x = x.to(self.device)
        if self.memory_bank is None:
            return 0.5, np.zeros((x.shape[2], x.shape[3]), dtype=np.float32)

        feats = self.forward(x)
        f2, f3 = feats
        f3 = torch.nn.functional.interpolate(f3, size=f2.shape[2:], mode='bilinear', align_corners=False)
        f_combined = torch.cat([f2, f3], dim=1)

        b, c, h, w = f_combined.shape
        f_combined = f_combined.permute(0, 2, 3, 1).reshape(-1, c)

        # GPU-accelerated nearest neighbor search
        if not hasattr(self, 'memory_bank_tensor') or self.memory_bank_tensor is None:
            self.memory_bank_tensor = torch.tensor(self.memory_bank, device=x.device)
            
        distances = torch.cdist(f_combined, self.memory_bank_tensor)
        min_distances, _ = torch.min(distances, dim=1)
        
        anomaly_map = min_distances.reshape(h, w).cpu().numpy()
        anomaly_score = float(np.max(anomaly_map))

        # Resize heatmap to original image size
        anomaly_map_resized = cv2.resize(anomaly_map, (x.shape[3], x.shape[2]))
        return anomaly_score, anomaly_map_resized
