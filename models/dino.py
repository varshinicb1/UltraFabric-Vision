import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from sklearn.neighbors import NearestNeighbors
from models.base import BaseModel


class DINOFeatureExtractor(BaseModel):
    """
    DINO ViT-S/8 feature extractor for anomaly detection.
    Uses k-NN against a fitted memory bank of "normal" features
    to produce per-patch anomaly scores — identical approach to PatchCore
    but with a self-supervised backbone.
    """

    def __init__(self, use_vits8=True):
        super().__init__()
        arch = 'dino_vits8' if use_vits8 else 'dino_vitb16'
        self.model = torch.hub.load('facebookresearch/dino:main', arch)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.embed_dim = 384 if use_vits8 else 768
        self.patch_size = 8 if use_vits8 else 16

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # Memory bank for anomaly detection (fitted during training)
        self.memory_bank = None
        self.knn = NearestNeighbors(n_neighbors=1, algorithm='auto', n_jobs=-1)

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------
    def _extract_patch_tokens(self, x):
        """Return spatial patch tokens (B, N_patches, D) from the last
        attention block, skipping the [CLS] token."""
        with torch.no_grad():
            feats = self.model.get_intermediate_layers(x, n=1)[0]
            patch_tokens = feats[:, 1:, :]  # (B, N, D)
        return patch_tokens

    def get_attention_maps(self, x):
        """Returns the self-attention maps from the last layer for all heads."""
        with torch.no_grad():
            attentions = self.model.get_last_selfattention(x)
            cls_attn = attentions[:, :, 0, 1:] 
            B, H, N = cls_attn.shape
            grid_size = int(N**0.5)
            attn_grid = cls_attn.reshape(B, H, grid_size, grid_size)
            return attn_grid.cpu().numpy()

    def forward(self, x):
        """Return [CLS] token embedding (for compatibility)."""
        with torch.no_grad():
            features = self.model(x)
        return features

    # ------------------------------------------------------------------
    # Fit / Predict (anomaly detection)
    # ------------------------------------------------------------------
    def fit(self, dataloader):
        """Build a memory bank of patch-level features from normal data."""
        all_features = []
        for x in dataloader:
            x = x.to(self.device)
            patch_tokens = self._extract_patch_tokens(x)  # (B, N, D)
            patch_tokens = patch_tokens.reshape(-1, self.embed_dim)
            all_features.append(patch_tokens.cpu().numpy())

        self.memory_bank = np.concatenate(all_features, axis=0)
        # Coreset subsampling
        n_samples = min(10000, len(self.memory_bank))
        idx = np.random.choice(len(self.memory_bank), n_samples, replace=False)
        self.memory_bank = self.memory_bank[idx]
        self.knn.fit(self.memory_bank)

    def save_memory_bank(self, path):
        """Persist the memory bank to disk."""
        import joblib
        joblib.dump(self.memory_bank, path)

    def load_memory_bank(self, path):
        """Load a previously saved memory bank."""
        import joblib
        import os
        if os.path.exists(path):
            self.memory_bank = joblib.load(path)
            self.knn.fit(self.memory_bank)
            return True
        return False

    def predict(self, x):
        """
        Returns (anomaly_score, anomaly_heatmap).
        If no memory bank has been fitted, returns a dummy fallback.
        """
        x = x.to(self.device)
        if self.memory_bank is None:
            b, c, h, w = x.shape
            return 0.0, np.zeros((h, w), dtype=np.float32)

        patch_tokens = self._extract_patch_tokens(x)  # (1, N, D)
        B, N, D = patch_tokens.shape
        grid_size = int(N ** 0.5)  # e.g. 28 for 224/8

        tokens_flat = patch_tokens.reshape(-1, D).cpu().numpy()
        distances, _ = self.knn.kneighbors(tokens_flat)

        anomaly_map = distances.reshape(grid_size, grid_size)
        anomaly_score = float(np.max(anomaly_map))

        # Upsample to original input resolution
        import cv2
        anomaly_map_resized = cv2.resize(anomaly_map, (x.shape[3], x.shape[2]))
        return anomaly_score, anomaly_map_resized
