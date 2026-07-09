import os
import cv2
import joblib
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from sklearn.neighbors import NearestNeighbors
from models.base import BaseModel, amp_context, batch_images


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
        self.memory_bank_tensor = None
        self.knn = NearestNeighbors(n_neighbors=1, algorithm='auto', n_jobs=-1)

        # When True, predict() also captures the last-layer self-attention in the
        # SAME forward pass and stashes it in self.last_attention (for dashboards),
        # avoiding a second full network forward. Off by default for max speed.
        self.capture_attention = False
        self.last_attention = None

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------
    def _extract_patch_tokens(self, x):
        """Return spatial patch tokens (B, N_patches, D) from the last
        attention block, skipping the [CLS] token."""
        with torch.no_grad(), amp_context(x.device):
            feats = self.model.get_intermediate_layers(x, n=1)[0]
            patch_tokens = feats[:, 1:, :]  # (B, N, D)
        return patch_tokens.float()

    def _extract_tokens_and_attn(self, x):
        """Single forward pass that returns BOTH the patch tokens (for scoring)
        and the last-layer self-attention grid (for visualization).

        This replaces the old pattern of calling get_intermediate_layers() and
        get_last_selfattention() separately — two full network passes — with one
        pass plus a cheap extra evaluation of only the final block. Falls back to
        the token-only path if the DINO API differs."""
        try:
            with torch.no_grad(), amp_context(x.device):
                m = self.model
                tok = m.prepare_tokens(x)
                n_blocks = len(m.blocks)
                attn = None
                for i, blk in enumerate(m.blocks):
                    if i == n_blocks - 1:
                        attn = blk(tok, return_attention=True)  # (B, heads, N, N)
                        tok = blk(tok)
                    else:
                        tok = blk(tok)
                tok = m.norm(tok)
                patch_tokens = tok[:, 1:, :].float()

            cls_attn = attn[:, :, 0, 1:]                 # CLS -> patches
            B, H, N = cls_attn.shape
            grid = int(N ** 0.5)
            attn_grid = cls_attn.reshape(B, H, grid, grid).float().cpu().numpy()
            return patch_tokens, attn_grid
        except Exception:
            # Robust fallback: tokens only, no attention.
            return self._extract_patch_tokens(x), None

    def get_attention_maps(self, x):
        """Returns the self-attention maps from the last layer for all heads.
        Kept for the (non-realtime) single-image upload path."""
        with torch.no_grad(), amp_context(x.device):
            attentions = self.model.get_last_selfattention(x)
            cls_attn = attentions[:, :, 0, 1:]
            B, H, N = cls_attn.shape
            grid_size = int(N**0.5)
            attn_grid = cls_attn.reshape(B, H, grid_size, grid_size)
            return attn_grid.float().cpu().numpy()

    def forward(self, x):
        """Return [CLS] token embedding (for compatibility)."""
        with torch.no_grad(), amp_context(x.device):
            features = self.model(x)
        return features

    # ------------------------------------------------------------------
    # Fit / Predict (anomaly detection)
    # ------------------------------------------------------------------
    def fit(self, dataloader):
        """Build a memory bank of patch-level features from normal data, then
        calibrate the score distribution."""
        all_features = []
        for batch in dataloader:
            x = batch_images(batch).to(self.device)
            patch_tokens = self._extract_patch_tokens(x)  # (B, N, D)
            patch_tokens = patch_tokens.reshape(-1, self.embed_dim)
            all_features.append(patch_tokens.cpu().numpy())

        self.memory_bank = np.concatenate(all_features, axis=0)
        # Coreset subsampling
        n_samples = min(10000, len(self.memory_bank))
        idx = np.random.choice(len(self.memory_bank), n_samples, replace=False)
        self.memory_bank = self.memory_bank[idx]
        self.knn.fit(self.memory_bank)
        self.memory_bank_tensor = torch.tensor(self.memory_bank, device=self.device, dtype=torch.float32)
        self.calibrate(dataloader)

    def calibrate(self, dataloader):
        """Record mean/std of raw anomaly scores over NORMAL data for z-scoring."""
        raw = []
        for batch in dataloader:
            x = batch_images(batch).to(self.device)
            raw.append(self._raw_score(x)[0])
        if raw:
            self.score_mean = float(np.mean(raw))
            self.score_std = float(np.std(raw) + 1e-6)

    def save_memory_bank(self, path):
        """Persist the memory bank and score calibration to disk."""
        joblib.dump({
            'memory_bank': self.memory_bank,
            'score_mean': self.score_mean,
            'score_std': self.score_std,
        }, path)

    def load_memory_bank(self, path):
        """Load a memory bank (new dict format or legacy raw-ndarray format)."""
        if os.path.exists(path):
            data = joblib.load(path)
            if isinstance(data, dict):
                self.memory_bank = data['memory_bank']
                self.score_mean = data.get('score_mean', 0.0)
                self.score_std = data.get('score_std', 1.0)
            else:
                self.memory_bank = data
            self.knn.fit(self.memory_bank)
            self.memory_bank_tensor = torch.tensor(self.memory_bank, device=self.device, dtype=torch.float32)
            return True
        return False

    def _raw_score(self, x):
        """Return (raw_score, anomaly_map@input_res) using GPU k-NN. Also updates
        self.last_attention when capture_attention is enabled (single pass)."""
        if self.capture_attention:
            patch_tokens, attn_grid = self._extract_tokens_and_attn(x)
            self.last_attention = attn_grid
        else:
            patch_tokens = self._extract_patch_tokens(x)
            self.last_attention = None

        B, N, D = patch_tokens.shape
        grid_size = int(N ** 0.5)

        # GPU-accelerated nearest-neighbour search (mirrors PatchCore). The old
        # sklearn kneighbors() ran on CPU over up to 10k vectors every frame and
        # was the single largest latency bottleneck.
        tokens_flat = patch_tokens.reshape(-1, D)
        if self.memory_bank_tensor is None:
            self.memory_bank_tensor = torch.tensor(self.memory_bank, device=x.device, dtype=torch.float32)
        distances = torch.cdist(tokens_flat, self.memory_bank_tensor)
        min_distances, _ = torch.min(distances, dim=1)

        anomaly_map = min_distances.reshape(grid_size, grid_size).cpu().numpy()
        raw_score = float(np.max(anomaly_map))
        anomaly_map_resized = cv2.resize(anomaly_map, (x.shape[3], x.shape[2]))
        return raw_score, anomaly_map_resized

    def predict(self, x):
        """
        Returns (anomaly_score, anomaly_heatmap).
        If no memory bank has been fitted, returns a dummy fallback.
        """
        if self.memory_bank is None:
            b, c, h, w = x.shape
            return 0.0, np.zeros((h, w), dtype=np.float32)
        raw_score, anomaly_map_resized = self._raw_score(x)
        return self._normalize_score(raw_score), anomaly_map_resized
