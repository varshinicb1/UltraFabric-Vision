import os
import torch
import torch.nn as nn
import numpy as np
from models.base import BaseModel, amp_context, batch_images


class ViTAutoencoder(BaseModel):
    """
    Vision-Transformer-based Autoencoder for anomaly detection.
    Trains on normal fabric images to reconstruct them. Anomalies produce
    high reconstruction error which serves as the anomaly heatmap.
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=256, depth=4, num_heads=8):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.num_patches = (img_size // patch_size) ** 2

        self.patch_embed = nn.Conv2d(in_chans, embed_dim,
                                     kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, batch_first=True,
            dim_feedforward=embed_dim * 4, dropout=0.1)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads, batch_first=True,
            dim_feedforward=embed_dim * 4, dropout=0.1)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=depth)

        self.decoder_pred = nn.Linear(embed_dim, patch_size * patch_size * in_chans)

    # ------------------------------------------------------------------
    def forward(self, x):
        # Encode
        x = self.patch_embed(x)
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)        # (B, N, C)
        x = x + self.pos_embed
        encoded = self.encoder(x)

        # Decode
        decoded = self.decoder(encoded, encoded)
        pred = self.decoder_pred(decoded)         # (B, N, P*P*3)

        # Reshape to image
        pred = pred.view(b, h, w, self.patch_size, self.patch_size, self.in_chans)
        pred = pred.permute(0, 5, 1, 3, 2, 4).contiguous()
        pred = pred.view(b, self.in_chans,
                         h * self.patch_size, w * self.patch_size)
        return pred

    # ------------------------------------------------------------------
    def save_weights(self, path):
        torch.save({
            'state_dict': self.state_dict(),
            'score_mean': self.score_mean,
            'score_std': self.score_std,
        }, path)

    def load_weights(self, path):
        if os.path.exists(path):
            data = torch.load(path, map_location=self.device, weights_only=False)
            if isinstance(data, dict) and 'state_dict' in data:
                self.load_state_dict(data['state_dict'])
                self.score_mean = data.get('score_mean', 0.0)
                self.score_std = data.get('score_std', 1.0)
            else:
                # Legacy format: raw state_dict
                self.load_state_dict(data)
            return True
        return False

    # ------------------------------------------------------------------
    def calibrate(self, dataloader):
        """Record mean/std of raw reconstruction-error scores over NORMAL data."""
        raw = []
        for batch in dataloader:
            x = batch_images(batch).to(self.device)
            raw.append(self._raw_score(x)[0])
        if raw:
            self.score_mean = float(np.mean(raw))
            self.score_std = float(np.std(raw) + 1e-6)

    def _raw_score(self, x):
        self.eval()
        with torch.no_grad(), amp_context(x.device):
            reconstructed = self.forward(x)

        error_map = torch.mean((x.float() - reconstructed.float()) ** 2, dim=1)  # (B, H, W)
        error_map = error_map.squeeze(0).cpu().numpy()
        raw_score = float(error_map.max())
        return raw_score, error_map

    def predict(self, x):
        raw_score, error_map = self._raw_score(x)
        return self._normalize_score(raw_score), error_map
