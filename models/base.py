import contextlib
import torch
import torch.nn as nn
from abc import ABC, abstractmethod

# Global switch for fp16 autocast on CUDA. Halves transformer latency with no
# meaningful accuracy change for these feature-distance detectors. Can be toggled
# from config at startup.
USE_AMP = True


def amp_context(device):
    """Autocast context for fast fp16 inference on CUDA (no-op on CPU)."""
    if USE_AMP and torch.cuda.is_available() and torch.device(device).type == 'cuda':
        return torch.autocast(device_type='cuda', dtype=torch.float16)
    return contextlib.nullcontext()


def batch_images(batch):
    """Return the image tensor from a dataloader batch, handling both plain-tensor
    loaders (train.py) and (image, label) loaders like ImageFolder (train_universal.py)."""
    if isinstance(batch, (list, tuple)):
        return batch[0]
    return batch


class BaseModel(nn.Module, ABC):
    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Per-model score calibration. Anomaly scores from PatchCore (L2 distance),
        # DINO (patch distance) and the ViT-AE (reconstruction MSE) live on wildly
        # different scales; storing the mean/std of each model's score over NORMAL
        # data lets predict() emit comparable z-scores so the ensemble average is
        # meaningful. Defaults (0, 1) => passthrough until calibrated.
        self.score_mean = 0.0
        self.score_std = 1.0

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def predict(self, x):
        """Returns anomaly score and heatmap."""
        pass

    def _normalize_score(self, raw_score):
        """Convert a raw anomaly score to a calibrated z-score."""
        std = self.score_std if self.score_std and self.score_std > 1e-6 else 1.0
        return (raw_score - self.score_mean) / std

    def to_device(self):
        self.to(self.device)
        return self
