"""
UltraFabric-Vision — embeddable inference SDK.
===============================================
A single, importable class that loads the calibrated ensemble and scores frames.
Use this to integrate defect detection directly into line-control software,
edge devices, or batch QA — no HTTP required.

    from fabric_engine import InferenceEngine
    engine = InferenceEngine()            # loads weights + calibrated threshold
    result = engine.predict_bgr(frame)    # frame: HxWx3 BGR numpy array (OpenCV)
    if result.is_defect:
        reject(result.score, result.heatmap)

The REST server (backend_api.py) and the batch CLI (predict.py) are thin wrappers
around this same engine, so every integration path shares one calibrated pipeline.
"""

import os
import time
from dataclasses import dataclass

import numpy as np
import torch

from models import base as model_base
from models.patchcore import PatchCore
from models.dino import DINOFeatureExtractor
from models.vit_autoencoder import ViTAutoencoder
from fusion.ensemble import EnsembleFusion
from app_utils.helpers import preprocess_frame, load_threshold, apply_heatmap
from app_utils.config import config


@dataclass
class InspectionResult:
    score: float          # calibrated fused anomaly score (z-score scale)
    is_defect: bool       # score > threshold
    threshold: float      # threshold used for this decision
    latency_ms: float     # pure model inference time (excludes decode/encode)
    heatmap: np.ndarray   # HxW float anomaly map at input resolution


class InferenceEngine:
    """Loads the calibrated ensemble once and scores frames. Thread-affinity:
    create one engine per process; a single engine call is not re-entrant."""

    def __init__(self, weights_dir=None, device=None, capture_attention=False, warmup=True):
        self.weights_dir = weights_dir or config.weights_dir
        self.device = torch.device(device or config.resolved_device())
        model_base.USE_AMP = config.use_amp
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True

        self._load()
        self.dino.capture_attention = capture_attention
        self.threshold = load_threshold(config.calibration_path, config.default_threshold)

        self.calibrated = os.path.exists(config.calibration_path)
        if warmup:
            self._warmup()

    def _load(self):
        pc = PatchCore().to(self.device)
        pc.load_memory_bank(os.path.join(self.weights_dir, 'patchcore_memory_bank.pkl'))

        dino = DINOFeatureExtractor().to(self.device)
        dino.load_memory_bank(os.path.join(self.weights_dir, 'dino_memory_bank.pkl'))

        vae = ViTAutoencoder().to(self.device)
        vae.load_weights(os.path.join(self.weights_dir, 'vit_ae_weights.pth'))

        self.patchcore, self.dino, self.vae = pc, dino, vae
        self.ensemble = EnsembleFusion([pc, dino, vae])

        if config.use_compile and self.device.type == 'cuda':
            try:
                pc.feature_extractor = torch.compile(pc.feature_extractor)
                dino.model = torch.compile(dino.model)
            except Exception:
                pass

    def _warmup(self):
        try:
            self.predict_bgr(np.zeros((config.input_height, config.input_width, 3), dtype=np.uint8))
        except Exception:
            pass

    def predict_bgr(self, img_bgr):
        """Score a BGR (OpenCV) image. Returns an InspectionResult."""
        tensor = preprocess_frame(img_bgr, (config.input_width, config.input_height),
                                  self.device, config.imagenet_mean, config.imagenet_std)
        t0 = time.time()
        with torch.no_grad():
            score, heatmap = self.ensemble.predict(tensor)
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        latency_ms = (time.time() - t0) * 1000.0
        return InspectionResult(
            score=float(score),
            is_defect=bool(score > self.threshold),
            threshold=float(self.threshold),
            latency_ms=latency_ms,
            heatmap=heatmap,
        )

    def overlay(self, img_bgr, result, alpha=0.5):
        """Return a heatmap-overlaid visualization at the input resolution."""
        import cv2
        base = cv2.resize(img_bgr, (config.input_width, config.input_height))
        return apply_heatmap(base, result.heatmap, alpha=alpha)
