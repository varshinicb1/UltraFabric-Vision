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
import json
import time
from dataclasses import dataclass

import numpy as np
import torch

from models import base as model_base
from models.patchcore import PatchCore
from models.dino import DINOFeatureExtractor
from models.vit_autoencoder import ViTAutoencoder
from fusion.ensemble import EnsembleFusion
from app_utils.helpers import preprocess_frame, load_threshold, apply_heatmap, detect_defect_regions
from app_utils.config import config


@dataclass
class InspectionResult:
    score: float          # calibrated anomaly score (z-score scale)
    is_defect: bool       # score > threshold AND a defect region above min size
    threshold: float      # threshold used for this decision
    latency_ms: float     # pure model inference time (excludes decode/encode)
    heatmap: np.ndarray   # HxW float anomaly map at input resolution
    mode: str = "accurate"  # which mode produced this result
    boxes: list = None    # sized defect regions [{x,y,w,h,area_frac}] at heatmap res
    defect_area_frac: float = 0.0  # total defective area as a fraction of the frame


class InferenceEngine:
    """Loads the calibrated detectors once and scores frames.

    Two modes:
      * ``accurate`` (default) fuses all three detectors -- highest accuracy.
      * ``fast`` runs a single strong detector -- lowest latency (~3x fewer
        transformer forward passes), for real-time use on constrained hardware.

    Thread-affinity: create one engine per process; a single call is not
    re-entrant."""

    _KEY2ATTR = {'patchcore': 'patchcore', 'dino': 'dino', 'vit_ae': 'vae'}

    def __init__(self, weights_dir=None, device=None, capture_attention=False,
                 warmup=True, mode='accurate'):
        self.weights_dir = weights_dir or config.weights_dir
        self.device = torch.device(device or config.resolved_device())
        model_base.USE_AMP = config.use_amp
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True

        self._load()
        self.dino.capture_attention = capture_attention
        self.mode = mode

        # Load calibration: ensemble threshold + per-model Fast-mode thresholds.
        self.threshold = load_threshold(config.calibration_path, config.default_threshold)
        self.calibrated = os.path.exists(config.calibration_path)
        self.model_thresholds = {}
        self.fast_model = 'dino'
        if self.calibrated:
            try:
                with open(config.calibration_path) as f:
                    cal = json.load(f)
                self.fast_model = cal.get('fast_model', 'dino')
                for k, v in cal.get('per_model', {}).items():
                    self.model_thresholds[k] = float(v.get('threshold', config.default_threshold))
            except Exception:
                pass
        # Fast mode needs its detector's attention iff visualization is requested.
        if capture_attention and self.fast_model == 'dino':
            self.dino.capture_attention = True

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

    def predict_bgr(self, img_bgr, mode=None):
        """Score a BGR (OpenCV) image. Returns an InspectionResult.

        ``mode`` overrides the engine default: ``'accurate'`` fuses all detectors,
        ``'fast'`` runs only the recommended single detector for lowest latency."""
        mode = mode or self.mode
        tensor = preprocess_frame(img_bgr, (config.input_width, config.input_height),
                                  self.device, config.imagenet_mean, config.imagenet_std)
        t0 = time.time()
        with torch.no_grad():
            if mode == 'fast':
                model = getattr(self, self._KEY2ATTR.get(self.fast_model, 'dino'))
                score, heatmap = model.predict(tensor)
                threshold = self.model_thresholds.get(self.fast_model, self.threshold)
            else:
                score, heatmap = self.ensemble.predict(tensor)
                threshold = self.threshold
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        latency_ms = (time.time() - t0) * 1000.0

        # Size- and coverage-gated defect regions: a frame counts as defective
        # only if the score exceeds the threshold AND there is a LOCALIZED
        # anomalous region above the minimum size. A whole-frame anomaly (wrong
        # material / blank frame) yields no boxes and is not called a defect.
        boxes, coverage = [], 0.0
        if score > threshold:
            boxes, coverage = detect_defect_regions(
                heatmap, config.min_defect_area_frac, config.defect_intensity_frac,
                config.max_defect_coverage_frac, return_coverage=True)
        area = round(sum(b['area_frac'] for b in boxes), 5)
        return InspectionResult(
            score=float(score),
            is_defect=bool(score > threshold and len(boxes) > 0),
            threshold=float(threshold),
            latency_ms=latency_ms,
            heatmap=heatmap,
            mode=mode,
            boxes=boxes,
            defect_area_frac=area,
        )

    def overlay(self, img_bgr, result, alpha=0.5):
        """Return a heatmap-overlaid visualization at the input resolution."""
        import cv2
        base = cv2.resize(img_bgr, (config.input_width, config.input_height))
        return apply_heatmap(base, result.heatmap, alpha=alpha)
