import os
import yaml
import sys


def _env_bool(name, default):
    v = os.environ.get(name)
    if v is None:
        return default
    return str(v).strip().lower() in ('1', 'true', 'yes', 'on')


def _env_float(name, default):
    v = os.environ.get(name)
    try:
        return float(v) if v is not None else default
    except (TypeError, ValueError):
        return default


def _env_int(name, default):
    v = os.environ.get(name)
    try:
        return int(v) if v is not None else default
    except (TypeError, ValueError):
        return default


class Config:
    """Runtime configuration. Every field can be overridden by an environment
    variable (prefix ``UFV_``) so the same image deploys unchanged across dev,
    staging and the factory floor — twelve-factor style."""

    def __init__(self, config_path=None):
        # 'auto' resolves to cuda when available, else cpu (see resolved_device()).
        self.device = os.environ.get('UFV_DEVICE', 'auto')
        self.input_width = _env_int('UFV_INPUT_WIDTH', 224)
        self.input_height = _env_int('UFV_INPUT_HEIGHT', 224)
        self.batch_size = _env_int('UFV_BATCH_SIZE', 1)

        # API server
        self.api_host = os.environ.get('UFV_HOST', '0.0.0.0')
        self.api_port = _env_int('UFV_PORT', 8000)
        # Optional shared-secret header (X-API-Key). Empty => auth disabled.
        self.api_key = os.environ.get('UFV_API_KEY', '')

        # ImageNet normalization stats — MUST match what the models were trained/
        # fitted with (see train.py / train_universal.py). Kept here as the single
        # source of truth so training and inference can never drift apart.
        self.imagenet_mean = (0.485, 0.456, 0.406)
        self.imagenet_std = (0.229, 0.224, 0.225)

        # ---- Inference performance flags ----
        # AMP (fp16 autocast) roughly halves transformer latency on CUDA with no
        # measurable accuracy change for these feature-distance models.
        self.use_amp = _env_bool('UFV_USE_AMP', True)
        # torch.compile can add another speedup but needs Triton, which is flaky on
        # Windows. OFF by default; turn on for Linux industrial (GPU) deployments.
        self.use_compile = _env_bool('UFV_USE_COMPILE', False)

        # ---- Detection thresholding ----
        # Fallback threshold on the (normalized) fused score, used only when a
        # calibrated weights/calibration.json is not present. Calibrate for real
        # deployments via `python calibrate.py`.
        self.default_threshold = _env_float('UFV_DEFAULT_THRESHOLD', 3.0)
        # Minimum defect size, as a fraction of the frame area. Connected
        # anomalous regions smaller than this are ignored, so tiny texture
        # specks / single-patch noise are not reported as defects. 0.004 of a
        # 224x224 frame is ~200 px. Raise to be stricter about defect size.
        self.min_defect_area_frac = _env_float('UFV_MIN_DEFECT_AREA', 0.004)
        # Relative intensity (0-1) at which the fused heatmap is thresholded when
        # extracting defect regions for localization / sizing.
        self.defect_intensity_frac = _env_float('UFV_DEFECT_INTENSITY', 0.55)
        # If a single-frame anomaly covers MORE than this fraction of the frame,
        # it is not a localized defect but a whole-frame anomaly (wrong material,
        # blank/black frame, out-of-distribution) -- no defect boxes are drawn.
        # This prevents "boxes everywhere" on non-fabric input.
        self.max_defect_coverage_frac = _env_float('UFV_MAX_COVERAGE', 0.45)
        # Temporal smoothing window over the fused score. Small so real transient
        # defects are not averaged away at line speed.
        self.temporal_window = _env_int('UFV_TEMPORAL_WINDOW', 3)

        # Paths
        # Detect if running as a PyInstaller bundle
        if hasattr(sys, '_MEIPASS'):
            self.base_dir = sys._MEIPASS
        else:
            self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            
        self.experiments_dir = os.path.join(self.base_dir, 'experiments_data')
        self.reports_dir = os.path.join(self.base_dir, 'reports_output')
        self.weights_dir = os.path.join(self.base_dir, 'weights')
        self.calibration_path = os.path.join(self.weights_dir, 'calibration.json')
        
        # Ensure dirs exist (only if not frozen, or to some other writable path)
        if not hasattr(sys, '_MEIPASS'):
            os.makedirs(self.experiments_dir, exist_ok=True)
            os.makedirs(self.reports_dir, exist_ok=True)
        
        config_path = config_path or os.environ.get('UFV_CONFIG')
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                loaded_config = yaml.safe_load(f)
                for k, v in loaded_config.items():
                    setattr(self, k, v)

    def resolved_device(self):
        """Resolve 'auto'/'cuda'/'cpu' to an actual torch device string, falling
        back to CPU if CUDA was requested but is unavailable."""
        import torch
        want = str(self.device).lower()
        if want in ('auto', 'cuda', 'gpu'):
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return 'cpu'


config = Config()
