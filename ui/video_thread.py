import cv2
import torch
import numpy as np
import time
from PyQt5.QtCore import QThread, pyqtSignal
from api.input_stream import VideoStream
from fusion.ensemble import EnsembleFusion
from temporal.smoothing import TemporalSmoother
from app_utils.helpers import resize_and_pad, apply_heatmap, get_defect_boxes, preprocess_frame, load_threshold
from app_utils.config import config

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray, float, bool, float)

    def __init__(self, source=0, models=[]):
        super().__init__()
        self.stream = VideoStream(source)
        self.fusion = EnsembleFusion(models) if models else None
        self.smoother = TemporalSmoother(window_size=config.temporal_window)
        self._run_flag = True
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Shared, calibrated threshold on the fused z-score (single source of truth
        # with the web backend). Falls back to config default before calibration.
        self.threshold = load_threshold(config.calibration_path, config.default_threshold)
        
    def run(self):
        while self._run_flag:
            start_time = time.time()
            ret, cv_img = self.stream.read()
            if not ret:
                break
                
            smoothed_score = 0.0
            is_anomalous = False
            display_img = cv_img.copy()

            if self.fusion:
                # Preprocess
                input_tensor = self.preprocess(cv_img)
                # Predict
                score, heatmap = self.fusion.predict(input_tensor)
                
                # Temporal smoothing
                self.smoother.add(score)
                smoothed_score = self.smoother.get_smoothed_score()
                
                is_anomalous = smoothed_score > self.threshold
                
                # Visual enhancements for "World's Best"
                if is_anomalous:
                    # 1. Overlay Heatmap
                    display_img = apply_heatmap(cv_img, heatmap, alpha=0.6)
                    
                    # 2. Extract and draw Bounding Boxes (YOLO-style)
                    boxes = get_defect_boxes(heatmap, threshold=0.6)
                    for (bx, by, bw, bh) in boxes:
                        # Rescale box from 224x224 to original resolution
                        h, w = cv_img.shape[:2]
                        rx, ry = w/224, h/224
                        cv2.rectangle(display_img, (int(bx*rx), int(by*ry)), (int((bx+bw)*rx), int((by+bh)*ry)), (0, 0, 255), 3)
                        cv2.putText(display_img, "DEFECT", (int(bx*rx), int(by*ry)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # 3. Global warning border
                    h, w = cv_img.shape[:2]
                    cv2.rectangle(display_img, (0, 0), (w, h), (0, 0, 255), 10)
                else:
                    display_img = cv_img
            
            # Signal the UI (Image, Score, IsAnomalous, Latency)
            latency_ms = (time.time() - start_time) * 1000
            self.change_pixmap_signal.emit(display_img, float(smoothed_score), bool(is_anomalous), float(latency_ms))
            
            # Control frame rate to approx 30 FPS
            elapsed = time.time() - start_time
            sleep_time = max(0.001, 0.033 - elapsed)
            time.sleep(sleep_time)
            
        self.stream.release()

    def preprocess(self, img):
        # Canonical preprocessing shared with training and the web backend.
        return preprocess_frame(img, (224, 224), self.device,
                                config.imagenet_mean, config.imagenet_std)

    def stop(self):
        self._run_flag = False
        self.wait()
