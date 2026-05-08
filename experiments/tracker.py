import os
import json
import time
from datetime import datetime
import pandas as pd
from app_utils.config import config

class ExperimentTracker:
    def __init__(self):
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join(config.experiments_dir, self.experiment_id)
        os.makedirs(self.log_dir, exist_ok=True)
        self.logs = []
        self.metrics = {}

    def log_inference(self, model_name, inference_time, anomaly_score, is_anomalous):
        log_entry = {
            "timestamp": time.time(),
            "model": model_name,
            "inference_time_ms": inference_time * 1000,
            "anomaly_score": float(anomaly_score),
            "is_anomalous": bool(is_anomalous)
        }
        self.logs.append(log_entry)

    def log_evaluation(self, auroc, f1_score, avg_inference_time):
        self.metrics = {
            "auroc": float(auroc),
            "f1_score": float(f1_score),
            "avg_inference_time_ms": float(avg_inference_time * 1000)
        }
        
    def save(self):
        # Save detailed logs
        if self.logs:
            df = pd.DataFrame(self.logs)
            df.to_csv(os.path.join(self.log_dir, "inference_logs.csv"), index=False)
            
        # Save metrics
        if self.metrics:
            with open(os.path.join(self.log_dir, "metrics.json"), 'w') as f:
                json.dump(self.metrics, f, indent=4)
                
        return self.log_dir
