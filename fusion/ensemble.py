import numpy as np
import cv2


class EnsembleFusion:
    """
    Fuses predictions from multiple anomaly detection models.
    Handles heatmaps of different spatial resolutions by resizing
    them to a common size before weighted averaging.
    """

    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights if weights else [1.0 / len(models)] * len(models)

    def predict(self, x):
        scores = []
        heatmaps = []
        target_h, target_w = x.shape[2], x.shape[3]

        for model in self.models:
            score, heatmap = model.predict(x)
            scores.append(score)
            # Ensure all heatmaps are the same spatial size
            if heatmap.shape[0] != target_h or heatmap.shape[1] != target_w:
                heatmap = cv2.resize(heatmap.astype(np.float32),
                                     (target_w, target_h))
            heatmaps.append(heatmap)

        final_score = float(np.average(scores, weights=self.weights))
        final_heatmap = np.average(heatmaps, axis=0, weights=self.weights)

        return final_score, final_heatmap
