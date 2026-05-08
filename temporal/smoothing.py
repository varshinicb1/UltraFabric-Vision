from collections import deque
import numpy as np

class TemporalSmoother:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.history = deque(maxlen=window_size)
        
    def add(self, score):
        self.history.append(score)
        
    def get_smoothed_score(self):
        if not self.history:
            return 0.0
        return np.mean(self.history)

    def is_trend_increasing(self):
        if len(self.history) < 2:
            return False
        return self.history[-1] > self.history[0]
