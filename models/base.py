import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseModel(nn.Module, ABC):
    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    @abstractmethod
    def forward(self, x):
        pass
        
    @abstractmethod
    def predict(self, x):
        """Returns anomaly score and heatmap."""
        pass
        
    def to_device(self):
        self.to(self.device)
        return self
