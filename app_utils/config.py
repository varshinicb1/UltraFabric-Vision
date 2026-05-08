import os
import yaml
import sys

class Config:
    def __init__(self, config_path=None):
        self.device = 'cuda' # Default, will be overwritten by torch.cuda.is_available() check in main
        self.input_width = 224
        self.input_height = 224
        self.batch_size = 1
        
        # Paths
        # Detect if running as a PyInstaller bundle
        if hasattr(sys, '_MEIPASS'):
            self.base_dir = sys._MEIPASS
        else:
            self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            
        self.experiments_dir = os.path.join(self.base_dir, 'experiments_data')
        self.reports_dir = os.path.join(self.base_dir, 'reports_output')
        
        # Ensure dirs exist (only if not frozen, or to some other writable path)
        if not hasattr(sys, '_MEIPASS'):
            os.makedirs(self.experiments_dir, exist_ok=True)
            os.makedirs(self.reports_dir, exist_ok=True)
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                loaded_config = yaml.safe_load(f)
                for k, v in loaded_config.items():
                    setattr(self, k, v)

config = Config()
