import os
import torch
import cv2
import numpy as np
import time
import psutil
from tqdm import tqdm
import torch.multiprocessing as mp

# Add the project root to sys.path
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models.patchcore import PatchCore
from models.dino import DINOFeatureExtractor
from models.vit_autoencoder import ViTAutoencoder
from fusion.ensemble import EnsembleFusion
from app_utils.helpers import resize_and_pad

def get_system_stats():
    cpu_usage = psutil.cpu_percent()
    memory_usage = psutil.virtual_memory().percent
    return cpu_usage, memory_usage

def stress_test_worker(worker_id, stop_event, results_queue):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights_dir = os.path.join(project_root, 'weights')
    
    # Load Ensemble Stack
    models = []
    pc = PatchCore().to(device)
    pc.load_memory_bank(os.path.join(weights_dir, 'patchcore_memory_bank.pkl'))
    models.append(pc)
    
    dino = DINOFeatureExtractor().to(device)
    dino.load_memory_bank(os.path.join(weights_dir, 'dino_memory_bank.pkl'))
    models.append(dino)
    
    vae = ViTAutoencoder().to(device)
    vae.load_weights(os.path.join(weights_dir, 'vit_ae_weights.pth'))
    models.append(vae)
    
    fusion = EnsembleFusion(models)
    
    # Dummy input
    dummy_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img_resized = resize_and_pad(dummy_img, (224, 224))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = ((tensor - mean) / std).unsqueeze(0).to(device)
    
    count = 0
    start_time = time.time()
    
    while not stop_event.is_set():
        with torch.no_grad():
            score, _ = fusion.predict(tensor)
        count += 1
        if count % 10 == 0:
            results_queue.put((worker_id, count))
            
    results_queue.put((worker_id, count))

def run_stress_test(duration=30, num_workers=1):
    print(f"=== FabricAI Pro Suite Stress Test ===")
    print(f"Duration: {duration}s | Workers: {num_workers}")
    print(f"Initializing models...")
    
    stop_event = mp.Event()
    results_queue = mp.Queue()
    workers = []
    
    for i in range(num_workers):
        p = mp.Process(target=stress_test_worker, args=(i, stop_event, results_queue))
        p.start()
        workers.append(p)
    
    start_time = time.time()
    total_inferences = 0
    worker_counts = {i: 0 for i in range(num_workers)}
    
    try:
        with tqdm(total=duration, desc="Stress Testing") as pbar:
            last_time = start_time
            while time.time() - start_time < duration:
                # Update progress bar
                current_time = time.time()
                elapsed = current_time - last_time
                if elapsed >= 1.0:
                    pbar.update(int(elapsed))
                    last_time = current_time
                
                # Check results queue
                while not results_queue.empty():
                    wid, count = results_queue.get()
                    worker_counts[wid] = count
                
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nInterrupting stress test...")
    
    stop_event.set()
    for p in workers:
        p.join()
        
    total_inferences = sum(worker_counts.values())
    total_duration = time.time() - start_time
    fps = total_inferences / total_duration
    
    cpu_usage, mem_usage = get_system_stats()
    
    print("\n" + "="*40)
    print("      STRESS TEST RESULTS")
    print("="*40)
    print(f"Total Inferences:  {total_inferences}")
    print(f"Avg Throughput:    {fps:.2f} FPS")
    print(f"Final CPU Usage:   {cpu_usage}%")
    print(f"Final RAM Usage:   {mem_usage}%")
    print("="*40)

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    # Increase workers if on a high-spec machine
    run_stress_test(duration=20, num_workers=1)
