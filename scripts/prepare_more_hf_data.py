import os
import cv2
import numpy as np
from datasets import load_dataset
from PIL import Image
import tqdm

def add_synthetic_defect(image_np):
    """Add a synthetic defect (e.g., dark stain or hole) to a real fabric image."""
    img_defect = image_np.copy()
    h, w = img_defect.shape[:2]
    
    # Try multiple defects sometimes
    num_defects = np.random.randint(1, 4)
    for _ in range(num_defects):
        defect_type = np.random.rand()
        if defect_type > 0.6:
            # Stain
            center_x = np.random.randint(20, w-20)
            center_y = np.random.randint(20, h-20)
            radius = np.random.randint(10, 40)
            cv2.circle(img_defect, (center_x, center_y), radius, (np.random.randint(20, 80),)*3, -1)
            img_defect = cv2.GaussianBlur(img_defect, (15, 15), 0)
        elif defect_type > 0.3:
            # Tear/Line defect
            x1, y1 = np.random.randint(10, w-50), np.random.randint(10, h-50)
            x2, y2 = x1 + np.random.randint(20, 80), y1 + np.random.randint(-20, 20)
            cv2.line(img_defect, (x1, y1), (x2, y2), (200, 200, 200), np.random.randint(2, 6))
        else:
            # Hole / Burn mark
            center_x = np.random.randint(20, w-20)
            center_y = np.random.randint(20, h-20)
            radius = np.random.randint(5, 15)
            cv2.circle(img_defect, (center_x, center_y), radius, (10, 10, 10), -1)
            
    return img_defect

def main():
    print("Loading Cached HuggingFace dataset: SimTho/IndustrialTextileDataset...")
    # This will load instantly from cache
    dataset = load_dataset("SimTho/IndustrialTextileDataset")
    train_data = dataset['train']
    
    total_images = len(train_data)
    print(f"Total images found in dataset: {total_images}")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data_real_massive')
    
    train_good_dir = os.path.join(data_dir, 'train', 'good')
    test_good_dir = os.path.join(data_dir, 'test', 'good')
    test_defect_dir = os.path.join(data_dir, 'test', 'defect')
    
    os.makedirs(train_good_dir, exist_ok=True)
    os.makedirs(test_good_dir, exist_ok=True)
    os.makedirs(test_defect_dir, exist_ok=True)
    
    # Let's split: 4000 train, 946 test_good, 946 test_defect
    train_count = 4000
    test_good_count = (total_images - train_count) // 2
    
    print(f"Extracting {train_count} real fabric images for training...")
    for i in tqdm.tqdm(range(train_count)):
        img = train_data[i]['image'].convert('RGB')
        img.save(os.path.join(train_good_dir, f"real_train_{i:04d}.jpg"), quality=95)
        
    print(f"Extracting {test_good_count} real fabric images for testing (good)...")
    for i in tqdm.tqdm(range(train_count, train_count + test_good_count)):
        img = train_data[i]['image'].convert('RGB')
        img.save(os.path.join(test_good_dir, f"real_test_good_{i:04d}.jpg"), quality=95)
        
    print(f"Extracting {total_images - train_count - test_good_count} real fabric images with defects for testing...")
    for i in tqdm.tqdm(range(train_count + test_good_count, total_images)):
        img = train_data[i]['image'].convert('RGB')
        img_np = np.array(img)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        img_defect_bgr = add_synthetic_defect(img_bgr)
        cv2.imwrite(os.path.join(test_defect_dir, f"real_test_defect_{i:04d}.jpg"), img_defect_bgr)
        
    print(f"Massive real fabric dataset successfully extracted to: {data_dir}")
    print("Update 'train.py' to point to 'data_real_massive/train/good' to use this massive dataset.")

if __name__ == "__main__":
    main()
