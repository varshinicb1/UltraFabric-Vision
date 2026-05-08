import os
import cv2
import numpy as np
from datasets import load_dataset
from PIL import Image

def add_synthetic_defect(image_np):
    """Add a synthetic defect (e.g., dark stain or hole) to a real fabric image."""
    img_defect = image_np.copy()
    h, w = img_defect.shape[:2]
    
    if np.random.rand() > 0.5:
        # Stain
        center_x = np.random.randint(20, w-20)
        center_y = np.random.randint(20, h-20)
        radius = np.random.randint(10, 30)
        cv2.circle(img_defect, (center_x, center_y), radius, (50, 50, 70), -1)
        img_defect = cv2.GaussianBlur(img_defect, (15, 15), 0)
    else:
        # Tear/Line defect
        x1, y1 = np.random.randint(10, w-50), np.random.randint(10, h-50)
        x2, y2 = x1 + np.random.randint(20, 50), y1 + np.random.randint(-10, 10)
        cv2.line(img_defect, (x1, y1), (x2, y2), (200, 200, 200), 3)
        
    return img_defect

def main():
    print("Loading HuggingFace dataset: SimTho/IndustrialTextileDataset...")
    dataset = load_dataset("SimTho/IndustrialTextileDataset")
    train_data = dataset['train']
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data_real')
    
    train_good_dir = os.path.join(data_dir, 'train', 'good')
    test_good_dir = os.path.join(data_dir, 'test', 'good')
    test_defect_dir = os.path.join(data_dir, 'test', 'defect')
    
    os.makedirs(train_good_dir, exist_ok=True)
    os.makedirs(test_good_dir, exist_ok=True)
    os.makedirs(test_defect_dir, exist_ok=True)
    
    print("Extracting 500 real fabric images for training...")
    for i in range(500):
        img = train_data[i]['image'].convert('RGB')
        img.save(os.path.join(train_good_dir, f"real_train_{i:04d}.png"))
        
    print("Extracting 50 real fabric images for testing (good)...")
    for i in range(500, 550):
        img = train_data[i]['image'].convert('RGB')
        img.save(os.path.join(test_good_dir, f"real_test_good_{i:04d}.png"))
        
    print("Extracting 50 real fabric images and applying simulated defects for testing...")
    for i in range(550, 600):
        img = train_data[i]['image'].convert('RGB')
        img_np = np.array(img)
        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # Add defect
        img_defect_bgr = add_synthetic_defect(img_bgr)
        
        cv2.imwrite(os.path.join(test_defect_dir, f"real_test_defect_{i:04d}.png"), img_defect_bgr)
        
    print(f"Real fabric dataset successfully extracted to: {data_dir}")
    print("You can now update 'train.py' to point to 'data_real/train/good' if you wish to train on real textures.")

if __name__ == "__main__":
    main()
