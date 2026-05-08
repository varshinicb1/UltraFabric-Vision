import os
import cv2
import numpy as np

def generate_fabric(width=224, height=224):
    """Generate a synthetic fabric texture using sine waves and noise."""
    base = np.zeros((height, width, 3), dtype=np.uint8)
    # Background color (e.g., beige)
    base[:] = (200, 220, 230)
    
    # Add vertical threads
    for x in range(0, width, 4):
        cv2.line(base, (x, 0), (x, height), (180, 200, 210), 1)
        
    # Add horizontal threads
    for y in range(0, height, 4):
        cv2.line(base, (0, y), (width, y), (180, 200, 210), 1)
        
    # Add noise
    noise = np.random.normal(0, 10, (height, width, 3)).astype(np.int16)
    fabric = np.clip(base + noise, 0, 255).astype(np.uint8)
    return fabric

def add_defect(image):
    """Add a synthetic defect (e.g., dark stain or hole)."""
    img_defect = image.copy()
    h, w = img_defect.shape[:2]
    
    # Randomly choose defect type
    if np.random.rand() > 0.5:
        # Stain
        center_x = np.random.randint(20, w-20)
        center_y = np.random.randint(20, h-20)
        radius = np.random.randint(10, 30)
        cv2.circle(img_defect, (center_x, center_y), radius, (50, 50, 70), -1)
        # Blur the stain to make it look embedded
        img_defect = cv2.GaussianBlur(img_defect, (15, 15), 0)
    else:
        # Tear/Line defect
        x1, y1 = np.random.randint(10, w-50), np.random.randint(10, h-50)
        x2, y2 = x1 + np.random.randint(20, 50), y1 + np.random.randint(-10, 10)
        cv2.line(img_defect, (x1, y1), (x2, y2), (255, 255, 255), 3)
        
    return img_defect

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    
    train_good_dir = os.path.join(data_dir, 'train', 'good')
    test_good_dir = os.path.join(data_dir, 'test', 'good')
    test_defect_dir = os.path.join(data_dir, 'test', 'defect')
    
    os.makedirs(train_good_dir, exist_ok=True)
    os.makedirs(test_good_dir, exist_ok=True)
    os.makedirs(test_defect_dir, exist_ok=True)
    
    print("Generating training data (normal fabric)...")
    for i in range(150):
        img = generate_fabric()
        cv2.imwrite(os.path.join(train_good_dir, f"train_good_{i:03d}.png"), img)
        
    print("Generating testing data (normal fabric)...")
    for i in range(30):
        img = generate_fabric()
        cv2.imwrite(os.path.join(test_good_dir, f"test_good_{i:03d}.png"), img)
        
    print("Generating testing data (defective fabric)...")
    for i in range(30):
        img = generate_fabric()
        img = add_defect(img)
        cv2.imwrite(os.path.join(test_defect_dir, f"test_defect_{i:03d}.png"), img)
        
    print(f"Synthetic dataset generated successfully at: {data_dir}")

if __name__ == "__main__":
    main()
