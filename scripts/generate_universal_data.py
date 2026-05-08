import os
import cv2
import numpy as np

def generate_fabric_universal(width=224, height=224, color_type='beige', density=4):
    """Generate a synthetic fabric texture with color and density variations."""
    base = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Universal Color Palette
    colors = {
        'beige': (200, 220, 230),
        'blue': (180, 100, 80),
        'grey': (150, 150, 150),
        'white': (240, 240, 240),
        'dark': (40, 40, 50)
    }
    bg_color = colors.get(color_type, (200, 220, 230))
    base[:] = bg_color
    
    # Dynamic Thread Color (slightly darker than BG)
    thread_color = tuple(max(0, c - 20) for c in bg_color)
    
    # Add vertical threads
    for x in range(0, width, density):
        cv2.line(base, (x, 0), (x, height), thread_color, 1)
        
    # Add horizontal threads
    for y in range(0, height, density):
        cv2.line(base, (0, y), (width, y), thread_color, 1)
        
    # Add noise (simulating sensor grain)
    noise = np.random.normal(0, 10, (height, width, 3)).astype(np.int16)
    fabric = np.clip(base + noise, 0, 255).astype(np.uint8)
    return fabric

def add_defect_universal(image):
    """Add advanced synthetic defects (stains, holes, oil spots)."""
    img_defect = image.copy()
    h, w = img_defect.shape[:2]
    defect_type = np.random.choice(['stain', 'tear', 'oil', 'knot'])
    
    if defect_type == 'stain':
        center = (np.random.randint(20, w-20), np.random.randint(20, h-20))
        radius = np.random.randint(5, 25)
        cv2.circle(img_defect, center, radius, (40, 40, 60), -1)
        img_defect = cv2.GaussianBlur(img_defect, (15, 15), 0)
    elif defect_type == 'tear':
        x1, y1 = np.random.randint(10, w-50), np.random.randint(10, h-50)
        x2, y2 = x1 + np.random.randint(10, 40), y1 + np.random.randint(10, 40)
        cv2.line(img_defect, (x1, y1), (x2, y2), (255, 255, 255), 2)
    elif defect_type == 'oil':
        # Translucent dark spot
        overlay = img_defect.copy()
        center = (np.random.randint(20, w-20), np.random.randint(20, h-20))
        cv2.ellipse(overlay, center, (30, 15), np.random.randint(0, 180), 0, 360, (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.4, img_defect, 0.6, 0, img_defect)
    else: # knot
        center = (np.random.randint(20, w-20), np.random.randint(20, h-20))
        cv2.circle(img_defect, center, 3, (10, 10, 10), -1)

    return img_defect

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data_universal')
    
    os.makedirs(os.path.join(data_dir, 'train', 'good'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'test', 'good'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'test', 'defect'), exist_ok=True)
    
    color_options = ['beige', 'blue', 'grey', 'white', 'dark']
    
    print("Generating Universal Dataset...")
    # Train set (diverse colors/densities)
    for i in range(250):
        c = np.random.choice(color_options)
        d = np.random.choice([3, 4, 6])
        img = generate_fabric_universal(color_type=c, density=d)
        cv2.imwrite(os.path.join(data_dir, 'train', 'good', f"u_train_{i:03d}.png"), img)
        
    # Test set
    for i in range(50):
        c = np.random.choice(color_options)
        img = generate_fabric_universal(color_type=c)
        cv2.imwrite(os.path.join(data_dir, 'test', 'good', f"u_test_good_{i:03d}.png"), img)
        
        # Defect version
        img_defect = add_defect_universal(img)
        cv2.imwrite(os.path.join(data_dir, 'test', 'defect', f"u_test_defect_{i:03d}.png"), img_defect)
        
    print(f"Universal dataset generated at: {data_dir}")

if __name__ == "__main__":
    main()
