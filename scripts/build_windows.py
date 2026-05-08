import os
import subprocess
import sys

def build_windows():
    print("Starting PyInstaller build for FabricAI Pro Suite...")
    
    # Define paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    main_script = os.path.join(base_dir, "main.py")
    
    if not os.path.exists(main_script):
        print(f"Error: {main_script} not found.")
        sys.exit(1)
        
    cmd = [
        "pyinstaller",
        "--name=FabricAI_Pro",
        "--windowed", # Don't open console
        "--onefile",
        f"--add-data=models{os.pathsep}models",
        f"--add-data=ui{os.pathsep}ui",
        "--hidden-import=PyQt5",
        "--hidden-import=torch",
        "--hidden-import=torchvision",
        "--hidden-import=pyqtgraph",
        main_script
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        print("Build completed successfully. Check the 'dist' folder for the .exe file.")
    except subprocess.CalledProcessError as e:
        print(f"Build failed: {e}")

if __name__ == "__main__":
    build_windows()
