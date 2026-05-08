import os
import subprocess
import sys

def build_linux():
    print("Starting PyInstaller build for FabricAI Pro Suite on Linux...")
    
    # Define paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    main_script = os.path.join(base_dir, "main.py")
    
    if not os.path.exists(main_script):
        print(f"Error: {main_script} not found.")
        sys.exit(1)
        
    cmd = [
        sys.executable,
        "-m", "PyInstaller",
        "--name=FabricAI_Pro_Linux",
        "--windowed", 
        "--onefile",
        f"--add-data=models{os.pathsep}models",
        f"--add-data=ui{os.pathsep}ui",
        f"--add-data=weights{os.pathsep}weights",
        f"--add-data=app_utils{os.pathsep}app_utils",
        f"--add-data=api{os.pathsep}api",
        f"--add-data=fusion{os.pathsep}fusion",
        f"--add-data=temporal{os.pathsep}temporal",
        f"--add-data=experiments{os.pathsep}experiments",
        "--hidden-import=PyQt5",
        "--hidden-import=torch",
        "--hidden-import=torchvision",
        "--hidden-import=pyqtgraph",
        "--hidden-import=yaml",
        main_script
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        print("Build completed successfully. Check the 'dist' folder for the executable file.")
    except subprocess.CalledProcessError as e:
        print(f"Build failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    build_linux()
