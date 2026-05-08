import os
import urllib.request
import zipfile
import tarfile

def download_file(url, dest_path):
    print(f"Downloading {url}...")
    try:
        urllib.request.urlretrieve(url, dest_path)
        print(f"Successfully downloaded to {dest_path}")
        return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return False

def extract_file(file_path, extract_to):
    print(f"Extracting {file_path}...")
    try:
        if file_path.endswith('.zip'):
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        elif file_path.endswith('.tar.gz') or file_path.endswith('.tgz') or file_path.endswith('.tar.xz'):
            with tarfile.open(file_path, 'r:*') as tar_ref:
                tar_ref.extractall(extract_to)
        print(f"Extracted to {extract_to}")
    except Exception as e:
        print(f"Extraction failed: {e}")

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(base_dir, 'datasets_collection')
    os.makedirs(dataset_dir, exist_ok=True)
    
    print("==================================================")
    print(" FABRIC & TEXTILE DEFECT DATASET COLLECTOR")
    print("==================================================")
    
    # 1. TILDA Textile Texture Database (Subset)
    # The full TILDA dataset is older but a classic in textile inspection.
    print("\n1. Collecting Public Fabric Repositories...")
    
    # We will write a markdown file with links to datasets that require forms/authentication
    restricted_datasets = """# Comprehensive List of Fabric Defect Datasets

Due to academic licensing, some of the best fabric datasets require you to fill out a short university/company form before downloading. You cannot script their download directly.

### 1. AITEX Fabric Image Database
- **Description**: One of the most famous datasets. Contains 245 images of 7 different fabric types with 14 different defect types (stains, thread errors, etc.) and ground-truth segmentation masks.
- **Link**: https://www.aitex.es/afid/ (Requires simple form)

### 2. MVTec AD (Carpet & Leather)
- **Description**: The gold standard for anomaly detection. Carpet and Leather categories simulate textile/fabric textures perfectly.
- **Link**: https://www.mvtec.com/company/research/datasets/mvtec-ad

### 3. ZJU-Leaper (Alibaba Tianchi)
- **Description**: A massive industrial dataset of fabric defects from the Alibaba Tianchi competition. 
- **Link**: Search "Alibaba Tianchi Fabric Defect" on Kaggle/Tianchi.

### 4. DAGM 2007 (Texture Defect Dataset)
- **Description**: Contains artificial defects generated on background textures. Classes 1-6 are highly similar to fabric weaves.
- **Link**: https://hci.iwr.uni-heidelberg.de/content/weakly-supervised-learning-industrial-optical-inspection
"""
    
    with open(os.path.join(dataset_dir, 'RESTRICTED_DATASETS_LINKS.md'), 'w') as f:
        f.write(restricted_datasets)
    print("Saved RESTRICTED_DATASETS_LINKS.md for datasets requiring forms.")

    # Let's download a small public open-source texture dataset sample (KTH-TIPS fabric textures)
    # We will download the Cotton dataset from KTH-TIPS as a representation of normal fabric variations
    kth_url = "https://www.csc.kth.se/cvap/databases/kth-tips/kth-tips-grey.tar.gz"
    kth_dest = os.path.join(dataset_dir, "kth-tips-grey.tar.gz")
    
    print("\n2. Downloading KTH-TIPS (Texture dataset containing Cotton/Fabric textures)...")
    if download_file(kth_url, kth_dest):
        extract_file(kth_dest, os.path.join(dataset_dir, "KTH_TIPS"))
        os.remove(kth_dest) # Cleanup
        
    print("\nDataset collection complete! Check the 'datasets_collection' folder.")

if __name__ == "__main__":
    main()
