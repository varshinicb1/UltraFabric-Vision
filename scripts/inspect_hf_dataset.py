import os
from datasets import load_dataset

def main():
    print("Downloading HuggingFace dataset: SimTho/IndustrialTextileDataset...")
    dataset = load_dataset("SimTho/IndustrialTextileDataset")
    print(dataset)
    
    # Check features of train split
    if 'train' in dataset:
        print("Features:", dataset['train'].features)
        print("Sample:", dataset['train'][0])

if __name__ == "__main__":
    main()
