import os
import shutil
import random
from pathlib import Path

def split_yolo_dataset(dataset_path, output_path, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """
    Split YOLO format dataset into train, val, and test sets.
    
    Args:
        dataset_path: Path to folder containing images and labels subdirectories
        output_path: Path where split datasets will be saved
        train_ratio: Proportion for training set (default 0.7)
        val_ratio: Proportion for validation set (default 0.2)
        test_ratio: Proportion for test set (default 0.1)
    """
    
    # Validate ratios
    if not abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.01:
        raise ValueError("Ratios must sum to 1.0")
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        Path(f"{output_path}/{split}/images").mkdir(parents=True, exist_ok=True)
        Path(f"{output_path}/{split}/labels").mkdir(parents=True, exist_ok=True)
    
    # Get list of image files
    images_dir = f"{dataset_path}/images"
    labels_dir = f"{dataset_path}/labels"
    
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
    random.shuffle(image_files)
    
    # Calculate split indices
    total = len(image_files)
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    
    train_files = image_files[:train_count]
    val_files = image_files[train_count:train_count + val_count]
    test_files = image_files[train_count + val_count:]
    
    # Copy files to respective directories
    for split, files in [('train', train_files), ('val', val_files), ('test', test_files)]:
        for file in files:
            # Copy image
            src_img = f"{images_dir}/{file}"
            dst_img = f"{output_path}/{split}/images/{file}"
            shutil.copy2(src_img, dst_img)
            
            # Copy corresponding label
            label_file = Path(file).stem + '.txt'
            src_label = f"{labels_dir}/{label_file}"
            dst_label = f"{output_path}/{split}/labels/{label_file}"
            if os.path.exists(src_label):
                shutil.copy2(src_label, dst_label)
    
    print(f"Dataset split completed!")
    print(f"Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}")

# Usage
if __name__ == "__main__":
    split_yolo_dataset(
        dataset_path=r"C:\Users\Yifei\Documents\cryo\revised_annotation\all",
        output_path=r"C:\Users\Yifei\Documents\cryo\revised_annotation\split",
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1
    )