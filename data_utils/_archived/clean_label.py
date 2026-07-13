"""
Label Cleaning Script for YOLO Instance Segmentation

This script:
1. Maps all class IDs (except class 6) to class 0
2. Removes all instances with class ID 6
3. Creates cleaned label files
4. Validates the cleaned labels
"""

import os
import shutil
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple


def clean_label_file(
    input_path: str,
    output_path: str,
    class_mapping: Dict[int, int] = None,
    classes_to_ignore: List[int] = None
) -> Tuple[int, int, int]:
    """
    Clean a single label file.
    
    Args:
        input_path: Path to input label file
        output_path: Path to output label file
        class_mapping: Dictionary mapping old class IDs to new ones
        classes_to_ignore: List of class IDs to remove
        
    Returns:
        (original_instances, cleaned_instances, ignored_instances)
    """
    if class_mapping is None:
        class_mapping = {}  # Will map all to 0 by default
    
    if classes_to_ignore is None:
        classes_to_ignore = []
    
    original_instances = 0
    cleaned_instances = 0
    ignored_instances = 0
    
    cleaned_lines = []
    
    try:
        with open(input_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            original_instances += 1
            
            tokens = line.split()
            if len(tokens) < 7:  # Need at least class + 3 points
                print(f"  Warning: Invalid line (too few tokens): {line[:50]}...")
                continue
            
            try:
                old_class_id = int(tokens[0])
            except ValueError:
                print(f"  Warning: Invalid class ID: {tokens[0]}")
                continue
            
            # Check if this class should be ignored
            if old_class_id in classes_to_ignore:
                ignored_instances += 1
                continue
            
            # Map class ID
            if old_class_id in class_mapping:
                new_class_id = class_mapping[old_class_id]
            else:
                new_class_id = 0  # Default: map all to class 0
            
            # Validate coordinates
            try:
                coords = [float(x) for x in tokens[1:]]
                
                # Check if coordinates are in valid range
                if not all(0 <= c <= 1 for c in coords):
                    print(f"  Warning: Coordinates out of range [0,1]: {line[:50]}...")
                    # Clamp to [0, 1]
                    coords = [max(0, min(1, c)) for c in coords]
                
                # Rebuild line with new class ID
                new_line = f"{new_class_id} " + " ".join(f"{c:.6f}" for c in coords)
                cleaned_lines.append(new_line)
                cleaned_instances += 1
                
            except ValueError as e:
                print(f"  Warning: Invalid coordinates: {e}")
                continue
        
        # Write cleaned file
        if cleaned_lines:
            with open(output_path, 'w') as f:
                f.write('\n'.join(cleaned_lines) + '\n')
        else:
            # Create empty file if no valid instances
            Path(output_path).touch()
    
    except Exception as e:
        print(f"  Error processing {input_path}: {e}")
        return 0, 0, 0
    
    return original_instances, cleaned_instances, ignored_instances


def clean_dataset(
    input_dir: str,
    output_dir: str,
    class_mapping: Dict[int, int] = None,
    classes_to_ignore: List[int] = None,
    copy_images: bool = True
):
    """
    Clean all label files in a dataset.
    
    Args:
        input_dir: Input directory with images/ and labels/ folders
        output_dir: Output directory
        class_mapping: Dictionary mapping old class IDs to new ones
        classes_to_ignore: List of class IDs to remove
        copy_images: Whether to copy images to output directory
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    input_images = input_path / 'images'
    input_labels = input_path / 'labels'
    output_images = output_path / 'images'
    output_labels = output_path / 'labels'
    
    # Create output directories
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)
    
    # Get all label files
    label_files = list(input_labels.glob('*.txt'))
    
    if len(label_files) == 0:
        print(f"⚠️  No label files found in {input_labels}")
        return
    
    print(f"\nProcessing {len(label_files)} label files...")
    
    total_original = 0
    total_cleaned = 0
    total_ignored = 0
    files_with_changes = 0
    
    for label_file in tqdm(label_files, desc="Cleaning labels"):
        input_label = label_file
        output_label = output_labels / label_file.name
        
        original, cleaned, ignored = clean_label_file(
            str(input_label),
            str(output_label),
            class_mapping,
            classes_to_ignore
        )
        
        total_original += original
        total_cleaned += cleaned
        total_ignored += ignored
        
        if original != cleaned:
            files_with_changes += 1
        
        # Copy corresponding image if requested
        if copy_images:
            image_name = label_file.stem
            # Try different image extensions
            for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
                image_file = input_images / f"{image_name}{ext}"
                if image_file.exists():
                    output_image = output_images / f"{image_name}{ext}"
                    shutil.copy2(image_file, output_image)
                    break
    
    # Print statistics
    print("\n" + "="*70)
    print("CLEANING SUMMARY")
    print("="*70)
    print(f"Label files processed: {len(label_files)}")
    print(f"Files modified: {files_with_changes}")
    print(f"\nInstances:")
    print(f"  Original: {total_original}")
    print(f"  Cleaned: {total_cleaned}")
    print(f"  Ignored: {total_ignored}")
    print(f"  Removed: {total_original - total_cleaned}")
    
    if classes_to_ignore:
        print(f"\nIgnored class IDs: {classes_to_ignore}")
    
    if class_mapping:
        print(f"\nClass mapping applied:")
        for old_id, new_id in sorted(class_mapping.items()):
            print(f"  {old_id} → {new_id}")
    else:
        print(f"\nAll classes mapped to: 0")
    
    print("="*70 + "\n")


def verify_cleaned_labels(label_dir: str, expected_classes: List[int] = None):
    """
    Verify that cleaned labels only contain expected class IDs.
    
    Args:
        label_dir: Directory with label files
        expected_classes: List of expected class IDs (default: [0])
    """
    if expected_classes is None:
        expected_classes = [0]
    
    label_path = Path(label_dir)
    label_files = list(label_path.glob('*.txt'))
    
    print(f"\nVerifying {len(label_files)} label files...")
    
    found_classes = set()
    invalid_files = []
    
    for label_file in tqdm(label_files, desc="Verifying"):
        try:
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                tokens = line.split()
                class_id = int(tokens[0])
                found_classes.add(class_id)
                
                if class_id not in expected_classes:
                    invalid_files.append((label_file.name, class_id))
        
        except Exception as e:
            print(f"  Error reading {label_file.name}: {e}")
    
    print("\n" + "="*70)
    print("VERIFICATION RESULTS")
    print("="*70)
    print(f"Expected classes: {expected_classes}")
    print(f"Found classes: {sorted(found_classes)}")
    
    if invalid_files:
        print(f"\n⚠️  WARNING: {len(invalid_files)} files contain unexpected class IDs:")
        for filename, class_id in invalid_files[:10]:
            print(f"  {filename}: class {class_id}")
        if len(invalid_files) > 10:
            print(f"  ... and {len(invalid_files) - 10} more")
    else:
        print(f"\n✅ All labels contain only expected class IDs!")
    
    print("="*70 + "\n")


def analyze_class_distribution(data_dir: str):
    """
    Analyze class distribution in dataset before cleaning.
    
    Args:
        data_dir: Directory with labels/ folder
    """
    labels_path = Path(data_dir) / 'labels'
    label_files = list(labels_path.glob('*.txt'))
    
    print(f"\nAnalyzing {len(label_files)} label files...")
    
    class_counts = {}
    total_instances = 0
    
    for label_file in tqdm(label_files, desc="Analyzing"):
        try:
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                tokens = line.split()
                if len(tokens) < 7:
                    continue
                
                try:
                    class_id = int(tokens[0])
                    class_counts[class_id] = class_counts.get(class_id, 0) + 1
                    total_instances += 1
                except ValueError:
                    continue
        
        except Exception as e:
            continue
    
    print("\n" + "="*70)
    print("CLASS DISTRIBUTION ANALYSIS")
    print("="*70)
    print(f"Total instances: {total_instances}")
    print(f"\nClass distribution:")
    for class_id in sorted(class_counts.keys()):
        count = class_counts[class_id]
        percentage = (count / total_instances) * 100
        print(f"  Class {class_id}: {count:4d} instances ({percentage:5.2f}%)")
    print("="*70 + "\n")
    
    return class_counts


# ============================================================================
# Main Script
# ============================================================================

def main():
    """Main cleaning pipeline"""
    
    print("\n" + "="*70)
    print("YOLO LABEL CLEANING SCRIPT")
    print("="*70)
    
    # Configuration
    CONFIG = {
        'train_dir': r'C:\Users\Yifei\Documents\cryo\revised_annotation\split\train',
        'val_dir': r'C:\Users\Yifei\Documents\cryo\revised_annotation\split\test',
        'output_train_dir': r'C:\Users\Yifei\Documents\cryo\revised_annotation\split_cleaned\train',
        'output_val_dir': r'C:\Users\Yifei\Documents\cryo\revised_annotation\split_cleaned\test',
        
        # Class mapping: Map ALL classes to 0 (vesicle)
        'class_mapping': {
            0: 0,
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0,
            # 6 is ignored (see below)
            7: 0,
            8: 0,
            9: 0,
        },
        
        # Classes to ignore (remove from dataset)
        'classes_to_ignore': [6],  # Ignore class 6
    }
    
    print("\nConfiguration:")
    print(f"  Input train dir: {CONFIG['train_dir']}")
    print(f"  Input val dir: {CONFIG['val_dir']}")
    print(f"  Output train dir: {CONFIG['output_train_dir']}")
    print(f"  Output val dir: {CONFIG['output_val_dir']}")
    print(f"  Classes to ignore: {CONFIG['classes_to_ignore']}")
    print(f"  Mapping all other classes to: 0")
    print("="*70)
    
    # Step 1: Analyze original distribution
    print("\n" + "="*70)
    print("STEP 1: ANALYZING ORIGINAL CLASS DISTRIBUTION")
    print("="*70)
    
    # print("\nTrain set:")
    # train_dist = analyze_class_distribution(CONFIG['train_dir'])
    
    print("\nValidation set:")
    val_dist = analyze_class_distribution(CONFIG['val_dir'])
    
    # # Step 2: Clean train set
    # print("\n" + "="*70)
    # print("STEP 2: CLEANING TRAINING SET")
    # print("="*70)
    
    # clean_dataset(
    #     input_dir=CONFIG['train_dir'],
    #     output_dir=CONFIG['output_train_dir'],
    #     class_mapping=CONFIG['class_mapping'],
    #     classes_to_ignore=CONFIG['classes_to_ignore'],
    #     copy_images=True
    # )
    
    # Step 3: Clean validation set
    print("\n" + "="*70)
    print("STEP 3: CLEANING VALIDATION SET")
    print("="*70)
    
    clean_dataset(
        input_dir=CONFIG['val_dir'],
        output_dir=CONFIG['output_val_dir'],
        class_mapping=CONFIG['class_mapping'],
        classes_to_ignore=CONFIG['classes_to_ignore'],
        copy_images=True
    )
    
    # Step 4: Verify cleaned labels
    print("\n" + "="*70)
    print("STEP 4: VERIFYING CLEANED LABELS")
    print("="*70)
    
    # print("\nVerifying train set:")
    # verify_cleaned_labels(
    #     str(Path(CONFIG['output_train_dir']) / 'labels'),
    #     expected_classes=[0]
    # )
    
    print("\nVerifying validation set:")
    verify_cleaned_labels(
        str(Path(CONFIG['output_val_dir']) / 'labels'),
        expected_classes=[0]
    )
    
    # Complete
    print("\n" + "="*70)
    print("LABEL CLEANING COMPLETE!")
    print("="*70)
    print("\nCleaned dataset locations:")
    print(f"  Train: {CONFIG['output_train_dir']}")
    print(f"  Val: {CONFIG['output_val_dir']}")
    print("\nNext steps:")
    print("  1. Update paths in train_yolo_segmentation.py:")
    print(f"     'train_dir': r'{CONFIG['output_train_dir']}'")
    print(f"     'val_dir': r'{CONFIG['output_val_dir']}'")
    print("  2. Run: python train_yolo_segmentation.py --config")
    print("="*70 + "\n")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Clean YOLO label files')
    parser.add_argument('--analyze-only', action='store_true',
                       help='Only analyze class distribution without cleaning')
    parser.add_argument('--train-dir', type=str,
                       help='Training data directory')
    parser.add_argument('--val-dir', type=str,
                       help='Validation data directory')
    
    args = parser.parse_args()
    
    if args.analyze_only:
        if args.train_dir:
            print("\nAnalyzing training set:")
            analyze_class_distribution(args.train_dir)
        if args.val_dir:
            print("\nAnalyzing validation set:")
            analyze_class_distribution(args.val_dir)
    else:
        main()