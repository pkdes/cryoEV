"""
YOLO Instance Segmentation Training Script for Cryo-EM Vesicles
Fixed version with consistent prediction handling and optimized thresholds.
"""

import csv
import os
import yaml
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
import numpy as np
from PIL import Image
import cv2
from scipy.ndimage import label as label_connected_components
from scipy.optimize import linear_sum_assignment


def verify_yolo_polygon_format(label_file: str) -> bool:
    """Verify label file is in YOLO polygon format (class_id x1 y1 x2 y2 ...)"""
    try:
        with open(label_file, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                tokens = line.strip().split()
                if len(tokens) < 7:
                    return False
                int(tokens[0])  # Verify class ID
                coords = [float(x) for x in tokens[1:]]
                if not all(0 <= c <= 1 for c in coords):
                    return False
        return True
    except Exception as e:
        print(f"Error verifying {label_file}: {e}")
        return False


def prepare_yolo_dataset(source_data_dir: str, output_dir: str, split_name: str = 'train') -> Dict[str, int]:
    """Prepare dataset in YOLO format"""
    source_images = Path(source_data_dir) / 'images'
    source_labels = Path(source_data_dir) / 'labels'
    
    output_images = Path(output_dir) / 'images' / split_name
    output_labels = Path(output_dir) / 'labels' / split_name
    
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)
    
    stats = {'n_images': 0, 'n_labels': 0, 'n_instances': 0, 'images_without_labels': 0}
    
    for img_path in tqdm(source_images.glob('*'), desc=f"Preparing {split_name}"):
        if img_path.suffix.lower() not in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
            continue
        
        shutil.copy2(img_path, output_images / img_path.name)
        stats['n_images'] += 1
        
        label_file = source_labels / f"{img_path.stem}.txt"
        if label_file.exists() and verify_yolo_polygon_format(str(label_file)):
            shutil.copy2(label_file, output_labels / f"{img_path.stem}.txt")
            stats['n_labels'] += 1
            with open(label_file) as f:
                stats['n_instances'] += len([l for l in f if l.strip()])
        else:
            stats['images_without_labels'] += 1
    
    return stats


def visualize_augmented_samples(data_yaml: str, model_size: str = 'n', imgsz: int = 1024, 
                               n_samples: int = 16, output_dir: str = None, seed: int = 42,
                               use_yolov11: bool = True):
    """
    Visualize augmented training samples to verify augmentation quality.
    
    Augmentations shown:
    - HSV color jitter (hue, saturation, brightness)
    - Random crops (85-95% of image, then resize)
    - Horizontal and vertical flips
    - NO rotation (to avoid black border artifacts)
    
    Args:
        data_yaml: Path to YOLO dataset config
        model_size: Model size (affects augmentation pipeline)
        imgsz: Image size for training
        n_samples: Number of samples to visualize (will show in grid)
        output_dir: Where to save visualization
        seed: Random seed for reproducibility
        use_yolov11: If True, use YOLOv11; if False, use YOLOv8
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from ultralytics import YOLO
    from ultralytics.data import YOLODataset
    from ultralytics.data.augment import Compose, Format
    import torch
    
    print("\n" + "="*80)
    print("VISUALIZING AUGMENTED TRAINING SAMPLES")
    print("="*80 + "\n")
    
    print("[!]  IMPORTANT NOTE:")
    print("This visualization shows augmentations applied to IMAGES ONLY.")
    print("Masks shown are from original positions (not transformed).")
    print("During actual training, YOLO correctly transforms both images AND masks together.")
    print("Use this to verify: no artifacts, good color/crop diversity, vesicles visible.")
    print("To see actual augmented batches: check training_output/*/train_batch*.jpg\n")
    
    # Load dataset config
    import yaml
    with open(data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)
    
    # Create model to get augmentation transforms
    if use_yolov11:
        model_name = f'yolo11{model_size}-seg.pt'
        print(f"Using YOLOv11 ({model_name})")
    else:
        model_name = f'yolov8{model_size}-seg.pt'
        print(f"Using YOLOv8 ({model_name})")
    
    model = YOLO(model_name)
    
    # Get the training augmentation pipeline
    print("Augmentation Pipeline (Artifact-Free Strategy):")
    print("  Correct Order: Flip -> Crop -> Color")
    print("  1. Random flips (horizontal & vertical, 50% each)")
    print("  2. Random crops (85-95% of flipped image, then resize back)")
    print("  3. HSV color jitter (hue +/-2%, saturation 0.4-1.6x, brightness 0.6-1.4x)")
    print("  + Mosaic (combines 4 images)")
    print("  + Copy-Paste (0.3 probability)")
    print("  + MixUp (0.15 probability)")
    print("  [x] NO rotation (disabled to avoid black border artifacts)")
    print("  [x] NO shear/perspective (disabled to avoid artifacts)")
    print("")
    
    # Load raw dataset
    data_path = Path(data_config['path'])
    train_images = data_path / 'images' / 'train'
    train_labels = data_path / 'labels' / 'train'
    
    if not train_images.exists():
        print(f"Error: Training images not found at {train_images}")
        return
    
    # Get image list
    img_files = sorted(list(train_images.glob('*')))
    img_files = [f for f in img_files if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']]
    
    if len(img_files) == 0:
        print("Error: No images found in training set")
        return
    
    print(f"Found {len(img_files)} training images")
    print(f"Generating {n_samples} augmented samples...\n")
    
    # Setup output directory
    if output_dir is None:
        output_dir = Path(data_yaml).parent.parent / 'augmentation_samples'
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate grid size
    n_cols = 4
    n_rows = (n_samples + n_cols - 1) // n_cols
    
    # Create visualization
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Randomly select images and apply augmentation
    np.random.seed(seed)
    selected_indices = np.random.choice(len(img_files), size=min(n_samples, len(img_files)), replace=False)
    
    for idx, img_idx in enumerate(selected_indices):
        if idx >= n_samples:
            break
            
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        # Load image
        img_path = img_files[img_idx]
        img = np.array(Image.open(img_path).convert('RGB'))
        
        # Load labels if they exist
        label_path = train_labels / f"{img_path.stem}.txt"
        masks = []
        if label_path.exists():
            h, w = img.shape[:2]
            with open(label_path) as f:
                import cv2
                for line in f:
                    if not line.strip():
                        continue
                    parts = line.strip().split()
                    coords = [float(x) for x in parts[1:]]
                    polygon = np.array([(coords[i]*w, coords[i+1]*h) 
                                       for i in range(0, len(coords), 2)], dtype=np.float32)
                    masks.append(polygon)
        
        # Apply augmentation - CORRECT ORDER: flip -> crop -> color
        aug_img = img.copy()
        h, w = img.shape[:2]
        
        # 1. Random flips FIRST
        flip_h = np.random.rand() > 0.5
        flip_v = np.random.rand() > 0.5
        if flip_h:
            aug_img = cv2.flip(aug_img, 1)  # Horizontal
        if flip_v:
            aug_img = cv2.flip(aug_img, 0)  # Vertical
        
        # 2. Random crop SECOND (85-95% of flipped image) then resize
        crop_ratio = np.random.uniform(0.85, 0.95)
        crop_h = int(h * crop_ratio)
        crop_w = int(w * crop_ratio)
        top = np.random.randint(0, h - crop_h + 1)
        left = np.random.randint(0, w - crop_w + 1)
        aug_img = aug_img[top:top+crop_h, left:left+crop_w]
        aug_img = cv2.resize(aug_img, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # 3. Random HSV color jitter LAST
        if np.random.rand() > 0.3:
            aug_img_hsv = cv2.cvtColor(aug_img, cv2.COLOR_RGB2HSV).astype(np.float32)
            aug_img_hsv[..., 0] += np.random.uniform(-0.02, 0.02) * 180
            aug_img_hsv[..., 1] *= np.random.uniform(0.4, 1.6)
            aug_img_hsv[..., 2] *= np.random.uniform(0.6, 1.4)
            aug_img_hsv = np.clip(aug_img_hsv, 0, 255).astype(np.uint8)
            aug_img = cv2.cvtColor(aug_img_hsv, cv2.COLOR_HSV2RGB)
        
        # Display augmented image (NO masks - would be misaligned)
        ax.imshow(aug_img)
        
        # Add augmentation info to title
        aug_info = []
        if flip_h:
            aug_info.append('H-flip')
        if flip_v:
            aug_info.append('V-flip')
        aug_info.append(f'crop={crop_ratio:.0%}')
        
        ax.set_title(f'Sample {idx+1} ({", ".join(aug_info)})', fontsize=9)
        ax.axis('off')
    
    # Hide unused subplots
    for idx in range(len(selected_indices), n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    output_path = output_dir / 'augmented_samples_grid.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Augmentation samples saved to: {output_path}")
    
    # Also create individual samples with more detailed info
    print(f"\nGenerating detailed individual samples...")
    detail_dir = output_dir / 'detailed'
    detail_dir.mkdir(exist_ok=True)
    
    for i in range(min(4, n_samples)):  # Save first 4 in detail
        img_idx = selected_indices[i]
        img_path = img_files[img_idx]
        img = np.array(Image.open(img_path).convert('RGB'))
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original
        axes[0].imshow(img)
        axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Augmented version 1
        aug1 = apply_random_augmentation(img, img_path, train_labels)
        axes[1].imshow(aug1)
        axes[1].set_title('Augmented Version 1', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        # Augmented version 2
        aug2 = apply_random_augmentation(img, img_path, train_labels)
        axes[2].imshow(aug2)
        axes[2].set_title('Augmented Version 2', fontsize=12, fontweight='bold')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(detail_dir / f'sample_{i+1}_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"[OK] Detailed samples saved to: {detail_dir}")
    print(f"\n{'='*80}")
    print("AUGMENTATION QUALITY CHECK:")
    print("  [OK] Are vesicle boundaries still visible?")
    print("  [OK] Are objects still recognizable after transformations?")
    print("  [OK] Is the augmentation diversity sufficient?")
    print("  [OK] No black artifacts or weird borders?")
    print("\n[!]  Note: Masks shown are NOT transformed (visualization only).")
    print("During training, YOLO transforms both images AND masks correctly.")
    print("To see actual augmented batches with transformed masks:")
    print(f"  -> Check: {Path(data_yaml).parent.parent / 'training' / '*' / 'train_batch*.jpg'}")
    print("="*80 + "\n")


def apply_random_augmentation(img: np.ndarray, img_path: Path, labels_dir: Path) -> np.ndarray:
    """
    Apply random augmentation to an image for visualization.
    Order matters: flip -> crop -> color jitter
    """
    import cv2
    
    aug_img = img.copy()
    h, w = aug_img.shape[:2]
    
    # 1. Random flips FIRST (before cropping)
    if np.random.rand() > 0.5:
        aug_img = cv2.flip(aug_img, 1)  # Horizontal flip
    
    if np.random.rand() > 0.5:
        aug_img = cv2.flip(aug_img, 0)  # Vertical flip
    
    # 2. Random crop SECOND (85-95% of flipped image) then resize back
    crop_ratio = np.random.uniform(0.85, 0.95)
    crop_h = int(h * crop_ratio)
    crop_w = int(w * crop_ratio)
    
    # Random crop position
    top = np.random.randint(0, h - crop_h + 1)
    left = np.random.randint(0, w - crop_w + 1)
    
    aug_img = aug_img[top:top+crop_h, left:left+crop_w]
    aug_img = cv2.resize(aug_img, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # 3. Random HSV color augmentation LAST
    if np.random.rand() > 0.3:
        aug_img_hsv = cv2.cvtColor(aug_img, cv2.COLOR_RGB2HSV).astype(np.float32)
        aug_img_hsv[..., 0] += np.random.uniform(-0.02, 0.02) * 180  # Hue
        aug_img_hsv[..., 1] *= np.random.uniform(0.4, 1.6)            # Saturation
        aug_img_hsv[..., 2] *= np.random.uniform(0.6, 1.4)            # Value/brightness
        aug_img_hsv = np.clip(aug_img_hsv, 0, 255).astype(np.uint8)
        aug_img = cv2.cvtColor(aug_img_hsv, cv2.COLOR_HSV2RGB)
    
    return aug_img


def rotate_and_crop(img: np.ndarray, angle: float) -> np.ndarray:
    """
    DEPRECATED: No longer using rotation to avoid artifacts.
    Keeping function for compatibility but it won't be called.
    """
    return img


def create_yolo_yaml(dataset_root: str, output_path: str, class_names: List[str] = None):
    """Create YOLO dataset configuration YAML"""
    if class_names is None:
        class_names = ['vesicle']
    
    config = {
        'path': str(Path(dataset_root).absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'names': {i: name for i, name in enumerate(class_names)},
        'nc': len(class_names)
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"[OK] Created config: {output_path}")


def train_yolo_segmentation(data_yaml: str, model_size: str = 'n', epochs: int = 300, imgsz: int = 1024,
                           batch_size: int = 16, device: str = '0', project: str = 'yolo_results',
                           name: str = 'vesicle_seg', patience: int = 50, overlap_mask: bool = True,
                           use_v11: bool = True, workers: int = 8, optimizer: str = 'AdamW',
                           lr0: float = 0.002, lrf: float = 0.001,
                           hsv_h: float = 0.015, hsv_s: float = 0.7, hsv_v: float = 0.4,
                           degrees: float = 45.0, fliplr: float = 0.5, flipud: float = 0.5,
                           mosaic: float = 1.0, mixup: float = 0.15, copy_paste: float = 0.3,
                           **kwargs):
    """
    Train YOLO instance segmentation model with optimized hyperparameters for cryo-EM vesicles.
    
    Augmentation strategy optimized to avoid artifacts:
    - Primary: Flips, crops, color jitter (no artifacts)
    - Secondary: Moderate rotation (+/-45 deg instead of +/-90 deg)
    - Advanced: Mosaic, copy-paste, mixup for instance learning
    
    Args:
        use_v11: If True, use YOLOv11 (default). If False, use YOLOv8.
    """
    from ultralytics import YOLO
    
    # Select model version
    if use_v11:
        model_name = f'yolo11{model_size}-seg.pt'
        version_str = "YOLOv11"
    else:
        model_name = f'yolov8{model_size}-seg.pt'
        version_str = "YOLOv8"
    
    print(f"\nTraining {model_name} ({version_str}) | Epochs: {epochs} | Batch: {batch_size} | Device: {device}")
    print("Augmentation optimized to avoid artifacts: flips + crops + moderate rotation")
    print("Primary augmentations: color jitter, random crops, flips\n")
    
    model = YOLO(model_name)

    import time, torch
    torch.cuda.reset_peak_memory_stats()
    t_start = time.time()

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        project=project,
        name=name,
        patience=patience,
        overlap_mask=overlap_mask,

        # Mask generation
        mask_ratio=4,

        # Data augmentation
        hsv_h=hsv_h,
        hsv_s=hsv_s,
        hsv_v=hsv_v,
        degrees=degrees,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=flipud,
        fliplr=fliplr,
        mosaic=mosaic,
        mixup=mixup,
        copy_paste=copy_paste,

        # Optimizer
        optimizer=optimizer,
        lr0=lr0,
        lrf=lrf,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=5.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,

        # Loss weights
        box=7.5, cls=0.5, dfl=1.5,

        # Training settings
        amp=True,
        fraction=1.0,
        profile=False,
        close_mosaic=10,

        # System
        workers=workers,
        seed=42,
        deterministic=False,

        # Output
        verbose=True,
        plots=True,
        save=True,
        save_period=50,
        exist_ok=False,

        **kwargs
    )

    wall_secs = time.time() - t_start
    h, rem = divmod(int(wall_secs), 3600)
    m, s   = divmod(rem, 60)
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0

    # Read best epoch from results.csv
    import csv as _csv
    results_csv = Path(results.save_dir) / 'results.csv'
    best_ep, best_map50b, best_map50m = 0, 0.0, 0.0
    if results_csv.exists():
        with open(results_csv, newline='', encoding='utf-8') as _f:
            rows = list(_csv.DictReader(_f))
        rows = [{k.strip(): v for k, v in r.items()} for r in rows]
        for i, r in enumerate(rows):
            v = float(r.get('metrics/mAP50(M)', 0) or 0)
            if v > best_map50m:
                best_map50m = v
                best_map50b = float(r.get('metrics/mAP50(B)', 0) or 0)
                best_ep = i + 1

    summary_lines = [
        "=" * 60,
        "TRAINING RUN SUMMARY",
        "=" * 60,
        f"Wall time:        {h:02d}:{m:02d}:{s:02d}",
        f"Peak GPU memory:  {peak_gb:.2f} GB  (of {total_gb:.1f} GB)",
        f"Epochs completed: {len(rows) if results_csv.exists() else '?'} / {epochs}",
        f"Best epoch:       {best_ep}",
        f"Best mAP50(B):    {best_map50b:.4f}",
        f"Best mAP50(M):    {best_map50m:.4f}",
        f"Hyperparams:      model=yolo{'11' if use_v11 else 'v8'}{model_size}-seg  "
        f"optimizer={optimizer}  lr0={lr0}  lrf={lrf}  imgsz={imgsz}  batch={batch_size}",
        "=" * 60,
    ]
    summary = "\n".join(summary_lines)
    print("\n" + summary)
    (Path(results.save_dir) / 'run_summary.txt').write_text(summary + "\n", encoding='utf-8')

    print(f"[OK] Training complete: {results.save_dir}")
    return results


def validate_yolo_model(model_path: str, data_yaml: str, imgsz: int = 1024, batch_size: int = 16,
                       device: str = '0', split: str = 'val', project: str = None,
                       name: str = 'val'):
    """Validate YOLO model"""
    from ultralytics import YOLO
    
    model = YOLO(model_path)
    val_kwargs = dict(data=data_yaml, split=split, imgsz=imgsz, batch=batch_size,
                      device=device, verbose=False)
    if project is not None:
        val_kwargs['project'] = project
        val_kwargs['name'] = name
    results = model.val(**val_kwargs)
    
    if results.seg:
        print(f"[OK] Validation Metrics:")
        print(f"  Box mAP@0.5: {results.box.map50:.4f}")
        print(f"  Box mAP@0.5:0.95: {results.box.map:.4f}")
        print(f"  Mask mAP@0.5: {results.seg.map50:.4f}")
        print(f"  Mask mAP@0.5:0.95: {results.seg.map:.4f}")
    
    return results


def load_predictions_from_model(model_path: str, img_path: str, imgsz: int, 
                                conf: float, iou: float, device: str) -> Tuple[List[np.ndarray], List[float]]:
    """
    Load predictions directly from model with specified thresholds.
    Returns masks and their confidence scores.
    """
    from ultralytics import YOLO
    import cv2
    
    model = YOLO(model_path)
    result = model.predict(source=str(img_path), imgsz=imgsz, conf=conf, iou=iou, 
                          device=device, verbose=False, retina_masks=True)[0]
    
    img = Image.open(img_path)
    w, h = img.size
    
    pred_masks = []
    confidences = []
    
    if result.masks is not None and result.boxes is not None:
        for mask, box in zip(result.masks.data, result.boxes):
            mask_2d = mask.cpu().numpy().squeeze()  # handle (1,H,W) or (H,W,1)
            mask_np = cv2.resize(mask_2d, (w, h), interpolation=cv2.INTER_LINEAR)
            mask_bool = mask_np > 0.5
            pred_masks.append(mask_bool)
            confidences.append(float(box.conf.cpu().numpy()[0]))
    
    return pred_masks, confidences


def load_gt_masks_from_labels(label_path: str, img_width: int, img_height: int) -> List[np.ndarray]:
    """Load ground truth masks from YOLO polygon format"""
    import cv2
    
    gt_masks = []
    
    if not label_path.exists():
        return gt_masks
    
    with open(label_path) as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split()
            coords = [float(x) for x in parts[1:]]
            polygon = np.array([(int(coords[i]*img_width), int(coords[i+1]*img_height)) 
                               for i in range(0, len(coords), 2)], dtype=np.int32)
            
            mask = np.zeros((img_height, img_width), dtype=bool)
            mask_uint8 = np.zeros((img_height, img_width), dtype=np.uint8)
            cv2.fillPoly(mask_uint8, [polygon], 1)
            mask = mask_uint8 > 0
            gt_masks.append(mask)
    
    return gt_masks


def load_predictions_with_classes_from_model(model_path: str, img_path: str, imgsz: int,
                                             conf: float, iou: float, device: str) -> Tuple[List[np.ndarray], List[float], List[int]]:
    """Load predicted masks, confidences, and class IDs from model output."""
    from ultralytics import YOLO
    import cv2

    model = YOLO(model_path)
    result = model.predict(source=str(img_path), imgsz=imgsz, conf=conf, iou=iou,
                          device=device, verbose=False, retina_masks=True)[0]

    img = Image.open(img_path)
    w, h = img.size

    pred_masks = []
    confidences = []
    class_ids = []

    if result.masks is not None and result.boxes is not None:
        for mask, box in zip(result.masks.data, result.boxes):
            mask_2d = mask.cpu().numpy().squeeze()
            mask_np = cv2.resize(mask_2d, (w, h), interpolation=cv2.INTER_LINEAR)
            pred_masks.append(mask_np > 0.5)
            confidences.append(float(box.conf.cpu().numpy()[0]))
            class_ids.append(int(box.cls.cpu().numpy()[0]))

    return pred_masks, confidences, class_ids


def load_gt_masks_and_classes_from_labels(label_path: str, img_width: int, img_height: int) -> Tuple[List[np.ndarray], List[int]]:
    """Load ground-truth masks and their class IDs from YOLO polygon labels."""
    import cv2

    gt_masks = []
    gt_classes = []

    if not label_path.exists():
        return gt_masks, gt_classes

    with open(label_path) as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split()
            class_id = int(parts[0])
            coords = [float(x) for x in parts[1:]]
            polygon = np.array([(int(coords[i] * img_width), int(coords[i + 1] * img_height))
                               for i in range(0, len(coords), 2)], dtype=np.int32)

            mask_uint8 = np.zeros((img_height, img_width), dtype=np.uint8)
            cv2.fillPoly(mask_uint8, [polygon], 1)
            gt_masks.append(mask_uint8 > 0)
            gt_classes.append(class_id)

    return gt_masks, gt_classes


def calculate_per_class_segmentation_metrics(model_path: str, img_dir: str, label_dir: str,
                                             imgsz: int, conf: float, iou: float, device: str,
                                             class_names: List[str], match_threshold: float = 0.5) -> Dict[str, Dict]:
    """Calculate object-level metrics separately for each class using class-aware matching."""
    img_paths = sorted(Path(img_dir).glob('*'))
    img_paths = [p for p in img_paths if p.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']]

    class_stats = {
        idx: {
            'class_name': class_name,
            'tp': 0,
            'fp': 0,
            'fn': 0,
            'n_gt': 0,
            'n_pred': 0,
            'matched_iou_sum': 0.0,
        }
        for idx, class_name in enumerate(class_names)
    }

    for img_path in tqdm(img_paths, desc="Computing per-class metrics"):
        img = Image.open(img_path)
        w, h = img.size

        pred_masks, _, pred_classes = load_predictions_with_classes_from_model(
            model_path, str(img_path), imgsz, conf, iou, device
        )

        label_path = Path(label_dir) / f"{img_path.stem}.txt"
        gt_masks, gt_classes = load_gt_masks_and_classes_from_labels(label_path, w, h)

        for class_id in class_stats:
            pred_masks_cls = [m for m, c in zip(pred_masks, pred_classes) if c == class_id]
            gt_masks_cls = [m for m, c in zip(gt_masks, gt_classes) if c == class_id]

            matches, unmatched_preds, unmatched_gts = match_objects_hungarian(
                pred_masks_cls, gt_masks_cls, iou_threshold=match_threshold
            )

            stats = class_stats[class_id]
            stats['tp'] += len(matches)
            stats['fp'] += len(unmatched_preds)
            stats['fn'] += len(unmatched_gts)
            stats['n_pred'] += len(pred_masks_cls)
            stats['n_gt'] += len(gt_masks_cls)

            for pred_idx, gt_idx in matches:
                pred_mask = pred_masks_cls[pred_idx]
                gt_mask = gt_masks_cls[gt_idx]
                intersection = (pred_mask & gt_mask).sum()
                union = (pred_mask | gt_mask).sum()
                if union > 0:
                    stats['matched_iou_sum'] += intersection / union

    results = {}
    for class_id, stats in class_stats.items():
        tp, fp, fn = stats['tp'], stats['fp'], stats['fn']
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        avg_iou = stats['matched_iou_sum'] / tp if tp > 0 else 0.0
        results[stats['class_name']] = {
            'class_id': class_id,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'n_gt': stats['n_gt'],
            'n_pred': stats['n_pred'],
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'avg_iou_matched': avg_iou,
        }

    return results


def save_per_class_metrics(per_class_metrics: Dict[str, Dict], output_path: Path, split_name: str) -> None:
    """Save per-class metrics to a readable text file and CSV companion."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write(f"PER-CLASS SEGMENTATION METRICS - {split_name.upper()}\n")
        f.write("=" * 60 + "\n\n")
        for class_name, metrics in per_class_metrics.items():
            f.write(f"Class: {class_name} (id={metrics['class_id']})\n")
            f.write(f"  GT Objects:     {metrics['n_gt']}\n")
            f.write(f"  Pred Objects:   {metrics['n_pred']}\n")
            f.write(f"  TP / FP / FN:   {metrics['tp']} / {metrics['fp']} / {metrics['fn']}\n")
            f.write(f"  Precision:      {metrics['precision']:.4f}\n")
            f.write(f"  Recall:         {metrics['recall']:.4f}\n")
            f.write(f"  F1 Score:       {metrics['f1']:.4f}\n")
            f.write(f"  IoU (matched):  {metrics['avg_iou_matched']:.4f}\n\n")

    csv_path = output_path.with_suffix('.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['class_name', 'class_id', 'n_gt', 'n_pred', 'tp', 'fp', 'fn', 'precision', 'recall', 'f1', 'avg_iou_matched']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for class_name, metrics in per_class_metrics.items():
            writer.writerow({'class_name': class_name, **metrics})


def match_objects_hungarian(pred_masks: List[np.ndarray], gt_masks: List[np.ndarray], 
                            iou_threshold: float = 0.5) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Match predicted and ground truth objects using Hungarian algorithm.
    
    Returns:
        matches: List of (pred_idx, gt_idx) tuples for matched objects
        unmatched_preds: List of unmatched prediction indices
        unmatched_gts: List of unmatched ground truth indices
    """
    if len(pred_masks) == 0 or len(gt_masks) == 0:
        return [], list(range(len(pred_masks))), list(range(len(gt_masks)))
    
    # Compute IoU matrix
    iou_matrix = np.zeros((len(pred_masks), len(gt_masks)))
    for i, pred_mask in enumerate(pred_masks):
        for j, gt_mask in enumerate(gt_masks):
            intersection = (pred_mask & gt_mask).sum()
            union = (pred_mask | gt_mask).sum()
            iou_matrix[i, j] = intersection / union if union > 0 else 0
    
    # Hungarian algorithm for optimal assignment
    pred_indices, gt_indices = linear_sum_assignment(-iou_matrix)
    
    matches = []
    for pred_idx, gt_idx in zip(pred_indices, gt_indices):
        if iou_matrix[pred_idx, gt_idx] >= iou_threshold:
            matches.append((pred_idx, gt_idx))
    
    matched_pred_set = set([m[0] for m in matches])
    matched_gt_set = set([m[1] for m in matches])
    
    unmatched_preds = [i for i in range(len(pred_masks)) if i not in matched_pred_set]
    unmatched_gts = [i for i in range(len(gt_masks)) if i not in matched_gt_set]
    
    return matches, unmatched_preds, unmatched_gts

def visualize_predictions_with_matching(model_path: str, source_dir: str, label_dir: str, output_dir: str,
                                       imgsz: int = 1024, conf: float = 0.25, iou: float = 0.7,
                                       device: str = '0', match_threshold: float = 0.5,
                                       class_names: List[str] = None):
    """
    Visualize predictions broken out per class.
    Layout: [Original] [Class 0 overlay] [Class 1 overlay] ...

    Each class panel uses the same color scheme:
      Green  = TP (correct detection of this class)
      Yellow = FP (predicted this class but no matching GT)
      Red    = FN (GT was this class but missed)
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    if class_names is None:
        class_names = ['EV', 'multilayer EV']

    COLOR = {'tp': np.array([0, 220, 0]), 'fp': np.array([255, 220, 0]), 'fn': np.array([255, 40, 40])}
    ALPHA = 0.6

    legend_elements = [
        Patch(facecolor=COLOR['tp']/255, alpha=0.8, label='TP – correct'),
        Patch(facecolor=COLOR['fp']/255, alpha=0.8, label='FP – hallucination'),
        Patch(facecolor=COLOR['fn']/255, alpha=0.8, label='FN – missed'),
    ]

    output_vis_dir = Path(output_dir) / 'visualizations'
    output_vis_dir.mkdir(parents=True, exist_ok=True)

    img_paths = sorted(Path(source_dir).glob('*'))
    print(f"Found {len(img_paths)} images in {source_dir}")

    for img_path in tqdm(img_paths, desc="Visualizing with matching"):
        if img_path.suffix.lower() not in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
            continue

        img = np.array(Image.open(img_path).convert('RGB'))
        h, w = img.shape[:2]

        pred_masks, confidences, pred_classes = load_predictions_with_classes_from_model(
            model_path, str(img_path), imgsz, conf, iou, device
        )
        label_path = Path(label_dir) / f"{img_path.stem}.txt"
        gt_masks, gt_classes = load_gt_masks_and_classes_from_labels(label_path, w, h)

        matches, unmatched_preds, unmatched_gts = match_objects_hungarian(
            pred_masks, gt_masks, iou_threshold=match_threshold
        )

        n_cols = 1 + len(class_names)
        fig, axes = plt.subplots(1, n_cols, figsize=(11 * n_cols, 10))

        # Column 0: original
        axes[0].imshow(img)
        axes[0].set_title('Original', fontsize=12, fontweight='bold')
        axes[0].axis('off')

        # One column per class
        for cid, cname in enumerate(class_names):
            overlay = img.copy().astype(float)
            tp_c, fp_c, fn_c = 0, 0, 0

            for pred_idx, gt_idx in matches:
                if pred_classes[pred_idx] == cid and gt_classes[gt_idx] == cid:
                    overlay[pred_masks[pred_idx]] = overlay[pred_masks[pred_idx]] * (1-ALPHA) + COLOR['tp'] * ALPHA
                    tp_c += 1
                elif pred_classes[pred_idx] == cid:
                    overlay[pred_masks[pred_idx]] = overlay[pred_masks[pred_idx]] * (1-ALPHA) + COLOR['fp'] * ALPHA
                    fp_c += 1
                elif gt_classes[gt_idx] == cid:
                    overlay[gt_masks[gt_idx]] = overlay[gt_masks[gt_idx]] * (1-ALPHA) + COLOR['fn'] * ALPHA
                    fn_c += 1

            for pred_idx in unmatched_preds:
                if (pred_idx < len(pred_classes)) and pred_classes[pred_idx] == cid:
                    overlay[pred_masks[pred_idx]] = (
                        overlay[pred_masks[pred_idx]] * (1-ALPHA) + COLOR['fp'] * ALPHA)
                    fp_c += 1

            for gt_idx in unmatched_gts:
                if (gt_idx < len(gt_classes)) and gt_classes[gt_idx] == cid:
                    overlay[gt_masks[gt_idx]] = (
                        overlay[gt_masks[gt_idx]] * (1-ALPHA) + COLOR['fn'] * ALPHA)
                    fn_c += 1

            prec = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0.0
            rec  = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0.0
            ax = axes[cid + 1]
            ax.imshow(overlay.astype(np.uint8))
            ax.set_title(f'{cname}\nTP={tp_c}  FP={fp_c}  FN={fn_c}'
                         f'   P={prec:.3f}  R={rec:.3f}', fontsize=11, fontweight='bold')
            ax.axis('off')
            ax.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.7)

        all_tp, all_fp, all_fn = len(matches), len(unmatched_preds), len(unmatched_gts)
        fig.suptitle(
            f'{img_path.name}  —  Overall: TP={all_tp} FP={all_fp} FN={all_fn}'
            f'   conf≥{conf:.2f}  match_iou≥{match_threshold:.2f}',
            fontsize=11, y=1.01
        )
        plt.tight_layout()
        plt.savefig(output_vis_dir / f"{img_path.stem}_matched.png", dpi=150, bbox_inches='tight')
        plt.close()

    print(f"[OK] Visualizations saved to {output_vis_dir}")


def calculate_segmentation_metrics(model_path: str, img_dir: str, label_dir: str,
                                   imgsz: int, conf: float, iou: float, device: str,
                                   match_threshold: float = 0.5) -> Dict:
    """
    Calculate comprehensive metrics using the SAME prediction method as visualization.
    
    CRITICAL: Uses model.predict() directly instead of reading saved txt files
    to ensure consistency with visualization.
    """
    
    img_paths = sorted(Path(img_dir).glob('*'))
    img_paths = [p for p in img_paths if p.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']]
    
    print(f"Found {len(img_paths)} images for metrics calculation")
    
    total_iou_matched = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    total_semantic_intersection = 0
    total_semantic_union = 0
    
    n_processed = 0
    
    # Store per-image stats for debugging
    debug_stats = []
    
    for img_path in tqdm(img_paths, desc="Computing metrics"):
        # Get image dimensions
        img = Image.open(img_path)
        w, h = img.size
        
        # Load predictions using model (SAME as visualization)
        pred_masks, confidences = load_predictions_from_model(
            model_path, str(img_path), imgsz, conf, iou, device
        )
        
        # Load ground truth
        label_path = Path(label_dir) / f"{img_path.stem}.txt"
        gt_masks = load_gt_masks_from_labels(label_path, w, h)
        
        # Match objects
        matches, unmatched_preds, unmatched_gts = match_objects_hungarian(
            pred_masks, gt_masks, iou_threshold=match_threshold
        )
        
        # Count TP, FP, FN
        tp = len(matches)
        fp = len(unmatched_preds)
        fn = len(unmatched_gts)
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
        # Calculate instance-level IoU for matched objects
        for pred_idx, gt_idx in matches:
            pred_mask = pred_masks[pred_idx]
            gt_mask = gt_masks[gt_idx]
            
            intersection = (pred_mask & gt_mask).sum()
            union = (pred_mask | gt_mask).sum()
            
            if union > 0:
                total_iou_matched += intersection / union
        
        # Calculate semantic segmentation style IoU
        if len(pred_masks) > 0:
            pred_mask_combined = np.logical_or.reduce([m for m in pred_masks])
        else:
            pred_mask_combined = np.zeros((h, w), dtype=bool)
            
        if len(gt_masks) > 0:
            gt_mask_combined = np.logical_or.reduce([m for m in gt_masks])
        else:
            gt_mask_combined = np.zeros((h, w), dtype=bool)
        
        semantic_intersection = (pred_mask_combined & gt_mask_combined).sum()
        semantic_union = (pred_mask_combined | gt_mask_combined).sum()
        total_semantic_intersection += semantic_intersection
        total_semantic_union += semantic_union
        
        # Store debug info
        debug_stats.append({
            'image': img_path.name,
            'n_pred': len(pred_masks),
            'n_gt': len(gt_masks),
            'tp': tp, 'fp': fp, 'fn': fn
        })
        
        n_processed += 1
    
    # Calculate metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # IoU metrics
    avg_iou_matched = total_iou_matched / total_tp if total_tp > 0 else 0
    total_objects = total_tp + total_fp + total_fn
    avg_iou_all_objects = total_iou_matched / total_objects if total_objects > 0 else 0
    semantic_iou = total_semantic_intersection / total_semantic_union if total_semantic_union > 0 else 0
    
    # Print debug info for images with anomalies
    print("\n=== Debug Info ===")
    anomalies = [s for s in debug_stats if s['n_pred'] > s['n_gt'] * 3 or s['n_pred'] < s['n_gt'] / 3]
    if anomalies:
        print("Images with unusual prediction counts:")
        for stat in anomalies[:5]:  # Show first 5
            print(f"  {stat['image']}: Pred={stat['n_pred']}, GT={stat['n_gt']}, "
                  f"TP={stat['tp']}, FP={stat['fp']}, FN={stat['fn']}")
    else:
        print("No major anomalies detected in prediction counts")
    
    return {
        'object_precision': precision,
        'object_recall': recall,
        'object_f1': f1,
        'avg_iou_matched': avg_iou_matched,
        'avg_iou_all_objects': avg_iou_all_objects,
        'semantic_iou': semantic_iou,
        'total_gt_objects': total_tp + total_fn,
        'total_pred_objects': total_tp + total_fp,
        'true_positives': total_tp,
        'false_positives': total_fp,
        'false_negatives': total_fn,
        'n_images_processed': n_processed,
        'debug_stats': debug_stats
    }


def optimize_thresholds(model_path: str, img_dir: str, label_dir: str, 
                       imgsz: int, device: str) -> Dict:
    """
    Find optimal conf, iou, and match thresholds by grid search.
    
    Returns best parameters based on F1 score.
    """
    print("\n" + "="*60)
    print("THRESHOLD OPTIMIZATION")
    print("="*60)
    
    # Test ranges
    conf_values = [0.15, 0.20, 0.25, 0.30, 0.35]
    iou_values = [0.5, 0.6, 0.7, 0.8]
    match_values = [0.3, 0.4, 0.5, 0.6]
    
    best_f1 = 0
    best_params = None
    results = []
    
    print(f"\nTesting {len(conf_values)} x {len(iou_values)} x {len(match_values)} = "
          f"{len(conf_values) * len(iou_values) * len(match_values)} combinations...\n")
    
    total_tests = len(conf_values) * len(iou_values) * len(match_values)
    pbar = tqdm(total=total_tests, desc="Optimizing")
    
    for conf in conf_values:
        for iou in iou_values:
            for match in match_values:
                metrics = calculate_segmentation_metrics(
                    model_path, img_dir, label_dir, imgsz, conf, iou, device, match
                )
                
                f1 = metrics['object_f1']
                results.append({
                    'conf': conf, 'iou': iou, 'match': match,
                    'precision': metrics['object_precision'],
                    'recall': metrics['object_recall'],
                    'f1': f1,
                    'iou_matched': metrics['avg_iou_matched']
                })
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_params = {'conf': conf, 'iou': iou, 'match': match}
                    best_metrics = metrics
                
                pbar.update(1)
    
    pbar.close()
    
    # Display results
    print("\n" + "="*60)
    print("OPTIMIZATION RESULTS")
    print("="*60)
    print(f"\nBest Parameters (by F1 score):")
    print(f"  conf_threshold:  {best_params['conf']:.2f}")
    print(f"  iou_threshold:   {best_params['iou']:.2f}")
    print(f"  match_threshold: {best_params['match']:.2f}")
    print(f"\nBest Metrics:")
    print(f"  F1 Score:        {best_metrics['object_f1']:.4f}")
    print(f"  Precision:       {best_metrics['object_precision']:.4f}")
    print(f"  Recall:          {best_metrics['object_recall']:.4f}")
    print(f"  IoU (matched):   {best_metrics['avg_iou_matched']:.4f}")
    
    # Show top 5 configurations
    print(f"\nTop 5 Configurations:")
    sorted_results = sorted(results, key=lambda x: x['f1'], reverse=True)
    for i, r in enumerate(sorted_results[:5], 1):
        print(f"  {i}. conf={r['conf']:.2f}, iou={r['iou']:.2f}, match={r['match']:.2f} -> "
              f"F1={r['f1']:.3f}, P={r['precision']:.3f}, R={r['recall']:.3f}")
    
    print("="*60 + "\n")
    
    return best_params, best_metrics, results


def save_metrics_to_file(metrics: Dict, output_path: Path, split_name: str, params: Dict):
    """Save metrics to a text file"""
    with open(output_path, 'w') as f:
        f.write(f"="*60 + "\n")
        f.write(f"SEGMENTATION METRICS - {split_name.upper()}\n")
        f.write(f"="*60 + "\n\n")
        
        f.write(f"Parameters Used:\n")
        f.write(f"  conf_threshold:  {params['conf_threshold']:.2f}\n")
        f.write(f"  iou_threshold:   {params['iou_threshold']:.2f}\n")
        f.write(f"  match_threshold: {params['match_threshold']:.2f}\n\n")
        
        f.write(f"Dataset:\n")
        f.write(f"  Images Processed:     {metrics['n_images_processed']}\n")
        f.write(f"  Total GT Objects:     {metrics['total_gt_objects']}\n")
        f.write(f"  Total Pred Objects:   {metrics['total_pred_objects']}\n\n")
        
        f.write(f"Object-Level Detection:\n")
        f.write(f"  True Positives (TP):  {metrics['true_positives']}\n")
        f.write(f"  False Positives (FP): {metrics['false_positives']}\n")
        f.write(f"  False Negatives (FN): {metrics['false_negatives']}\n")
        f.write(f"  Precision:            {metrics['object_precision']:.4f}\n")
        f.write(f"  Recall:               {metrics['object_recall']:.4f}\n")
        f.write(f"  F1 Score:             {metrics['object_f1']:.4f}\n\n")
        
        f.write(f"Pixel-Level IoU Metrics:\n")
        f.write(f"  IoU (matched only):   {metrics['avg_iou_matched']:.4f}\n")
        f.write(f"  IoU (with FP/FN=0):   {metrics['avg_iou_all_objects']:.4f}\n")
        f.write(f"  Semantic IoU:         {metrics['semantic_iou']:.4f}\n")
        f.write(f"="*60 + "\n")
    
    print(f"[OK] Metrics saved to {output_path}")


def print_metrics_summary(metrics: Dict, split_name: str, params: Dict):
    """Print metrics summary to console"""
    print(f"\n{'='*60}")
    print(f"METRICS SUMMARY - {split_name.upper()}")
    print(f"{'='*60}")
    print(f"Parameters: conf={params['conf_threshold']:.2f}, iou={params['iou_threshold']:.2f}, match={params['match_threshold']:.2f}")
    print(f"Images: {metrics['n_images_processed']} | GT Objects: {metrics['total_gt_objects']} | Pred Objects: {metrics['total_pred_objects']}")
    print(f"TP={metrics['true_positives']}, FP={metrics['false_positives']}, FN={metrics['false_negatives']}")
    print(f"Precision: {metrics['object_precision']:.4f} | Recall: {metrics['object_recall']:.4f} | F1: {metrics['object_f1']:.4f}")
    print(f"IoU (matched): {metrics['avg_iou_matched']:.4f}")
    print(f"{'='*60}\n")


def save_run_manifest(output_path: Path, config: Dict, dataset_yaml: Path,
                      split_stats: Dict[str, Dict[str, int]], best_model_path: str = None,
                      validation_summary: Dict = None):
    """Save a compact manifest tying a training run to dataset sources and settings."""
    manifest = {
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'experiment_name': config['experiment_name'],
        'dataset_id': config.get('dataset_id', 'unspecified_dataset'),
        'run_notes': config.get('run_notes', ''),
        'class_names': config['class_names'],
        'source_data': {
            'train_dir': config['train_dir'],
            'val_dir': config['val_dir'],
            'test_dir': config['test_dir'],
            'prepared_dataset_yaml': str(dataset_yaml),
        },
        'training_params': {
            'model_size': config['model_size'],
            'use_yolov11': config['USE_YOLOV11'],
            'epochs': config['epochs'],
            'imgsz': config['imgsz'],
            'batch_size': config['batch_size'],
            'patience': config['patience'],
            'device': config['device'],
            'conf_threshold': config['conf_threshold'],
            'iou_threshold': config['iou_threshold'],
            'match_threshold': config['match_threshold'],
        },
        'split_stats': split_stats,
        'best_model_path': best_model_path,
        'validation_summary': validation_summary or {},
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(manifest, f, sort_keys=False)


def append_run_summary_logs(output_root: Path, config: Dict,
                            split_stats: Dict[str, Dict[str, int]],
                            best_model_path: str = None,
                            validation_summary: Dict = None):
    """Append a one-line summary for this run to CSV and Markdown logs."""
    validation_summary = validation_summary or {}
    model_tag = f"y11{config['model_size']}-seg" if config['USE_YOLOV11'] else f"y8{config['model_size']}-seg"
    timestamp = datetime.now().isoformat(timespec='seconds')

    row = {
        'timestamp': timestamp,
        'experiment_name': config['experiment_name'],
        'dataset_id': config.get('dataset_id', ''),
        'model': model_tag,
        'classes': '|'.join(config['class_names']),
        'epochs': config['epochs'],
        'imgsz': config['imgsz'],
        'batch_size': config['batch_size'],
        'patience': config['patience'],
        'train_images': split_stats.get('train', {}).get('n_images', ''),
        'train_instances': split_stats.get('train', {}).get('n_instances', ''),
        'val_images': split_stats.get('val', {}).get('n_images', ''),
        'val_instances': split_stats.get('val', {}).get('n_instances', ''),
        'test_images': split_stats.get('test', {}).get('n_images', ''),
        'test_instances': split_stats.get('test', {}).get('n_instances', ''),
        'box_map50': f"{validation_summary['box_map50']:.4f}" if 'box_map50' in validation_summary else '',
        'box_map50_95': f"{validation_summary['box_map50_95']:.4f}" if 'box_map50_95' in validation_summary else '',
        'mask_map50': f"{validation_summary['mask_map50']:.4f}" if 'mask_map50' in validation_summary else '',
        'mask_map50_95': f"{validation_summary['mask_map50_95']:.4f}" if 'mask_map50_95' in validation_summary else '',
        'best_model_path': best_model_path or '',
        'run_notes': config.get('run_notes', ''),
    }

    csv_path = output_root / 'EXPERIMENT_LOG.csv'
    fieldnames = list(row.keys())
    file_exists = csv_path.exists()
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    md_path = output_root / 'EXPERIMENT_LOG.md'
    if not md_path.exists():
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write('# Experiment Log\n\n')
            f.write('| timestamp | experiment | dataset | model | epochs | imgsz | box mAP50 | mask mAP50 | notes |\n')
            f.write('|---|---|---|---|---:|---:|---:|---:|---|\n')
    with open(md_path, 'a', encoding='utf-8') as f:
        f.write(
            f"| {row['timestamp']} | {row['experiment_name']} | {row['dataset_id']} | {row['model']} | "
            f"{row['epochs']} | {row['imgsz']} | {row['box_map50'] or '-'} | {row['mask_map50'] or '-'} | {row['run_notes']} |\n"
        )


def main():
    """
    Main training and evaluation pipeline.
    
    Workflow:
    1. Prepare datasets (train/val/test)
    2. Train model (optional, set DO_TRAINING=True)
    3. Load trained model
    4. Run inference on all splits
    5. Save all results to output folder
    """
    
    # =====================================================================
    # CONFIGURATION
    # =====================================================================
    CONFIG = {
        # Data paths - stratified EV/multilayer-EV dataset
        'train_dir': r'C:\Users\pujan\Desktop\cryoEV\Carney CryoEV\training outputs\roboflow20260604_stratified\images\train',
        'val_dir':   r'C:\Users\pujan\Desktop\cryoEV\Carney CryoEV\training outputs\roboflow20260604_stratified\images\val',
        'test_dir':  r'C:\Users\pujan\Desktop\cryoEV\Carney CryoEV\training outputs\roboflow20260604_stratified\images\test',
        'output_root': r'C:\Users\pujan\Desktop\cryoEV\Carney CryoEV\training outputs\roboflow20260604_stratified',
        'dataset_id': 'roboflow20260604_stratified',
        'run_notes': 'yolov8s-seg, SGD lr0=0.01, imgsz=768, batch=2, 300ep, patience=75 — improved over AdamW nano baseline',
        'run_label': 'SGD_s_768_Run1',

        # Model configuration
        'class_names': ['EV', 'multilayer EV'],
        'model_size': 's',       # was 'n' — small model for better capacity
        'experiment_name': None,  # Auto-generated below
        'USE_YOLOV11': False,     # YOLOv8

        # Training parameters
        'epochs': 300,
        'imgsz': 768,
        'batch_size': 2,
        'patience': 75,           # was 50
        'device': '0',
        
        # Inference parameters
        'conf_threshold': 0.35,   # Confidence threshold for predictions
        'iou_threshold': 0.5,     # NMS IoU threshold
        'match_threshold': 0.30,  # Evaluation matching threshold
        
        # Workflow control
        'DO_TRAINING': True,            # Train a new model
        'DO_THRESHOLD_OPTIMIZATION': False,
        'DO_INFERENCE': True,           # Then evaluate on val/test
        'VISUALIZE_AUGMENTATION': False,
        'INFERENCE_SPLITS': ['val', 'test'],    # Used only if DO_INFERENCE=True
        
        # If not training, specify path to existing model (leave None to auto-fill)
        'pretrained_model_path': None,

        # Set to an existing dataset.yaml path to skip the prepare step entirely.
        # Leave None to run prepare_yolo_dataset() as normal.
        'dataset_yaml_override': r'C:\Users\pujan\Desktop\cryoEV\Carney CryoEV\training outputs\roboflow20260604_stratified\dataset.yaml',
    }

    if not CONFIG.get('experiment_name'):
        timestamp_tag = datetime.now().strftime('%Y-%m-%d_%H%M%S')
        model_tag = f"y11{CONFIG['model_size']}" if CONFIG['USE_YOLOV11'] else f"y8{CONFIG['model_size']}"
        class_tag = f"{len(CONFIG['class_names'])}class"
        label_tag = f"_{CONFIG['run_label']}" if CONFIG.get('run_label') else ''
        CONFIG['experiment_name'] = f"{timestamp_tag}_{model_tag}_{class_tag}_{CONFIG['imgsz']}_ep{CONFIG['epochs']}{label_tag}"

    if not CONFIG.get('pretrained_model_path'):
        CONFIG['pretrained_model_path'] = str(
            Path(CONFIG['output_root']) / 'runs' / CONFIG['experiment_name'] / 'weights' / 'best.pt'
        )
    
    print("\n" + "="*80)
    print(" "*20 + "YOLO INSTANCE SEGMENTATION PIPELINE")
    print(" "*25 + f"({'YOLOv11' if CONFIG['USE_YOLOV11'] else 'YOLOv8'})")
    print("="*80 + "\n")
    
    # =====================================================================
    # SETUP PATHS
    # =====================================================================
    output_root = Path(CONFIG['output_root'])
    yolo_dataset_root = output_root / 'dataset'
    runs_root = output_root / 'runs'
    
    # Create directories
    yolo_dataset_root.mkdir(parents=True, exist_ok=True)
    runs_root.mkdir(parents=True, exist_ok=True)
    
    # =====================================================================
    # STEP 1: PREPARE DATASETS
    # =====================================================================
    print("="*80)
    print("STEP 1: DATASET PREPARATION")
    print("="*80 + "\n")

    if CONFIG.get('dataset_yaml_override'):
        yaml_path = Path(CONFIG['dataset_yaml_override'])
        print(f"[OK] Using existing dataset yaml: {yaml_path}")
        split_stats = {'train': {}, 'val': {}, 'test': {}}
    else:
        # Prepare train split
        train_stats = prepare_yolo_dataset(CONFIG['train_dir'], str(yolo_dataset_root), 'train')
        print(f"[OK] Train: {train_stats['n_images']} images, {train_stats['n_instances']} instances")

        # Prepare val split
        val_stats = prepare_yolo_dataset(CONFIG['val_dir'], str(yolo_dataset_root), 'val')
        print(f"[OK] Val:   {val_stats['n_images']} images, {val_stats['n_instances']} instances")

        # Prepare test split (if exists)
        test_dir = Path(CONFIG['test_dir'])
        if test_dir.exists() and 'test' in CONFIG['INFERENCE_SPLITS']:
            test_stats = prepare_yolo_dataset(CONFIG['test_dir'], str(yolo_dataset_root), 'test')
            print(f"[OK] Test:  {test_stats['n_images']} images, {test_stats['n_instances']} instances")
        else:
            if 'test' in CONFIG['INFERENCE_SPLITS']:
                print(f"[!] Test directory not found: {test_dir}")
                CONFIG['INFERENCE_SPLITS'].remove('test')

        # Create YOLO config file
        yaml_path = yolo_dataset_root / 'dataset.yaml'
        create_yolo_yaml(str(yolo_dataset_root), str(yaml_path), CONFIG['class_names'])

        split_stats = {
            'train': train_stats,
            'val': val_stats,
            'test': test_stats if 'test_stats' in locals() else {},
        }
    
    # =====================================================================
    # OPTIONAL: VISUALIZE AUGMENTATIONS
    # =====================================================================
    if CONFIG.get('VISUALIZE_AUGMENTATION', False):
        print("\n" + "="*80)
        print("VISUALIZING AUGMENTED TRAINING SAMPLES")
        print("="*80 + "\n")
        
        visualize_augmented_samples(
            data_yaml=str(yaml_path),
            model_size=CONFIG['model_size'],
            imgsz=CONFIG['imgsz'],
            n_samples=16,
            output_dir=str(output_root / 'augmentation_samples'),
            use_yolov11=CONFIG['USE_YOLOV11']
        )
    
    # =====================================================================
    # STEP 2: TRAIN MODEL (Optional)
    # =====================================================================
    if CONFIG['DO_TRAINING']:
        print("\n" + "="*80)
        print("STEP 2: MODEL TRAINING")
        print("="*80 + "\n")

        # Publish the intended run path immediately so the live monitor can
        # follow the current run instead of the previously completed one.
        experiment_log_root = output_root.parent.parent
        expected_training_output = runs_root / CONFIG['experiment_name']
        (experiment_log_root / 'LATEST_RUN.txt').write_text(
            f"experiment_name: {CONFIG['experiment_name']}\n"
            f"training_dir: {expected_training_output}\n"
            f"best_model_path: {expected_training_output / 'weights' / 'best.pt'}\n"
            f"dataset_yaml: {yaml_path}\n"
            f"experiment_log_csv: {experiment_log_root / 'EXPERIMENT_LOG.csv'}\n"
            f"experiment_log_md: {experiment_log_root / 'EXPERIMENT_LOG.md'}\n",
            encoding='utf-8'
        )
        
        results = train_yolo_segmentation(
            data_yaml=str(yaml_path),
            model_size=CONFIG['model_size'],
            epochs=CONFIG['epochs'],
            imgsz=CONFIG['imgsz'],
            batch_size=CONFIG['batch_size'],
            device=CONFIG['device'],
            project=str(runs_root),
            name=CONFIG['experiment_name'],
            patience=CONFIG['patience'],
            use_v11=CONFIG['USE_YOLOV11'],
            optimizer='SGD',
            lr0=0.01,
            lrf=0.001,
            copy_paste=0.3,
            mosaic=0.8,
            workers=0,
        )
        
        # Use the actual Ultralytics save directory in case the run name is adjusted.
        training_output = Path(results.save_dir)
        CONFIG['experiment_name'] = training_output.name
        best_model_path = training_output / 'weights' / 'best.pt'
        print(f"\n[OK] Training complete!")
        print(f"[OK] Best model saved to: {best_model_path}")
        
        # Validate on validation set
        print("\n" + "="*80)
        print("VALIDATION ON VAL SET")
        print("="*80 + "\n")
        val_results = validate_yolo_model(
            str(best_model_path), 
            str(yaml_path), 
            imgsz=CONFIG['imgsz'],
            batch_size=CONFIG['batch_size'], 
            device=CONFIG['device'],
            project=str(training_output),
            name='validation'
        )
        validation_summary = {
            'box_map50': float(val_results.box.map50),
            'box_map50_95': float(val_results.box.map),
            'mask_map50': float(val_results.seg.map50),
            'mask_map50_95': float(val_results.seg.map),
        }
        save_run_manifest(
            training_output / 'run_manifest.yaml',
            CONFIG,
            yaml_path,
            split_stats,
            best_model_path=str(best_model_path),
            validation_summary=validation_summary,
        )
        experiment_log_root = output_root.parent.parent
        append_run_summary_logs(
            experiment_log_root,
            CONFIG,
            split_stats,
            best_model_path=str(best_model_path),
            validation_summary=validation_summary,
        )
        (experiment_log_root / 'LATEST_RUN.txt').write_text(
            f"experiment_name: {CONFIG['experiment_name']}\n"
            f"training_dir: {training_output}\n"
            f"best_model_path: {best_model_path}\n"
            f"dataset_yaml: {yaml_path}\n"
            f"experiment_log_csv: {experiment_log_root / 'EXPERIMENT_LOG.csv'}\n"
            f"experiment_log_md: {experiment_log_root / 'EXPERIMENT_LOG.md'}\n",
            encoding='utf-8'
        )
    else:
        print("\n" + "="*80)
        print("STEP 2: LOADING PRETRAINED MODEL")
        print("="*80 + "\n")
        
        best_model_path = Path(CONFIG['pretrained_model_path'])
        
        if not best_model_path.exists():
            print(f"[ERR] Error: Model not found at {best_model_path}")
            print(f"   Please set DO_TRAINING=True to train a new model")
            print(f"   Or update 'pretrained_model_path' in CONFIG")
            return
        
        training_output = best_model_path.parent.parent  # runs/<exp>/weights/best.pt -> runs/<exp>/
        CONFIG['experiment_name'] = training_output.name
        print(f"[OK] Loaded model: {best_model_path}")
    
    # =====================================================================
    # STEP 3: OPTIMIZE THRESHOLDS (Optional)
    # =====================================================================
    if CONFIG['DO_THRESHOLD_OPTIMIZATION']:
        print("\n" + "="*80)
        print("STEP 3: THRESHOLD OPTIMIZATION ON VAL SET")
        print("="*80 + "\n")
        
        val_images_dir = yolo_dataset_root / 'images' / 'val'
        val_labels_dir = yolo_dataset_root / 'labels' / 'val'
        
        best_params, best_metrics, all_results = optimize_thresholds(
            str(best_model_path),
            str(val_images_dir),
            str(val_labels_dir),
            CONFIG['imgsz'],
            CONFIG['device']
        )
        
        # Update config with optimized values
        CONFIG['conf_threshold'] = best_params['conf']
        CONFIG['iou_threshold'] = best_params['iou']
        CONFIG['match_threshold'] = best_params['match']
        
        print(f"[OK] Updated CONFIG with optimized thresholds")
        
        # Save optimization results
        opt_results_file = training_output / 'threshold_optimization_results.txt'
        with open(opt_results_file, 'w') as f:
            f.write("Threshold Optimization Results\n")
            f.write("="*60 + "\n\n")
            f.write(f"Best Parameters:\n")
            f.write(f"  conf_threshold:  {best_params['conf']:.2f}\n")
            f.write(f"  iou_threshold:   {best_params['iou']:.2f}\n")
            f.write(f"  match_threshold: {best_params['match']:.2f}\n\n")
            f.write(f"Best Metrics:\n")
            f.write(f"  F1 Score:  {best_metrics['object_f1']:.4f}\n")
            f.write(f"  Precision: {best_metrics['object_precision']:.4f}\n")
            f.write(f"  Recall:    {best_metrics['object_recall']:.4f}\n")
        print(f"[OK] Optimization results saved to {opt_results_file}")
    
    # =====================================================================
    # STEP 4: RUN INFERENCE ON ALL SPLITS
    # =====================================================================
    all_metrics = {}
    if CONFIG['DO_INFERENCE']:
        print("\n" + "="*80)
        print("STEP 4: INFERENCE ON ALL SPLITS")
        print("="*80 + "\n")
        
        for split in CONFIG['INFERENCE_SPLITS']:
            print(f"\n{'='*80}")
            print(f"PROCESSING {split.upper()} SET")
            print(f"{'='*80}\n")
            
            split_images_dir = yolo_dataset_root / 'images' / split
            split_labels_dir = yolo_dataset_root / 'labels' / split
            split_output_dir = training_output / 'inference' / split
            split_output_dir.mkdir(parents=True, exist_ok=True)
            
            if not split_images_dir.exists():
                print(f"[!] Skipping {split}: directory not found")
                continue
            
            # Create visualizations
            print(f"Creating visualizations for {split} set...")
            visualize_predictions_with_matching(
                str(best_model_path),
                str(split_images_dir),
                str(split_labels_dir),
                str(split_output_dir),
                imgsz=CONFIG['imgsz'],
                conf=CONFIG['conf_threshold'],
                iou=CONFIG['iou_threshold'],
                device=CONFIG['device'],
                match_threshold=CONFIG['match_threshold']
            )
            
            # Calculate metrics
            print(f"\nCalculating metrics for {split} set...")
            metrics = calculate_segmentation_metrics(
                model_path=str(best_model_path),
                img_dir=str(split_images_dir),
                label_dir=str(split_labels_dir),
                imgsz=CONFIG['imgsz'],
                conf=CONFIG['conf_threshold'],
                iou=CONFIG['iou_threshold'],
                device=CONFIG['device'],
                match_threshold=CONFIG['match_threshold']
            )
            
            all_metrics[split] = metrics
            
            # Save metrics to file
            metrics_file = split_output_dir / f'metrics_{split}.txt'
            save_metrics_to_file(metrics, metrics_file, split, CONFIG)

            # Calculate and save per-class metrics when class labels are available
            per_class_metrics = calculate_per_class_segmentation_metrics(
                model_path=str(best_model_path),
                img_dir=str(split_images_dir),
                label_dir=str(split_labels_dir),
                imgsz=CONFIG['imgsz'],
                conf=CONFIG['conf_threshold'],
                iou=CONFIG['iou_threshold'],
                device=CONFIG['device'],
                class_names=CONFIG['class_names'],
                match_threshold=CONFIG['match_threshold']
            )
            save_per_class_metrics(per_class_metrics, split_output_dir / f'metrics_per_class_{split}.txt', split)

            if len(CONFIG['class_names']) > 1:
                print(f"Per-class results for {split}:")
                for class_name, class_metrics in per_class_metrics.items():
                    print(
                        f"  {class_name:<12} TP={class_metrics['tp']:<4} FP={class_metrics['fp']:<4} "
                        f"FN={class_metrics['fn']:<4} P={class_metrics['precision']:.4f} "
                        f"R={class_metrics['recall']:.4f} F1={class_metrics['f1']:.4f}"
                    )
                print("")
            
            # Print summary
            print_metrics_summary(metrics, split, CONFIG)
    
    # =====================================================================
    # FINAL SUMMARY
    # =====================================================================
    print("\n" + "="*80)
    print("PIPELINE COMPLETE - FINAL SUMMARY")
    print("="*80 + "\n")
    
    print("Configuration:")
    print(f"  Model: {best_model_path}")
    print(f"  conf_threshold:  {CONFIG['conf_threshold']:.2f}")
    print(f"  iou_threshold:   {CONFIG['iou_threshold']:.2f}")
    print(f"  match_threshold: {CONFIG['match_threshold']:.2f}\n")
    
    if CONFIG['DO_INFERENCE'] and all_metrics:
        print("Results Summary:")
        print(f"{'Split':<10} {'F1':<8} {'Precision':<12} {'Recall':<10} {'IoU':<10}")
        print("-" * 60)
        for split, metrics in all_metrics.items():
            print(f"{split:<10} {metrics['object_f1']:<8.4f} "
                  f"{metrics['object_precision']:<12.4f} "
                  f"{metrics['object_recall']:<10.4f} "
                  f"{metrics['avg_iou_matched']:<10.4f}")
    
    print(f"\n{'='*80}")
    print("Output Structure:")
    print(f"{'='*80}")
    print(f"{output_root}/")
    print(f"+-- dataset/")
    print(f"|   +-- images/")
    print(f"|   |   +-- train/")
    print(f"|   |   +-- val/")
    print(f"|   |   \-- test/")
    print(f"|   +-- labels/")
    print(f"|   |   +-- train/")
    print(f"|   |   +-- val/")
    print(f"|   |   \-- test/")
    print(f"|   \-- dataset.yaml")
    print(f"\-- runs/")
    print(f"    \-- {CONFIG['experiment_name']}/")
    print(f"        +-- weights/  (best.pt, last.pt)")
    print(f"        +-- results.csv")
    print(f"        +-- run_manifest.yaml")
    if CONFIG['DO_INFERENCE'] and all_metrics:
        print(f"        \-- inference/")
        for split in CONFIG['INFERENCE_SPLITS']:
            if split in all_metrics:
                print(f"            +-- {split}/")
                print(f"            |   +-- visualizations/")
                print(f"            |   \-- metrics_{split}.txt")
    else:
        print(f"        \-- inference/  [set DO_INFERENCE=True to generate]")
    print(f"{'='*80}\n")
    
    print(f"[OK] All results saved to: {output_root}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
