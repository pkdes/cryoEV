"""
YOLO Instance Segmentation Training Script for Cryo-EM Vesicles
Fixed version with consistent prediction handling and optimized thresholds.
"""

import csv
import yaml
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
import numpy as np
from PIL import Image
import cv2
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
                           mask_ratio: int = 4, rect: bool = False,
                           use_v11: bool = True, workers: int = 8, optimizer: str = 'AdamW',
                           lr0: float = 0.002, lrf: float = 0.001,
                           hsv_h: float = 0.015, hsv_s: float = 0.7, hsv_v: float = 0.4,
                           degrees: float = 45.0, translate: float = 0.1, scale: float = 0.5,
                           shear: float = 0.0, perspective: float = 0.0,
                           fliplr: float = 0.5, flipud: float = 0.5,
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

    import time, torch, gc
    torch.cuda.reset_peak_memory_stats()
    t_start = time.time()

    try:
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
            rect=rect,

            # Mask generation
            mask_ratio=mask_ratio,

            # Data augmentation
            hsv_h=hsv_h,
            hsv_s=hsv_s,
            hsv_v=hsv_v,
            degrees=degrees,
            translate=translate,
            scale=scale,
            shear=shear,
            perspective=perspective,
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
    finally:
        # Always release the model + CUDA cache, success or failure. Without
        # this, a crashed run (e.g. OOM) can leave GPU memory fragmented/held
        # by the exception traceback, causing the NEXT run in a sweep to fail
        # even if its own config would otherwise fit comfortably.
        del model
        gc.collect()
        torch.cuda.empty_cache()

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


def save_predictions_yolo_format(pred_masks: List[np.ndarray], confidences: List[float], class_ids: List[int],
                                 img_width: int, img_height: int, output_path) -> None:
    """
    Persist predictions as polygon lines: 'class_id confidence x1 y1 x2 y2 ...' (normalized
    coords) -- same format as the GT label files plus a confidence column. Lets downstream
    analyses (multilayer containment, confidence sweeps, etc.) reuse predictions without
    re-running inference. Each mask is reduced to its largest external contour, the same
    single-polygon-per-instance precision already used for GT labels.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for mask, conf, cls_id in zip(pred_masks, confidences, class_ids):
        mask_u8 = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        polygon = max(contours, key=cv2.contourArea).reshape(-1, 2)
        if len(polygon) < 3:
            continue
        coords = []
        for x, y in polygon:
            coords.append(f"{x / img_width:.6f}")
            coords.append(f"{y / img_height:.6f}")
        lines.append(f"{cls_id} {conf:.4f} " + " ".join(coords))
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding='utf-8')


def load_predictions_yolo_format(label_path, img_width: int, img_height: int) -> Tuple[List[np.ndarray], List[float], List[int]]:
    """Inverse of save_predictions_yolo_format(): rasterize cached prediction polygons back to masks."""
    label_path = Path(label_path)
    pred_masks, confidences, class_ids = [], [], []
    if not label_path.exists():
        return pred_masks, confidences, class_ids
    with open(label_path) as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split()
            class_id = int(parts[0])
            conf = float(parts[1])
            coords = [float(x) for x in parts[2:]]
            polygon = np.array([(int(coords[i] * img_width), int(coords[i + 1] * img_height))
                               for i in range(0, len(coords), 2)], dtype=np.int32)
            mask_u8 = np.zeros((img_height, img_width), dtype=np.uint8)
            cv2.fillPoly(mask_u8, [polygon], 1)
            pred_masks.append(mask_u8 > 0)
            confidences.append(conf)
            class_ids.append(class_id)
    return pred_masks, confidences, class_ids


def load_prediction_polygons(label_path, img_width: int, img_height: int) -> Tuple[List[np.ndarray], List[float]]:
    """Read a prediction cache file as float polygons in pixel coords (no rasterization) plus confidences.
    Complements load_predictions_yolo_format() where the polygon itself is needed (drawing, ellipse fits)."""
    label_path = Path(label_path)
    polys, confs = [], []
    if not label_path.exists():
        return polys, confs
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 8:
            continue
        confs.append(float(parts[1]))
        polys.append(np.array(parts[2:], dtype=np.float32).reshape(-1, 2) * [img_width, img_height])
    return polys, confs


def polygon_to_mask(polygon: np.ndarray, height: int, width: int) -> np.ndarray:
    """Rasterize a float pixel-coord polygon. Rounds (not truncates) vertices: truncation roughens
    the edge and biases circularity low (~0.87 vs 0.90 against live-inference masks)."""
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(polygon).astype(np.int32)], 1)
    return mask > 0

def export_predictions_for_split(model_path: str, img_dir: str, output_dir, imgsz: int, conf: float,
                                 iou: float, device: str) -> None:
    """Run inference once over every image in a split and cache the raw predicted polygons to disk."""
    output_dir = Path(output_dir)
    img_paths = sorted(Path(img_dir).glob('*'))
    img_paths = [p for p in img_paths if p.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']]

    for img_path in tqdm(img_paths, desc=f"Caching predictions ({output_dir.name})"):
        img = Image.open(img_path)
        w, h = img.size
        pred_masks, confidences, class_ids = load_predictions_with_classes_from_model(
            model_path, str(img_path), imgsz, conf, iou, device
        )
        save_predictions_yolo_format(pred_masks, confidences, class_ids, w, h,
                                     output_dir / f"{img_path.stem}.txt")


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
    ALPHA = 0.25  # light fill only -- outlines (below) carry the main signal so nested/overlapping
                  # masks stay legible instead of blending into an indistinguishable smear
    OUTLINE_THICKNESS = 3

    def _draw_mask(overlay: np.ndarray, mask: np.ndarray, color: np.ndarray) -> None:
        """Light alpha fill + a thick contour outline, so overlapping/nested masks stay legible."""
        overlay[mask] = overlay[mask] * (1 - ALPHA) + color * ALPHA
        mask_uint8 = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color.tolist(), thickness=OUTLINE_THICKNESS)

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
                    _draw_mask(overlay, pred_masks[pred_idx], COLOR['tp'])
                    tp_c += 1
                elif pred_classes[pred_idx] == cid:
                    _draw_mask(overlay, pred_masks[pred_idx], COLOR['fp'])
                    fp_c += 1
                elif gt_classes[gt_idx] == cid:
                    _draw_mask(overlay, gt_masks[gt_idx], COLOR['fn'])
                    fn_c += 1

            for pred_idx in unmatched_preds:
                if (pred_idx < len(pred_classes)) and pred_classes[pred_idx] == cid:
                    _draw_mask(overlay, pred_masks[pred_idx], COLOR['fp'])
                    fp_c += 1

            for gt_idx in unmatched_gts:
                if (gt_idx < len(gt_classes)) and gt_classes[gt_idx] == cid:
                    _draw_mask(overlay, gt_masks[gt_idx], COLOR['fn'])
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
