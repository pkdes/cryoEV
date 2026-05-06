"""
Cross-Grid Inference & Comparison
==================================
Runs YOLO instance segmentation on all 4 cryo-EM grids, tags each detection
with its grid ID and condition, then produces cross-grid comparison plots and
a combined CSV.

Grid layout (2x2 condition design):
  - Grid 4 & Grid 6 → Condition A (no-PLL, film type 1)
  - Grid 5 & Grid 7 → Condition B (no-PLL, film type 2)

Usage:
    python analysis/cross_grid_comparison.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import re
import csv
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # non-interactive backend — safe for headless runs
import matplotlib.pyplot as plt

from inference.inference import extract_instances_yolo
from inference.perf_log import PerformanceLogger
from analysis.morphology import analyze_instances

# ---------------------------------------------------------------------------
# Configuration — edit these paths if needed
# ---------------------------------------------------------------------------
IMAGES_DIR = Path(
    r"C:\Users\ML-2619\Desktop\Pujan Cryo\20260430 - PLL-film exp analysis\images"
)
MODEL_WEIGHTS = Path(
    r"C:\Users\ML-2619\Desktop\Pujan Cryo\cryo-ev pipeline\Model Training by Yifei"
    r"\round_2\results_yolov8_heavy_augmentation\training\vesicle_instance_seg_v2\weights\best.pt"
)
OUTPUT_DIR = Path(
    r"C:\Users\ML-2619\Desktop\Pujan Cryo\20260430 - PLL-film exp analysis\cross_grid_results"
)

# YOLO inference parameters
IMGSZ   = 1024
CONF    = 0.25
IOU     = 0.7
DEVICE  = 'cpu'   # set to 'cuda' if a GPU is available
SAVE_OVERLAYS = True

# Optional: pixel size in nm/px (set to None to keep everything in pixels)
PIXEL_SIZE: Optional[float] = None

# 2x2 condition mapping: grid number → condition label
# 2x2 design: grid type (A = grids 4/6, B = grids 5/7) × PLL treatment
GRID_CONDITIONS: Dict[int, str] = {
    4: "Type A, no PLL",
    5: "Type B, no PLL",
    6: "Type A, PLL",
    7: "Type B, PLL",
}

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
# ---------------------------------------------------------------------------


def save_detection_overlay(
    image_path: Path,
    masks: List[np.ndarray],
    confidences: List[float],
    out_dir: Path,
    alpha: float = 0.35,
) -> Optional[Path]:
    """Save a per-image overlay showing all predicted instance masks."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return None

    # Convert BGR->RGB for plotting overlays, then back to BGR for saving.
    base = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    overlay = base.copy()

    for idx, mask in enumerate(masks):
        if mask.shape[:2] != base.shape[:2]:
            continue
        color = np.array([
            (37 * idx + 90) % 255,
            (71 * idx + 130) % 255,
            (109 * idx + 170) % 255,
        ], dtype=np.uint8)
        overlay[mask] = color

        mask_u8 = (mask.astype(np.uint8)) * 255
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(overlay, contours, -1, (255, 255, 255), 1)

    blended = cv2.addWeighted(base, 1.0 - alpha, overlay, alpha, 0.0)

    # Put detection count and mean confidence in the top-left corner.
    count = len(masks)
    mean_conf = float(np.mean(confidences)) if confidences else 0.0
    label = f"Detections: {count} | Mean conf: {mean_conf:.3f}"
    cv2.putText(
        blended,
        label,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{image_path.stem}_overlay.png"
    cv2.imwrite(str(out_path), cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
    return out_path


def _grid_id_from_filename(name: str) -> Optional[int]:
    """Extract the leading grid number from a filename like 'grid4_...'."""
    m = re.match(r'grid(\d+)', name, re.IGNORECASE)
    return int(m.group(1)) if m else None


def collect_images() -> Dict[int, List[Path]]:
    """Return {grid_id: [image_paths]} for all supported image files, excluding rejects folder."""
    grid_images: Dict[int, List[Path]] = {}
    for path in sorted(IMAGES_DIR.rglob('*')):
        # Skip files in the rejects subfolder
        if 'rejects' in path.parts:
            continue
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        gid = _grid_id_from_filename(path.name)
        if gid is None:
            continue
        grid_images.setdefault(gid, []).append(path)
    return grid_images


def run_inference_all_grids(
    grid_images: Dict[int, List[Path]],
    perf: PerformanceLogger,
    overlay_root: Optional[Path] = None,
) -> tuple[List[Dict], int]:
    """
    Run YOLO inference on every image across all grids.

    Returns a flat list of morphology records, each augmented with:
        - grid_id
        - condition
        - image_name
        - global_object_id (unique across all images)
    """
    all_records: List[Dict] = []
    n_processed = 0
    global_obj_id = 0

    grid_ids = sorted(grid_images.keys())
    for gid in grid_ids:
        images = grid_images[gid]
        condition = GRID_CONDITIONS.get(gid, f"grid{gid}")
        print(f"\n[Grid {gid} | {condition}] — {len(images)} image(s)")

        for img_path in images:
            print(f"  Inferring: {img_path.name} ...", end=' ', flush=True)
            try:
                t0 = time.perf_counter()
                masks, confs, _bboxes, polys = extract_instances_yolo(
                    model_path=str(MODEL_WEIGHTS),
                    img_path=str(img_path),
                    imgsz=IMGSZ,
                    conf=CONF,
                    iou=IOU,
                    device=DEVICE,
                )
                inference_s = time.perf_counter() - t0
            except Exception as exc:
                print(f"ERROR ({exc})")
                continue

            perf.log_image(
                image_path=img_path,
                n_raw_detections=len(masks),
                inference_time_s=inference_s,
                image=None,
            )
            n_processed += 1

            records = analyze_instances(
                instance_masks=masks,
                confidences=confs,
                pixel_size=PIXEL_SIZE,
                polygon_points_list=polys,
            )

            if overlay_root is not None:
                overlay_dir = overlay_root / f"grid{gid}"
                save_detection_overlay(
                    image_path=img_path,
                    masks=masks,
                    confidences=confs,
                    out_dir=overlay_dir,
                )

            print(f"{len(records)} detections")

            for rec in records:
                rec['grid_id']          = gid
                rec['condition']        = condition
                rec['image_name']       = img_path.name
                rec['global_object_id'] = global_obj_id
                global_obj_id += 1
                all_records.append(rec)

    return all_records, n_processed


def save_combined_csv(records: List[Dict], out_path: Path) -> None:
    """Write all records to a single CSV file."""
    if not records:
        print("No records — CSV not written.")
        return

    # Ensure metadata columns come first
    meta_cols = ['global_object_id', 'grid_id', 'condition', 'image_name',
                 'object_id', 'confidence']
    morph_cols = [k for k in records[0].keys() if k not in meta_cols]
    fieldnames = [c for c in meta_cols if c in records[0]] + morph_cols

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)
    print(f"\nCombined CSV saved → {out_path}  ({len(records)} rows)")


def save_per_image_summary(records: List[Dict], out_path: Path) -> None:
    """Generate and save a summary of detection counts per image."""
    if not records:
        print("No records — summary not written.")
        return

    # Group by image and count detections
    by_image = {}
    for rec in records:
        img = rec.get('image_name', 'unknown')
        gid = rec.get('grid_id', 'unknown')
        cond = rec.get('condition', 'unknown')
        by_image.setdefault(img, {'grid_id': gid, 'condition': cond, 'count': 0})
        by_image[img]['count'] += 1

    # Sort by grid, then by image name
    sorted_items = sorted(by_image.items(), key=lambda x: (x[1]['grid_id'], x[0]))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['grid_id', 'condition', 'image_name', 'detection_count'])
        writer.writeheader()
        for img_name, data in sorted_items:
            writer.writerow({
                'grid_id': data['grid_id'],
                'condition': data['condition'],
                'image_name': img_name,
                'detection_count': data['count'],
            })

    # Also print to stdout for quick review
    print("\nDetections per image:")
    print("─" * 120)
    print(f"{'Grid':<6} {'Condition':<20} {'Image Name':<70} {'Count':>8}")
    print("─" * 120)
    for img_name, data in sorted_items:
        print(f"{data['grid_id']:<6} {data['condition']:<20} {img_name:<70} {data['count']:>8}")
    print("─" * 120)
    print(f"\nPer-image summary saved → {out_path}")


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

# Consistent colours per grid
_GRID_COLOURS = {4: '#1f77b4', 5: '#ff7f0e', 6: '#2ca02c', 7: '#d62728'}
_COND_COLOURS = {
    'Type A, no PLL': '#1f77b4',  # blue
    'Type B, no PLL': '#ff7f0e',  # orange
    'Type A, PLL':    '#2ca02c',  # green
    'Type B, PLL':    '#d62728',  # red
}


def _records_by_group(records: List[Dict], group_key: str) -> Dict:
    groups: Dict = {}
    for r in records:
        k = r[group_key]
        groups.setdefault(k, []).append(r)
    return groups


def _extract(records: List[Dict], key: str) -> np.ndarray:
    return np.array([r[key] for r in records if r.get(key) is not None], dtype=float)


def plot_per_grid_distributions(records: List[Dict], out_dir: Path) -> None:
    """Four-panel distribution plot, one series per grid."""
    by_grid = _records_by_group(records, 'grid_id')
    metrics = [
        ('equivalent_diameter', 'Equivalent Diameter (px)', (0, None)),
        ('circularity',         'Circularity',               (0, 1)),
        ('major_axis',          'Major Axis (px)',            (0, None)),
        ('aspect_ratio',        'Aspect Ratio',               (1, None)),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle('Per-grid morphology distributions', fontsize=14)

    for ax, (key, label, xlim) in zip(axes.flat, metrics):
        for gid, grecs in sorted(by_grid.items()):
            vals = _extract(grecs, key)
            if len(vals) == 0:
                continue
            ax.hist(vals, bins=30, alpha=0.5, label=f'Grid {gid}',
                    color=_GRID_COLOURS.get(gid), edgecolor='none')
        ax.set_xlabel(label)
        ax.set_ylabel('Count')
        ax.set_title(label)
        if xlim[0] is not None:
            ax.set_xlim(left=xlim[0])
        if xlim[1] is not None:
            ax.set_xlim(right=xlim[1])
        ax.legend(fontsize=8)

    fig.tight_layout()
    path = out_dir / 'per_grid_distributions.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved → {path}")


def plot_condition_distributions(records: List[Dict], out_dir: Path) -> None:
    """Per-condition (A vs B) overlaid histograms."""
    by_cond = _records_by_group(records, 'condition')
    metrics = [
        ('equivalent_diameter', 'Equivalent Diameter (px)'),
        ('circularity',         'Circularity'),
        ('major_axis',          'Major Axis (px)'),
        ('aspect_ratio',        'Aspect Ratio'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle('Condition A vs B — morphology distributions', fontsize=14)

    for ax, (key, label) in zip(axes.flat, metrics):
        for cond, crecs in sorted(by_cond.items()):
            vals = _extract(crecs, key)
            if len(vals) == 0:
                continue
            ax.hist(vals, bins=30, alpha=0.6, label=cond,
                    color=_COND_COLOURS.get(cond), edgecolor='none',
                    density=True)
        ax.set_xlabel(label)
        ax.set_ylabel('Density')
        ax.set_title(label)
        ax.legend(fontsize=8)

    fig.tight_layout()
    path = out_dir / 'condition_AvsB_distributions.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved → {path}")


def plot_count_summary(records: List[Dict], grid_images: Dict[int, List[Path]],
                       out_dir: Path) -> None:
    """
    Bar charts of:
      1. Total detections per grid
      2. Mean detections per image per grid
    Grids are grouped by condition with colour coding.
    """
    by_grid = _records_by_group(records, 'grid_id')
    grid_ids = sorted(grid_images.keys())

    total_counts = [len(by_grid.get(g, [])) for g in grid_ids]
    n_images     = [len(grid_images[g]) for g in grid_ids]
    mean_counts  = [t / n if n > 0 else 0
                    for t, n in zip(total_counts, n_images)]
    colours      = [_GRID_COLOURS.get(g, 'grey') for g in grid_ids]
    labels       = [f"Grid {g}\n({GRID_CONDITIONS.get(g, '?')})" for g in grid_ids]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle('EV counts per grid', fontsize=14)

    bars1 = ax1.bar(labels, total_counts, color=colours)
    ax1.set_ylabel('Total detections')
    ax1.set_title('Total detections')
    for bar, val in zip(bars1, total_counts):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 str(val), ha='center', va='bottom', fontsize=9)

    bars2 = ax2.bar(labels, mean_counts, color=colours)
    ax2.set_ylabel('Mean detections / image')
    ax2.set_title('Mean detections per image')
    for bar, val in zip(bars2, mean_counts):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f'{val:.1f}', ha='center', va='bottom', fontsize=9)

    fig.tight_layout()
    path = out_dir / 'detection_counts.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved → {path}")


def plot_condition_boxplots(records: List[Dict], out_dir: Path) -> None:
    """Box plots comparing Condition A vs B for key metrics."""
    by_cond = _records_by_group(records, 'condition')
    cond_labels = sorted(by_cond.keys())
    metrics = [
        ('equivalent_diameter', 'Equivalent Diameter (px)'),
        ('circularity',         'Circularity'),
        ('major_axis',          'Major Axis (px)'),
        ('aspect_ratio',        'Aspect Ratio'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    fig.suptitle('Condition A vs B — boxplots', fontsize=14)

    for ax, (key, label) in zip(axes.flat, metrics):
        data = [_extract(by_cond[c], key) for c in cond_labels]
        bp = ax.boxplot(data, tick_labels=cond_labels, patch_artist=True,
                        medianprops=dict(color='black', linewidth=2))
        for patch, cond in zip(bp['boxes'], cond_labels):
            patch.set_facecolor(_COND_COLOURS.get(cond, 'grey'))
            patch.set_alpha(0.6)
        ax.set_ylabel(label)
        ax.set_title(label)

    fig.tight_layout()
    path = out_dir / 'condition_AvsB_boxplots.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved → {path}")


def plot_per_grid_boxplots(records: List[Dict], out_dir: Path) -> None:
    """Box plots by individual grid (4 grids side-by-side) for key metrics."""
    by_grid = _records_by_group(records, 'grid_id')
    grid_ids = sorted(by_grid.keys())
    metrics = [
        ('equivalent_diameter', 'Equivalent Diameter (px)'),
        ('circularity',         'Circularity'),
        ('major_axis',          'Major Axis (px)'),
        ('aspect_ratio',        'Aspect Ratio'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle('Per-grid boxplots', fontsize=14)

    for ax, (key, label) in zip(axes.flat, metrics):
        data   = [_extract(by_grid[g], key) for g in grid_ids]
        labels = [f'Grid {g}' for g in grid_ids]
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True,
                        medianprops=dict(color='black', linewidth=2))
        for patch, gid in zip(bp['boxes'], grid_ids):
            patch.set_facecolor(_GRID_COLOURS.get(gid, 'grey'))
            patch.set_alpha(0.6)
        ax.set_ylabel(label)
        ax.set_title(label)

    fig.tight_layout()
    path = out_dir / 'per_grid_boxplots.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("cryoEV cross-grid inference & comparison")
    print("=" * 60)
    print(f"Images  : {IMAGES_DIR}")
    print(f"Weights : {MODEL_WEIGHTS}")
    print(f"Output  : {OUTPUT_DIR}")
    print()

    # Validate paths
    if not IMAGES_DIR.exists():
        sys.exit(f"ERROR: Images directory not found:\n  {IMAGES_DIR}")
    if not MODEL_WEIGHTS.exists():
        sys.exit(f"ERROR: Model weights not found:\n  {MODEL_WEIGHTS}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Collect images
    grid_images = collect_images()
    if not grid_images:
        sys.exit("ERROR: No images found — check IMAGES_DIR and filename convention.")

    print("Images found per grid:")
    for gid in sorted(grid_images):
        cond = GRID_CONDITIONS.get(gid, '?')
        print(f"  Grid {gid} ({cond}):  {len(grid_images[gid])} images")

    # 2. Inference
    n_images_total = sum(len(v) for v in grid_images.values())
    run_id = datetime.now().strftime('cross_grid_%Y%m%d_%H%M%S')
    perf = PerformanceLogger(output_dir=OUTPUT_DIR, session_id=run_id)
    perf.log_session_start(
        model_path=MODEL_WEIGHTS,
        n_images=n_images_total,
        device=DEVICE,
        imgsz=IMGSZ,
        conf=CONF,
        iou=IOU,
    )

    session_t0 = time.perf_counter()
    overlay_root = OUTPUT_DIR / 'overlays' if SAVE_OVERLAYS else None
    all_records, n_processed = run_inference_all_grids(grid_images, perf, overlay_root=overlay_root)
    perf.log_session_end(
        total_time_s=time.perf_counter() - session_t0,
        n_processed=n_processed,
    )

    if not all_records:
        sys.exit("No detections returned — check model weights and images.")

    print(f"\nTotal detections: {len(all_records)}")

    # 3. Save combined CSV
    save_combined_csv(all_records, OUTPUT_DIR / 'all_detections.csv')

    # 4. Save per-image summary
    save_per_image_summary(all_records, OUTPUT_DIR / 'detections_per_image.csv')

    # 5. Plots
    print("\nGenerating plots...")
    plot_per_grid_distributions(all_records, OUTPUT_DIR)
    plot_condition_distributions(all_records, OUTPUT_DIR)
    plot_count_summary(all_records, grid_images, OUTPUT_DIR)
    plot_condition_boxplots(all_records, OUTPUT_DIR)
    plot_per_grid_boxplots(all_records, OUTPUT_DIR)

    print("\nDone.  All outputs saved to:")
    print(f"  {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
