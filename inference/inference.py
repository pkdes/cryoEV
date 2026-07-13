"""
Inference and Visualization Script for Cryo-EM Instance Segmentation

Features:
- Run YOLO instance segmentation inference on test images
- Extract per-object confidence scores
- Interactive review of low-confidence detections
- Morphology analysis (ellipse fitting, diameter, circularity)
- Save prediction masks, review decisions, and morphology data
"""

import os
import sys
import csv
import time
from datetime import datetime

# Ensure project root is on the path so sibling-package imports work
# regardless of the working directory.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from inference.perf_log import PerformanceLogger

from analysis.morphology import (
    analyze_instances, save_morphology_csv,
    draw_ellipses_on_image, draw_polygons_on_image, plot_morphology_distributions,
)


# ============================================================================
# Instance Extraction
# ============================================================================

def extract_instances_yolo(
    model_path: str,
    img_path: str,
    imgsz: int = 1024,
    conf: float = 0.25,
    iou: float = 0.7,
    device: str = 'cpu'
) -> Tuple[List[np.ndarray], List[float], List[Dict], List[Optional[np.ndarray]]]:
    """
    Extract individual object instances from a YOLO segmentation model.

    Args:
        model_path: Path to YOLO .pt weights
        img_path: Path to input image
        imgsz: Inference image size
        conf: Confidence threshold for YOLO
        iou: IoU threshold for NMS
        device: Device string ('cpu' or 'cuda')

    Returns:
        instance_masks: List of boolean masks
        confidences: List of confidence scores
        bbox_info: List of dicts with bbox coordinates
        polygon_points: List of Nx2 polygon arrays in image coordinates
    """
    from ultralytics import YOLO

    model = YOLO(model_path)
    result = model.predict(source=str(img_path), imgsz=imgsz, conf=conf, iou=iou,
                           device=device, verbose=False, retina_masks=True)[0]

    instance_masks: List[np.ndarray] = []
    confidences: List[float] = []
    bbox_info: List[Dict] = []
    polygon_points: List[Optional[np.ndarray]] = []

    if result.masks is not None and result.boxes is not None:
        if hasattr(result.masks, 'xy') and result.masks.xy is not None:
            polygons_xy = result.masks.xy
        else:
            polygons_xy = [None] * len(result.boxes)

        for raw_mask, box, poly in zip(result.masks.data, result.boxes, polygons_xy):
            # Normalise to a clean 2-D boolean mask regardless of what
            # the predictor returns (could be (H,W), (1,H,W), (H,W,1), uint8, etc.)
            raw_mask = raw_mask.detach().cpu().numpy() if hasattr(raw_mask, 'cpu') else raw_mask
            mask = np.asarray(raw_mask).squeeze()
            if mask.ndim != 2:
                continue
            mask = mask.astype(bool)
            coords = np.argwhere(mask)
            if len(coords) == 0:
                continue

            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)

            poly_points = None
            if poly is not None:
                p = np.asarray(poly, dtype=np.float32)
                if p.ndim == 2 and p.shape[1] == 2 and len(p) >= 5:
                    poly_points = p

            instance_masks.append(mask)
            confidences.append(float(box.conf.cpu().numpy()[0]))
            polygon_points.append(poly_points)
            bbox_info.append({
                'x_min': int(x_min), 'y_min': int(y_min),
                'x_max': int(x_max), 'y_max': int(y_max),
                'area': int(mask.sum()),
            })

    return instance_masks, confidences, bbox_info, polygon_points


# ============================================================================
# Interactive Review
# ============================================================================

def interactive_review_objects(
    image: np.ndarray,
    instance_masks: List[np.ndarray],
    confidences: List[float],
    threshold: float = 0.5,
    skip_review: bool = False
) -> Tuple[List[np.ndarray], List[float], List[Dict], bool]:
    """
    Review detected objects by confidence threshold.

    Objects with confidence >= threshold are automatically accepted.
    Objects below threshold are either rejected automatically
    (skip_review=True) or shown in a matplotlib popup for manual
    accept/reject (skip_review=False).

    Args:
        image: Original grayscale image (H, W)
        instance_masks: List of boolean masks per object
        confidences: List of confidence scores per object
        threshold: Confidence threshold for acceptance
        skip_review: If True, reject all below-threshold objects
            without showing popups.

    Returns:
        filtered_masks: Masks of accepted objects
        filtered_confs: Confidences of accepted objects
        decisions: List of dicts recording each decision
        exited: True if the user clicked Exit during review
    """
    filtered_masks: List[np.ndarray] = []
    filtered_confs: List[float] = []
    decisions: List[Dict] = []

    for idx, (mask, conf) in enumerate(zip(instance_masks, confidences)):
        coords = np.argwhere(mask)
        if len(coords) == 0:
            continue
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        if conf >= threshold:
            decision = 'accepted'
            filtered_masks.append(mask)
            filtered_confs.append(conf)
        elif skip_review:
            decision = 'rejected'
        else:
            decision = _show_review_popup(
                image, instance_masks, confidences, idx, threshold
            )
            if decision == 'exit':
                return filtered_masks, filtered_confs, decisions, True
            if decision == 'accepted':
                filtered_masks.append(mask)
                filtered_confs.append(conf)

        decisions.append({
            'object_index': idx,
            'confidence': conf,
            'decision': decision,
            'bbox_x_min': int(x_min),
            'bbox_y_min': int(y_min),
            'bbox_x_max': int(x_max),
            'bbox_y_max': int(y_max),
        })

    return filtered_masks, filtered_confs, decisions, False


def _show_review_popup(
    image: np.ndarray,
    instance_masks: List[np.ndarray],
    confidences: List[float],
    current_idx: int,
    threshold: float
) -> str:
    """Show a matplotlib popup for reviewing a single object.

    Returns 'accepted', 'rejected', or 'exit'.
    """
    result = {'decision': 'rejected'}

    fig, (ax_orig, ax_overlay) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"Object {current_idx} | Confidence: {confidences[current_idx]:.3f} "
        f"(threshold: {threshold:.3f})",
        fontsize=13, fontweight='bold'
    )

    # --- Left panel: original image (clean, no annotations) ---
    if len(image.shape) == 2:
        display_img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        display_img = image.copy()

    ax_orig.imshow(display_img)
    ax_orig.set_title("Original")
    ax_orig.axis('off')

    # --- Right panel: overlay with all masks in green, current object circled in red ---
    current_mask = instance_masks[current_idx]
    coords = np.argwhere(current_mask)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    cx = int((x_min + x_max) / 2)
    cy = int((y_min + y_max) / 2)
    radius = int(max(x_max - x_min, y_max - y_min) / 2 * 1.3)

    overlay = display_img.copy()
    for i, (mask, conf) in enumerate(zip(instance_masks, confidences)):
        if conf >= threshold:
            overlay[mask] = (0, 200, 0)    # green = above threshold
        else:
            overlay[mask] = (255, 255, 0)  # yellow = below threshold
    blended = cv2.addWeighted(display_img, 0.6, overlay, 0.4, 0)
    cv2.circle(blended, (cx, cy), radius, (255, 0, 0), 3)

    ax_overlay.imshow(blended)
    ax_overlay.set_title("Overlay (red circle = under review)")
    ax_overlay.axis('off')

    # --- Buttons ---
    ax_accept = fig.add_axes([0.25, 0.02, 0.15, 0.06])
    ax_reject = fig.add_axes([0.43, 0.02, 0.15, 0.06])
    ax_exit = fig.add_axes([0.61, 0.02, 0.15, 0.06])
    btn_accept = Button(ax_accept, 'Accept', color='lightgreen', hovercolor='green')
    btn_reject = Button(ax_reject, 'Reject', color='lightsalmon', hovercolor='red')
    btn_exit = Button(ax_exit, 'Exit', color='lightgray', hovercolor='gray')

    def on_accept(event):
        result['decision'] = 'accepted'
        plt.close(fig)

    def on_reject(event):
        result['decision'] = 'rejected'
        plt.close(fig)

    def on_exit(event):
        result['decision'] = 'exit'
        plt.close(fig)

    btn_accept.on_clicked(on_accept)
    btn_reject.on_clicked(on_reject)
    btn_exit.on_clicked(on_exit)

    plt.show()
    return result['decision']


# ============================================================================
# Saving Decisions
# ============================================================================

def save_review_decisions(
    decisions: List[Dict],
    output_path: str
):
    """
    Save review decisions to a CSV file.

    Columns: object_index, confidence, decision,
             bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max
    """
    fieldnames = [
        'object_index', 'confidence', 'decision',
        'bbox_x_min', 'bbox_y_min', 'bbox_x_max', 'bbox_y_max',
    ]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(decisions)
    print(f"Saved review decisions to {output_path}")


# ============================================================================
# High-Level Orchestrator
# ============================================================================

def predict_with_review(
    image_path: str,
    output_dir: str,
    # YOLO params
    yolo_model_path: str,
    imgsz: int = 1024,
    yolo_conf: float = 0.25,
    yolo_iou: float = 0.7,
    device: str = 'cpu',
    # Review params
    confidence_threshold: float = 0.5,
    skip_review: bool = True,
    # Morphology params
    pixel_size: Optional[float] = None,
) -> Tuple[List[np.ndarray], List[float], List[Dict], List[Dict], bool]:
    """
    End-to-end: predict -> extract instances -> review -> morphology -> save.

    Args:
        image_path: Path to input image
        output_dir: Directory to save outputs (masks, decisions CSV)
        yolo_model_path: Path to YOLO .pt weights
        imgsz: Inference image size
        yolo_conf: YOLO confidence threshold
        yolo_iou: YOLO NMS IoU threshold
        device: Device string ('cpu' or 'cuda')
        confidence_threshold: Objects below this are rejected (skip_review=True)
            or shown for manual review (skip_review=False)
        skip_review: If True, auto-reject below-threshold objects without popups
        pixel_size: Physical size per pixel (e.g., nm/px) for morphology.
            If None, measurements are in pixels.

    Returns:
        (accepted_masks, accepted_confidences, decisions, morph_records, exited)
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    image = image.squeeze()  # ensure (H, W), not (H, W, 1)

    # --- Extract instances (timed) ---
    _t0 = time.perf_counter()
    instance_masks, confidences, bbox_info, instance_polygons = extract_instances_yolo(
        yolo_model_path, image_path, imgsz, yolo_conf, yolo_iou, device
    )
    _inference_time_s = time.perf_counter() - _t0

    # --- Perf record (written to output_dir/performance_log.csv) ---
    _session_id = f"single_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    _perf = PerformanceLogger(output_dir=output_dir, session_id=_session_id)
    _perf.log_session_start(
        model_path=yolo_model_path, n_images=1,
        device=device, imgsz=imgsz, conf=yolo_conf, iou=yolo_iou,
    )
    _perf.log_image(
        image_path=image_path, n_raw_detections=len(instance_masks),
        inference_time_s=_inference_time_s, image=image,
    )

    print(f"Detected {len(instance_masks)} objects")

    # --- Review ---
    filtered_masks, filtered_confs, decisions, exited = interactive_review_objects(
        image, instance_masks, confidences,
        threshold=confidence_threshold, skip_review=skip_review
    )

    if exited:
        print("User exited review. Stopping.")
        return filtered_masks, filtered_confs, decisions, [], True

    n_accepted = sum(1 for d in decisions if d['decision'] == 'accepted')
    n_rejected = sum(1 for d in decisions if d['decision'] == 'rejected')
    print(f"Accepted: {n_accepted}, Rejected: {n_rejected}")

    accepted_indices = [d['object_index'] for d in decisions if d['decision'] == 'accepted']
    filtered_polygons = [
        instance_polygons[i] if i < len(instance_polygons) else None
        for i in accepted_indices
    ]

    # --- Morphology analysis ---
    morph_records = analyze_instances(
        filtered_masks,
        confidences=filtered_confs,
        pixel_size=pixel_size,
        polygon_points_list=filtered_polygons,
    )

    # --- Save outputs ---
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    base_name = Path(image_path).stem

    # Save accepted binary mask
    combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for m in filtered_masks:
        combined_mask[m] = 255
    cv2.imwrite(str(output_path / f"{base_name}_mask.png"), combined_mask)

    # Save decisions CSV
    save_review_decisions(decisions, str(output_path / f"{base_name}_decisions.csv"))

    # Save morphology CSV
    save_morphology_csv(morph_records, str(output_path / f"{base_name}_morphology.csv"))

    # Save ellipse overlay
    ellipse_vis = draw_ellipses_on_image(image, filtered_masks, morph_records)
    cv2.imwrite(
        str(output_path / f"{base_name}_ellipses.png"),
        cv2.cvtColor(ellipse_vis, cv2.COLOR_RGB2BGR)
    )

    # Save polygon (true boundary) + ellipse overlay, for comparing fit quality
    overlay_vis = draw_polygons_on_image(image, filtered_polygons, morph_records)
    cv2.imwrite(
        str(output_path / f"{base_name}_overlay.png"),
        cv2.cvtColor(overlay_vis, cv2.COLOR_RGB2BGR)
    )

    # Save morphology distribution plot
    if morph_records:
        plot_morphology_distributions(
            morph_records, pixel_size=pixel_size,
            save_path=str(output_path / f"{base_name}_morphology_distributions.png")
        )

    _perf.log_session_end(total_time_s=time.perf_counter() - _t0, n_processed=1)

    return filtered_masks, filtered_confs, decisions, morph_records, False


# ============================================================================
# Visualization Helpers
# ============================================================================

def create_overlay_image(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.5
) -> np.ndarray:
    """Create overlay of mask on image. Returns RGB overlay image (H, W, 3)."""
    if len(image.shape) == 2:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image_rgb = image.copy()

    colored_mask = np.zeros_like(image_rgb)
    colored_mask[mask > 0] = color

    overlay = cv2.addWeighted(image_rgb, 1.0, colored_mask, alpha, 0)
    return overlay


# ============================================================================
# Main Script
# ============================================================================

if __name__ == '__main__':
    """
    YOLO inference with morphology analysis on test images.
    """
    import glob as glob_module
    import torch

    CONFIG = {
        'yolo_model_path': r'C:\Users\Yifei\Documents\cryo\revised_annotation\results_yolov8_heavy_augmentation\training\vesicle_instance_seg_v2\weights\best.pt',
        'yolo_conf': 0.25,
        'yolo_iou': 0.7,

        'test_data_dir': r'C:\Users\Yifei\Documents\cryo\revised_annotation\split_cleaned\test',
        'output_dir': r'C:\Users\Yifei\Documents\cryo\revised_annotation\results_yolov8_heavy_augmentation',
        'image_size': 1024,
        'confidence_threshold': 0.5,
        'skip_review': False,  # Set to False to interactively review below-threshold detections
        'pixel_size': None,  # Set to e.g. 3.5 for nm/px conversion
    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")

    # Gather test images
    img_dir = os.path.join(CONFIG['test_data_dir'], 'images')
    image_paths = sorted(
        glob_module.glob(os.path.join(img_dir, '*.jpg'))
        + glob_module.glob(os.path.join(img_dir, '*.png'))
        + glob_module.glob(os.path.join(img_dir, '*.tif'))
    )
    print(f"Found {len(image_paths)} test images in {img_dir}\n")

    all_morph_records = []

    _batch_session_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    _batch_perf = PerformanceLogger(
        output_dir=CONFIG['output_dir'],
        session_id=_batch_session_id,
        log_name="performance_log.csv",
    )
    _batch_perf.log_session_start(
        model_path=CONFIG['yolo_model_path'],
        n_images=len(image_paths),
        device=device,
        imgsz=CONFIG['image_size'],
        conf=CONFIG['yolo_conf'],
        iou=CONFIG['yolo_iou'],
    )
    _batch_t0 = time.perf_counter()
    _batch_processed = 0

    for img_path in image_paths:
        print(f"Processing: {os.path.basename(img_path)}")
        _img_t0 = time.perf_counter()
        _, _, _, morph_records, exited = predict_with_review(
            image_path=img_path,
            output_dir=CONFIG['output_dir'],
            yolo_model_path=CONFIG['yolo_model_path'],
            imgsz=CONFIG['image_size'],
            yolo_conf=CONFIG['yolo_conf'],
            yolo_iou=CONFIG['yolo_iou'],
            device=device,
            confidence_threshold=CONFIG['confidence_threshold'],
            skip_review=CONFIG['skip_review'],
            pixel_size=CONFIG['pixel_size'],
        )
        # Tag each record with its source image
        for rec in morph_records:
            rec['image'] = os.path.basename(img_path)
        all_morph_records.extend(morph_records)
        _img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        _batch_perf.log_image(
            image_path=img_path,
            n_raw_detections=sum(1 for d in _ if True) if _ is not None else 0,
            inference_time_s=time.perf_counter() - _img_t0,
            image=_img,
        )
        _batch_processed += 1
        print()
        if exited:
            break

    _batch_perf.log_session_end(
        total_time_s=time.perf_counter() - _batch_t0,
        n_processed=_batch_processed,
    )

    # Save combined morphology CSV for all images
    if all_morph_records:
        combined_csv = os.path.join(CONFIG['output_dir'], 'morphology_all.csv')
        save_morphology_csv(all_morph_records, combined_csv)
        plot_morphology_distributions(
            all_morph_records, pixel_size=CONFIG['pixel_size'],
            save_path=os.path.join(CONFIG['output_dir'], 'morphology_distributions_all.png')
        )

    print(f"\nInference complete! {len(all_morph_records)} vesicles analyzed across {len(image_paths)} images.")
