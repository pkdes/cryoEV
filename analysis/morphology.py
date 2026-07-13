"""
Morphology Analysis for Segmented Extracellular Vesicles

Fits ellipses to segmented instance masks and computes size/shape
descriptors: equivalent diameter, major/minor axes, aspect ratio,
circularity, and solidity.

Designed to plug directly into the instance extraction outputs from
extract_instances_unet() or extract_instances_yolo().
"""

import csv
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ============================================================================
# Ellipse Fitting
# ============================================================================

def fit_ellipse(mask: np.ndarray) -> Optional[Dict]:
    """
    Fit an ellipse to a single binary instance mask using OpenCV.

    Args:
        mask: Boolean or binary mask (H, W) for one object.

    Returns:
        Dict with ellipse parameters, or None if fitting fails
        (e.g., fewer than 5 contour points).
    """
    mask_uint8 = (mask.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        return None

    # Use the largest contour (handles any small noise fragments)
    contour = max(contours, key=cv2.contourArea)

    # cv2.fitEllipse requires at least 5 points
    if len(contour) < 5:
        return None

    (cx, cy), (axis_minor, axis_major), angle = cv2.fitEllipse(contour)

    # OpenCV returns (width, height) which may not be ordered;
    # ensure major >= minor
    if axis_minor > axis_major:
        axis_minor, axis_major = axis_major, axis_minor
        angle = (angle + 90) % 180

    return {
        'center_x': cx,
        'center_y': cy,
        'major_axis': axis_major,   # length in pixels (diameter, not radius)
        'minor_axis': axis_minor,
        'angle': angle,             # degrees, orientation of major axis
    }


def fit_ellipse_from_polygon(polygon_points: np.ndarray) -> Optional[Dict]:
    """
    Fit an ellipse directly from polygon vertices.

    This avoids rasterization artifacts that can appear when fitting from
    thresholded mask contours.
    """
    if polygon_points is None:
        return None

    pts = np.asarray(polygon_points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 5:
        return None

    contour = pts.reshape(-1, 1, 2)
    (cx, cy), (axis_minor, axis_major), angle = cv2.fitEllipse(contour)

    if axis_minor > axis_major:
        axis_minor, axis_major = axis_major, axis_minor
        angle = (angle + 90) % 180

    return {
        'center_x': cx,
        'center_y': cy,
        'major_axis': axis_major,
        'minor_axis': axis_minor,
        'angle': angle,
    }


# ============================================================================
# Morphology Metrics
# ============================================================================

def compute_morphology(
    mask: np.ndarray,
    pixel_size: Optional[float] = None,
    polygon_points: Optional[np.ndarray] = None,
) -> Optional[Dict]:
    """
    Compute morphology descriptors for a single instance mask.

    Metrics:
        area:               Mask area in pixels (or physical units if pixel_size given)
        equivalent_diameter: Diameter of a circle with the same area
        major_axis:         Major axis of the fitted ellipse
        minor_axis:         Minor axis of the fitted ellipse
        aspect_ratio:       major / minor (1.0 = perfect circle)
        circularity:        4*pi*area / perimeter^2 (1.0 = perfect circle)
        solidity:           area / convex_hull_area
        ellipse_center_x/y: Center of the fitted ellipse
        ellipse_angle:      Orientation of the major axis (degrees)

    Args:
        mask: Boolean/binary mask (H, W) for one object.
        pixel_size: Physical size per pixel (e.g., nm/px).  If provided,
            length quantities are reported in physical units and area in
            units^2.  If None, everything is in pixels.

    Returns:
        Dict of morphology descriptors, or None if the mask is too small.
    """
    mask_uint8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)

    area_px = float(mask.sum())
    perimeter_px = cv2.arcLength(contour, closed=True)

    if area_px == 0 or perimeter_px == 0:
        return None

    # Equivalent diameter (diameter of circle with same area)
    eq_diameter_px = np.sqrt(4 * area_px / np.pi)

    # Circularity
    circularity = (4 * np.pi * area_px) / (perimeter_px ** 2)
    circularity = min(circularity, 1.0)  # clip numerical overshoot

    # Solidity
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area_px / hull_area if hull_area > 0 else 0.0

    # Prefer fitting from polygon vertices when available; fallback to mask contour.
    ellipse = fit_ellipse_from_polygon(polygon_points)
    if ellipse is None:
        ellipse = fit_ellipse(mask)

    scale = pixel_size if pixel_size is not None else 1.0
    unit = '' if pixel_size is None else '_phys'

    result = {
        'area': area_px * (scale ** 2),
        'perimeter': perimeter_px * scale,
        'equivalent_diameter': eq_diameter_px * scale,
        'circularity': circularity,
        'solidity': solidity,
    }

    if ellipse is not None:
        result.update({
            'major_axis': ellipse['major_axis'] * scale,
            'minor_axis': ellipse['minor_axis'] * scale,
            'aspect_ratio': ellipse['major_axis'] / ellipse['minor_axis']
                            if ellipse['minor_axis'] > 0 else float('inf'),
            'ellipse_center_x': ellipse['center_x'],
            'ellipse_center_y': ellipse['center_y'],
            'ellipse_angle': ellipse['angle'],
        })
    else:
        result.update({
            'major_axis': None,
            'minor_axis': None,
            'aspect_ratio': None,
            'ellipse_center_x': None,
            'ellipse_center_y': None,
            'ellipse_angle': None,
        })

    return result


# ============================================================================
# Batch Analysis
# ============================================================================

def analyze_instances(
    instance_masks: List[np.ndarray],
    confidences: Optional[List[float]] = None,
    pixel_size: Optional[float] = None,
    polygon_points_list: Optional[List[Optional[np.ndarray]]] = None,
) -> List[Dict]:
    """
    Compute morphology descriptors for a list of instance masks.

    Args:
        instance_masks: List of boolean masks, one per object
            (as returned by extract_instances_unet / extract_instances_yolo).
        confidences: Optional list of confidence scores per object.
        pixel_size: Physical size per pixel (e.g., nm/px).

    Returns:
        List of dicts, one per object, with all morphology metrics
        plus an 'object_id' field (0-indexed).
    """
    results: List[Dict] = []

    for idx, mask in enumerate(instance_masks):
        polygon_points = None
        if polygon_points_list is not None and idx < len(polygon_points_list):
            polygon_points = polygon_points_list[idx]

        morph = compute_morphology(mask, pixel_size=pixel_size, polygon_points=polygon_points)
        if morph is None:
            continue

        record = {'object_id': idx}
        if confidences is not None:
            record['confidence'] = confidences[idx]
        record.update(morph)
        results.append(record)

    return results


# ============================================================================
# I/O
# ============================================================================

def save_morphology_csv(
    records: List[Dict],
    output_path: str,
):
    """Save morphology records to CSV."""
    if not records:
        print("No records to save.")
        return

    fieldnames = list(records[0].keys())
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"Saved morphology data ({len(records)} objects) to {output_path}")


# ============================================================================
# Visualization
# ============================================================================

def draw_ellipses_on_image(
    image: np.ndarray,
    instance_masks: List[np.ndarray],
    records: List[Dict],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw fitted ellipses on the image.

    Args:
        image: Grayscale or RGB image.
        instance_masks: List of boolean masks (used only for mask overlay).
        records: Morphology records from analyze_instances().
        color: Ellipse outline color (BGR).
        thickness: Line thickness.

    Returns:
        RGB image with ellipses drawn.
    """
    if len(image.shape) == 2:
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        vis = image.copy()

    for rec in records:
        cx = rec.get('ellipse_center_x')
        cy = rec.get('ellipse_center_y')
        major = rec.get('major_axis')
        minor = rec.get('minor_axis')
        angle = rec.get('ellipse_angle')

        if any(v is None for v in [cx, cy, major, minor, angle]):
            continue

        center = (int(round(cx)), int(round(cy)))
        axes = (int(round(major / 2)), int(round(minor / 2)))
        cv2.ellipse(vis, center, axes, angle, 0, 360, color, thickness)

    return vis


def draw_polygons_on_image(
    image: np.ndarray,
    polygon_points_list: List[Optional[np.ndarray]],
    records: Optional[List[Dict]] = None,
    polygon_color: Tuple[int, int, int] = (0, 255, 0),
    ellipse_color: Tuple[int, int, int] = (255, 0, 0),
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw raw segmentation polygon outlines (the true detected boundary) on
    the image. If `records` is given, also draws each object's fitted
    ellipse in a second color so the two can be compared directly.

    Args:
        image: Grayscale or RGB image.
        polygon_points_list: List of Nx2 polygon vertex arrays (one per
            object), as returned by extract_instances_yolo(). Entries may
            be None (skipped).
        records: Optional morphology records from analyze_instances(), used
            to overlay the fitted ellipse for comparison.
        polygon_color: Polygon outline color (RGB).
        ellipse_color: Ellipse outline color (RGB).
        thickness: Line thickness.

    Returns:
        RGB image with polygon outlines (and optionally ellipses) drawn.
    """
    if len(image.shape) == 2:
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        vis = image.copy()

    if records:
        vis = draw_ellipses_on_image(vis, [], records, color=ellipse_color, thickness=thickness)

    for poly in polygon_points_list:
        if poly is None:
            continue
        pts = np.asarray(poly, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3:
            continue
        cv2.polylines(vis, [pts.astype(np.int32)], isClosed=True, color=polygon_color, thickness=thickness)

    return vis


def plot_morphology_distributions(
    records: List[Dict],
    pixel_size: Optional[float] = None,
    save_path: Optional[str] = None,
):
    """
    Plot histograms of key morphology descriptors.

    Generates a 2x2 figure:
        - Equivalent diameter
        - Major / Minor axis
        - Circularity
        - Aspect ratio
    """
    if not records:
        print("No records to plot.")
        return

    unit = 'px' if pixel_size is None else 'nm'

    diameters = [r['equivalent_diameter'] for r in records if r['equivalent_diameter'] is not None]
    majors = [r['major_axis'] for r in records if r['major_axis'] is not None]
    minors = [r['minor_axis'] for r in records if r['minor_axis'] is not None]
    circularities = [r['circularity'] for r in records if r['circularity'] is not None]
    aspect_ratios = [r['aspect_ratio'] for r in records if r['aspect_ratio'] is not None]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    axes[0, 0].hist(diameters, bins=30, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel(f'Equivalent Diameter ({unit})')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Size Distribution')

    axes[0, 1].hist(majors, bins=30, alpha=0.6, label='Major axis', edgecolor='black')
    axes[0, 1].hist(minors, bins=30, alpha=0.6, label='Minor axis', edgecolor='black')
    axes[0, 1].set_xlabel(f'Axis Length ({unit})')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Ellipse Axes')
    axes[0, 1].legend()

    axes[1, 0].hist(circularities, bins=30, edgecolor='black', alpha=0.7, range=(0, 1))
    axes[1, 0].set_xlabel('Circularity')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Circularity (1.0 = perfect circle)')

    axes[1, 1].hist(aspect_ratios, bins=30, edgecolor='black', alpha=0.7)
    axes[1, 1].set_xlabel('Aspect Ratio (major / minor)')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Aspect Ratio (1.0 = circular)')

    n = len(records)
    fig.suptitle(f'Vesicle Morphology (n = {n})', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved morphology plot to {save_path}")
    else:
        plt.show()

    plt.close()
