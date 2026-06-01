"""Compare ellipse fitting methods against segmentation masks.

Methods:
- OpenCV least-squares ellipse fit (cv2.fitEllipse)
- Hough transform ellipse fit (skimage.transform.hough_ellipse)

Outputs:
- Per-image overlay PNGs with both fits overlaid on the raw image
- Per-object metrics CSV (IoU and Dice against segmentation mask)
- Summary CSV with method-level averages
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from skimage.feature import canny
from skimage.transform import hough_ellipse


def _ellipse_from_cv2(contour_xy: np.ndarray) -> Optional[dict]:
    pts = np.asarray(contour_xy, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 5:
        return None

    (cx, cy), (axis_a, axis_b), angle = cv2.fitEllipse(pts.reshape(-1, 1, 2))
    major = max(axis_a, axis_b)
    minor = min(axis_a, axis_b)
    if axis_a < axis_b:
        angle = (angle + 90.0) % 180.0

    return {
        "cx": float(cx),
        "cy": float(cy),
        "major": float(major),
        "minor": float(minor),
        "angle": float(angle),
    }


def _ellipse_from_hough(mask: np.ndarray, contour_xy: np.ndarray) -> Optional[dict]:
    ys, xs = np.where(mask > 0)
    if len(xs) < 20:
        return None

    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    pad = 8
    x0 = max(x0 - pad, 0)
    y0 = max(y0 - pad, 0)
    x1 = min(x1 + pad, mask.shape[1] - 1)
    y1 = min(y1 + pad, mask.shape[0] - 1)

    roi = mask[y0 : y1 + 1, x0 : x1 + 1].astype(np.uint8)
    edges = canny(roi.astype(float), sigma=1.0)

    h, w = roi.shape
    min_size = max(6, int(min(h, w) * 0.15))
    max_size = max(min_size + 2, int(max(h, w) * 0.9))

    try:
        result = hough_ellipse(edges, accuracy=2, threshold=8, min_size=min_size, max_size=max_size)
    except Exception:
        return None

    if result is None or len(result) == 0:
        return None

    best = result[np.argmax(result["accumulator"])]

    yc = float(best["yc"] + y0)
    xc = float(best["xc"] + x0)
    a = float(best["a"])
    b = float(best["b"])
    orientation = float(best["orientation"])

    major = 2.0 * max(a, b)
    minor = 2.0 * min(a, b)
    angle_deg = np.degrees(orientation)
    if a < b:
        angle_deg = (angle_deg + 90.0) % 180.0

    return {
        "cx": xc,
        "cy": yc,
        "major": major,
        "minor": minor,
        "angle": float(angle_deg),
    }


def _rasterize_contour(contour_xy: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape_hw, dtype=np.uint8)
    pts = np.asarray(contour_xy, dtype=np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [pts], color=1)
    return mask


def _rasterize_ellipse(ellipse: dict, shape_hw: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape_hw, dtype=np.uint8)
    center = (int(round(ellipse["cx"])), int(round(ellipse["cy"])))
    axes = (max(1, int(round(ellipse["major"] / 2))), max(1, int(round(ellipse["minor"] / 2))))
    angle = float(ellipse["angle"])
    cv2.ellipse(mask, center, axes, angle, 0, 360, color=1, thickness=-1)
    return mask


def _iou_dice(mask_true: np.ndarray, mask_pred: np.ndarray) -> tuple[float, float]:
    m1 = mask_true.astype(bool)
    m2 = mask_pred.astype(bool)
    inter = float(np.logical_and(m1, m2).sum())
    union = float(np.logical_or(m1, m2).sum())
    s1 = float(m1.sum())
    s2 = float(m2.sum())

    iou = inter / union if union > 0 else 0.0
    dice = (2.0 * inter) / (s1 + s2) if (s1 + s2) > 0 else 0.0
    return iou, dice


def _draw_overlay(
    image_gray: np.ndarray,
    contours: list[np.ndarray],
    lsq_ellipses: list[dict] | None = None,
    hough_ellipses: list[dict] | None = None,
) -> np.ndarray:
    vis = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
    lsq_ellipses = lsq_ellipses or []
    hough_ellipses = hough_ellipses or []

    for contour in contours:
        c = np.asarray(contour, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [c], isClosed=True, color=(255, 255, 255), thickness=1)

    for ell in lsq_ellipses:
        center = (int(round(ell["cx"])), int(round(ell["cy"])))
        axes = (max(1, int(round(ell["major"] / 2))), max(1, int(round(ell["minor"] / 2))))
        cv2.ellipse(vis, center, axes, ell["angle"], 0, 360, color=(0, 255, 0), thickness=2)

    for ell in hough_ellipses:
        center = (int(round(ell["cx"])), int(round(ell["cy"])))
        axes = (max(1, int(round(ell["major"] / 2))), max(1, int(round(ell["minor"] / 2))))
        cv2.ellipse(vis, center, axes, ell["angle"], 0, 360, color=(255, 0, 255), thickness=2)

    return vis


def run_comparison(boundaries_dir: Path, images_dir: Path, output_dir: Path) -> tuple[int, int]:
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    boundary_files = sorted(boundaries_dir.glob("*_boundaries.json"))

    for bf in boundary_files:
        payload = json.loads(bf.read_text(encoding="utf-8"))
        image_name = payload.get("image")
        if not image_name:
            continue

        image_path = images_dir / image_name
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"[skip] Could not read {image_path}")
            continue

        lsq_draw: list[dict] = []
        hough_draw: list[dict] = []
        contours_draw: list[np.ndarray] = []

        for obj in payload.get("objects", []):
            obj_id = int(obj.get("object_id", -1))
            contour = np.asarray(obj.get("contour_xy", []), dtype=np.float32)
            if contour.ndim != 2 or contour.shape[0] < 5 or contour.shape[1] != 2:
                continue

            mask_true = _rasterize_contour(contour, image.shape)

            lsq = _ellipse_from_cv2(contour)
            hough = _ellipse_from_hough(mask_true, contour)

            lsq_iou = lsq_dice = np.nan
            if lsq is not None:
                lsq_mask = _rasterize_ellipse(lsq, image.shape)
                lsq_iou, lsq_dice = _iou_dice(mask_true, lsq_mask)
                lsq_draw.append(lsq)

            hough_iou = hough_dice = np.nan
            if hough is not None:
                hough_mask = _rasterize_ellipse(hough, image.shape)
                hough_iou, hough_dice = _iou_dice(mask_true, hough_mask)
                hough_draw.append(hough)

            contours_draw.append(contour)
            rows.append(
                {
                    "image": image_name,
                    "object_id": obj_id,
                    "lsq_iou": lsq_iou,
                    "lsq_dice": lsq_dice,
                    "hough_iou": hough_iou,
                    "hough_dice": hough_dice,
                    "lsq_major": lsq["major"] if lsq else np.nan,
                    "lsq_minor": lsq["minor"] if lsq else np.nan,
                    "hough_major": hough["major"] if hough else np.nan,
                    "hough_minor": hough["minor"] if hough else np.nan,
                }
            )

        base_stem = Path(image_name).stem

        overlay_lsq = _draw_overlay(image, contours_draw, lsq_ellipses=lsq_draw)
        out_overlay_lsq = output_dir / f"{base_stem}_ellipse_compare_lsq_overlay.png"
        cv2.imwrite(str(out_overlay_lsq), overlay_lsq)

        overlay_hough = _draw_overlay(image, contours_draw, hough_ellipses=hough_draw)
        out_overlay_hough = output_dir / f"{base_stem}_ellipse_compare_hough_overlay.png"
        cv2.imwrite(str(out_overlay_hough), overlay_hough)

        overlay_combined = _draw_overlay(image, contours_draw, lsq_draw, hough_draw)
        out_overlay_combined = output_dir / f"{base_stem}_ellipse_compare_overlay.png"
        cv2.imwrite(str(out_overlay_combined), overlay_combined)

    metrics_csv = output_dir / "ellipse_method_comparison.csv"
    if rows:
        fieldnames = list(rows[0].keys())
        with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

    summary_csv = output_dir / "ellipse_method_summary.csv"
    if rows:
        arr = rows
        lsq_iou = np.array([r["lsq_iou"] for r in arr], dtype=float)
        hough_iou = np.array([r["hough_iou"] for r in arr], dtype=float)
        lsq_dice = np.array([r["lsq_dice"] for r in arr], dtype=float)
        hough_dice = np.array([r["hough_dice"] for r in arr], dtype=float)

        summary = [
            {
                "metric": "objects_total",
                "lsq": int(len(arr)),
                "hough": int(np.sum(~np.isnan(hough_iou))),
            },
            {
                "metric": "mean_iou",
                "lsq": float(np.nanmean(lsq_iou)),
                "hough": float(np.nanmean(hough_iou)) if np.any(~np.isnan(hough_iou)) else np.nan,
            },
            {
                "metric": "median_iou",
                "lsq": float(np.nanmedian(lsq_iou)),
                "hough": float(np.nanmedian(hough_iou)) if np.any(~np.isnan(hough_iou)) else np.nan,
            },
            {
                "metric": "mean_dice",
                "lsq": float(np.nanmean(lsq_dice)),
                "hough": float(np.nanmean(hough_dice)) if np.any(~np.isnan(hough_dice)) else np.nan,
            },
            {
                "metric": "median_dice",
                "lsq": float(np.nanmedian(lsq_dice)),
                "hough": float(np.nanmedian(hough_dice)) if np.any(~np.isnan(hough_dice)) else np.nan,
            },
        ]

        with open(summary_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["metric", "lsq", "hough"])
            w.writeheader()
            w.writerows(summary)

    return len(boundary_files), len(rows)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compare LSQ vs Hough ellipse fitting from segmentation boundaries.")
    p.add_argument("--boundaries-dir", type=Path, required=True, help="Folder with *_boundaries.json")
    p.add_argument("--images-dir", type=Path, required=True, help="Folder with source images")
    p.add_argument("--output-dir", type=Path, required=True, help="Folder for overlays and metrics")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    n_images, n_objects = run_comparison(args.boundaries_dir, args.images_dir, args.output_dir)
    print(f"Processed {n_images} boundary files and {n_objects} objects.")
    print(f"Output dir: {args.output_dir}")


if __name__ == "__main__":
    main()
