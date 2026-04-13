#!/usr/bin/env python3
"""Enhance grayscale cryo-EM images and tile large fields-of-view to a training-like size.

Why this helps
--------------
YOLO inference rescales each input to the configured `imgsz`. Very large source images can
therefore shrink objects far more than the images used during training. This script addresses
that by:

1. applying mild contrast normalization + CLAHE, and
2. splitting large images into smaller tiles with source traceability.

Example
-------
python -m data_utils.enhance_and_tile_images \
    --input-dir "C:\\path\\to\\SKOV\\images" \
    --output-dir "C:\\path\\to\\SKOV_enhanced" \
    --reference-height 1024 \
    --reference-width 1440
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def load_source_lookup(input_dir: Path) -> Dict[str, Dict[str, str]]:
    """Load filename -> provenance mapping from a sibling `source_manifest.csv` if present."""
    manifest_path = input_dir.parent / "source_manifest.csv"
    if not manifest_path.exists():
        return {}

    lookup: Dict[str, Dict[str, str]] = {}
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = row.get("harvested_filename") or row.get("original_filename") or ""
            if key:
                lookup[key] = row
    return lookup


def normalize_contrast(image: np.ndarray, low_pct: float = 1.0, high_pct: float = 99.0) -> np.ndarray:
    """Percentile normalize grayscale image to uint8."""
    arr = np.asarray(image, dtype=np.float32)
    lo = float(np.percentile(arr, low_pct))
    hi = float(np.percentile(arr, high_pct))
    if hi <= lo:
        lo = float(arr.min())
        hi = float(arr.max()) if float(arr.max()) > float(arr.min()) else float(arr.min()) + 1.0
    scaled = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return (scaled * 255.0).astype(np.uint8)


def enhance_grayscale(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: int = 8,
    denoise_strength: float = 0.0,
) -> np.ndarray:
    """Apply mild denoising + contrast enhancement suitable for cryo-EM grayscale images."""
    img8 = normalize_contrast(image)
    if denoise_strength and denoise_strength > 0:
        img8 = cv2.fastNlMeansDenoising(img8, None, h=float(denoise_strength), templateWindowSize=7, searchWindowSize=21)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    enhanced = clahe.apply(img8)
    return enhanced


def compute_tile_edges(length: int, reference_length: int) -> np.ndarray:
    n_tiles = max(1, math.ceil(length / max(reference_length, 1)))
    return np.linspace(0, length, n_tiles + 1, dtype=int)


def process_image(
    image_path: Path,
    output_dir: Path,
    source_lookup: Dict[str, Dict[str, str]],
    reference_height: int,
    reference_width: int,
    clip_limit: float,
    tile_grid_size: int,
    denoise_strength: float,
    disable_tiling: bool,
    resize_to_reference: bool,
) -> List[Dict[str, object]]:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    enhanced = enhance_grayscale(
        image,
        clip_limit=clip_limit,
        tile_grid_size=tile_grid_size,
        denoise_strength=denoise_strength,
    )
    original_h, original_w = enhanced.shape[:2]
    if resize_to_reference:
        enhanced = cv2.resize(enhanced, (reference_width, reference_height), interpolation=cv2.INTER_AREA)
    h, w = enhanced.shape[:2]
    if disable_tiling:
        y_edges = np.asarray([0, h], dtype=int)
        x_edges = np.asarray([0, w], dtype=int)
    else:
        y_edges = compute_tile_edges(h, reference_height)
        x_edges = compute_tile_edges(w, reference_width)

    source_info = source_lookup.get(image_path.name, {})
    records: List[Dict[str, object]] = []

    for row_idx in range(len(y_edges) - 1):
        for col_idx in range(len(x_edges) - 1):
            y0, y1 = int(y_edges[row_idx]), int(y_edges[row_idx + 1])
            x0, x1 = int(x_edges[col_idx]), int(x_edges[col_idx + 1])
            tile = enhanced[y0:y1, x0:x1]
            if tile.size == 0:
                continue

            out_name = f"{image_path.stem}__r{row_idx+1:02d}_c{col_idx+1:02d}.png"
            out_path = output_dir / out_name
            cv2.imwrite(str(out_path), tile)

            records.append(
                {
                    "processed_filename": out_name,
                    "processed_path": str(out_path),
                    "source_filename": image_path.name,
                    "source_path": str(image_path),
                    "origin_source_path": source_info.get("source_path", ""),
                    "origin_original_filename": source_info.get("original_filename", image_path.name),
                    "tile_row": row_idx + 1,
                    "tile_col": col_idx + 1,
                    "y0": y0,
                    "y1": y1,
                    "x0": x0,
                    "x1": x1,
                    "tile_height": y1 - y0,
                    "tile_width": x1 - x0,
                    "original_height": original_h,
                    "original_width": original_w,
                    "contrast_method": "percentile_norm_plus_clahe",
                    "preprocess_mode": (
                        "enhanced_resized_fullframe" if resize_to_reference else "enhanced_fullres"
                    ) if disable_tiling else "enhanced_tiled",
                    "clip_limit": clip_limit,
                    "clahe_grid": tile_grid_size,
                    "denoise_strength": denoise_strength,
                }
            )

    return records


def write_manifest(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path, help="Directory containing source images")
    parser.add_argument("--output-dir", required=True, type=Path, help="Where enhanced/tiled images should be written")
    parser.add_argument("--reference-height", type=int, default=1024, help="Training-like reference image height")
    parser.add_argument("--reference-width", type=int, default=1440, help="Training-like reference image width")
    parser.add_argument("--clip-limit", type=float, default=2.0, help="CLAHE clip limit")
    parser.add_argument("--tile-grid-size", type=int, default=8, help="CLAHE grid size")
    parser.add_argument("--denoise-strength", type=float, default=0.0, help="Optional fastNlMeans denoising strength (e.g. 4-8)")
    parser.add_argument("--disable-tiling", action="store_true", help="Keep each image full-resolution instead of splitting into tiles")
    parser.add_argument("--resize-to-reference", action="store_true", help="After enhancement, resize each full image to the reference height/width")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    images = sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS])
    if not images:
        raise FileNotFoundError(f"No supported images found in {input_dir}")

    source_lookup = load_source_lookup(input_dir)
    all_rows: List[Dict[str, object]] = []

    for image_path in images:
        rows = process_image(
            image_path=image_path,
            output_dir=output_dir,
            source_lookup=source_lookup,
            reference_height=args.reference_height,
            reference_width=args.reference_width,
            clip_limit=args.clip_limit,
            tile_grid_size=args.tile_grid_size,
            denoise_strength=args.denoise_strength,
            disable_tiling=args.disable_tiling,
            resize_to_reference=args.resize_to_reference,
        )
        all_rows.extend(rows)

    write_manifest(output_dir.parent / "preprocess_manifest.csv", all_rows)
    print(f"Processed {len(images)} image(s) into {len(all_rows)} enhanced tile(s).")
    print(f"Outputs: {output_dir}")
    print(f"Manifest: {output_dir.parent / 'preprocess_manifest.csv'}")


if __name__ == "__main__":
    main()
