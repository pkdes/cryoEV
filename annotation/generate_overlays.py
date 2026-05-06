"""
generate_overlays.py
--------------------
Retroactively generate missing overlays for existing annotation session folders.

For each session under ANNOTATION_ROOT (or a single session path you pass):
  - reviewed/overlays/     : model predictions (yellow) + user corrections (white)
  - reviewed_detect/overlays/ : model predictions only (yellow boxes / polygons)

Run from the repo root:
    python -m annotation.generate_overlays
or with a specific root:
    python -m annotation.generate_overlays --root "C:/path/to/annotation_outputs"
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# YOLO label readers
# ---------------------------------------------------------------------------

def _read_yolo_polygons(label_path: Path, w: int, h: int) -> List[np.ndarray]:
    """Read a YOLO polygon label file and return absolute-pixel polygons."""
    polys: List[np.ndarray] = []
    if not label_path.exists():
        return polys
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 7:  # class + at least 3 x,y pairs
            continue
        coords = [float(v) for v in parts[1:]]
        if len(coords) % 2 != 0:
            continue
        pts = np.array(coords, dtype=np.float32).reshape(-1, 2)
        pts[:, 0] *= w
        pts[:, 1] *= h
        polys.append(pts.astype(np.int32))
    return polys


def _read_yolo_boxes(label_path: Path, w: int, h: int) -> List[np.ndarray]:
    """Read a YOLO box label file and return absolute-pixel polygon outlines."""
    polys: List[np.ndarray] = []
    if not label_path.exists():
        return polys
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        _, cx, cy, bw, bh = [float(v) for v in parts[:5]]
        x0 = (cx - bw / 2) * w
        y0 = (cy - bh / 2) * h
        x1 = (cx + bw / 2) * w
        y1 = (cy + bh / 2) * h
        poly = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.int32)
        polys.append(poly)
    return polys


# ---------------------------------------------------------------------------
# Drawing helper
# ---------------------------------------------------------------------------

def _draw_polygons(
    image: np.ndarray,
    polygons: List[np.ndarray],
    color: tuple,
    thickness: int = 2,
) -> np.ndarray:
    if image.ndim == 2:
        canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        canvas = image.copy()
    for pts in polygons:
        pts_arr = np.asarray(pts, dtype=np.int32)
        if pts_arr.ndim != 2 or pts_arr.shape[0] < 2:
            continue
        cv2.polylines(canvas, [pts_arr.reshape(-1, 1, 2)], isClosed=True,
                      color=color, thickness=thickness)
    return canvas


# ---------------------------------------------------------------------------
# Per-session processing
# ---------------------------------------------------------------------------

def _process_session(session_dir: Path, overwrite: bool) -> int:
    """Generate any missing overlays for one session folder.  Returns # written."""
    reviewed_images_dir = session_dir / "reviewed" / "images"
    reviewed_labels_dir = session_dir / "reviewed" / "labels"
    reviewed_overlays_dir = session_dir / "reviewed" / "overlays"
    detect_labels_dir = session_dir / "reviewed_detect" / "labels"
    detect_overlays_dir = session_dir / "reviewed_detect" / "overlays"

    if not reviewed_images_dir.exists():
        return 0

    image_paths = sorted(reviewed_images_dir.glob("*"))
    image_paths = [p for p in image_paths
                   if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}]
    if not image_paths:
        return 0

    n_written = 0
    for img_path in image_paths:
        stem = img_path.stem
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"    [skip] Cannot read: {img_path.name}")
            continue
        h, w = image.shape[:2]

        # --- reviewed/overlays : auto (yellow) + reviewed (white) -------
        reviewed_overlay_path = reviewed_overlays_dir / f"{stem}_overlay.png"
        if overwrite or not reviewed_overlay_path.exists():
            reviewed_polys = _read_yolo_polygons(reviewed_labels_dir / f"{stem}.txt", w, h)
            # No auto polygons available retroactively; draw reviewed only.
            canvas = _draw_polygons(image, reviewed_polys, color=(255, 255, 255), thickness=2)
            reviewed_overlays_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(reviewed_overlay_path), canvas)
            n_written += 1

        # --- reviewed_detect/overlays : model box predictions (cyan) ----
        detect_overlay_path = detect_overlays_dir / f"{stem}_overlay.png"
        if overwrite or not detect_overlay_path.exists():
            detect_polys = _read_yolo_boxes(detect_labels_dir / f"{stem}.txt", w, h)
            if detect_polys or (detect_labels_dir / f"{stem}.txt").exists():
                canvas2 = _draw_polygons(image, detect_polys, color=(0, 255, 255), thickness=2)
                detect_overlays_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(detect_overlay_path), canvas2)
                n_written += 1

    return n_written


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate missing overlays for annotation session folders."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(r"C:\Users\ML-2619\Desktop\Pujan Cryo\cryo-ev pipeline\annotation_outputs"),
        help="Root folder containing session_* sub-folders.",
    )
    parser.add_argument(
        "--session",
        type=Path,
        default=None,
        help="Process a single session folder instead of scanning --root.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate overlays even if they already exist.",
    )
    args = parser.parse_args()

    if args.session:
        sessions = [args.session]
    else:
        # Include both session_YYYYMMDD_HHMMSS and legacy named folders.
        sessions = sorted(
            [p for p in args.root.iterdir() if p.is_dir() and p.name != "archive"],
        )

    total = 0
    for session_dir in sessions:
        n = _process_session(session_dir, overwrite=args.overwrite)
        if n:
            print(f"  {session_dir.name}: wrote {n} overlay(s)")
        else:
            print(f"  {session_dir.name}: nothing to do")
        total += n

    print(f"\nDone. {total} overlay file(s) written.")


if __name__ == "__main__":
    main()
