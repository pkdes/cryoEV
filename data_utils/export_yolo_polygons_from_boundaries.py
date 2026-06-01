"""Export YOLO polygon TXT labels from boundary JSON files.

Boundary JSON format expected (created by prior standard EV workflow):
{
  "image": "name.png",
  "objects": [
    {"object_id": 0, "n_points": 120, "contour_xy": [[x, y], ...]},
    ...
  ]
}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from annotation.io import write_yolo_polygon_labels


def _load_boundaries(path: Path) -> tuple[str | None, list[np.ndarray]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    image_name = payload.get("image")
    polygons: list[np.ndarray] = []

    for obj in payload.get("objects", []):
        pts = np.asarray(obj.get("contour_xy", []), dtype=np.float32)
        if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] != 2:
            continue
        polygons.append(pts)

    return image_name, polygons


def export_labels(
    boundaries_dir: Path,
    images_dir: Path,
    output_labels_dir: Path,
    class_id: int = 0,
) -> tuple[int, int]:
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(boundaries_dir.glob("*_boundaries.json"))
    n_files = 0
    n_polys = 0

    for boundary_file in files:
        image_name, polygons = _load_boundaries(boundary_file)
        if not polygons:
            continue

        if image_name:
            image_path = images_dir / image_name
        else:
            stem = boundary_file.name.replace("_boundaries.json", "")
            candidates = [
                images_dir / f"{stem}.png",
                images_dir / f"{stem}.jpg",
                images_dir / f"{stem}.jpeg",
                images_dir / f"{stem}.tif",
                images_dir / f"{stem}.tiff",
                images_dir / f"{stem}.bmp",
            ]
            image_path = next((p for p in candidates if p.exists()), None)

        if image_path is None or not Path(image_path).exists():
            print(f"[skip] Could not locate source image for {boundary_file.name}")
            continue

        img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"[skip] Could not read image {image_path}")
            continue

        h, w = img.shape[:2]
        label_name = Path(image_path).stem + ".txt"
        label_path = output_labels_dir / label_name
        write_yolo_polygon_labels(label_path, polygons, width=w, height=h, class_id=class_id)

        n_files += 1
        n_polys += len(polygons)

    return n_files, n_polys


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Export YOLO polygon labels from boundary JSON outputs.")
    p.add_argument("--boundaries-dir", type=Path, required=True, help="Folder containing *_boundaries.json files")
    p.add_argument("--images-dir", type=Path, required=True, help="Folder containing source images")
    p.add_argument("--output-labels-dir", type=Path, default=None, help="Destination folder for YOLO .txt labels")
    p.add_argument("--class-id", type=int, default=0, help="YOLO class id to write for all polygons")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    output = args.output_labels_dir or (args.boundaries_dir / "yolo_polygon_labels")

    n_files, n_polys = export_labels(
        boundaries_dir=args.boundaries_dir,
        images_dir=args.images_dir,
        output_labels_dir=output,
        class_id=args.class_id,
    )
    print(f"Exported YOLO polygon labels for {n_files} images ({n_polys} polygons) to: {output}")


if __name__ == "__main__":
    main()
