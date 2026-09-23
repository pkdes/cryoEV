#!/usr/bin/env python3
"""Size and morphology profile of a folder of cryo-EM images, from cached model predictions.

No model is loaded here. Run inference once first, then profile from its prediction files:

    python inference/predict_models.py --images <dir> --out <pred_out> --models v3_run1_20260824
    python inference/batch_size_profile.py --input-dir <dir>         --predictions-dir <pred_out>/v3_run1_20260824/predictions --output-dir <profile_out>

Any `<stem>.txt` prediction folder works (e.g. a training run's predictions/test).

Outputs
-------
- per-image masks / decisions / morphology CSVs
- `morphology_all.csv` for all accepted objects
- `per_image_summary.csv` for image-level counts and diameter summaries
- `summary.csv` with folder-level metrics
- `morphology_distributions_all.png`
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from statistics import mean, median
from typing import Dict, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2

from analysis.morphology import plot_morphology_distributions, save_morphology_csv
from inference.inference import review_and_measure
from training.train_yolo import load_prediction_polygons, polygon_to_mask

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def iter_images(input_dir: Path) -> List[Path]:
    return sorted(
        path for path in input_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def summarize_values(values: List[float]) -> Dict[str, float | str]:
    if not values:
        return {
            "mean": "",
            "median": "",
            "min": "",
            "max": "",
        }
    return {
        "mean": float(mean(values)),
        "median": float(median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path, help="Folder of images to profile")
    parser.add_argument("--predictions-dir", required=True, type=Path,
                        help="Folder of cached predictions (<stem>.txt), e.g. <out>/<model_id>/predictions from predict_models.py")
    parser.add_argument("--output-dir", required=True, type=Path, help="Folder to save outputs")
    parser.add_argument("--confidence-threshold", type=float, default=0.5, help="Acceptance threshold after detection")
    parser.add_argument("--pixel-size", type=float, default=None, help="Optional physical pixel size, e.g. nm/px")
    parser.add_argument("--review", action="store_true", help="Enable manual review for below-threshold detections")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    args.predictions_dir = args.predictions_dir.resolve()
    if not args.predictions_dir.is_dir():
        raise FileNotFoundError(f"Predictions folder not found: {args.predictions_dir}")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = iter_images(input_dir)
    if not image_paths:
        raise FileNotFoundError(f"No supported images found under: {input_dir}")

    print(f"Found {len(image_paths)} image(s) under {input_dir}")

    all_records: List[Dict] = []
    per_image_rows: List[Dict] = []

    for image_path in image_paths:
        print(f"\nProcessing: {image_path.name}")
        pred_path = args.predictions_dir / f"{image_path.stem}.txt"
        if not pred_path.exists():
            print(f"  No prediction file ({pred_path.name}); skipping.")
            continue
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        h, w = image.shape[:2]
        polygons, confidences = load_prediction_polygons(pred_path, w, h)
        masks = [polygon_to_mask(poly, h, w) for poly in polygons]
        _, _, decisions, morph_records, exited = review_and_measure(
            image=image,
            image_name=image_path.stem,
            instance_masks=masks,
            confidences=confidences,
            instance_polygons=polygons,
            output_dir=str(output_dir / "per_image"),
            confidence_threshold=args.confidence_threshold,
            skip_review=not args.review,
            pixel_size=args.pixel_size,
        )
        if exited:
            print("Stopped early by user during manual review.")
            break

        image_records = []
        for rec in morph_records:
            rec = dict(rec)
            rec["image"] = image_path.name
            rec["image_rel_path"] = image_path.relative_to(input_dir).as_posix()
            all_records.append(rec)
            image_records.append(rec)

        diameters = [float(r["equivalent_diameter"]) for r in image_records if r.get("equivalent_diameter") is not None]
        confidences = [float(r["confidence"]) for r in image_records if r.get("confidence") is not None]
        per_image_rows.append(
            {
                "image": image_path.name,
                "image_rel_path": image_path.relative_to(input_dir).as_posix(),
                "n_detected_objects": len(decisions),
                "n_accepted_objects": len(image_records),
                "mean_confidence": float(mean(confidences)) if confidences else "",
                "mean_equivalent_diameter": float(mean(diameters)) if diameters else "",
                "median_equivalent_diameter": float(median(diameters)) if diameters else "",
            }
        )

    if all_records:
        save_morphology_csv(all_records, str(output_dir / "morphology_all.csv"))
        plot_morphology_distributions(all_records, pixel_size=args.pixel_size, save_path=str(output_dir / "morphology_distributions_all.png"))
    else:
        print("No accepted morphology records were produced.")

    write_csv(output_dir / "per_image_summary.csv", per_image_rows)

    diameters = [float(r["equivalent_diameter"]) for r in all_records if r.get("equivalent_diameter") is not None]
    confidences = [float(r["confidence"]) for r in all_records if r.get("confidence") is not None]
    circularities = [float(r["circularity"]) for r in all_records if r.get("circularity") is not None]
    aspects = [float(r["aspect_ratio"]) for r in all_records if r.get("aspect_ratio") is not None]

    folder_summary = {
        "input_dir": str(input_dir),
        "n_images": len(image_paths),
        "n_images_with_accepted_objects": sum(1 for row in per_image_rows if int(row["n_accepted_objects"]) > 0),
        "n_objects": len(all_records),
        "mean_confidence": summarize_values(confidences)["mean"],
        "mean_equivalent_diameter": summarize_values(diameters)["mean"],
        "median_equivalent_diameter": summarize_values(diameters)["median"],
        "min_equivalent_diameter": summarize_values(diameters)["min"],
        "max_equivalent_diameter": summarize_values(diameters)["max"],
        "mean_circularity": summarize_values(circularities)["mean"],
        "mean_aspect_ratio": summarize_values(aspects)["mean"],
    }
    write_csv(output_dir / "summary.csv", [folder_summary])

    print("\nSummary")
    for key, value in folder_summary.items():
        print(f"- {key}: {value}")
    print(f"\nOutputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
