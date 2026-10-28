#!/usr/bin/env python3
"""Size profile from CACHED prediction files, with no model load.

This is CLEANUP_CANDIDATES R1. `batch_size_profile.py` produces the same CSVs
but only via live inference, so re-measuring at a new pixel size, or with a
different confidence cut, meant re-running the GPU. `predict_models.py` has
already written `<model_id>/predictions/<stem>.txt`; this reads those back and
emits the identical schema, so `aggregate_size_results.py` and
`compare_size_profiles.py` run downstream unchanged.

--pixel-size is nm/px of the images AS PREDICTED ON (the plot labels and the
morphology fields are in nm). For a scale-matched export it is constant across
images, which is the whole point of exporting at one spacing.

Example
-------
python -m inference.size_profile_from_predictions \
    --predictions-dir <out>/v4_y8m_20260924/predictions \
    --images-dir <yolo_input>/accept \
    --output-dir <out>/v4_y8m_20260924/morphology \
    --pixel-size 1.47288
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean, median

from PIL import Image

from analysis.morphology import plot_morphology_distributions, save_morphology_csv
from training.train_yolo import load_predictions_yolo_format

Image.MAX_IMAGE_PIXELS = None
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def write_csv(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions-dir", required=True, type=Path)
    ap.add_argument("--images-dir", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--pixel-size", type=float, default=None, help="nm/px of the predicted images")
    ap.add_argument("--confidence-threshold", type=float, default=0.5,
                    help="acceptance threshold applied after detection, as in batch_size_profile")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    from analysis.morphology import analyze_instances

    images_dir = args.images_dir.resolve()
    image_paths = sorted(p for p in images_dir.rglob("*")
                         if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not image_paths:
        raise FileNotFoundError(f"No supported images found under: {images_dir}")

    print(f"{len(image_paths)} image(s); predictions from {args.predictions_dir}")
    print(f"pixel size {args.pixel_size} nm/px, confidence >= {args.confidence_threshold}")

    all_records, per_image_rows = [], []
    for image_path in image_paths:
        with Image.open(image_path) as im:
            w, h = im.size
        txt = args.predictions_dir / f"{image_path.stem}.txt"
        masks, confs, _ = load_predictions_yolo_format(txt, w, h)
        n_detected = len(masks)

        keep = [i for i, c in enumerate(confs) if c >= args.confidence_threshold]
        recs = analyze_instances([masks[i] for i in keep],
                                 confidences=[confs[i] for i in keep],
                                 pixel_size=args.pixel_size)
        for rec in recs:
            rec = dict(rec)
            rec["image"] = image_path.name
            rec["image_rel_path"] = image_path.relative_to(images_dir).as_posix()
            all_records.append(rec)

        diams = [float(r["equivalent_diameter"]) for r in recs
                 if r.get("equivalent_diameter") is not None]
        kept_conf = [confs[i] for i in keep]
        per_image_rows.append({
            "image": image_path.name,
            "image_rel_path": image_path.relative_to(images_dir).as_posix(),
            "n_detected_objects": n_detected,
            "n_accepted_objects": len(recs),
            "mean_confidence": float(mean(kept_conf)) if kept_conf else "",
            "mean_equivalent_diameter": float(mean(diams)) if diams else "",
            "median_equivalent_diameter": float(median(diams)) if diams else "",
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if all_records:
        save_morphology_csv(all_records, str(args.output_dir / "morphology_all.csv"))
        if not args.no_plot:
            plot_morphology_distributions(
                all_records, pixel_size=args.pixel_size,
                save_path=str(args.output_dir / "morphology_distributions_all.png"))
    else:
        print("No accepted morphology records were produced.")
    write_csv(args.output_dir / "per_image_summary.csv", per_image_rows)

    n_det = sum(r["n_detected_objects"] for r in per_image_rows)
    print(f"detected {n_det} -> accepted {len(all_records)} object(s) "
          f"over {len(per_image_rows)} image(s) -> {args.output_dir}")


if __name__ == "__main__":
    main()
