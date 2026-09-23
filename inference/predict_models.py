"""
Run one or more registered EV models (models/models.yaml) over a folder of images.

Per model, writes to <out>/<model_id>/:
  predictions/<stem>.txt  -- standard prediction cache format ('class conf x1 y1 ...', normalized),
                             readable by rank_predictions.py, multilayer_role_analysis.py, etc.
  overlays/<stem>.png     -- predicted polygon outlines on the image
Plus <out>/summary.csv: one row per image x model (n_detections, mean_conf).

Each model runs at its own registered imgsz -- do not override it, or the comparison is skewed.

Usage:
  python inference/predict_models.py --images <dir> --out <dir> [--models all|id1,id2] [--conf 0.25] [--device cuda]
"""

import argparse
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import numpy as np
import yaml

from training.train_yolo import export_predictions_for_split, load_prediction_polygons
from analysis.morphology import draw_polygons_on_image

REPO = Path(__file__).resolve().parent.parent
REGISTRY = REPO / "models" / "models.yaml"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--models", default="all", help="'all' or comma-separated model_ids from models.yaml")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    registry = yaml.safe_load(REGISTRY.read_text())["models"]
    ids = list(registry) if args.models == "all" else args.models.split(",")
    unknown = [i for i in ids if i not in registry]
    if unknown:
        sys.exit(f"Unknown model_id(s) {unknown}; available: {list(registry)}")

    images = sorted(p for p in args.images.iterdir()
                    if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.tif', '.tiff'))
    rows = []
    for mid in ids:
        spec = registry[mid]
        pred_dir = args.out / mid / "predictions"
        ov_dir = args.out / mid / "overlays"
        ov_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[{mid}] imgsz={spec['imgsz']}")
        export_predictions_for_split(str(REPO / spec["weights"]), str(args.images), pred_dir,
                                     imgsz=spec["imgsz"], conf=args.conf, iou=args.iou, device=args.device)
        for img_path in images:
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            h, w = img.shape[:2]
            polys, confs = load_prediction_polygons(pred_dir / f"{img_path.stem}.txt", w, h)
            vis = draw_polygons_on_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), polys)
            cv2.putText(vis, f"{mid}: n={len(polys)}", (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
            cv2.imwrite(str(ov_dir / f"{img_path.stem}.png"), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            rows.append({"image": img_path.name, "model_id": mid, "n_detections": len(polys),
                         "mean_conf": round(float(np.mean(confs)), 4) if confs else ""})

    with open(args.out / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "model_id", "n_detections", "mean_conf"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nDone. {len(images)} images x {len(ids)} models -> {args.out}")


if __name__ == "__main__":
    main()
