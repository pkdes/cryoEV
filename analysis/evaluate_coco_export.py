"""
Score frozen model predictions against a labeled Roboflow COCO export, split by
how each image was labeled (roboflow_upload.py's split_role):

  blind-eval     labeled from scratch      -> the honest P/R/F1
  pre-annotated  labeled from predictions  -> biased toward the model; for comparison

Matching, IoU and the confidence sweep are confidence_threshold_sweep.py's own
(full-resolution mask IoU, Hungarian, IoU >= 0.5); only the inputs differ: one
COCO file, a flat predictions folder, and the manifest for the roles.

Roboflow renames exported images (`<stem>_png.rf.<hash>.jpg`), so each export
image is mapped back to the original stem through COCO `extra.name` when present,
otherwise by stripping the `_<ext>.rf.<hash>` suffix. Predictions are normalized,
so a resized export still scores correctly.

    python analysis/evaluate_coco_export.py --coco-json <export>/_annotations.coco.json \
        --pred-dir ".../predictions/v4_y8m_20260924/predictions" \
        --manifest ".../3d.../MANIFEST.csv" --out <dir>
"""
import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "training"))
sys.path.insert(0, str(Path(__file__).parent))
from train_yolo import load_predictions_yolo_format
from multilayer_containment import load_image_annotations
from confidence_threshold_sweep import (CONF_VALUES, GT_EXCLUDE, compute_iou_matrix,
                                        score_at_threshold)


def original_stem(img):
    name = (img.get("extra") or {}).get("name") or img["file_name"]
    name = Path(name).name
    m = re.match(r"(.+)_(png|jpg|jpeg|tif|tiff)\.rf\.[0-9a-f]+\.\w+$", name, re.I)
    return m.group(1) if m else Path(name).stem


def load_records(coco_json, pred_dir, roles):
    by_role, unmatched = defaultdict(list), []
    for img, entries_all in load_image_annotations(coco_json):
        stem = original_stem(img)
        pred = pred_dir / f"{stem}.txt"
        if not pred.exists():
            unmatched.append(stem)
            continue
        gt = [m for _, cat, m in entries_all if cat not in GT_EXCLUDE]
        masks, confs, _ = load_predictions_yolo_format(str(pred), img["width"], img["height"])
        by_role[roles.get(stem, "unknown")].append({
            "stem": stem, "n_gt": len(gt), "confidences": np.array(confs),
            "iou_matrix": compute_iou_matrix(masks, gt)})
    return by_role, unmatched


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coco-json", required=True, type=Path)
    ap.add_argument("--pred-dir", required=True, type=Path)
    ap.add_argument("--manifest", type=Path, default=None, help="for split_role; else all 'unknown'")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    roles = {}
    if args.manifest:
        with open(args.manifest, newline="", encoding="utf-8") as fh:
            roles = {Path(r["output_name"]).stem: r.get("split_role") or "unknown"
                     for r in csv.DictReader(fh)}
    by_role, unmatched = load_records(args.coco_json, args.pred_dir, roles)
    if unmatched:
        print(f"WARNING {len(unmatched)} export images have no prediction file, e.g. {unmatched[:2]}")

    rows = []
    for role in sorted(by_role):
        recs = by_role[role]
        print(f"\n{role}: {len(recs)} images, {sum(r['n_gt'] for r in recs)} labeled objects")
        print("  conf   P      R      F1     TP    FP    FN")
        for c in CONF_VALUES:
            s = score_at_threshold(recs, c)
            rows.append({"split_role": role, "n_images": len(recs), "conf": c, **s})
            print(f"  {c:.2f}  {s['precision']:.3f}  {s['recall']:.3f}  {s['f1']:.3f}  "
                  f"{s['tp']:5d} {s['fp']:5d} {s['fn']:5d}")
    if args.out and rows:
        args.out.mkdir(parents=True, exist_ok=True)
        with open(args.out / "eval_by_role.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\n-> {args.out / 'eval_by_role.csv'}")


if __name__ == "__main__":
    main()
