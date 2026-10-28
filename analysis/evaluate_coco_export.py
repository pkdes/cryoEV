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

Multi-class models (EV / non-EV / Undetermined) and single-class baselines are
compared through --view, which decides what counts as an EV on both sides:

  fair       GT: EV layers only; Low conf EV, VLDL and non-EV are IGNORED (a
             prediction on one is neither TP nor FP). Preds: --ev-classes only
             (Undetermined dropped). EV finding on equal terms.
  practical  GT: EV layers + Low conf EV; VLDL/non-EV are not EVs, so an EV
             prediction on one is an FP. Preds: --ev-classes + --undet-class.
             What a user of the model would see.
  (omitted)  original behavior: every prediction vs GT minus GT_EXCLUDE.

--group-csv groups images by a column of any CSV keyed by image name instead of
the manifest's split_role, e.g. make_balanced_split.py's split_assignment.csv by
batch. --nonev-class also scores that predicted class against VLDL + non-EV GT,
and --undet-class (practical view) scores Undetermined against Low conf EV GT.
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


NON_EV_CATS = {"VLDL", "non-EV"}
LOW_CONF_CATS = {"Low conf EV"}


def drop_ignored(masks, confs, ignore_gt, target_gt):
    """Remove predictions that sit on an ignored GT object (IoU >= 0.5, and better
    than their best target match) -- crowd-style "don't care" regions."""
    if not ignore_gt or not masks:
        return masks, confs
    ign = compute_iou_matrix(masks, ignore_gt).max(axis=1)
    tgt = compute_iou_matrix(masks, target_gt).max(axis=1) if target_gt else np.zeros(len(masks))
    keep = ~((ign >= 0.5) & (ign > tgt))
    return [m for m, k in zip(masks, keep) if k], [c for c, k in zip(confs, keep) if k]


def load_records(coco_json, pred_dir, roles, view=None, ev_classes=(0,), undet_class=None,
                 target="ev", nonev_class=None):
    by_role, unmatched = defaultdict(list), []
    ignore_cats = LOW_CONF_CATS | NON_EV_CATS if view == "fair" else set()
    keep_classes = set(ev_classes) | ({undet_class} if view == "practical" and undet_class is not None else set())
    for img, entries_all in load_image_annotations(coco_json):
        stem = original_stem(img)
        pred = pred_dir / f"{stem}.txt"
        if not pred.exists():
            pred = pred_dir / f"{Path(img['file_name']).stem}.txt"   # run_experiments' cache naming
        if not pred.exists():
            unmatched.append(stem)
            continue
        masks, confs, classes = load_predictions_yolo_format(str(pred), img["width"], img["height"])
        if target in ("nonev", "undet"):
            gt_cats, cls = (NON_EV_CATS, nonev_class) if target == "nonev" else (LOW_CONF_CATS, undet_class)
            gt = [m for _, cat, m in entries_all if cat in gt_cats]
            sel = [i for i, c in enumerate(classes) if c == cls]
            masks, confs = [masks[i] for i in sel], [confs[i] for i in sel]
        elif view:
            gt = [m for _, cat, m in entries_all if cat not in GT_EXCLUDE | NON_EV_CATS | ignore_cats]
            ignore_gt = [m for _, cat, m in entries_all if cat in ignore_cats]
            sel = [i for i, c in enumerate(classes) if c in keep_classes]
            masks, confs = drop_ignored([masks[i] for i in sel], [confs[i] for i in sel], ignore_gt, gt)
        else:
            gt = [m for _, cat, m in entries_all if cat not in GT_EXCLUDE]
        role = roles.get(stem) or roles.get(Path(img["file_name"]).stem) or "unknown"
        by_role[role].append({
            "stem": stem, "n_gt": len(gt), "confidences": np.array(confs),
            "iou_matrix": compute_iou_matrix(masks, gt)})
    return by_role, unmatched


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coco-json", required=True, type=Path)
    ap.add_argument("--pred-dir", required=True, type=Path)
    ap.add_argument("--manifest", type=Path, default=None, help="for split_role; else all 'unknown'")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--group-csv", type=Path, help="CSV keyed by image name; groups by --group-col")
    ap.add_argument("--group-col", default="batch")
    ap.add_argument("--view", choices=["fair", "practical"])
    ap.add_argument("--ev-classes", default="0", help="Predicted class ids that mean EV")
    ap.add_argument("--undet-class", type=int, help="Predicted class id for Undetermined")
    ap.add_argument("--nonev-class", type=int, help="Also score this predicted class vs VLDL + non-EV GT")
    ap.add_argument("--label", default="", help="Model label written into the CSV")
    args = ap.parse_args()

    roles = {}
    if args.manifest:
        with open(args.manifest, newline="", encoding="utf-8") as fh:
            roles = {Path(r["output_name"]).stem: r.get("split_role") or "unknown"
                     for r in csv.DictReader(fh)}
    if args.group_csv:
        with open(args.group_csv, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if not r.get("split", "").startswith("DROPPED"):
                    roles[re.sub(r"_(png|jpg|jpeg|tif|tiff)$", "", r["image"])] = r[args.group_col]
    ev_classes = [int(c) for c in args.ev_classes.split(",")]
    targets = [("ev", args.view or "all")] + ([("nonev", "nonev")] if args.nonev_class is not None else [])
    if args.undet_class is not None and args.view == "practical":
        targets.append(("undet", "undet"))   # Undetermined preds vs Low conf EV GT

    rows = []
    for target, view_name in targets:
        by_role, unmatched = load_records(args.coco_json, args.pred_dir, roles, args.view, ev_classes,
                                          args.undet_class, target, args.nonev_class)
        if unmatched:
            print(f"WARNING {len(unmatched)} export images have no prediction file, e.g. {unmatched[:2]}")
        groups = dict(by_role)
        if args.group_csv:
            groups["ALL"] = [r for recs in by_role.values() for r in recs]
        for role in sorted(groups):
            recs = groups[role]
            print(f"\n[{target}/{view_name}] {role}: {len(recs)} images, {sum(r['n_gt'] for r in recs)} labeled objects")
            print("  conf   P      R      F1     TP    FP    FN")
            for c in CONF_VALUES:
                s = score_at_threshold(recs, c)
                rows.append({"model": args.label, "target": target, "view": view_name, "split_role": role,
                             "n_images": len(recs), "conf": c, **s})
                print(f"  {c:.2f}  {s['precision']:.3f}  {s['recall']:.3f}  {s['f1']:.3f}  "
                      f"{s['tp']:5d} {s['fp']:5d} {s['fn']:5d}")
    if args.out and rows:
        args.out.mkdir(parents=True, exist_ok=True)
        name = f"eval_{args.label}_{args.view}.csv" if args.view else "eval_by_role.csv"
        with open(args.out / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\n-> {args.out / name}")


if __name__ == "__main__":
    main()
