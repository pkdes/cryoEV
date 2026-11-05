"""
Apply the multilayer-containment idea to a trained model's PREDICTIONS
(rather than ground truth, as in multilayer_containment.py) and validate the
resulting multilayer/standalone/inner-layer classification against the
original (pre-collapse) COCO ground truth categories (EV/MV/inner/etc.).

Workflow per image:
  1. Get the predicted "EV" polygons: from cached prediction files with
     --predictions-dir (the predict_models.py / run_experiments.py format,
     <dir>/<split>/<stem>.txt or <dir>/<stem>.txt, kept at conf >= --conf),
     or, by default, by running the model live (--model-path).
  2. Run the same containment test among predictions: a predicted polygon
     that fully contains >=1 other predicted polygon is classified
     "multilayer"; a predicted polygon fully contained inside another is
     classified "inner-layer"; everything else is "standalone".
  3. Match predictions to ground-truth polygons (Hungarian, IoU>=0.5) so we
     know each matched prediction's TRUE category (MV / inner / EV / ...).
  4. Score the multilayer classification (and inner-layer classification)
     as precision/recall against those true categories.

The live path stays the default: cached polygons are single largest contours,
rasterized by load_predictions_yolo_format(), which truncates vertices to int
(CLEANUP_CANDIDATES "R2 vertex truncation", undecided), so cached masks are not
the live masks. On v5 F5 / S1 valid (26 images, 2026-11-05) every TP/FP/FN, per
image, was identical either way; Run_14's own images are not on the laptop, so
its default setup is unchecked. Switching the default is the user's call.

--model <model_id> (models.yaml) keeps only the EV class of a multi-class model,
on both paths; without it every prediction is used, as before.

    python analysis/multilayer_containment_predictions.py                    # Run_14, live (as before)
    python analysis/multilayer_containment_predictions.py --predictions-dir <run>/predictions
"""

import argparse

import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent / "training"))
from train_yolo import (load_predictions_with_classes_from_model, load_predictions_yolo_format,
                        match_objects_hungarian)

sys.path.insert(0, str(Path(__file__).parent))
from multilayer_containment import load_image_annotations, find_containment, SOURCE_ROOT
sys.path.insert(0, str(Path(__file__).parent.parent))
from inference.model_classes import ev_class_ids

ROOT = Path(__file__).parent.parent.parent
DATASET = ROOT / "training outputs" / "roboflow_20260713_all_layer_singleclass"
MODEL_PATH = DATASET / "runs" / "2026-07-13_181034_y8_1440_24b_300ep_Run_14" / "weights" / "best.pt"

IMGSZ = 1440
CONF = 0.25
NMS_IOU = 0.7
MATCH_IOU = 0.5

# COCO split name -> converted YOLO dataset split dir name (see prepare_all_layer_singleclass.py)
SPLIT_MAP = {"train": "train", "valid": "val"}  # --splits names not listed here are used as-is


def classify_predictions(pred_masks):
    """Containment among predictions only -> multilayer / inner-layer / standalone per pred index."""
    entries = [(i, "pred", m) for i, m in enumerate(pred_masks)]
    contains = find_containment(entries)
    contained_by_someone = set()
    for parent, children in contains.items():
        contained_by_someone |= children

    role = {}
    layer_count = {}
    for i in range(len(pred_masks)):
        n_contained = len(contains.get(i, set()))
        if n_contained > 0:
            role[i] = "multilayer"
            layer_count[i] = n_contained
        elif i in contained_by_someone:
            role[i] = "inner-layer"
            layer_count[i] = 0
        else:
            role[i] = "standalone"
            layer_count[i] = 0
    return role, layer_count


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions-dir", type=Path, default=None,
                    help="read cached prediction files instead of running the model")
    ap.add_argument("--model-path", type=Path, default=MODEL_PATH, help="weights for the live path")
    ap.add_argument("--dataset-dir", type=Path, default=DATASET, help="YOLO dataset holding images/<split>/")
    ap.add_argument("--source-root", type=Path, default=SOURCE_ROOT,
                    help="COCO export with <split>/_annotations.coco.json (pre-collapse categories)")
    ap.add_argument("--splits", default=",".join(SPLIT_MAP), help="COCO splits, e.g. train,valid")
    ap.add_argument("--imgsz", type=int, default=IMGSZ)
    ap.add_argument("--conf", type=float, default=CONF,
                    help="live: predict conf; cached: keep predictions with conf >= this")
    ap.add_argument("--nms-iou", type=float, default=NMS_IOU)
    ap.add_argument("--match-iou", type=float, default=MATCH_IOU)
    ap.add_argument("--device", default="0")
    ap.add_argument("--model", default=None,
                    help="model_id in models.yaml: keep only the EV class of a multi-class model")
    return ap.parse_args(argv)


def get_predictions(img_path, yolo_split, args, keep_classes):
    """(masks, confidences) for one image, from the cache or the live model."""
    if args.predictions_dir is not None:
        txt = args.predictions_dir / yolo_split / f"{img_path.stem}.txt"
        if not txt.exists():
            txt = args.predictions_dir / f"{img_path.stem}.txt"
        if not txt.exists():
            # predict_models.py writes a file (maybe empty) for every image, so a missing one is an
            # error, not "no detections". On Windows a path over 260 characters also reads as missing.
            raise FileNotFoundError(f"no cached prediction for {img_path.name} under {args.predictions_dir} "
                                    f"(or the path exceeds MAX_PATH: {len(str(txt))} chars)")
        from PIL import Image
        with Image.open(img_path) as im:
            w, h = im.size
        masks, confs, classes = load_predictions_yolo_format(txt, w, h)
        keep = [i for i, c in enumerate(confs) if c >= args.conf]
    else:
        masks, confs, classes = load_predictions_with_classes_from_model(
            str(args.model_path), str(img_path), args.imgsz, args.conf, args.nms_iou, device=args.device
        )
        keep = list(range(len(masks)))
    if keep_classes is not None:
        keep = [i for i in keep if classes[i] in keep_classes]
    return [masks[i] for i in keep], [confs[i] for i in keep]


def main(argv=None):
    args = parse_args(argv)
    keep_classes = ev_class_ids(args.model) if args.model else None
    print("predictions: " + (f"cached {args.predictions_dir} (conf >= {args.conf})" if args.predictions_dir
                             else f"live {args.model_path} (conf={args.conf}, nms_iou={args.nms_iou}, "
                                  f"imgsz={args.imgsz})"))
    print("classes: " + (f"EV class {sorted(keep_classes)} only" if keep_classes else "all predictions"))

    # Confusion counts for the "multilayer" classification (predicted role=='multilayer' vs true category=='MV')
    ml_tp = ml_fp = ml_fn = 0
    # Confusion counts for "inner-layer" classification (predicted role=='inner-layer' vs true category=='inner')
    inner_tp = inner_fp = inner_fn = 0

    n_gt_mv_undetected = 0  # GT MV polygons with no matching prediction at all
    n_gt_inner_undetected = 0

    per_image_rows = []

    for coco_split in args.splits.split(","):
        yolo_split = SPLIT_MAP.get(coco_split, coco_split)
        coco_path = args.source_root / coco_split / "_annotations.coco.json"
        img_dir = args.dataset_dir / "images" / yolo_split

        for img, gt_entries in load_image_annotations(coco_path):
            img_path = img_dir / img["file_name"]
            if not img_path.exists() or not gt_entries:
                continue

            gt_masks = [m for _, _, m in gt_entries]
            gt_cats = [cat for _, cat, _ in gt_entries]

            pred_masks, confidences = get_predictions(img_path, yolo_split, args, keep_classes)
            pred_role, pred_layers = classify_predictions(pred_masks)

            matches, unmatched_preds, unmatched_gts = match_objects_hungarian(
                pred_masks, gt_masks, iou_threshold=args.match_iou
            )
            matched_gt_idx = {gt_idx for _, gt_idx in matches}

            row = {"image": img["file_name"], "n_pred": len(pred_masks), "n_gt": len(gt_entries),
                   "ml_tp": 0, "ml_fp": 0, "ml_fn": 0, "inner_tp": 0, "inner_fp": 0, "inner_fn": 0}

            for pred_idx, gt_idx in matches:
                true_cat = gt_cats[gt_idx]
                role = pred_role[pred_idx]

                is_true_mv = true_cat == "MV"
                is_pred_ml = role == "multilayer"
                if is_pred_ml and is_true_mv:
                    ml_tp += 1; row["ml_tp"] += 1
                elif is_pred_ml and not is_true_mv:
                    ml_fp += 1; row["ml_fp"] += 1
                elif (not is_pred_ml) and is_true_mv:
                    ml_fn += 1; row["ml_fn"] += 1

                is_true_inner = true_cat == "inner"
                is_pred_inner = role == "inner-layer"
                if is_pred_inner and is_true_inner:
                    inner_tp += 1; row["inner_tp"] += 1
                elif is_pred_inner and not is_true_inner:
                    inner_fp += 1; row["inner_fp"] += 1
                elif (not is_pred_inner) and is_true_inner:
                    inner_fn += 1; row["inner_fn"] += 1

            # GT polygons never detected at all (no matching prediction) also count as
            # missed multilayer/inner detections, not just misclassifications.
            for gt_idx, (_, cat, _) in enumerate(gt_entries):
                if gt_idx in matched_gt_idx:
                    continue
                if cat == "MV":
                    ml_fn += 1; row["ml_fn"] += 1; n_gt_mv_undetected += 1
                elif cat == "inner":
                    inner_fn += 1; row["inner_fn"] += 1; n_gt_inner_undetected += 1

            if row["n_pred"] or row["n_gt"]:
                per_image_rows.append(row)

    def prf(tp, fp, fn):
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        return p, r, f1

    ml_p, ml_r, ml_f1 = prf(ml_tp, ml_fp, ml_fn)
    in_p, in_r, in_f1 = prf(inner_tp, inner_fp, inner_fn)

    print("=" * 72)
    print("MULTILAYER CLASSIFICATION ON PREDICTIONS -- "
          + (f"cached: {args.predictions_dir}" if args.predictions_dir
             else f"model: {args.model_path.parent.parent.name}"))
    print(f"conf={args.conf} match_iou={args.match_iou} containment_threshold=0.9")
    print("=" * 72)
    print("\n-- 'Multilayer' classification (predicted contains >=1 other pred) vs true category=='MV' --")
    print(f"TP={ml_tp} FP={ml_fp} FN={ml_fn}  (of which {n_gt_mv_undetected} GT MV polygons were never detected)")
    print(f"Precision: {ml_p:.4f}  Recall: {ml_r:.4f}  F1: {ml_f1:.4f}")

    print("\n-- 'Inner-layer' classification (predicted contained inside another pred) vs true category=='inner' --")
    print(f"TP={inner_tp} FP={inner_fp} FN={inner_fn}  (of which {n_gt_inner_undetected} GT inner polygons were never detected)")
    print(f"Precision: {in_p:.4f}  Recall: {in_r:.4f}  F1: {in_f1:.4f}")

    print("\nPer-image detail:")
    for r in per_image_rows:
        print(f"  {r['image']}: n_pred={r['n_pred']} n_gt={r['n_gt']} "
              f"ML(tp/fp/fn)={r['ml_tp']}/{r['ml_fp']}/{r['ml_fn']} "
              f"Inner(tp/fp/fn)={r['inner_tp']}/{r['inner_fp']}/{r['inner_fn']}")


if __name__ == "__main__":
    main()
