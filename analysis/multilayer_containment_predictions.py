"""
Apply the multilayer-containment idea to a trained model's PREDICTIONS
(rather than ground truth, as in multilayer_containment.py) and validate the
resulting multilayer/standalone/inner-layer classification against the
original (pre-collapse) COCO ground truth categories (EV/MV/inner/etc.).

Workflow per image:
  1. Run the model to get predicted "EV" polygons.
  2. Run the same containment test among predictions: a predicted polygon
     that fully contains >=1 other predicted polygon is classified
     "multilayer"; a predicted polygon fully contained inside another is
     classified "inner-layer"; everything else is "standalone".
  3. Match predictions to ground-truth polygons (Hungarian, IoU>=0.5) so we
     know each matched prediction's TRUE category (MV / inner / EV / ...).
  4. Score the multilayer classification (and inner-layer classification)
     as precision/recall against those true categories.
"""

import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent / "training"))
from train_yolo import load_predictions_with_classes_from_model, match_objects_hungarian

sys.path.insert(0, str(Path(__file__).parent))
from multilayer_containment import load_image_annotations, find_containment, SOURCE_ROOT

ROOT = Path(__file__).parent.parent.parent
DATASET = ROOT / "training outputs" / "roboflow_20260713_all_layer_singleclass"
MODEL_PATH = DATASET / "runs" / "2026-07-13_181034_y8_1440_24b_300ep_Run_14" / "weights" / "best.pt"

IMGSZ = 1440
CONF = 0.25
NMS_IOU = 0.7
MATCH_IOU = 0.5

# COCO split name -> converted YOLO dataset split dir name (see prepare_all_layer_singleclass.py)
SPLIT_MAP = {"train": "train", "valid": "val"}


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


def main():
    # Confusion counts for the "multilayer" classification (predicted role=='multilayer' vs true category=='MV')
    ml_tp = ml_fp = ml_fn = 0
    # Confusion counts for "inner-layer" classification (predicted role=='inner-layer' vs true category=='inner')
    inner_tp = inner_fp = inner_fn = 0

    n_gt_mv_undetected = 0  # GT MV polygons with no matching prediction at all
    n_gt_inner_undetected = 0

    per_image_rows = []

    for coco_split, yolo_split in SPLIT_MAP.items():
        coco_path = SOURCE_ROOT / coco_split / "_annotations.coco.json"
        img_dir = DATASET / "images" / yolo_split

        for img, gt_entries in load_image_annotations(coco_path):
            img_path = img_dir / img["file_name"]
            if not img_path.exists() or not gt_entries:
                continue

            gt_masks = [m for _, _, m in gt_entries]
            gt_cats = [cat for _, cat, _ in gt_entries]

            pred_masks, confidences, pred_classes = load_predictions_with_classes_from_model(
                str(MODEL_PATH), str(img_path), IMGSZ, CONF, NMS_IOU, device='0'
            )
            pred_role, pred_layers = classify_predictions(pred_masks)

            matches, unmatched_preds, unmatched_gts = match_objects_hungarian(
                pred_masks, gt_masks, iou_threshold=MATCH_IOU
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
    print(f"MULTILAYER CLASSIFICATION ON PREDICTIONS -- model: {MODEL_PATH.parent.parent.name}")
    print(f"conf={CONF} match_iou={MATCH_IOU} containment_threshold=0.9")
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
