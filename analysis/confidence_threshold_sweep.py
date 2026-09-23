"""
Confidence-threshold sweep against ALREADY-CACHED predictions -- no
re-inference, no retraining. The cache was written by run_experiments.py's
export_predictions_for_split() at conf=0.25, so this sweep can only raise
the effective threshold from there; nothing below 0.25 is recoverable
without rerunning inference.

Outputs:
  <dataset>/runs/<model-run-dir>/confidence_sweep.csv   (split, conf, P, R, F1, tp, fp, fn)
  console table per split + combined, marking the best-F1 threshold.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).parent.parent / "training"))
from train_yolo import load_predictions_yolo_format

sys.path.insert(0, str(Path(__file__).parent))
from multilayer_containment import load_image_annotations

ROOT = Path(__file__).parent.parent.parent / "CryoAI"

GT_EXCLUDE = {"non-EV", "items-nfaR"}
MATCH_IOU = 0.5
CONF_VALUES = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]


def compute_iou_matrix(pred_masks: list, gt_masks: list) -> np.ndarray:
    """Full-resolution pixel-mask IoU between every (pred, gt) pair -- the expensive O(n*m)
    step. Computed ONCE per image, independent of confidence threshold, so the sweep below
    only reruns the cheap Hungarian assignment on threshold-filtered rows of this matrix."""
    iou = np.zeros((len(pred_masks), len(gt_masks)))
    for i, pm in enumerate(pred_masks):
        for j, gm in enumerate(gt_masks):
            inter = (pm & gm).sum()
            union = (pm | gm).sum()
            iou[i, j] = inter / union if union > 0 else 0.0
    return iou


def load_split_records(coco_path: Path, img_dir: Path, pred_dir: Path):
    """Load per-image GT masks + cached predictions, and precompute the full pred x gt IoU
    matrix once. The sweep filters/re-matches in-memory without touching the masks again."""
    records = []
    for img, entries_all in load_image_annotations(coco_path):
        entries = [(aid, cat, m) for aid, cat, m in entries_all if cat not in GT_EXCLUDE]
        img_path = img_dir / img["file_name"]
        pred_path = pred_dir / f"{img_path.stem}.txt"
        if not img_path.exists() or not pred_path.exists():
            continue
        w, h = img["width"], img["height"]
        pred_masks, confidences, pred_classes = load_predictions_yolo_format(str(pred_path), w, h)
        gt_masks = [m for _, _, m in entries]
        iou_matrix = compute_iou_matrix(pred_masks, gt_masks)
        records.append({
            "n_gt": len(gt_masks), "confidences": np.array(confidences),
            "iou_matrix": iou_matrix,
        })
    return records


def match_from_iou(iou_matrix: np.ndarray, n_pred: int, n_gt: int, iou_threshold: float):
    """Hungarian assignment on an already-computed IoU matrix -- no mask access, cheap."""
    if n_pred == 0 or n_gt == 0:
        return 0, n_pred, n_gt
    pred_idx, gt_idx = linear_sum_assignment(-iou_matrix)
    tp = int((iou_matrix[pred_idx, gt_idx] >= iou_threshold).sum())
    return tp, n_pred - tp, n_gt - tp


def score_at_threshold(records: list, conf_threshold: float):
    tp_total = fp_total = fn_total = 0
    for rec in records:
        keep = rec["confidences"] >= conf_threshold
        sub_iou = rec["iou_matrix"][keep, :]
        tp, fp, fn = match_from_iou(sub_iou, int(keep.sum()), rec["n_gt"], MATCH_IOU)
        tp_total += tp
        fp_total += fp
        fn_total += fn
    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) else 1.0
    recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp_total, "fp": fp_total, "fn": fn_total}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="roboflow_20260713_all_layer_singleclass",
                         help="Folder name under 'training outputs/'")
    parser.add_argument("--source-name", default="roboflow - 20260713 all layer label",
                         help="Folder name under 'training images/' holding the raw Roboflow COCO export")
    parser.add_argument("--model-run-dir", default="2026-07-13_181034_y8_1440_24b_300ep_Run_14",
                         help="Run folder name under '<dataset>/runs/' whose cached predictions to sweep")
    args = parser.parse_args()

    source_root = ROOT / "training images" / args.source_name
    dataset_root = ROOT / "training outputs" / args.dataset
    run_dir = dataset_root / "runs" / args.model_run_dir
    out_dir = dataset_root / "runs" / args.model_run_dir  # per run, so sweeps of different runs don't overwrite each other
    out_dir.mkdir(parents=True, exist_ok=True)

    split_map = {"train": "train", "valid": "val"}
    if (source_root / "test" / "_annotations.coco.json").exists():
        split_map["test"] = "test"

    per_split_records = {}
    for coco_split, yolo_split in split_map.items():
        coco_path = source_root / coco_split / "_annotations.coco.json"
        img_dir = dataset_root / "images" / yolo_split
        pred_dir = run_dir / "predictions" / yolo_split
        per_split_records[yolo_split] = load_split_records(coco_path, img_dir, pred_dir)

    combined_records = [r for recs in per_split_records.values() for r in recs]
    all_groups = list(per_split_records.items()) + [("combined", combined_records)]

    rows = []
    for split_label, records in all_groups:
        scored = [(conf, score_at_threshold(records, conf)) for conf in CONF_VALUES]
        best_conf, best_metrics = max(scored, key=lambda cm: cm[1]["f1"])
        for conf, m in scored:
            rows.append({"split": split_label, "conf_threshold": conf, **m})

        print("=" * 90)
        print(f"CONFIDENCE SWEEP -- {split_label.upper()}")
        print("=" * 90)
        print(f"{'conf':>6}{'precision':>12}{'recall':>10}{'f1':>10}{'tp':>7}{'fp':>7}{'fn':>7}")
        for conf, m in scored:
            marker = "  <== best F1" if conf == best_conf else ""
            print(f"{conf:>6.2f}{m['precision']:>12.4f}{m['recall']:>10.4f}{m['f1']:>10.4f}{m['tp']:>7}{m['fp']:>7}{m['fn']:>7}{marker}")
        print()

    csv_path = out_dir / "confidence_sweep.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "conf_threshold", "precision", "recall", "f1", "tp", "fp", "fn"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[DONE] Sweep written to {csv_path}")


if __name__ == "__main__":
    main()
