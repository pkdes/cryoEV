"""Detection-vs-review matching metrics, shared by the annotation tools.

Moved verbatim from hitl_annotator.py so the web QC tool and the tkinter
annotator score reviews with the same code.
"""

from typing import Dict, List

import numpy as np
from scipy.optimize import linear_sum_assignment


def _iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(inter / union) if union > 0 else 0.0


def _evaluate_predictions(auto_masks: List[np.ndarray], reviewed_masks: List[np.ndarray], iou_threshold: float = 0.5) -> Dict[str, float]:
    n_pred = len(auto_masks)
    n_gt = len(reviewed_masks)

    if n_pred == 0 and n_gt == 0:
        return {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "mean_matched_iou": 1.0,
            "pred_count": 0,
            "reviewed_count": 0,
            "count_error": 0,
        }

    if n_pred == 0:
        return {
            "tp": 0,
            "fp": 0,
            "fn": n_gt,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "mean_matched_iou": 0.0,
            "pred_count": 0,
            "reviewed_count": n_gt,
            "count_error": -n_gt,
        }

    if n_gt == 0:
        return {
            "tp": 0,
            "fp": n_pred,
            "fn": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "mean_matched_iou": 0.0,
            "pred_count": n_pred,
            "reviewed_count": 0,
            "count_error": n_pred,
        }

    iou_matrix = np.zeros((n_pred, n_gt), dtype=np.float32)
    for i, pred_mask in enumerate(auto_masks):
        for j, gt_mask in enumerate(reviewed_masks):
            iou_matrix[i, j] = _iou(pred_mask, gt_mask)

    pred_idx, gt_idx = linear_sum_assignment(-iou_matrix)

    matched_ious = []
    for i, j in zip(pred_idx, gt_idx):
        if iou_matrix[i, j] >= iou_threshold:
            matched_ious.append(float(iou_matrix[i, j]))

    tp = len(matched_ious)
    fp = n_pred - tp
    fn = n_gt - tp

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_matched_iou": float(np.mean(matched_ious)) if matched_ious else 0.0,
        "pred_count": n_pred,
        "reviewed_count": n_gt,
        "count_error": n_pred - n_gt,
    }
