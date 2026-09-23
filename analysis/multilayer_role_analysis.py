"""
Full role classification (multilayer / inner-layer / standalone) on a trained
model's predictions, validated against ground truth as a proper confusion
matrix -- extends multilayer_containment_predictions.py with:
  - a (containment_threshold, max_size_ratio) sweep, reusing one cached
    inference pass (inference is run exactly once per image)
  - a 4x4 confusion matrix (3 real classes + background/FN/FP), image + text
  - per-class P/R/F1 for ALL three classes, including plain "standalone" EVs
  - role-colored prediction overlays (correct vs. misclassified outline)
  - cropped failure-mode galleries, one folder per failure category

Outputs all saved under:
  training outputs/roboflow_20260713_all_layer_singleclass/multilayer_role_analysis/
    threshold_sweep.csv
    confusion_matrix_<tag>.png / .txt          (for the chosen combo)
    overlays/<split>/*.png                     (role-colored, chosen combo)
    failure_crops/<category>/*.png             (chosen combo)
"""

import argparse
import sys
import shutil
from pathlib import Path
from collections import defaultdict
import csv

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "training"))
from train_yolo import (
    load_predictions_with_classes_from_model, match_objects_hungarian,
    load_predictions_yolo_format, save_predictions_yolo_format,
)

sys.path.insert(0, str(Path(__file__).parent))
from multilayer_containment import load_image_annotations, classify_roles

ROOT = Path(__file__).parent.parent.parent / "CryoAI"

# All overridden by CLI args in the __main__ block below; module-level
# defaults here match the original 20260713 campaign for backward compat.
SOURCE_ROOT = ROOT / "training images" / "roboflow - 20260713 all layer label"
DATASET = ROOT / "training outputs" / "roboflow_20260713_all_layer_singleclass"
MODEL_PATH = DATASET / "runs" / "2026-07-13_181034_y8_1440_24b_300ep_Run_14" / "weights" / "best.pt"
OUT_ROOT = DATASET / "multilayer_role_analysis"
# Cache directory convention matches run_experiments.py's export_predictions_for_split():
# <run_dir>/predictions/<split>/<image_stem>.txt.
PRED_CACHE_DIR = MODEL_PATH.parent.parent / "predictions"

IMGSZ = 1440
CONF = 0.25
NMS_IOU = 0.7
MATCH_IOU = 0.5

SPLIT_MAP = {"train": "train", "valid": "val"}  # coco split -> yolo dataset split dir
GT_CATEGORY_EXCLUDE = {"non-EV", "items-nfaR"}  # not real EV instances, excluded from the role question

CLASSES = ["multilayer", "inner-layer", "standalone"]
ROLE_COLOR = {
    "multilayer": np.array([255, 140, 0]),   # orange
    "inner-layer": np.array([170, 0, 220]),  # purple
    "standalone": np.array([0, 180, 255]),   # blue
}
CORRECT_COLOR = np.array([0, 220, 0])   # green outline = matches GT role
WRONG_COLOR = np.array([255, 40, 40])   # red outline = misclassified vs GT role
HALLUC_COLOR = np.array([80, 80, 80])   # gray = prediction with no matching real object at all


# NOTE: the ground-truth "true role" is derived geometrically from GT masks
# via classify_roles() -- the SAME function used for predictions below -- not
# from the flat annotator category (MV/inner/EV/...). A middle layer of a
# 3+-layer stack is annotator-labeled "inner" but is geometrically BOTH
# contained (inner-layer) AND containing (multilayer); classify_roles()
# correctly resolves that to "multilayer" (checks "contains something" first)
# for both GT and predictions, so the comparison is apples-to-apples. Using
# the flat category instead would score ~89/299 correct GT detections as
# errors purely because the 2-tier MV/inner scheme can't express a 3rd tier.


# ---------------------------------------------------------------------------
# Step 1: run inference ONCE per image, cache everything the sweep needs.
# ---------------------------------------------------------------------------

def build_cache():
    cache = []
    n_from_cache = n_from_inference = 0
    for coco_split, yolo_split in SPLIT_MAP.items():
        coco_path = SOURCE_ROOT / coco_split / "_annotations.coco.json"
        img_dir = DATASET / "images" / yolo_split
        cache_dir = PRED_CACHE_DIR / yolo_split

        for img, gt_entries_all in tqdm(list(load_image_annotations(coco_path)), desc=f"Loading predictions ({coco_split})"):
            img_path = img_dir / img["file_name"]
            if not img_path.exists():
                continue
            gt_entries = [(aid, cat, m) for aid, cat, m in gt_entries_all if cat not in GT_CATEGORY_EXCLUDE]
            if not gt_entries:
                continue

            w, h = img["width"], img["height"]
            cache_path = cache_dir / f"{img_path.stem}.txt"
            if cache_path.exists():
                pred_masks, confidences, pred_classes = load_predictions_yolo_format(str(cache_path), w, h)
                n_from_cache += 1
            else:
                pred_masks, confidences, pred_classes = load_predictions_with_classes_from_model(
                    str(MODEL_PATH), str(img_path), IMGSZ, CONF, NMS_IOU, device='0'
                )
                save_predictions_yolo_format(pred_masks, confidences, pred_classes, w, h, cache_path)
                n_from_inference += 1
            gt_masks = [m for _, _, m in gt_entries]
            matches, unmatched_preds, unmatched_gts = match_objects_hungarian(
                pred_masks, gt_masks, iou_threshold=MATCH_IOU
            )
            cache.append({
                "split": yolo_split, "image": img["file_name"], "img_path": img_path,
                "pred_masks": pred_masks, "gt_entries": gt_entries,
                "matches": matches, "unmatched_preds": unmatched_preds, "unmatched_gts": unmatched_gts,
            })
    print(f"Predictions loaded from cache: {n_from_cache}, freshly inferred (and cached for next time): {n_from_inference}")
    return cache


# ---------------------------------------------------------------------------
# Step 2: role classification + confusion matrix for one (threshold, ratio) combo.
# ---------------------------------------------------------------------------

def evaluate(cache, containment_threshold, max_size_ratio, collect_details=False):
    """Confusion[true_label][pred_label] with 'background' for FN (missed) / FP (hallucinated).

    Both true (GT) and predicted roles come from the identical classify_roles()
    geometry, evaluated at the SAME (containment_threshold, max_size_ratio) --
    only then is the comparison fair.
    """
    labels = CLASSES + ["background"]
    confusion = {t: {p: 0 for p in labels} for t in labels}
    details = defaultdict(list) if collect_details else None  # category -> list of (record, gt_idx/pred_idx)

    for rec in cache:
        role, layers = classify_roles(rec["pred_masks"], containment_threshold, max_size_ratio)
        gt_masks = [m for _, _, m in rec["gt_entries"]]
        gt_role, _ = classify_roles(gt_masks, containment_threshold, max_size_ratio)

        for pred_idx, gt_idx in rec["matches"]:
            t = gt_role[gt_idx]
            p = role[pred_idx]
            confusion[t][p] += 1
            if collect_details and t != p:
                details[f"{t}_as_{p}"].append((rec, gt_idx, pred_idx))

        for gt_idx in rec["unmatched_gts"]:
            t = gt_role[gt_idx]
            confusion[t]["background"] += 1
            if collect_details:
                details[f"{t}_undetected"].append((rec, gt_idx, None))

        for pred_idx in rec["unmatched_preds"]:
            p = role[pred_idx]
            confusion["background"][p] += 1
            if collect_details:
                details[f"hallucinated_{p}"].append((rec, None, pred_idx))

    per_class = {}
    for c in CLASSES:
        tp = confusion[c][c]
        fp = sum(confusion[t][c] for t in labels if t != c)
        fn = sum(confusion[c][p] for p in labels if p != c)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[c] = {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}

    return confusion, per_class, details


# ---------------------------------------------------------------------------
# Step 3: sweep
# ---------------------------------------------------------------------------

def run_sweep(cache):
    thresholds = [0.80, 0.85, 0.90, 0.95, 0.99]
    size_ratios = [None, 0.75, 0.50]

    rows = []
    for thr in thresholds:
        for ratio in size_ratios:
            _, per_class, _ = evaluate(cache, thr, ratio)
            avg_f1 = sum(per_class[c]["f1"] for c in CLASSES) / len(CLASSES)
            row = {"containment_threshold": thr, "max_size_ratio": ratio if ratio else "none", "avg_f1": avg_f1}
            for c in CLASSES:
                row[f"{c}_P"] = per_class[c]["precision"]
                row[f"{c}_R"] = per_class[c]["recall"]
                row[f"{c}_F1"] = per_class[c]["f1"]
            rows.append(row)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_ROOT / "threshold_sweep.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[OK] Sweep saved to {csv_path}")

    print(f"\n{'thr':>5} {'size_ratio':>10} {'ml_P':>6} {'ml_R':>6} {'ml_F1':>6}  "
          f"{'in_P':>6} {'in_R':>6} {'in_F1':>6}  {'st_P':>6} {'st_R':>6} {'st_F1':>6}  {'avgF1':>6}")
    for r in rows:
        print(f"{r['containment_threshold']:>5} {str(r['max_size_ratio']):>10} "
              f"{r['multilayer_P']:>6.3f} {r['multilayer_R']:>6.3f} {r['multilayer_F1']:>6.3f}  "
              f"{r['inner-layer_P']:>6.3f} {r['inner-layer_R']:>6.3f} {r['inner-layer_F1']:>6.3f}  "
              f"{r['standalone_P']:>6.3f} {r['standalone_R']:>6.3f} {r['standalone_F1']:>6.3f}  {r['avg_f1']:>6.3f}")

    best = max(rows, key=lambda r: r["avg_f1"])
    return best, rows


# ---------------------------------------------------------------------------
# Step 4: confusion matrix image + text for the chosen combo
# ---------------------------------------------------------------------------

def save_confusion_matrix(confusion, tag: str):
    labels = CLASSES + ["background"]
    mat = np.array([[confusion[t][p] for p in labels] for t in labels])

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(mat, cmap="Blues")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted role"); ax.set_ylabel("True role")
    ax.set_title(f"Multilayer role confusion matrix ({tag})\n"
                 f"rows/cols 'background' = missed (FN) / hallucinated (FP)")
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = mat[i, j]
            if i == len(labels) - 1 and j == len(labels) - 1:
                continue  # background-background is not meaningful
            ax.text(j, i, str(val), ha="center", va="center",
                     color="white" if val > mat.max() / 2 else "black", fontsize=11)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    png_path = OUT_ROOT / f"confusion_matrix_{tag}.png"
    plt.savefig(png_path, dpi=150)
    plt.close()

    txt_path = OUT_ROOT / f"confusion_matrix_{tag}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Multilayer role confusion matrix ({tag})\n")
        f.write("rows=true, cols=predicted; 'background' = missed(FN)/hallucinated(FP)\n\n")
        header = " " * 14 + "".join(f"{l:>13}" for l in labels)
        f.write(header + "\n")
        for t in labels:
            f.write(f"{t:>13} " + "".join(f"{confusion[t][p]:>13}" for p in labels) + "\n")
    print(f"[OK] Confusion matrix saved to {png_path} and {txt_path}")
    return png_path


# ---------------------------------------------------------------------------
# Step 5: role-colored overlays for the chosen combo
# ---------------------------------------------------------------------------

def _draw_outline(overlay, mask, color, thickness=3):
    mask_u8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, color.tolist(), thickness=thickness)


def generate_overlays(cache, containment_threshold, max_size_ratio, tag: str, clear: bool = True):
    out_dir = OUT_ROOT / "overlays"
    if clear and out_dir.exists():
        shutil.rmtree(out_dir)  # clear stale files from a prior run/combo before regenerating
    for rec in tqdm(cache, desc=f"Generating role overlays ({tag})"):
        role, layers = classify_roles(rec["pred_masks"], containment_threshold, max_size_ratio)
        gt_masks = [m for _, _, m in rec["gt_entries"]]
        gt_role, _ = classify_roles(gt_masks, containment_threshold, max_size_ratio)
        gt_of_pred = {pi: gi for pi, gi in rec["matches"]}

        raw = np.array(Image.open(rec["img_path"]).convert("RGB"))
        overlay = raw.copy().astype(float)

        for i, mask in enumerate(rec["pred_masks"]):
            fill_color = ROLE_COLOR[role[i]]
            overlay[mask] = overlay[mask] * 0.75 + fill_color * 0.25
            if i in gt_of_pred:
                t = gt_role[gt_of_pred[i]]
                outline = CORRECT_COLOR if t == role[i] else WRONG_COLOR
            else:
                outline = HALLUC_COLOR
            _draw_outline(overlay, mask, outline)

        split_dir = out_dir / rec["split"]
        split_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(12, 8.5))
        ax.imshow(overlay.astype(np.uint8))
        ax.set_title(f"{rec['image']}  (containment_thr={containment_threshold}, max_size_ratio={max_size_ratio})",
                     fontsize=9)
        ax.axis("off")
        patches = (
            [Patch(facecolor=ROLE_COLOR[c] / 255, label=f"fill: {c}") for c in CLASSES] +
            [Patch(edgecolor=CORRECT_COLOR / 255, facecolor="none", linewidth=2, label="outline: correct"),
             Patch(edgecolor=WRONG_COLOR / 255, facecolor="none", linewidth=2, label="outline: misclassified"),
             Patch(edgecolor=HALLUC_COLOR / 255, facecolor="none", linewidth=2, label="outline: hallucinated (no GT match)")]
        )
        ax.legend(handles=patches, loc="upper right", fontsize=7, framealpha=0.8)
        plt.tight_layout()
        plt.savefig(split_dir / f"{Path(rec['image']).stem}_roles.png", dpi=150, bbox_inches="tight")
        plt.close()
    print(f"[OK] Role overlays saved under {out_dir}")


# ---------------------------------------------------------------------------
# Step 6: cropped failure-mode galleries
# ---------------------------------------------------------------------------

def generate_failure_crops(cache, containment_threshold, max_size_ratio, pad=150):
    out_dir = OUT_ROOT / "failure_crops"
    if out_dir.exists():
        shutil.rmtree(out_dir)  # clear stale files from a prior run/combo before regenerating
    _, _, details = evaluate(cache, containment_threshold, max_size_ratio, collect_details=True)

    for category, items in details.items():
        cat_dir = out_dir / category
        cat_dir.mkdir(parents=True, exist_ok=True)
        for n, (rec, gt_idx, pred_idx) in enumerate(items):
            role, _ = classify_roles(rec["pred_masks"], containment_threshold, max_size_ratio)
            raw = np.array(Image.open(rec["img_path"]).convert("RGB"))
            h, w = raw.shape[:2]
            overlay = raw.copy().astype(float)

            focus_mask = None
            if gt_idx is not None:
                _, cat, gmask = rec["gt_entries"][gt_idx]
                _draw_outline(overlay, gmask, np.array([255, 40, 40]), thickness=4)  # red = GT
                focus_mask = gmask
            if pred_idx is not None:
                pmask = rec["pred_masks"][pred_idx]
                _draw_outline(overlay, pmask, ROLE_COLOR[role[pred_idx]], thickness=3)  # role color = prediction
                focus_mask = pmask if focus_mask is None else focus_mask

            ys, xs = np.where(focus_mask)
            cy, cx = int(ys.mean()), int(xs.mean())
            y0, y1 = max(0, cy - pad), min(h, cy + pad)
            x0, x1 = max(0, cx - pad), min(w, cx + pad)
            crop = overlay.astype(np.uint8)[y0:y1, x0:x1]

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(crop)
            ax.set_title(f"{category}\n{rec['image']}", fontsize=8)
            ax.axis("off")
            plt.tight_layout()
            plt.savefig(cat_dir / f"{n:03d}_{Path(rec['image']).stem}.png", dpi=150, bbox_inches="tight")
            plt.close()
        print(f"  {category}: {len(items)} example(s) -> {cat_dir}")

    print(f"[OK] Failure-mode crops saved under {out_dir}")


def main():
    print(f"Source:      {SOURCE_ROOT}")
    print(f"Dataset:     {DATASET}")
    print(f"Model:       {MODEL_PATH}")
    print(f"Output root: {OUT_ROOT}\n")

    cache = build_cache()
    print(f"\nCached {len(cache)} images with predictions + GT matches.\n")

    print("=" * 100)
    print("THRESHOLD / SIZE-RATIO SWEEP (combined across all splits)")
    print("=" * 100)
    best, rows = run_sweep(cache)
    print(f"\nBest combo by average F1: containment_threshold={best['containment_threshold']}, "
          f"max_size_ratio={best['max_size_ratio']} (avg_f1={best['avg_f1']:.4f})")

    thr = best["containment_threshold"]
    ratio = None if best["max_size_ratio"] == "none" else best["max_size_ratio"]
    base_tag = f"thr{thr}_ratio{best['max_size_ratio']}"

    out_overlays = OUT_ROOT / "overlays"
    if out_overlays.exists():
        shutil.rmtree(out_overlays)  # clear once; per-split calls below append with clear=False

    present_splits = sorted({rec["split"] for rec in cache})
    for split_label in [*present_splits, "combined"]:
        sub_cache = cache if split_label == "combined" else [r for r in cache if r["split"] == split_label]
        if not sub_cache:
            continue
        tag = f"{base_tag}_{split_label}"

        confusion, per_class, _ = evaluate(sub_cache, thr, ratio)
        print(f"\n=== Per-class P/R/F1 -- {split_label.upper()} (n={len(sub_cache)} images) ===")
        for c in CLASSES:
            m = per_class[c]
            print(f"  {c:>12}: P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f} "
                  f"(TP={m['tp']} FP={m['fp']} FN={m['fn']})")

        save_confusion_matrix(confusion, tag)
        generate_overlays(sub_cache, thr, ratio, tag, clear=False)

    # Failure-mode crops: pooled across all splits (diagnostic examples, not a
    # per-split reporting artifact).
    generate_failure_crops(cache, thr, ratio)

    print(f"\n[DONE] All outputs under: {OUT_ROOT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="roboflow_20260713_all_layer_singleclass",
                         help="Folder name under 'training outputs/'")
    parser.add_argument("--source-name", default="roboflow - 20260713 all layer label",
                         help="Folder name under 'training images/' holding the raw Roboflow COCO export")
    parser.add_argument("--model-run-dir", default="2026-07-13_181034_y8_1440_24b_300ep_Run_14",
                         help="Run folder name under '<dataset>/runs/' whose weights/best.pt to evaluate")
    cli_args = parser.parse_args()

    SOURCE_ROOT = ROOT / "training images" / cli_args.source_name
    DATASET = ROOT / "training outputs" / cli_args.dataset
    MODEL_PATH = DATASET / "runs" / cli_args.model_run_dir / "weights" / "best.pt"
    OUT_ROOT = DATASET / "multilayer_role_analysis"
    PRED_CACHE_DIR = MODEL_PATH.parent.parent / "predictions"

    SPLIT_MAP = {"train": "train", "valid": "val"}
    if (DATASET / "images" / "test").exists():
        SPLIT_MAP["test"] = "test"

    main()
