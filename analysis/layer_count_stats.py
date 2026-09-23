"""
Layer-count / grouping stats for multilayer EVs, using the containment-derived
`layer_count` from classify_roles() (number of polygons geometrically nested
inside each 'multilayer'-classified outer boundary).

Computed on BOTH:
  - ground truth (all 31 images, train+valid) -- the reliable reference
  - Run 14's cached predictions -- to see whether the model's detections
    recover a similar layer-count distribution to the true one

No new inference: predictions are read from the cache written by
run_experiments.py / multilayer_role_analysis.py
(<run_dir>/predictions/<split>/*.txt).

Outputs under training outputs/roboflow_20260713_all_layer_singleclass/layer_count_stats/:
  multilayer_objects_gt.csv / _pred.csv   -- one row per multilayer EV object
  layer_count_histogram.png               -- GT vs pred distribution
  size_vs_layer_count.png                 -- outer-polygon area vs layer count
  summary.txt
"""

import argparse
import csv
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from multilayer_containment import load_image_annotations, classify_roles

sys.path.insert(0, str(Path(__file__).parent.parent / "training"))
from train_yolo import load_predictions_yolo_format

ROOT = Path(__file__).parent.parent.parent / "CryoAI"

# All overridden by CLI args in the __main__ block below; module-level
# defaults here match the original 20260713 campaign for backward compat.
SOURCE_ROOT = ROOT / "training images" / "roboflow - 20260713 all layer label"
DATASET = ROOT / "training outputs" / "roboflow_20260713_all_layer_singleclass"
MODEL_RUN_DIR = DATASET / "runs" / "2026-07-13_181034_y8_1440_24b_300ep_Run_14"
OUT_ROOT = DATASET / "layer_count_stats"

CONTAINMENT_THRESHOLD = 0.99
MAX_SIZE_RATIO = 0.75  # same combo validated in multilayer_role_analysis.py

SPLIT_MAP = {"train": "train", "valid": "val"}
GT_CATEGORY_EXCLUDE = {"non-EV", "items-nfaR"}


def equiv_diameter_px(area_px: float) -> float:
    return 2.0 * np.sqrt(area_px / np.pi)


def collect_gt_records():
    records = []
    for coco_split, yolo_split in SPLIT_MAP.items():
        coco_path = SOURCE_ROOT / coco_split / "_annotations.coco.json"
        for img, entries_all in load_image_annotations(coco_path):
            entries = [(aid, cat, m) for aid, cat, m in entries_all if cat not in GT_CATEGORY_EXCLUDE]
            if not entries:
                continue
            masks = [m for _, _, m in entries]
            ids = [aid for aid, _, _ in entries]
            role, layer_count = classify_roles(masks, CONTAINMENT_THRESHOLD, MAX_SIZE_RATIO)
            for idx, aid in enumerate(ids):
                if role[idx] != "multilayer":
                    continue
                area = int(masks[idx].sum())
                records.append({
                    "split": yolo_split, "image": img["file_name"], "id": aid,
                    "layer_count": layer_count[idx], "area_px": area,
                    "equiv_diameter_px": round(equiv_diameter_px(area), 2),
                })
    return records


def collect_pred_records():
    records = []
    for coco_split, yolo_split in SPLIT_MAP.items():
        coco_path = SOURCE_ROOT / coco_split / "_annotations.coco.json"
        img_dir = DATASET / "images" / yolo_split
        cache_dir = MODEL_RUN_DIR / "predictions" / yolo_split

        for img, entries_all in load_image_annotations(coco_path):
            img_path = img_dir / img["file_name"]
            cache_path = cache_dir / f"{img_path.stem}.txt"
            if not img_path.exists() or not cache_path.exists():
                continue
            w, h = img["width"], img["height"]
            pred_masks, confidences, pred_classes = load_predictions_yolo_format(str(cache_path), w, h)
            if not pred_masks:
                continue
            role, layer_count = classify_roles(pred_masks, CONTAINMENT_THRESHOLD, MAX_SIZE_RATIO)
            for idx in range(len(pred_masks)):
                if role[idx] != "multilayer":
                    continue
                area = int(pred_masks[idx].sum())
                records.append({
                    "split": yolo_split, "image": img["file_name"], "id": idx,
                    "layer_count": layer_count[idx], "area_px": area,
                    "equiv_diameter_px": round(equiv_diameter_px(area), 2),
                })
    return records


def save_csv(records, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "image", "id", "layer_count", "area_px", "equiv_diameter_px"])
        writer.writeheader()
        writer.writerows(records)


def summarize(records, label: str) -> str:
    if not records:
        return f"{label}: no multilayer objects found\n"
    counts = [r["layer_count"] for r in records]
    lines = [
        f"{label}: n={len(records)} multilayer objects",
        f"  layer_count: mean={np.mean(counts):.2f} median={np.median(counts):.1f} "
        f"min={min(counts)} max={max(counts)}",
    ]
    dist = Counter(counts)
    lines.append("  distribution (layer_count -> n_objects):")
    for k in sorted(dist):
        lines.append(f"    {k:>3} layers: {dist[k]}")
    if len(records) >= 2:
        areas = [r["area_px"] for r in records]
        corr = np.corrcoef(areas, counts)[0, 1]
        lines.append(f"  Pearson correlation (outer area vs layer_count): {corr:.3f}")
    return "\n".join(lines) + "\n"


def plot_histogram(gt_records, pred_records, out_path: Path):
    gt_counts = [r["layer_count"] for r in gt_records]
    pred_counts = [r["layer_count"] for r in pred_records]
    max_layers = max(gt_counts + pred_counts) if (gt_counts or pred_counts) else 1
    bins = np.arange(1, max_layers + 2) - 0.5

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist([gt_counts, pred_counts], bins=bins, label=[f"Ground truth (n={len(gt_counts)})",
                                                          f"Predictions (n={len(pred_counts)})"],
            color=["#1f77b4", "#ff7f0e"], alpha=0.85)
    ax.set_xlabel("Layer count (# polygons nested inside the outer boundary)")
    ax.set_ylabel("Number of multilayer EV objects")
    ax.set_title("Multilayer EV layer-count distribution: GT vs. predictions")
    ax.set_xticks(range(1, max_layers + 1))
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_size_vs_layers(gt_records, pred_records, out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 6))
    if gt_records:
        ax.scatter([r["equiv_diameter_px"] for r in gt_records], [r["layer_count"] for r in gt_records],
                   alpha=0.6, label="Ground truth", color="#1f77b4")
    if pred_records:
        ax.scatter([r["equiv_diameter_px"] for r in pred_records], [r["layer_count"] for r in pred_records],
                   alpha=0.6, label="Predictions", color="#ff7f0e")
    ax.set_xlabel("Outer boundary equivalent diameter (px)")
    ax.set_ylabel("Layer count")
    ax.set_title("Multilayer EV size vs. layer count")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    print("Collecting ground-truth multilayer objects...")
    gt_records = collect_gt_records()
    print(f"  {len(gt_records)} found")

    print("Collecting cached prediction multilayer objects...")
    pred_records = collect_pred_records()
    print(f"  {len(pred_records)} found")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    save_csv(gt_records, OUT_ROOT / "multilayer_objects_gt.csv")
    save_csv(pred_records, OUT_ROOT / "multilayer_objects_pred.csv")

    present_splits = sorted({r["split"] for r in gt_records} | {r["split"] for r in pred_records})
    summary = ""
    for split_label in [*present_splits, "combined"]:
        if split_label == "combined":
            gt_sub, pred_sub = gt_records, pred_records
        else:
            gt_sub = [r for r in gt_records if r["split"] == split_label]
            pred_sub = [r for r in pred_records if r["split"] == split_label]
        summary += summarize(gt_sub, f"GROUND TRUTH ({split_label.upper()})") + "\n"
        summary += summarize(pred_sub, f"PREDICTIONS ({split_label.upper()})") + "\n"

    print("\n" + summary)
    (OUT_ROOT / "summary.txt").write_text(summary, encoding="utf-8")

    plot_histogram(gt_records, pred_records, OUT_ROOT / "layer_count_histogram.png")
    plot_size_vs_layers(gt_records, pred_records, OUT_ROOT / "size_vs_layer_count.png")

    print(f"[DONE] All outputs under: {OUT_ROOT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="roboflow_20260713_all_layer_singleclass",
                         help="Folder name under 'training outputs/'")
    parser.add_argument("--source-name", default="roboflow - 20260713 all layer label",
                         help="Folder name under 'training images/' holding the raw Roboflow COCO export")
    parser.add_argument("--model-run-dir", default="2026-07-13_181034_y8_1440_24b_300ep_Run_14",
                         help="Run folder name under '<dataset>/runs/' whose cached predictions to use")
    cli_args = parser.parse_args()

    SOURCE_ROOT = ROOT / "training images" / cli_args.source_name
    DATASET = ROOT / "training outputs" / cli_args.dataset
    MODEL_RUN_DIR = DATASET / "runs" / cli_args.model_run_dir
    OUT_ROOT = DATASET / "layer_count_stats"

    SPLIT_MAP = {"train": "train", "valid": "val"}
    if (DATASET / "images" / "test").exists():
        SPLIT_MAP["test"] = "test"

    main()
