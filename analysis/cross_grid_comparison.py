r"""
Generic Multi-Dataset Inference and Comparison
==============================================
Runs YOLO instance segmentation across one or more datasets, tags each EV
instance with dataset metadata, and generates combined CSV outputs plus
comparison plots.

Default behavior remains compatible with the previous grid workflow:
- If no input directories are provided, it scans IMAGES_DIR.
- Dataset IDs are inferred from filename patterns like grid4_... when present.
- Otherwise, dataset IDs fall back to parent folder names.

You can also:
- Provide explicit dataset directories
- Use folder names as dataset labels
- Provide labels from CLI
- Prompt labels via GUI dialog boxes

Examples:
    python analysis/cross_grid_comparison.py

    python analysis/cross_grid_comparison.py \
      --input-dirs "D:\exp\setA" "D:\exp\setB" \
      --dataset-inference directory

    python analysis/cross_grid_comparison.py \
      --input-dirs "D:\exp\all_images" \
      --dataset-inference regex \
      --dataset-regex "(sample\\d+)" \
      --gui-labels
"""

import argparse
import csv
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from analysis.morphology import analyze_instances
from inference.inference import extract_instances_yolo
from inference.perf_log import PerformanceLogger

# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------
IMAGES_DIR = Path(
    r"C:\Users\ML-2619\Desktop\Pujan Cryo\20260430 - PLL-film exp analysis\images"
)
MODEL_WEIGHTS = Path(
    r"C:\Users\ML-2619\Desktop\Pujan Cryo\cryo-ev pipeline\Model Training by Yifei"
    r"\round_2\results_yolov8_heavy_augmentation\training\vesicle_instance_seg_v2\weights\best.pt"
)
OUTPUT_DIR = Path(
    r"C:\Users\ML-2619\Desktop\Pujan Cryo\20260430 - PLL-film exp analysis\cross_grid_results"
)

IMGSZ = 1024
CONF = 0.25
IOU = 0.7
DEVICE = "cpu"
SAVE_OVERLAYS = True
PIXEL_SIZE: Optional[float] = None

# Backward-compatible mapping for known grid IDs.
GRID_CONDITIONS: Dict[int, str] = {
    4: "Type A, no PLL",
    5: "Type B, no PLL",
    6: "Type A, PLL",
    7: "Type B, PLL",
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
DEFAULT_GRID_REGEX = r"grid(\d+)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run generic multi-dataset EV inference and comparisons.",
    )
    parser.add_argument(
        "--input-dirs",
        nargs="+",
        default=None,
        help="One or more input directories to scan recursively.",
    )
    parser.add_argument(
        "--dataset-inference",
        choices=["auto", "directory", "parent", "grid", "regex"],
        default="auto",
        help=(
            "How dataset IDs are inferred per image: "
            "auto (grid then parent), directory (input dir name), "
            "parent (image parent folder), grid (grid# in filename), "
            "regex (custom regex on filename)."
        ),
    )
    parser.add_argument(
        "--dataset-regex",
        default=DEFAULT_GRID_REGEX,
        help="Regex used when dataset-inference=regex (or for grid parsing).",
    )
    parser.add_argument(
        "--dataset-labels",
        nargs="+",
        default=None,
        help="Optional labels in sorted dataset ID order (must match dataset count).",
    )
    parser.add_argument(
        "--gui-labels",
        action="store_true",
        help="Prompt for dataset labels with a small Tkinter GUI dialog.",
    )
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Output directory.")
    parser.add_argument("--model-path", default=str(MODEL_WEIGHTS), help="YOLO weights path.")
    parser.add_argument("--imgsz", type=int, default=IMGSZ)
    parser.add_argument("--conf", type=float, default=CONF)
    parser.add_argument("--iou", type=float, default=IOU)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--pixel-size", type=float, default=PIXEL_SIZE)
    parser.add_argument(
        "--save-overlays",
        dest="save_overlays",
        action="store_true",
        default=SAVE_OVERLAYS,
        help="Save per-image segmentation overlays.",
    )
    parser.add_argument(
        "--no-save-overlays",
        dest="save_overlays",
        action="store_false",
        help="Disable saving per-image segmentation overlays.",
    )
    return parser.parse_args()


def _is_reject_path(path: Path) -> bool:
    return any(part.lower() == "rejects" for part in path.parts)


def _grid_id_from_name(name: str, regex: str = DEFAULT_GRID_REGEX) -> Optional[int]:
    m = re.search(regex, name, re.IGNORECASE)
    if not m:
        return None
    grp = m.group(1) if m.groups() else m.group(0)
    try:
        return int(grp)
    except ValueError:
        return None


def _safe_id(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]+", "_", text.strip())
    return cleaned.strip("_") or "dataset"


def infer_dataset_id(path: Path, root: Path, mode: str, regex: str) -> str:
    filename = path.name

    if mode == "directory":
        return _safe_id(root.name)

    if mode == "parent":
        return _safe_id(path.parent.name)

    if mode == "grid":
        gid = _grid_id_from_name(filename, DEFAULT_GRID_REGEX)
        return f"grid{gid}" if gid is not None else _safe_id(path.parent.name)

    if mode == "regex":
        m = re.search(regex, filename, re.IGNORECASE)
        if m:
            token = m.group(1) if m.groups() else m.group(0)
            return _safe_id(token)
        return _safe_id(path.parent.name)

    # auto: prefer grid token, otherwise parent folder name
    gid = _grid_id_from_name(filename, DEFAULT_GRID_REGEX)
    if gid is not None:
        return f"grid{gid}"
    return _safe_id(path.parent.name)


def collect_images(
    input_dirs: List[Path],
    inference_mode: str,
    dataset_regex: str,
) -> Dict[str, List[Path]]:
    dataset_images: Dict[str, List[Path]] = {}
    for root in input_dirs:
        for path in sorted(root.rglob("*")):
            if _is_reject_path(path):
                continue
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            dataset_id = infer_dataset_id(path, root, inference_mode, dataset_regex)
            dataset_images.setdefault(dataset_id, []).append(path)
    return dataset_images


def resolve_dataset_labels(
    dataset_ids: List[str],
    cli_labels: Optional[List[str]],
    gui_labels: bool,
) -> Dict[str, str]:
    if cli_labels is not None:
        if len(cli_labels) != len(dataset_ids):
            raise ValueError(
                f"--dataset-labels expected {len(dataset_ids)} values, got {len(cli_labels)}."
            )
        return dict(zip(dataset_ids, cli_labels))

    defaults: Dict[str, str] = {}
    for did in dataset_ids:
        gid = _grid_id_from_name(did, r"grid(\d+)$")
        if gid is not None and gid in GRID_CONDITIONS:
            defaults[did] = GRID_CONDITIONS[gid]
        else:
            defaults[did] = did

    if not gui_labels:
        return defaults

    try:
        import tkinter as tk
        from tkinter import simpledialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)

        resolved: Dict[str, str] = {}
        for did in dataset_ids:
            prompt = (
                "Enter display label for dataset:\n"
                f"{did}\n\n"
                "Leave blank to keep default."
            )
            answer = simpledialog.askstring(
                "Dataset Label",
                prompt,
                initialvalue=defaults[did],
                parent=root,
            )
            resolved[did] = answer.strip() if answer and answer.strip() else defaults[did]

        root.destroy()
        return resolved
    except Exception:
        print("GUI label prompt unavailable; using default labels.")
        return defaults


def save_detection_overlay(
    image_path: Path,
    masks: List[np.ndarray],
    confidences: List[float],
    out_dir: Path,
    alpha: float = 0.35,
) -> Optional[Path]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return None

    base = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    overlay = base.copy()

    for idx, mask in enumerate(masks):
        if mask.shape[:2] != base.shape[:2]:
            continue

        color = np.array(
            [(37 * idx + 90) % 255, (71 * idx + 130) % 255, (109 * idx + 170) % 255],
            dtype=np.uint8,
        )
        overlay[mask] = color

        mask_u8 = (mask.astype(np.uint8)) * 255
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(overlay, contours, -1, (255, 255, 255), 1)

    blended = cv2.addWeighted(base, 1.0 - alpha, overlay, alpha, 0.0)
    mean_conf = float(np.mean(confidences)) if confidences else 0.0
    label = f"Detections: {len(masks)} | Mean conf: {mean_conf:.3f}"
    cv2.putText(
        blended,
        label,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{image_path.stem}_overlay.png"
    cv2.imwrite(str(out_path), cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
    return out_path


def run_inference_all_datasets(
    dataset_images: Dict[str, List[Path]],
    dataset_labels: Dict[str, str],
    perf: PerformanceLogger,
    model_path: Path,
    imgsz: int,
    conf: float,
    iou: float,
    device: str,
    pixel_size: Optional[float],
    overlay_root: Optional[Path],
) -> Tuple[List[Dict], int]:
    all_records: List[Dict] = []
    n_processed = 0
    global_obj_id = 0

    for dataset_id in sorted(dataset_images.keys()):
        images = dataset_images[dataset_id]
        label = dataset_labels.get(dataset_id, dataset_id)
        print(f"\n[Dataset {dataset_id} | {label}] - {len(images)} image(s)")

        for img_path in images:
            print(f"  Inferring: {img_path.name} ...", end=" ", flush=True)
            try:
                t0 = time.perf_counter()
                masks, confs, _bboxes, polys = extract_instances_yolo(
                    model_path=str(model_path),
                    img_path=str(img_path),
                    imgsz=imgsz,
                    conf=conf,
                    iou=iou,
                    device=device,
                )
                inference_s = time.perf_counter() - t0
            except Exception as exc:
                print(f"ERROR ({exc})")
                continue

            perf.log_image(
                image_path=img_path,
                n_raw_detections=len(masks),
                inference_time_s=inference_s,
                image=None,
            )
            n_processed += 1

            records = analyze_instances(
                instance_masks=masks,
                confidences=confs,
                pixel_size=pixel_size,
                polygon_points_list=polys,
            )

            if overlay_root is not None:
                save_detection_overlay(
                    image_path=img_path,
                    masks=masks,
                    confidences=confs,
                    out_dir=overlay_root / dataset_id,
                )

            print(f"{len(records)} detections")

            for rec in records:
                rec["dataset_id"] = dataset_id
                rec["dataset_label"] = label
                rec["condition"] = label
                rec["image_name"] = img_path.name
                rec["global_object_id"] = global_obj_id
                gid = _grid_id_from_name(dataset_id, r"grid(\d+)$")
                rec["grid_id"] = gid if gid is not None else ""
                global_obj_id += 1
                all_records.append(rec)

    return all_records, n_processed


def save_combined_csv(records: List[Dict], out_path: Path) -> None:
    if not records:
        print("No records - CSV not written.")
        return

    meta_cols = [
        "global_object_id",
        "dataset_id",
        "dataset_label",
        "grid_id",
        "condition",
        "image_name",
        "object_id",
        "confidence",
    ]
    morph_cols = [k for k in records[0].keys() if k not in meta_cols]
    fieldnames = [c for c in meta_cols if c in records[0]] + morph_cols

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    print(f"\nCombined CSV saved -> {out_path} ({len(records)} rows)")


def save_per_image_summary(records: List[Dict], out_path: Path) -> None:
    if not records:
        print("No records - summary not written.")
        return

    by_image: Dict[Tuple[str, str], Dict] = {}
    for rec in records:
        key = (rec.get("dataset_id", "unknown"), rec.get("image_name", "unknown"))
        if key not in by_image:
            by_image[key] = {
                "dataset_id": rec.get("dataset_id", "unknown"),
                "dataset_label": rec.get("dataset_label", "unknown"),
                "image_name": rec.get("image_name", "unknown"),
                "detection_count": 0,
            }
        by_image[key]["detection_count"] += 1

    rows = sorted(by_image.values(), key=lambda r: (str(r["dataset_id"]), str(r["image_name"])))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["dataset_id", "dataset_label", "image_name", "detection_count"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\nDetections per image:")
    print("-" * 110)
    print(f"{'Dataset':<18} {'Label':<24} {'Image Name':<58} {'Count':>8}")
    print("-" * 110)
    for row in rows:
        print(
            f"{str(row['dataset_id']):<18} "
            f"{str(row['dataset_label']):<24} "
            f"{str(row['image_name']):<58} "
            f"{int(row['detection_count']):>8}"
        )
    print("-" * 110)
    print(f"\nPer-image summary saved -> {out_path}")


def _records_by_group(records: List[Dict], group_key: str) -> Dict[str, List[Dict]]:
    groups: Dict[str, List[Dict]] = {}
    for rec in records:
        key = str(rec.get(group_key, "unknown"))
        groups.setdefault(key, []).append(rec)
    return groups


def _extract(records: List[Dict], key: str) -> np.ndarray:
    return np.array([r[key] for r in records if r.get(key) is not None], dtype=float)


def _make_color_map(keys: List[str], cmap_name: str = "tab20") -> Dict[str, str]:
    cmap = plt.get_cmap(cmap_name)
    n = max(len(keys), 1)
    return {k: mcolors.to_hex(cmap(i / n)) for i, k in enumerate(keys)}


def plot_per_dataset_distributions(records: List[Dict], out_dir: Path) -> None:
    by_dataset = _records_by_group(records, "dataset_id")
    dataset_ids = sorted(by_dataset.keys())
    colors = _make_color_map(dataset_ids)

    metrics = [
        ("equivalent_diameter", "Equivalent Diameter (px)", (0, None)),
        ("circularity", "Circularity", (0, 1)),
        ("major_axis", "Major Axis (px)", (0, None)),
        ("aspect_ratio", "Aspect Ratio", (1, None)),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Per-dataset morphology distributions", fontsize=14)

    for ax, (key, label, xlim) in zip(axes.flat, metrics):
        for did in dataset_ids:
            vals = _extract(by_dataset[did], key)
            if len(vals) == 0:
                continue
            ax.hist(vals, bins=30, alpha=0.5, label=did, color=colors[did], edgecolor="none")
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        ax.set_title(label)
        if xlim[0] is not None:
            ax.set_xlim(left=xlim[0])
        if xlim[1] is not None:
            ax.set_xlim(right=xlim[1])
        ax.legend(fontsize=8)

    fig.tight_layout()
    path = out_dir / "per_dataset_distributions.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved -> {path}")


def plot_condition_distributions(records: List[Dict], out_dir: Path) -> None:
    by_cond = _records_by_group(records, "condition")
    cond_ids = sorted(by_cond.keys())
    colors = _make_color_map(cond_ids)

    metrics = [
        ("equivalent_diameter", "Equivalent Diameter (px)"),
        ("circularity", "Circularity"),
        ("major_axis", "Major Axis (px)"),
        ("aspect_ratio", "Aspect Ratio"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Condition distributions", fontsize=14)

    for ax, (key, label) in zip(axes.flat, metrics):
        for cond in cond_ids:
            vals = _extract(by_cond[cond], key)
            if len(vals) == 0:
                continue
            ax.hist(
                vals,
                bins=30,
                alpha=0.6,
                label=cond,
                color=colors[cond],
                edgecolor="none",
                density=True,
            )
        ax.set_xlabel(label)
        ax.set_ylabel("Density")
        ax.set_title(label)
        ax.legend(fontsize=8)

    fig.tight_layout()
    path = out_dir / "condition_distributions.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved -> {path}")


def plot_count_summary(records: List[Dict], dataset_images: Dict[str, List[Path]], out_dir: Path) -> None:
    by_dataset = _records_by_group(records, "dataset_id")
    dataset_ids = sorted(dataset_images.keys())
    colors = _make_color_map(dataset_ids)

    total_counts = [len(by_dataset.get(did, [])) for did in dataset_ids]
    n_images = [len(dataset_images[did]) for did in dataset_ids]
    mean_counts = [tot / n if n > 0 else 0.0 for tot, n in zip(total_counts, n_images)]

    labels = []
    for did in dataset_ids:
        sample = by_dataset.get(did, [{}])[0]
        dlabel = str(sample.get("dataset_label", did))
        labels.append(f"{did}\n({dlabel})")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("EV counts per dataset", fontsize=14)

    bars1 = ax1.bar(labels, total_counts, color=[colors[d] for d in dataset_ids])
    ax1.set_ylabel("Total detections")
    ax1.set_title("Total detections")
    for bar, val in zip(bars1, total_counts):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, str(val), ha="center", va="bottom", fontsize=9)

    bars2 = ax2.bar(labels, mean_counts, color=[colors[d] for d in dataset_ids])
    ax2.set_ylabel("Mean detections / image")
    ax2.set_title("Mean detections per image")
    for bar, val in zip(bars2, mean_counts):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{val:.1f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    path = out_dir / "detection_counts.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved -> {path}")


def plot_condition_boxplots(records: List[Dict], out_dir: Path) -> None:
    by_cond = _records_by_group(records, "condition")
    cond_labels = sorted(by_cond.keys())
    colors = _make_color_map(cond_labels)

    metrics = [
        ("equivalent_diameter", "Equivalent Diameter (px)"),
        ("circularity", "Circularity"),
        ("major_axis", "Major Axis (px)"),
        ("aspect_ratio", "Aspect Ratio"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    fig.suptitle("Condition boxplots", fontsize=14)

    for ax, (key, label) in zip(axes.flat, metrics):
        data = [_extract(by_cond[c], key) for c in cond_labels]
        bp = ax.boxplot(data, tick_labels=cond_labels, patch_artist=True, medianprops={"color": "black", "linewidth": 2})
        for patch, cond in zip(bp["boxes"], cond_labels):
            patch.set_facecolor(colors[cond])
            patch.set_alpha(0.6)
        ax.set_ylabel(label)
        ax.set_title(label)

    fig.tight_layout()
    path = out_dir / "condition_boxplots.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved -> {path}")


def plot_per_dataset_boxplots(records: List[Dict], out_dir: Path) -> None:
    by_dataset = _records_by_group(records, "dataset_id")
    dataset_ids = sorted(by_dataset.keys())
    colors = _make_color_map(dataset_ids)

    metrics = [
        ("equivalent_diameter", "Equivalent Diameter (px)"),
        ("circularity", "Circularity"),
        ("major_axis", "Major Axis (px)"),
        ("aspect_ratio", "Aspect Ratio"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Per-dataset boxplots", fontsize=14)

    for ax, (key, label) in zip(axes.flat, metrics):
        data = [_extract(by_dataset[d], key) for d in dataset_ids]
        bp = ax.boxplot(data, tick_labels=dataset_ids, patch_artist=True, medianprops={"color": "black", "linewidth": 2})
        for patch, did in zip(bp["boxes"], dataset_ids):
            patch.set_facecolor(colors[did])
            patch.set_alpha(0.6)
        ax.set_ylabel(label)
        ax.set_title(label)

    fig.tight_layout()
    path = out_dir / "per_dataset_boxplots.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved -> {path}")


def main() -> None:
    args = parse_args()

    input_dirs = [Path(p) for p in (args.input_dirs or [str(IMAGES_DIR)])]
    output_dir = Path(args.output_dir)
    model_path = Path(args.model_path)

    print("=" * 64)
    print("cryoEV multi-dataset inference and comparison")
    print("=" * 64)
    print("Input directories:")
    for d in input_dirs:
        print(f"  - {d}")
    print(f"Model:   {model_path}")
    print(f"Output:  {output_dir}")
    print(f"Mode:    {args.dataset_inference}")
    print()

    missing_dirs = [d for d in input_dirs if not d.exists()]
    if missing_dirs:
        missing_text = "\n".join(str(d) for d in missing_dirs)
        sys.exit(f"ERROR: input directory not found:\n{missing_text}")

    if not model_path.exists():
        sys.exit(f"ERROR: model weights not found:\n{model_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_images = collect_images(input_dirs, args.dataset_inference, args.dataset_regex)
    if not dataset_images:
        sys.exit("ERROR: no images found after filtering; check input folders and rejects layout.")

    dataset_ids = sorted(dataset_images.keys())
    dataset_labels = resolve_dataset_labels(dataset_ids, args.dataset_labels, args.gui_labels)

    print("Datasets found:")
    for did in dataset_ids:
        print(f"  {did} ({dataset_labels.get(did, did)}): {len(dataset_images[did])} images")

    n_images_total = sum(len(v) for v in dataset_images.values())
    run_id = datetime.now().strftime("multi_dataset_%Y%m%d_%H%M%S")

    perf = PerformanceLogger(output_dir=output_dir, session_id=run_id)
    perf.log_session_start(
        model_path=model_path,
        n_images=n_images_total,
        device=args.device,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
    )

    t0 = time.perf_counter()
    overlay_root = output_dir / "overlays" if args.save_overlays else None
    all_records, n_processed = run_inference_all_datasets(
        dataset_images=dataset_images,
        dataset_labels=dataset_labels,
        perf=perf,
        model_path=model_path,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        pixel_size=args.pixel_size,
        overlay_root=overlay_root,
    )
    perf.log_session_end(total_time_s=time.perf_counter() - t0, n_processed=n_processed)

    if not all_records:
        sys.exit("No detections returned; check model, thresholds, and images.")

    print(f"\nTotal detections: {len(all_records)}")

    save_combined_csv(all_records, output_dir / "all_detections.csv")
    save_per_image_summary(all_records, output_dir / "detections_per_image.csv")

    print("\nGenerating plots...")
    plot_per_dataset_distributions(all_records, output_dir)
    plot_condition_distributions(all_records, output_dir)
    plot_count_summary(all_records, dataset_images, output_dir)
    plot_condition_boxplots(all_records, output_dir)
    plot_per_dataset_boxplots(all_records, output_dir)

    print("\nDone. All outputs saved to:")
    print(f"  {output_dir}")


if __name__ == "__main__":
    main()
