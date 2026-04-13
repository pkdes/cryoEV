#!/usr/bin/env python3
"""Regenerate poster-friendly YOLO training/validation figures with larger fonts.

Outputs editable/vector copies (`.svg`, `.pdf`) plus high-resolution `.png` files that
open cleanly in Illustrator.

Example
-------
python -m visualization.export_yolo_poster_figures \
    --run-dir "C:\\Users\\ML-2619\\Desktop\\Pujan Cryo\\cryo-ev pipeline\\Model Training by Yifei\\round_2\\results_yolov8_heavy_augmentation\\training\\vesicle_instance_seg_v2"
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.ndimage import gaussian_filter1d
from ultralytics import YOLO


STYLE = {
    "font.size": 18,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
    "figure.titlesize": 24,
}


def export_figure(fig: plt.Figure, base_path: Path) -> None:
    """Save the same figure as PNG, PDF, and SVG."""
    base_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".svg"), bbox_inches="tight")


def load_results_csv(results_csv: Path) -> Dict[str, np.ndarray]:
    with results_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise ValueError(f"No rows found in {results_csv}")

    data: Dict[str, np.ndarray] = {}
    for key in reader.fieldnames or []:
        values = []
        for row in rows:
            raw = row.get(key, "")
            try:
                values.append(float(raw))
            except (TypeError, ValueError):
                values.append(np.nan)
        data[key] = np.asarray(values, dtype=float)
    return data


def prettify_metric_name(name: str) -> str:
    replacements = {
        "train/": "Train ",
        "val/": "Val ",
        "metrics/": "",
        "(B)": " (Box)",
        "(M)": " (Mask)",
        "box_loss": "Box Loss",
        "seg_loss": "Seg Loss",
        "cls_loss": "Cls Loss",
        "dfl_loss": "DFL Loss",
        "precision": "Precision",
        "recall": "Recall",
        "mAP50-95": "mAP50-95",
        "mAP50": "mAP50",
        "lr/pg0": "LR pg0",
        "lr/pg1": "LR pg1",
        "lr/pg2": "LR pg2",
    }
    pretty = name
    for old, new in replacements.items():
        pretty = pretty.replace(old, new)
    return pretty


def plot_training_results(results_csv: Path, out_dir: Path) -> None:
    """Regenerate the YOLO `results.png` panel with larger fonts and vector export."""
    data = load_results_csv(results_csv)
    x = data.get("epoch")
    if x is None:
        raise KeyError(f"`epoch` column not found in {results_csv}")

    loss_keys = [k for k in data if "loss" in k]
    metric_keys = [k for k in data if "metrics/" in k]
    lr_keys = [k for k in data if k.startswith("lr/")]

    columns = loss_keys[: len(loss_keys) // 2] + metric_keys[: len(metric_keys) // 2]
    columns += loss_keys[len(loss_keys) // 2 :] + metric_keys[len(metric_keys) // 2 :]
    if lr_keys:
        columns += lr_keys[:1]

    n_cols = 4
    n_rows = int(np.ceil(len(columns) / n_cols))

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 4.2 * n_rows), constrained_layout=True)
        axes = np.atleast_1d(axes).ravel()

        for ax, key in zip(axes, columns):
            y = data[key]
            valid = np.isfinite(y)
            if not valid.any():
                ax.set_visible(False)
                continue

            ax.plot(x[valid], y[valid], color="#1f77b4", marker="o", markersize=3.5, linewidth=2.2, label="result")
            if valid.sum() >= 5:
                smooth = gaussian_filter1d(y[valid], sigma=2)
                ax.plot(x[valid], smooth, color="#ff7f0e", linestyle="--", linewidth=2.4, label="smooth")

            ax.set_title(prettify_metric_name(key), pad=10)
            ax.set_xlabel("Epoch")
            ax.grid(True, alpha=0.25)
            if "precision" in key or "recall" in key or "mAP" in key:
                ax.set_ylim(0, 1.02)
            if key.startswith("lr/"):
                ax.set_yscale("log")

        for ax in axes[len(columns):]:
            ax.set_visible(False)

        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
        fig.suptitle(f"Training Metrics: {results_csv.parent.name}", y=1.02)
        export_figure(fig, out_dir / "results_poster")
        plt.close(fig)


def plot_confusion_matrix(matrix: np.ndarray, class_names: List[str], out_dir: Path, normalize: bool) -> None:
    """Create a poster-sized confusion matrix export with larger annotation text."""
    matrix = np.asarray(matrix, dtype=float)
    if normalize:
        col_sums = matrix.sum(axis=0, keepdims=True) + 1e-9
        display = matrix / col_sums
        display[display < 0.005] = np.nan
        title = "Confusion Matrix Normalized"
        annot_fmt = lambda v: f"{v:.2f}"
        stem = "confusion_matrix_normalized_poster"
    else:
        display = matrix.copy()
        display[display == 0] = np.nan
        title = "Confusion Matrix Counts"
        annot_fmt = lambda v: f"{int(round(v))}"
        stem = "confusion_matrix_counts_poster"

    labels = [*class_names, "background"]

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8.5, 7.2), constrained_layout=True)
        im = ax.imshow(display, cmap="Blues", vmin=0)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=13)

        for i in range(display.shape[0]):
            for j in range(display.shape[1]):
                val = display[i, j]
                if np.isnan(val):
                    continue
                color = "white" if val > (0.45 * np.nanmax(display)) else "black"
                ax.text(j, i, annot_fmt(val), ha="center", va="center", fontsize=18, fontweight="bold", color=color)

        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=0)
        ax.set_yticklabels(labels)
        ax.set_xlabel("True label")
        ax.set_ylabel("Predicted label")
        ax.set_title(title, pad=12)
        export_figure(fig, out_dir / stem)
        plt.close(fig)


def curve_slug(prefix: str, xlabel: str, ylabel: str) -> str:
    label = f"{prefix}_{ylabel.lower()}_{xlabel.lower()}"
    label = label.replace("-", "_").replace(" ", "_")
    while "__" in label:
        label = label.replace("__", "_")
    return label


def plot_validation_curves(metrics, out_dir: Path) -> None:
    """Export all box/mask validation curves from a fresh `model.val()` run."""
    names = list(metrics.names.values()) if isinstance(metrics.names, dict) else list(metrics.names)
    curve_titles = list(metrics.curves)

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 4, figsize=(22, 10), constrained_layout=True)
        axes = axes.ravel()

        for idx, (ax, item, title) in enumerate(zip(axes, metrics.curves_results, curve_titles)):
            x, y, xlabel, ylabel = item
            x = np.asarray(x)
            y = np.asarray(y)
            if y.ndim == 1:
                y = y[np.newaxis, :]

            prefix = "box" if idx < 4 else "mask"
            solo_fig, solo_ax = plt.subplots(figsize=(8.8, 6.2), constrained_layout=True)
            for class_idx, class_y in enumerate(y):
                class_name = names[class_idx] if class_idx < len(names) else f"class_{class_idx}"
                ax.plot(x, class_y, linewidth=2.6, label=class_name)
                solo_ax.plot(x, class_y, linewidth=3.0, label=class_name)

            solo_ax.set_xlim(0, 1)
            solo_ax.set_ylim(0, 1.02)
            solo_ax.set_xlabel(xlabel)
            solo_ax.set_ylabel(ylabel)
            solo_ax.set_title(title)
            solo_ax.grid(True, alpha=0.25)
            solo_ax.legend(frameon=False)
            export_figure(solo_fig, out_dir / curve_slug(prefix, xlabel, ylabel))
            plt.close(solo_fig)

            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.02)
            ax.grid(True, alpha=0.25)
            ax.legend(frameon=False)

        export_figure(fig, out_dir / "validation_curves_poster")
        plt.close(fig)


def write_readme(out_dir: Path, validation_dir: Path | None) -> None:
    lines = [
        "Poster-friendly YOLO figure exports",
        "",
        "Editable/vector files are provided as `.svg` and `.pdf` for Illustrator.",
        "High-resolution `.png` copies are also included.",
        "",
        "Key files:",
        "- results_poster.*",
        "- validation_curves_poster.*",
        "- confusion_matrix_normalized_poster.*",
        "- confusion_matrix_counts_poster.*",
    ]
    if validation_dir is not None:
        lines += ["", f"Ultralytics refresh outputs: {validation_dir}"]
    (out_dir / "README.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_local_dataset_yaml(run_dir: Path, dataset_dir: Path) -> Path:
    names = {0: "vesicle"}
    source_yaml = dataset_dir / "dataset.yaml"
    if source_yaml.exists():
        loaded = yaml.safe_load(source_yaml.read_text(encoding="utf-8")) or {}
        names = loaded.get("names", names)

    payload = {
        "path": str(dataset_dir),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": names,
        "nc": len(names),
    }
    out_yaml = run_dir / "poster_exports" / "dataset_local.yaml"
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    out_yaml.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return out_yaml


def regenerate_validation_artifacts(run_dir: Path, dataset_dir: Path, device: str, imgsz: int):
    weights = run_dir / "weights" / "best.pt"
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")

    local_yaml = build_local_dataset_yaml(run_dir, dataset_dir)
    project_dir = run_dir / "poster_exports"
    model = YOLO(str(weights))
    metrics = model.val(
        data=str(local_yaml),
        split="val",
        imgsz=imgsz,
        batch=4,
        device=device,
        plots=True,
        project=str(project_dir),
        name="validation_refresh",
        exist_ok=True,
    )
    return metrics, project_dir / "validation_refresh"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path, help="Path to the YOLO run folder containing results.csv")
    parser.add_argument("--dataset-dir", type=Path, default=None, help="Optional local dataset directory override")
    parser.add_argument("--device", default="0", help="YOLO validation device (default: 0)")
    parser.add_argument("--imgsz", type=int, default=1024, help="Validation image size")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    results_csv = run_dir / "results.csv"
    if not results_csv.exists():
        raise FileNotFoundError(f"results.csv not found under {run_dir}")

    dataset_dir = args.dataset_dir.resolve() if args.dataset_dir else run_dir.parents[1] / "dataset"
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    out_dir = run_dir / "poster_exports" / "editable"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Creating poster exports in: {out_dir}")
    plot_training_results(results_csv, out_dir)

    metrics, validation_dir = regenerate_validation_artifacts(run_dir, dataset_dir, device=args.device, imgsz=args.imgsz)
    plot_validation_curves(metrics, out_dir)
    plot_confusion_matrix(metrics.confusion_matrix.matrix, list(metrics.names.values()), out_dir, normalize=False)
    plot_confusion_matrix(metrics.confusion_matrix.matrix, list(metrics.names.values()), out_dir, normalize=True)
    write_readme(out_dir, validation_dir)

    print("Done. Generated files:")
    for path in sorted(out_dir.iterdir()):
        print(f" - {path.name}")


if __name__ == "__main__":
    main()
