#!/usr/bin/env python3
"""Compare EV size distributions across multiple morphology CSV files.

Creates publication-ready size comparison plots including:
- density curves (KDE)
- ECDF curves
- violin plots
- summary statistics CSV

Example
-------
python -m visualization.compare_size_profiles \
  --input "MDA-MB-231=C:\\path\\to\\morphology_all.csv" \
  --input "SKOV=C:\\path\\to\\morphology_all.csv" \
  --input "HEK=C:\\path\\to\\morphology_all.csv" \
  --output-dir "C:\\path\\to\\size_plots"
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde


STYLE = {
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "figure.titlesize": 18,
}

COLORS = ["#4C78A8", "#E45756", "#54A24B", "#B279A2", "#F58518"]


def parse_input_spec(spec: str) -> Tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Invalid --input '{spec}'. Expected LABEL=PATH")
    label, path = spec.split("=", 1)
    label = label.strip()
    csv_path = Path(path.strip()).expanduser().resolve()
    if not label:
        raise ValueError(f"Invalid label in --input '{spec}'")
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")
    return label, csv_path


def load_metric_values(csv_path: Path, metric: str, scale_factor: float = 1.0) -> List[float]:
    values: List[float] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            raw = row.get(metric, "")
            if raw in {None, "", "None"}:
                continue
            try:
                value = float(raw)
            except ValueError:
                continue
            if np.isfinite(value):
                values.append(value * scale_factor)
    return values


def save_figure(fig: plt.Figure, base_path: Path) -> None:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".svg"), bbox_inches="tight")


def estimate_mode(values: List[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError("Cannot estimate mode for empty values")
    if arr.size == 1 or np.allclose(arr, arr[0]):
        return float(arr[0])

    grid = np.linspace(float(arr.min()), float(arr.max()), 512)
    kde = gaussian_kde(arr)
    return float(grid[np.argmax(kde(grid))])


def summarize(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "mode_estimate": estimate_mode(values),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "q1": float(np.percentile(arr, 25)),
        "q3": float(np.percentile(arr, 75)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def plot_histogram(ax: plt.Axes, series: Dict[str, List[float]], xlabel: str) -> None:
    all_values = np.concatenate([np.asarray(v, dtype=float) for v in series.values() if v])
    xmin = max(0.0, float(all_values.min()) * 0.9)
    xmax = float(all_values.max()) * 1.05
    bins = np.linspace(xmin, xmax, 28)

    for idx, (label, values) in enumerate(series.items()):
        arr = np.asarray(values, dtype=float)
        color = COLORS[idx % len(COLORS)]
        ax.hist(arr, bins=bins, alpha=0.4, edgecolor="black", linewidth=0.8, label=f"{label} (n={arr.size})", color=color)

    ax.set_title("Binned EV counts")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("EV count")
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=False)


def plot_violin(ax: plt.Axes, series: Dict[str, List[float]], xlabel: str) -> None:
    labels = list(series.keys())
    data = [np.asarray(series[label], dtype=float) for label in labels]

    parts = ax.violinplot(data, showmeans=False, showmedians=True, showextrema=True)
    for idx, body in enumerate(parts["bodies"]):
        body.set_facecolor(COLORS[idx % len(COLORS)])
        body.set_edgecolor("black")
        body.set_alpha(0.4)

    for key in ("cbars", "cmins", "cmaxes", "cmedians"):
        if key in parts:
            parts[key].set_edgecolor("black")
            parts[key].set_linewidth(1.0)

    ax.set_xticks(np.arange(1, len(labels) + 1), labels)

    rng = np.random.default_rng(42)
    for idx, values in enumerate(data, start=1):
        if values.size == 0:
            continue
        jitter = rng.uniform(-0.08, 0.08, size=values.size)
        point_color = COLORS[(idx - 1) % len(COLORS)]
        ax.scatter(
            np.full(values.size, idx) + jitter,
            values,
            s=11,
            alpha=0.32,
            color=point_color,
            edgecolors="black",
            linewidths=0.15,
            zorder=3,
        )

    ax.set_title("Violin plot + individual EVs")
    ax.set_xlabel("EV line")
    ax.set_ylabel(xlabel)
    ax.grid(True, axis="y", alpha=0.2)


def format_stats_block(summary_rows: List[Dict[str, float | str]]) -> str:
    lines = ["Per-line summary (nm)"]
    for row in summary_rows:
        lines.append(
            f"{row['line']}: n={row['n']} | μ={row['mean']:.1f} | med={row['median']:.1f} | mode≈{row['mode_estimate']:.1f}"
        )
    return "\n".join(lines)


def add_stats_box(ax: plt.Axes, summary_rows: List[Dict[str, float | str]], *, x: float, y: float, ha: str) -> None:
    ax.text(
        x,
        y,
        format_stats_block(summary_rows),
        transform=ax.transAxes,
        ha=ha,
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.82, "edgecolor": "#999999"},
    )


def write_summary_csv(path: Path, summary_rows: List[Dict[str, float | str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)


def write_summary_txt(path: Path, summary_rows: List[Dict[str, float | str]]) -> None:
    lines = ["EV size summary", ""]
    for row in summary_rows:
        lines.extend([
            f"{row['line']}",
            f"  n = {row['n']}",
            f"  mean = {row['mean']:.2f}",
            f"  median = {row['median']:.2f}",
            f"  mode_estimate = {row['mode_estimate']:.2f}",
            f"  IQR = ({row['q1']:.2f}, {row['q3']:.2f})",
            f"  range = ({row['min']:.2f}, {row['max']:.2f})",
            "",
        ])
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, help="Repeated LABEL=PATH to morphology_all.csv")
    parser.add_argument("--metric", default="equivalent_diameter", help="Morphology metric to compare")
    parser.add_argument("--metric-label", default="Equivalent diameter (px)", help="Axis label for the chosen metric")
    parser.add_argument("--scale-factor", type=float, default=1.0, help="Optional multiplier to convert the metric values before plotting (e.g. nm/px)")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for plots and summary CSV")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    series: Dict[str, List[float]] = {}
    summary_rows: List[Dict[str, float | str]] = []

    for spec in args.input:
        label, csv_path = parse_input_spec(spec)
        values = load_metric_values(csv_path, metric=args.metric, scale_factor=args.scale_factor)
        if not values:
            raise ValueError(f"No valid `{args.metric}` values found in {csv_path}")
        series[label] = values
        stats = summarize(values)
        summary_rows.append({
            "line": label,
            "metric": args.metric,
            "scale_factor": args.scale_factor,
            **stats,
            "source_csv": str(csv_path),
        })

    write_summary_csv(output_dir / f"{args.metric}_summary.csv", summary_rows)
    write_summary_txt(output_dir / f"{args.metric}_summary.txt", summary_rows)

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), constrained_layout=True)
        plot_histogram(axes[0], series, xlabel=args.metric_label)
        plot_violin(axes[1], series, xlabel=args.metric_label)
        add_stats_box(axes[0], summary_rows, x=0.98, y=0.50, ha="right")
        fig.suptitle(f"EV size comparison: {args.metric.replace('_', ' ').title()}")
        save_figure(fig, output_dir / f"{args.metric}_comparison_overview")
        plt.close(fig)

        fig2, ax2 = plt.subplots(figsize=(6.5, 5.4), constrained_layout=True)
        plot_violin(ax2, series, xlabel=args.metric_label)
        add_stats_box(ax2, summary_rows, x=0.02, y=0.98, ha="left")
        save_figure(fig2, output_dir / f"{args.metric}_violin")
        plt.close(fig2)

    print(f"Saved plots to: {output_dir}")
    for row in summary_rows:
        print(
            f"- {row['line']}: n={row['n']}, mean={row['mean']:.2f}, median={row['median']:.2f}, "
            f"mode≈{row['mode_estimate']:.2f}, IQR=({row['q1']:.2f}, {row['q3']:.2f})"
        )


if __name__ == "__main__":
    main()
