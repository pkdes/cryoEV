"""
Aggregate per-image EV detection/morphology results into sample/square/area
level summaries and QC plots.

Reads the `morphology_all.csv` and `per_image_summary.csv` produced by
`inference.batch_size_profile` (or `analysis.cross_grid_comparison`), derives
sample/square/area grouping from the image filename, and produces:
  - objects_per_image.png/.csv          (object count per image, by sample)
  - size_violin_by_sample.png/.csv       (equivalent_diameter violin + points)
  - qc_by_square.png/.csv                (size + count QC, grouped by square)
  - qc_by_area.png/.csv                  (size + count QC, grouped by area)

Reuses plotting/summary helpers from cross_grid_comparison.py and
compare_size_profiles.py rather than re-implementing them.
"""

import argparse
import csv
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis.cross_grid_comparison import _make_color_map, _extract
from visualization.compare_size_profiles import (
    plot_violin, summarize, write_summary_csv, write_summary_txt, add_stats_box,
)

UNKNOWN = "unknown"


def parse_groups(image_name: str) -> Dict[str, str]:
    """Derive sample/square/area group keys from an image filename, e.g.
    'grid1_sq1_area1_hmmontage_sub0_0000.png' -> sample='1', square='1_sq1',
    area='1_sq1_area1'. '1_sq1_area1_hm1.png' parses the same way."""
    m_sample = re.search(r"(\d+)", image_name)
    sample = m_sample.group(1) if m_sample else UNKNOWN

    m_sq = re.search(r"sq(\d+)", image_name)
    m_area = re.search(r"area(\d+)", image_name)

    square = f"{sample}_sq{m_sq.group(1)}" if (m_sq and sample != UNKNOWN) else UNKNOWN
    area = f"{square}_area{m_area.group(1)}" if (m_area and square != UNKNOWN) else UNKNOWN

    return {"sample": sample, "square": square, "area": area}


def load_csv_rows(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def annotate_groups(rows: List[Dict], image_key: str) -> List[Dict]:
    for row in rows:
        row.update(parse_groups(row[image_key]))
    return rows


def write_group_summary_csv(path: Path, rows: List[Dict], group_col: str, metric: str) -> None:
    by_group: Dict[str, List[float]] = {}
    for row in rows:
        val = row.get(metric)
        if val in (None, "", "None"):
            continue
        by_group.setdefault(row[group_col], []).append(float(val))

    summary_rows = []
    for group, values in sorted(by_group.items()):
        stats = summarize(values)
        summary_rows.append({"group": group, "metric": metric, **stats})

    if summary_rows:
        write_summary_csv(path, summary_rows)
        print(f"Saved -> {path}")


def plot_objects_per_image(per_image_rows: List[Dict], out_dir: Path) -> None:
    samples = sorted({r["sample"] for r in per_image_rows})
    colors = _make_color_map(samples)

    # Sort within each sample for a readable bar order
    ordered = sorted(per_image_rows, key=lambda r: (r["sample"], r["image"]))
    counts = [int(r["n_accepted_objects"]) for r in ordered]
    bar_colors = [colors[r["sample"]] for r in ordered]

    fig, ax = plt.subplots(figsize=(max(10, len(ordered) * 0.12), 5.5))
    ax.bar(range(len(ordered)), counts, color=bar_colors, width=0.9)
    ax.set_xlabel("Image (grouped by sample)")
    ax.set_ylabel("Accepted objects per image")
    ax.set_title("EV objects detected per image")

    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[s]) for s in samples]
    ax.legend(handles, [f"Sample {s}" for s in samples], fontsize=9, loc="upper right")
    ax.set_xticks([])

    fig.tight_layout()
    path = out_dir / "objects_per_image.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved -> {path}")

    csv_path = out_dir / "objects_per_image.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sample", "square", "area", "image", "n_accepted_objects"])
        writer.writeheader()
        for r in ordered:
            writer.writerow({k: r[k] for k in writer.fieldnames})
    print(f"Saved -> {csv_path}")


def plot_objects_per_image_violin(per_image_rows: List[Dict], out_dir: Path) -> None:
    """Violin + individual points of accepted-object-count per image, grouped
    by sample -- one point per image, so skew/outlier images are visible."""
    samples = sorted({r["sample"] for r in per_image_rows})
    series: Dict[str, List[float]] = {}
    for s in samples:
        vals = [float(r["n_accepted_objects"]) for r in per_image_rows if r["sample"] == s]
        if vals:
            series[f"Sample {s}"] = vals

    if not series:
        print("No data for objects-per-image violin plot.")
        return

    summary_rows = []
    for label, values in series.items():
        stats = summarize(values)
        summary_rows.append({"line": label, "metric": "n_accepted_objects", "scale_factor": 1.0, **stats, "source_csv": ""})
    write_summary_csv(out_dir / "objects_per_image_by_sample_summary.csv", summary_rows)

    fig, ax = plt.subplots(figsize=(7.5, 5.6))
    plot_violin(ax, series, xlabel="Accepted objects per image")
    add_stats_box(ax, summary_rows, x=0.02, y=0.98, ha="left")
    fig.tight_layout()
    path = out_dir / "objects_per_image_violin_by_sample.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved -> {path}")


def plot_size_violin_by_sample(records: List[Dict], out_dir: Path, metric: str = "equivalent_diameter") -> None:
    samples = sorted({r["sample"] for r in records})
    series: Dict[str, List[float]] = {}
    for s in samples:
        vals = _extract([r for r in records if r["sample"] == s], metric)
        if len(vals) > 0:
            series[f"Sample {s}"] = list(vals)

    if not series:
        print("No data for size violin plot.")
        return

    summary_rows = []
    for label, values in series.items():
        stats = summarize(values)
        summary_rows.append({"line": label, "metric": metric, "scale_factor": 1.0, **stats, "source_csv": ""})
    write_summary_csv(out_dir / f"{metric}_by_sample_summary.csv", summary_rows)
    write_summary_txt(out_dir / f"{metric}_by_sample_summary.txt", summary_rows)

    fig, ax = plt.subplots(figsize=(7.5, 5.6))
    plot_violin(ax, series, xlabel=f"{metric.replace('_', ' ').title()} (nm)")
    add_stats_box(ax, summary_rows, x=0.02, y=0.98, ha="left")
    fig.tight_layout()
    path = out_dir / "size_violin_by_sample.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved -> {path}")


def plot_qc_panel(records: List[Dict], per_image_rows: List[Dict], group_col: str,
                   out_dir: Path, out_name: str) -> None:
    groups = sorted(g for g in {r[group_col] for r in records} if g != UNKNOWN)
    if not groups:
        print(f"No groups found for '{group_col}' QC panel.")
        return
    colors = _make_color_map(groups)

    size_data = [_extract([r for r in records if r[group_col] == g], "equivalent_diameter") for g in groups]
    count_data = [
        [int(r["n_accepted_objects"]) for r in per_image_rows if r[group_col] == g]
        for g in groups
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(10, len(groups) * 1.1), 5.5))
    fig.suptitle(f"QC by {group_col}", fontsize=14)

    bp1 = ax1.boxplot(size_data, tick_labels=groups, patch_artist=True,
                       medianprops={"color": "black", "linewidth": 2})
    for patch, g in zip(bp1["boxes"], groups):
        patch.set_facecolor(colors[g])
        patch.set_alpha(0.6)
    ax1.set_ylabel("Equivalent diameter (nm)")
    ax1.set_title("Size distribution")
    ax1.tick_params(axis="x", rotation=45)

    bp2 = ax2.boxplot(count_data, tick_labels=groups, patch_artist=True,
                       medianprops={"color": "black", "linewidth": 2})
    for patch, g in zip(bp2["boxes"], groups):
        patch.set_facecolor(colors[g])
        patch.set_alpha(0.6)
    ax2.set_ylabel("Accepted objects per image")
    ax2.set_title("Object count per image")
    ax2.tick_params(axis="x", rotation=45)

    fig.tight_layout()
    path = out_dir / f"{out_name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved -> {path}")

    write_group_summary_csv(out_dir / f"{out_name}_size_summary.csv", records, group_col, "equivalent_diameter")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--morphology-csv", required=True, type=Path)
    parser.add_argument("--per-image-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    records = annotate_groups(load_csv_rows(args.morphology_csv), image_key="image")
    per_image_rows = annotate_groups(load_csv_rows(args.per_image_csv), image_key="image")

    n_unknown = sum(1 for r in per_image_rows if r["sample"] == UNKNOWN)
    print(f"{len(per_image_rows)} images, {len(records)} objects; "
          f"{n_unknown} image(s) with no parseable sample id")

    plot_objects_per_image(per_image_rows, out_dir)
    plot_objects_per_image_violin(per_image_rows, out_dir)
    plot_size_violin_by_sample(records, out_dir)
    plot_qc_panel(records, per_image_rows, "square", out_dir, "qc_by_square")
    plot_qc_panel(records, per_image_rows, "area", out_dir, "qc_by_area")

    print(f"\nDone. Outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
