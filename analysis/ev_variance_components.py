#!/usr/bin/env python3
"""Variance decomposition and sampling-design analysis for EV detections.

This answers a different question from `aggregate_size_results.py`. That script
describes what was found per square and per area; this one asks how much of the
spread is real structure versus sampling noise, and therefore how many areas
must be imaged to characterize a grid.

The design is a two-level unbalanced nest: areas (images) within squares. For
each response -- EV abundance per image and EV equivalent diameter -- it
estimates

    sigma2_between   variance between squares
    sigma2_within    variance between areas inside a square
    ICC              sigma2_between / (sigma2_between + sigma2_within)

by method-of-moments from a one-way random-effects ANOVA, puts a CLUSTER
bootstrap interval on them (resampling squares, not images -- with only a
dozen-odd squares, resampling images would badly understate the uncertainty on
sigma2_between), and tests the between-square signal against a permutation null
built by shuffling the image-to-square assignment. That null framing follows
`embedding/evaluate.py`, the repo's existing precedent for "above chance is a
number".

Those two components then give the standard error of a grid-level mean estimated
from m squares x k areas each,

    SE = sqrt(sigma2_between / m + sigma2_within / (m * k))

which is the sampling rule. The same formula explains why, once
sigma2_within / k is small next to sigma2_between, extra areas inside a square
stop buying precision and extra squares are the only thing that helps.

Size DISTRIBUTION convergence is a separate question from the precision of its
mean, so it gets a separate, assumption-free treatment: subsample k areas,
compare the pooled diameter distribution to the full-data one by KS distance,
and report how many areas are needed before that distance stops shrinking.

Reads only CSVs -- no model load, per the repo's "run inference once, analyze
many times" rule.

    python analysis/ev_variance_components.py \
        --morphology-csv  .../morphology/morphology_all.csv \
        --per-image-csv   .../morphology/per_image_summary.csv \
        --output-dir      .../variance/v3_run1_20260824 \
        --image-area-um2  1.868 --label v3_run1_20260824
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis.aggregate_size_results import parse_groups
from visualization.compare_size_profiles import STYLE, COLORS, save_figure, summarize

# Relative-SE targets the sampling table is solved for.
SE_TARGETS = (0.10, 0.15, 0.20, 0.25)
# KS distance below which the pooled size distribution is called stable.
KS_TARGET = 0.10


# ── loading ──────────────────────────────────────────────────────────────────

def load_per_image(path: Path) -> List[Dict]:
    """Per-image rows tagged with their square, from per_image_summary.csv.

    Images whose square cannot be parsed off the filename are dropped loudly --
    a silent regex miss here would collapse groups and quietly invalidate every
    variance estimate downstream.
    """
    rows, skipped = [], []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            g = parse_groups(r["image"])
            if g["square"] == "unknown":
                skipped.append(r["image"])
                continue
            rows.append({
                "image": r["image"],
                "square": g["square"],
                "area": g["area"],
                "n_accepted": int(r["n_accepted_objects"] or 0),
                "mean_diameter": _f(r.get("mean_equivalent_diameter")),
                "median_diameter": _f(r.get("median_equivalent_diameter")),
            })
    if skipped:
        print(f"  WARNING: {len(skipped)} image(s) with unparseable square: "
              f"{', '.join(skipped[:5])}")
    return rows


def load_objects_by_image(path: Path, metric: str) -> Dict[str, List[float]]:
    """metric value per object, keyed by image. Tolerates the empty strings
    morphology.py writes when an ellipse fit fails."""
    out: Dict[str, List[float]] = defaultdict(list)
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            v = _f(r.get(metric))
            if v is not None and math.isfinite(v):
                out[r["image"]].append(v)
    return dict(out)


def _f(v):
    if v is None or v == "" or v == "None":
        return None
    try:
        return float(v)
    except ValueError:
        return None


# ── variance components ──────────────────────────────────────────────────────

def variance_components(groups: Dict[str, Sequence[float]]) -> Dict[str, float]:
    """One-way random-effects variance components, unbalanced, by ANOVA
    method-of-moments.

    `groups` maps a square id to that square's per-image values. Returns the two
    components, the ICC, the ANOVA F, and the design constants. sigma2_between is
    truncated at zero: the MoM estimator can go negative when the true between
    component is near zero, and a negative variance is not reportable.

    Calibration, measured on 1000 synthetic datasets using the real 20260727
    design (14 squares, 2-4 areas, N=45): both variance components come back
    essentially unbiased, and sigma2_within reproduces a directly computed pooled
    within-group variance exactly. ICC is biased LOW by about 0.02-0.03, being a
    ratio of two noisy estimates, and biased HIGH when the truth is near zero
    because of the truncation above -- an ICC of ~0.06 is what this estimator
    returns for a true ICC of 0.
    """
    keys = [k for k, v in groups.items() if len(v) > 0]
    k = len(keys)
    if k < 2:
        raise ValueError("need >= 2 groups for a variance decomposition")

    n_i = np.array([len(groups[key]) for key in keys], dtype=float)
    means = np.array([float(np.mean(groups[key])) for key in keys])
    N = float(n_i.sum())
    grand = float(sum(float(np.sum(groups[key])) for key in keys) / N)

    ss_between = float(np.sum(n_i * (means - grand) ** 2))
    ss_within = float(sum(float(np.sum((np.asarray(groups[key]) - means[i]) ** 2))
                          for i, key in enumerate(keys)))
    df_b, df_w = k - 1, N - k

    ms_between = ss_between / df_b
    ms_within = ss_within / df_w if df_w > 0 else float("nan")

    # Unbalanced-design effective group size.
    n0 = (N - float(np.sum(n_i ** 2)) / N) / df_b

    s2_within = ms_within
    s2_between_raw = (ms_between - ms_within) / n0
    s2_between = max(0.0, s2_between_raw)
    total = s2_between + s2_within
    icc = s2_between / total if total > 0 else 0.0

    return {
        "k_groups": k, "n_total": int(N), "n0": n0,
        "grand_mean": grand,
        "ms_between": ms_between, "ms_within": ms_within,
        "df_between": int(df_b), "df_within": int(df_w),
        "sigma2_between": s2_between,
        "sigma2_between_raw": s2_between_raw,
        "sigma2_within": s2_within,
        "sigma2_total": total,
        "icc": icc,
        "f_statistic": ms_between / ms_within if ms_within > 0 else float("nan"),
        "sd_between": math.sqrt(s2_between),
        "sd_within": math.sqrt(s2_within) if s2_within >= 0 else float("nan"),
        "cv_between": math.sqrt(s2_between) / grand if grand else float("nan"),
        "cv_within": math.sqrt(s2_within) / grand if grand else float("nan"),
    }


def cluster_bootstrap(groups: Dict[str, Sequence[float]], n_boot: int,
                      rng: np.random.Generator) -> Dict[str, Tuple[float, float]]:
    """Percentile CIs on the components, resampling SQUARES with replacement.

    Each drawn square carries all of its images, which is what keeps the
    within-square correlation intact. Draws that happen to yield fewer than two
    distinct squares are discarded.

    KNOWN LIMIT, measured: at this design (14 squares) the nominal 95% interval
    on ICC actually covers the truth about 85-90% of the time. Widening the
    quantiles barely helps (a nominal 99% interval reaches only ~91%), so the
    shortfall is intrinsic to having 14 clusters, not a quantile choice. Read
    these intervals as mildly optimistic rather than exact.
    """
    keys = list(groups)
    stats = defaultdict(list)
    for _ in range(n_boot):
        pick = rng.choice(len(keys), size=len(keys), replace=True)
        resampled = {f"b{j}": groups[keys[i]] for j, i in enumerate(pick)}
        try:
            vc = variance_components(resampled)
        except ValueError:
            continue
        for f in ("sigma2_between", "sigma2_within", "icc", "grand_mean",
                  "cv_between", "cv_within"):
            stats[f].append(vc[f])
    return {f: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
            for f, v in stats.items() if v}


def permutation_f(groups: Dict[str, Sequence[float]], n_perm: int,
                  rng: np.random.Generator) -> Dict[str, float]:
    """Permutation null for the ANOVA F: reshuffle which square each image
    belongs to, holding the group-size pattern fixed."""
    obs = variance_components(groups)["f_statistic"]
    sizes = [len(v) for v in groups.values()]
    pooled = np.concatenate([np.asarray(v, dtype=float) for v in groups.values()])

    null = []
    for _ in range(n_perm):
        shuffled = rng.permutation(pooled)
        cut, perm = 0, {}
        for i, s in enumerate(sizes):
            perm[f"p{i}"] = shuffled[cut:cut + s]
            cut += s
        null.append(variance_components(perm)["f_statistic"])
    null = np.asarray(null, dtype=float)
    null = null[np.isfinite(null)]

    p = (1.0 + float(np.sum(null >= obs))) / (1.0 + null.size)
    return {
        "f_observed": obs,
        "null_mean": float(null.mean()),
        "null_sd": float(null.std(ddof=1)),
        "z_vs_null": (obs - float(null.mean())) / float(null.std(ddof=1)),
        "p_permutation": p,
        "n_perm": int(null.size),
    }


# ── per-square table ─────────────────────────────────────────────────────────

def per_square_table(groups: Dict[str, Sequence[float]],
                     vc: Dict[str, float]) -> List[Dict]:
    """Per-square mean with a CI built from the POOLED within-square variance.

    A square's own SD is reported descriptively, but with n_i of 2-4 it is far
    too unstable to build an interval from, so the interval uses
    sqrt(sigma2_within / n_i) -- the pooled estimate, which has 31 df here
    instead of 1-3. Squares with n<3 are flagged rather than dropped.
    """
    s2w = vc["sigma2_within"]
    rows = []
    for key in sorted(groups, key=_square_sort_key):
        v = np.asarray(groups[key], dtype=float)
        se = math.sqrt(s2w / v.size) if s2w > 0 else 0.0
        own_sd = float(v.std(ddof=1)) if v.size > 1 else float("nan")
        mean = float(v.mean())
        rows.append({
            "square": key,
            "n_images": int(v.size),
            "mean": mean,
            "median": float(np.median(v)),
            "sd_own": own_sd,
            "cv_own": own_sd / mean if mean else float("nan"),
            "se_pooled": se,
            "ci_lo": mean - 1.96 * se,
            "ci_hi": mean + 1.96 * se,
            "delta_from_grid": mean - vc["grand_mean"],
            "low_n_flag": "n<3" if v.size < 3 else "",
        })
    return rows


def _square_sort_key(key: str):
    import re
    m = re.search(r"sq(\d+)", key)
    return (int(m.group(1)) if m else 0, key)


# ── sampling design ──────────────────────────────────────────────────────────

def analytic_se(s2_between: float, s2_within: float, m: int, k: int) -> float:
    """SE of the grid mean estimated as the unweighted mean of m square means,
    each from k areas."""
    return math.sqrt(s2_between / m + s2_within / (m * k))


def time_cost_table(vc: Dict[str, float], *, t_square: float, t_position: float,
                    t_expose: float, n_shift_values: Sequence[int],
                    m_values: Sequence[int], p_values: Sequence[int],
                    budget_s: Sequence[float]) -> List[Dict]:
    """Precision per HOUR, not per image.

    Microscope time is not spent per field of view. A new square costs
    eucentricity, a new stage position costs a move plus autofocus, and only the
    exposure itself is per-FOV -- so with beam-image shift many FOVs at one
    position are nearly free:

        T = m*t_square + m*p*t_position + m*p*n_shift*t_expose

    THE UNMEASURED QUANTITY, and the reason every row is reported as a bracket:
    a 4x4 beam-shift montage spans about 5.5 um at this field of view, while the
    within-square area separations this dataset actually sampled are 12.2-50.3
    um. Nothing here was measured at the beam-shift scale. So sigma2_within,
    which was estimated from stage-moved areas, does NOT transfer to tiled FOVs.
    Rather than guess, each design is evaluated at both extremes:

        optimistic  -- every FOV is an independent area, k = p * n_shift
        pessimistic -- all FOVs at one position collapse to one area, k = p

    The truth is in between and depends on the EV clustering length, which one
    4x4 montage per square in three or four squares would measure directly.
    A recommendation is only trustworthy where both brackets agree.
    """
    mean = vc["grand_mean"]
    s2b, s2w = vc["sigma2_between"], vc["sigma2_within"]
    rows = []
    for budget in budget_s:
        best: Dict[str, Dict] = {}
        for m in m_values:
            for p in p_values:
                for n in n_shift_values:
                    t = m * t_square + m * p * t_position + m * p * n * t_expose
                    if t > budget:
                        continue
                    for lbl, k in (("optimistic", p * n), ("pessimistic", p)):
                        rel = analytic_se(s2b, s2w, m, k) / mean if mean else float("nan")
                        if lbl not in best or rel < best[lbl]["rel_se"]:
                            best[lbl] = {
                                "budget_hours": budget / 3600.0,
                                "fov_independence": lbl,
                                "m_squares": m, "positions_per_square": p,
                                "fovs_per_position": n, "total_images": m * p * n,
                                "minutes_used": t / 60.0,
                                "effective_k_areas": k,
                                "rel_se": rel,
                            }
        for lbl in ("optimistic", "pessimistic"):
            if lbl in best:
                rows.append(best[lbl])
    return rows


def sampling_power_table(vc: Dict[str, float], ci: Dict[str, Tuple[float, float]],
                         m_values: Sequence[int], k_values: Sequence[int]) -> List[Dict]:
    """Relative SE over the (m, k) design grid, with a band from the bootstrap
    CI on the variance components."""
    mean = vc["grand_mean"]
    lo_b, hi_b = ci.get("sigma2_between", (vc["sigma2_between"],) * 2)
    lo_w, hi_w = ci.get("sigma2_within", (vc["sigma2_within"],) * 2)
    rows = []
    for m in m_values:
        for k in k_values:
            se = analytic_se(vc["sigma2_between"], vc["sigma2_within"], m, k)
            rows.append({
                "m_squares": m, "k_areas_per_square": k, "total_images": m * k,
                "se_abs": se,
                "rel_se": se / mean if mean else float("nan"),
                "rel_se_lo": analytic_se(lo_b, lo_w, m, k) / mean if mean else float("nan"),
                "rel_se_hi": analytic_se(hi_b, hi_w, m, k) / mean if mean else float("nan"),
                "ci_halfwidth_95_pct_of_mean":
                    1.96 * se / mean if mean else float("nan"),
            })
    return rows


def squares_required(vc: Dict[str, float], ci: Dict[str, Tuple[float, float]],
                     k_values: Sequence[int], targets: Sequence[float],
                     m_max: int = 200) -> List[Dict]:
    """Smallest m hitting each target relative SE, at the point estimate and at
    the pessimistic end of the bootstrap CI."""
    mean = vc["grand_mean"]
    hi_b = ci.get("sigma2_between", (vc["sigma2_between"],) * 2)[1]
    hi_w = ci.get("sigma2_within", (vc["sigma2_within"],) * 2)[1]

    def solve(s2b, s2w, k, target):
        for m in range(1, m_max + 1):
            if analytic_se(s2b, s2w, m, k) / mean <= target:
                return m
        return None

    rows = []
    for target in targets:
        for k in k_values:
            m = solve(vc["sigma2_between"], vc["sigma2_within"], k, target)
            m_cons = solve(hi_b, hi_w, k, target)
            rows.append({
                "target_rel_se": target,
                "k_areas_per_square": k,
                "m_squares_needed": m if m else f">{m_max}",
                "total_images": m * k if m else "",
                "m_squares_needed_conservative": m_cons if m_cons else f">{m_max}",
                "total_images_conservative": m_cons * k if m_cons else "",
            })
    return rows


def empirical_power(groups: Dict[str, Sequence[float]], m_values: Sequence[int],
                    k_values: Sequence[int], n_rep: int,
                    rng: np.random.Generator) -> List[Dict]:
    """Observed spread of the grid-mean estimate, by subsampling the real data.

    Draws m squares without replacement (only those with at least k images) and
    k images without replacement inside each, then forms the same estimator the
    analytic formula describes: the unweighted mean of square means. This is the
    check on whether that formula can be trusted on this dataset.
    """
    keys = list(groups)
    rows = []
    for k in k_values:
        eligible = [key for key in keys if len(groups[key]) >= k]
        if len(eligible) < 2:
            continue
        for m in m_values:
            if m > len(eligible):
                continue
            ests = []
            for _ in range(n_rep):
                pick = rng.choice(len(eligible), size=m, replace=False)
                sq_means = []
                for i in pick:
                    v = np.asarray(groups[eligible[i]], dtype=float)
                    sel = rng.choice(v.size, size=k, replace=False)
                    sq_means.append(float(v[sel].mean()))
                ests.append(float(np.mean(sq_means)))
            ests = np.asarray(ests)
            rows.append({
                "m_squares": m, "k_areas_per_square": k, "total_images": m * k,
                "n_eligible_squares": len(eligible),
                "empirical_sd_of_estimate": float(ests.std(ddof=1)),
                "empirical_rel_se": float(ests.std(ddof=1) / ests.mean()),
                "mean_of_estimates": float(ests.mean()),
            })
    return rows


# ── size-distribution convergence ────────────────────────────────────────────

def ks_convergence(by_image: Dict[str, List[float]], images: Sequence[str],
                   k_values: Sequence[int], n_rep: int,
                   rng: np.random.Generator) -> List[Dict]:
    """KS distance from a k-area subsample's pooled size distribution to the
    full-data one.

    Because the subsample is a subset of the reference, this measures the
    STABILITY of the measured distribution, not its accuracy against truth --
    the full-data distribution is itself an estimate. Reported as median and
    90th-percentile KS so the answer is "k areas puts 90% of draws within X".
    """
    usable = [im for im in images if by_image.get(im)]
    full = np.sort(np.concatenate([np.asarray(by_image[im]) for im in usable]))
    n_full = full.size
    f_full = np.searchsorted(full, full, side="right") / n_full

    rows = []
    for k in k_values:
        if k > len(usable):
            continue
        dists = []
        for _ in range(n_rep):
            pick = rng.choice(len(usable), size=k, replace=False)
            sub = np.sort(np.concatenate(
                [np.asarray(by_image[usable[i]]) for i in pick]))
            f_sub = np.searchsorted(sub, full, side="right") / sub.size
            dists.append(float(np.max(np.abs(f_sub - f_full))))
        d = np.asarray(dists)
        rows.append({
            "k_areas": k,
            "median_ks": float(np.median(d)),
            "p90_ks": float(np.percentile(d, 90)),
            "mean_n_objects": float(n_full * k / len(usable)),
            "frac_draws_below_target": float(np.mean(d < KS_TARGET)),
        })
    return rows


# ── plots ────────────────────────────────────────────────────────────────────

def plot_forest(rows: List[Dict], vc: Dict[str, float], out: Path, *,
                xlabel: str, title: str) -> None:
    """Per-square means with pooled-SE intervals, ordered by effect."""
    ordered = sorted(rows, key=lambda r: r["mean"])
    y = np.arange(len(ordered))
    mean = np.array([r["mean"] for r in ordered])
    se = np.array([r["se_pooled"] for r in ordered])

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8.5, 0.42 * len(ordered) + 2.4))
        grid = vc["grand_mean"]
        # A square is "flat" when its interval spans the grid mean.
        inside = (mean - 1.96 * se <= grid) & (mean + 1.96 * se >= grid)
        ax.axvline(grid, color="0.35", lw=1.4, zorder=1,
                   label=f"grid mean = {grid:.3g}")
        ax.axvspan(grid - 1.96 * math.sqrt(vc["sigma2_within"]),
                   grid + 1.96 * math.sqrt(vc["sigma2_within"]),
                   color="0.85", zorder=0,
                   label="±1.96 σ_within (single-area spread)")
        for i, (r, m_, s_, ins) in enumerate(zip(ordered, mean, se, inside)):
            c = "0.45" if ins else COLORS[1]
            ax.errorbar(m_, i, xerr=1.96 * s_, fmt="o", color=c, ecolor=c,
                        capsize=3, ms=6, zorder=3)
            lab = f"  n={r['n_images']}"
            if r["low_n_flag"]:
                lab += " *"
            ax.text(ax.get_xlim()[1], i, lab, va="center", fontsize=9,
                    color="0.3")
        ax.set_yticks(y)
        ax.set_yticklabels([r["square"] for r in ordered])
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.grid(alpha=0.2, axis="x")
        ax.legend(frameon=False, loc="lower right", fontsize=9)
        fig.text(0.01, 0.005,
                 "Intervals are ±1.96·sqrt(σ²_within / n_i), pooled within-square "
                 "variance.  * = n<3 images.  Red = interval excludes the grid mean.",
                 fontsize=8, color="0.4")
        save_figure(fig, out)
        plt.close(fig)


def plot_variance_decomposition(blocks: List[Tuple[str, Dict, Dict]],
                                out: Path) -> None:
    """Stacked between/within variance as a share of total, one bar per
    response, ICC annotated."""
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(1.9 * len(blocks) + 3.4, 5.2))
        x = np.arange(len(blocks))
        for i, (name, vc, ci) in enumerate(blocks):
            tot = vc["sigma2_total"] or 1.0
            fb = vc["sigma2_between"] / tot
            ax.bar(i, fb, color=COLORS[0], width=0.55,
                   label="between squares" if i == 0 else None)
            ax.bar(i, 1 - fb, bottom=fb, color=COLORS[2], width=0.55,
                   label="within square (area to area)" if i == 0 else None)
            lo, hi = ci.get("icc", (float("nan"),) * 2)
            ax.text(i, 1.02, f"ICC {vc['icc']:.2f}\n[{lo:.2f}, {hi:.2f}]",
                    ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels([b[0] for b in blocks])
        ax.set_ylim(0, 1.22)
        ax.set_ylabel("share of total variance")
        ax.set_title("Where the variability lives")
        ax.legend(frameon=False, loc="lower left", fontsize=9)
        ax.grid(alpha=0.2, axis="y")
        save_figure(fig, out)
        plt.close(fig)


def plot_sampling_power(power: List[Dict], emp: List[Dict], vc: Dict[str, float],
                        out: Path, *, response: str) -> None:
    """Left: relative SE vs squares imaged, one line per k. Right: the same
    against total images, which is the fixed-microscope-time view."""
    ks = sorted({r["k_areas_per_square"] for r in power})
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
        for j, k in enumerate(ks):
            sel = sorted([r for r in power if r["k_areas_per_square"] == k],
                         key=lambda r: r["m_squares"])
            m = [r["m_squares"] for r in sel]
            rel = [r["rel_se"] for r in sel]
            c = COLORS[j % len(COLORS)]
            axes[0].plot(m, rel, "-o", color=c, ms=3.5, label=f"k={k}")
            axes[0].fill_between(m, [r["rel_se_lo"] for r in sel],
                                 [r["rel_se_hi"] for r in sel], color=c, alpha=0.12)
            axes[1].plot([r["total_images"] for r in sel], rel, "-o", color=c, ms=3.5,
                         label=f"k={k}")
        for e in emp:
            j = ks.index(e["k_areas_per_square"]) if e["k_areas_per_square"] in ks else 0
            axes[0].plot(e["m_squares"], e["empirical_rel_se"], "x",
                         color=COLORS[j % len(COLORS)], ms=7, mew=1.8)
        axes[0].plot([], [], "kx", ms=7, mew=1.8, label="observed (subsampled)")

        for ax, xl in ((axes[0], "squares imaged (m)"),
                       (axes[1], "total images imaged (m × k)")):
            for t in SE_TARGETS:
                ax.axhline(t, color="0.8", lw=0.8, ls=":")
                ax.text(ax.get_xlim()[1], t, f" {t:.0%}", fontsize=8,
                        color="0.5", va="center")
            ax.set_xlabel(xl)
            ax.set_ylabel("relative SE of the grid mean")
            ax.set_ylim(0, min(1.0, max(r["rel_se"] for r in power) * 1.05))
            ax.grid(alpha=0.2)
        axes[0].legend(frameon=False, fontsize=9, title="areas/square")
        axes[0].set_title(f"Precision vs squares sampled — {response}")
        axes[1].set_title("Precision per unit of microscope time")
        fig.text(0.01, 0.005,
                 f"Bands are the 95% cluster-bootstrap CI on the variance components. "
                 f"σ²_between={vc['sigma2_between']:.3g}, "
                 f"σ²_within={vc['sigma2_within']:.3g}, ICC={vc['icc']:.2f}.",
                 fontsize=8, color="0.4")
        save_figure(fig, out)
        plt.close(fig)


def plot_ks_convergence(rows: List[Dict], out: Path) -> None:
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7.6, 5.2))
        k = [r["k_areas"] for r in rows]
        ax.plot(k, [r["median_ks"] for r in rows], "-o", color=COLORS[0], ms=4,
                label="median")
        ax.plot(k, [r["p90_ks"] for r in rows], "-s", color=COLORS[1], ms=4,
                label="90th percentile")
        ax.axhline(KS_TARGET, color="0.5", ls="--", lw=1,
                   label=f"KS = {KS_TARGET:.2f}")
        hit = [r["k_areas"] for r in rows if r["p90_ks"] <= KS_TARGET]
        if hit:
            ax.axvline(hit[0], color=COLORS[2], lw=1.2, ls=":")
            ax.text(hit[0], ax.get_ylim()[1] * 0.92, f" {hit[0]} areas",
                    color=COLORS[2], fontsize=10)
        ax.set_xlabel("areas pooled (k)")
        ax.set_ylabel("KS distance to full-data size distribution")
        ax.set_title("Size-distribution stability vs areas sampled")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, fontsize=9)
        fig.text(0.01, 0.005,
                 "Subsamples are subsets of the reference, so this measures "
                 "stability of the measured distribution, not accuracy against truth.",
                 fontsize=8, color="0.4")
        save_figure(fig, out)
        plt.close(fig)


# ── output ───────────────────────────────────────────────────────────────────

def write_csv(path: Path, rows: List[Dict]) -> None:
    """Rows may carry response-specific extras (overdispersion for counts, the
    object-level component for sizes), so the header is the union of all keys in
    first-seen order rather than just the first row's."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: List[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--morphology-csv", required=True, type=Path)
    ap.add_argument("--per-image-csv", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--metric", default="equivalent_diameter",
                    help="per-object morphology column for the size analysis")
    ap.add_argument("--image-area-um2", type=float, default=None,
                    help="field of view per image; enables density (objects/um^2) reporting")
    ap.add_argument("--label", default="run")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--n-subsample", type=int, default=2000,
                    help="reps for the empirical power and KS curves")
    ap.add_argument("--seed", type=int, default=20260727)
    # Acquisition costs for the precision-per-hour model. Defaults are the
    # operator-confirmed Glacios figures for this workflow.
    ap.add_argument("--t-square", type=float, default=180.0,
                    help="seconds per new square (eucentricity, setup)")
    ap.add_argument("--t-position", type=float, default=45.0,
                    help="seconds per new stage position (move + autofocus)")
    ap.add_argument("--t-expose", type=float, default=10.0,
                    help="seconds per FOV (beam shift + exposure)")
    ap.add_argument("--n-shift", type=int, default=16,
                    help="FOVs reachable per position by beam-image shift")
    args = ap.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print("=" * 74)
    print(f"EV variance components and sampling design — {args.label}")
    print("=" * 74)

    per_image = load_per_image(args.per_image_csv)
    by_image = load_objects_by_image(args.morphology_csv, args.metric)

    # Abundance: counts per image, grouped by square.
    counts = defaultdict(list)
    for r in per_image:
        counts[r["square"]].append(float(r["n_accepted"]))
    counts = dict(counts)

    n_img = len(per_image)
    n_obj = sum(len(v) for v in by_image.values())
    print(f"\nimages {n_img}   squares {len(counts)}   objects {n_obj}")
    print("areas per square: " + ", ".join(
        f"{k.split('_')[-1]}={len(v)}" for k, v in
        sorted(counts.items(), key=lambda kv: _square_sort_key(kv[0]))))

    summary: Dict[str, object] = {"label": args.label, "n_images": n_img,
                                  "n_objects": n_obj, "n_squares": len(counts)}
    vc_rows, blocks = [], []

    # ── abundance ────────────────────────────────────────────────────────────
    vc_c = variance_components(counts)
    ci_c = cluster_bootstrap(counts, args.n_boot, rng)
    perm_c = permutation_f(counts, args.n_perm, rng)

    # Poisson reference: a uniform random EV field gives variance == mean.
    all_counts = np.concatenate([np.asarray(v) for v in counts.values()])
    overdispersion = float(all_counts.var(ddof=1) / all_counts.mean())

    print(f"\n--- ABUNDANCE (objects per image) ---")
    _report(vc_c, ci_c, perm_c, unit="objects/image")
    print(f"  overdispersion (var/mean)  : {overdispersion:.2f}   "
          f"(1.0 = Poisson, uniform random field)")
    if args.image_area_um2:
        print(f"  density                    : "
              f"{vc_c['grand_mean'] / args.image_area_um2:.2f} objects/um^2")

    vc_rows.append(_vc_row("abundance_per_image", vc_c, ci_c, perm_c,
                           extra={"overdispersion_var_over_mean": overdispersion}))
    blocks.append(("abundance\n(per image)", vc_c, ci_c))

    sq_counts = per_square_table(counts, vc_c)
    write_csv(out / "per_square_abundance.csv", sq_counts)
    plot_forest(sq_counts, vc_c, out / "forest_abundance_by_square",
                xlabel="mean EV detections per image",
                title=f"EV abundance by square — {args.label}")

    m_vals = list(range(2, 21)) + [25, 30, 40]
    k_vals = [1, 2, 3, 4, 5, 6]
    power = sampling_power_table(vc_c, ci_c, m_vals, k_vals)
    emp = empirical_power(counts, [2, 3, 4, 6, 8, 10, 12, 14], [1, 2, 3],
                          args.n_subsample, rng)
    req = squares_required(vc_c, ci_c, k_vals, SE_TARGETS)
    write_csv(out / "sampling_power_abundance.csv", power)
    write_csv(out / "sampling_power_empirical.csv", emp)
    write_csv(out / "squares_required_abundance.csv", req)

    time_rows = time_cost_table(
        vc_c, t_square=args.t_square, t_position=args.t_position,
        t_expose=args.t_expose,
        n_shift_values=sorted({1, 4, 9, args.n_shift}),
        m_values=list(range(1, 61)), p_values=list(range(1, 9)),
        budget_s=[1800, 3600, 7200, 14400])
    write_csv(out / "sampling_power_time.csv", time_rows)
    print(f"\n  best design per time budget  (square {args.t_square:.0f}s, "
          f"position {args.t_position:.0f}s, FOV {args.t_expose:.0f}s):")
    print(f"    {'hours':>5} {'independence':>12} {'design':>22} "
          f"{'images':>7} {'rel SE':>7}")
    for r in time_rows:
        design = (f"{r['m_squares']}sq x {r['positions_per_square']}pos "
                  f"x {r['fovs_per_position']}fov")
        print(f"    {r['budget_hours']:5.1f} {r['fov_independence']:>12} "
              f"{design:>22} {r['total_images']:7d} {100 * r['rel_se']:6.1f}%")
    print("    brackets the unmeasured beam-shift scale; trust where both agree")
    plot_sampling_power(power, emp, vc_c, out / "sampling_power_abundance",
                        response="EV abundance")

    print("\n  squares needed for a target precision on the grid mean:")
    print(f"    {'target':>7}  {'k=1':>12} {'k=2':>12} {'k=3':>12} {'k=4':>12}")
    for t in SE_TARGETS:
        cells = []
        for k in (1, 2, 3, 4):
            r = next(x for x in req if x["target_rel_se"] == t
                     and x["k_areas_per_square"] == k)
            cells.append(f"{r['m_squares_needed']}/{r['m_squares_needed_conservative']}")
        print(f"    {t:>6.0%}  " + " ".join(f"{c:>12}" for c in cells))
    print("    cells are m_squares  point-estimate / conservative (upper CI)")

    # ── size ─────────────────────────────────────────────────────────────────
    img_to_sq = {r["image"]: r["square"] for r in per_image}
    per_img_mean = defaultdict(list)
    for im, vals in by_image.items():
        if im in img_to_sq and vals:
            per_img_mean[img_to_sq[im]].append(float(np.mean(vals)))
    per_img_mean = dict(per_img_mean)

    vc_s = variance_components(per_img_mean)
    ci_s = cluster_bootstrap(per_img_mean, args.n_boot, rng)
    perm_s = permutation_f(per_img_mean, args.n_perm, rng)

    # Within-image object spread: the third level, pooled.
    within_img = [float(np.var(v, ddof=1)) for v in by_image.values() if len(v) > 1]
    s2_object = float(np.mean(within_img)) if within_img else float("nan")

    print(f"\n--- SIZE ({args.metric}, per-image mean, nm) ---")
    _report(vc_s, ci_s, perm_s, unit="nm")
    print(f"  sigma2_object (within image): {s2_object:.3g}   "
          f"(sd {math.sqrt(s2_object):.2f} nm)")

    # Robustness check: the same decomposition on per-image MEDIAN diameter.
    # The median is insensitive to the upper-tail truncation caused by missed
    # large vesicles, so a divergent ICC here would mean the size result is
    # driven by the tail rather than by the bulk of the distribution.
    per_img_median = defaultdict(list)
    for im, vals in by_image.items():
        if im in img_to_sq and vals:
            per_img_median[img_to_sq[im]].append(float(np.median(vals)))
    vc_smed = (variance_components(dict(per_img_median))
               if len(per_img_median) > 1 else vc_s)
    print(f"  ICC on per-image MEDIAN      : {vc_smed['icc']:.3f}   "
          f"(robustness check vs {vc_s['icc']:.3f} on means)")

    vc_rows.append(_vc_row("size_per_image_mean", vc_s, ci_s, perm_s,
                           extra={"sigma2_object_within_image": s2_object}))
    blocks.append((f"size\n(per-image mean)", vc_s, ci_s))

    sq_size = per_square_table(per_img_mean, vc_s)
    write_csv(out / "per_square_size.csv", sq_size)
    plot_forest(sq_size, vc_s, out / "forest_size_by_square",
                xlabel=f"mean {args.metric} per image (nm)",
                title=f"EV size by square — {args.label}")

    power_s = sampling_power_table(vc_s, ci_s, m_vals, k_vals)
    req_s = squares_required(vc_s, ci_s, k_vals, SE_TARGETS)
    write_csv(out / "sampling_power_size.csv", power_s)
    write_csv(out / "squares_required_size.csv", req_s)
    plot_sampling_power(power_s, [], vc_s, out / "sampling_power_size",
                        response="EV size")

    # ── size-distribution convergence ────────────────────────────────────────
    ks_rows = ks_convergence(by_image, [r["image"] for r in per_image],
                             list(range(1, min(len(per_image), 40) + 1)),
                             args.n_subsample, rng)
    write_csv(out / "size_distribution_convergence.csv", ks_rows)
    plot_ks_convergence(ks_rows, out / "size_ks_convergence")
    hit = [r["k_areas"] for r in ks_rows if r["p90_ks"] <= KS_TARGET]
    max_k = ks_rows[-1]["k_areas"] if ks_rows else 0
    reached = str(hit[0]) if hit else f">{max_k} (not reached)"
    print("\n--- SIZE DISTRIBUTION CONVERGENCE ---")
    print(f"  areas for 90% of draws within KS<{KS_TARGET}: {reached}")
    for r in ks_rows:
        if r["k_areas"] in (1, 2, 3, 5, 10, 15, 20, 30):
            print(f"    k={r['k_areas']:<3} median KS {r['median_ks']:.3f}   "
                  f"p90 {r['p90_ks']:.3f}")

    write_csv(out / "variance_components.csv", vc_rows)
    write_csv(out / "permutation_tests.csv", [
        dict(response="abundance_per_image", **perm_c),
        dict(response="size_per_image_mean", **perm_s),
    ])
    plot_variance_decomposition(blocks, out / "variance_decomposition")

    summary.update({
        "abundance": _vc_row("abundance_per_image", vc_c, ci_c, perm_c,
                             extra={"overdispersion_var_over_mean": overdispersion}),
        "size": _vc_row("size_per_image_mean", vc_s, ci_s, perm_s,
                        extra={"sigma2_object_within_image": s2_object,
                               "icc_median_robustness_check": vc_smed["icc"]}),
        "ks_target_areas": hit[0] if hit else None,
        "image_area_um2": args.image_area_um2,
    })
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nwrote {len(list(out.glob('*')))} file(s) to {out}")


def _group_images(per_image: List[Dict]) -> Dict[str, List[str]]:
    g = defaultdict(list)
    for r in per_image:
        g[r["square"]].append(r["image"])
    return dict(g)


def _vc_row(response: str, vc: Dict, ci: Dict, perm: Dict,
            extra: Dict | None = None) -> Dict:
    row = {"response": response}
    row.update({k: v for k, v in vc.items()})
    for f, (lo, hi) in ci.items():
        row[f"{f}_ci_lo"], row[f"{f}_ci_hi"] = lo, hi
    row.update({f"perm_{k}": v for k, v in perm.items()})
    if extra:
        row.update(extra)
    return row


def _report(vc: Dict, ci: Dict, perm: Dict, *, unit: str) -> None:
    def rng_(f):
        lo, hi = ci.get(f, (float("nan"),) * 2)
        return f"[{lo:.3g}, {hi:.3g}]"
    print(f"  grand mean                 : {vc['grand_mean']:.3f} {unit} "
          f"{rng_('grand_mean')}")
    print(f"  sigma2_between (squares)   : {vc['sigma2_between']:.4g} "
          f"{rng_('sigma2_between')}   sd {vc['sd_between']:.3g}")
    print(f"  sigma2_within  (areas)     : {vc['sigma2_within']:.4g} "
          f"{rng_('sigma2_within')}   sd {vc['sd_within']:.3g}")
    print(f"  ICC                        : {vc['icc']:.3f} {rng_('icc')}")
    print(f"  CV between / within        : {vc['cv_between']:.3f} / "
          f"{vc['cv_within']:.3f}")
    print(f"  ANOVA F({vc['df_between']},{vc['df_within']})"
          f"{'':<14.14}= {vc['f_statistic']:.3f}")
    print(f"  permutation null           : F {perm['null_mean']:.3f} "
          f"± {perm['null_sd']:.3f}  ->  z = {perm['z_vs_null']:+.2f}, "
          f"p = {perm['p_permutation']:.4f}")
    if vc["sigma2_between_raw"] < 0:
        print(f"  NOTE: raw MoM sigma2_between was negative "
              f"({vc['sigma2_between_raw']:.3g}); truncated to 0, i.e. no "
              f"detectable between-square variance beyond noise.")


if __name__ == "__main__":
    main()
