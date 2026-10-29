#!/usr/bin/env python3
"""Three-level variance split for EV detections: square / position / field.

ev_variance_components.py splits variance two ways, between squares and
between areas inside a square, because 20260727 imaged one field per stage
position. It could only bracket whether fields tiled at one position (beam or
image shift, a few um apart) are independent samples or near-copies; see the
time_cost_table docstring there. A montage session images several fields per
position, so the middle level becomes measurable:

    y_ijk = mu + a_i (square) + b_ij (position in square) + e_ijk (field)

Estimator: ANOVA method of moments for the unbalanced nested random model
(Searle 1971, sec. 6.3; Sokal & Rohlf), components truncated at zero with the
raw value kept:

    E[MS_field] = s2_f
    E[MS_pos]   = s2_f + c1 s2_p
    E[MS_sq]    = s2_f + c2 s2_p + c3 s2_s
    c1 = (N - sum_ij n_ij^2 / n_i) / (b - a)
    c2 = (sum_ij n_ij^2 / n_i - sum_ij n_ij^2 / N) / (a - 1)
    c3 = (N - sum_i n_i^2 / N) / (a - 1)

which is n, n and p*n in a balanced design. s2_square is solved with the raw
s2_p, then each component is truncated. With one position per square the
middle level vanishes (s2_p not estimable, reported nan) and the square/field split is
exactly ev_variance_components.variance_components -- the regression control
below checks that on 20260727.

Uncertainty:
  bootstrap  resample SQUARES with replacement, each carrying its positions
             and fields (cluster bootstrap, as in ev_variance_components; with
             ~11 squares the 95% interval under-covers -- read it as optimistic)
  permutation  position effect: shuffle fields among the positions of the same
             square, statistic MS_pos / MS_field. Square effect: shuffle whole
             positions between squares, statistic MS_sq / MS_pos.

Design precision for the grid mean from m squares x p positions x n fields:

    SE = sqrt(s2_s/m + s2_p/(m p) + s2_f/(m p n))

and per microscope hour with T = m t_sq + m p t_pos + m p n t_field, the times
measured from the session's mdoc timestamps (timing.json) rather than assumed.

Reads only CSVs (field_table.csv from build_field_positions.py in
Cryo-image-processing). Two controls run first and stop the script if either
fails:
  synthetic   balanced and unbalanced data with known components, recovered
              on average within 10% (MoM is unbiased before truncation)
  0727        with one position per square, reproduces the 20260727
              variance_components.csv abundance row exactly

    python -m analysis.ev_variance_nested --field-table <session>/5. Model + labeling/sampling/field_table.csv \
        --output-dir <session>/5. Model + labeling/sampling/variance --label "20261027 grid2"
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analysis.ev_variance_components import (load_per_image,  # noqa: E402
                                             variance_components)

# nested data: {square: {position: [field values]}}
Nested = Dict[str, Dict[str, List[float]]]


def nested_components(d: Nested) -> Dict[str, float]:
    sq = [s for s in d if any(len(v) for v in d[s].values())]
    a = len(sq)
    if a < 2:
        raise ValueError("need >= 2 squares")
    cells = [(s, p, np.asarray(v, float)) for s in sq for p, v in d[s].items() if len(v)]
    b = len(cells)
    N = float(sum(len(v) for _, _, v in cells))
    allv = np.concatenate([v for _, _, v in cells])
    grand = float(allv.mean())
    n_i = {s: float(sum(len(v) for s2, _, v in cells if s2 == s)) for s in sq}
    m_i = {s: float(np.concatenate([v for s2, _, v in cells if s2 == s]).mean()) for s in sq}

    ss_s = sum(n_i[s] * (m_i[s] - grand) ** 2 for s in sq)
    ss_p = sum(len(v) * (v.mean() - m_i[s]) ** 2 for s, _, v in cells)
    ss_f = sum(float(((v - v.mean()) ** 2).sum()) for _, _, v in cells)
    df_s, df_p, df_f = a - 1, b - a, N - b
    ms_s = ss_s / df_s
    ms_p = ss_p / df_p if df_p > 0 else float("nan")
    ms_f = ss_f / df_f if df_f > 0 else float("nan")

    sum_nij2_ni = sum(len(v) ** 2 / n_i[s] for s, _, v in cells)
    sum_nij2 = sum(len(v) ** 2 for _, _, v in cells)
    c1 = (N - sum_nij2_ni) / df_p if df_p > 0 else float("nan")
    c2 = (sum_nij2_ni - sum_nij2 / N) / df_s
    c3 = (N - sum(n ** 2 for n in n_i.values()) / N) / df_s

    if df_f <= 0:
        raise ValueError("need more than one field in some position")
    s2_f = ms_f
    if df_p > 0:
        s2_p_raw = (ms_p - ms_f) / c1
        # solve with the raw (untruncated) s2_p: that is the MoM solution
        s2_s_raw = (ms_s - ms_f - c2 * s2_p_raw) / c3
    else:
        # one position per square: the two-level case, s2_p not estimable
        s2_p_raw = float("nan")
        s2_s_raw = (ms_s - ms_f) / c3
    s2_p = max(0.0, s2_p_raw) if math.isfinite(s2_p_raw) else float("nan")
    s2_s = max(0.0, s2_s_raw)
    parts = [x for x in (s2_s, s2_p, s2_f) if math.isfinite(x)]
    tot = sum(parts)
    return {
        "squares": a, "positions": b, "fields": int(N), "grand_mean": grand,
        "ms_square": ms_s, "ms_position": ms_p, "ms_field": ms_f,
        "df_square": df_s, "df_position": df_p, "df_field": df_f,
        "c1": c1, "c2": c2, "c3": c3,
        "s2_square": s2_s, "s2_square_raw": s2_s_raw,
        "s2_position": s2_p, "s2_position_raw": s2_p_raw,
        "s2_field": s2_f, "s2_total": tot,
        "share_square": s2_s / tot if tot else float("nan"),
        "share_position": s2_p / tot if tot and math.isfinite(s2_p) else float("nan"),
        "share_field": s2_f / tot if tot and math.isfinite(s2_f) else float("nan"),
        "F_position": ms_p / ms_f if df_p > 0 and df_f > 0 and ms_f > 0 else float("nan"),
        "F_square": ms_s / ms_p if df_p > 0 and ms_p > 0 else float("nan"),
    }


KEYS = ("s2_square", "s2_position", "s2_field", "share_square",
        "share_position", "share_field", "grand_mean")


def bootstrap(d: Nested, n_boot: int, rng) -> Dict[str, tuple]:
    keys = list(d)
    out = defaultdict(list)
    for _ in range(n_boot):
        pick = rng.choice(len(keys), len(keys), replace=True)
        try:
            vc = nested_components({f"b{j}": d[keys[i]] for j, i in enumerate(pick)})
        except ValueError:
            continue
        for k in KEYS:
            if math.isfinite(vc[k]):
                out[k].append(vc[k])
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in out.items()}


def permutation(d: Nested, n_perm: int, rng) -> Dict[str, float]:
    obs = nested_components(d)
    null_p, null_s = [], []
    pos_list = [(s, p, list(v)) for s in d for p, v in d[s].items() if len(v)]
    for _ in range(n_perm):
        # fields among the positions of their own square
        dp = {}
        for s in d:
            sizes = [(p, len(v)) for p, v in d[s].items() if len(v)]
            pooled = rng.permutation(np.concatenate([np.asarray(v, float) for v in d[s].values() if len(v)]))
            cut, dp[s] = 0, {}
            for p, n in sizes:
                dp[s][p] = pooled[cut:cut + n]
                cut += n
        null_p.append(nested_components(dp)["F_position"])
        # whole positions between squares, square sizes (in positions) fixed
        order = rng.permutation(len(pos_list))
        ds, cut = {}, 0
        for s in d:
            k = sum(1 for p, v in d[s].items() if len(v))
            ds[s] = {f"q{j}": pos_list[i][2] for j, i in enumerate(order[cut:cut + k])}
            cut += k
        null_s.append(nested_components(ds)["F_square"])

    def p(obs_v, null):
        null = np.asarray([x for x in null if math.isfinite(x)])
        return (1.0 + float((null >= obs_v).sum())) / (1.0 + null.size) if math.isfinite(obs_v) else float("nan")
    return {"F_position": obs["F_position"], "p_position": p(obs["F_position"], null_p),
            "F_square": obs["F_square"], "p_square": p(obs["F_square"], null_s),
            "n_perm": n_perm}


def design_se(vc, m, p, n):
    s2p = vc["s2_position"] if math.isfinite(vc["s2_position"]) else 0.0
    return math.sqrt(vc["s2_square"] / m + s2p / (m * p) + vc["s2_field"] / (m * p * n))


def design_table(vc, ci, timing, budgets_h=(0.5, 1, 2, 4)) -> List[Dict]:
    rows = []
    mean = vc["grand_mean"]
    for m in (3, 5, 8, 11, 15, 20, 30):
        for p in (1, 2, 3, 4):
            for n in (1, 4, 9, 16):
                minutes = (m * timing["t_square_s"] + m * p * timing["t_position_s"]
                           + m * p * n * timing["t_field_s"]) / 60.0
                rows.append({"m_squares": m, "positions_per_square": p, "fields_per_position": n,
                             "total_fields": m * p * n, "minutes": round(minutes, 1),
                             "rel_se": design_se(vc, m, p, n) / mean})
    best = []
    for h in budgets_h:
        ok = [r for r in rows if r["minutes"] <= h * 60]
        if ok:
            r = min(ok, key=lambda r: r["rel_se"])
            best.append({"budget_hours": h, **r})
    return rows, best


# ── controls ────────────────────────────────────────────────────────────────

def simulate(rng, a, p_sizes, n_sizes, s2s, s2p, s2f, mu=20.0) -> Nested:
    d = {}
    for i in range(a):
        ai = rng.normal(0, math.sqrt(s2s))
        d[f"s{i}"] = {}
        for j in range(p_sizes[i]):
            bij = rng.normal(0, math.sqrt(s2p))
            d[f"s{i}"][f"p{j}"] = list(mu + ai + bij + rng.normal(0, math.sqrt(s2f), n_sizes[i][j]))
    return d


def control_synthetic(rng, reps=2000) -> List[Dict]:
    truth = {"s2_square": 4.0, "s2_position": 2.0, "s2_field": 10.0}
    rows = []
    for name, a, ps, ns in (
            ("balanced 11x3x9", 11, [3] * 11, [[9] * 3] * 11),
            ("unbalanced", 11, [3, 3, 2, 3, 3, 1, 3, 3, 2, 3, 3],
             [[9, 9, 6], [9, 9, 9], [9, 5], [9, 9, 9], [7, 9, 9], [9], [9, 9, 9],
              [9, 9, 9], [9, 8], [9, 9, 9], [6, 9, 9]])):
        est = defaultdict(list)
        for _ in range(reps):
            vc = nested_components(simulate(rng, a, ps, ns, truth["s2_square"],
                                            truth["s2_position"], truth["s2_field"]))
            est["s2_square"].append(vc["s2_square_raw"])
            est["s2_position"].append(vc["s2_position_raw"])
            est["s2_field"].append(vc["s2_field"])
        for k, t in truth.items():
            m = float(np.mean(est[k]))
            rows.append({"control": "synthetic", "case": name, "component": k, "truth": t,
                         "mean_estimate": round(m, 3), "rel_bias": round(m / t - 1, 3),
                         "pass": abs(m / t - 1) <= 0.10})
    return rows


def control_0727(per_image_csv: Path, reference_csv: Path) -> List[Dict]:
    per = load_per_image(per_image_csv)
    groups = defaultdict(list)
    for r in per:
        groups[r["square"]].append(float(r["n_accepted"]))
    nested = {s: {"one": v} for s, v in groups.items()}
    vc = nested_components(nested)
    two = variance_components(dict(groups))
    ref = next(r for r in csv.DictReader(open(reference_csv, newline=""))
               if r["response"] == "abundance_per_image")
    rows = []
    for mine, theirs, refk in (("s2_square", "sigma2_between", "sigma2_between"),
                               ("s2_field", "sigma2_within", "sigma2_within"),
                               ("grand_mean", "grand_mean", "grand_mean")):
        a, b, c = vc[mine], two[theirs], float(ref[refk])
        rows.append({"control": "0727 2-level", "case": refk, "component": mine,
                     "truth": round(c, 6), "mean_estimate": round(a, 6),
                     "rel_bias": round(a / c - 1, 9) if c else 0.0,
                     "pass": abs(a - b) < 1e-9 and abs(a - c) < 1e-6})
    return rows


# ── main ────────────────────────────────────────────────────────────────────

def load_nested(field_table: Path, value: str, usable_only: bool,
                covariate: str = None) -> tuple:
    """Nested values; with `covariate`, each value is adjusted to the mean
    covariate by an OLS slope fitted across fields (a fixed effect removed
    before the split). Returns (nested, slope or None)."""
    rows = []
    for r in csv.DictReader(open(field_table, newline="", encoding="utf-8")):
        if usable_only and r.get("usable", "").strip().lower() in ("no", "n", "0", "false"):
            continue
        v = r.get(value, "")
        if v in ("", "nan"):
            continue
        x = float(r[covariate]) if covariate else 0.0
        rows.append((f"sq{int(r['square']):02d}", f"pos{int(r['position']):02d}", float(v), x))
    slope = None
    if covariate:
        y = np.array([r[2] for r in rows])
        x = np.array([r[3] for r in rows])
        slope = float(np.polyfit(x, y, 1)[0])
        rows = [(s, p, v - slope * (xx - x.mean()), xx) for s, p, v, xx in rows]
    d = defaultdict(lambda: defaultdict(list))
    for s, p, v, _ in rows:
        d[s][p].append(v)
    return {s: dict(p) for s, p in d.items()}, slope


def write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--field-table", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--timing", type=Path, default=None, help="timing.json; default beside the field table")
    ap.add_argument("--label", default="run")
    ap.add_argument("--responses", default="ev_core_per_um2,median_diam_core_nm",
                    help="field_table columns to decompose")
    ap.add_argument("--covariate", default="mdoc_defocus_um",
                    help="field_table column removed as a fixed effect in the adjusted subset")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-perm", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20261027)
    ap.add_argument("--control-0727-per-image", type=Path, default=Path(
        r"C:\Users\pujan\Desktop\Cryo-image-processing\20260727\processed\yolo_out\accept"
        r"\v3_run1_20260824\morphology\per_image_summary.csv"))
    ap.add_argument("--control-0727-reference", type=Path, default=Path(
        r"C:\Users\pujan\Desktop\Cryo-image-processing\20260727\processed\yolo_out\accept"
        r"\v3_run1_20260824\variance\variance_components.csv"))
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    print("controls (must pass before any result is computed)")
    ctl = control_synthetic(rng) + control_0727(args.control_0727_per_image, args.control_0727_reference)
    write_csv(out / "controls.csv", ctl)
    for r in ctl:
        print(f"  {r['control']:13s} {r['case']:16s} {r['component']:12s} truth {r['truth']:<10} "
              f"est {r['mean_estimate']:<10} {'PASS' if r['pass'] else 'FAIL'}")
    if not all(r["pass"] for r in ctl):
        sys.exit("control failed: no results written")

    timing = json.load(open(args.timing or args.field_table.parent / "timing.json"))
    vc_rows, design_rows, best_rows, sq_rows = [], [], [], []
    for resp in args.responses.split(","):
        for subset, usable_only, cov in (("all fields", False, None),
                                         ("usable fields", True, None),
                                         ("usable, defocus-adjusted", True, args.covariate)):
            d, slope = load_nested(args.field_table, resp, usable_only, cov)
            vc = nested_components(d)
            ci = bootstrap(d, args.n_boot, rng)
            pm = permutation(d, args.n_perm, rng)
            row = {"response": resp, "subset": subset, "covariate_slope": slope,
                   **{k: vc[k] for k in vc}, **pm}
            for k, (lo, hi) in ci.items():
                row[f"{k}_ci_lo"], row[f"{k}_ci_hi"] = lo, hi
            vc_rows.append(row)
            print(f"\n{resp} [{subset}]  {vc['squares']} sq, {vc['positions']} pos, {vc['fields']} fields, "
                  f"mean {vc['grand_mean']:.3f}"
                  + (f", slope vs {args.covariate} {slope:.3f}/um" if slope is not None else ""))
            for lvl in ("square", "position", "field"):
                lo, hi = ci.get(f"share_{lvl}", (float("nan"), float("nan")))
                print(f"  {lvl:9s} s2 {vc['s2_' + lvl]:9.4f}  share {vc['share_' + lvl]:5.2f} "
                      f"(95% {lo:.2f}-{hi:.2f})")
            print(f"  permutation p: position {pm['p_position']:.4f}, square {pm['p_square']:.4f}")
            rows, best = design_table(vc, ci, timing)
            for r in rows:
                design_rows.append({"response": resp, "subset": subset, **r})
            for r in best:
                best_rows.append({"response": resp, "subset": subset, **r})
            for s in sorted(d):
                vals = [x for v in d[s].values() for x in v]
                sq_rows.append({"response": resp, "subset": subset, "square": s,
                                "positions": len(d[s]), "fields": len(vals),
                                "mean": float(np.mean(vals)),
                                "position_means": "; ".join(f"{p}={np.mean(v):.3f}"
                                                            for p, v in sorted(d[s].items()))})
    write_csv(out / "variance_nested.csv", vc_rows)
    write_csv(out / "design_precision.csv", design_rows)
    write_csv(out / "design_best_per_budget.csv", best_rows)
    write_csv(out / "per_square.csv", sq_rows)
    json.dump({"label": args.label, "seed": args.seed, "timing": timing,
               "n_boot": args.n_boot, "n_perm": args.n_perm}, open(out / "summary.json", "w"), indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
