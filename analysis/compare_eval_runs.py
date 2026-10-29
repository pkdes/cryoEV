"""
Collect several models' evaluation outputs into one per-class comparison table, with
paired bootstrap confidence intervals against a reference model.

Inputs (written per model by evaluate_coco_export.py --per-image and
multilayer_role_analysis.py --fixed ... --out-dir <role-dir>/<model>_<view>):

  <eval-dir>/eval_<model>_<split>_<view>.csv            detection P/R/F1 per conf, per group
  <eval-dir>/eval_<model>_<split>_<view>_per_image.csv  per-image TP/FP/FN (+ summed matched IoU)
  <eval-dir>/testmap_<model>.json                       test box/mask mAP (optional)
  <role-dir>/<model>_<view>/role_f1_*.csv, role_per_image_*.csv

Detection is reported at conf 0.25 and at the conf that maximizes F1 on **val** (never
chosen on test). Groups are pooled into subsets: ALL, old (every session but lipo) and
lipo. The bootstrap resamples test images with replacement (same images for both
models) and reports the 95% interval of (model - reference).

    python analysis/compare_eval_runs.py --eval-dir <dir> --role-dir <dir> --ref F1b --out TABLE.md
"""
import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

SUBSETS = ("ALL", "old", "lipo")


def subset_of(group):
    return "lipo" if "lipo" in group.lower() else "old"


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def read(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load_detection(eval_dir):
    """{(model, split, target, view, subset, conf): [tp, fp, fn]} pooled from the per-group summaries."""
    pool = defaultdict(lambda: [0, 0, 0])
    for f in eval_dir.glob("eval_*.csv"):
        if f.name.endswith("_per_image.csv"):
            continue
        m = re.match(r"eval_(.+)_(test|val)_(fair|practical)\.csv$", f.name)
        if not m:
            continue
        model, split, file_view = m.groups()
        for r in read(f):
            if r["target"] != "ev" and file_view != "practical":
                continue  # non-EV / Undetermined rows are written to both view files; count them once
            g = r["split_role"]
            for s in (["ALL"] if g == "ALL" else [subset_of(g)]):
                k = (model, split, r["target"], r["view"], s, round(float(r["conf"]), 2))
                for i, c in enumerate(("tp", "fp", "fn")):
                    pool[k][i] += int(r[c])
    return pool


def load_per_image(eval_dir):
    """{(model, target, view, conf): {image: (tp, fp, fn, sum_iou, subset)}} for test."""
    out = defaultdict(dict)
    for f in eval_dir.glob("eval_*_test_*_per_image.csv"):
        model = re.match(r"eval_(.+)_test_(fair|practical)_per_image\.csv$", f.name).group(1)
        for r in read(f):
            out[(model, r["target"], r["view"], round(float(r["conf"]), 2))][r["image"]] = (
                int(r["tp"]), int(r["fp"]), int(r["fn"]), float(r["sum_iou"]), subset_of(r["split_role"]))
    return out


def load_roles(role_dir):
    """summary {(model, view, subset, role): (P, R, F1)}, per-image {(model, view): {(image, role): (tp, fp, fn, subset)}}."""
    summ, per = {}, defaultdict(dict)
    for d in role_dir.iterdir() if role_dir.exists() else []:
        m = re.match(r"(.+)_(fair|practical)$", d.name)
        if not m:
            continue
        model, view = m.groups()
        for f in d.glob("role_f1_*.csv"):
            for r in read(f):
                sub = r["subset"]
                sub = "ALL" if sub == "combined" else ("lipo" if "lipo" in sub.lower() else "old") if sub.startswith("group-") else None
                if sub:
                    summ[(model, view, sub, r["role"])] = (float(r["precision"]), float(r["recall"]), float(r["f1"]))
        for f in d.glob("role_per_image_*.csv"):
            for r in read(f):
                per[(model, view)][(r["image"], r["role"])] = (int(r["tp"]), int(r["fp"]), int(r["fn"]),
                                                              "lipo" if "lipo" in r["group"].lower() else "old")
    return summ, per


def bootstrap(per_a, per_b, subset, n=2000, seed=0):
    """95% CI of F1(a) - F1(b) from {image: (tp, fp, fn, ...)}, resampling shared images."""
    imgs = sorted(i for i in set(per_a) & set(per_b) if subset == "ALL" or per_a[i][-1] == subset)
    if len(imgs) < 3:
        return None
    a = np.array([per_a[i][:3] for i in imgs], dtype=float)
    b = np.array([per_b[i][:3] for i in imgs], dtype=float)

    def f1(x):
        tp, fp, fn = x[..., 0], x[..., 1], x[..., 2]
        return np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 0.0)

    idx = np.random.default_rng(seed).integers(0, len(imgs), size=(n, len(imgs)))
    d = f1(a[idx].sum(axis=1)) - f1(b[idx].sum(axis=1))
    return float(f1(a.sum(0)) - f1(b.sum(0))), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-dir", type=Path, required=True)
    ap.add_argument("--role-dir", type=Path, required=True)
    ap.add_argument("--ref", help="Reference model for the bootstrap differences")
    ap.add_argument("--models", help="Comma-separated order to report (default: all found)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    det = load_detection(a.eval_dir)
    per_img = load_per_image(a.eval_dir)
    role_sum, role_per = load_roles(a.role_dir)
    found = sorted({k[0] for k in det} | {k[0] for k in role_sum})
    models = [m for m in a.models.split(",") if m in found] if a.models else found
    L = [f"# Model comparison (S1 test, {len(models)} models)", ""]

    L += ["## Test mask / box mAP50-95 (Ultralytics val, overlap_mask=False)", "",
          "| Model | Mask mAP50-95 | Box mAP50-95 | Per-class mask mAP50-95 |", "|---|--:|--:|---|"]
    for m in models:
        f = a.eval_dir / f"testmap_{m}.json"
        if f.exists():
            j = json.loads(f.read_text())
            pc = ", ".join(f"{k} {v:.3f}" for k, v in j.get("per_class_mask_map50_95", {}).items())
            L.append(f"| {m} | {j['mask_map50_95']:.3f} | {j['box_map50_95']:.3f} | {pc} |")

    L += ["", "## Detection, per class (F1 at IoU 0.5)", "",
          "conf* = the conf with the best F1 on val (ALL), applied to test. Matched IoU = mean mask IoU of true positives at 0.25.", "",
          "| Class / view | Subset | Model | P@0.25 | R@0.25 | F1@0.25 | conf* | F1@conf* | Matched IoU |",
          "|---|---|---|--:|--:|--:|--:|--:|--:|"]
    tv = sorted({(k[2], k[3]) for k in det}, key=lambda x: ("ev", "nonev", "undet").index(x[0]) * 2 + (x[1] == "fair"))
    for target, view in tv:
        for s in SUBSETS:
            for m in models:
                k25 = (m, "test", target, view, s, 0.25)
                if k25 not in det:
                    continue
                p, r, f = prf(*det[k25])
                confs = sorted({k[5] for k in det if k[:5] == (m, "val", target, view, "ALL")})
                cstar = max(confs, key=lambda c: prf(*det[(m, "val", target, view, "ALL", c)])[2]) if confs else None
                fstar = prf(*det[(m, "test", target, view, s, cstar)])[2] if cstar is not None else None
                pi = per_img.get((m, target, view, 0.25), {})
                tp = sum(v[0] for v in pi.values() if s == "ALL" or v[4] == s)
                iou = sum(v[3] for v in pi.values() if s == "ALL" or v[4] == s) / tp if tp else None
                n_gt = det[k25][0] + det[k25][2]
                L.append(f"| {target} / {view} | {s} (n_gt={n_gt}) | {m} | {p:.3f} | {r:.3f} | {f:.3f} | "
                         f"{'' if cstar is None else f'{cstar:.2f}'} | {'' if fstar is None else f'{fstar:.3f}'} | "
                         f"{'' if iou is None else f'{iou:.3f}'} |")

    L += ["", "## Multilayer roles (containment 0.99, size ratio 0.75)", "",
          "| View | Subset | Model | Multilayer P | R | F1 | Inner-layer P | R | F1 |", "|---|---|---|--:|--:|--:|--:|--:|--:|"]
    for view in ("practical", "fair"):
        for s in SUBSETS:
            for m in models:
                ml, inn = role_sum.get((m, view, s, "multilayer")), role_sum.get((m, view, s, "inner-layer"))
                if ml and inn:
                    L.append(f"| {view} | {s} | {m} | {ml[0]:.3f} | {ml[1]:.3f} | {ml[2]:.3f} | {inn[0]:.3f} | {inn[1]:.3f} | {inn[2]:.3f} |")

    if a.ref and a.ref in models:
        L += ["", f"## Difference from {a.ref}, with 95% paired bootstrap interval (2000 resamples of test images)", "",
              "An interval that excludes 0 is a difference the 39-image test set can resolve.", "",
              "| Measure | Subset | Model | Diff | 95% interval |", "|---|---|---|--:|---|"]
        measures = [(f"{t} / {v} F1@0.25", per_img.get((a.ref, t, v, 0.25)), lambda m, t=t, v=v: per_img.get((m, t, v, 0.25)))
                    for t, v in tv]
        for view in ("practical", "fair"):
            for role in ("multilayer", "inner-layer"):
                def rp(m, view=view, role=role):
                    d = role_per.get((m, view))
                    return {img: v for (img, rl), v in d.items() if rl == role} if d else None
                measures.append((f"{role} F1 ({view})", rp(a.ref), rp))
        for name, ref_pi, get in measures:
            if not ref_pi:
                continue
            for s in SUBSETS:
                for m in models:
                    if m == a.ref or not get(m):
                        continue
                    b = bootstrap(get(m), ref_pi, s)
                    if b:
                        L.append(f"| {name} | {s} | {m} | {b[0]:+.3f} | [{b[1]:+.3f}, {b[2]:+.3f}] |")

    a.out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
