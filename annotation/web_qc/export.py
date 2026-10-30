#!/usr/bin/env python3
"""Export web-QC reviews as training labels, and score the model against them.

Only images marked Done are used.

    python -m annotation.web_qc.export \
        --images-dir <tiles> --predictions-dir <model>/predictions \
        --review-dir <out>/qc_review [--analysis-conf 0.5]

Writes <review-dir>/export/:
  images/, labels/ (YOLO-seg, class ids from classes.yaml), data.yaml,
  _annotations.coco.json (imports into Roboflow)
and <review-dir>/metrics/:
  per_image.csv          P/R/F1/IoU at --analysis-conf (annotation.metrics)
  by_threshold.csv       the same, pooled, at each confidence cut
  precision_by_conf.csv  share of model outlines kept as EV, per confidence bin
  confusion.csv          model's class -> reviewer's class (or rejected)
  morphology_reviewed.csv  sizes of accepted EV-type outlines; role = REVIEWER's class
  morphology_model.csv     same tiles, model at --analysis-conf, geometric roles
                           (= size_profile_from_predictions + add_layer_roles)
  summary.txt
  figures/, qc_metrics.xlsx, qc_report.html   (annotation.web_qc.report)
and <review-dir>/export/reanalysis/ (prediction-cache format, `0 conf x y ...`,
EV-type outlines only; drawn = conf 1.0), export/compare/ + compare_all.pdf
(original model outlines | revised, changes colour-coded).

Truth = outlines accepted with an EV-type class (is_ev in classes.yaml), plus
outlines the reviewer drew. Predictions are the ORIGINAL model polygons from
the prediction files, so an edited outline is scored by IoU against the model's
version, not against itself. Thresholds below the cut the reviewer worked at
are not reported: outlines hidden below it were never judged.

Multi-class models: model class ids listed under model_class_to_class in the
classes file (classes_roboflow.yaml: 1 = non-EV) are not EV predictions, so
they are left out of the EV scores, precision_by_conf and morphology_model.
They still appear in confusion.csv. classes.yaml has no such key: unchanged.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from PIL import Image

from analysis.add_layer_roles import roles_for
from analysis.morphology import analyze_instances, save_morphology_csv
from annotation.metrics import _evaluate_predictions
from annotation.web_qc import report
from annotation.web_qc.server import (HERE, fixed_model_classes, load_classes,
                                      read_prediction_classes, read_predictions)
from training.train_yolo import load_predictions_yolo_format


def mask(poly_px, w, h):
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [np.round(np.asarray(poly_px)).astype(np.int32)], 1)
    return m.astype(bool)


def load_done(review_dir: Path):
    # One name per line (as server.py writes it); .split() broke names containing spaces.
    order = [l.strip() for l in (review_dir / "images_list.txt").read_text(encoding="utf-8").splitlines()
             if l.strip()]
    docs = []
    for name in order:
        p = review_dir / (Path(name).stem + ".json")
        if p.exists():
            d = json.loads(p.read_text(encoding="utf-8"))
            if d.get("done"):
                docs.append(d)
    return docs


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        if not rows:
            return
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def export_labels(docs, cfg, images_dir, out):
    ids = {c["name"]: c["id"] for c in cfg["classes"]}
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    coco = {"images": [], "annotations": [],
            "categories": [{"id": c["id"], "name": c["name"]} for c in cfg["classes"]]}
    n_obj = 0
    for k, d in enumerate(docs):
        w, h = d["width"], d["height"]
        shutil.copy2(images_dir / d["image"], out / "images" / d["image"])
        keep = [o for o in d["objects"] if o["status"] == "accepted"]
        lines = []
        coco["images"].append({"id": k, "file_name": d["image"], "width": w, "height": h})
        for o in keep:
            flat = [v for xy in o["poly"] for v in xy]
            lines.append(" ".join([str(ids[o["cls"]])] + ["%.6f" % v for v in flat]))
            px = [[x * w, y * h] for x, y in o["poly"]]
            xs, ys = [p[0] for p in px], [p[1] for p in px]
            a = 0.5 * abs(sum(px[i][0] * px[i - 1][1] - px[i - 1][0] * px[i][1]
                              for i in range(len(px))))
            coco["annotations"].append({
                "id": n_obj, "image_id": k, "category_id": ids[o["cls"]],
                "segmentation": [[round(v, 2) for p in px for v in p]],
                "bbox": [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)],
                "area": a, "iscrowd": 0})
            n_obj += 1
        (out / "labels" / (Path(d["image"]).stem + ".txt")).write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    (out / "_annotations.coco.json").write_text(json.dumps(coco), encoding="utf-8")
    names = {c["id"]: c["name"] for c in cfg["classes"]}
    (out / "data.yaml").write_text(
        "path: .\ntrain: images\nval: images\nnames:\n"
        + "".join(f"  {i}: {n}\n" for i, n in sorted(names.items())), encoding="utf-8")
    return n_obj


def export_reanalysis(docs, cfg, out):
    """Accepted EV-type outlines in the model's own cache format.

    Class id is always 0, as the single-class model writes it; the reviewer's
    class goes into morphology_reviewed.csv instead. Kept model outlines keep
    their confidence, drawn ones get 1.0.
    """
    is_ev = {c["name"]: bool(c["is_ev"]) for c in cfg["classes"]}
    out.mkdir(parents=True, exist_ok=True)
    for d in docs:
        lines = []
        for o in d["objects"]:
            if o["status"] == "accepted" and is_ev[o["cls"]]:
                conf = o["conf"] if o["source"] == "model" and o.get("conf") is not None else 1.0
                lines.append(" ".join(["0", "%.4f" % conf]
                                      + ["%.6f" % v for xy in o["poly"] for v in xy]))
        (out / (Path(d["image"]).stem + ".txt")).write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def morphology(docs, cfg, pred_dir, reanalysis_dir, analysis_conf, pixel_size):
    """(reviewed rows, model rows) for the same Done tiles, same sizing code.

    Both rasterize through load_predictions_yolo_format, so an untouched tile
    gives identical numbers. Reviewed: no confidence cut (accepted = kept) and
    role from the reviewer's class. Model: as the deck's analysis.
    """
    class_to_role = {v: k for k, v in cfg["role_to_class"].items()}
    is_ev = {c["name"]: bool(c["is_ev"]) for c in cfg["classes"]}
    fixed = fixed_model_classes(cfg)
    reviewed, model = [], []
    for d in docs:
        w, h, stem = d["width"], d["height"], Path(d["image"]).stem
        sample = d["image"].split("_")[0]
        cls = [o["cls"] for o in d["objects"] if o["status"] == "accepted" and is_ev[o["cls"]]]
        masks, confs, _ = load_predictions_yolo_format(reanalysis_dir / f"{stem}.txt", w, h)
        assert len(masks) == len(cls), d["image"]
        for rec in analyze_instances(masks, confidences=confs, pixel_size=pixel_size):
            c = cls[rec["object_id"]]
            reviewed.append({**rec, "image": d["image"], "sample": sample,
                             "reviewer_class": c, "role": class_to_role.get(c, "standalone")})

        masks, confs, mcls = load_predictions_yolo_format(pred_dir / f"{stem}.txt", w, h)
        kept = [i for i, c in enumerate(confs) if c >= analysis_conf and mcls[i] not in fixed]
        km = [masks[i] for i in kept]
        role, layers = roles_for(km) if km else ({}, {})
        for rec in analyze_instances(km, confidences=[confs[i] for i in kept],
                                     pixel_size=pixel_size):
            i = rec["object_id"]
            model.append({**rec, "image": d["image"], "sample": sample,
                          "role": role[i], "layer_count": layers[i]})
    return reviewed, model


def morph_rows(reviewed, model):
    """Per sample: EVs (inner layers excluded), median diameter, % multilayer; model vs reviewed."""
    def stats(rows):
        by = defaultdict(list)
        for r in rows:
            if r["role"] != "inner-layer":
                by[r["sample"]].append(r)
        out = {}
        for smp, rs in by.items():
            dia = [float(r["equivalent_diameter"]) for r in rs
                   if r.get("equivalent_diameter") is not None]
            out[smp] = (len(rs), float(np.median(dia)) if dia else float("nan"),
                        100.0 * sum(r["role"] == "multilayer" for r in rs) / len(rs))
        return out

    nan = (0, float("nan"), float("nan"))
    m, r = stats(model), stats(reviewed)
    return [(smp, *(x for pair in zip(m.get(smp, nan), r.get(smp, nan)) for x in pair))
            for smp in sorted(set(m) | set(r))]


MORPH_HEADER = ["sample", "EVs model", "EVs reviewed", "median nm model", "median nm reviewed",
                "% multilayer model", "% multilayer reviewed"]


def morph_summary(rows):
    lines = ["same tiles, model (conf cut, geometric roles) vs reviewed (reviewer's classes):",
             "  %-8s%12s%18s%16s" % ("sample", "EVs m/r", "median nm m/r", "% multi m/r")]
    for smp, em, er, dm, dr, pm, pr in rows:
        lines.append("  %-8s%12s%18s%16s" % (smp, "%d/%d" % (em, er), "%.0f/%.0f" % (dm, dr),
                                            "%.1f/%.1f" % (pm, pr)))
    return lines


def score(docs, cfg, pred_dir, analysis_conf, iou):
    is_ev = {c["name"]: bool(c["is_ev"]) for c in cfg["classes"]}
    fixed = fixed_model_classes(cfg)
    floor = max([d.get("conf_cut") or 0.0 for d in docs] + [0.0])
    cuts = [round(t, 2) for t in np.arange(0.05, 0.951, 0.05) if t >= floor - 1e-9]
    if analysis_conf not in cuts and analysis_conf >= floor:
        cuts = sorted(set(cuts + [analysis_conf]))

    per_image, pooled = [], {t: Counter() for t in cuts}
    bins, confusion = defaultdict(Counter), Counter()
    for d in docs:
        w, h = d["width"], d["height"]
        txt = pred_dir / (Path(d["image"]).stem + ".txt")
        polys, confs = read_predictions(txt, w, h)
        mcls = read_prediction_classes(txt) if fixed else [0] * len(polys)
        ev_pred = [k for k in range(len(polys)) if mcls[k] not in fixed]
        polys, confs = [polys[k] for k in ev_pred], [confs[k] for k in ev_pred]
        truth = [mask([[x * w, y * h] for x, y in o["poly"]], w, h) for o in d["objects"]
                 if o["status"] == "accepted" and is_ev[o["cls"]]]
        pred_masks = [mask(p, w, h) for p in polys]
        for t in cuts:
            r = _evaluate_predictions([m for m, c in zip(pred_masks, confs) if c >= t],
                                      truth, iou_threshold=iou)
            pooled[t].update({k: r[k] for k in ("tp", "fp", "fn")})
            if abs(t - analysis_conf) < 1e-9:
                per_image.append({"image": d["image"], **r})
        for o in d["objects"]:
            if o["source"] != "model" or o["status"] == "pending":
                continue
            kept = o["status"] == "accepted" and is_ev[o["cls"]]
            if not fixed or mcls[int(o["id"][1:])] not in fixed:   # id "m<i>" = line i
                b = min(int(o["conf"] * 10), 9) / 10
                bins[b]["n"] += 1
                bins[b]["kept_as_ev"] += kept
            confusion[(o["model_class"], o["cls"] if o["status"] == "accepted" else "rejected")] += 1
        n_drawn = sum(o["source"] == "human" and is_ev[o["cls"]] for o in d["objects"])
        if per_image and per_image[-1]["image"] == d["image"]:
            per_image[-1]["n_drawn_by_reviewer"] = n_drawn

    by_t = []
    for t in cuts:
        c = pooled[t]
        p = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else float("nan")
        r = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else float("nan")
        by_t.append({"conf_cut": t, "tp": c["tp"], "fp": c["fp"], "fn": c["fn"],
                     "precision": round(p, 4), "recall": round(r, 4),
                     "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0})
    by_conf = [{"conf_bin": f"{b:.1f}-{b + 0.1:.1f}", "n_reviewed": c["n"],
                "kept_as_ev": c["kept_as_ev"],
                "precision": round(c["kept_as_ev"] / c["n"], 4)}
               for b, c in sorted(bins.items())]
    conf_rows = [{"model_class": a, "reviewer": b, "n": n}
                 for (a, b), n in sorted(confusion.items(), key=lambda kv: -kv[1])]
    return per_image, by_t, by_conf, conf_rows, floor


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images-dir", required=True, type=Path)
    ap.add_argument("--predictions-dir", required=True, type=Path)
    ap.add_argument("--review-dir", required=True, type=Path)
    ap.add_argument("--classes", type=Path, default=HERE / "classes.yaml")
    ap.add_argument("--analysis-conf", type=float, default=0.5)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--pixel-size", type=float, default=None,
                    help="nm/px of the tiles, for sizes in nm (EVCore: 1.4724)")
    args = ap.parse_args()

    cfg = load_classes(args.classes)
    docs = load_done(args.review_dir)
    if not docs:
        raise SystemExit("no images marked Done yet")
    exp = args.review_dir / "export"
    n_obj = export_labels(docs, cfg, args.images_dir, exp)
    export_reanalysis(docs, cfg, exp / "reanalysis")
    compare_rows = report.export_compare(docs, cfg, args.images_dir, args.predictions_dir, exp)
    reviewed, model = morphology(docs, cfg, args.predictions_dir, exp / "reanalysis",
                                 args.analysis_conf, args.pixel_size)
    per_image, by_t, by_conf, conf_rows, floor = score(
        docs, cfg, args.predictions_dir, args.analysis_conf, args.iou)
    m = args.review_dir / "metrics"
    write_csv(m / "per_image.csv", per_image)
    write_csv(m / "by_threshold.csv", by_t)
    write_csv(m / "precision_by_conf.csv", by_conf)
    write_csv(m / "confusion.csv", conf_rows)
    for name, rows in (("morphology_reviewed.csv", reviewed), ("morphology_model.csv", model)):
        if rows:
            save_morphology_csv(rows, str(m / name))

    at = next((r for r in by_t if abs(r["conf_cut"] - args.analysis_conf) < 1e-9), None)
    lines = [f"images done: {len(docs)}   exported outlines: {n_obj}",
             f"reviewed down to confidence {floor:.2f}; IoU match >= {args.iou}"]
    if at:
        lines.append(f"at conf >= {args.analysis_conf}: precision {at['precision']:.3f}  "
                     f"recall {at['recall']:.3f}  F1 {at['f1']:.3f}  "
                     f"(tp {at['tp']}, fp {at['fp']}, fn {at['fn']})")
    lines.append("by threshold: " + "  ".join(
        f"{r['conf_cut']:.2f}:P{r['precision']:.2f}/R{r['recall']:.2f}" for r in by_t))
    lines.append("outlines left pending (e.g. hidden below the cut) are not reviewed and not exported")
    mrows = morph_rows(reviewed, model)
    lines += morph_summary(mrows)
    lines.append("2 tiles per sample: compare model vs reviewed here, not with the deck's 81 fields")
    (m / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    # Readable outputs. The xlsx goes last: if it is open in Excel the rest is already written.
    figs = report.make_figures(m / "figures", by_t, by_conf, conf_rows, reviewed, model,
                               args.analysis_conf, floor)
    changes = Counter()
    for _, _, n in compare_rows:
        changes.update(n)
    headline = [("images done", len(docs)), ("outlines exported", n_obj)]
    if at:
        headline += [(f"precision @ {args.analysis_conf}", f"{at['precision']:.2f}"),
                     (f"recall @ {args.analysis_conf}", f"{at['recall']:.2f}"),
                     (f"F1 @ {args.analysis_conf}", f"{at['f1']:.2f}")]
    headline += [(k, changes[k]) for k in ("rejected", "edited", "reclassed", "drawn") if changes[k]]
    cm_rows, cm_cols, cm = report.confusion_matrix(conf_rows) if conf_rows else ([], [], None)

    def rows(dicts):
        return (list(dicts[0].keys()), [list(r.values()) for r in dicts]) if dicts else ([], [])
    sheets = [("Summary", ["item", "value"], headline + [("reviewed down to confidence", floor),
                                                        ("IoU match", args.iou)]),
              ("Per image", *rows(per_image)),
              ("By threshold", *rows(by_t)),
              ("Precision by conf", *rows(by_conf)),
              ("Confusion", ["model \\ reviewer"] + cm_cols,
               [[r] + list(map(int, cm[i])) for i, r in enumerate(cm_rows)]),
              ("Sizes per sample", MORPH_HEADER, mrows),
              ("Changes per image", ["image", "kept", "reclassed", "edited", "drawn", "rejected", "unreviewed"],
               [[nm] + [n[k] for k in ("kept", "reclassed", "edited", "drawn", "rejected", "pending")]
                for nm, _, n in compare_rows])]
    report.write_html(m / "qc_report.html", headline, sheets, figs, compare_rows)
    xl = report.write_xlsx(m / "qc_metrics.xlsx", sheets)
    print(f"-> {exp}  and  {m}")
    print(f"   open {m / 'qc_report.html'}  (figures, tables, original vs revised)  ·  {xl.name}")


if __name__ == "__main__":
    main()
