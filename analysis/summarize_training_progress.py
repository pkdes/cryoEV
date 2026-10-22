"""Brief training-progress summary: three comparison tables for slides.

A. Old dedicated multilayer class vs single-class + post-hoc containment
B. More training data / a more balanced split
C. Model size (nano -> small -> medium) on the same dataset

Reads the harvested tables from compile_run_report.py (run it first, or this
script will). Every role number is put on one setting, containment 0.99 / size
ratio 0.75 on the test split: missing ones are recomputed from each run's cached
predictions with multilayer_role_analysis.build_cache()/evaluate() (CPU only, no
inference) and kept in role_fill_cache.json so reruns are instant.

Outputs (CryoAI/secondary analyses/training_progress_summary/):
    table_A_multilayer_approach.csv, table_B_training_data.csv, table_C_model_size.csv
    SUMMARY.md, training_progress_summary.html, role_fill_cache.json

Usage:
    python analysis/summarize_training_progress.py
"""
import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import compile_run_report as crr  # noqa: E402

CRYOAI = crr.CRYOAI
REPORT_DIR = CRYOAI / "secondary analyses" / "cross_run_report"
OUT_DIR = CRYOAI / "secondary analyses" / "training_progress_summary"
CONTAINMENT = (0.99, 0.75)

V3 = ("roboflow_20260824_cryoai_v3_singleclass", "roboflow - 20260824 cryoai v3")
V4 = ("roboflow_20260924_cryoai_v4_singleclass", "roboflow - 20260924 cryoai v4")
R = dict(
    c3="2026-06-03_215157_y8_640_8b_200ep",
    s100="2026-06-04_201830_y8_640_4b_200ep_Run_1", s75="2026-06-04_203358_y8_640_4b_200ep_Run_2",
    s50="2026-06-04_204500_y8_640_4b_200ep_Run_3", s25="2026-06-04_205501_y8_640_4b_200ep_Run_4",
    s768="2026-06-05_004849_y8_768_2b_300ep_Run_2",
    v3n="2026-08-24_190215_y8_1440_24b_300ep_Run_1", v3s="2026-09-23_134823_y8s_1440_24b_300ep_Run_2",
    v3m="2026-09-23_142437_y8m_1440_16b_300ep_Run_3",
    v4n="2026-09-24_134221_y8n_1440_24b_300ep_Run_1", v4m="2026-09-24_172246_y8m_1440_16b_300ep_Run_2",
    v3base="v3_model_baseline",
)
DATASET_OF = {R[k]: V3 for k in ("v3n", "v3s", "v3m")} | {R[k]: V4 for k in ("v4n", "v4m", "v3base")}
PARAMS_M = {"yolov8n": 3.4, "yolov8s": 11.8, "yolov8m": 27.3}  # yolov8-seg parameter counts (Ultralytics)


def load_report():
    if not (REPORT_DIR / "runs_master.csv").exists():
        sys.argv = [sys.argv[0]]
        crr.main()
    runs = {r["run"]: r for r in csv.DictReader((REPORT_DIR / "runs_master.csv").open(encoding="utf-8"))}
    pc = list(csv.DictReader((REPORT_DIR / "per_class_metrics_long.csv").open(encoding="utf-8")))
    return runs, pc


def f(x):
    return crr.fnum(x)


def native(pc, run, split, cls):
    return next((r for r in pc if r["run"] == run and r["split"] == split and r["cls"] == cls
                 and r["scheme"] == "native class"), None)


def sweep(run, split="test"):
    """P/R/F1 at conf 0.25 and the best-F1 cutoff, from the run's cached predictions."""
    ds = DATASET_OF[run][0]
    p = CRYOAI / "training outputs" / ds / "runs" / run / "confidence_sweep.csv"
    rows = [r for r in csv.DictReader(p.open()) if r["split"] == split]
    at = next(r for r in rows if abs(float(r["conf_threshold"]) - 0.25) < 1e-6)
    best = max(rows, key=lambda r: float(r["f1"]))
    return dict(P=f(at["precision"]), R=f(at["recall"]), F1=f(at["f1"]),
                best_F1=f(best["f1"]), best_conf=f(best["conf_threshold"]))


def roles(run, cache_file):
    """Test-split role P/R/F1 at CONTAINMENT, recomputed from cached predictions once."""
    store = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    key = f"{run}|test|{CONTAINMENT[0]}|{CONTAINMENT[1]}"
    if key not in store:
        import multilayer_role_analysis as mra
        ds, src = DATASET_OF[run]
        mra.SOURCE_ROOT = CRYOAI / "training images" / src
        mra.DATASET = CRYOAI / "training outputs" / ds
        mra.MODEL_PATH = mra.DATASET / "runs" / run / "weights" / "best.pt"
        mra.PRED_CACHE_DIR = mra.MODEL_PATH.parent.parent / "predictions"
        mra.SPLIT_MAP = {"test": "test"}
        cache = mra.build_cache()
        _, per_class, _ = mra.evaluate(cache, *CONTAINMENT)
        store[key] = dict(n_images=len(cache), **per_class)
        cache_file.write_text(json.dumps(store, indent=1))
        print(f"[roles] {run}: " + "  ".join(f"{c} F1={per_class[c]['f1']:.3f}" for c in ("multilayer", "inner-layer")))
    return store[key]


def r3(v):
    return None if v is None else round(v, 3)


def table_a(runs, pc, role):
    rows = []
    for key, approach in (("c3", "3-class: EV / multilayer EV / non-EV"),
                          ("s100", "2-class: EV / multilayer EV"), ("s768", "2-class: EV / multilayer EV")):
        run = R[key]
        ml = native(pc, run, "val", "multilayer EV")
        nat = [r for r in pc if r["run"] == run and r["split"] == "val" and r["scheme"] == "native class"]
        tp, fp, fn = (sum(int(r[k]) for r in nat) for k in ("tp", "fp", "fn"))
        rows.append(dict(approach=approach, run=run, model=runs[run]["model"], imgsz=runs[run]["imgsz"],
                         train_images=runs[run]["train_images"], scored_on="val",
                         images=runs[run]["val_images"], multilayer_labels=int(ml["n_gt"]),
                         multilayer_P=r3(f(ml["precision"])), multilayer_R=r3(f(ml["recall"])),
                         multilayer_F1=r3(f(ml["f1"])), all_objects_F1=r3(crr.prf(tp, fp, fn)[2])))
    for key in ("v3n", "v4n", "v4m"):
        run = R[key]
        ro = role(run)["multilayer"]
        rows.append(dict(approach="1-class EV (every layer) + containment", run=run, model=runs[run]["model"],
                         imgsz=runs[run]["imgsz"], train_images=runs[run]["train_images"], scored_on="test",
                         images=role(run)["n_images"], multilayer_labels=ro["tp"] + ro["fn"],
                         multilayer_P=r3(ro["precision"]), multilayer_R=r3(ro["recall"]),
                         multilayer_F1=r3(ro["f1"]), all_objects_F1=r3(sweep(run)["F1"])))
    return rows


def table_b(runs, pc, role):
    rows = []
    for key, label, data in (("v3base", "v3 split", "v3 Run_1 weights"), ("v4n", "v4 split", "v4 Run_1")):
        run = R[key]
        src = runs[R["v3n"]] if key == "v3base" else runs[run]
        sw, ro = sweep(run), role(run)
        rows.append(dict(block="B1: same y8n config, same 50 v4 test images", setting=label, run=data,
                         train_images=src["train_images"], val_images=src["val_images"],
                         test_images=src["test_images"],
                         F1_at_0_25=r3(sw["F1"]), best_F1=r3(sw["best_F1"]), best_conf=sw["best_conf"],
                         multilayer_F1=r3(ro["multilayer"]["f1"]), inner_F1=r3(ro["inner-layer"]["f1"]),
                         val_mask_mAP50_95=r3(f(src["val_mask_mAP50_95"]))))
    for key, pct in (("s25", "25%"), ("s50", "50%"), ("s75", "75%"), ("s100", "100%")):
        run = R[key]
        ev, ml = native(pc, run, "val", "EV"), native(pc, run, "val", "multilayer EV")
        rows.append(dict(block="B2: June 2-class data scaling, same 26-image val", setting=f"{pct} of train",
                         run=run, train_images=runs[run]["train_images"], val_images=runs[run]["val_images"],
                         test_images="", EV_F1=r3(f(ev["f1"])), multilayer_R=r3(f(ml["recall"])),
                         multilayer_F1=r3(f(ml["f1"])), val_mask_mAP50_95=r3(f(runs[run]["val_mask_mAP50_95"]))))
    return rows


def table_c(runs, pc, role):
    rows = []
    for block, keys in (("C1: v3 data (51 train / 99 test)", ("v3n", "v3s", "v3m")),
                        ("C2: v4 data (100 train / 50 test)", ("v4n", "v4m"))):
        for key in keys:
            run = R[key]
            m, sw, ro = runs[run], sweep(run), role(run)
            ev = native(pc, run, "test", "EV")
            rows.append(dict(block=block, model=m["model"], params_M=PARAMS_M.get(m["model"]), run=run,
                             batch=m["batch"], train_min=round(f(m["wall_time_s"]) / 60) if m["wall_time_s"] else None,
                             val_mask_mAP50_95=r3(f(m["val_mask_mAP50_95"])),
                             P=r3(sw["P"]), R=r3(sw["R"]), F1_at_0_25=r3(sw["F1"]),
                             best_F1=r3(sw["best_F1"]), best_conf=sw["best_conf"],
                             mask_IoU=r3(f(ev["mask_iou"])) if ev else None,
                             multilayer_F1=r3(ro["multilayer"]["f1"]), inner_F1=r3(ro["inner-layer"]["f1"])))
    return rows


TAKEAWAYS = {
    "A": ("Replacing the dedicated multilayer class with one EV class plus containment raised multilayer "
          "F1 from 0-0.52 to 0.70-0.84 and multilayer recall from 0.39-0.47 to 0.69-0.92, while overall "
          "detection F1 rose from 0.58-0.74 to 0.83-0.85.",
          "Old runs are scored on their own validation sets and new ones on held-out test sets, with "
          "different images and label schemes, so this is not a like-for-like comparison; the common-test-set "
          "evaluation will make it one. Containment on the ground-truth outlines reproduces the annotators' "
          "MV/inner labels at F1 0.996, so the geometry step is not the bottleneck."),
    "B": ("Same labels, same y8n model, same 50 test images: moving from 51 to 100 training images (v3 to v4 "
          "split) raised overall F1 by about 0.01 and multilayer and inner-layer F1 by about 0.07. In the June "
          "2-class runs, EV F1 rose steadily with each step up in training data.",
          "In B2 the multilayer numbers are not monotonic (the 25% run beat the 50% run); with 64 multilayer "
          "labels in val, differences of that size are within noise."),
    "C": ("On v3 (51 training images), y8s and y8m tied y8n on overall detection (best F1 0.850-0.854) but "
          "already improved multilayer F1 (0.70 to 0.78). On v4 (100 training images), y8m beats y8n on "
          "everything: F1 at 0.25 +0.027, multilayer F1 +0.105, inner-layer F1 +0.088.",
          "Each block compares models on the same test images. The y8m gain grows with more training data, "
          "and costs about twice the training time."),
}


def md_table(rows, cols):
    out = "| " + " | ".join(h for h, _ in cols) + " |\n|" + "---|" * len(cols) + "\n"
    for r in rows:
        out += "| " + " | ".join("" if r.get(k) is None else str(r.get(k)) for _, k in cols) + " |\n"
    return out


COLS = {
    "A": [("Approach", "approach"), ("Model", "model"), ("imgsz", "imgsz"), ("Train img", "train_images"),
          ("Scored on", "scored_on"), ("Images", "images"), ("Multilayer labels", "multilayer_labels"),
          ("Multilayer P", "multilayer_P"), ("Multilayer R", "multilayer_R"), ("Multilayer F1", "multilayer_F1"),
          ("All-object F1", "all_objects_F1")],
    "B1": [("Split", "setting"), ("Weights", "run"), ("Train / val / test", "split"), ("F1 @ 0.25", "F1_at_0_25"),
           ("Best F1", "best_F1"), ("at conf", "best_conf"), ("Multilayer F1", "multilayer_F1"),
           ("Inner F1", "inner_F1"), ("Val mask mAP50-95", "val_mask_mAP50_95")],
    "B2": [("Training data", "setting"), ("Train img", "train_images"), ("EV F1", "EV_F1"),
           ("Multilayer R", "multilayer_R"), ("Multilayer F1", "multilayer_F1"), ("Val mask mAP50-95", "val_mask_mAP50_95")],
    "C": [("Model", "model"), ("Params (M)", "params_M"), ("Batch", "batch"), ("Train min", "train_min"),
          ("Val mask mAP50-95", "val_mask_mAP50_95"), ("P", "P"), ("R", "R"), ("F1 @ 0.25", "F1_at_0_25"),
          ("Best F1", "best_F1"), ("at conf", "best_conf"), ("Mask IoU", "mask_IoU"),
          ("Multilayer F1", "multilayer_F1"), ("Inner F1", "inner_F1")],
}


def write_md(path, A, B, C):
    for r in B:
        r["split"] = f"{r['train_images']} / {r['val_images']} / {r['test_images'] or '-'}"
    L = ["# Training progress, May to September 2026\n",
         f"Generated {datetime.now():%Y-%m-%d} by `CryoEV Github/analysis/summarize_training_progress.py`. "
         "Scoring: detections matched one-to-one to labels at mask IoU >= 0.5. F1 @ 0.25 = confidence cutoff 0.25; "
         "best F1 = best cutoff found on that set. Multilayer and inner-layer F1 for containment models use "
         f"containment {CONTAINMENT[0]} / size ratio {CONTAINMENT[1]}, test split. Full detail: `../cross_run_report/`.\n"]
    for key, title, blocks in (("A", "A. Dedicated multilayer class vs containment", [("A", A)]),
                               ("B", "B. More training data, more balanced split",
                                [("B1", [r for r in B if r["block"].startswith("B1")]),
                                 ("B2", [r for r in B if r["block"].startswith("B2")])]),
                               ("C", "C. Model size: nano, small, medium",
                                [("C", [r for r in C if r["block"].startswith("C1")]),
                                 ("C", [r for r in C if r["block"].startswith("C2")])])):
        L.append(f"## {title}\n\n**{TAKEAWAYS[key][0]}**\n")
        for cols, rows in blocks:
            if rows and "block" in rows[0]:
                L.append(f"*{rows[0]['block']}*\n")
            L.append(md_table(rows, COLS[cols]))
        L.append(f"_{TAKEAWAYS[key][1]}_\n")
    path.write_text("\n".join(L), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    out = Path(ap.parse_args().out_dir)
    out.mkdir(parents=True, exist_ok=True)
    runs, pc = load_report()
    role = lambda run: roles(run, out / "role_fill_cache.json")
    A, B, C = table_a(runs, pc, role), table_b(runs, pc, role), table_c(runs, pc, role)
    for name, rows in (("table_A_multilayer_approach.csv", A), ("table_B_training_data.csv", B),
                       ("table_C_model_size.csv", C)):
        cols = []
        for r in rows:
            cols += [k for k in r if k not in cols]
        with (out / name).open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
    write_md(out / "SUMMARY.md", A, B, C)
    data = dict(generated=f"{datetime.now():%Y-%m-%d}", containment=CONTAINMENT, A=A, B=B, C=C,
                takeaways=TAKEAWAYS)
    tpl = (Path(__file__).parent / "training_progress_template.html").read_text(encoding="utf-8")
    blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    (out / "training_progress_summary.html").write_text(tpl.replace("/*DATA*/null", blob), encoding="utf-8")
    print(f"[DONE] {out}")


if __name__ == "__main__":
    main()
