"""Compile a cross-run report of every YOLO training run, from metrics already on disk.

Read-only harvest (no inference, no GPU). Run folders on disk are the source of
truth; EXPERIMENT_LOG.csv is only cross-checked, since it misses runs and has a few
wrong rows. Each metric keeps its split, thresholds and source file, because the
eras were scored differently and are not directly comparable.

Outputs (CryoAI/secondary analyses/cross_run_report/):
    runs_master.csv            one row per run, status main | appendix
    per_class_metrics_long.csv run x split x class/role
    imagesets.csv              raw exports: images per split, annotations per category
    data_quality_notes.csv     log vs disk discrepancies
    report_data.json           everything above
    cross_run_report.html      sortable/filterable page (cross_run_report_template.html + the data)
    REPORT.md

Usage:
    python analysis/compile_run_report.py
"""
import argparse
import csv
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import yaml

CRYOAI = Path(__file__).resolve().parent.parent.parent / "CryoAI"
OUT_ROOT = CRYOAI / "training outputs"
IMG_ROOT = CRYOAI / "training images"
IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
MIN_EPOCHS = 30

# One entry per dataset folder under training outputs/. "era" groups the summary.
CAMPAIGNS = {
    "quick_bench": dict(
        era="1. May: Yifei baseline", imageset="roboflow - 202512 Yifei",
        labels="Yifei 202512 (1 class: vesicle)", approach="1-class vesicle",
        split_method="Yifei export split", note="Re-training Yifei's dataset to benchmark the setup"),
    "roboflow_202512_yifei": dict(
        era="1. May: Yifei baseline", imageset="roboflow - 202512 Yifei",
        labels="Yifei 202512 (1 class: vesicle)", approach="1-class vesicle",
        split_method="Yifei export split", note="Sanity check only"),
    "roboflow_20260529": dict(
        era="2. Jun: 3-class", imageset="roboflow - 20260529",
        labels="20260529 export (EV / multilayer EV / non-EV EP)",
        approach="3-class: dedicated multilayer EV class + non-EV",
        split_method="Roboflow export split", note="non-EV EP never detected; multilayer EV ~0-2 of 7 found"),
    "roboflow20260604_backup": dict(
        era="3. Jun: 2-class", imageset="roboflow20260604",
        labels="20260604 export (EV / multilayer EV; non-EV dropped)",
        approach="2-class: dedicated multilayer EV class",
        split_method="random 80/10/10 re-split", note="Only 5 multilayer instances in val"),
    "roboflow20260604_stratified": dict(
        era="3. Jun: 2-class", imageset="roboflow20260604",
        labels="20260604 export (EV / multilayer EV; non-EV dropped)",
        approach="2-class: dedicated multilayer EV class",
        split_method="stratified re-split (half of multilayer images in val)",
        note="Multilayer recall ceiling ~0.39 -> motivated the single-class approach"),
    "vldl": dict(
        era="4. Jun: VLDL side experiment", imageset="vldl_imageset",
        labels="VLDL tiles (EV / non-EV EP)", approach="2-class EV / non-EV EP (tiles)",
        split_method="tiles from 2 images", note="Side experiment on lipoprotein contamination"),
    "roboflow_20260713_all_layer_singleclass": dict(
        era="5. Jul-Sep: single-class EV + containment", imageset="roboflow - 20260713 all layer label",
        labels="All-layer v1 (every membrane layer its own polygon)",
        approach="1-class EV (all layers) + post-hoc containment",
        split_method="Roboflow export split (no test)", note="Established overlap_mask=False and imgsz 1440 + rect"),
    "roboflow_20260824_cryoai_v3_singleclass": dict(
        era="5. Jul-Sep: single-class EV + containment", imageset="roboflow - 20260824 cryoai v3",
        labels="All-layer v3 (adds Overlapping)", approach="1-class EV (all layers) + post-hoc containment",
        split_method="Roboflow export split", note="First campaign with a held-out test set"),
    "roboflow_20260924_cryoai_v4_singleclass": dict(
        era="5. Jul-Sep: single-class EV + containment", imageset="roboflow - 20260924 cryoai v4",
        labels="All-layer v3 labels (unchanged)", approach="1-class EV (all layers) + post-hoc containment",
        split_method="Roboflow re-split, rebalanced 100/20/50", note="Same images/labels as v3, more in train"),
}
STRATIFIED_SUBSETS = {"stratified_25pct": "25%", "stratified_50pct": "50%", "stratified_75pct": "75%"}

# Role analyses live per dataset, not per run; which run each folder scored comes from its log.
ROLE_FOLDERS = {
    ("roboflow_20260713_all_layer_singleclass", "multilayer_role_analysis"): "2026-07-13_181034_y8_1440_24b_300ep_Run_14",
    ("roboflow_20260824_cryoai_v3_singleclass", "multilayer_role_analysis"): "2026-08-24_190215_y8_1440_24b_300ep_Run_1",
    ("roboflow_20260924_cryoai_v4_singleclass", "multilayer_role_analysis_y8n"): "2026-09-24_134221_y8n_1440_24b_300ep_Run_1",
    ("roboflow_20260924_cryoai_v4_singleclass", "multilayer_role_analysis_y8m"): "2026-09-24_172246_y8m_1440_16b_300ep_Run_2",
    ("roboflow_20260924_cryoai_v4_singleclass", "multilayer_role_analysis_v3model"): "v3_model_baseline",
}
ROLES = ["multilayer", "inner-layer", "standalone"]

# Harmonized headline columns: old native classes and new geometric roles side by side.
HEADLINE = {"single": {"EV": "native", "standalone": "role"},
            "multilayer": {"multilayer EV": "native", "multilayer": "role"},
            "inner": {"inner-layer": "role"},
            "nonEV": {"non-EV EP": "native"}}


def rebase(p):
    """Map a path recorded on another machine (C:/Users/pujan/...) onto this CryoAI folder."""
    s = str(p).replace("\\", "/")
    for anchor in ("training outputs/", "training images/"):
        if anchor in s:
            return CRYOAI / (anchor + s.split(anchor, 1)[1])
    return Path(p)


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def count_split(img_dir):
    """Images in a YOLO split dir, plus instances per class id from its labels dir."""
    if not img_dir.is_dir():
        return None, None
    imgs = [f for f in img_dir.iterdir() if f.suffix.lower() in IMG_EXT]
    lbl_dir = Path(str(img_dir).replace("\\", "/").replace("/images", "/labels"))
    inst = Counter()
    if lbl_dir.is_dir():
        for f in lbl_dir.glob("*.txt"):
            for line in f.read_text().splitlines():
                if line.strip():
                    inst[int(float(line.split()[0]))] += 1
    return len(imgs), inst


def dataset_splits(data_path):
    """Image and instance counts per split for the dataset a run actually trained on."""
    yml = rebase(data_path)
    if yml.is_dir():  # quick_bench pointed at a raw export; local copy has a suffix
        cands = [yml / "dataset.yaml"] + sorted(IMG_ROOT.glob(yml.name + "*/dataset.yaml"))
        yml = next((c for c in cands if c.exists()), yml)
    if not yml.is_file():
        if not (yml.parent / "images").is_dir():
            return {}, [], str(yml)
        # the yaml is gone but its images/labels dirs remain
        d = {"path": str(yml.parent), "train": "images/train", "val": "images/val", "test": "images/test"}
    else:
        d = yaml.safe_load(yml.read_text())
    base = rebase(d["path"]) if d.get("path") else yml.parent
    if not base.exists():
        base = yml.parent
    names = d.get("names", [])
    names = [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)
    out = {}
    for split in ("train", "val", "test"):
        if d.get(split):
            n, inst = count_split(base / d[split])
            if n is not None:
                out[split] = (n, inst)
    return out, names, str(yml.relative_to(CRYOAI)) if CRYOAI in yml.parents else str(yml)


def run_date(run):
    m = re.search(r"(\d{4})-?(\d{2})-?(\d{2})", run.name)
    if m:
        return "-".join(m.groups())
    f = next((run / n for n in ("results.csv", "args.yaml") if (run / n).exists()), run)
    return datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d")


def best_epoch(results_csv):
    rows = list(csv.DictReader(results_csv.open()))
    rows = [{k.strip(): v for k, v in r.items()} for r in rows]
    if not rows:
        return None

    def fit(r):
        return sum(0.1 * (fnum(r.get(f"metrics/mAP50({t})")) or 0) + 0.9 * (fnum(r.get(f"metrics/mAP50-95({t})")) or 0)
                   for t in ("B", "M"))
    b = max(rows, key=fit)
    g = lambda k: fnum(b.get(k))
    mp, mr = g("metrics/precision(M)"), g("metrics/recall(M)")
    return dict(epochs_done=len(rows), best_epoch=int(float(b["epoch"])) + (0 if rows[0]["epoch"].strip() == "1" else 1),
                val_box_P=g("metrics/precision(B)"), val_box_R=g("metrics/recall(B)"),
                val_box_mAP50=g("metrics/mAP50(B)"), val_box_mAP50_95=g("metrics/mAP50-95(B)"),
                val_mask_P=mp, val_mask_R=mr,
                val_mask_F1=(2 * mp * mr / (mp + mr)) if mp and mr else None,
                val_mask_mAP50=g("metrics/mAP50(M)"), val_mask_mAP50_95=g("metrics/mAP50-95(M)"),
                train_time_s=fnum(rows[-1].get("time")))


def wall_time(run):
    f = run / "run_summary.txt"
    if f.exists():
        m = re.search(r"Wall time:\s*(\d+):(\d+):(\d+)", f.read_text())
        if m:
            h, mi, s = map(int, m.groups())
            return h * 3600 + mi * 60 + s
    return None


def aug_summary(a):
    keys = ["mosaic", "mixup", "copy_paste", "degrees", "flipud", "fliplr", "scale"]
    vals = {k: a.get(k) for k in keys}
    if all(not fnum(v) for v in vals.values()):
        return "none"
    return " ".join(f"{k}={v}" for k, v in vals.items() if fnum(v))


def per_class_files(run):
    """(split, csv path, thresholds) for every per-class metrics CSV a run has."""
    found = []
    for split in ("train", "val", "test"):
        for f in (run / f"per_class_metrics_{split}.csv", run / "inference" / split / f"metrics_per_class_{split}.csv"):
            if f.exists():
                found.append((split, f))
    if (run / "per_class_metrics.csv").exists():
        found.append(("val", run / "per_class_metrics.csv"))
    out = []
    for split, f in found:
        conf, match = 0.25, 0.5
        mt = f.parent / f"metrics_{split}.txt"
        if mt.exists():
            t = mt.read_text()
            c = re.search(r"conf_threshold:\s*([\d.]+)", t)
            m = re.search(r"match_threshold:\s*([\d.]+)", t)
            conf, match = (float(c.group(1)) if c else conf), (float(m.group(1)) if m else match)
        out.append((split, f, conf, match))
    return out


def parse_role_matrix(txt):
    rows = {}
    for line in txt.read_text().splitlines():
        parts = line.split()
        if parts and parts[0] in ROLES + ["background"] and len(parts) == 5:
            rows[parts[0]] = list(map(int, parts[1:]))
    labels = ROLES + ["background"]
    out = {}
    for i, role in enumerate(ROLES):
        tp = rows[role][i]
        fn = sum(rows[role]) - tp
        fp = sum(rows[l][i] for l in labels) - tp
        out[role] = (tp, fp, fn)
    return out


def role_rows(campaign_dir):
    """Per-split role P/R/F1 from saved confusion matrices, prefer containment 0.99 / ratio 0.75."""
    rows = []
    for (camp, folder), run_name in ROLE_FOLDERS.items():
        d = OUT_ROOT / camp / folder
        if camp != campaign_dir or not d.is_dir():
            continue
        mats = sorted(d.glob("confusion_matrix_thr*_ratio*.txt"))
        thr_avail = sorted({re.search(r"thr([\d.]+)_", m.name).group(1) for m in mats})
        thr = "0.99" if "0.99" in thr_avail else thr_avail[-1]
        for m in mats:
            mm = re.match(r"confusion_matrix_thr([\d.]+)_ratio([\w.]+?)(?:_(train|val|test|combined))?\.txt$", m.name)
            if not mm or mm.group(1) != thr:
                continue
            split = mm.group(3) or "train+val"
            for role, (tp, fp, fn) in parse_role_matrix(m).items():
                p, r, f1 = prf(tp, fp, fn)
                rows.append(dict(run=run_name, campaign=camp, split=split, scheme="geometric role", cls=role,
                                 n_gt=tp + fn, n_pred=tp + fp, tp=tp, fp=fp, fn=fn,
                                 precision=p, recall=r, f1=f1, mask_iou=None, conf=0.25, match_iou=0.5,
                                 containment=f"{thr} / {mm.group(2)}",
                                 source=str(m.relative_to(CRYOAI))))
    return rows


def conf_sweep(run, split):
    f = run / "confidence_sweep.csv"
    if not f.exists():
        return {}
    rows = [r for r in csv.DictReader(f.open()) if r["split"] == split]
    if not rows:
        return {}
    at = next((r for r in rows if abs(float(r["conf_threshold"]) - 0.25) < 1e-6), None)
    best = max(rows, key=lambda r: float(r["f1"]))
    return dict(sweep_F1_at_0_25=fnum(at["f1"]) if at else None,
                sweep_best_F1=fnum(best["f1"]), sweep_best_conf=fnum(best["conf_threshold"]))


def find_runs():
    runs = []
    for camp in sorted(p for p in OUT_ROOT.iterdir() if p.is_dir()):
        if camp.name.startswith("vldl_run"):
            runs.append(("vldl", camp, False))
        elif camp.name == "quick_bench":
            runs += [(camp.name, r, False) for r in sorted(camp.iterdir()) if (r / "args.yaml").exists()]
        else:
            for sub, probe in (("runs", False), ("probes", True)):
                if (camp / sub).is_dir():
                    runs += [(camp.name, r, probe) for r in sorted((camp / sub).iterdir()) if r.is_dir()]
    return runs


def read_log():
    f = CRYOAI / "EXPERIMENT_LOG.csv"
    rows = list(csv.reader(f.open(encoding="utf-8")))
    hdr = rows[0] + ["peak_gpu_memory_gb", "total_training_time_sec", "avg_epoch_time_sec", "optimizer", "augmentations"]
    return [dict(zip(hdr, r)) for r in rows[1:] if r]


def imagesets():
    out = []
    for d in sorted(p for p in IMG_ROOT.iterdir() if p.is_dir()):
        for split_dir in sorted(p for p in d.iterdir() if p.is_dir()):
            js = split_dir / "_annotations.coco.json"
            if js.exists():
                c = json.loads(js.read_text(encoding="utf-8"))
                cats = {k["id"]: k["name"] for k in c["categories"]}
                cnt = Counter(cats[a["category_id"]] for a in c["annotations"])
                out.append(dict(imageset=d.name, split=split_dir.name, format="COCO", images=len(c["images"]),
                                annotations=sum(cnt.values()),
                                per_category="; ".join(f"{k}={v}" for k, v in sorted(cnt.items()))))
        yml = next((y for y in (d / "dataset.yaml", d / "data.yaml") if y.exists()), None)
        if yml and (d / "images").is_dir():
            names = yaml.safe_load(yml.read_text()).get("names", [])
            names = [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)
            for split_dir in sorted((d / "images").iterdir()):
                n, inst = count_split(split_dir)
                if n is not None:
                    out.append(dict(imageset=d.name, split=split_dir.name, format="YOLO", images=n,
                                    annotations=sum(inst.values()),
                                    per_category="; ".join(f"{names[k] if k < len(names) else k}={v}" for k, v in sorted(inst.items()))))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=str(CRYOAI / "secondary analyses" / "cross_run_report"))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log = read_log()
    log_by_name = {r["experiment_name"]: r for r in log}
    notes, master, long_rows = [], [], []
    role_cache = {}

    for camp, run, probe in find_runs():
        meta = dict(CAMPAIGNS.get(camp, dict(era="?", imageset="?", labels="?", approach="?", split_method="?", note="")))
        a = yaml.safe_load((run / "args.yaml").read_text()) if (run / "args.yaml").exists() else {}
        row = dict(era=meta["era"], campaign=camp, run=run.name, date=run_date(run),
                   imageset=meta["imageset"], labels=meta["labels"], approach=meta["approach"],
                   split_method=meta["split_method"], campaign_note=meta["note"])
        if run.name == "v3_model_baseline":
            row.update(date="2026-09-24", approach=meta["approach"] + " (evaluation only: v3 Run_1 weights on v4 splits)")

        # Split actually trained on (handles the stratified 25/50/75% subsets)
        data = a.get("data") or (OUT_ROOT / camp / "dataset.yaml")
        splits, names, used = dataset_splits(data)
        sub = next((v for k, v in STRATIFIED_SUBSETS.items() if k in str(data)), None)
        if sub:
            row["split_method"] += f", {sub} train subset"
        row["dataset_yaml"] = used
        row["classes"] = " / ".join(names)
        for s in ("train", "val", "test"):
            n, inst = splits.get(s, (None, None))
            row[f"{s}_images"] = n
            row[f"{s}_instances"] = sum(inst.values()) if inst else (0 if n else None)
        # Parameters
        row.update(model=Path(str(a.get("model", ""))).stem.replace("-seg", ""), epochs=a.get("epochs"),
                   imgsz=a.get("imgsz"), batch=a.get("batch"), patience=a.get("patience"),
                   optimizer=a.get("optimizer"), lr0=a.get("lr0"), lrf=a.get("lrf"),
                   overlap_mask=a.get("overlap_mask"), rect=a.get("rect"), mask_ratio=a.get("mask_ratio"),
                   seed=a.get("seed"), augmentation=aug_summary(a) if a else "")
        if run.name == "v3_model_baseline":
            row.update(model="yolov8n", imgsz=1440)
        # Training outcome
        rc = run / "results.csv"
        be = best_epoch(rc) if rc.exists() else None
        row.update(be or {})
        row["wall_time_s"] = wall_time(run) or (be or {}).get("train_time_s")
        row["has_weights"] = (run / "weights" / "best.pt").exists()

        # Status
        if probe:
            status, why = "appendix", "probe (2-epoch memory check)"
        elif "sanity" in run.name:
            status, why = "appendix", "sanity check"
        elif run.name == "v3_model_baseline":
            status, why = "main", "evaluation only"
        elif not rc.exists():
            status, why = "appendix", "no results.csv (crashed / out of memory / never started)"
        elif not row["has_weights"]:
            status, why = "appendix", "no weights"
        elif row["epochs_done"] < MIN_EPOCHS:
            status, why = "appendix", f"stopped after {row['epochs_done']} epochs"
        else:
            status, why = "main", ""
        row.update(status=status, status_reason=why)

        # Per-class object-level metrics
        for split, f, conf, match in per_class_files(run):
            for r in csv.DictReader(f.open()):
                long_rows.append(dict(run=run.name, campaign=camp, split=split, scheme="native class",
                                      cls=r["class_name"], n_gt=int(r["n_gt"]), n_pred=int(r["n_pred"]),
                                      tp=int(r["tp"]), fp=int(r["fp"]), fn=int(r["fn"]),
                                      precision=fnum(r["precision"]), recall=fnum(r["recall"]), f1=fnum(r["f1"]),
                                      mask_iou=fnum(r["avg_iou_matched"]), conf=conf, match_iou=match,
                                      containment="", source=str(f.relative_to(CRYOAI))))
        if camp not in role_cache:
            role_cache[camp] = role_rows(camp)
        long_rows += [r for r in role_cache[camp] if r["run"] == run.name]

        # Log cross-check
        lg = log_by_name.get(run.name)
        row["in_experiment_log"] = bool(lg)
        if not lg and status == "main" and run.name != "v3_model_baseline":
            notes.append(dict(run=run.name, campaign=camp, issue="Run on disk but missing from EXPERIMENT_LOG.csv"))
        if lg:
            lp = lg.get("best_model_path", "").replace("\\", "/")
            if lp and "training outputs/" in lp and f"training outputs/{camp}/" not in lp:
                notes.append(dict(run=run.name, campaign=camp,
                                  issue=f"Log path points to '{lp.split('training outputs/')[1].split('/')[0]}', run is in '{camp}'"))
            lt = lg.get("train_images")
            if lt and row["train_images"] is not None and lt.isdigit() and int(lt) != row["train_images"]:
                notes.append(dict(run=run.name, campaign=camp,
                                  issue=f"Log says {lt} train images; dataset it trained on has {row['train_images']}"))
        master.append(row)

    names_on_disk = {r["run"] for r in master}
    for lg in log:
        if lg["experiment_name"] not in names_on_disk:
            notes.append(dict(run=lg["experiment_name"], campaign=lg["dataset_id"],
                              issue="In EXPERIMENT_LOG.csv but no run folder on disk"))
            master.append(dict(era="", campaign=lg["dataset_id"], run=lg["experiment_name"],
                               date=lg["timestamp"][:10], status="appendix",
                               status_reason="log only, no run folder (failed run)",
                               model=lg.get("model"), imgsz=lg.get("imgsz"), batch=lg.get("batch_size")))
    notes.append(dict(run="(file)", campaign="", issue="EXPERIMENT_LOG.csv header has 21 columns; rows 3+ have 26 fields"))

    # Headline numbers: test if scored, else val. Micro-averaged "all" sums TP/FP/FN over native classes.
    by_run = {}
    for r in long_rows:
        by_run.setdefault(r["run"], []).append(r)
    for row in master:
        rs = by_run.get(row["run"], [])
        if not row.get("classes") and rs:  # dataset yaml lost; class names survive in the metrics
            row["classes"] = " / ".join(dict.fromkeys(r["cls"] for r in rs if r["scheme"] == "native class"))
        splits = {r["split"] for r in rs}
        hs = next((s for s in ("test", "val", "train+val") if s in splits), None)
        row["headline_split"] = hs or ""
        if not hs:
            continue
        nat = [r for r in rs if r["split"] == hs and r["scheme"] == "native class"]
        if nat:
            tp, fp, fn = (sum(r[k] for r in nat) for k in ("tp", "fp", "fn"))
            p, rcl, f1 = prf(tp, fp, fn)
            iou = sum((r["mask_iou"] or 0) * r["tp"] for r in nat) / tp if tp else None
            row.update(all_P=p, all_R=rcl, all_F1=f1, all_mask_IoU=iou,
                       score_conf=nat[0]["conf"], score_match_iou=nat[0]["match_iou"])
        # In single-class runs native "EV" is every layer, not single-layer EVs
        multiclass = len({r["cls"] for r in rs if r["scheme"] == "native class"}) > 1
        for col, wanted in HEADLINE.items():
            ok = lambda r: r["cls"] in wanted and (r["cls"] != "EV" or multiclass)
            hit = next((r for r in rs if r["split"] == hs and ok(r)), None)
            if not hit:  # 0713 Run_14's role analysis was only saved for train+val combined
                hit = next((r for r in rs if r["split"] == "train+val" and ok(r)), None)
            if hit and hit["split"] != hs:
                row["role_split"] = hit["split"]
            if hit:
                row.update({f"{col}_P": hit["precision"], f"{col}_R": hit["recall"], f"{col}_F1": hit["f1"],
                            f"{col}_n_gt": hit["n_gt"], f"{col}_source": wanted[hit["cls"]]})
        row["role_containment"] = next((r["containment"] for r in rs if r["scheme"] == "geometric role"), "")
        row.update(conf_sweep(OUT_ROOT / row["campaign"] / "runs" / row["run"], hs))

    master.sort(key=lambda r: (r["status"] != "main", r.get("date") or "", r["run"]))
    imsets = imagesets()

    def write_csv(name, rows):
        cols = []
        for r in rows:
            cols += [k for k in r if k not in cols]
        with (out_dir / name).open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in rows)

    write_csv("runs_master.csv", master)
    write_csv("per_class_metrics_long.csv", long_rows)
    write_csv("imagesets.csv", imsets)
    write_csv("data_quality_notes.csv", notes)
    data = dict(generated=datetime.now().strftime("%Y-%m-%d"), campaigns=CAMPAIGNS, runs=master,
                per_class=long_rows, imagesets=imsets, notes=notes)
    (out_dir / "report_data.json").write_text(json.dumps(data, default=str, indent=1), encoding="utf-8")
    # Same data into the self-contained HTML page (template lives next to this script)
    tpl = (Path(__file__).parent / "cross_run_report_template.html").read_text(encoding="utf-8")
    blob = json.dumps(data, default=str, separators=(",", ":")).replace("</", "<\/")
    (out_dir / "cross_run_report.html").write_text(tpl.replace("/*DATA*/null", blob), encoding="utf-8")
    write_markdown(out_dir / "REPORT.md", master, long_rows, imsets, notes)
    n_main = sum(r["status"] == "main" for r in master)
    print(f"{len(master)} runs ({n_main} main, {len(master) - n_main} appendix), "
          f"{len(long_rows)} per-class rows, {len(notes)} notes -> {out_dir}")


def fmt(v, nd=3):
    if v is None or v == "":
        return ""
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def md_table(rows, cols):
    head = "| " + " | ".join(h for h, _ in cols) + " |\n|" + "---|" * len(cols) + "\n"
    return head + "".join("| " + " | ".join(fmt(r.get(k)) if isinstance(k, str) else k(r) for _, k in cols) + " |\n"
                          for r in rows)


def write_markdown(path, master, long_rows, imsets, notes):
    main = [r for r in master if r["status"] == "main"]
    app = [r for r in master if r["status"] != "main"]
    split = lambda r: "/".join(fmt(r.get(f"{s}_images")) or "-" for s in ("train", "val", "test"))
    L = ["# Cross-run training report (May 2026 to now)\n",
         "Generated by `CryoEV Github/analysis/compile_run_report.py` from metrics already on disk (no re-scoring).\n",
         "## Read this first\n",
         "- **Numbers are not directly comparable across eras.** Before v3 there was no test set, so older runs are "
         "scored on validation (5-37 images). Thresholds also differ: conf 0.25 / match IoU 0.5 unless noted; "
         "the May sanity runs used conf 0.35 / match 0.30.",
         "- **Multilayer means two different things.** Before July it is a *native class* the model predicted "
         "(`multilayer EV`, outer outline only). From July it is a *geometric role*: the model predicts every layer "
         "as one EV class, and a detection that contains others is multilayer. `single` = native EV (old) or "
         "standalone role (new); `inner` exists only in the new scheme.",
         "- **all_*** = TP/FP/FN summed across native classes on the headline split; for single-class runs that is "
         "every membrane layer. **Mask IoU** = mean mask IoU of matched detections (mask fit).",
         "- **Val mask mAP** comes from Ultralytics at the best epoch, on each run's own val set.",
         "- Role scores exist only for runs that had a role analysis: 0713 Run_14, v3 Run_1, v4 Run_1/Run_2 and "
         "the v3 baseline. v4 y8m per-split matrices were saved at containment 0.85, the others at 0.99 "
         "(the threshold changes role F1 by at most 0.017).\n",
         "## Best run per campaign\n"]
    best = {}
    for r in main:
        k = r["campaign"]
        key = r.get("all_F1") or r.get("val_mask_mAP50_95") or 0
        if k not in best or key > (best[k].get("all_F1") or best[k].get("val_mask_mAP50_95") or 0):
            best[k] = r
    best_rows = sorted(best.values(), key=lambda r: r["date"])
    L.append(md_table(best_rows, [
        ("Era", "era"), ("Campaign", "campaign"), ("Run", "run"), ("Approach", "approach"),
        ("Train/val/test", split), ("Model", "model"), ("imgsz", "imgsz"), ("Scored on", "headline_split"),
        ("All F1", "all_F1"), ("Single F1", "single_F1"), ("Multilayer F1", "multilayer_F1"),
        ("Inner F1", "inner_F1"), ("Mask IoU", "all_mask_IoU"), ("Val mask mAP50-95", "val_mask_mAP50_95")]))
    L.append("\n## Image sets\n")
    L.append(md_table(imsets, [("Image set", "imageset"), ("Split", "split"), ("Format", "format"),
                               ("Images", "images"), ("Annotations", "annotations"), ("Per category", "per_category")]))
    L.append("\n## All completed runs\n")
    L.append(md_table(main, [
        ("Date", "date"), ("Campaign", "campaign"), ("Run", "run"), ("Classes", "classes"),
        ("Split method", "split_method"), ("Train/val/test img", split), ("Model", "model"),
        ("imgsz", "imgsz"), ("Batch", "batch"), ("Epochs (done/max)", lambda r: f"{fmt(r.get('epochs_done'))}/{fmt(r.get('epochs'))}"),
        ("Opt", "optimizer"), ("lr0", "lr0"), ("overlap_mask", "overlap_mask"), ("rect", "rect"),
        ("mask_ratio", "mask_ratio"), ("Aug", lambda r: "none" if r.get("augmentation") == "none" else "std"),
        ("Val box mAP50-95", "val_box_mAP50_95"), ("Val mask P", "val_mask_P"), ("Val mask R", "val_mask_R"),
        ("Val mask mAP50-95", "val_mask_mAP50_95"), ("Scored on", "headline_split"),
        ("All P", "all_P"), ("All R", "all_R"), ("All F1", "all_F1"), ("Mask IoU", "all_mask_IoU"),
        ("Single F1", "single_F1"), ("Multilayer P", "multilayer_P"), ("Multilayer R", "multilayer_R"),
        ("Multilayer F1", "multilayer_F1"), ("Inner F1", "inner_F1"), ("non-EV F1", "nonEV_F1")]))
    L.append("\n## Per-class and per-role metrics (every split saved)\n")
    L.append(md_table(sorted(long_rows, key=lambda r: (r["campaign"], r["run"], r["split"], r["scheme"])), [
        ("Run", "run"), ("Split", "split"), ("Scheme", "scheme"), ("Class / role", "cls"), ("GT", "n_gt"),
        ("Pred", "n_pred"), ("TP", "tp"), ("FP", "fp"), ("FN", "fn"), ("P", "precision"), ("R", "recall"),
        ("F1", "f1"), ("Mask IoU", "mask_iou"), ("Conf / match", lambda r: f"{r['conf']} / {r['match_iou']}"),
        ("Containment", "containment")]))
    L.append("\n## Appendix: sanity checks, probes, failed and short runs\n")
    L.append(md_table(app, [("Date", "date"), ("Campaign", "campaign"), ("Run", "run"), ("Model", "model"),
                            ("imgsz", "imgsz"), ("Batch", "batch"), ("Epochs done", "epochs_done"),
                            ("Why not in main table", "status_reason")]))
    L.append("\n## Data-quality notes (EXPERIMENT_LOG.csv vs disk)\n")
    L.append(md_table(notes, [("Run", "run"), ("Campaign", "campaign"), ("Issue", "issue")]))
    path.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
