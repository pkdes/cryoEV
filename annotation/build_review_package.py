"""
Build an offline, browser-based label review package from a Roboflow COCO export
plus a training run's cached predictions.

The package is a folder (and a .zip of it) holding `index.html`, `data.js` and
downscaled grayscale JPEGs. Opening index.html in any browser needs no Python,
no install and no network. The reviewer edits the existing annotations with
the model's predictions drawn as an overlay, then exports a JSON of edits.
annotation/apply_review_edits.py merges that JSON back into a new COCO export.

Each image is flagged by where labels and predictions disagree (IoU 0.5 Hungarian
match, same as the training evaluation, on masks rasterized at 1/8 scale for speed):
  - labels with no matching prediction   (possibly spurious / wrong label)
  - predictions with no matching label   (possibly missed object)
Images are listed most-flagged first.

Usage (training-run layout):
  python annotation/build_review_package.py \
      --source-name "roboflow - 20260924 cryoai v4" \
      --dataset roboflow_20260924_cryoai_v4_singleclass \
      --model-run-dir 2026-09-24_172246_y8m_1440_16b_300ep_Run_2

Direct inputs (any COCO, e.g. roboflow_pull_ids.py's current.coco.json, a flat image
folder and a flat predictions folder, both keyed by image stem):
  python annotation/build_review_package.py --coco-json <pull>/current.coco.json \
      --images-dir <flat images> --pred-dir <predictions> --split-csv <S1>/split_assignment.csv \
      --out-dir <package> --class-aware --flag-conf 0.50 [--seed-edits <dir>]

--class-aware matches labels and predictions class-agnostically, then flags matched
pairs whose groups differ. The groups come from --class-groups (default
annotation/class_groups.yaml): ev_type = EV, MV, inner, Incomplete, Overlapping;
non_ev_type = VLDL, non-EV; low_conf = Low conf EV; predicted class 0 = ev_type,
1 = non_ev_type. low_conf labels match like any other, but are never class-
disagree and their unmatched count is kept apart (flc), so it doesn't rank images.
The grouping used is recorded in data.js as "class_groups".

data.js per image: gt [{id, c, p, u?, x?, lc?}], pr [{s, c, p, u?, x?}], fg (unmatched
labels), fp (unmatched predictions), fx (class-disagree pairs), flc (unmatched Low conf
EV). u = unmatched, x = matched across groups, lc = low_conf label. Without
--class-aware, fx/flc are 0 and matching is as before (non-EV labels not matched).
"""
import argparse
import csv
import json
import re
import shutil
import sys
import warnings
import zipfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "training"))
from train_yolo import match_objects_hungarian  # noqa: E402

ROOT = REPO.parent / "CryoAI"
TEMPLATE = Path(__file__).resolve().parent / "review_tool" / "index.html"
SPLIT_MAP = {"train": "train", "valid": "val", "test": "test"}  # coco split -> yolo split
NOT_MATCHED = {"non-EV", "items-nfaR"}  # dropped from training, so never flagged (single-class mode)
MATCH_SCALE = 1 / 8

CLASS_GROUPS = Path(__file__).resolve().parent / "class_groups.yaml"
LOW_CONF = "low_conf"  # the group that is never class-disagree and not ranked
SPLIT_ORDER = {"test": 0, "valid": 1, "val": 1, "train": 2}
RF_SUFFIX = re.compile(r"\.rf\.[0-9a-f]+\.\w+$")


def load_class_groups(path=CLASS_GROUPS):
    """class_groups.yaml -> {file, groups, model_classes, label_group}; label_group maps a label class to its group."""
    y = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    label_group = {c: g for g, names in y["groups"].items() for c in names}
    return {"file": str(Path(path)), "groups": y["groups"],
            "model_classes": {int(k): v for k, v in y["model_classes"].items()}, "label_group": label_group}


def stem_of(file_name):
    """Export name `x_png.rf.<hash>.jpg` -> `x_png`; Roboflow name `x.png` -> `x_png`."""
    name = Path(file_name).name
    if RF_SUFFIX.search(name):
        return RF_SUFFIX.sub("", name)
    p = Path(name)
    return f"{p.stem}_{p.suffix[1:]}" if p.suffix else name


def rasterize(ring_norm, w, h):
    """Normalized flat ring -> bool mask at MATCH_SCALE of the original image size."""
    mw, mh = max(1, round(w * MATCH_SCALE)), max(1, round(h * MATCH_SCALE))
    pts = np.round(np.asarray(ring_norm, dtype=np.float64).reshape(-1, 2) * [mw, mh]).astype(np.int32)
    mask = np.zeros((mh, mw), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)


def load_predictions(path, class_names):
    """Prediction cache file -> [{s, c, p, k, sr}] (normalized ring), keeping the class (load_prediction_polygons
    drops it). s is the confidence rounded for display; sr (raw) and k (class id) are for flagging only."""
    out = []
    if not Path(path).exists():
        return out
    for line in Path(path).read_text().splitlines():
        v = line.split()
        if len(v) < 8:
            continue
        k = int(v[0])
        out.append({"s": round(float(v[1]), 3), "sr": float(v[1]), "c": class_names[k] if k < len(class_names) else str(k),
                    "k": k, "p": [[round(float(x), 5) for x in np.asarray(v[2:], dtype=np.float32)]]})
    return out


def flag_image(gt, pr, w, h, groups, flag_conf):
    """Set u / x / lc on gt and pr items in place; return (fg, fp, fx, flc). groups = load_class_groups()
    output for class-aware flags, None for the original class-agnostic ones."""
    lg = groups["label_group"] if groups else {}
    if groups:
        for c in sorted({g["c"] for g in gt} - set(lg) - {"items-nfaR"}):
            warnings.warn(f"label class {c!r} is not in {groups['file']}: its labels are not matched or flagged",
                          stacklevel=2)
        gt_idx = [i for i, g in enumerate(gt) if g["c"] in lg]
    else:
        gt_idx = [i for i, g in enumerate(gt) if g["c"] not in NOT_MATCHED]
    pr_idx = [i for i, p in enumerate(pr) if p.get("sr", p["s"]) >= flag_conf]  # raw conf, not the rounded one
    matches, unmatched_p, unmatched_g = match_objects_hungarian(
        [rasterize(pr[i]["p"][0], w, h) for i in pr_idx], [rasterize(gt[i]["p"][0], w, h) for i in gt_idx], 0.5)
    fx = flc = 0
    for p in unmatched_p:
        pr[pr_idx[p]]["u"] = 1
    for g in unmatched_g:
        gt[gt_idx[g]]["u"] = 1
    if groups:
        for g in gt:
            if lg.get(g["c"]) == LOW_CONF:
                g["lc"] = True
        for p, g in matches:
            gg, pp = gt[gt_idx[g]], pr[pr_idx[p]]
            grp = lg[gg["c"]]
            if grp != LOW_CONF and groups["model_classes"].get(pp["k"]) != grp:
                gg["x"] = pp["x"] = True
                fx += 1
        flc = sum(1 for g in unmatched_g if lg[gt[gt_idx[g]]["c"]] == LOW_CONF)
    fg = len(unmatched_g) - flc
    return fg, len(unmatched_p), fx, flc


def gt_items(anns, cat_names, w, h):
    gt = []
    for a in anns:
        rings = [[round(v / (w if k % 2 == 0 else h), 5) for k, v in enumerate(seg)]
                 for seg in a.get("segmentation", []) if isinstance(seg, list) and len(seg) >= 6]
        if rings:
            gt.append({"id": a["id"], "c": cat_names.get(a["category_id"], "EV"), "p": rings})
    return gt


def run_layout_inputs(args):
    """(file, split, w, h, anns, cat_names, image_path, pred_path) from the training-run layout."""
    source_root = ROOT / "training images" / args.source_name
    run_dir = ROOT / "training outputs" / args.dataset / "runs" / args.model_run_dir
    for coco_split, yolo_split in SPLIT_MAP.items():
        coco_path = source_root / coco_split / "_annotations.coco.json"
        if not coco_path.exists():
            continue
        coco = json.loads(coco_path.read_text(encoding="utf-8"))
        cat_names = {c["id"]: c["name"] for c in coco["categories"]}
        anns_by_img = {}
        for a in coco["annotations"]:
            anns_by_img.setdefault(a["image_id"], []).append(a)
        for img in coco["images"]:
            stem = Path(img["file_name"]).stem
            yield (img["file_name"], coco_split, img["width"], img["height"], anns_by_img.get(img["id"], []),
                   cat_names, source_root / coco_split / img["file_name"],
                   run_dir / "predictions" / yolo_split / f"{stem}.txt", coco, {})


def direct_inputs(args):
    """Same tuples from --coco-json / --images-dir / --pred-dir, keyed by image stem."""
    coco = json.loads(args.coco_json.read_text(encoding="utf-8"))
    cat_names = {c["id"]: c["name"] for c in coco["categories"]}
    split_of = {}
    if args.split_csv:
        with open(args.split_csv, newline="", encoding="utf-8") as fh:
            split_of = {r["image"]: r["split"] for r in csv.DictReader(fh)}
    anns_by_img = {}
    for a in coco["annotations"]:
        anns_by_img.setdefault(a["image_id"], []).append(a)
    images_by_stem = {p.stem: p for p in args.images_dir.iterdir()
                      if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff")}
    for img in coco["images"]:
        stem = (img.get("extra") or {}).get("name") or stem_of(img["file_name"])
        if stem not in images_by_stem:
            print(f"[skip] no image for {stem}")
            continue
        split = split_of.get(stem) or (img.get("extra") or {}).get("split") or "train"
        split = {"val": "valid"}.get(split, split)
        yield (img["file_name"], split, img["width"], img["height"], anns_by_img.get(img["id"], []),
               cat_names, images_by_stem[stem], args.pred_dir / f"{stem}.txt", coco,
               {"stem": stem, "rf_split": (img.get("extra") or {}).get("split", "")})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-name", help="Folder under 'training images/' (Roboflow COCO export)")
    ap.add_argument("--dataset", help="Folder under 'training outputs/' holding the run")
    ap.add_argument("--model-run-dir", help="Run folder under '<dataset>/runs/' whose predictions to show")
    ap.add_argument("--coco-json", type=Path, help="Direct input: one COCO file (all splits)")
    ap.add_argument("--images-dir", type=Path, help="Direct input: flat folder of images named <stem>.<ext>")
    ap.add_argument("--pred-dir", type=Path, help="Direct input: flat folder of <stem>.txt prediction files")
    ap.add_argument("--split-csv", type=Path, help="Direct input: CSV with image (stem), split; else COCO extra.split")
    ap.add_argument("--out-dir", type=Path, help="Package folder (default: 'annotation review/<output-name>')")
    ap.add_argument("--output-name", default=None, help="Folder under 'annotation review/' (default: <source>_review_<date>)")
    ap.add_argument("--run-label", default=None, help="Predictions label shown in the page (default: the run dir)")
    ap.add_argument("--pred-classes", default="EV,non-EV", help="Predicted class id -> name, comma-separated")
    ap.add_argument("--class-aware", action="store_true", help="Group-aware flags (see above)")
    ap.add_argument("--class-groups", type=Path, default=CLASS_GROUPS, help="Class grouping file for --class-aware")
    ap.add_argument("--flag-conf", type=float, default=0.0,
                    help="Only predictions at or above this confidence are matched and flagged (all are shown)")
    ap.add_argument("--seed-edits", type=Path, default=None,
                    help="Folder of <file>.json edit records to pre-load into <package>/edits/")
    ap.add_argument("--no-zip", action="store_true", help="Skip the .zip files")
    ap.add_argument("--with-edits", action="store_true", help="Also copy edits/ into each part folder")
    ap.add_argument("--extra-classes", default="",
                    help="Comma-separated classes to offer on top of the export's own (collapsed or dropped at training time)")
    ap.add_argument("--width", type=int, default=1440, help="Display width of the packaged images")
    ap.add_argument("--quality", type=int, default=72, help="JPEG quality of the packaged images")
    ap.add_argument("--part-size", type=int, default=40,
                    help="Also split into self-contained parts of this many images (0 = no split); "
                         "40 keeps each zip under ~30 MB")
    args = ap.parse_args()

    direct = args.coco_json is not None
    if direct:
        if not (args.images_dir and args.pred_dir):
            ap.error("--coco-json needs --images-dir and --pred-dir")
        source, run = str(args.coco_json), args.run_label or str(args.pred_dir)
        inputs = direct_inputs(args)
    else:
        if not (args.source_name and args.dataset and args.model_run_dir):
            ap.error("give --source-name, --dataset and --model-run-dir, or --coco-json")
        source, run = args.source_name, args.run_label or args.model_run_dir
        inputs = run_layout_inputs(args)
    name = args.output_name or (args.out_dir.name if args.out_dir else
                                f"{args.source_name.replace(' ', '_')}_review_{datetime.now():%Y%m%d}")
    out = args.out_dir or ROOT / "annotation review" / name
    (out / "images").mkdir(parents=True, exist_ok=True)
    pred_classes = [c.strip() for c in args.pred_classes.split(",")]
    groups = load_class_groups(args.class_groups) if args.class_aware else None
    steps = ["match: Hungarian IoU >= 0.5 on 1/8-scale masks",
             f"predictions matched/flagged at conf >= {args.flag_conf:.2f}",
             f"class-aware groups from {args.class_groups}: {groups['groups']}, model {groups['model_classes']}"
             if args.class_aware
             else f"class ignored; labels {sorted(NOT_MATCHED)} not matched",
             "order: test -> valid -> train, each by fg + fp + fx" if args.class_aware else "order: by fg + fp"]
    print("[steps] " + "; ".join(steps))

    categories, images = None, []
    for file, split, w, h, anns, cat_names, img_path, pred_path, coco, extra in tqdm(list(inputs), desc="Packaging"):
        if categories is None:
            categories = [c["name"] for c in coco["categories"] if c["name"] != "items-nfaR"]
            categories += [c.strip() for c in args.extra_classes.split(",") if c.strip() and c.strip() not in categories]
        gt = gt_items(anns, cat_names, w, h)
        pr = [p for p in load_predictions(pred_path, pred_classes) if len(p["p"][0]) >= 6]
        fg, fp, fx, flc = flag_image(gt, pr, w, h, groups, args.flag_conf)
        for p in pr:
            p.pop("k", None)
            p.pop("sr", None)

        stem = extra.get("stem") or Path(file).stem
        im = Image.open(img_path).convert("L")
        if im.width > args.width:
            im = im.resize((args.width, round(im.height * args.width / im.width)), Image.LANCZOS)
        jpg = f"images/{stem}.jpg"
        im.save(out / jpg, "JPEG", quality=args.quality)
        rec = {"file": file, "split": split, "w": w, "h": h, "src": jpg,
               "gt": gt, "pr": pr, "fg": fg, "fp": fp, "fx": fx, "flc": flc}
        if extra.get("rf_split"):
            rec["rf_split"] = extra["rf_split"]
        images.append(rec)

    if args.class_aware:
        images.sort(key=lambda r: (SPLIT_ORDER.get(r["split"], 3), -(r["fg"] + r["fp"] + r["fx"])))
    else:
        images.sort(key=lambda r: -(r["fg"] + r["fp"]))
    data = {"id": name, "source": source, "run": run,
            "created": datetime.now().isoformat(timespec="seconds"), "categories": categories, "images": images}
    if groups:
        data["class_groups"] = {k: groups[k] for k in ("file", "groups", "model_classes")}
        data["flag_conf"] = args.flag_conf
    if args.seed_edits:
        seed = out / "edits"
        seed.mkdir(exist_ok=True)
        valid = {r["file"] for r in images}
        n = 0
        for f in sorted(args.seed_edits.glob("*.json")):
            rec = json.loads(f.read_text(encoding="utf-8"))
            if rec.get("file") in valid:
                (seed / f.name).write_text(json.dumps(rec), encoding="utf-8")
                n += 1
            else:
                print(f"[seed skip] {f.name}: file not in package")
        print(f"[OK] seeded {n} edit records -> {seed}")
    write_packages(out, data, args.part_size, zip_it=not args.no_zip, with_edits=args.with_edits)
    print(f"[OK] {len(images)} images, {sum(len(r['gt']) for r in images)} labels, "
          f"{sum(len(r['pr']) for r in images)} predictions")
    print(f"[OK] Flags: {sum(r['fg'] for r in images)} unmatched labels, "
          f"{sum(r['fp'] for r in images)} unmatched predictions, {sum(r['fx'] for r in images)} class-disagree, "
          f"{sum(r['flc'] for r in images)} unmatched Low conf EV (not ranked)")


def write_packages(out: Path, data: dict, part_size: int = 0, zip_it: bool = True, with_edits: bool = False):
    """Write index.html + data.js next to out/images and zip it. With part_size > 0, also split the
    (flag-sorted) images into self-contained parts <out>_partN/ + .zip, each small enough to send or
    download on its own; the most-flagged images land in part 1."""
    def write(folder: Path, d: dict):
        (folder / "data.js").write_text("window.REVIEW_DATA = " + json.dumps(d, separators=(",", ":")) + ";\n",
                                        encoding="utf-8")
        shutil.copy(TEMPLATE, folder / "index.html")
        if not zip_it:
            print(f"[OK] {folder.name}: {len(d['images'])} images")
            return
        zip_path = folder.with_suffix(".zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as z:  # JPEGs don't compress further
            for f in sorted(folder.rglob("*")):
                if f.is_file():
                    z.write(f, Path(folder.name) / f.relative_to(folder))
        print(f"[OK] {folder.name}: {len(d['images'])} images, zip {zip_path.stat().st_size / 2**20:.0f} MiB")

    write(out, data)
    if part_size <= 0:
        return
    chunks = [data["images"][k:k + part_size] for k in range(0, len(data["images"]), part_size)]
    for n, chunk in enumerate(chunks, 1):
        part = out.parent / f"{out.name}_part{n}"
        (part / "images").mkdir(parents=True, exist_ok=True)
        for r in chunk:
            shutil.copy(out / r["src"], part / r["src"])
        if with_edits and (out / "edits").is_dir():
            files = {r["file"] for r in chunk}
            (part / "edits").mkdir(exist_ok=True)
            for f in (out / "edits").glob("*.json"):
                if json.loads(f.read_text(encoding="utf-8")).get("file") in files:
                    shutil.copy(f, part / "edits" / f.name)
        write(part, {**data, "id": f"{data['id']}_part{n}", "images": chunk})

if __name__ == "__main__":
    main()
