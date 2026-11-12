"""
Build a "compare" review package: two label sets on the same images, where only the
disagreements become Quick check cards. Set A is a COCO label set (the package's base,
which apply_review_edits.py merges decisions back into); set B is a model (weights, run
here), a folder of cached YOLO-seg predictions, or a second COCO (human vs human).

  python annotation/compare_package.py --labels <lv>/labels.coco.json \
      --images-root "<S1>/train" --images-root "<S1>/valid" \
      --select-csv <S1>/split_assignment.csv --where "batch=20260512 A549,20260512 PMSC" --where split=train,valid \
      --b-model models/v5_evnonev_y8m_20261029/best.pt --imgsz 1440 \
      --flag-conf 0.55 --decoys 0.10 --owner pujan-web --b-label "v5 @0.55" \
      --out <CryoAI-laptop>/review/lab_compare_v5_cell_20261110

Cards (the fair-view EV population of analysis/evaluate_coco_export.py):
  FP card = a set-B EV object at >= flag_conf with no matching set-A label ("B found, not in A")
  FN card = a set-A EV-type label (EV, MV, inner, Incomplete, Overlapping) with no matching B object
Low conf EV and non-EV/VLDL labels never become cards; B's non-EV objects are shown, never carded.
Matching is build_review_package.flag_image (Hungarian, IoU 0.5, 1/8-scale masks, class-aware groups).

Known-answer decoys (--decoys, a fraction of the real cards, half each kind), mixed in by the same rules:
  accept decoy: a set-A EV label that B also found, shown as an FN card with B's match hidden -> answer: keep
  reject decoy: a B-style outline on empty background (no A or B object within a margin) -> answer: reject
The answer key is written to <out>/../_work_<name>/decoys.json, never into data.js, so it is not served.
read_compare_package() / strip_decoys() turn decisions into audit counts and drop decoy decisions.

Writes <out>/{index.html, data.js, images/, review_package.json (kind "compare"), .review_key}.
--reuse-key <dir> copies another package's .review_key (e.g. to serve from a review slot whose key the
portal already holds). The key is never printed.
"""
import argparse
import csv
import json
import math
import random
import secrets
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "training"))
import build_review_package as B  # noqa: E402

TOOL = "annotation/compare_package.py"
EV_GROUP = "ev_type"
DECOY_MARGIN = 1.5  # reject decoys keep this many radii away from every A and B object


def parse_where(items):
    """["batch=a,b", "split=train,valid"] -> {"batch": {"a","b"}, "split": {...}}"""
    out = {}
    for it in items or []:
        k, _, v = it.partition("=")
        out[k.strip()] = {x.strip() for x in v.split(",") if x.strip()}
    return out


def select_stems(select_csv, where, key="image"):
    if not select_csv:
        return None
    with open(select_csv, newline="", encoding="utf-8") as fh:
        return {r[key] for r in csv.DictReader(fh) if all(r.get(k) in v for k, v in where.items())}


def stage_images(coco, roots, stems, work):
    """Copy the selected images into one flat folder named <stem>.<ext> (the builder's lookup key)."""
    flat = work / "images"
    flat.mkdir(parents=True, exist_ok=True)
    want = {B.image_key(im) for im in coco["images"]}
    if stems is not None:
        want &= stems
    found = {}
    for root in roots:
        for p in Path(root).iterdir():
            if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff") and B.stem_of(p.name) in want:
                found.setdefault(B.stem_of(p.name), p)
    for stem, p in found.items():
        dst = flat / f"{stem}{p.suffix.lower()}"
        if not dst.exists():
            shutil.copy2(p, dst)
    missing = sorted(want - set(found))
    return flat, sorted(found), missing


def coco_as_preds(coco_b, stems, out_dir, groups):
    """Second COCO -> YOLO-seg txt per stem (conf 1.0; class 0 for EV-type labels, 1 otherwise)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cats = {c["id"]: c["name"] for c in coco_b["categories"]}
    by_img = {}
    for a in coco_b["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)
    for im in coco_b["images"]:
        stem = (im.get("extra") or {}).get("name") or B.stem_of(im["file_name"])
        if stem not in stems:
            continue
        lines = []
        for a in by_img.get(im["id"], []):
            k = 0 if groups["label_group"].get(cats.get(a["category_id"])) == EV_GROUP else 1
            for seg in a.get("segmentation", []):
                if isinstance(seg, list) and len(seg) >= 6:
                    xy = [v / (im["width"] if i % 2 == 0 else im["height"]) for i, v in enumerate(seg)]
                    lines.append(f"{k} 1.0 " + " ".join(f"{v:.6f}" for v in xy))
        (out_dir / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
    return out_dir


def ring_box(r):
    a = np.asarray(r, dtype=float).reshape(-1, 2)
    return a[:, 0].min(), a[:, 1].min(), a[:, 0].max(), a[:, 1].max()


def circle(cx, cy, rad_x, rad_y, n=24, rng=None, jitter=0.06):
    """A closed ring; with rng, the radius wobbles smoothly (low harmonics + a little noise) like a model outline."""
    out = []
    ph = [rng.uniform(0, 2 * math.pi) for _ in range(3)] if rng else [0, 0, 0]
    amp = [rng.uniform(0, jitter) for _ in range(3)] if rng else [0, 0, 0]
    for k in range(n):
        a = 2 * math.pi * k / n
        f = 1 + sum(amp[h] * math.sin((h + 2) * a + ph[h]) for h in range(3)) + (rng.uniform(-0.01, 0.01) if rng else 0)
        out += [round(cx + rad_x * f * math.cos(2 * math.pi * k / n), 5),
                round(cy + rad_y * f * math.sin(2 * math.pi * k / n), 5)]
    return out


def refine_cards(rec, groups):
    """Keep u only on the fair-view EV population; recount fg/fp."""
    lg = groups["label_group"]
    for g in rec["gt"]:
        if g.get("u") and lg.get(g["c"]) != EV_GROUP:
            g.pop("u")
    for p in rec["pr"]:
        if p.get("u") and p.get("k") != 0:
            p.pop("u")
    rec["fg"] = sum(1 for g in rec["gt"] if g.get("u"))
    rec["fp"] = sum(1 for p in rec["pr"] if p.get("u"))


def add_decoys(images, groups, frac, rng, flag_conf):
    """Mix in known-answer cards; return the answer key (never written into data.js)."""
    if not images:
        return []
    n_real = sum(r["fg"] + r["fp"] for r in images)
    n_each = max(5, round(n_real * frac / 2)) if frac > 0 else 0
    lg, key = groups["label_group"], []
    # accept decoys: EV labels matched by a B EV object at >= flag_conf (two sources agree)
    pool = []
    for i, r in enumerate(images):
        for gi, g in enumerate(r["gt"]):
            if g.get("u") or g.get("x") or lg.get(g["c"]) != EV_GROUP or g["c"] != "EV":
                continue
            pool.append((i, gi))
    rng.shuffle(pool)
    used = set()
    for i, gi in pool:
        if len([k for k in key if k["kind"] == "accept"]) >= n_each:
            break
        r, g = images[i], images[i]["gt"][gi]
        # hide B's match: drop the B EV object overlapping this label most (display only; never applied)
        gx0, gy0, gx1, gy1 = ring_box(g["p"][0])
        best, best_ov = None, 0.0
        for pi, p in enumerate(r["pr"]):
            if p.get("_drop") or p.get("k") != 0 or p.get("sr", p["s"]) < flag_conf:
                continue
            px0, py0, px1, py1 = ring_box(p["p"][0])
            ix = max(0, min(gx1, px1) - max(gx0, px0)) * max(0, min(gy1, py1) - max(gy0, py0))
            un = (gx1 - gx0) * (gy1 - gy0) + (px1 - px0) * (py1 - py0) - ix
            ov = ix / un if un > 0 else 0
            if ov > best_ov:
                best, best_ov = pi, ov
        if best is None or best_ov < 0.5 or (i, best) in used:
            continue
        r["pr"][best]["_drop"] = True
        used.add((i, best))
        g["u"] = 1
        key.append({"kind": "accept", "answer": "keep", "file": r["file"], "obj": f"g{g['id']}"})
    # reject decoys: an outline on empty background, sized like this image's EVs
    order = list(range(len(images)))
    rng.shuffle(order)
    tries = 0
    while len([k for k in key if k["kind"] == "reject"]) < n_each and tries < 5000:
        tries += 1
        r = images[order[tries % len(order)]]
        boxes = [ring_box(o["p"][0]) for o in r["gt"] + r["pr"]]
        if not boxes:
            continue
        sizes = sorted(max(b[2] - b[0], (b[3] - b[1]) * r["h"] / r["w"]) for b in boxes)
        d = sizes[len(sizes) // 2]  # median diameter, in x-normalized units
        rx, ry = d / 2, d / 2 * r["w"] / r["h"]
        cx, cy = rng.uniform(0.08, 0.92), rng.uniform(0.08, 0.92)
        clear = all(cx + rx * DECOY_MARGIN < b[0] or cx - rx * DECOY_MARGIN > b[2] or
                    cy + ry * DECOY_MARGIN < b[1] or cy - ry * DECOY_MARGIN > b[3] for b in boxes)
        if not clear:
            continue
        s = round(rng.uniform(flag_conf, min(0.9, flag_conf + 0.3)), 3)
        did = len(key)
        r["pr"].append({"s": s, "sr": s, "c": "EV", "k": 0, "p": [circle(cx, cy, rx, ry, n=rng.randint(28, 44), rng=rng)],
                        "u": 1, "_decoy": did})
        key.append({"kind": "reject", "answer": "no", "file": r["file"], "obj": None, "_decoy": did})
    for r in images:
        r["pr"] = [p for p in r["pr"] if not p.get("_drop")]
        for i, p in enumerate(r["pr"]):  # final indices, after the drops
            if "_decoy" in p:
                next(k for k in key if k.get("_decoy") == p["_decoy"])["obj"] = f"p{i}"
                p.pop("_decoy")
    for k in key:
        k.pop("_decoy", None)
    return key


DEDUP_IOU = 0.5  # two models' finds at mask IoU >= this are one object (one card)


def parse_extra(text):
    """NAME=WEIGHTS@CONF[@CLASSES] -> (name, weights, conf, classes). WEIGHTS may be a folder of <stem>.txt
    predictions instead. CLASSES: comma list (default EV,non-EV)."""
    name, _, rest = text.partition("=")
    parts = rest.split("@")
    if not name or len(parts) < 2:
        raise SystemExit(f"--b-extra {text!r}: expected NAME=WEIGHTS@CONF[@CLASSES]")
    return name, Path(parts[0]), float(parts[1]), parts[2] if len(parts) > 2 else "EV,non-EV"


def predict_cached(weights, flat, pred_dir, staged, args):
    """Predictions for the staged images (conf floor 0.25), re-made when the weights, settings or images change."""
    sys.path.insert(0, str(HERE.parent / "training"))
    from train_yolo import export_predictions_for_split
    want = {"weights": str(weights), "imgsz": args.imgsz, "nms": args.nms, "max_det": args.max_det, "images": staged}
    stamp = pred_dir / "_settings.json"
    if pred_dir.exists() and (not stamp.exists() or json.loads(stamp.read_text(encoding="utf-8")) != want):
        shutil.rmtree(pred_dir)  # cached predictions were made with other settings or images
    if not any(pred_dir.glob("*.txt")):
        export_predictions_for_split(str(weights), str(flat), pred_dir, args.imgsz, 0.25, args.nms, args.device,
                                     max_det=args.max_det)
        stamp.write_text(json.dumps(want), encoding="utf-8")
    return pred_dir


def add_model_cards(rec, pr_other, name, conf, ref_conf, w, h, groups, dedup_iou=DEDUP_IOU):
    """Union another model's disagreements into an image record already flagged against set B (the reference):
    a set-A label this model misses becomes a card too; this model's unmatched EV finds (>= its own conf) become cards,
    merged with an overlapping reference card (IoU >= dedup_iou) instead of duplicating it. A find overlapping a
    reference EV outline that matched a label is not carded (the object is labeled); one overlapping a reference outline
    that is not a card (below the reference conf, or another class) replaces that outline. Who found each card goes in
    the item's "_m" (set of model names; written to the work folder, never data.js). -> (new FN cards, new FP cards,
    merged, skipped)."""
    g2 = [{k: v for k, v in g.items() if k not in ("u", "x", "lc")} for g in rec["gt"]]
    p2 = [{k: v for k, v in p.items() if k not in ("u", "x")} for p in pr_other]
    B.flag_image(g2, p2, w, h, groups, conf)
    tmp = {"gt": g2, "pr": p2}
    refine_cards(tmp, groups)
    n_fn = 0
    for g, go in zip(rec["gt"], g2):
        if go.get("u"):
            if not g.get("u"):
                g["u"] = 1
                n_fn += 1
            g.setdefault("_m", set()).add(name)
    masks = [B.rasterize(p["p"][0], w, h) for p in rec["pr"]]
    n_fp = merged = skipped = 0
    for p in sorted((p for p in p2 if p.get("u")), key=lambda p: -p.get("sr", p["s"])):
        m = B.rasterize(p["p"][0], w, h)
        hit = None
        for j, mk in enumerate(masks):
            if rec["pr"][j].get("_drop"):
                continue
            inter = np.logical_and(m, mk).sum()
            if inter and inter / np.logical_or(m, mk).sum() >= dedup_iou:
                hit = j
                break
        if hit is not None and rec["pr"][hit].get("u"):
            rec["pr"][hit].setdefault("_m", set()).add(name)
            merged += 1
            continue
        if hit is not None:
            ref = rec["pr"][hit]
            if ref.get("k") == 0 and ref.get("sr", ref["s"]) >= ref_conf:
                skipped += 1  # the reference's EV outline here matched a label
                continue
            ref["_drop"] = True  # a non-card reference outline: this model's card replaces it
        q = dict(p, u=1, _m={name})
        rec["pr"].append(q)
        masks.append(m)
        n_fp += 1
    return n_fn, n_fp, merged, skipped


def build(args):
    if any((args.out / "edits").glob("*.json")) and not args.force:
        raise SystemExit(f"{args.out} already has review edits: a rebuild would point them at different objects. "
                         "Build into a new folder, or pass --force.")
    coco = json.loads(args.labels.read_text(encoding="utf-8"))
    groups = B.load_class_groups(args.class_groups)
    name = args.out.name
    work = args.out.parent / f"_work_{name}"
    work.mkdir(parents=True, exist_ok=True)
    stems = select_stems(args.select_csv, parse_where(args.where))
    flat, staged, missing = stage_images(coco, args.images_root, stems, work)
    print(f"[images] {len(staged)} staged" + (f", {len(missing)} selected but not found: {missing[:5]}" if missing else ""))
    if args.b_model:
        pred_dir = predict_cached(args.b_model, flat, work / "preds", staged, args)
        b_desc = {"kind": "model", "weights": str(args.b_model), "imgsz": args.imgsz, "nms": args.nms,
                  "max_det": args.max_det, "conf_floor": 0.25}
    elif args.b_preds:
        pred_dir, b_desc = args.b_preds, {"kind": "predictions", "dir": str(args.b_preds)}
    elif args.b_coco:
        pred_dir = coco_as_preds(json.loads(args.b_coco.read_text(encoding="utf-8")), set(staged), work / "b_as_preds",
                                 groups)
        b_desc = {"kind": "labels", "coco": str(args.b_coco)}
    else:
        raise SystemExit("give --b-model, --b-preds or --b-coco")

    extras = []
    for text in getattr(args, "b_extra", None) or []:
        if not (args.b_model or args.b_preds):
            raise SystemExit("--b-extra needs a reference model (--b-model or --b-preds)")
        ename, weights, econf, eclasses = parse_extra(text)
        edir = weights if weights.is_dir() else predict_cached(weights, flat, work / f"preds_{ename}", staged, args)
        extras.append((ename, edir, econf, eclasses, str(weights)))
    if extras:
        b_desc = dict(b_desc, flag_conf=args.flag_conf, extra=[{"name": e[0], "weights": e[4], "flag_conf": e[2],
                                                                "classes": e[3]} for e in extras])
    union = {"fn": 0, "fp": 0, "merged": 0, "skipped": 0}
    ns = argparse.Namespace(coco_json=args.labels, images_dir=flat, pred_dir=Path(pred_dir), split_csv=args.split_csv)
    images, categories = [], None
    for file, split, w, h, anns, cat_names, img_path, pred_path, _coco, extra in B.direct_inputs(ns):
        if categories is None:
            categories = [c["name"] for c in coco["categories"] if c["name"] != "items-nfaR"]
        gt = B.gt_items(anns, cat_names, w, h)
        pr = [p for p in B.load_predictions(pred_path, args.pred_classes.split(",")) if len(p["p"][0]) >= 6]
        fg, fp, fx, flc = B.flag_image(gt, pr, w, h, groups, args.flag_conf)
        rec = {"file": file, "split": split, "w": w, "h": h, "src": f"images/{extra['stem']}.jpg", "gt": gt, "pr": pr,
               "fg": fg, "fp": fp, "fx": fx, "flc": flc, "_img": img_path, "_stem": extra["stem"]}
        refine_cards(rec, groups)
        if extras:
            ref_name = args.b_label or "B"
            for o in rec["pr"] + rec["gt"]:
                if o.get("u"):
                    o["_m"] = {ref_name}
            for ename, edir, econf, eclasses, _w in extras:
                epr = [p for p in B.load_predictions(Path(edir) / Path(pred_path).name, eclasses.split(","))
                       if len(p["p"][0]) >= 6]
                for k, v in zip(("fn", "fp", "merged", "skipped"),
                                add_model_cards(rec, epr, ename, econf, args.flag_conf, w, h, groups)):
                    union[k] += v
            rec["fg"] = sum(1 for g in rec["gt"] if g.get("u"))
            rec["fp"] = sum(1 for p in rec["pr"] if p.get("u") and not p.get("_drop"))
        images.append(rec)
    real = {"fp": sum(r["fp"] for r in images), "fn": sum(r["fg"] for r in images)}
    real_by = {r["file"]: (r["fp"], r["fg"], r["split"]) for r in images}
    rng = random.Random(args.seed)
    key = add_decoys(images, groups, args.decoys, rng, args.flag_conf)
    found_by = []
    for r in images:
        r["fg"] = sum(1 for g in r["gt"] if g.get("u"))
        r["fp"] = sum(1 for p in r["pr"] if p.get("u"))
        for i, p in enumerate(r["pr"]):
            p.pop("k", None)
            p.pop("sr", None)
            m = p.pop("_m", None)
            if m and p.get("u"):
                found_by.append({"file": r["file"], "obj": f"p{i}", "card": "fp", "models": sorted(m)})
        for g in r["gt"]:
            m = g.pop("_m", None)
            if m and g.get("u"):
                found_by.append({"file": r["file"], "obj": f"g{g['id']}", "card": "fn", "missed_by": sorted(m)})
    images.sort(key=lambda r: -(r["fg"] + r["fp"]))

    out = args.out
    (out / "images").mkdir(parents=True, exist_ok=True)
    from PIL import Image
    for r in images:
        im = Image.open(r.pop("_img")).convert("L")
        if im.width > args.width:
            im = im.resize((args.width, round(im.height * args.width / im.width)), Image.LANCZOS)
        im.save(out / r["src"], "JPEG", quality=args.quality)
        r.pop("_stem")
    built = datetime.now().isoformat(timespec="seconds")
    data = {"id": name, "kind": "compare", "source": str(args.labels), "run": args.b_label or b_desc.get("weights", ""),
            "created": built, "categories": categories, "images": images, "flag_conf": args.flag_conf,
            "cand_conf": args.flag_conf,
            "class_groups": {k: groups[k] for k in ("file", "groups", "model_classes")}}
    if getattr(args, "source_title", None):  # the label set's name on the page's "Working on" bar (else its folder)
        data["source_title"] = args.source_title
    B.write_packages(out, data, 0, zip_it=False)
    if args.index_html:
        shutil.copy(args.index_html, out / "index.html")
    key_file = out / ".review_key"
    if args.reuse_key:
        shutil.copy(Path(args.reuse_key) / ".review_key", key_file)
    elif not key_file.exists():
        key_file.write_text(secrets.token_urlsafe(24), encoding="utf-8")
    n_dec = {"accept": sum(k["kind"] == "accept" for k in key), "reject": sum(k["kind"] == "reject" for k in key)}
    (out / "review_package.json").write_text(json.dumps({
        "schema": "review_package/1", "id": name, "logged": built, "recorded_by": TOOL, "path": str(out.resolve()),
        "kind": "compare", "compare": {"a": {"kind": "labels", "coco": str(args.labels)}, "b": b_desc,
                                       "flag_conf": args.flag_conf, "label": args.b_label},
        "base": str(args.labels), "base_kind": "label_version", "n_images": len(images), "built": built,
        "tool": TOOL, "status": "built", "owner": args.owner, "reviewers": args.reviewer or [],
        "cards": {"fp": real["fp"], "fn": real["fn"], "decoys": n_dec}}, indent=1), encoding="utf-8")
    if extras:
        (work / "cards_models.json").write_text(json.dumps({"package": name, "union": union, "cards": found_by},
                                                           indent=1), encoding="utf-8")
        print(f"[union] extra models add FN {union['fn']}, FP {union['fp']}; merged with a reference card "
              f"{union['merged']}; not carded (reference matched a label) {union['skipped']}")
    (work / "decoys.json").write_text(json.dumps({"package": name, "seed": args.seed, "decoys": key}, indent=1),
                                      encoding="utf-8")
    with open(work / "cards_per_image.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["file", "split", "fp_cards", "fn_cards"])
        for f, (a, b, s) in sorted(real_by.items()):
            wr.writerow([f, s, a, b])
    print(f"[OK] {name}: {len(images)} images; real cards FP {real['fp']} + FN {real['fn']}; "
          f"decoys accept {n_dec['accept']} + reject {n_dec['reject']}; key -> {work / 'decoys.json'}")
    return out, work


def read_compare_package(pkg: Path, work: Path):
    """Decisions -> audit counts (real cards only) + decoy accuracy. Decision keys: p<k> (add:/no), g<id> (keep:/del)."""
    data = json.loads((pkg / "data.js").read_text(encoding="utf-8").split("=", 1)[1].rstrip().rstrip(";"))
    key = {(d["file"], d["obj"]): d for d in json.loads((work / "decoys.json").read_text(encoding="utf-8"))["decoys"]}
    edits = {}
    for f in (pkg / "edits").glob("*.json"):
        try:
            rec = json.loads(f.read_text(encoding="utf-8"))
        except ValueError:
            continue
        e = rec.get("edit", rec)
        edits[rec.get("file") or f.name[:-5]] = (e.get("dec") or {})
    rows = []
    for r in data["images"]:
        dec = edits.get(r["file"], {})
        for i, p in enumerate(r["pr"]):
            if p.get("u"):
                v = (dec.get(f"p{i}") or {}).get("v")
                rows.append({"file": r["file"], "split": r["split"], "card": "fp", "obj": f"p{i}",
                             "answer": "real" if v and str(v).startswith("add:") else "not" if v == "no" else "skip",
                             "decoy": key.get((r["file"], f"p{i}"))})
        for g in r["gt"]:
            if g.get("u"):
                v = (dec.get(f"g{g['id']}") or {}).get("v")
                ans = "skip" if not v else "not" if v == "del" else "real" if str(v) == "keep:EV" or (
                    str(v).startswith("keep:") and v[5:] in ("EV", "MV", "inner", "Incomplete", "Overlapping")) \
                    else "reclass"
                rows.append({"file": r["file"], "split": r["split"], "card": "fn", "obj": f"g{g['id']}",
                             "answer": ans, "decoy": key.get((r["file"], f"g{g['id']}"))})
    return rows


def strip_decoys(edits_dir: Path, work: Path, out_dir: Path, now_ms=None) -> dict:
    """Copy every edit record from edits_dir to out_dir with the decoy answers taken out, so a known-answer card
    can never reach the labels. Reject decoy p<k> (a synthetic outline): its t:p<k> tag goes; its decision is
    removed, or, when the record's saved polygon list holds an item with src p<k> (that item would otherwise
    survive without a decision), replaced by "no" stamped newer than anything in the record. Accept decoy g<id>
    is a real label: its corrections (reshape, class, tag) stay; only a "del" is removed. Other records are
    copied byte for byte. -> {"files", "changed": [file], "reject_dropped", "reject_no", "accept_del_dropped"}."""
    key = json.loads((Path(work) / "decoys.json").read_text(encoding="utf-8"))["decoys"]
    by_file = {}
    for d in key:
        by_file.setdefault(d["file"], []).append(d)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    stats = {"files": 0, "changed": [], "reject_dropped": 0, "reject_no": 0, "accept_del_dropped": 0}
    for f in sorted(Path(edits_dir).glob("*.json")):
        raw = f.read_bytes()
        stats["files"] += 1
        rec = json.loads(raw.decode("utf-8"))
        e = rec.get("edit", rec)
        dec = e.get("dec") or {}
        touched = False
        for d in by_file.get(rec.get("file"), []):
            obj = d["obj"]
            if d["kind"] == "reject":
                if dec.pop("t:" + obj, None) is not None:
                    touched = True
                in_list = any(isinstance(it, dict) and it.get("src") == obj for it in (e.get("gt") or []))
                if in_list:
                    stamps = [v.get("t", 0) for v in dec.values() if isinstance(v, dict)] + [
                        v.get("server_t", 0) for v in dec.values() if isinstance(v, dict)] + [
                        e.get("gtT") or 0, e.get("t") or 0, now_ms or 0]
                    t = max(x for x in stamps if isinstance(x, (int, float))) + 1
                    dec[obj] = {"v": "no", "t": t, "server_t": t, "who": "decoy-strip"}
                    stats["reject_no"] += 1
                    touched = True
                elif dec.pop(obj, None) is not None:
                    stats["reject_dropped"] += 1
                    touched = True
            elif (dec.get(obj) or {}).get("v") == "del":
                dec.pop(obj)
                stats["accept_del_dropped"] += 1
                touched = True
        if touched:
            e["dec"] = dec
            (out_dir / f.name).write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
            stats["changed"].append(f.name)
        else:
            (out_dir / f.name).write_bytes(raw)
    return stats


def edits_sha256(edits_dir: Path) -> str:
    """One fingerprint for a folder of edit records: sha256 over the sorted names and their bytes."""
    import hashlib
    h = hashlib.sha256()
    for f in sorted(Path(edits_dir).glob("*.json")):
        h.update(f.name.encode("utf-8") + b"\x00" + f.read_bytes() + b"\x00")
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", type=Path, required=True, help="Set A: COCO label set (the base decisions apply to)")
    ap.add_argument("--images-root", type=Path, action="append", required=True, help="Folder(s) holding the images")
    ap.add_argument("--select-csv", type=Path, help="CSV with an image column (stem) to select from")
    ap.add_argument("--where", action="append", help="col=v1,v2 filter on --select-csv (repeatable, AND)")
    ap.add_argument("--split-csv", type=Path, help="CSV image,split (the split shown on cards)")
    ap.add_argument("--b-model", type=Path, help="Set B: model weights (predictions run here)")
    ap.add_argument("--b-preds", type=Path, help="Set B: folder of <stem>.txt YOLO-seg predictions")
    ap.add_argument("--b-coco", type=Path, help="Set B: a second COCO label set (human vs human)")
    ap.add_argument("--b-label", default="", help="How set B is named on the page")
    ap.add_argument("--b-extra", action="append", metavar="NAME=WEIGHTS@CONF[@CLASSES]",
                    help="Another model whose disagreements are carded too (repeatable; with --b-model). Cards: any "
                         "label a model misses; every model's unmatched EV finds, merged where they overlap")
    ap.add_argument("--source-title", help="How set A (the label set) is named on the page (default: its folder name)")
    ap.add_argument("--imgsz", type=int, default=1440)
    ap.add_argument("--nms", type=float, default=0.7)
    ap.add_argument("--max-det", type=int, default=1000)
    ap.add_argument("--device", default="0")
    ap.add_argument("--pred-classes", default="EV,non-EV")
    ap.add_argument("--class-groups", type=Path, default=B.CLASS_GROUPS)
    ap.add_argument("--flag-conf", type=float, default=0.55)
    ap.add_argument("--decoys", type=float, default=0.10, help="Decoy cards as a fraction of real cards (0 = none)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--owner", required=True)
    ap.add_argument("--reviewer", action="append", help="Extra people who may open it (portal ids)")
    ap.add_argument("--reuse-key", type=Path, help="Copy this package folder's .review_key")
    ap.add_argument("--index-html", type=Path, help="index.html to ship (default: the repo's review tool)")
    ap.add_argument("--width", type=int, default=1440)
    ap.add_argument("--quality", type=int, default=72)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--force", action="store_true", help="Rebuild even though <out>/edits/ has decisions")
    build(ap.parse_args())


if __name__ == "__main__":
    main()
