"""
Turn a label review package's saved decisions (edits/*.json from review_server.py
or the page's export) into explicit label operations, and check each one against
what Roboflow holds now (a roboflow_pull_ids.py / roboflow_snapshot.py COCO).

Each image's labels are rebuilt with review_server.effective_labels() (the same
rules as the page's effGt(): base list or Editor list, decisions on top, per-item
t, n<uid> drawn outlines), and the result is diffed against the package's labels:

  delete    a label removed (decision "del", or gone from an Editor-saved list)
  reclass   class changed, outline unchanged
  reshape   outline changed (class may change too)
  add       a new label (accepted prediction "add:<cls>", drawn "new:<cls>", or an
            outline drawn in an older Editor list, key "editor")
  keep      a flagged label confirmed as is           (no label change)
  reject    a flagged prediction rejected ("no")       (no label change)

Each operation is then matched by geometry to the current labels of the same
image (or of its kept twin, for a dropped duplicate) and classified:

  applied-ready   still to do, and its target is unambiguous
  already-done    the current labels already show the result (or nothing to do)
  conflict        it contradicts the current labels, another add on the same image,
                  or the twin's decision on the same object
  orphan          its target is gone (image or label not found)
  needs-human     can't be decided by geometry: an add inside (or around) an existing
                  label, a duplicate's delete of a label only its own copy has, or a
                  twin decision that agrees but is itself unresolved

Rules:
- Existing labels match at IoU >= --match-iou (0.9). A delete or reclass is
  applied-ready only while the current label still has the September "from" class.
- An add is already done when a current label of its class overlaps it at IoU >=
  --add-iou (0.5); another class there is a conflict; an add that lies >= --contain
  (0.8) inside a label, or holds one, is needs-human.
- Adds on one image are compared with each other first: same-class pairs at IoU >=
  --add-iou collapse to one (a prediction or drawn key beats an older Editor-list
  copy, then the newest), different-class pairs are both conflicts.
- A dropped duplicate's operation is compared with its twin's operation that
  overlaps it most (IoU >= --add-iou): the same result is covered by the twin
  (already-done if the twin's own operation is applied-ready or already-done, else
  needs-human); a different result, including keep/keep with different classes,
  is a conflict.
- Two applied-ready operations on one current label are conflicts.

Known-answer control (printed, and in control.json): every original label on an
edited image that no decision touched should still be in the current labels at
IoU >= --match-iou with the same class, unless Roboflow was edited since. A low
rate means the geometry matching is broken, and the statuses can't be trusted.

Writes --out/resolution.csv (one row per operation) and, with --seed-dir plus
--pred-dir, the applied-ready operations as edit records for a new package built
from the same --current-coco, so a reviewer sees them as pending decisions:
g<annotation id> for label changes, p<index> for an add with a new prediction at
IoU >= --add-iou, else n<uid> "new:<cls>" + p (a drawn outline). It also checks
that no two seeded outlines on one image overlap at IoU >= --add-iou. Read-only
otherwise.

    python annotation/resolve_review_edits.py --package ".../annotation review/cryoai_v4_review" \\
        --current-coco ".../roboflow_pull_<stamp>/current.coco.json" \\
        --dup-csv ".../tracking/prelipo_images.csv" --out ".../september_edits" \\
        [--pred-dir ".../predictions" --seed-dir ".../september_edits/seed_edits"]
"""
import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_review_package import load_predictions, stem_of  # noqa: E402
from review_server import _migrate as migrate, effective_labels  # noqa: E402

RASTER = 2048  # IoU grid: the longer image side is rasterized at this many pixels
STATUSES = ("applied-ready", "already-done", "conflict", "orphan", "needs-human")
COLS = ["file", "stem", "target_stem", "n", "key", "who", "t", "op", "from", "to", "status", "note",
        "twin_note", "cur_id", "cur_class", "iou", "contain", "seed_key", "old", "new"]


def read_data_js(path):
    return json.loads(Path(path).read_text(encoding="utf-8").split("=", 1)[1].strip().rstrip(";"))


def same_poly(a, b):
    return json.dumps(a) == json.dumps(b)


def operations(im, e):
    """Diff the package labels vs review_server.effective_labels() -> op dicts (polygons: normalized ring lists)."""
    e = migrate(e)
    final = effective_labels(im, e)
    by_id = {g["id"]: g for g in final if g.get("id") is not None}
    dec = e["dec"]
    ops = []

    def meta(k):
        d = dec.get(k) or {}
        return {"key": k if d else "editor", "who": d.get("who", ""), "t": d.get("t", e.get("gtT") or e.get("t", 0))}

    for g in im["gt"]:
        k = f"g{g['id']}"
        f = by_id.get(g["id"])
        if f is None:
            ops.append({"op": "delete", "from": g["c"], "to": "", "old": g["p"], "new": None, **meta(k)})
        elif not same_poly(f["p"], g["p"]):
            ops.append({"op": "reshape", "from": g["c"], "to": f["c"], "old": g["p"], "new": f["p"], **meta(k)})
        elif f["c"] != g["c"]:
            ops.append({"op": "reclass", "from": g["c"], "to": f["c"], "old": g["p"], "new": g["p"], **meta(k)})
        elif k in dec and (dec[k].get("v") or "").startswith("keep:"):
            ops.append({"op": "keep", "from": g["c"], "to": g["c"], "old": g["p"], "new": g["p"], **meta(k)})
    for f in final:
        if f.get("id") is None:
            ops.append({"op": "add", "from": "", "to": f["c"], "old": None, "new": f["p"], **meta(f.get("src") or "")})
    for k, d in dec.items():
        if k[0] == "p" and d.get("v") == "no":
            idx = int(k[1:])
            if idx < len(im["pr"]):
                ops.append({"op": "reject", "from": "", "to": "", "old": im["pr"][idx]["p"], "new": None, **meta(k)})
    return ops


def outcome(op):
    """What an operation leaves on the object: deleted, nothing (rejected prediction), or a label of a class."""
    return {"delete": "deleted", "reject": "none"}.get(op["op"], op["to"])


def poly_of(op):
    return op["old"] or op["new"]


def to_px(rings, w, h):
    return [np.asarray(r, np.float64).reshape(-1, 2) * (w, h) for r in rings]


def overlap(a, b, w, h):
    """(IoU, fraction of a inside b, fraction of b inside a) of two normalized ring lists, rasterized inside
    their joint box on a RASTER-px grid."""
    s = RASTER / max(w, h)
    pa, pb = to_px(a, w * s, h * s), to_px(b, w * s, h * s)
    pts = np.vstack(pa + pb)
    lo = np.floor(pts.min(0)).astype(int)
    hi = np.ceil(pts.max(0)).astype(int) + 1
    ma, mb = np.zeros((hi[1] - lo[1], hi[0] - lo[0]), np.uint8), np.zeros((hi[1] - lo[1], hi[0] - lo[0]), np.uint8)
    cv2.fillPoly(ma, [np.round(r - lo).astype(np.int32) for r in pa], 1)
    cv2.fillPoly(mb, [np.round(r - lo).astype(np.int32) for r in pb], 1)
    inter, union = np.logical_and(ma, mb).sum(), np.logical_or(ma, mb).sum()
    na, nb = ma.sum(), mb.sum()
    return (float(inter / union) if union else 0.0, float(inter / na) if na else 0.0, float(inter / nb) if nb else 0.0)


def iou(a, b, w, h):
    return overlap(a, b, w, h)[0]


def box(rings):
    pts = np.vstack([np.asarray(r).reshape(-1, 2) for r in rings])
    return {"lo": pts.min(0), "hi": pts.max(0)}


def near(poly, labels):
    b = box(poly)
    return [lab for lab in labels if (lab["lo"] < b["hi"]).all() and (b["lo"] < lab["hi"]).all()]


def best(poly, labels, w, h):
    """(iou, label) of the current label overlapping poly most; bbox prefilter."""
    top = (0.0, None)
    for lab in near(poly, labels):
        v = iou(poly, lab["p"], w, h)
        if v > top[0]:
            top = (v, lab)
    return top


def max_contain(poly, labels, w, h):
    """(fraction, label, 'inside'|'holds'): the largest share of poly inside one label, or of one label inside poly."""
    top = (0.0, None, "")
    for lab in near(poly, labels):
        _, a_in_b, b_in_a = overlap(poly, lab["p"], w, h)
        if a_in_b > top[0]:
            top = (a_in_b, lab, "inside")
        if b_in_a > top[0]:
            top = (b_in_a, lab, "holds")
    return top


def classify(op, labels, w, h, match_iou=0.9, add_iou=0.5, contain=0.8):
    """One operation vs the current labels of its target image -> (status, label or None, iou, note, contain)."""
    if op["op"] == "add":
        v, lab = best(op["new"], labels, w, h)
        if v >= add_iou:
            when = "already there in September" if lab.get("pre") else "added in Roboflow since"
            if lab["c"] == op["to"]:
                return "already-done", lab, v, f"a {lab['c']} label covers it ({when})", ""
            return "conflict", lab, v, f"overlaps a {lab['c']} label ({when}); separate layer or duplicate?", ""
        f, labc, how = max_contain(op["new"], labels, w, h)
        if f >= contain:
            what = f"lies {f:.0%} inside" if how == "inside" else f"holds {f:.0%} of"
            return "needs-human", labc, v, f"add {what} a {labc['c']} label (IoU {v:.2f})", f"{how} {f:.2f}"
        return "applied-ready", None, v, "", f"{how} {f:.2f}" if how else ""
    if op["op"] == "reject":
        v, lab = best(op["old"], labels, w, h)
        if v >= add_iou and not lab.get("pre"):
            return "conflict", lab, v, f"a {lab['c']} label added in Roboflow since sits on the rejected prediction", ""
        return "already-done", None, v, ("no label change needed" if v < add_iou else
                                         f"duplicate prediction on a {lab['c']} label already there in September"), ""
    v, lab = best(op["old"], labels, w, h)
    if op["op"] == "reshape":
        vn, labn = best(op["new"], labels, w, h)
        if vn >= match_iou and vn > v and labn["c"] == op["to"]:
            return "already-done", labn, vn, "current outline matches the reshaped one", ""
    if v < match_iou:
        if op["op"] == "delete":
            return (("conflict", lab, v, "label changed shape since", "") if v >= add_iou else
                    ("already-done", None, v, "label no longer in Roboflow", ""))
        return (("conflict", lab, v, "label changed shape since", "") if v >= add_iou else
                ("orphan", None, v, "label not found", ""))
    cur = lab["c"]
    if op["op"] == "delete":
        return (("applied-ready", lab, v, "", "") if cur == op["from"] else
                ("conflict", lab, v, f"now {cur} (September deleted a {op['from']})", ""))
    if op["op"] == "keep":
        return ("already-done", lab, v, "kept as is", "") if cur == op["from"] else ("conflict", lab, v, f"now {cur}", "")
    if op["op"] == "reclass":
        if cur == op["to"]:
            return "already-done", lab, v, "already reclassed", ""
        return ("applied-ready", lab, v, "", "") if cur == op["from"] else ("conflict", lab, v, f"now {cur}", "")
    return (("applied-ready", lab, v, "", "") if cur in (op["from"], op["to"]) else
            ("conflict", lab, v, f"now {cur}", ""))  # reshape


def dedupe_adds(rows, w, h, add_iou=0.5):
    """Adds of one source image vs each other: same class at IoU >= add_iou -> one kept, the rest already-done;
    different class -> both conflicts. Sets status/note in place on rows that have none yet."""
    adds = [r for r in rows if r["op"] == "add"]
    rank = sorted(adds, key=lambda r: (r["key"] == "editor", -(r["t"] or 0)))
    for i, a in enumerate(rank):
        for b in rank[i + 1:]:
            if iou(a["new"], b["new"], w, h) < add_iou:
                continue
            if a["to"] == b["to"]:
                if not b["status"] and a["status"] != "already-done":
                    b["status"], b["note"] = "already-done", f"duplicate of add {a['key']} ({a['to']}) on this image"
            else:
                for x, y in ((a, b), (b, a)):
                    if x["status"] != "conflict":
                        x["status"], x["note"] = "conflict", f"overlaps add {y['key']} ({y['to']}) on this image"


def index_coco(coco):
    """{stem: image dict + labels [{id, c, p (normalized), lo, hi}]} from a COCO with extra.name = stem."""
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    cur = {}
    for img in coco["images"]:
        cur[(img.get("extra") or {}).get("name") or stem_of(img["file_name"])] = {**img, "labels": []}
    by_id = {img["id"]: s for s, img in cur.items()}
    for a in coco["annotations"]:
        img = cur[by_id[a["image_id"]]]
        w, h = img["width"], img["height"]
        rings = [[v / (w if k % 2 == 0 else h) for k, v in enumerate(seg)] for seg in a["segmentation"] if len(seg) >= 6]
        img["labels"].append({"id": a["id"], "c": cats[a["category_id"]], "p": rings, **box(rings)})
    return cur


def resolve(pkg_images, edits, cur_img, dup_of, match_iou=0.9, add_iou=0.5, contain=0.8):
    """pkg_images {file: data.js image}, edits {file: record}, cur_img from index_coco(), dup_of {stem: twin}.
    -> (rows, control, unknown files)."""
    pkg_by_stem = {stem_of(f): im for f, im in pkg_images.items()}
    for stem, img in cur_img.items():   # labels already in the package in September
        old = pkg_by_stem.get(stem)
        old_labels = [{"p": g["p"], "c": g["c"], **box(g["p"])} for g in old["gt"]] if old else []
        for lab in img["labels"]:
            lab["pre"] = bool(old_labels) and best(lab["p"], old_labels, img["width"], img["height"])[0] >= match_iou

    rows, unknown, ctrl = [], [], {"untouched": 0, "found": 0, "misses": []}
    for file, e in edits.items():
        im = pkg_images.get(file)
        if im is None:
            unknown.append(file)
            continue
        stem = stem_of(file)
        ops = operations(im, e)
        for n, op in enumerate(ops):
            rows.append({"file": file, "stem": stem, "target_stem": dup_of.get(stem, stem), "n": n, **op,
                         "status": "", "note": "", "twin_note": "", "cur_id": "", "cur_class": "", "iou": "",
                         "contain": "", "seed_key": ""})
        img = cur_img.get(stem)   # control on the image itself (a duplicate's own Roboflow copy)
        if img is None:
            continue
        touched = {json.dumps(o["old"]) for o in ops if o["old"]}
        for g in im["gt"]:
            if json.dumps(g["p"]) in touched:
                continue
            v, lab = best(g["p"], img["labels"], img["width"], img["height"])
            ctrl["untouched"] += 1
            if v >= match_iou and lab["c"] == g["c"]:
                ctrl["found"] += 1
            else:
                ctrl["misses"].append({"stem": stem, "id": g["id"], "c": g["c"], "iou": round(v, 3),
                                       "cur": lab["c"] if lab else ""})
    ctrl["rate"] = round(ctrl["found"] / ctrl["untouched"], 4) if ctrl["untouched"] else None

    by_stem = defaultdict(list)
    for r in rows:
        by_stem[r["stem"]].append(r)
    for stem, rs_ in by_stem.items():
        img = cur_img.get(rs_[0]["target_stem"]) or cur_img.get(stem)
        if img:
            dedupe_adds(rs_, img["width"], img["height"], add_iou)

    def settle(r):
        img = cur_img.get(r["target_stem"])
        if img is None:
            r["status"], r["note"] = "orphan", "image not in the current labels"
            return
        status, lab, v, note, cont = classify(r, img["labels"], img["width"], img["height"], match_iou, add_iou, contain)
        r["status"], r["note"], r["iou"], r["contain"] = status, note, round(v, 3), cont
        if lab:
            r["cur_id"], r["cur_class"] = lab["id"], lab["c"]

    own = [r for r in rows if r["stem"] == r["target_stem"]]
    dups = [r for r in rows if r["stem"] != r["target_stem"]]
    for r in own:
        if not r["status"]:
            settle(r)
    for r in dups:   # after the twins' own operations have a status
        twin_ops = by_stem.get(r["target_stem"], [])
        tw = cur_img.get(r["target_stem"]) or {"width": 1, "height": 1}
        scored = [(iou(poly_of(r), poly_of(t), tw["width"], tw["height"]), t) for t in twin_ops]
        v, t = max(scored, key=lambda x: x[0], default=(0.0, None))
        if t is not None and v >= add_iou:
            same = outcome(t) == outcome(r)
            what = f"{t['op']}{':' + t['to'] if t['to'] else ''} ({t['key']}, IoU {v:.2f})"
            if same:
                r["twin_note"] = f"twin made the same call: {what}, {t['status']}"
                if t["status"] in ("applied-ready", "already-done"):
                    r["status"], r["note"] = "already-done", "covered by the twin's own decision"
                else:
                    r["status"], r["note"] = "needs-human", f"twin's same decision is itself {t['status']}"
            else:
                r["twin_note"] = f"twin decided {what}"
                r["status"], r["note"] = "conflict", "twin has a different decision on the same object"
            continue
        if r["status"]:
            continue
        settle(r)
        if r["op"] == "delete" and r["status"] == "already-done":
            mine = cur_img.get(r["stem"])
            if mine and best(r["old"], mine["labels"], mine["width"], mine["height"])[0] >= match_iou:
                r["status"], r["note"] = "needs-human", ("not in the twin, but still in the dropped duplicate's "
                                                         "own Roboflow copy")

    claims = defaultdict(list)
    for r in rows:
        if r["status"] == "applied-ready" and r["cur_id"] != "":
            claims[(r["target_stem"], r["cur_id"])].append(r)
    for rs_ in claims.values():
        if len({(r["op"], r["to"]) for r in rs_}) > 1:
            for r in rs_:
                r["status"], r["note"] = "conflict", "another decision targets the same current label"
    return rows, ctrl, unknown


def seed_records(rows, cur_img, preds_of, add_iou=0.5):
    """Applied-ready ops -> {stem: edit record}, keyed to a package built from the same COCO and predictions.
    preds_of(stem) -> that package's pr list. Sets r['seed_key']."""
    per_img = defaultdict(dict)
    for r in rows:
        if r["status"] != "applied-ready":
            continue
        img = cur_img[r["target_stem"]]
        seed = f"september:{r['file']}:{r['key']}"
        d = {"t": r["t"], "who": r["who"] or "September review", "seed": seed}
        dec = per_img[r["target_stem"]]
        if r["op"] == "delete":
            r["seed_key"] = f"g{r['cur_id']}"
            dec[r["seed_key"]] = {"v": "del", **d}
        elif r["op"] == "reclass":
            r["seed_key"] = f"g{r['cur_id']}"
            dec[r["seed_key"]] = {"v": f"keep:{r['to']}", **d}
        elif r["op"] == "reshape":
            r["seed_key"] = f"g{r['cur_id']}"
            dec[r["seed_key"]] = {"v": f"keep:{r['to']}", "p": r["new"], **d}
        elif r["op"] == "add":
            preds = preds_of(r["target_stem"])
            scored = sorted(((iou(r["new"], p["p"], img["width"], img["height"]), k) for k, p in enumerate(preds)),
                            reverse=True)
            free = [(v, k) for v, k in scored if f"p{k}" not in dec]
            if free and free[0][0] >= add_iou:
                r["seed_key"] = f"p{free[0][1]}"
                dec[r["seed_key"]] = {"v": f"add:{r['to']}", "p": r["new"], **d}
            else:
                r["seed_key"] = "nsep" + hashlib.md5(seed.encode()).hexdigest()[:10]
                dec[r["seed_key"]] = {"v": f"new:{r['to']}", "p": r["new"], **d}
    out = {}
    for stem, dec in per_img.items():
        if dec:
            out[stem] = {"file": cur_img[stem]["file_name"],
                         "edit": {"dec": dec, "t": max(x["t"] for x in dec.values()), "modified": True,
                                  "reviewed": False, "reviewedT": 0}}
    return out


def seeded_overlaps(records, cur_img, add_iou=0.5):
    """Pairs of seeded outlines (adds, reshapes) on one image that overlap at IoU >= add_iou; should be none."""
    bad = []
    for stem, rec in records.items():
        img = cur_img[stem]
        polys = [(k, d["p"]) for k, d in rec["edit"]["dec"].items() if d.get("p")]
        for i, (ka, a) in enumerate(polys):
            for kb, b in polys[i + 1:]:
                v = iou(a, b, img["width"], img["height"])
                if v >= add_iou:
                    bad.append((stem, ka, kb, round(v, 3)))
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--package", required=True, type=Path, help="review package folder (data.js, edits/)")
    ap.add_argument("--edits-dir", type=Path, default=None, help="default: <package>/edits")
    ap.add_argument("--current-coco", required=True, type=Path)
    ap.add_argument("--dup-csv", type=Path, help="CSV with stem, duplicate_of (dropped copy -> kept twin)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--match-iou", type=float, default=0.9)
    ap.add_argument("--add-iou", type=float, default=0.5)
    ap.add_argument("--contain", type=float, default=0.8, help="an add this much inside a label (or holding one) "
                                                               "is needs-human")
    ap.add_argument("--pred-dir", type=Path, help="new package's predictions, to key seeded adds as p<index>")
    ap.add_argument("--pred-classes", default="EV,non-EV")
    ap.add_argument("--seed-dir", type=Path, help="write applied-ready ops as edit records here")
    args = ap.parse_args()

    data = read_data_js(args.package / "data.js")
    pkg_images = {im["file"]: im for im in data["images"]}
    edits = {}
    for f in sorted((args.edits_dir or args.package / "edits").glob("*.json")):
        rec = json.loads(f.read_text(encoding="utf-8"))
        edits[rec["file"]] = rec["edit"]
    dup_of = {}
    if args.dup_csv:
        with open(args.dup_csv, newline="", encoding="utf-8") as fh:
            dup_of = {r["stem"]: r["duplicate_of"] for r in csv.DictReader(fh) if r.get("duplicate_of")}
    cur_img = index_coco(json.loads(args.current_coco.read_text(encoding="utf-8")))

    rows, ctrl, unknown = resolve(pkg_images, edits, cur_img, dup_of, args.match_iou, args.add_iou, args.contain)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "control.json").write_text(json.dumps(ctrl, indent=1), encoding="utf-8")
    print(f"control: {ctrl['found']} of {ctrl['untouched']} untouched original labels still in the current labels "
          f"(IoU >= {args.match_iou}, same class): {ctrl['rate']}")

    if args.seed_dir:
        classes = [c.strip() for c in args.pred_classes.split(",")]
        cache = {}

        def preds_of(stem):
            if stem not in cache:
                ps = load_predictions(args.pred_dir / f"{stem}.txt", classes) if args.pred_dir else []
                cache[stem] = [p for p in ps if len(p["p"][0]) >= 6]
            return cache[stem]
        records = seed_records(rows, cur_img, preds_of, args.add_iou)
        args.seed_dir.mkdir(parents=True, exist_ok=True)
        for rec in records.values():
            (args.seed_dir / f"{rec['file']}.json").write_text(json.dumps(rec), encoding="utf-8")
        keys = Counter(k[0] for rec in records.values() for k in rec["edit"]["dec"])
        print(f"seeded {sum(keys.values())} decisions on {len(records)} images -> {args.seed_dir} "
              f"(g: {keys['g']}, p: {keys['p']}, n drawn: {keys['n']})")
        bad = seeded_overlaps(records, cur_img, args.add_iou)
        print(f"seeded outlines overlapping each other at IoU >= {args.add_iou}: {len(bad)}" + (f" {bad}" if bad else ""))

    with open(args.out / "resolution.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        for r in rows:
            w.writerow({**{k: r.get(k, "") for k in COLS},
                        "old": json.dumps(r["old"]) if r["old"] else "", "new": json.dumps(r["new"]) if r["new"] else ""})
    print(f"{len(edits)} edit records, {len(rows)} operations -> {args.out / 'resolution.csv'}")
    if unknown:
        print(f"  {len(unknown)} edit records name a file not in the package: {unknown[:3]}")
    print("  " + ", ".join(f"{k}: {v}" for k, v in sorted(Counter(r["status"] for r in rows).items())))
    print("  " + ", ".join(f"{k}: {v}" for k, v in sorted(Counter((r["op"], r["status"]) for r in rows).items())))


if __name__ == "__main__":
    main()
