"""
Apply a review package's saved edits (<package>/edits/<file>.json from review_server.py) to the
package's own base COCO pull, and write a new, dated label version (docs/LABEL_CHANGES.md, contract v1):

  <out-root>/lv_<YYYYMMDD_HHMM>/labels.coco.json   the base with the edits applied
                               changes.jsonl      one row per object change (schema label_change/1)
                               conflicts.csv      the rows that were not applied, with the reason
                               version.json       inputs, hashes, counts

Each image's labels are rebuilt with review_server.effective_labels() (the reference applier, the same
rules as effGt() in review_tool/index.html) and diffed against the package labels per object:

  keep             g<id> "keep:<cls>" that changes nothing (logged, no label change)
  reclass          class changed, outline unchanged       (base segmentation kept, category_id changed)
  reshape          outline changed                         (segmentation from the edit)
  reclass+reshape  both
  delete           label gone ("del", or removed from an Editor-saved list)
  add              accepted prediction p<k>
  new              outline drawn in the tool: n<uid>, or a list item with no key (dec_key "gt")
  reject           prediction refused ("no"): logged for model scoring, no label change

Conflicts (status "conflict", also in conflicts.csv; nothing is written for them, so a base label keeps
its base class and outline):
  duplicate_outline      a changed/added outline overlaps another final outline at mask IoU >= --dup-iou
                         (default 0.9, full-resolution masks). Both sides conflict when both are changes;
                         the reason notes whether the classes differ.
  g_missing_from_base    a g<id> decision or list item whose id is not a base annotation of that image
  pred_index_out_of_range  p<k> with k not in 0..len(pr)-1
  invalid_polygon        an outline that, in pixels and clamped to the image, has a ring with < 3 distinct
                         points or zero area
  image_not_in_package / image_not_in_base   a record whose file has no data.js / base COCO image
  duplicate_id           the same base id twice in one image's labels

Not logged (no label change, no review verdict): null (undone) decisions, n<uid>/p<k> "del" of an
outline that was drawn/accepted and then removed again. Counted in version.json "skipped".

COCO output: images and categories as in the base (+ any class missing from the base, flagged in
version.json); annotations sorted by id; base ids kept, new ids max(base id)+1, ... in base image order;
bbox and area recomputed from the segmentation for every annotation; edited outlines are normalized x
image w/h, clamped to [0, w] x [0, h], rounded to 0.01 px; several rings stay one annotation. No time or
version id inside labels.coco.json, so the same inputs give the same file hash.

Read-only on the base, package and edits. Refuses --out-root under Box or Cryo-image-processing, and
never overwrites an existing lv_ folder.

  python annotation/apply_review_edits.py --base <pull>/current.coco.json --package <review package> [--dry-run]
      [--edits <dir>] [--out-root <CryoAI-laptop>/labels] [--people annotation/people.json] [--stamp YYYYMMDD_HHMM]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
from review_server import _migrate, effective_labels, load_people  # noqa: E402
from inference.inference_check import PROTECTED_ROOTS, _under  # noqa: E402

SCHEMA = "label_change/1"
DUP_IOU = 0.9
PEOPLE = HERE / "people.json"
CLASS_GROUPS = HERE / "class_groups.yaml"
INTERNAL = ("u", "x", "lc", "seed", "src", "t")  # data.js / tool fields, never written to COCO
CHANGE_OPS = ("reclass", "reshape", "reclass+reshape", "add", "new")
OPS = ("keep", "reclass", "reshape", "reclass+reshape", "delete", "add", "new", "reject")
ROW_FIELDS = ["schema", "label_version", "base", "pkg", "image", "rf_id", "dec_key", "op", "ann_old", "ann_new",
              "cls_from", "cls_to", "who_raw", "who", "who_auth", "t", "rx", "seed", "reviewed", "status", "reason"]
CONFLICT_FIELDS = ["image", "rf_id", "file", "dec_key", "op", "reason", "ann_old", "cls_from", "cls_to",
                   "who_raw", "who", "t", "seed"]
RF_SUFFIX = re.compile(r"\.rf\.[0-9a-f]+\.\w+$")


# ---------------------------------------------------------------- inputs

def read_data_js(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8").split("=", 1)[1].strip().rstrip(";"))


def stem_of(file_name: str) -> str:
    """Same rule as build_review_package.stem_of (kept here so this script needs no cv2/PIL/train_yolo chain)."""
    name = Path(file_name).name
    if RF_SUFFIX.search(name):
        return RF_SUFFIX.sub("", name)
    p = Path(name)
    return f"{p.stem}_{p.suffix[1:]}" if p.suffix else name


def sha256_file(path: Path) -> str | None:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def git_sha() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True,
                             text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO,
                               capture_output=True, text=True).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def roster_lookup(people: list):
    """who_raw -> roster id ('unknown' if no alias matches). Case- and space-insensitive; an id matches itself."""
    table = {}
    for p in people:
        for name in [p.get("id"), p.get("name"), *(p.get("aliases") or [])]:
            if isinstance(name, str) and name.strip():
                table.setdefault(name.strip().casefold(), p["id"])
    return lambda raw: table.get(raw.strip().casefold(), "unknown") if isinstance(raw, str) else "unknown"


def load_records(edits_dir: Path) -> list:
    """[(path, wrapper)] sorted by file name. Unreadable files abort: an edit must never be lost silently."""
    out = []
    for f in sorted(Path(edits_dir).glob("*.json")):
        try:
            rec = json.loads(f.read_text(encoding="utf-8"))
        except ValueError as e:
            raise SystemExit(f"[ERROR] {f}: not JSON ({e})")
        if not isinstance(rec, dict) or "file" not in rec or not isinstance(rec.get("edit"), dict):
            raise SystemExit(f"[ERROR] {f}: not an edit record {{file, edit}}")
        out.append((f, rec))
    return out


def out_root_problem(root: Path, protected=PROTECTED_ROOTS) -> str | None:
    """Why --out-root may not be used, or None (inference_check's pattern: protected roots at any depth)."""
    root = Path(os.path.abspath(root))
    for p in protected:
        if _under(root, Path(os.path.abspath(p))):
            return f"--out-root {root} is under {p}, which is read-only; write label versions elsewhere"
    return None


# ---------------------------------------------------------------- geometry

def to_px(rings, w, h):
    """Normalized rings -> pixel rings, clamped to the image, rounded to 0.01 px."""
    out = []
    for r in rings:
        out.append([round(min(max(float(v) * (w if k % 2 == 0 else h), 0.0), float(w if k % 2 == 0 else h)), 2)
                    for k, v in enumerate(r)])
    return out


def ring_area(r) -> float:
    xs, ys = r[0::2], r[1::2]
    n = len(xs)
    return abs(sum(xs[i] * ys[(i + 1) % n] - xs[(i + 1) % n] * ys[i] for i in range(n))) / 2


def polygon_problem(rings) -> str | None:
    """None if every ring is a usable polygon (>= 3 distinct finite points, non-zero area)."""
    if not isinstance(rings, list) or not rings:
        return "no rings"
    for r in rings:
        if not isinstance(r, list) or len(r) < 6 or len(r) % 2:
            return "ring with < 3 points"
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in r):
            return "non-numeric coordinate"
        if len({(r[i], r[i + 1]) for i in range(0, len(r), 2)}) < 3:
            return "ring with < 3 distinct points after clamping"
        if ring_area(r) <= 0:
            return "ring with zero area after clamping"
    return None


def bbox_area(seg):
    xs = [v for r in seg for v in r[0::2]]
    ys = [v for r in seg for v in r[1::2]]
    x0, y0 = min(xs), min(ys)
    return [round(x0, 2), round(y0, 2), round(max(xs) - x0, 2), round(max(ys) - y0, 2)], \
        round(sum(ring_area(r) for r in seg), 2)


def mask_iou(a, b) -> float:
    """IoU of two pixel ring lists (union of rings), rasterized at full resolution over their joint box."""
    ba, bb = bbox_area(a)[0], bbox_area(b)[0]
    if ba[0] > bb[0] + bb[2] or bb[0] > ba[0] + ba[2] or ba[1] > bb[1] + bb[3] or bb[1] > ba[1] + ba[3]:
        return 0.0
    x0, y0 = math.floor(min(ba[0], bb[0])), math.floor(min(ba[1], bb[1]))
    x1, y1 = math.ceil(max(ba[0] + ba[2], bb[0] + bb[2])), math.ceil(max(ba[1] + ba[3], bb[1] + bb[3]))
    shape = (y1 - y0 + 1, x1 - x0 + 1)

    def m(seg):
        mask = np.zeros(shape, dtype=np.uint8)
        cv2.fillPoly(mask, [np.round(np.asarray(r, dtype=np.float64).reshape(-1, 2) - [x0, y0]).astype(np.int32)
                            for r in seg], 1)
        return mask.astype(bool)
    ma, mb = m(a), m(b)
    union = np.logical_or(ma, mb).sum()
    return float(np.logical_and(ma, mb).sum() / union) if union else 0.0


# ---------------------------------------------------------------- one image

def _same(a, b) -> bool:
    return json.dumps(a) == json.dumps(b)  # exact, as resolve_review_edits.same_poly


def _pred_index(key: str, im: dict):
    try:
        k = int(key[1:])
    except ValueError:
        return None
    return k if 0 <= k < len(im.get("pr") or []) else None


def image_ops(im: dict, rec: dict, base_anns: dict, w: int, h: int, dup_iou: float = DUP_IOU):
    """One image's edit record -> (ops, skipped). base_anns: {id: COCO annotation} of this image only.

    Each op: {dec_key, op, ann_old, cls_from, cls_to, p (normalized, final), px (pixel, final), d (decision
    or None), item (the final label), conflict (reason or None)}. Final labels = base anns with every
    non-conflict op applied."""
    edit = rec["edit"]
    e = _migrate(edit)
    dec = e["dec"]
    final = effective_labels(im, edit)
    pkg_gt = {g["id"]: g for g in im["gt"]}
    ops, skipped = [], Counter()

    def op_row(key, op, d, item=None, ann_old=None, cls_from=None, cls_to=None, conflict=None):
        ops.append({"dec_key": key, "op": op, "d": d, "item": item, "ann_old": ann_old, "cls_from": cls_from,
                    "cls_to": cls_to, "p": item["p"] if item else None, "px": None, "conflict": conflict})

    def explains(gkey, f):
        """The g decision accounts for this object's final state (else the change came from the list)."""
        d = dec.get(gkey)
        v = d.get("v") if d else None
        if not v:
            return False
        if f is None:
            return v == "del"
        return v == f"keep:{f['c']}" and (d.get("p") is None or _same(d["p"], f["p"]))

    seen_ids = set()
    final_by_id = {}
    for f in final:
        if f.get("id") is None:
            continue
        fid = f["id"]
        key = f"g{fid}"
        d = dec.get(key) if explains(key, f) else None
        if fid in seen_ids:
            op_row(key if d else "gt", "reshape", d, f, fid, None, f["c"], "duplicate_id")
            continue
        seen_ids.add(fid)
        final_by_id[fid] = f
        if fid not in base_anns:
            op_row(key if d else "gt", "reshape" if not (fid in pkg_gt and _same(f["p"], pkg_gt[fid]["p"])) else
                   "reclass", d, f, fid, (pkg_gt.get(fid) or {}).get("c"), f["c"], "g_missing_from_base")
            continue
        g = pkg_gt.get(fid)
        if g is None:  # a base annotation the package dropped (no valid ring) but a list carries: treat as reshape
            op_row(key if d else "gt", "reshape", d, f, fid, None, f["c"])
            continue
        moved, recl = not _same(f["p"], g["p"]), f["c"] != g["c"]
        if moved or recl:
            op_row(key if d else "gt", "reclass+reshape" if moved and recl else "reshape" if moved else "reclass",
                   d, f, fid, g["c"], f["c"])
        elif d and (d.get("v") or "").startswith("keep:"):
            op_row(key, "keep", d, f, fid, g["c"], g["c"])
    for g in im["gt"]:
        if g["id"] in final_by_id or g["id"] in seen_ids:
            continue
        key = f"g{g['id']}"
        d = dec.get(key) if explains(key, None) else None
        op_row(key if d else "gt", "delete", d, None, g["id"], g["c"], None,
               None if g["id"] in base_anns else "g_missing_from_base")
    for f in final:
        if f.get("id") is not None:
            continue
        src = f.get("src")
        if src and src[0] == "p":
            d = dec.get(src)
            op_row(src, "add", d if d and d.get("v") else None, f, None, None, f["c"],
                   None if _pred_index(src, im) is not None else "pred_index_out_of_range")
        elif src:
            d = dec.get(src)
            op_row(src, "new", d if d and d.get("v") else None, f, None, None, f["c"])
        else:
            op_row("gt", "new", None, f, None, None, f["c"])
    present_src = {f.get("src") for f in final if f.get("src")}
    for key, d in dec.items():
        v = d.get("v")
        if key[0] == "g":
            try:
                gid = int(key[1:])
            except ValueError:
                gid = None
            if gid in base_anns or gid in pkg_gt:
                if not v:
                    skipped["undone"] += 1
                continue
            if v:  # a decision on a label this image doesn't have: effective_labels ignores it
                op_row(key, "delete" if v == "del" else "reshape" if d.get("p") else "reclass" if v.startswith("keep:")
                       else "keep", d, None, gid, None, v[5:] if v.startswith("keep:") else None, "g_missing_from_base")
            else:
                skipped["undone"] += 1
            continue
        if not v:
            skipped["undone"] += 1
            continue
        if key in present_src:
            continue
        if key[0] == "p" and v == "no":
            k = _pred_index(key, im)
            op_row(key, "reject", d, None, None, None, None, None if k is not None else "pred_index_out_of_range")
        elif key[0] == "p" and v.startswith("add:"):
            if _pred_index(key, im) is None:
                op_row(key, "add", d, {"id": None, "c": v[4:], "p": d.get("p")}, None, None, v[4:],
                       "pred_index_out_of_range")
            else:
                skipped["removed_again"] += 1  # list newer than the decision dropped it
        elif key[0] == "n" and v.startswith("new:") and not d.get("p"):
            op_row(key, "new", d, None, None, None, v[4:], "invalid_polygon")
        elif v in ("del", "no"):
            skipped["removed_again"] += 1
        else:
            skipped["superseded"] += 1

    # pixels + polygon validity for everything that writes an outline
    for o in ops:
        if o["op"] in ("reshape", "reclass+reshape", "add", "new") and o["p"] is not None and not o["conflict"]:
            try:
                px = to_px(o["p"], w, h)
            except (TypeError, ValueError):
                px = None
            prob = polygon_problem(px) if px is not None else "not a list of numeric rings"
            if prob:
                o["conflict"] = f"invalid_polygon: {prob}"
            else:
                o["px"] = px

    # duplicate outlines, repeated until stable (a reverted change can expose another pair)
    def candidates():
        """Final outlines (pixel) on this image: (op or None, key, class, px). Base labels as in the base, with
        the non-conflict ops applied."""
        cur = {aid: (None, f"g{aid}", a["category_name"], a["segmentation"]) for aid, a in base_anns.items()}
        out = []
        for o in ops:
            if o["conflict"]:
                continue
            if o["op"] == "delete":
                cur.pop(o["ann_old"], None)
            elif o["op"] in ("reclass",):
                cur[o["ann_old"]] = (o, o["dec_key"], o["cls_to"], base_anns[o["ann_old"]]["segmentation"])
            elif o["op"] in ("reshape", "reclass+reshape"):
                cur[o["ann_old"]] = (o, o["dec_key"], o["cls_to"], o["px"])
            elif o["op"] in ("add", "new"):
                out.append((o, o["dec_key"], o["cls_to"], o["px"]))
        return [cur[k] for k in sorted(cur)] + out

    while True:
        cands = candidates()
        hits = {}
        for i in range(len(cands)):
            for j in range(i + 1, len(cands)):
                a, b = cands[i], cands[j]
                if a[0] is None and b[0] is None:
                    continue  # two untouched base labels: not an edit
                v = mask_iou(a[3], b[3])
                if v >= dup_iou:
                    kind = "different class" if a[2] != b[2] else "same class"
                    for x, y in ((a, b), (b, a)):
                        if x[0] is not None and id(x[0]) not in hits:
                            hits[id(x[0])] = (x[0], f"duplicate_outline: IoU {v:.3f} with {y[1]} ({y[2]}), {kind}")
        if not hits:
            break
        for o, reason in hits.values():
            o["conflict"] = reason
    return ops, skipped


# ---------------------------------------------------------------- run

def apply(base_path: Path, package: Path, edits_dir: Path, people: list, dup_iou: float = DUP_IOU):
    """Everything but writing: -> (coco_out, rows (without label_version), conflicts, info)."""
    base = json.loads(Path(base_path).read_text(encoding="utf-8"))
    data = read_data_js(Path(package) / "data.js")
    who_of = roster_lookup(people)
    cat_id = {c["name"]: c["id"] for c in base["categories"]}
    cat_name = {c["id"]: c["name"] for c in base["categories"]}
    images_by_file = {i["file_name"]: i for i in base["images"]}
    anns_by_img = {}
    for a in base["annotations"]:
        anns_by_img.setdefault(a["image_id"], {})[a["id"]] = {**a, "category_name": cat_name.get(a["category_id"])}
    pkg_images = {im["file"]: im for im in data["images"]}
    base_name, pkg_id = Path(base_path).parent.name, data.get("id", "")

    # The package must come from this base: every data.js label is a base annotation of the same image and class
    mismatched = []
    for im in data["images"]:
        b = images_by_file.get(im["file"])
        if b is None:
            continue
        mine = anns_by_img.get(b["id"], {})
        for g in im["gt"]:
            a = mine.get(g["id"])
            if a is not None and a["category_name"] != g["c"]:
                mismatched.append(f"{im['file']} g{g['id']}: package {g['c']}, base {a['category_name']}")
        if (b["width"], b["height"]) != (im["w"], im["h"]):
            mismatched.append(f"{im['file']}: package {im['w']}x{im['h']}, base {b['width']}x{b['height']}")
    if mismatched:
        raise SystemExit("[ERROR] the package was not built from this base (or the base changed):\n  "
                         + "\n  ".join(mismatched[:10]) + (f"\n  ... {len(mismatched)} in all" if len(mismatched) > 10 else ""))

    records = load_records(edits_dir)
    for f, rec in records:
        if rec.get("pkg") and pkg_id and rec["pkg"] != pkg_id:
            raise SystemExit(f"[ERROR] {f.name}: record is for package {rec['pkg']!r}, this package is {pkg_id!r}")
    by_file = {}
    for f, rec in records:
        if rec["file"] in by_file:
            raise SystemExit(f"[ERROR] two records for {rec['file']!r}: {by_file[rec['file']][0].name}, {f.name}")
        by_file[rec["file"]] = (f, rec)

    rows, skipped, applied_images, n_unchanged = [], Counter(), 0, 0
    per_image = {}  # base image id -> ops
    order = {i["file_name"]: k for k, i in enumerate(base["images"])}
    for file in sorted(by_file, key=lambda f: (order.get(f, len(order)), f)):
        f, rec = by_file[file]
        e = _migrate(rec["edit"])
        if not (e["dec"] or e.get("modified")):
            n_unchanged += 1
            continue
        b, im = images_by_file.get(file), pkg_images.get(file)
        stem = (b.get("extra") or {}).get("name") if b else None
        common = {"image": stem or stem_of(file), "rf_id": (b.get("extra") or {}).get("roboflow_id") if b else None,
                  "reviewed": bool(e.get("reviewed")), "file": file}
        if b is None or im is None:
            rows.append({**common, "dec_key": "*", "op": "keep", "ann_old": None, "cls_from": None, "cls_to": None,
                         "d": None, "item": None, "rx": None, "conflict": "image_not_in_base" if b is None
                         else "image_not_in_package"})
            continue
        applied_images += 1
        ops, sk = image_ops(im, rec, anns_by_img.get(b["id"], {}), b["width"], b["height"], dup_iou)
        skipped.update(sk)
        per_image[b["id"]] = ops
        rx = rec.get("rx") or {}
        for o in ops:
            rows.append({**common, **o, "rx": rx.get(o["dec_key"]), "_op": o})  # _op: ann_new is set below

    # COCO: base + applied ops; new ids after the base max, in base image order then op order
    cats = [dict(c) for c in base["categories"]]
    new_cats = []

    def category(name):
        if name not in cat_id:
            cid = max([c["id"] for c in cats] or [0]) + 1
            cat_id[name] = cid
            cats.append({"id": cid, "name": name})
            new_cats.append(name)
        return cat_id[name]

    next_id = max([a["id"] for a in base["annotations"]] or [0]) + 1
    out_anns = {}
    for a in base["annotations"]:
        out_anns[a["id"]] = dict(a)
    for img in base["images"]:
        for o in per_image.get(img["id"], []):
            if o["conflict"] or o["op"] in ("keep", "reject"):
                continue
            if o["op"] == "delete":
                out_anns.pop(o["ann_old"], None)
                o["ann_new"] = None
            elif o["op"] == "reclass":
                out_anns[o["ann_old"]]["category_id"] = category(o["cls_to"])
                o["ann_new"] = o["ann_old"]
            elif o["op"] in ("reshape", "reclass+reshape"):
                a = out_anns[o["ann_old"]]
                a["category_id"], a["segmentation"] = category(o["cls_to"]), o["px"]
                o["ann_new"] = o["ann_old"]
            elif o["op"] in ("add", "new"):
                out_anns[next_id] = {"id": next_id, "image_id": img["id"], "category_id": category(o["cls_to"]),
                                     "segmentation": o["px"], "iscrowd": 0}
                o["ann_new"] = next_id
                next_id += 1
    anns = []
    for aid in sorted(out_anns):
        a = {k: v for k, v in out_anns[aid].items() if k not in INTERNAL and k != "category_name"}
        a["bbox"], a["area"] = bbox_area(a["segmentation"])
        anns.append(a)
    info = dict(base.get("info") or {})
    info["label_change"] = {"schema": SCHEMA, "base": base_name, "pkg": pkg_id}
    coco = {**{k: v for k, v in base.items() if k not in ("info", "images", "annotations", "categories")},
            "info": info, "images": base["images"], "annotations": anns, "categories": cats}

    out_rows, conflicts = [], []
    for r in rows:
        d, item = r.get("d"), r.get("item")
        who_raw = d.get("who") if d else None
        t = d.get("t") if d else (item.get("t") if item and item.get("t") is not None else
                                  (_migrate(by_file[r["file"]][1]["edit"]).get("gtT") if r["dec_key"] == "gt" else None))
        seed = (d.get("seed") if d else None) or (item.get("seed") if item else None)
        row = {"schema": SCHEMA, "label_version": None, "base": base_name, "pkg": pkg_id, "image": r["image"],
               "rf_id": r["rf_id"], "dec_key": r["dec_key"], "op": r["op"], "ann_old": r["ann_old"],
               "ann_new": None if r["conflict"] else (r["_op"].get("ann_new") if r["op"] not in ("keep", "reject") else r["ann_old"]),
               "cls_from": r["cls_from"], "cls_to": r["cls_to"], "who_raw": who_raw, "who": who_of(who_raw),
               "who_auth": None, "t": t, "rx": r["rx"], "seed": seed, "reviewed": r["reviewed"],
               "status": "conflict" if r["conflict"] else "applied", "reason": r["conflict"]}
        out_rows.append(row)
        if r["conflict"]:
            conflicts.append({"file": r["file"], **{k: row[k] for k in CONFLICT_FIELDS if k in row}})
    counts = Counter(r["op"] for r in out_rows if r["status"] == "applied")
    info_out = {"n_records": len(records), "n_images_applied": applied_images, "n_records_without_changes": n_unchanged,
                "counts_by_op": {k: counts.get(k, 0) for k in OPS},
                "counts_conflict_by_op": dict(sorted(Counter(r["op"] for r in out_rows if r["status"] == "conflict").items())),
                "n_conflicts": len(conflicts), "skipped": dict(sorted(skipped.items())),
                "new_categories": [{"id": cat_id[n], "name": n,
                                    "n_annotations": sum(1 for a in anns if a["category_id"] == cat_id[n]),
                                    "note": "not in the base categories; pushing it to Roboflow needs the user's OK"}
                                   for n in new_cats],
                "n_annotations_base": len(base["annotations"]), "n_annotations_out": len(anns),
                "base_name": base_name, "pkg": pkg_id}
    return coco, out_rows, conflicts, info_out


def coco_bytes(coco: dict) -> bytes:
    return (json.dumps(coco, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def conflicts_csv(conflicts: list) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=CONFLICT_FIELDS, lineterminator="\n")
    w.writeheader()
    for c in conflicts:
        w.writerow({k: c.get(k) for k in CONFLICT_FIELDS})
    return buf.getvalue()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, type=Path, help="The package's base pull: <pull>/current.coco.json")
    ap.add_argument("--package", required=True, type=Path, help="Review package folder (data.js)")
    ap.add_argument("--edits", type=Path, default=None, help="Edit records folder (default: <package>/edits)")
    ap.add_argument("--out-root", type=Path, default=None,
                    help="Where lv_<stamp>/ is created (default: <base pull folder>/../labels)")
    ap.add_argument("--people", type=Path, default=PEOPLE, help="Reviewer roster (default: annotation/people.json)")
    ap.add_argument("--class-groups", type=Path, default=CLASS_GROUPS, help="Recorded by hash in version.json")
    ap.add_argument("--dup-iou", type=float, default=DUP_IOU, help="Mask IoU at or above which two outlines are duplicates")
    ap.add_argument("--stamp", default=None, help="Version stamp YYYYMMDD_HHMM (default: now)")
    ap.add_argument("--dry-run", action="store_true", help="Report only; write nothing")
    a = ap.parse_args(argv)

    base, package = a.base.resolve(), a.package.resolve()
    edits_dir = (a.edits or package / "edits").resolve()
    out_root = (a.out_root or base.parent.parent / "labels").resolve()
    problem = out_root_problem(out_root)
    if problem:
        raise SystemExit(f"[ERROR] {problem}")
    stamp = a.stamp or datetime.now().strftime("%Y%m%d_%H%M")
    if not re.fullmatch(r"\d{8}_\d{4}", stamp):
        raise SystemExit(f"[ERROR] --stamp {stamp!r}: expected YYYYMMDD_HHMM")
    lv = f"lv_{stamp}"
    out = out_root / lv
    if out.exists():
        raise SystemExit(f"[ERROR] {out} exists; label versions are never overwritten")
    for p, what in ((base, "--base"), (package / "data.js", "--package data.js"), (edits_dir, "--edits")):
        if not p.exists():
            raise SystemExit(f"[ERROR] {what} not found: {p}")
    people = load_people(a.people)
    print("[steps] effective_labels() per record -> per-object diff vs package labels; "
          f"duplicates at mask IoU >= {a.dup_iou}; pixels = normalized x w/h, clamped, 0.01 px; "
          "bbox/area recomputed; conflicts not written", flush=True)
    print(f"[in] base {base}\n[in] package {package}\n[in] edits {edits_dir}\n[in] people {a.people} ({len(people)})", flush=True)

    coco, rows, conflicts, info = apply(base, package, edits_dir, people, a.dup_iou)
    for r in rows:
        r["label_version"] = lv
    labels = coco_bytes(coco)
    version = {"id": lv, "base": info["base_name"], "base_path": str(base), "base_sha256": sha256_file(base),
               "pkg": info["pkg"], "package_path": str(package), "edits_dir": str(edits_dir),
               "n_records": info["n_records"], "n_images_applied": info["n_images_applied"],
               "n_records_without_changes": info["n_records_without_changes"],
               "counts_by_op": info["counts_by_op"], "counts_conflict_by_op": info["counts_conflict_by_op"],
               "n_conflicts": info["n_conflicts"], "skipped": info["skipped"],
               "new_categories": info["new_categories"], "new_categories_flag": bool(info["new_categories"]),
               "n_annotations_base": info["n_annotations_base"], "n_annotations_out": info["n_annotations_out"],
               "labels_sha256": hashlib.sha256(labels).hexdigest(), "dup_iou": a.dup_iou,
               "class_groups_sha256": sha256_file(a.class_groups), "people_sha256": sha256_file(a.people),
               "tool": f"apply_review_edits.py {git_sha()}", "created": datetime.now().isoformat(timespec="seconds")}
    print(f"[OK] {info['n_records']} records, {info['n_images_applied']} images applied; "
          f"annotations {info['n_annotations_base']} -> {info['n_annotations_out']}")
    print("[OK] applied by op: " + ", ".join(f"{k} {v}" for k, v in info["counts_by_op"].items()))
    print(f"[OK] conflicts: {info['n_conflicts']} {info['counts_conflict_by_op']}; skipped {info['skipped']}")
    for c in conflicts:
        print(f"  [conflict] {c['image']} {c['dec_key']} {c['op']}: {c['reason']}")
    if info["new_categories"]:
        print(f"[WARN] classes not in the base, added as categories: {[c['name'] for c in info['new_categories']]}")
    if a.dry_run:
        print(f"[dry-run] nothing written (would write {out})")
        return version
    out.mkdir(parents=True, exist_ok=False)
    (out / "labels.coco.json").write_bytes(labels)
    (out / "changes.jsonl").write_text("".join(json.dumps({k: r[k] for k in ROW_FIELDS}, ensure_ascii=False) + "\n"
                                               for r in rows), encoding="utf-8")
    (out / "conflicts.csv").write_text(conflicts_csv(conflicts), encoding="utf-8")
    (out / "version.json").write_text(json.dumps(version, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] wrote {out}")
    return version


if __name__ == "__main__":
    main()
