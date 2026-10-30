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
  duplicate_outline      outlines linked at mask IoU >= --dup-iou (0.9, full resolution) with different
                         classes: every change in the group; or a change that makes two labels the same
  g_missing_from_base    a g<id> decision or list item whose id is not a base annotation of that image
  pred_index_out_of_range  p<k> with k not in 0..len(pr)-1 (p-1 too, as in the page)
  invalid_polygon        an outline that, in pixels and clamped to the image, has a ring with < 3 distinct
                         points or zero area
  image_not_in_package   a record whose file has no data.js image
  duplicate_id           the same base id twice in one image's labels
duplicate-dropped: a same-class copy of an outline that stays (a label with an id, else the earliest add/new by
t, then key order); in conflicts.csv with severity "dropped". Base pairs that already overlap are not flagged
(version.json base_duplicates). An add/new over another class's label at IoU >= 0.5 is applied with a warning
(severity "warn"). Outlines are compared numerically (1e-9), so 0 and 0.0 are the same outline.

Not logged (no label change, no review verdict): null (undone) decisions, n<uid>/p<k> "del" of an
outline that was drawn/accepted and then removed again. Counted in version.json "skipped".

Refused before anything is written: a package whose labels differ from the base (per image the same ids, per
label the class and an outline within 0.5 px), a data.js source that is neither --base nor a byte-identical
copy, duplicate image names, a record for another package.

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
from review_server import _migrate, _num, effective_labels, load_people  # noqa: E402
from data_utils.path_guard import PROTECTED_ROOTS, normalize, protected_problem  # noqa: E402

SCHEMA = "label_change/1"
DUP_IOU = 0.9
NEAR_IOU = 0.5  # an add/new over a label of another class at this IoU (below DUP_IOU) is applied with a warning
PEOPLE = HERE / "people.json"
CLASS_GROUPS = HERE / "class_groups.yaml"
INTERNAL = ("u", "x", "lc", "seed", "src", "t")  # data.js / tool fields, never written to COCO
CHANGE_OPS = ("reclass", "reshape", "reclass+reshape", "add", "new")
OPS = ("keep", "reclass", "reshape", "reclass+reshape", "delete", "add", "new", "reject")
ROW_FIELDS = ["schema", "label_version", "base", "pkg", "image", "rf_id", "dec_key", "op", "ann_old", "ann_new",
              "cls_from", "cls_to", "who_raw", "who", "who_auth", "t", "rx", "seed", "reviewed", "status", "reason"]
CONFLICT_FIELDS = ["severity", "image", "rf_id", "file", "dec_key", "op", "reason", "ann_old", "cls_from", "cls_to",
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
    """Why --out-root may not be used, or None (data_utils/path_guard.py, shared with inference_check)."""
    problem = protected_problem(root, protected)
    return f"--out-root {problem}; write label versions elsewhere" if problem else None


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

SAME_TOL = 1e-9  # normalized units: equal outlines up to float noise (0 vs 0.0 is equal)


def _same(a, b, tol: float = SAME_TOL) -> bool:
    """Two ring lists are the same outline: same ring and vertex counts, every coordinate within tol. Numeric, so
    0 == 0.0 (a json.dumps comparison calls those different and logs an edge-touching keep as a reshape)."""
    try:
        return (len(a) == len(b) and all(len(ra) == len(rb) for ra, rb in zip(a, b))
                and all(abs(float(x) - float(y)) <= tol for ra, rb in zip(a, b) for x, y in zip(ra, rb)))
    except (TypeError, ValueError):
        return False


def _pred_index(key: str, im: dict):
    """p<k> -> k if it names one of the image's predictions, else None. Like the page (im.pr[k] is undefined for
    k < 0), "p-1" is out of range, although Python's pr[-1] would pick the last prediction."""
    if not re.fullmatch(r"p\d+", key):
        return None
    k = int(key[1:])
    return k if k < len(im.get("pr") or []) else None


def _union_find(n, edges):
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i, j in edges:
        parent[find(i)] = find(j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return sorted(groups.values())


def base_duplicate_pairs(base_anns: dict, dup_iou: float = DUP_IOU, cache=None) -> dict:
    """{(id_a, id_b): IoU} for base labels of one image that already overlap at >= dup_iou."""
    ids = sorted(base_anns)
    out = {}
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            v = _iou_cached(base_anns[a]["segmentation"], base_anns[b]["segmentation"], cache)
            if v >= dup_iou:
                out[(a, b)] = round(v, 3)
    return out


def _iou_cached(a, b, cache):
    if cache is None:
        return mask_iou(a, b)
    k = (id(a), id(b)) if id(a) < id(b) else (id(b), id(a))
    if k not in cache:
        cache[k] = mask_iou(a, b)
    return cache[k]


def image_ops(im: dict, rec: dict, base_anns: dict, w: int, h: int, dup_iou: float = DUP_IOU, base_dups=(), cache=None):
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
        if key in present_src:  # its object came from a list item (logged as add/new), whatever the decision says
            continue
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
        cur = {aid: (None, f"g{aid}", a["category_name"], a["segmentation"], aid) for aid, a in base_anns.items()}
        out = []
        for o in ops:
            if o["conflict"] or o.get("dropped"):
                continue
            if o["op"] == "delete":
                cur.pop(o["ann_old"], None)
            elif o["op"] in ("reclass",):
                cur[o["ann_old"]] = (o, o["dec_key"], o["cls_to"], base_anns[o["ann_old"]]["segmentation"], o["ann_old"])
            elif o["op"] in ("reshape", "reclass+reshape"):
                cur[o["ann_old"]] = (o, o["dec_key"], o["cls_to"], o["px"], o["ann_old"])
            elif o["op"] in ("add", "new"):
                out.append((o, o["dec_key"], o["cls_to"], o["px"], None))
        return [cur[k] for k in sorted(cur)] + out

    def when(o):
        d, item = o["d"], o["item"] or {}
        t = d.get("t") if d else item.get("t", e.get("gtT"))
        return t if _num(t) else math.inf

    # Duplicate outlines: clusters of final outlines linked at IoU >= dup_iou (pairs that already overlapped in the
    # base are not links). Repeated until stable, since a reverted change brings its base outline back.
    #   classes differ in a cluster         -> every change in it is a conflict
    #   same class, one label with an id    -> it stays; each add/new is dropped (status duplicate-dropped)
    #   same class, only adds/new           -> the earliest (t, then key order) stays; the rest are dropped
    #   same class, two labels with ids     -> the changed ones are conflicts (a label is never dropped)
    order = {id(o): k for k, o in enumerate(ops)}
    while True:
        cands = candidates()
        edges, best = [], {}
        for i in range(len(cands)):
            for j in range(i + 1, len(cands)):
                a, b = cands[i], cands[j]
                if a[0] is None and b[0] is None:
                    continue  # two untouched base labels: not an edit
                if a[4] is not None and b[4] is not None and (min(a[4], b[4]), max(a[4], b[4])) in base_dups:
                    continue  # already overlapped in the base: reported in version.json, not an edit's doing
                v = _iou_cached(a[3], b[3], cache)
                if v >= dup_iou:
                    edges.append((i, j))
                    for x, y in ((i, j), (j, i)):
                        if v > best.get(x, (0,))[0]:
                            best[x] = (v, y)
        acted = False
        for group in _union_find(len(cands), edges):
            members = [cands[k] for k in group]
            if len(group) < 2 or all(m[0] is None for m in members):
                continue
            def why(k, kind):
                v, other = best[k]
                return f"duplicate_outline: IoU {v:.3f} with {cands[other][1]} ({cands[other][2]}), {kind}"
            if len({m[2] for m in members}) > 1:
                for k in group:
                    if cands[k][0] is not None:
                        cands[k][0]["conflict"] = why(k, "different class")
                        acted = True
                continue
            labeled = [k for k in group if cands[k][4] is not None]
            added = sorted((k for k in group if cands[k][4] is None), key=lambda k: (when(cands[k][0]), order[id(cands[k][0])]))
            if len(labeled) >= 2:
                for k in labeled:
                    if cands[k][0] is not None:
                        cands[k][0]["conflict"] = why(k, "same class, two labels")
                        acted = True
            keep = labeled[0] if len(labeled) == 1 else None if labeled else added[0]
            for k in added:
                if k != keep:
                    cands[k][0]["dropped"] = why(k, "same class") + f"; kept {cands[keep][1] if keep is not None else 'the labels'}"
                    acted = True
        if not acted:
            break

    # Near duplicates: an add/new over an existing label of another class at IoU >= NEAR_IOU is applied, with a warning
    final_cands = candidates()
    for c in final_cands:
        o = c[0]
        if o is None or o["op"] not in ("add", "new"):
            continue
        for x in final_cands:
            if x[4] is not None and x[2] != c[2]:
                v = _iou_cached(c[3], x[3], cache)
                if v >= NEAR_IOU:
                    o["warn"] = f"near_duplicate: IoU {v:.3f} with {x[1]} ({x[2]}), different class; applied"
                    break
    return ops, skipped


# ---------------------------------------------------------------- run

DRIFT_PX = 0.5  # a package outline further than this from its base annotation means another base


def _rings(a):
    """A COCO annotation's polygon rings, as build_review_package.gt_items keeps them (>= 3 points)."""
    return [s for s in a.get("segmentation") or [] if isinstance(s, list) and len(s) >= 6]


def outline_offset_px(p, rings, w, h) -> float:
    """Largest vertex offset (px) between a data.js outline p (normalized) and base rings (px); inf if the ring or
    vertex counts differ."""
    if len(p) != len(rings) or any(len(a) != len(b) for a, b in zip(p, rings)):
        return math.inf
    return max((abs(float(x) * (w if k % 2 == 0 else h) - float(y)) for a, b in zip(p, rings)
                for k, (x, y) in enumerate(zip(a, b))), default=0.0)


def check_source(source, base_path) -> str:
    """data.js "source" vs --base: the same file, or a byte-identical copy of it; else refuse."""
    if not source:
        print("[WARN] data.js has no source: cannot check it names --base (the per-label check still runs)")
        return "absent"
    if normalize(source) == normalize(base_path):
        return "same path"
    if Path(source).is_file() and sha256_file(source) == sha256_file(base_path):
        return "identical copy"
    raise SystemExit(f"[ERROR] data.js was built from {source}, not --base {base_path} (and the files differ)")

def apply(base_path: Path, package: Path, edits_dir: Path, people: list, dup_iou: float = DUP_IOU):
    """Everything but writing: -> (coco_out, rows (without label_version), conflicts, info)."""
    base = json.loads(Path(base_path).read_text(encoding="utf-8"))
    data = read_data_js(Path(package) / "data.js")
    who_of = roster_lookup(people)
    cat_id = {c["name"]: c["id"] for c in base["categories"]}
    cat_name = {c["id"]: c["name"] for c in base["categories"]}
    for what, names in (("base file_name", [i["file_name"] for i in base["images"]]),
                        ("base image stem", [(i.get("extra") or {}).get("name") or stem_of(i["file_name"]) for i in base["images"]]),
                        ("package file", [im["file"] for im in data["images"]])):
        dups = sorted(n for n, k in Counter(names).items() if k > 1)
        if dups:
            raise SystemExit(f"[ERROR] duplicate {what}: {dups[:5]}{' ...' if len(dups) > 5 else ''}; images must be unique")
    images_by_file = {i["file_name"]: i for i in base["images"]}
    anns_by_img = {}
    for a in base["annotations"]:
        anns_by_img.setdefault(a["image_id"], {})[a["id"]] = {**a, "category_name": cat_name.get(a["category_id"])}
    pkg_images = {im["file"]: im for im in data["images"]}
    base_name, pkg_id = Path(base_path).parent.name, data.get("id", "")
    source_check = check_source(data.get("source"), base_path)

    # The package must come from this base: per image the same label ids, and per label the same class and outline
    # (data.js p = base segmentation / (w, h), rounded; matched within DRIFT_PX)
    mismatched = []
    for im in data["images"]:
        b = images_by_file.get(im["file"])
        if b is None:
            mismatched.append(f"{im['file']}: not in the base")
            continue
        if (b["width"], b["height"]) != (im["w"], im["h"]):
            mismatched.append(f"{im['file']}: package {im['w']}x{im['h']}, base {b['width']}x{b['height']}")
            continue
        mine = {aid: a for aid, a in anns_by_img.get(b["id"], {}).items() if _rings(a)}  # what gt_items packages
        pkg_ids = [g["id"] for g in im["gt"]]
        if sorted(pkg_ids) != sorted(mine):
            extra, missing = sorted(set(pkg_ids) - set(mine)), sorted(set(mine) - set(pkg_ids))
            mismatched.append(f"{im['file']}: label ids differ (package only {extra[:5]}, base only {missing[:5]}"
                              f"{', repeated ids' if len(pkg_ids) != len(set(pkg_ids)) else ''})")
            continue
        for g in im["gt"]:
            a = mine[g["id"]]
            if a["category_name"] != g["c"]:
                mismatched.append(f"{im['file']} g{g['id']}: package {g['c']}, base {a['category_name']}")
            off = outline_offset_px(g["p"], _rings(a), b["width"], b["height"])
            if off > DRIFT_PX:
                mismatched.append(f"{im['file']} g{g['id']}: package outline is {off:.1f} px from the base's")
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
    cache = {}
    base_dups = {i["id"]: base_duplicate_pairs(anns_by_img.get(i["id"], {}), dup_iou, cache) for i in base["images"]}
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
        ops, sk = image_ops(im, rec, anns_by_img.get(b["id"], {}), b["width"], b["height"], dup_iou, base_dups[b["id"]], cache)
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
            if o["conflict"] or o.get("dropped") or o["op"] in ("keep", "reject"):
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
               "ann_new": None if r["conflict"] or r.get("dropped") else (r["_op"].get("ann_new") if r["op"] not in ("keep", "reject") else r["ann_old"]),
               "cls_from": r["cls_from"], "cls_to": r["cls_to"], "who_raw": who_raw, "who": who_of(who_raw),
               "who_auth": None, "t": t, "rx": r["rx"], "seed": seed, "reviewed": r["reviewed"],
               "status": "conflict" if r["conflict"] else "duplicate-dropped" if r.get("dropped") else "applied",
               "reason": r["conflict"] or r.get("dropped") or r.get("warn")}
        out_rows.append(row)
        if row["reason"]:
            sev = {"conflict": "conflict", "duplicate-dropped": "dropped"}.get(row["status"], "warn")
            conflicts.append({"file": r["file"], "severity": sev, **{k: row[k] for k in CONFLICT_FIELDS if k in row}})
    counts = Counter(r["op"] for r in out_rows if r["status"] == "applied")
    info_out = {"n_records": len(records), "n_images_applied": applied_images, "n_records_without_changes": n_unchanged,
                "counts_by_op": {k: counts.get(k, 0) for k in OPS},
                "counts_conflict_by_op": dict(sorted(Counter(r["op"] for r in out_rows if r["status"] == "conflict").items())),
                "counts_dropped_by_op": dict(sorted(Counter(r["op"] for r in out_rows if r["status"] == "duplicate-dropped").items())),
                "n_conflicts": sum(1 for c in conflicts if c["severity"] == "conflict"),
                "n_dropped": sum(1 for c in conflicts if c["severity"] == "dropped"),
                "n_warnings": sum(1 for c in conflicts if c["severity"] == "warn"),
                "skipped": dict(sorted(skipped.items())), "source_check": source_check,
                "base_duplicates": [{"image": (i.get("extra") or {}).get("name") or stem_of(i["file_name"]), "ids": list(k), "iou": v}
                                    for i in base["images"] for k, v in sorted(base_dups[i["id"]].items())],
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
    raw_root = a.out_root or base.parent.parent / "labels"
    problem = out_root_problem(raw_root)  # as typed (UNC, \\?\, trailing spaces), then resolved
    out_root = raw_root.resolve() if not problem else raw_root
    problem = problem or out_root_problem(out_root)
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
               "n_conflicts": info["n_conflicts"], "counts_dropped_by_op": info["counts_dropped_by_op"],
               "n_dropped": info["n_dropped"], "n_warnings": info["n_warnings"], "skipped": info["skipped"],
               "source_check": info["source_check"], "base_duplicates": info["base_duplicates"],
               "new_categories": info["new_categories"], "new_categories_flag": bool(info["new_categories"]),
               "n_annotations_base": info["n_annotations_base"], "n_annotations_out": info["n_annotations_out"],
               "labels_sha256": hashlib.sha256(labels).hexdigest(), "dup_iou": a.dup_iou,
               "class_groups_sha256": sha256_file(a.class_groups), "people_sha256": sha256_file(a.people),
               "tool": f"apply_review_edits.py {git_sha()}", "created": datetime.now().isoformat(timespec="seconds")}
    print(f"[OK] {info['n_records']} records, {info['n_images_applied']} images applied; "
          f"annotations {info['n_annotations_base']} -> {info['n_annotations_out']}")
    print("[OK] applied by op: " + ", ".join(f"{k} {v}" for k, v in info["counts_by_op"].items()))
    print(f"[OK] conflicts: {info['n_conflicts']} {info['counts_conflict_by_op']}; duplicate-dropped {info['n_dropped']} "
          f"{info['counts_dropped_by_op']}; warnings {info['n_warnings']}; skipped {info['skipped']}")
    print(f"[OK] data.js source: {info['source_check']}; base duplicates (not flagged): {len(info['base_duplicates'])}")
    for c in conflicts:
        print(f"  [{c['severity']}] {c['image']} {c['dec_key']} {c['op']}: {c['reason']}")
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
