"""
Agreement between two annotation passes on the same tiles (e.g. a first
annotator vs the reviewer who corrected them, or a human vs model outlines):
which outlines were kept, reshaped, reclassed, deleted or added, per-class F1,
and class agreement (confusion matrix + Cohen's kappa) on the matched pairs.

    python analysis/annotation_agreement.py --pre A.coco.json --post B.coco.json \
        --out <dir> [--pre-label AU --post-label PKD] [--tiles t1,t2]
        [--mirror-map map.csv] [--image-dir DIR] [--title "..."]

Inputs are COCO files (a Roboflow export or roboflow_snapshot.py's
current.coco.json). Tiles are paired by original stem (COCO extra.name, or the
Roboflow export name minus `_<ext>.rf.<hash>`; evaluate_coco_export.original_stem).
Classes are matched BY NAME (snapshot category ids differ). Coordinates are
normalized per image, so the same field at 1440 and 2880 px compares directly;
IoU is computed in the larger of the two frames.

Matching follows evaluate_coco_export / confidence_threshold_sweep: mask IoU,
Hungarian assignment, a pair counts at IoU >= MATCH_IOU (0.5). Matching is
class-agnostic (geometry first), so a reclassed outline is one matched pair
with two classes, not a deletion plus an addition. IoU is rasterized per pair
inside the pair's joint bounding box (roboflow_snapshot.polygon_iou's method),
because whole-frame masks for 300+ outlines at 2880 px do not fit in memory.

Per matched pair:  kept (same class, IoU >= KEPT_IOU 0.98), reshaped (same
class, IoU < 0.98), reclassed (class differs, whatever the shape).
Unmatched pre -> deleted; unmatched post -> added.

Neither pass is ground truth, so the headline is a breakdown of every object
found by EITHER pass (n_union = matched + only_pre + only_post):
  agreed        both passes have it, same class (kept + reshaped)
  class_differs both passes have it, different class
  only_pre      only the first pass has it   (op "deleted")
  only_post     only the second pass has it  (op "added")
each also as a fraction of n_union. Secondary, symmetric summary:
  Dice (= "F1") = 2 * matched / (n_pre + n_post), class-agnostic.
  recall / precision are kept in the CSVs as "% of pre found by post" and
  "% of post found by pre"; they are not accuracy against a truth.
  per-class F1 = 2 * (matched with class c on both sides) / (n_pre_c + n_post_c)
  kappa = Cohen's kappa on the classes of matched pairs.

--mirror-map: CSV with columns tile, blind_name, mirror (v = up-down flip,
h = left-right, hv = both; v4check_selection_PRIVATE.csv). PRE tiles named
blind_name are renamed to `tile` and flipped back before matching.

Writes (refuses an existing --out): per_tile.csv, per_class.csv,
confusion.csv, pairs.csv (every outline's fate), REPORT.md, and with
--image-dir overlays/<tile>.jpg (green = both passes, same class; magenta = both,
class differs; thick blue = only pre; thick orange = only post). With --crops as well:
crops/<category>_p<N>.jpg contact sheets, one cell per disagreement (pre's outlines left, post's right; only in
pre, only in post, class differs), sorted by class, plus crops/index.csv
(sheet, cell, tile, classes, IoU, centre and size in px, and nm with
--nm-per-px, given for the image files' resolution).
"""
import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).parent))
from evaluate_coco_export import original_stem  # noqa: E402
from confidence_threshold_sweep import MATCH_IOU  # noqa: E402

KEPT_IOU = 0.98
SKIP_CATS = {"items-nfaR"}          # Roboflow's workspace supercategory, never an outline
NONE = "(none)"


# ---------------------------------------------------------------- loading
def load_coco(path, tiles=None):
    """{stem: {"w", "h", "objs": [(class_name, Nx2 normalized polygon)]}}."""
    coco = json.loads(Path(path).read_text(encoding="utf-8"))
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    out = {}
    for img in coco["images"]:
        stem = original_stem(img)
        if tiles and stem not in tiles:
            continue
        out[stem] = {"w": img["width"], "h": img["height"], "objs": [], "file_name": img["file_name"]}
    by_id = {img["id"]: original_stem(img) for img in coco["images"]}
    for a in coco["annotations"]:
        stem = by_id.get(a["image_id"])
        if stem not in out:
            continue
        name = cats.get(a["category_id"], "?")
        if name in SKIP_CATS:
            continue
        seg = a.get("segmentation")
        if not seg:
            continue
        flat = seg[0] if isinstance(seg[0], list) else seg
        pts = np.asarray(flat, float).reshape(-1, 2)
        if len(pts) < 3:
            continue
        rec = out[stem]
        out[stem]["objs"].append((name, pts / (rec["w"], rec["h"])))
    return out


def read_mirror_map(path):
    """{blind_stem: (tile_stem, flip)} from a selection CSV."""
    m = {}
    with open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r.get("blind_name") and r.get("tile") and r.get("mirror") is not None:
                m[Path(r["blind_name"]).stem] = (r["tile"], r["mirror"].strip())
    return m


def flip(pts, how):
    p = pts.copy()
    if "h" in how:
        p[:, 0] = 1.0 - p[:, 0]
    if "v" in how:
        p[:, 1] = 1.0 - p[:, 1]
    return p


def apply_mirror(data, mmap):
    out = {}
    for stem, rec in data.items():
        if stem in mmap:
            tile, how = mmap[stem]
            rec = dict(rec, objs=[(c, flip(p, how)) for c, p in rec["objs"]], renamed_from=stem)
            stem = tile
        out[stem] = rec
    return out


# ---------------------------------------------------------------- geometry
def poly_iou(p, q):
    """IoU of two pixel polygons, rasterized inside their joint bounding box."""
    pts = np.vstack([p, q])
    x0, y0 = np.floor(pts.min(0)).astype(int)
    x1, y1 = np.ceil(pts.max(0)).astype(int) + 1
    a = np.zeros((y1 - y0, x1 - x0), np.uint8)
    b = np.zeros_like(a)
    cv2.fillPoly(a, [np.round(p - (x0, y0)).astype(np.int32)], 1)
    cv2.fillPoly(b, [np.round(q - (x0, y0)).astype(np.int32)], 1)
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    return inter / union if union else 0.0


def iou_matrix(pre, post, w, h):
    P = [p * (w, h) for _, p in pre]
    Q = [q * (w, h) for _, q in post]
    m = np.zeros((len(P), len(Q)))
    if not P or not Q:
        return m
    bp = np.array([[*p.min(0), *p.max(0)] for p in P])
    bq = np.array([[*q.min(0), *q.max(0)] for q in Q])
    over = ((bp[:, None, 0] < bq[None, :, 2]) & (bq[None, :, 0] < bp[:, None, 2]) &
            (bp[:, None, 1] < bq[None, :, 3]) & (bq[None, :, 1] < bp[:, None, 3]))
    for i, j in zip(*np.nonzero(over)):
        m[i, j] = poly_iou(P[i], Q[j])
    return m


# ---------------------------------------------------------------- comparison
def compare_tile(pre, post, w, h):
    """Every outline's fate on one tile: list of dicts (pre_idx, post_idx, pre_cls,
    post_cls, iou, op)."""
    m = iou_matrix(pre, post, w, h)
    rows, used_pre, used_post = [], set(), set()
    if m.size:
        pi, qi = linear_sum_assignment(-m)
        for i, j in zip(pi, qi):
            if m[i, j] < MATCH_IOU:
                continue
            a, b = pre[i][0], post[j][0]
            op = "reclassed" if a != b else ("kept" if m[i, j] >= KEPT_IOU else "reshaped")
            rows.append({"pre_idx": int(i), "post_idx": int(j), "pre_cls": a, "post_cls": b,
                         "iou": round(float(m[i, j]), 4), "op": op})
            used_pre.add(i)
            used_post.add(j)
    for i, (c, _) in enumerate(pre):
        if i not in used_pre:
            rows.append({"pre_idx": i, "post_idx": "", "pre_cls": c, "post_cls": NONE, "iou": "", "op": "deleted"})
    for j, (c, _) in enumerate(post):
        if j not in used_post:
            rows.append({"pre_idx": "", "post_idx": j, "pre_cls": NONE, "post_cls": c, "iou": "", "op": "added"})
    return rows


def cohen_kappa(pairs):
    """Cohen's kappa for (a, b) label pairs; 1.0 when both raters use one identical
    label throughout (no disagreement possible), None when there are no pairs."""
    n = len(pairs)
    if not n:
        return None
    po = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[k] * cb.get(k, 0) for k in ca) / (n * n)
    if pe >= 1.0:
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1 - pe)


def summarize(rows):
    n_pre = sum(r["op"] != "added" for r in rows)
    n_post = sum(r["op"] != "deleted" for r in rows)
    matched = [r for r in rows if r["op"] in ("kept", "reshaped", "reclassed")]
    ops = Counter(r["op"] for r in rows)
    k = len(matched)
    ious = [r["iou"] for r in matched]
    kap = cohen_kappa([(r["pre_cls"], r["post_cls"]) for r in matched])
    return {"n_pre": n_pre, "n_post": n_post, "matched": k,
            "recall": round(k / n_pre, 4) if n_pre else None,
            "precision": round(k / n_post, 4) if n_post else None,
            "f1": round(2 * k / (n_pre + n_post), 4) if n_pre + n_post else None,
            "mean_iou": round(float(np.mean(ious)), 4) if ious else None,
            "kept": ops["kept"], "reshaped": ops["reshaped"], "reclassed": ops["reclassed"],
            "deleted": ops["deleted"], "added": ops["added"],
            "class_agree": round(sum(r["pre_cls"] == r["post_cls"] for r in matched) / k, 4) if k else None,
            "kappa": round(kap, 4) if kap is not None else None,
            **breakdown(rows)}


def breakdown(rows):
    """Every object found by either pass, split by what the two passes say about it."""
    b = {"agreed": sum(r["op"] in ("kept", "reshaped") for r in rows),
         "class_differs": sum(r["op"] == "reclassed" for r in rows),
         "only_pre": sum(r["op"] == "deleted" for r in rows),
         "only_post": sum(r["op"] == "added" for r in rows)}
    n = sum(b.values())
    out = {"n_union": n, **b}
    for k in list(b):
        out[f"pct_{k}"] = round(b[k] / n, 4) if n else None
    return out


def per_class(rows):
    classes = sorted(({r["pre_cls"] for r in rows} | {r["post_cls"] for r in rows}) - {NONE})
    out = []
    for c in classes:
        n_pre = sum(r["pre_cls"] == c for r in rows)
        n_post = sum(r["post_cls"] == c for r in rows)
        both = sum(r["pre_cls"] == c and r["post_cls"] == c for r in rows)
        out.append({"class": c, "n_pre": n_pre, "n_post": n_post, "same_class_matched": both,
                    "to_other_class": sum(r["pre_cls"] == c and r["post_cls"] not in (c, NONE) for r in rows),
                    "deleted": sum(r["pre_cls"] == c and r["op"] == "deleted" for r in rows),
                    "added": sum(r["post_cls"] == c and r["op"] == "added" for r in rows),
                    "f1": round(2 * both / (n_pre + n_post), 4) if n_pre + n_post else None,
                    # objects that EITHER pass calls c
                    "union": sum(c in (r["pre_cls"], r["post_cls"]) for r in rows),
                    "agreed": both,
                    "class_differs": sum(r["op"] == "reclassed" and c in (r["pre_cls"], r["post_cls"]) for r in rows),
                    "only_pre": sum(r["pre_cls"] == c and r["op"] == "deleted" for r in rows),
                    "only_post": sum(r["post_cls"] == c and r["op"] == "added" for r in rows)})
        u = out[-1]["union"]
        out[-1]["pct_agreed"] = round(both / u, 4) if u else None
    return out


def confusion(rows):
    classes = sorted(({r["pre_cls"] for r in rows} | {r["post_cls"] for r in rows}) - {NONE}) + [NONE]
    cm = Counter((r["pre_cls"], r["post_cls"]) for r in rows)
    return classes, [[cm[(a, b)] for b in classes] for a in classes]


def compare(pre_data, post_data, tiles=None):
    """{tile: rows} for tiles present in both; also the stems missing on either side."""
    common = sorted(set(pre_data) & set(post_data))
    if tiles:
        common = [t for t in common if t in tiles]
    res = {}
    for t in common:
        a, b = pre_data[t], post_data[t]
        w, h = max(a["w"], b["w"]), max(a["h"], b["h"])
        res[t] = compare_tile(a["objs"], b["objs"], w, h)
    only_pre = sorted(set(pre_data) - set(post_data))
    only_post = sorted(set(post_data) - set(pre_data))
    return res, only_pre, only_post


# ---------------------------------------------------------------- outputs
PRE_BGR, POST_BGR = (214, 118, 31), (14, 127, 255)     # blue, orange
AGREE_BGR, RECLASS_BGR = (90, 190, 90), (190, 60, 190)  # green, magenta (overlays)


def overlay(img_path, pre, post, rows, out_path, pre_label, post_label, max_w=1800):
    im = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if im is None:
        return False
    h, w = im.shape[:2]
    s = min(1.0, max_w / w)
    if s < 1:
        im = cv2.resize(im, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    H, W = im.shape[:2]
    def poly(p, col, th):
        cv2.polylines(im, [np.round(p * (W, H)).astype(np.int32)], True, col, th, cv2.LINE_AA)
    for r in rows:                       # shared first, differences on top
        if r["op"] in ("kept", "reshaped"):
            poly(post[r["post_idx"]][1], AGREE_BGR, 1)
    for r in rows:
        if r["op"] == "reclassed":
            poly(post[r["post_idx"]][1], RECLASS_BGR, 2)
        elif r["op"] == "deleted":
            poly(pre[r["pre_idx"]][1], PRE_BGR, 3)
        elif r["op"] == "added":
            poly(post[r["post_idx"]][1], POST_BGR, 3)
    bar = np.full((44, W, 3), 255, np.uint8)
    txt = [("both, same class", AGREE_BGR), ("both, class differs", RECLASS_BGR),
           (f"only {pre_label}", PRE_BGR), (f"only {post_label}", POST_BGR)]
    x = 10
    for t, col in txt:
        cv2.rectangle(bar, (x, 14), (x + 24, 30), col, -1)
        cv2.putText(bar, t, (x + 32, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 1, cv2.LINE_AA)
        x += 40 + cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0][0] + 30
    cv2.imwrite(str(out_path), np.vstack([bar, im]), [cv2.IMWRITE_JPEG_QUALITY, 88])
    return True


CROP_PANEL, CROP_GAP, CROP_CAP, CROP_COLS, CROP_ROWS = 200, 6, 38, 3, 8
CROP_CELL_W = 2 * CROP_PANEL + 2 * CROP_GAP
CROP_ORDER = ("only_pre", "only_post", "class_differs")


def crop_category(op):
    return {"deleted": "only_pre", "added": "only_post", "reclassed": "class_differs"}.get(op)


def find_image(image_dir, tile):
    return next(iter(sorted(image_dir.glob(f"*__{tile}__*"))), None) or \
        next(iter(sorted(image_dir.glob(f"{tile}*"))), None)


def _put(img, text, org, scale, col, th=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, col, th, cv2.LINE_AA)


def crop_cell(im, pre, post, row, caption, sub, pre_label="pre", post_label="post"):
    """One disagreement as two side-by-side panels of the same crop: the first pass's
    outlines on the left (blue), the second pass's on the right (orange), so outlines
    the passes share never hide each other. The object in question is thick and its
    neighbours thin; caption strip underneath."""
    H, W = im.shape[:2]
    subj = []
    if row["pre_idx"] != "":
        subj.append(("pre", row["pre_idx"]))
    if row["post_idx"] != "":
        subj.append(("post", row["post_idx"]))
    polys = [(pre if s == "pre" else post)[i][1] * (W, H) for s, i in subj]
    pts = np.vstack(polys)
    (x0, y0), (x1, y1) = pts.min(0), pts.max(0)
    side = max(x1 - x0, y1 - y0)
    half = max(side * 0.5 + max(0.6 * side, 24), 40)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    X0, Y0 = int(max(0, cx - half)), int(max(0, cy - half))
    X1, Y1 = int(min(W, cx + half)), int(min(H, cy + half))
    crop = im[Y0:Y1, X0:X1]
    s = CROP_PANEL / max(crop.shape[:2])
    crop = cv2.resize(crop, (max(1, round(crop.shape[1] * s)), max(1, round(crop.shape[0] * s))),
                      interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
    oy, ox = (CROP_PANEL - crop.shape[0]) // 2, (CROP_PANEL - crop.shape[1]) // 2
    box = np.array([X0, Y0, X1, Y1])
    panels = []
    for objs, col, side_name, label in ((pre, PRE_BGR, "pre", pre_label), (post, POST_BGR, "post", post_label)):
        pan = np.full((CROP_PANEL, CROP_PANEL, 3), 255, np.uint8)
        pan[oy:oy + crop.shape[0], ox:ox + crop.shape[1]] = crop
        for i, (_, p) in enumerate(objs):
            pp = p * (W, H)
            if pp[:, 0].max() < box[0] or pp[:, 0].min() > box[2] or pp[:, 1].max() < box[1] or pp[:, 1].min() > box[3]:
                continue
            q = np.round((pp - (X0, Y0)) * s + (ox, oy)).astype(np.int32)
            cv2.polylines(pan, [q], True, col, 3 if (side_name, i) in subj else 1, cv2.LINE_AA)
        tw = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0][0]
        cv2.rectangle(pan, (0, 0), (tw + 8, 16), col, -1)
        _put(pan, label, (4, 12), 0.42, (255, 255, 255))
        panels.append(pan)
    gap = np.full((CROP_PANEL, CROP_GAP, 3), 255, np.uint8)
    cell = np.full((CROP_PANEL + CROP_CAP, CROP_CELL_W, 3), 255, np.uint8)
    cell[:CROP_PANEL, :2 * CROP_PANEL + CROP_GAP] = np.hstack([panels[0], gap, panels[1]])
    _put(cell, caption, (4, CROP_PANEL + 15), 0.42, (30, 30, 30))
    _put(cell, sub, (4, CROP_PANEL + 32), 0.36, (110, 110, 110))
    cell[:, -CROP_GAP:] = 255
    return cell, (X0, Y0, X1, Y1), side


def write_crops(res, pre, post, image_dir, out_dir, pre_label, post_label, nm_per_px=None,
                categories=CROP_ORDER):
    """Contact sheets of every disagreement, by category, sorted by class."""
    out_dir.mkdir()
    items = []
    for t, rows in res.items():
        for r in rows:
            cat = crop_category(r["op"])
            if cat in categories:
                items.append((cat, r["pre_cls"], r["post_cls"], t, r))
    order = {c: k for k, c in enumerate(CROP_ORDER)}
    items.sort(key=lambda x: (order[x[0]], x[1], x[2], x[3], str(x[4]["pre_idx"]), str(x[4]["post_idx"])))
    names = {"only_pre": f"only {pre_label}", "only_post": f"only {post_label}", "class_differs": "class differs"}
    images, index, per_page = {}, [], CROP_COLS * CROP_ROWS
    for cat in CROP_ORDER:
        cat_items = [x for x in items if x[0] == cat]
        for page in range(0, len(cat_items), per_page):
            chunk = cat_items[page:page + per_page]
            sheet_name = f"{cat}_p{page // per_page + 1}.jpg"
            cells = []
            for k, (_, a, b, t, r) in enumerate(chunk):
                if t not in images:
                    p = find_image(image_dir, t)
                    images[t] = cv2.imread(str(p), cv2.IMREAD_COLOR) if p else None
                im = images[t]
                if im is None:
                    continue
                n = page + k + 1
                cap = (f"{pre_label}: {a}  >  {post_label}: {b}" if cat == "class_differs"
                       else f"{names[cat]}: {a if cat == 'only_pre' else b}")
                cell, (X0, Y0, X1, Y1), side = crop_cell(im, pre[t]["objs"], post[t]["objs"], r, cap[:58],
                                                         f"#{n} {t}"[:60], pre_label, post_label)
                cells.append(cell)
                H, W = im.shape[:2]
                idx = int(r["pre_idx"] if cat != "only_post" else r["post_idx"])
                poly = (pre if cat != "only_post" else post)[t]["objs"][idx][1] * (W, H)
                area = cv2.contourArea(poly.astype(np.float32))
                rec = {"sheet": sheet_name, "n": n, "row": k // CROP_COLS + 1, "col": k % CROP_COLS + 1,
                       "category": cat, "tile": t, "pre_cls": a, "post_cls": b, "iou": r["iou"],
                       "pre_idx": r["pre_idx"], "post_idx": r["post_idx"],
                       "cx_px": round(float(poly[:, 0].mean()), 1), "cy_px": round(float(poly[:, 1].mean()), 1),
                       "diam_px": round(2 * float(np.sqrt(area / np.pi)), 1)}
                if nm_per_px:
                    rec["diam_nm"] = round(rec["diam_px"] * nm_per_px, 1)
                index.append(rec)
            if not cells:
                continue
            while len(cells) % CROP_COLS:
                cells.append(np.full_like(cells[0], 255))
            grid = np.vstack([np.hstack(cells[i:i + CROP_COLS]) for i in range(0, len(cells), CROP_COLS)])
            head = np.full((52, grid.shape[1], 3), 255, np.uint8)
            _put(head, f"{names[cat]}  ({page + 1}-{page + len(chunk)} of {len(cat_items)})", (10, 22), 0.65,
                 (20, 20, 20), 2)
            x = 10
            for t_, col in ((f"left: {pre_label}'s outlines", PRE_BGR), (f"right: {post_label}'s outlines", POST_BGR),
                            ("thick = the object in question, thin = neighbours", (120, 120, 120))):
                cv2.rectangle(head, (x, 32), (x + 18, 44), col, -1)
                _put(head, t_, (x + 24, 44), 0.45, (60, 60, 60))
                x += 40 + cv2.getTextSize(t_, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0]
            cv2.imwrite(str(out_dir / sheet_name), np.vstack([head, grid]), [cv2.IMWRITE_JPEG_QUALITY, 90])
    if index:
        write_csv(out_dir / "index.csv", index)
    return index


def write_csv(path, rows, fields=None):
    rows = list(rows)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields or list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)


def md_table(rows, cols):
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    return head + "".join("| " + " | ".join("" if r.get(c) is None else str(r.get(c)) for c in cols) + " |\n"
                          for r in rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pre", required=True, type=Path)
    ap.add_argument("--post", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--pre-label", default="pre")
    ap.add_argument("--post-label", default="post")
    ap.add_argument("--tiles", default=None, type=lambda s: {t.strip() for t in s.split(",") if t.strip()})
    ap.add_argument("--mirror-map", default=None, type=Path)
    ap.add_argument("--image-dir", default=None, type=Path)
    ap.add_argument("--title", default="Annotation agreement")
    ap.add_argument("--crops", action="store_true", help="contact sheets of every disagreement (needs --image-dir)")
    ap.add_argument("--crop-categories", default=",".join(CROP_ORDER),
                    help="subset of only_pre,only_post,class_differs (e.g. skip class_differs vs a 1-class model)")
    ap.add_argument("--nm-per-px", type=float, default=None, help="for diam_nm in crops/index.csv (image-file px)")
    args = ap.parse_args()
    if args.crops and not args.image_dir:
        sys.exit("--crops needs --image-dir")
    if args.out.exists():
        sys.exit(f"{args.out} exists; reports are never overwritten")

    pre = load_coco(args.pre)
    if args.mirror_map:
        pre = apply_mirror(pre, read_mirror_map(args.mirror_map))
    post = load_coco(args.post)
    res, only_pre, only_post = compare(pre, post, args.tiles)
    if not res:
        sys.exit("no tiles in common")

    args.out.mkdir(parents=True)
    all_rows, tile_rows = [], []
    for t, rows in res.items():
        s = summarize(rows)
        tile_rows.append({"tile": t, **s})
        all_rows += [{"tile": t, **r} for r in rows]
    total = {"tile": "ALL", **summarize(all_rows)}
    write_csv(args.out / "per_tile.csv", tile_rows + [total])
    pc = per_class(all_rows)
    write_csv(args.out / "per_class.csv", pc)
    write_csv(args.out / "pairs.csv", all_rows)
    classes, cm = confusion(all_rows)
    with open(args.out / "confusion.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow([f"{args.pre_label} \\ {args.post_label}"] + classes)
        for c, row in zip(classes, cm):
            wr.writerow([c] + row)

    if args.image_dir:
        (args.out / "overlays").mkdir()
        for t, rows in res.items():
            img = find_image(args.image_dir, t)
            if img:
                overlay(img, pre[t]["objs"], post[t]["objs"], rows, args.out / "overlays" / f"{t}.jpg",
                        args.pre_label, args.post_label)

    n_crops = None
    if args.crops:
        n_crops = len(write_crops(res, pre, post, args.image_dir, args.out / "crops",
                                  args.pre_label, args.post_label, args.nm_per_px,
                                  tuple(c.strip() for c in args.crop_categories.split(","))))

    A, B = args.pre_label, args.post_label
    bcols = ["tile", "n_pre", "n_post", "n_union", "agreed", "class_differs", "only_pre", "only_post",
             "pct_agreed", "f1", "mean_iou", "kappa"]
    cols = ["tile", "n_pre", "n_post", "matched", "f1", "recall", "precision", "mean_iou",
            "kept", "reshaped", "reclassed", "deleted", "added", "class_agree", "kappa"]

    def pct(v):
        return "" if v is None else f"{100 * v:.1f}%"
    md = [f"# {args.title}\n",
          f"- **pre ({A}):** `{args.pre}`",
          f"- **post ({B}):** `{args.post}`"]
    if args.mirror_map:
        md.append(f"- **Mirror map:** `{args.mirror_map.name}` (pre tiles flipped back to the post frame)")
    if only_pre or only_post:
        md.append(f"- **Not compared:** only in pre {only_pre or '-'}; only in post {only_post or '-'}")
    md += ["", "## Agreement breakdown",
           f"Neither pass is treated as truth. Every object found by either pass ({total['n_union']}) "
           "falls in exactly one row:", "",
           "| | objects | share |", "|---|--:|--:|",
           f"| Both, same class | {total['agreed']} | {pct(total['pct_agreed'])} |",
           f"| Both, class differs | {total['class_differs']} | {pct(total['pct_class_differs'])} |",
           f"| Only {A} | {total['only_pre']} | {pct(total['pct_only_pre'])} |",
           f"| Only {B} | {total['only_post']} | {pct(total['pct_only_post'])} |", "",
           f"- Of {A}'s {total['n_pre']} objects, {B} also has {pct(total['recall'])}; "
           f"of {B}'s {total['n_post']}, {A} also has {pct(total['precision'])}.",
           f"- Dice (the f1 column) = 2 x both / ({A} + {B}) = {total['f1']}: one symmetric number for "
           "\"found the same objects\", ignoring class. Class agreement on shared objects: "
           f"kappa {total['kappa']}. Shape agreement on shared objects: mean IoU {total['mean_iou']} "
           f"({total['reshaped']} same-class pairs below IoU {KEPT_IOU}).",
           f"- Matching: class-agnostic Hungarian on mask IoU >= {MATCH_IOU}.", ""]
    if n_crops is not None:
        md += [f"- Crops ({args.crop_categories}): {n_crops} disagreements in `crops/` (sheets by category, sorted by class; "
               "`crops/index.csv` maps each sheet cell to tile, classes, position, size).", ""]
    bpc = [{"class": r["class"], "either pass": r["union"], "both, same class": r["agreed"],
            "both, class differs": r["class_differs"], f"only {A}": r["only_pre"], f"only {B}": r["only_post"],
            "% agreed": pct(r["pct_agreed"])} for r in pc]
    md += ["## By class (objects either pass calls this class)",
           "An object reclassed from X to Y counts under both X and Y.", "", md_table(bpc, list(bpc[0])),
           "## Per tile", md_table(tile_rows + [total], bcols),
           f"## Confusion (rows {A}, columns {B})",
           md_table([{"": c, **dict(zip(classes, r))} for c, r in zip(classes, cm)], [""] + classes),
           "## Detail (edit-style terms: pre = first pass)",
           f"kept = same class and IoU >= {KEPT_IOU}; deleted = only in pre; added = only in post.", "",
           md_table(tile_rows + [total], cols), md_table(pc, list(pc[0]))]
    (args.out / "REPORT.md").write_text("\n".join(md), encoding="utf-8")
    print(md_table([total], bcols))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
