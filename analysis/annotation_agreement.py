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

  agreement F1 = 2 * matched / (n_pre + n_post); recall = matched / n_pre
  (pre as reference), precision = matched / n_post.
  per-class F1 = 2 * (matched with class c on both sides) / (n_pre_c + n_post_c)
  kappa = Cohen's kappa on the classes of matched pairs.

--mirror-map: CSV with columns tile, blind_name, mirror (v = up-down flip,
h = left-right, hv = both; v4check_selection_PRIVATE.csv). PRE tiles named
blind_name are renamed to `tile` and flipped back before matching.

Writes (refuses an existing --out): per_tile.csv, per_class.csv,
confusion.csv, pairs.csv (every outline's fate), REPORT.md, and with
--image-dir overlays/<tile>.jpg (pre in blue, post in orange; an outline with
no partner drawn thick, a class change medium).
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
            "kappa": round(kap, 4) if kap is not None else None}


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
                    "f1": round(2 * both / (n_pre + n_post), 4) if n_pre + n_post else None})
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


def overlay(img_path, pre, post, rows, out_path, pre_label, post_label, max_w=1800):
    im = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if im is None:
        return False
    h, w = im.shape[:2]
    s = min(1.0, max_w / w)
    if s < 1:
        im = cv2.resize(im, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    H, W = im.shape[:2]
    lost = {r["pre_idx"] for r in rows if r["op"] == "deleted"}
    new = {r["post_idx"] for r in rows if r["op"] == "added"}
    recl = {r["post_idx"] for r in rows if r["op"] == "reclassed"}
    for i, (_, p) in enumerate(pre):
        cv2.polylines(im, [np.round(p * (W, H)).astype(np.int32)], True, PRE_BGR, 4 if i in lost else 1, cv2.LINE_AA)
    for j, (_, q) in enumerate(post):
        th = 4 if j in new else 2 if j in recl else 1
        cv2.polylines(im, [np.round(q * (W, H)).astype(np.int32)], True, POST_BGR, th, cv2.LINE_AA)
    bar = np.full((44, W, 3), 255, np.uint8)
    txt = [(f"{pre_label} (thick = only in this pass)", PRE_BGR),
           (f"{post_label} (thick = only in this pass, medium = class differs)", POST_BGR)]
    x = 10
    for t, col in txt:
        cv2.rectangle(bar, (x, 14), (x + 24, 30), col, -1)
        cv2.putText(bar, t, (x + 32, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 1, cv2.LINE_AA)
        x += 40 + cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0][0] + 30
    cv2.imwrite(str(out_path), np.vstack([bar, im]), [cv2.IMWRITE_JPEG_QUALITY, 88])
    return True


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
    args = ap.parse_args()
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
            img = next(iter(sorted(args.image_dir.glob(f"*__{t}__*"))), None) or \
                next(iter(sorted(args.image_dir.glob(f"{t}*"))), None)
            if img:
                overlay(img, pre[t]["objs"], post[t]["objs"], rows, args.out / "overlays" / f"{t}.jpg",
                        args.pre_label, args.post_label)

    cols = ["tile", "n_pre", "n_post", "matched", "f1", "recall", "precision", "mean_iou",
            "kept", "reshaped", "reclassed", "deleted", "added", "class_agree", "kappa"]
    md = [f"# {args.title}\n",
          f"- **pre ({args.pre_label}):** `{args.pre}`",
          f"- **post ({args.post_label}):** `{args.post}`"]
    if args.mirror_map:
        md.append(f"- **Mirror map:** `{args.mirror_map.name}` (pre tiles flipped back to the post frame)")
    if only_pre or only_post:
        md.append(f"- **Not compared:** only in pre {only_pre or '-'}; only in post {only_post or '-'}")
    md += ["", f"Matching: class-agnostic Hungarian on mask IoU >= {MATCH_IOU}; kept = same class and "
               f"IoU >= {KEPT_IOU}; recall uses pre as the reference.", "",
           "## Per tile", md_table(tile_rows + [total], cols),
           "## Per class", md_table(pc, list(pc[0])),
           f"## Confusion (rows {args.pre_label}, columns {args.post_label})",
           md_table([{"": c, **dict(zip(classes, r))} for c, r in zip(classes, cm)], [""] + classes)]
    (args.out / "REPORT.md").write_text("\n".join(md), encoding="utf-8")
    print(md_table([total], cols))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
