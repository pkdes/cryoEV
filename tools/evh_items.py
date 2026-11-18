"""
eVHarmony items CSV (round 17): the candidates of one prediction set -> an items CSV for
`python -m annotation.object_set build --line-col line --first-items <gold> --app evharmony`.

    python tools/evh_items.py --ps <CryoAI-laptop>/predict/<ps> --out <items.csv> [--lo 0.25] [--hi 0.5]
        [--gold 40] [--seed 17]
        [--low 0.10 [--n-low 20]]                          (r17c: extra likely-Nope candidates)
        [--decoys 20 --derive <dir>]                        (r17c: known-Nope decoys in a derived prediction set)
        [--answered <old pred_obj_*> --who <reviewer id>]   (r17c: that reviewer's answered objects first)

Items: every polygon line of every image's prediction .txt with lo <= conf < hi (any class), read with
annotation/object_set.read_lines (the object key of contract C5: image, 0-based line). Columns: image, line, conf,
gold. The first --gold rows are a confidence-stratified pick (the candidates sorted by conf, cut into --gold equal
bins, one seeded pick per bin), marked gold=1; the rest follow in (image, line) order. Build with --first-items
<gold>, so the gold rows are the first cards for every reviewer; they become known answers once the set's owner
answers them (tools/evh_score.py scores everyone else against those answers). Read-only on the prediction set;
the same inputs and seed give the same bytes.

r17c, balanced gold (all optional; without them the output is byte-identical to before):
  --low L [--n-low N]  up to N (seeded pick) candidates with L <= conf < lo as well (likely Nopes). A set predicted
                       at conf >= lo has none: reported, not an error.
  --decoys N --derive DIR
                       DIR (new) = a derived copy of --ps (images, images.csv, status.json byte-copied; request.json
                       with a new ps id; the decoy note only in evh_derived.json, never read by /api/meta) whose prediction .txt files get N decoy polygon lines APPENDED
                       (every real line keeps its index, so answers carry over by (image, line)). A decoy = a real
                       candidate's polygon (vertex list, size and pixel staircase kept), turned by a random flip /
                       transpose, moved by whole pixels to a random place of one of the set's images that touches no
                       prediction of that image (any conf, 3 px margin) and no other decoy, and fully inside it; conf
                       and class are drawn from the candidates (conf written like the model's, 4 decimals). Build the
                       set with --ps DIR. The decoy list goes ONLY to <out stem>_decoys.csv next to the items CSV
                       (image, line, conf, cls, src_image, src_line, flip): never into the items CSV, items.json or
                       data.js, where a decoy row looks like any candidate. tools/evh_score.py --decoys counts them as
                       known Nopes.
  --answered PKG --who RID
                       the objects RID answered (match, any value) in the old object set PKG go first.
  --answered-attr ATTR (r17f) the attribute that counts as "answered" (default match; e.g. type for an ML/MV set);
                       --who may list several reviewer ids, comma-separated.
r17f (optional; without them the output is unchanged):
  --images-of PKG      only images that hold an item of the object set PKG (e.g. the 49 images of an ML/MV set
                       whose prediction set has more); decoys go on those images too
  --exclude IMG:LINE,...  objects left out (practice cards, How to judge examples shown elsewhere)
  With any of these, the first block (gold=1) = answered + decoys + low (+ the --gold stratified pick from the rest),
  then the rest (gold=0). Build with --first-items <the printed count>.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from annotation import object_set as O  # noqa: E402
from annotation import prediction_package as pp  # noqa: E402

COLUMNS = ["image", "line", "conf", "gold"]
DECOY_COLUMNS = ["image", "line", "conf", "cls", "src_image", "src_line", "flip"]
DERIVED = "evh_derived.json"       # the derived prediction set's provenance (the server never reads it)
MARGIN = 3                         # px kept free between a decoy and any prediction / other decoy
TRIES = 4000


def candidates(ps_dir, lo=0.25, hi=0.5, with_cls=False) -> list:
    """[(image, line, conf)] of every prediction line with lo <= conf < hi, in (image, line) order."""
    _req, _ps, _model, rows, pred_dir = pp._ps_inputs(Path(ps_dir))
    out = []
    for r in sorted(rows, key=lambda r: r["image"]):
        f = pred_dir / f"{r['image']}.txt"
        if not f.is_file():
            continue
        for o in O.read_lines(f):
            if o is not None and lo <= o["sr"] < hi:
                out.append((r["image"], o["line"], o["sr"], o["k"]) if with_cls else (r["image"], o["line"], o["sr"]))
    return out


def stratified(cands: list, n: int, seed: int) -> list:
    """n items, one seeded pick from each of n equal-count bins of the candidates sorted by conf."""
    if n <= 0:
        return []
    if n >= len(cands):
        return list(cands)
    rng = random.Random(seed)
    srt = sorted(cands, key=lambda c: (c[2], c[0], c[1]))
    picks = []
    for k in range(n):
        a, b = k * len(srt) // n, (k + 1) * len(srt) // n
        picks.append(srt[rng.randrange(a, b)])
    return picks


def items_csv(cands: list, gold: int, seed: int, first=()) -> str:
    """first (r17c): rows put first, before the stratified gold pick (which is then drawn from the rest)."""
    first = list(first)
    fs = set(first)
    rest = [c for c in cands if c not in fs]
    g = first + stratified(rest, gold, seed)
    gs = set(g)
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(COLUMNS)
    for c in g:
        w.writerow([c[0], c[1], f"{c[2]:.4f}", 1])
    for c in cands:
        if c not in gs:
            w.writerow([c[0], c[1], f"{c[2]:.4f}", 0])
    return buf.getvalue()


# ---------------------------------------------------------------- r17c: decoys
def _px_rings(ps_dir) -> tuple:
    """({image: (w, h)}, {image: [(line, conf, cls, [(x, y) px])]}) of every prediction line of the set."""
    _req, _ps, _model, rows, pred_dir = pp._ps_inputs(Path(ps_dir))
    size, polys = {}, {}
    for r in sorted(rows, key=lambda r: r["image"]):
        with Image.open(Path(ps_dir) / r["file"]) as im:
            w, h = im.size
        size[r["image"]] = (w, h)
        f = pred_dir / f"{r['image']}.txt"
        polys[r["image"]] = []
        if f.is_file():
            for ln in f.read_text().splitlines():
                v = ln.split()
                if len(v) < 8 or (len(v) - 2) % 2:
                    continue
                xy = [(float(v[i]) * w, float(v[i + 1]) * h) for i in range(2, len(v), 2)]
                polys[r["image"]].append((float(v[1]), int(v[0]), xy))
    return size, polys


def _mask(xy, w, h, grow=0):
    im = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(im)
    d.polygon(xy, fill=1)
    if grow:
        d.line(xy + xy[:1], fill=1, width=2 * grow + 1)
    return np.asarray(im, dtype=bool)


def _turn(xy, flip: int):
    """flip bits: 1 = mirror x, 2 = mirror y, 4 = transpose; whole-pixel coordinates stay whole."""
    xs, ys = [p[0] for p in xy], [p[1] for p in xy]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    out = []
    for x, y in xy:
        if flip & 1:
            x = x0 + x1 - x
        if flip & 2:
            y = y0 + y1 - y
        if flip & 4:
            x, y = x0 + (y - y0), y0 + (x - x0)
        out.append((x, y))
    return out


def make_decoys(ps_dir, cands: list, n: int, seed: int) -> list:
    """n decoys [{image, xy (px), conf, cls, src_image, src_line, flip}] (module doc). cands: (image, line, conf, cls)."""
    if n <= 0:
        return []
    if not cands:
        raise ValueError("no candidates to draw decoy shapes from")
    size, polys = _px_rings(ps_dir)
    rng = random.Random(f"decoys:{seed}")
    lines = {}
    _req, _ps, _model, _rows, pred_dir = pp._ps_inputs(Path(ps_dir))
    for img in {c[0] for c in cands}:
        lines[img] = [o for o in O.read_lines(pred_dir / f"{img}.txt")]
    occ = {}
    images = sorted({c[0] for c in cands})
    out = []
    for _ in range(n):
        for _t in range(TRIES):
            img = rng.choice(images)
            w, h = size[img]
            if img not in occ:
                m = np.zeros((h, w), dtype=bool)
                for _c, _k, xy in polys[img]:
                    m |= _mask(xy, w, h, MARGIN)
                occ[img] = m
            src = rng.choice(cands)
            sw, sh = size[src[0]]
            ring = lines[src[0]][src[1]]["p"][0]
            xy = [(round(ring[i] * sw), round(ring[i + 1] * sh)) for i in range(0, len(ring), 2)]
            flip = rng.randrange(8)
            xy = _turn(xy, flip)
            xs, ys = [p[0] for p in xy], [p[1] for p in xy]
            bw, bh = max(xs) - min(xs), max(ys) - min(ys)
            if bw + 2 * MARGIN >= w or bh + 2 * MARGIN >= h:
                continue
            ox = rng.randint(MARGIN, w - 1 - MARGIN - bw) - min(xs)
            oy = rng.randint(MARGIN, h - 1 - MARGIN - bh) - min(ys)
            xy = [(x + ox, y + oy) for x, y in xy]
            m = _mask(xy, w, h, MARGIN)
            if (m & occ[img]).any():
                continue
            occ[img] |= m
            pick = rng.choice(cands)            # conf and class from the candidates, independent of the shape
            out.append({"image": img, "xy": xy, "conf": pick[2], "cls": pick[3], "src_image": src[0],
                        "src_line": src[1], "flip": flip})
            break
        else:
            raise ValueError(f"no free place for decoy {len(out) + 1} after {TRIES} tries")
    return out


def _new_ps_id(src_id: str, seed: int, n: int) -> str:
    h = hashlib.sha256(f"{src_id}:{seed}:{n}:decoys".encode()).hexdigest()[:6]
    return f"ps_{src_id.split('_')[1]}_{h}"


def derive(ps_dir, out_dir, decoys: list, seed: int) -> tuple:
    """Write the derived prediction set (module doc). -> (out_dir, [decoy rows with their line])."""
    ps_dir, out_dir = Path(ps_dir), Path(out_dir)
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} exists: --derive makes a new folder")
    req, ps_id, model, rows, pred_dir = pp._ps_inputs(ps_dir)
    new_id = _new_ps_id(ps_id, seed, len(decoys))
    shutil.copytree(ps_dir, out_dir)            # byte copy: images, images.csv, status.json, predictions, logs
    req2 = dict(req)
    req2["ps"] = new_id
    # request.json's name / note / provenance reach the page (/api/meta): nothing there may say "decoy"
    req2["name"] = f"{req.get('name') or ps_id} (eVHarmony copy)"
    prov = dict(req.get("provenance") or {})
    prov["derived_from"] = [{"kind": "prediction_set", "id": ps_id}] + [
        x for x in prov.get("derived_from") or [] if not (isinstance(x, dict) and x.get("id") == ps_id)]
    req2["provenance"] = prov
    (out_dir / DERIVED).write_text(json.dumps({  # never read by the review server or /api/meta
        "schema": "evh_derived/1", "from": ps_id, "ps": new_id, "n_decoys": len(decoys), "seed": seed,
        "tool": "tools/evh_items.py --decoys", "note": "decoy polygon lines appended to the prediction files; the "
        "list of them is kept off-package (<items>_decoys.csv)"}, indent=1) + "\n", encoding="utf-8")
    (out_dir / "request.json").write_text(json.dumps(req2, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    size = {}
    for r in rows:
        with Image.open(ps_dir / r["file"]) as im:
            size[r["image"]] = im.size
    dpred = out_dir / "p" / model / "predictions"
    rows_out = []
    by_img = {}
    for d in decoys:
        by_img.setdefault(d["image"], []).append(d)
    for img, ds in sorted(by_img.items()):
        f = dpred / f"{img}.txt"
        raw = f.read_bytes() if f.is_file() else b""     # every byte of the model's file is kept; lines appended
        nl = b"\r\n" if b"\r\n" in raw else b"\n"
        n0 = len(raw.decode().splitlines())               # = object_set.read_lines' line count
        if raw and not raw.endswith((b"\n", b"\r")):
            raw += nl
        w, h = size[img]
        for k, d in enumerate(ds):
            coords = " ".join(f"{x / w:.6f} {y / h:.6f}" for x, y in d["xy"])
            raw += f"{d['cls']} {d['conf']:.4f} {coords}".encode() + nl
            rows_out.append({**d, "line": n0 + k})
        f.write_bytes(raw)
    return out_dir, new_id, rows_out


def decoys_csv(rows: list) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(DECOY_COLUMNS)
    for d in sorted(rows, key=lambda d: (d["image"], d["line"])):
        w.writerow([d["image"], d["line"], f"{d['conf']:.4f}", d["cls"], d["src_image"], d["src_line"], d["flip"]])
    return buf.getvalue()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ps", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--lo", type=float, default=0.25)
    ap.add_argument("--hi", type=float, default=0.5)
    ap.add_argument("--gold", type=int, default=40)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--low", type=float, help="r17c: also candidates with LOW <= conf < --lo (likely Nopes)")
    ap.add_argument("--n-low", type=int, default=20)
    ap.add_argument("--decoys", type=int, default=0, help="r17c: N known-Nope decoys (needs --derive)")
    ap.add_argument("--derive", type=Path, help="r17c: the derived prediction set's new folder")
    ap.add_argument("--answered", type=Path, help="r17c: an old object set whose answered objects go first")
    ap.add_argument("--who", help="r17c: with --answered, the reviewer(s) whose answers count (comma-separated)")
    ap.add_argument("--answered-attr", default="match", help="r17f: the attribute that counts as answered")
    ap.add_argument("--images-of", type=Path, help="r17f: only the images of this object set's items")
    ap.add_argument("--exclude", default="", help="r17f: IMG:LINE,... objects left out")
    a = ap.parse_args(argv)
    if a.decoys and not a.derive:
        ap.error("--decoys needs --derive <new folder>")
    if a.answered and not a.who:
        ap.error("--answered needs --who")
    cands = candidates(a.ps, a.lo, a.hi)
    only, excl = None, set()
    if a.images_of:
        only = {str(it["image"]) for it in O.read_items(a.images_of)["items"]}
        n0 = len(cands)
        cands = [c for c in cands if c[0] in only]
        print(f"[images-of] {len(only)} images of {a.images_of.name}: {len(cands)} of {n0} candidates")
    if a.exclude:
        for t in a.exclude.split(","):
            img, ln = t.strip().rsplit(":", 1)
            excl.add((img, int(ln)))
        n0 = len(cands)
        cands = [c for c in cands if (c[0], c[1]) not in excl]
        print(f"[exclude] {n0 - len(cands)} of {len(excl)} listed objects were candidates and are left out")
    if not cands:
        print(f"[evh_items] no candidates with {a.lo} <= conf < {a.hi}", file=sys.stderr)
        return 2
    r17c = bool(a.low is not None or a.decoys or a.answered or only is not None or excl)
    if not r17c:
        text = items_csv(cands, a.gold, a.seed)
        a.out.write_text(text, encoding="utf-8", newline="")
        ng = min(a.gold, len(cands))
        print(f"[OK] {a.out}: {len(cands)} candidates ({a.lo} <= conf < {a.hi}) on "
              f"{len({c[0] for c in cands})} images; first {ng} rows = gold (conf-stratified, seed {a.seed}). "
              f"Build with --first-items {ng} --app evharmony")
        return 0
    first = []
    if a.answered:
        from evh_carry import answered_objects
        done = set()
        for who in a.who.split(","):
            done |= answered_objects(a.answered, who.strip(), a.answered_attr)
        first += [c for c in cands if (c[0], c[1]) in done]
        print(f"[answered] {len(first)} of the candidates were answered by {a.who} in {a.answered.name} "
              f"({len(done)} answered there)")
    low = []
    if a.low is not None:
        pool = candidates(a.ps, a.low, a.lo)
        low = sorted(random.Random(f"low:{a.seed}").sample(pool, min(a.n_low, len(pool))))
        print(f"[low] {len(pool)} candidates with {a.low} <= conf < {a.lo}; {len(low)} added"
              + ("" if pool else " (the set was predicted at conf >= its threshold: re-predict lower for these)"))
    ps_used, drows = a.ps, []
    if a.decoys:
        dc = [c for c in candidates(a.ps, a.lo, a.hi, with_cls=True) if only is None or c[0] in only]
        ds = make_decoys(a.ps, dc, a.decoys, a.seed)
        ps_used, new_id, drows = derive(a.ps, a.derive, ds, a.seed)
        dfile = a.out.with_name(a.out.stem + "_decoys.csv")
        dfile.write_text(decoys_csv(drows), encoding="utf-8", newline="")
        print(f"[decoys] {len(drows)} decoys on {len({d['image'] for d in drows})} images -> derived prediction set "
              f"{new_id} at {a.derive}; list (off-package) {dfile}")
    dec = [(d["image"], d["line"], round(d["conf"], 4)) for d in drows]
    allc = sorted(cands + low + dec, key=lambda c: (c[0], c[1]))
    first = first + dec + low
    text = items_csv(allc, a.gold, a.seed, first)
    a.out.write_text(text, encoding="utf-8", newline="")
    nf = len(first) + min(a.gold, len(allc) - len(first))
    print(f"[OK] {a.out}: {len(allc)} rows ({len(cands)} candidates {a.lo} <= conf < {a.hi}, {len(low)} low, "
          f"{len(dec)} decoys) on {len({c[0] for c in allc})} images; first {nf} rows = gold. Build with "
          f"--ps {ps_used} --first-items {nf} --app evharmony")
    return 0


if __name__ == "__main__":
    sys.exit(main())
