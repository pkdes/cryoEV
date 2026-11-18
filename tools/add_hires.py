"""Add 2x ("hi") images to an existing review package, so the review page can draw sharper pixels on zoom (round 16a,
stage 2, TODO #85). Additive only: nothing a reviewer saved, and no existing file, changes.

  python tools/add_hires.py <package dir> [--ps-dir DIR | --source DIR] [--width 2880] [--quality 90]
                            [--only 0001,0002] [--preview OUT_DIR] [--do]

What it writes (with --do):
  images/<stem>.hi.jpg   grayscale JPEG, min(--width, source width) px wide, no EXIF; named by the package's own image
                         stem, never by the source name (source names can carry sample names)
  data.js                each passing image gets "hi": "images/<stem>.hi.jpg", "hi_w", "hi_h" (keys added only; checked)
  hires_manifest.json    what was added, from which source (sha256), the alignment numbers, when

Sources (one of):
  --ps-dir   the prediction set the package was built from (default: <laptop root>/predict/<data.js prediction_set>);
             images.csv maps image -> file. Used for adopted runs, whose files are the full-size originals.
  --source   a folder of originals, matched to images.csv orig_name by file stem (case-insensitive). Never under Box
             or the processing datasets (data_utils.config.protected_roots): stage a copy first.

Alignment check (the known-answer control): each hi image, shrunk to the package image's size, must line up with
the package image: phase-correlation shift < 1 px and normalized correlation >= the threshold. Two negatives run on
every call and must FAIL, or the whole call is refused: (1) image i's hi vs image j's package image, (2) the package
image shifted by 2 px (must measure a 2 px shift). The threshold is set halfway between the worst positive and the
best negative and reported. An image that fails keeps 1440 only and is listed.

Proof (with --do): md5 of every package file except edits/, data.js and the new *.hi.jpg is identical before and
after; data.js after == before plus only the hi keys. Writes run under the package's meta lock (the same one the
Reviews hub and object_set instructions use), each file via temp + os.replace.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from annotation import object_set as OS  # noqa: E402
from annotation import prediction_package as pp  # noqa: E402

HI_KEYS = ("hi", "hi_w", "hi_h")
MIN_GAIN = 1.25            # a source must be at least this much wider than the package image to be worth adding
SHIFT_MAX = 1.0            # px, at the package image's size


class HiresError(Exception):
    pass


# ---------------------------------------------------------------- image math
def _gray(im: Image.Image, size=None) -> np.ndarray:
    im = im.convert("L")
    if size and im.size != size:
        im = im.resize(size, Image.LANCZOS)
    a = np.asarray(im, dtype=np.float64)
    return (a - a.mean()) / (a.std() or 1.0)


def phase_shift(a: np.ndarray, b: np.ndarray) -> tuple:
    """(dy, dx, peak) of b relative to a by phase correlation, integer-pixel + parabolic subpixel."""
    fa, fb = np.fft.fft2(a), np.fft.fft2(b)
    r = fa * np.conj(fb)
    r /= np.abs(r) + 1e-12
    c = np.real(np.fft.ifft2(r))
    iy, ix = np.unravel_index(np.argmax(c), c.shape)

    def sub(cm, cc, cp):
        d = cm - 2 * cc + cp
        return 0.0 if d == 0 else 0.5 * (cm - cp) / d
    h, w = c.shape
    dy = iy + sub(c[(iy - 1) % h, ix], c[iy, ix], c[(iy + 1) % h, ix])
    dx = ix + sub(c[iy, (ix - 1) % w], c[iy, ix], c[iy, (ix + 1) % w])
    dy = dy - h if dy > h / 2 else dy
    dx = dx - w if dx > w / 2 else dx
    return float(dy), float(dx), float(c[iy, ix])


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    return float((a * b).mean())


def compare(base: np.ndarray, cand: np.ndarray) -> dict:
    dy, dx, peak = phase_shift(base, cand)
    return {"shift": round(float(np.hypot(dy, dx)), 3), "ncc": round(ncc(base, cand), 4), "peak": round(peak, 4)}


# ---------------------------------------------------------------- sources
def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _protected(p: Path) -> bool:
    from data_utils import config as C
    rp = p.resolve()
    for root in C.protected_roots():
        try:
            if rp.is_relative_to(Path(root).resolve()):
                return True
        except OSError:
            continue
    return False


def sources(pkg: Path, data: dict, ps_dir=None, source=None) -> dict:
    """{package image file: source path or None}."""
    if source is not None:
        source = Path(source)
        if _protected(source):
            raise HiresError(f"{source}: under Box or the processing datasets; stage a copy outside them first")
        if not source.is_dir():
            raise HiresError(f"{source}: not a folder")
    if ps_dir is None:
        from data_utils import config as C
        ps = data.get("prediction_set")
        if not isinstance(ps, str) or not ps:
            raise HiresError("data.js has no prediction_set; pass --ps-dir")
        ps_dir = C.laptop_root() / "predict" / ps
    ps_dir = Path(ps_dir)
    csv_p = ps_dir / "images.csv"
    if not csv_p.is_file():
        raise HiresError(f"{csv_p}: missing")
    with open(csv_p, newline="", encoding="utf-8") as f:
        rows = {r["image"]: r for r in csv.DictReader(f)}
    by_stem = {}
    if source is not None:
        for p in source.iterdir():
            if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
                by_stem.setdefault(p.stem.lower(), []).append(p)
    out = {}
    for im in data.get("images", []):
        stem = Path(im["file"]).stem
        r = rows.get(stem)
        if r is None:
            out[im["file"]] = None
            continue
        if source is None:
            p = ps_dir / r["file"]
            out[im["file"]] = p if p.is_file() else None
        else:
            hits = by_stem.get(Path(r.get("orig_name") or "").stem.lower(), [])
            out[im["file"]] = hits[0] if len(hits) == 1 else None
    return out


# ---------------------------------------------------------------- plan
def plan(pkg, ps_dir=None, source=None, width=2880, only=None) -> dict:
    pkg = Path(pkg)
    djs = pkg / "data.js"
    if not djs.is_file():
        raise HiresError(f"{pkg}: not a review package (no data.js)")
    data, _, _ = OS._split_datajs(djs.read_bytes())
    src = sources(pkg, data, ps_dir, source)
    rows, bases, his = [], {}, {}
    for im in data["images"]:
        f = im["file"]
        if only and Path(f).stem not in only:
            continue
        row = {"file": f, "src": str(src.get(f)) if src.get(f) else None, "status": "", "why": ""}
        rows.append(row)
        base_p = pkg / im.get("src", f"images/{f}")
        if not base_p.is_file() or src.get(f) is None:
            row.update(status="skip", why="no package image" if not base_p.is_file() else "no source found")
            continue
        with Image.open(base_p) as b0:
            bsize = b0.size
            bases[f] = _gray(b0)
        with Image.open(src[f]) as s0:
            s0.load()
            sw, sh = s0.size
            if sw < bsize[0] * MIN_GAIN:
                row.update(status="skip", why=f"source {sw} px is not wider than {bsize[0]} x {MIN_GAIN}")
                continue
            if abs(sh / sw - bsize[1] / bsize[0]) * bsize[1] > 1.5:
                row.update(status="fail", why=f"aspect differs: source {sw}x{sh} vs package {bsize[0]}x{bsize[1]}")
                continue
            tw = min(width, sw)
            hi = s0.convert("L").resize((tw, round(sh * tw / sw)), Image.LANCZOS)
        his[f] = hi
        row.update(hi_w=hi.width, hi_h=hi.height, src_w=sw, **compare(bases[f], _gray(hi, bsize)))
    pos = [f for f in his]
    neg = []
    # negative 1: image i's hi vs image j's package image (same size only)
    for i, f in enumerate(pos):
        g = next((g for g in pos[i + 1:] + pos[:i] if bases[g].shape == bases[f].shape), None)
        if g:
            neg.append({"kind": "mismatched pair", "a": f, "b": g, **compare(bases[g], _gray(his[f], bases[g].shape[::-1]))})
    # negative 2: the package image vs itself shifted 2 px must MEASURE ~2 px
    shift_checks = []
    for f in pos[:3]:
        b = bases[f]
        sh = np.roll(b, 2, axis=1)
        shift_checks.append({"file": f, **compare(b, sh)})
    pos_ncc = [r["ncc"] for r in rows if "ncc" in r]
    neg_ncc = [n["ncc"] for n in neg]
    ctrl = {"negatives": neg, "shift_checks": shift_checks, "threshold": None, "ok": False, "why": ""}
    if not pos_ncc:
        ctrl["why"] = "no candidate images"
    elif not neg:
        ctrl["why"] = "need at least 2 same-size images for the mismatched-pair negative"
    elif not all(1.5 <= s["shift"] <= 2.5 for s in shift_checks):
        ctrl["why"] = "the 2 px shift control did not measure 2 px: the shift method is broken"
    elif max(neg_ncc) >= sorted(pos_ncc)[len(pos_ncc) // 2]:
        ctrl["why"] = "negatives score as high as typical positives: the correlation does not separate; refused"
    else:
        good = [v for v in pos_ncc if v > max(neg_ncc)]
        ctrl["threshold"] = round((min(good) + max(neg_ncc)) / 2, 4)
        ctrl["ok"] = True
    for r in rows:
        if "ncc" not in r:
            continue
        if not ctrl["ok"]:
            r.update(status="fail", why="control failed")
        elif r["shift"] >= SHIFT_MAX:
            r.update(status="fail", why=f"shift {r['shift']} px")
        elif r["ncc"] < ctrl["threshold"]:
            r.update(status="fail", why=f"ncc {r['ncc']} < {ctrl['threshold']}")
        else:
            r["status"] = "pass"
    return {"pkg": str(pkg), "rows": rows, "control": ctrl, "_his": his, "_src": src, "width": width}


# ---------------------------------------------------------------- write
def _md5_tree(pkg: Path) -> dict:
    out = {}
    for p in sorted(pkg.rglob("*")):
        rel = p.relative_to(pkg).as_posix()
        if (not p.is_file() or rel.startswith("edits/") or rel in ("data.js", "hires_manifest.json")
                or rel.endswith(".hi.jpg") or rel.startswith("images/c_") or rel.endswith(".tmp") or "/." in "/" + rel):
            continue                                    # dot-files/folders: locks and the key (never rewritten)
        out[rel] = hashlib.md5(p.read_bytes()).hexdigest()
    return out


def _jpeg(im: Image.Image, quality: int) -> bytes:
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality, optimize=True)          # a fresh L image: no EXIF to carry
    return buf.getvalue()


def write(P: dict, quality=90, lock_wait=5.0) -> dict:
    pkg = Path(P["pkg"])
    passing = [r for r in P["rows"] if r["status"] == "pass"]
    if not P["control"]["ok"]:
        raise HiresError(f"control failed: {P['control']['why']}")
    if not passing:
        raise HiresError("no image passed; nothing written")
    before = _md5_tree(pkg)
    lk = pp.meta_lock(pkg.parent, pkg.name, lock_wait)
    try:
        djs = pkg / "data.js"
        raw = djs.read_bytes()
        data, kw, tail = OS._split_datajs(raw)
        orig, _, _ = OS._split_datajs(raw)
        by_file = {im["file"]: im for im in data["images"]}
        for r in passing:
            stem = Path(r["file"]).stem
            rel = f"images/{stem}.hi.jpg"
            OS._write_atomic(pkg / rel, _jpeg(P["_his"][r["file"]], quality))
            by_file[r["file"]].update(hi=rel, hi_w=r["hi_w"], hi_h=r["hi_h"])
        check = json.loads(json.dumps(data))
        for im in check["images"]:
            for k in HI_KEYS:
                im.pop(k, None)
        for im in orig["images"]:
            for k in HI_KEYS:
                im.pop(k, None)
        if check != orig:
            raise HiresError("internal: data.js would change beyond the hi keys; refused")
        OS._write_atomic(djs, OS.DATAJS_PREFIX + json.dumps(data, **kw).encode("utf-8") + tail)
        man = {"tool": "tools/add_hires.py", "t": pp._now(), "width": P["width"], "quality": quality,
               "control": {k: v for k, v in P["control"].items()},
               "images": [{k: r.get(k) for k in ("file", "status", "why", "src", "src_w", "hi_w", "hi_h", "shift", "ncc")}
                          | ({"src_sha256": _sha256(Path(r["src"]))} if r["status"] == "pass" else {})
                          for r in P["rows"]]}
        OS._write_atomic(pkg / "hires_manifest.json", (json.dumps(man, indent=1) + "\n").encode("utf-8"))
    finally:
        lk.close()
    after = _md5_tree(pkg)
    if before != after:
        changed = sorted(set(before.items()) ^ set(after.items()))
        raise HiresError(f"md5 proof FAILED: other files changed: {changed[:5]}")
    return {"written": len(passing), "md5_files_checked": len(before)}


CHI_KEYS = ("chi", "chi_box", "chi_w", "chi_h")
CHI_PAD = 1.5                # the crop covers it.crop expanded this much about its centre (card zoom / pan room)
CHI_MAX = 2560               # px: a larger crop is shrunk to this side (r17f: 1536 -> 2560; phone memory)
ITEM_ID_RE = __import__("re").compile(r"^[A-Za-z0-9_-]{1,64}$")


def _level_w(pkg: Path, data: dict, f: str, cache: dict) -> int:
    """r17f: the width in px of the finest whole-image level the page already has for image file f: its 2x image
    (hi_w), else the package's base JPEG."""
    if f not in cache:
        imr = next((x for x in data.get("images", []) if x.get("file") == f), {})
        w = imr.get("hi_w") or 0
        if not w and imr.get("src") and (pkg / imr["src"]).is_file():
            with Image.open(pkg / imr["src"]) as b:
                w = b.width
        cache[f] = int(w or 0)
    return cache[f]


def chi_box(crop) -> list:
    x0, y0, x1, y1 = crop
    cx, cy, hw, hh = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) * CHI_PAD / 2, (y1 - y0) * CHI_PAD / 2
    return [round(max(0.0, cx - hw), 6), round(max(0.0, cy - hh), 6), round(min(1.0, cx + hw), 6),
            round(min(1.0, cy + hh), 6)]


def write_crops(P: dict, quality=90, lock_wait=5.0) -> dict:
    """Round 16b: per object-set item, a full-resolution crop of the source (images/c_<item id>.jpg) + data.js item
    keys chi / chi_box / chi_w / chi_h. Only items on images that passed the alignment control. Additive, as write()."""
    pkg = Path(P["pkg"])
    if not P["control"]["ok"]:
        raise HiresError(f"control failed: {P['control']['why']}")
    ok_files = {r["file"] for r in P["rows"] if r["status"] == "pass"}
    before = _md5_tree(pkg)
    n, skipped = 0, []
    lk = pp.meta_lock(pkg.parent, pkg.name, lock_wait)
    try:
        djs = pkg / "data.js"
        raw = djs.read_bytes()
        data, kw, tail = OS._split_datajs(raw)
        orig, _, _ = OS._split_datajs(raw)
        file_of = {Path(im["file"]).stem: im["file"] for im in data["images"]}
        src_cache, lvl_cache = {}, {}
        for it in data.get("items", []):
            f = file_of.get(str(it.get("image")).split(".")[0])
            iid = str(it.get("id", ""))
            c = it.get("crop")
            if f not in ok_files or not ITEM_ID_RE.match(iid) or not (isinstance(c, list) and len(c) == 4
                                                                       and max(c) <= 1.5 and c[2] > c[0] and c[3] > c[1]):
                skipped.append(iid)
                continue
            if f not in src_cache:
                src_cache.clear()
                with Image.open(P["_src"][f]) as s0:
                    src_cache[f] = s0.convert("L")
            s = src_cache[f]
            box = chi_box(c)
            px = (round(box[0] * s.width), round(box[1] * s.height), round(box[2] * s.width), round(box[3] * s.height))
            im = s.crop(px)
            if max(im.size) > CHI_MAX:
                k = CHI_MAX / max(im.size)
                im = im.resize((max(1, round(im.width * k)), max(1, round(im.height * k))), Image.LANCZOS)
            # r17f: a crop no finer than the image's 2x level (or, without one, the base image) adds nothing: skipped
            lvl_w = _level_w(pkg, data, f, lvl_cache)
            if lvl_w and im.width / ((box[2] - box[0]) * lvl_w) <= 1.0:
                skipped.append(iid)
                continue
            rel = f"images/c_{iid}.jpg"
            OS._write_atomic(pkg / rel, _jpeg(im, quality))
            it.update(chi=rel, chi_box=box, chi_w=im.width, chi_h=im.height)
            n += 1
        check, base = json.loads(json.dumps(data)), orig
        for d in (check, base):
            for it in d.get("items", []):
                for k in CHI_KEYS:
                    it.pop(k, None)
        if check != base:
            raise HiresError("internal: data.js would change beyond the chi keys; refused")
        OS._write_atomic(djs, OS.DATAJS_PREFIX + json.dumps(data, **kw).encode("utf-8") + tail)
    finally:
        lk.close()
    after = _md5_tree(pkg)
    if before != after:
        raise HiresError("md5 proof FAILED: other files changed")
    return {"crops": n, "skipped": len(skipped)}


def preview(P: dict, out_dir: Path, quality=90, n=2):
    out_dir.mkdir(parents=True, exist_ok=True)
    done = []
    for r in [r for r in P["rows"] if r["status"] == "pass"][:n]:
        p = out_dir / f"{Path(r['file']).stem}.hi.jpg"
        p.write_bytes(_jpeg(P["_his"][r["file"]], quality))
        done.append(str(p))
    return done


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("pkg")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--ps-dir")
    g.add_argument("--source")
    ap.add_argument("--width", type=int, default=2880)
    ap.add_argument("--quality", type=int, default=90)
    ap.add_argument("--only", help="comma-separated image stems")
    ap.add_argument("--preview", help="write the first 2 passing hi images here (not into the package)")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--item-crops", action="store_true",
                    help="object sets: full-resolution crops per item (images/c_<id>.jpg + chi keys) instead of 2x images")
    ap.add_argument("--do", action="store_true")
    a = ap.parse_args(argv)
    only = set(a.only.split(",")) if a.only else None
    try:
        t0 = time.time()
        P = plan(a.pkg, a.ps_dir, a.source, a.width, only)
        c = P["control"]
        print(f"control: {'OK' if c['ok'] else 'FAILED'} threshold={c['threshold']} {c['why']}")
        for s in c["shift_checks"]:
            print(f"  2px-shift control {s['file']}: measured {s['shift']} px")
        if c["negatives"]:
            print(f"  mismatched-pair negatives: {len(c['negatives'])}, best ncc {max(n['ncc'] for n in c['negatives'])}")
        for r in P["rows"]:
            print(f"  {r['file']:>10} {r['status']:>5} shift={r.get('shift', '-')} ncc={r.get('ncc', '-')} "
                  f"{r.get('src_w', '')}->{r.get('hi_w', '')} {r['why']}")
        n = sum(r["status"] == "pass" for r in P["rows"])
        print(f"{n} of {len(P['rows'])} pass ({time.time() - t0:.0f} s)")
        if a.preview:
            for p in preview(P, Path(a.preview), a.quality):
                print("preview:", p)
        if a.do and a.item_crops:
            print(json.dumps(write_crops(P, a.quality)))
        elif a.do:
            print(json.dumps(write(P, a.quality)))
        else:
            print("dry run: nothing written to the package (add --do)")
        return 0 if c["ok"] else 2
    except (HiresError, OS.ObjectSetError) as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
