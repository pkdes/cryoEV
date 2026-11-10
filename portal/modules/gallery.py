"""
Gallery module (BUILD_CONTRACTS.md section 13; round 7 Phase 1): browse the montages and tiles of a raw conversion,
mark tiles keep / maybe / reject (with reasons, stars, a note), read a report of the run, download it.

    PREFIX = "/gallery"    ROLES = admin + advanced (as Predict's raw conversion: predict.RAW_ROLES)

    GET  /gallery/                       my conversions (admin: Mine / All); tick several -> Compare
    GET  /gallery/<cid>/                 the gallery of one conversion
    GET  /gallery/compare?c=<cid>&c=...  several conversions side by side (every cid access-checked)
    GET  /gallery/<cid>/report           the report page (also /gallery/compare/report?c=...)
    GET  /gallery/<cid>/report.zip       the report as a standalone HTML zip (also /gallery/compare/report.zip?c=...)
    GET  /gallery/<cid>/images.zip       tiles + montages + MANIFEST + acquisition.json, streamed (also .../compare/...)
    GET  /gallery/<cid>/api/state        {tiles, tags, state: {key: {flag, stars, tags, note, by}}} (this person's cid)
    POST /gallery/<cid>/api/event        {k, f, v} -> appended to raw_tiles/<cid>/gallery/events.jsonl
    GET  /gallery/<cid>/{t,ts,m,ms,mp,rep}/<name>   tile, tile thumbnail, montage, small / medium montage, report tile
    GET  /gallery/gallery.js             the page script (portal/static/gallery.js; no inline script anywhere)

Access: every route above asks predict.load_conversion -> ctx.can_see(user, convert.json owner). A cid that does not
exist and one that belongs to someone else give the identical 404 (per kind: page / json / image). Files are only
served by name from the conversion's own MANIFEST / layout.json (never a path from the URL) and checked with
data.is_contained. The POST needs the Gate's Origin check plus the header X-Requested-With: gallery.

Storage: <conversion>/gallery/events.jsonl, one JSON object per line, append only, under a thread lock + an OS lock
(data_utils/oslock.py, as portal/audit.py):
    {"k": tile output_name or montage name, "f": "flag"|"stars"|"tags"|"note", "v": ..., "who": portal id, "t": server UTC}
The latest value per (k, f) wins (`fold`). flag in keep|maybe|reject|"" ; stars 0-5 ; tags a subset of TAGS ; note <= 1000.
Lines that fail validation on read are ignored (a hand-edited file can't break a page).

Escaping: everything that comes from a CSV, a JSON file or a person (sample_label, notes, tags, names) goes through
html.escape; the standalone report is plain HTML with no script at all.
"""
from __future__ import annotations

import csv
import errno
import html
import io
import json
import math
import os
import re
import threading
import time
from collections import Counter, OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import FileResponse, PlainTextResponse, Response, StreamingResponse
from starlette.routing import Route

from data_utils.oslock import acquire_lock
from portal import zipstream
from portal.modules import data as D
from portal.modules import predict as P

PREFIX = "/gallery"
ROLES = P.RAW_ROLES
SCRIPT = Path(__file__).resolve().parent.parent / "static" / "gallery.js"
CSS_FILE = Path(__file__).resolve().parent.parent / "static" / "cj.css"
CSS_MARK = "/* cj-gallery:begin */"

TAGS = ("thick ice", "empty", "contaminated", "crystalline ice", "broken carbon", "aggregates",
        "out of focus / drift", "other")
FLAGS = ("keep", "maybe", "reject")
FIELDS = ("flag", "stars", "tags", "note")
NOTE_MAX = 1000
BODY_MAX = 16 * 1024
MAX_COMPARE = 6
MAX_REP = 60                      # starred images drawn in one report
SIZES = {"ms": 480, "mp": 1000}   # montage copies: report / gallery
REP_PX = 960
MAX_ZIPS = 2                      # concurrent downloads per person
ZIP_STALE_S = 120                 # a slot whose stream pulled no chunk for this long is dead (client gone) and is ignored
BODY_ERR = 413
EVENTS_MAX = 20 * 1024 * 1024     # events.jsonl stops growing here (507)
NICE_NM = (5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000)
GONE_TEXT = "There is no gallery here that you can open."
esc = lambda s: html.escape("" if s is None else str(s), quote=True)  # noqa: E731
u = lambda s: html.escape(quote(str(s), safe=""), quote=True)  # noqa: E731   (a name inside a URL path)
CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
RC = re.compile(r"_r(\d+)c(\d+)")


# ---------------------------------------------------------------- small readers
def _json(p: Path):
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _num(v, default=None):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    return x if math.isfinite(x) else default


def _text(v, n=200) -> str:
    return "" if v is None else str(v)[:n]


def natkey(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(s))]


BAD_NAME = re.compile(r'[\x00-\x1f<>:"|?*/\\]')
RESERVED = frozenset({"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10))})


def plain_name(name) -> bool:
    """A file name that is only a name: no separator, drive or stream colon, control or Windows-reserved character,
    no leading dot, no trailing dot / space, no device name (checked on the part before the FIRST dot)."""
    return (isinstance(name, str) and 0 < len(name) <= 200 and not BAD_NAME.search(name)
            and not name.startswith(".") and name == name.rstrip(" .")
            and name.split(".", 1)[0].strip().lower() not in RESERVED)


_CACHE: dict = {}
_CACHE_LOCK = threading.Lock()


def cached(path: Path, load):
    """load(path), kept while the file's mtime and size are the same (small bounded cache)."""
    try:
        st = path.stat()
    except OSError:
        return load(path)
    key = (str(path), st.st_mtime_ns, st.st_size)
    with _CACHE_LOCK:
        if key in _CACHE:
            return _CACHE[key]
    val = load(path)
    with _CACHE_LOCK:
        if len(_CACHE) > 200:
            _CACHE.clear()
        _CACHE[key] = val
    return val


def read_manifest(path: Path) -> list:
    try:
        with open(path, encoding="utf-8-sig", newline="") as f:
            return [dict(r) for r in csv.DictReader(f)]
    except (OSError, UnicodeDecodeError, csv.Error):
        return []


# ---------------------------------------------------------------- the conversion's contents
class Conv:
    """One raw conversion: tiles (MANIFEST), montages (layout.json, else grouped from MANIFEST), acquisition, mirror."""

    def __init__(self, d: Path, meta: dict, status: dict):
        self.dir, self.meta, self.status = d, meta, status or {}
        self.cid = d.name
        self.session = _text(meta.get("session"), 60)
        self.created = _text(meta.get("created"), 40)
        self.owner = meta.get("owner")
        rows = cached(d / "tiles" / "MANIFEST.csv", read_manifest)
        self.tiles: OrderedDict = OrderedDict()
        for r in rows:
            name = (r.get("output_name") or r.get("file") or "").strip()
            if not plain_name(name) or name in self.tiles:
                continue
            nm = _num(r.get("nm_per_px"))
            self.tiles[name] = {"name": name, "montage": _text(r.get("montage"), 120).strip() or "(no montage)",
                                "grid": _text(r.get("grid"), 40).strip(), "sample": _text(r.get("sample_label"), 120).strip(),
                                "nm": nm if nm and nm > 0 else None, "source": _text(r.get("source_file"), 200)}
        self.layout = cached(d / "montages" / "layout.json", _json)
        self.stitch = cached(d / "montages" / "stitch.json", _json)
        acq = cached(d / "acquisition.json", _json)
        self.acq = acq if isinstance(acq, dict) else {}
        mir = cached(d / "mirror.json", _json)
        self.mirror = mir if isinstance(mir, dict) else None
        self.montages = self._montages()
        self.keys = set(self.tiles) | {m["name"] for m in self.montages}

    def _montages(self) -> list:
        out: OrderedDict = OrderedDict()
        lm = self.layout.get("montages") if isinstance(self.layout, dict) else None
        for m in (lm if isinstance(lm, list) else []):
            if not isinstance(m, dict) or not plain_name(m.get("name")):
                continue
            f = m.get("file")
            f = f if plain_name(f) and (self.dir / "montages" / f).is_file() else None
            w, h = _num(m.get("width")), _num(m.get("height"))
            boxes = []
            for t in (m.get("tiles") if isinstance(m.get("tiles"), list) else []):
                if not isinstance(t, dict) or t.get("output_name") not in self.tiles:
                    continue
                x, y, bw, bh = (_num(t.get(k)) for k in ("x", "y", "w", "h"))
                if None in (x, y, bw, bh) or bw <= 0 or bh <= 0 or not (w and h and w > 0 and h > 0):
                    continue
                boxes.append({"k": t["output_name"], "x": x, "y": y, "w": bw, "h": bh,
                              "row": _num(t.get("row"), 0), "col": _num(t.get("col"), 0)})
            boxes.sort(key=lambda b: (b["row"], b["col"]))
            nm = _num(m.get("nm_per_px"))
            out[m["name"]] = {"name": m["name"], "file": f, "w": w if w and w > 0 else None,
                              "h": h if h and h > 0 else None, "nm": nm if nm and nm > 0 else None, "sb": bool(_num(m.get("scalebar_nm"))),
                              "grid": _text(m.get("grid"), 40).strip(), "sample": _text(m.get("sample_label"), 120).strip(),
                              "boxes": boxes, "tiles": [b["k"] for b in boxes]}
        for t in self.tiles.values():
            m = out.get(t["montage"])
            if m is None:
                m = out[t["montage"]] = {"name": t["montage"], "file": None, "w": None, "h": None, "nm": None,
                                         "sb": False, "grid": t["grid"], "sample": t["sample"], "boxes": [], "tiles": []}
            if t["name"] not in m["tiles"]:
                m["tiles"].append(t["name"])
            m["grid"] = m["grid"] or t["grid"]
            m["sample"] = m["sample"] or t["sample"]
        return sorted(out.values(), key=lambda m: (natkey(m["grid"] or "~"), natkey(m["name"])))

    def groups(self) -> list:
        """[(grid, sample, [montage, ...])] by grid, then sample."""
        g: OrderedDict = OrderedDict()
        for m in self.montages:
            g.setdefault((m["grid"], m["sample"]), []).append(m)
        return [(a, b, ms) for (a, b), ms in g.items()]

    def montage(self, name):
        return next((m for m in self.montages if m["name"] == name), None)

    def tile_path(self, name):
        return self.dir / "tiles" / name if name in self.tiles else None

    def montage_path(self, name):
        m = self.montage(name)
        return self.dir / "montages" / m["file"] if m and m["file"] else None

    def n_grids(self) -> int:
        return len({(t["grid"] or t["sample"]) for t in self.tiles.values()} | {m["grid"] for m in self.montages if m["grid"]})

    def backup(self):
        """(text, tone) for the index."""
        m = self.mirror
        if not m:
            return "Not backed up yet", "mut"
        when = lambda v: _text(v, 19).replace("T", " ")[:16]  # noqa: E731
        if m.get("error") or m.get("md5_ok") is not True:
            ok = when(m.get("last_ok_at"))
            return "Backup problem" + (f" (last OK {ok})" if ok else ""), "warn"
        txt = "Backed up " + chr(0x2713) + (" " + when(m.get("mirrored_at")) if m.get("mirrored_at") else "")
        if m.get("tiles_unverified"):
            return txt + " (tiles not md5-checked)", "ok"
        return txt, "ok"


def load_conv(cfg, user, can_see, cid):
    d, meta, st = P.load_conversion(cfg, user, can_see, cid)
    if d is None:
        return None
    return Conv(d, meta, st)


# ---------------------------------------------------------------- the verdicts: events.jsonl
_EV_LOCK = threading.Lock()
LOCK_WAIT_S = 5.0


def events_path(d: Path) -> Path:
    return d / "gallery" / "events.jsonl"


def clean_value(f, v):
    """-> the value to store, or ValueError with a plain message."""
    if f == "flag":
        if v in FLAGS or v == "":
            return v
        raise ValueError("flag must be keep, maybe, reject or empty")
    if f == "stars":
        if isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 5:
            return v
        raise ValueError("stars must be a whole number from 0 to 5")
    if f == "tags":
        if not isinstance(v, list) or len(v) > len(TAGS) or any(not isinstance(t, str) or t not in TAGS for t in v):
            raise ValueError("tags must come from the fixed list")
        return [t for t in TAGS if t in v]
    if f == "note":
        if not isinstance(v, str):
            raise ValueError("note must be text")
        v = CTRL.sub("", v)
        if len(v) > NOTE_MAX:
            raise ValueError(f"note is longer than {NOTE_MAX} characters")
        return v
    raise ValueError("unknown field")


def valid_event(rec, keys) -> bool:
    if not isinstance(rec, dict) or rec.get("k") not in keys or rec.get("f") not in FIELDS:
        return False
    try:
        return clean_value(rec["f"], rec.get("v")) == rec.get("v")
    except ValueError:
        return False


def read_events(d: Path, keys) -> list:
    out = []
    try:
        with open(events_path(d), "rb") as f:
            for raw in f:
                try:
                    rec = json.loads(raw.decode("utf-8"))
                except ValueError:
                    continue
                if valid_event(rec, keys):
                    out.append(rec)
    except OSError:
        pass
    return out


def blank() -> dict:
    return {"flag": "", "stars": 0, "tags": [], "note": "", "by": {}}


def fold(records) -> dict:
    """Latest value per (k, f) wins -> {k: {flag, stars, tags, note, by: {f: {who, t}}}}; empty values are dropped."""
    out: dict = {}
    for r in records:
        e = out.setdefault(r["k"], blank())
        e[r["f"]] = r["v"]
        e["by"][r["f"]] = {"who": _text(r.get("who"), 80), "t": _text(r.get("t"), 40)}
    return {k: e for k, e in out.items() if e["flag"] or e["stars"] or e["tags"] or e["note"]}


def state_of(c: Conv) -> dict:
    return fold(read_events(c.dir, c.keys))


def append_event(c: Conv, who, k, f, v) -> dict:
    """Validate, append one line under the locks. -> the folded entry for k. ValueError: refused."""
    if k not in c.keys:
        raise ValueError("unknown image")
    if f not in FIELDS:
        raise ValueError("unknown field")
    v = clean_value(f, v)
    rec = {"k": k, "f": f, "v": v, "who": _text(who, 80),
           "t": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    line = (json.dumps(rec, ensure_ascii=False) + "\n").encode("utf-8")
    path = events_path(c.dir)
    try:
        if path.stat().st_size > EVENTS_MAX:
            raise OSError(errno.ENOSPC, "the gallery file is full")
    except FileNotFoundError:
        pass
    with _EV_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        lk = acquire_lock(path.with_name(path.name + ".lock"), wait=LOCK_WAIT_S)
        if lk is None:
            raise OSError("the gallery file is locked by another process")
        try:
            with open(path, "ab+") as f2:
                f2.seek(0, os.SEEK_END)
                if f2.tell():
                    f2.seek(-1, os.SEEK_END)
                    if f2.read(1) != b"\n":              # a cut last line (crash): never glue onto it
                        f2.write(b"\n")
                f2.write(line)
                f2.flush()
        finally:
            lk.close()
    return fold(read_events(c.dir, c.keys)).get(k, blank())


# ---------------------------------------------------------------- images
_decoders = D._decoders


def small_jpeg(path: Path, cache_dir: Path, px: int):
    """A JPEG at most px wide / high, cached on disk by path + mtime + size + px (the data thumbnail scheme)."""
    from PIL import Image
    import hashlib
    try:
        st = path.stat()
        key = hashlib.sha256(f"{path}|{st.st_mtime_ns}|{st.st_size}|{px}".encode("utf-8", "replace")).hexdigest()[:40]
        cf = cache_dir / key[:2] / f"{key}.jpg"
        try:
            return cf.read_bytes()
        except OSError:
            pass
        Image.MAX_IMAGE_PIXELS = D.THUMB_MAX_PIXELS
        with _decoders:
            with Image.open(path) as im:
                if im.size[0] * im.size[1] > D.THUMB_MAX_PIXELS:
                    return None
                im.draft("RGB", (px * 2, px * 2))
                if im.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
                    im = D.to_8bit(im)
                elif im.mode not in ("L", "RGB"):
                    im = im.convert("RGB")
                im.thumbnail((px, px))
                buf = io.BytesIO()
                im.save(buf, "JPEG", quality=82, optimize=True)
        data = buf.getvalue()
        try:
            cf.parent.mkdir(parents=True, exist_ok=True)
            tmp = cf.with_name(cf.name + f".{os.getpid()}.{threading.get_ident()}.tmp")
            tmp.write_bytes(data)
            os.replace(tmp, cf)
        except OSError:
            pass
        return data
    except Exception:                                        # unreadable / too large / not an image
        return None


def nice_bar(nm_per_screen_px: float, width_px: float):
    """(nm, bar width in px): the largest round length that takes at most a quarter of the width."""
    best = None
    for nm in NICE_NM:
        if nm / nm_per_screen_px <= width_px * 0.25:
            best = nm
    return (best, best / nm_per_screen_px) if best else (None, None)


def rep_jpeg(path: Path, nm_per_px, px=REP_PX):
    """A tile at most px wide with a scalebar drawn from nm_per_px (the report's representative images)."""
    from PIL import Image, ImageDraw, ImageFont
    try:
        Image.MAX_IMAGE_PIXELS = D.THUMB_MAX_PIXELS
        with _decoders:
            with Image.open(path) as im:
                if im.size[0] * im.size[1] > D.THUMB_MAX_PIXELS:
                    return None
                w0 = im.size[0]
                if im.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
                    im = D.to_8bit(im)
                im = im.convert("RGB")
                im.thumbnail((px, px))
        w, h = im.size
        if nm_per_px:
            nm, bar = nice_bar(nm_per_px * w0 / w, w)
            if nm:
                dr = ImageDraw.Draw(im)
                m, bh = max(8, w // 60), max(4, w // 160)
                x0, y0 = m, h - m - bh
                dr.rectangle([x0 - 1, y0 - 1, x0 + bar + 1, y0 + bh + 1], fill=(0, 0, 0))
                dr.rectangle([x0, y0, x0 + bar, y0 + bh], fill=(255, 255, 255))
                try:
                    font = ImageFont.load_default(size=max(12, w // 55))
                except TypeError:
                    font = ImageFont.load_default()
                label = f"{nm} nm" if nm < 1000 else f"{nm / 1000:g} \u00b5m"
                dr.text((x0 + 1, y0 - 4), label, fill=(255, 255, 255), font=font, anchor="ls",
                        stroke_width=2, stroke_fill=(0, 0, 0))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=88)
        return buf.getvalue()
    except Exception:
        return None


# ---------------------------------------------------------------- counts and the report
def label_of(t: dict, box=None) -> str:
    m = RC.search(t["name"])
    if box is not None and box.get("row") is not None and "col" in box:
        return f"r{int(box['row'])}c{int(box['col'])}"
    if m:
        return f"r{int(m.group(1))}c{int(m.group(2))}"
    return (os.path.splitext(t["source"])[0] or t["name"])[:24]


def tally(names, state) -> dict:
    c = {"n": len(names), "keep": 0, "maybe": 0, "reject": 0, "reasons": Counter()}
    for n in names:
        e = state.get(n)
        if e and e["flag"] in FLAGS:
            c[e["flag"]] += 1
            if e["flag"] == "reject":
                c["reasons"].update(e["tags"])
    c["unmarked"] = c["n"] - c["keep"] - c["maybe"] - c["reject"]
    return c


def merge_tally(parts) -> dict:
    t = {"n": 0, "keep": 0, "maybe": 0, "reject": 0, "unmarked": 0, "reasons": Counter()}
    for p in parts:
        for k in ("n", "keep", "maybe", "reject", "unmarked"):
            t[k] += p[k]
        t["reasons"].update(p["reasons"])
    return t


def reasons_text(c: Counter) -> str:
    return ", ".join(f"{k} {v}" for k, v in sorted(c.items(), key=lambda kv: (-kv[1], kv[0])))


def grid_title(grid, sample) -> str:
    g = f"Grid {grid}" if grid else ""
    if sample and sample != g:
        return f"{g} " + chr(0xB7) + f" {sample}" if g else sample
    return g or "Grid"


def acq_rows(c: Conv) -> dict:
    out = {}
    for r in (c.acq.get("montages") if isinstance(c.acq.get("montages"), list) else []):
        if isinstance(r, dict) and r.get("montage") is not None:
            out[str(r["montage"])] = r
    return out


def report_data(convs, states) -> list:
    """The numbers behind the report, per conversion (also what the tests check)."""
    out = []
    for c in convs:
        state = states[c.cid]
        acq = acq_rows(c)
        mrows, grids = [], []
        for grid, sample, ms in c.groups():
            gm = []
            for m in ms:
                names = [n for n in m["tiles"] if n in c.tiles]
                tl = tally(names, state)
                nms = {c.tiles[n]["nm"] for n in names if c.tiles[n]["nm"]}
                mrows.append({"montage": m["name"], "grid": grid, "sample": sample, "acq": acq.get(m["name"]),
                              "tile_a_px": (round(next(iter(nms)) * 10, 3) if len(nms) == 1 else None),
                              "mixed_scale": len(nms) > 1, "mark": (state.get(m["name"]) or blank())["flag"],
                              "has_image": bool(m["file"]), **{k: tl[k] for k in ("n",)}})
                gm.append({"montage": m["name"], "grid": grid, "sample": sample, "tally": tl})
            grids.append({"grid": grid, "sample": sample, "montages": gm,
                          "tally": merge_tally(x["tally"] for x in gm)})
        stars, notes = [], []
        for k, e in state.items():
            kind = "tile" if k in c.tiles else "montage"
            t = c.tiles.get(k)
            m = c.montage(t["montage"]) if t else c.montage(k)
            info = {"k": k, "kind": kind, "grid": (t or m or {}).get("grid", ""), "sample": (t or m or {}).get("sample", ""),
                    "montage": m["name"] if m else "", "flag": e["flag"], "stars": e["stars"], "tags": e["tags"],
                    "note": e["note"], "nm": t["nm"] if t else (m or {}).get("nm"),
                    "label": label_of(t) if t else k, "by": e["by"].get("note")}
            if e["stars"] > 0:
                stars.append(info)
            if e["note"]:
                notes.append(info)
        stars.sort(key=lambda s: (-s["stars"], natkey(s["grid"]), natkey(s["k"])))
        notes.sort(key=lambda s: (natkey(s["grid"]), natkey(s["k"])))
        out.append({"conv": c, "montages": mrows, "grids": grids, "stars": stars, "notes": notes,
                    "total": merge_tally(g["tally"] for g in grids)})
    return out


def fmt(v, unit="") -> str:
    if v is None or v == "":
        return "-"
    if isinstance(v, float):
        v = f"{v:.4g}"
    return esc(v) + unit


SESSION_LABELS = {"instrument": "Instrument", "magnification": "Magnification", "pixel_size_a": "Pixel size (\u00c5/px)",
                  "defocus_um": "Defocus (\u00b5m)", "target_defocus_um": "Target defocus (\u00b5m)",
                  "dose": "Dose (e\u207b/\u00c5\u00b2)", "voltage_kv": "Voltage (kV)", "binning": "Binning",
                  "acq_dates": "Acquired", "n_mdoc": "Montage / mdoc files", "n_imaging": "Files at imaging magnification",
                  "n_other": "Files at search/overview magnification only", "other_magnification": "Other magnifications (search / overview)"}


def nice(v) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def day(t) -> str:
    return _text(t, 40).replace("T", " ")[:16]


def session_value(k, v) -> str:
    """One session_acquisition value as plain text: lists joined, {min, median, max} as a range, dates as 'a to b'."""
    if v is None:
        return ""
    if isinstance(v, list):
        return ", ".join(nice(x) + ("\u00d7" if k in ("magnification", "other_magnification") else "") for x in v)
    if isinstance(v, dict):
        if k == "acq_dates":
            a, b = day(v.get("first")), day(v.get("last"))
            return a if a == b or not b else f"{a} to {b}"
        if {"min", "median", "max"} <= set(v):
            return f"{nice(v['min'])} to {nice(v['max'])} (median {nice(v['median'])})"
        return json.dumps(v, ensure_ascii=False)[:200]
    return nice(v)


def session_items(c: Conv) -> list:
    """[(label, text)]: what the report's first block says about the session. The instrument appears once."""
    ses = c.acq.get("session") if isinstance(c.acq.get("session"), dict) else {}
    out = [("Converted", c.created.replace("T", " ")[:16]), ("Recipe", _text(c.meta.get("recipe"), 60)),
           ("Grids", str(c.n_grids())), ("Montages", str(len(c.montages))), ("Tiles", str(len(c.tiles)))]
    scope = c.acq.get("scope_model") or ses.get("instrument")
    if scope:
        out.insert(1, ("Instrument", str(scope)))
    for k, v in list(ses.items())[:30]:
        if k == "instrument" or v is None or v == []:
            continue
        out.append((SESSION_LABELS.get(k) or str(k).replace("_", " ")[:60], session_value(k, v)))
    return out


def render_report(data, url, standalone=False, generated="", who="") -> str:
    """The report body. url(kind, cid, name) -> image src or None (kind ms: small montage copy, rep: starred tile)."""
    sessions = ", ".join(esc(d["conv"].session or d["conv"].cid) for d in data)
    h = [f'<article class="cj-gallery-report"><header class="cj-gallery-rhead"><h1>Imaging run report</h1>'
         f'<p class="cj-mut">Sessions: {sessions}. Generated {esc(generated)}{(" by " + esc(who)) if who else ""}.</p></header>']
    for d in data:
        c, tot = d["conv"], d["total"]
        h.append(f'<section class="cj-gallery-rconv"><h2>Session {esc(c.session or c.cid)} '
                 f'<small class="cj-mut">{esc(c.cid)}</small></h2>')
        # session and acquisition
        kv = session_items(c)
        h.append('<h3>Session and acquisition</h3><dl class="cj-gallery-kv">'
                 + "".join(f"<dt>{esc(k)}</dt><dd>{fmt(v)}</dd>" for k, v in kv) + "</dl>")
        if not c.acq:
            h.append('<p class="cj-mut">No acquisition settings were recorded for this conversion.</p>')
        h.append('<div class="cj-gallery-tw"><table class="cj-gallery-tbl"><caption class="cj-gallery-sr">Acquisition per montage'
                 '</caption><thead><tr><th>Grid</th><th>Montage</th><th>Mag</th><th>Acquisition &Aring;/px</th>'
                 '<th>Tile &Aring;/px</th><th>Defocus <span class="cj-gallery-unit">(&micro;m)</span></th><th>Dose <span class="cj-gallery-unit">(e&#8315;/&Aring;&sup2;)</span></th><th>Acquired</th><th>Tiles</th></tr></thead><tbody>')
        for m in d["montages"]:
            a = m["acq"] or {}
            tile_a = "mixed" if m["mixed_scale"] else fmt(m["tile_a_px"])
            h.append(f'<tr><td>{esc(grid_title(m["grid"], m["sample"]))}</td><td class="cj-wrap">{esc(m["montage"])}</td>'
                     f'<td>{fmt(a.get("mag"))}</td><td>{fmt(a.get("pixel_A"))}</td><td>{tile_a}</td>'
                     f'<td>{fmt(a.get("defocus_um"))}</td><td>{fmt(a.get("dose"))}</td>'
                     f'<td>{fmt(_text(a.get("acquired"), 40).replace("T", " ") if a.get("acquired") else None)}</td>'
                     f'<td>{m["n"]}</td></tr>')
        h.append("</tbody></table></div>")
        # samples per grid
        h.append('<h3>Samples per grid</h3><div class="cj-gallery-tw"><table class="cj-gallery-tbl"><thead><tr><th>Grid</th>'
                 '<th>Sample</th><th>Montages</th><th>Tiles</th></tr></thead><tbody>')
        for g in d["grids"]:
            h.append(f'<tr><td>{esc(g["grid"] or "-")}</td><td>{esc(g["sample"] or "-")}</td>'
                     f'<td>{len(g["montages"])}</td><td>{g["tally"]["n"]}</td></tr>')
        h.append("</tbody></table></div>")
        # counts
        h.append('<h3>Review counts</h3><div class="cj-gallery-tw"><table class="cj-gallery-tbl cj-gallery-counts-tbl">'
                 '<thead><tr><th>Grid / sample</th><th>Montage</th><th>Tiles</th><th>Keep</th><th>Maybe</th>'
                 '<th>Reject</th><th>Unmarked</th><th>Reject reasons</th></tr></thead><tbody>')

        def crow(cls, a, b, t):
            return (f'<tr class="{cls}"><td>{a}</td><td class="cj-wrap">{b}</td>'
                    + "".join(f'<td data-col="{k}">{t[k]}</td>' for k in ("n", "keep", "maybe", "reject", "unmarked"))
                    + f'<td>{esc(reasons_text(t["reasons"])) or "-"}</td></tr>')
        for g in d["grids"]:
            for m in g["montages"]:
                h.append(crow("", esc(grid_title(g["grid"], g["sample"])), esc(m["montage"]), m["tally"]))
            if len(g["montages"]) > 1:
                h.append(crow("cj-gallery-sub", esc(grid_title(g["grid"], g["sample"])), "all montages", g["tally"]))
        h.append(crow("cj-gallery-tot", "Total", "", tot) + "</tbody></table></div>")
        # representative images
        h.append("<h3>Representative images</h3>")
        if d["stars"]:
            h.append('<div class="cj-gallery-figs">')
            for s in d["stars"][:MAX_REP]:
                src = url("rep", c.cid, s["k"]) if s["kind"] == "tile" else url("ms", c.cid, s["k"])
                img = (f'<img src="{esc(src)}" alt="{esc(s["k"])}" loading="lazy">' if src else
                       '<div class="cj-gallery-noimg">image not available</div>')
                tags = f' &middot; {esc(", ".join(s["tags"]))}' if s["tags"] else ""
                h.append(f'<figure class="cj-gallery-fig">{img}<figcaption><b>{esc(grid_title(s["grid"], s["sample"]))}</b> '
                         f'&middot; {esc(s["label"])} &middot; {"&#9733;" * s["stars"]} {s["stars"]}/5'
                         f'{(" &middot; " + esc(s["flag"])) if s["flag"] else ""}{tags}'
                         f'{("<br>" + esc(s["note"])) if s["note"] else ""}</figcaption></figure>')
            h.append("</div>")
            if len(d["stars"]) > MAX_REP:
                h.append(f'<p class="cj-mut">{len(d["stars"]) - MAX_REP} more starred images are not shown.</p>')
        else:
            h.append('<p class="cj-mut">No images starred yet.</p>')
        # notes
        h.append("<h3>Notes</h3>")
        if d["notes"]:
            h.append('<ul class="cj-gallery-notes">')
            for s in d["notes"]:
                by = s["by"] or {}
                h.append(f'<li><b>{esc(grid_title(s["grid"], s["sample"]))} &middot; {esc(s["label"])}</b>'
                         f'{(" (" + esc(s["flag"]) + ")") if s["flag"] else ""}: '
                         f'<span class="cj-gallery-notetext">{esc(s["note"])}</span>'
                         f'<span class="cj-mut"> {esc(by.get("who", ""))} {esc(_text(by.get("t"), 19).replace("T", " "))}</span></li>')
            h.append("</ul>")
        else:
            h.append('<p class="cj-mut">No notes.</p>')
        # montages
        h.append("<h3>Montages</h3>")
        shown = [m for g in c.groups() for m in g[2] if m["file"]]
        if shown:
            h.append('<div class="cj-gallery-figs cj-gallery-mfigs">')
            tl = {m["montage"]: m for m in d["montages"]}
            gt = {m["montage"]: m["tally"] for g in d["grids"] for m in g["montages"]}
            for m in shown:
                t = gt.get(m["name"]) or tally([], {})
                src = url("ms", c.cid, m["name"])
                img = f'<img src="{esc(src)}" alt="Montage {esc(m["name"])}" loading="lazy">' if src else ""
                row = tl.get(m["name"], {})
                h.append(f'<figure class="cj-gallery-fig">{img}<figcaption><b>{esc(m["name"])}</b> '
                         f'&middot; {esc(grid_title(m["grid"], m["sample"]))} &middot; {t["n"]} tiles: {t["keep"]} keep, '
                         f'{t["maybe"]} maybe, {t["reject"]} reject</figcaption></figure>')
            h.append("</div>")
        else:
            h.append('<p class="cj-mut">No stitched montages for this conversion (the gallery works from the tiles).</p>')
        h.append("</section>")
    h.append("</article>")
    return "".join(h)


STANDALONE_TOKENS = """
body.cj-page{--bg:#fff;--fg:#16202a;--mut:#58636f;--card:#fff;--line:#dde3e9;--acc:#1d5f99;--accbg:#e3eef8;--prod:#12703f;
--prodbg:#ddf2e5;--warn:#8a5300;--warnbg:#fbefd9;--cj-rej:#b3261e;--cj-rejbg:#fbe4e2;--cj-font:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
--cj-mono:ui-monospace,Consolas,monospace;--cj-r:10px;margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 var(--cj-font);color-scheme:light}
@media (prefers-color-scheme:dark){body.cj-page{--bg:#0f151b;--fg:#e6ecf1;--mut:#9aa8b5;--card:#172029;--line:#2a3642;--acc:#7db8ec;
--accbg:#16304a;--prod:#6fd49a;--prodbg:#133326;--warn:#f0c06a;--warnbg:#3a2b10;--cj-rej:#ff9b94;--cj-rejbg:#3d1a18;color-scheme:dark}}
@media print{body.cj-page{--bg:#fff;--fg:#000;--mut:#444;--line:#bbb;color-scheme:light}}
.cj-mut{color:var(--mut);font-size:.85rem}.cj-wrap{overflow-wrap:anywhere}
main{max-width:1100px;margin:0 auto;padding:20px 24px 32px}
"""


def standalone_css() -> str:
    try:
        css = CSS_FILE.read_text(encoding="utf-8")
        return STANDALONE_TOKENS + css[css.index(CSS_MARK):css.index("/* cj-gallery:end */")]
    except (OSError, ValueError):
        return STANDALONE_TOKENS


def standalone_html(data, images: dict, generated: str, who: str) -> bytes:
    """report.html: plain HTML + inline CSS, no script, images by relative path (images: (kind, cid, name) -> arcname)."""
    body = render_report(data, lambda kind, cid, name: (lambda a: quote(a, safe="/") if a else None)(images.get((kind, cid, name))),
                         True, generated, who)
    title = "Imaging run report " + ", ".join(d["conv"].session or d["conv"].cid for d in data)
    doc = ('<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
           '<meta name="viewport" content="width=device-width,initial-scale=1"><meta name="color-scheme" content="light dark">'
           f'<title>{esc(title)}</title><style>{standalone_css()}</style></head>'
           f'<body class="cj-page cj-gallery-standalone"><main>{body}</main></body></html>\n')
    return doc.encode("utf-8")


def verdicts_csv(c: Conv, state) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    def cell(v):                                              # a spreadsheet must not run a note as a formula
        v = str(v)
        return "'" + v if v[:1] in ("=", "+", "-", "@", "\t", "\r") else v
    w.writerow(["key", "kind", "montage", "flag", "stars", "reasons", "note"])
    for k in list(c.tiles) + [m["name"] for m in c.montages]:
        e = state.get(k)
        if e:
            w.writerow([cell(x) for x in (k, "tile" if k in c.tiles else "montage",
                                          c.tiles[k]["montage"] if k in c.tiles else k, e["flag"], e["stars"],
                                          "; ".join(e["tags"]), e["note"].replace("\n", " "))])
    return ("﻿" + buf.getvalue()).encode("utf-8")


def report_entries(cfg, data, who: str):
    """[(arcname, bytes)] for the standalone report zip."""
    cache = Path(cfg.state_dir) / "gallery_thumbs"
    images, entries, used = {}, [], set()
    for d in data:
        c = d["conv"]
        need = [("rep", s["k"]) for s in d["stars"][:MAX_REP] if s["kind"] == "tile"] \
            + [("ms", m["name"]) for g in c.groups() for m in g[2] if m["file"]] \
            + [("ms", s["k"]) for s in d["stars"][:MAX_REP] if s["kind"] != "tile" and c.montage_path(s["k"])]
        for kind, name in need:
            if (kind, c.cid, name) in images:
                continue
            if kind == "rep":
                p, t = c.tile_path(name), c.tiles[name]
                b = rep_jpeg(p, t["nm"]) if p and p.is_file() and D.is_contained(p, c.dir) else None
            else:
                p = c.montage_path(name)
                b = small_jpeg(p, cache, SIZES["ms"]) if p and D.is_contained(p, c.dir) else None
            if b:
                arc, n = f"images/{c.cid}/{kind}_{zipstream.safe_arcname(name)}.jpg", 1
                while arc.lower() in used:                     # two names can sanitise to one: keep every reference true
                    n += 1
                    arc = f"images/{c.cid}/{kind}_{zipstream.safe_arcname(name)}_{n}.jpg"
                used.add(arc.lower())
                images[(kind, c.cid, name)] = arc
                entries.append((arc, b))
    when = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return [("report.html", standalone_html(data, images, when, who))] + entries


def images_entries(c: Conv, state, prefix: str):
    ents = []
    for n in c.tiles:
        p = c.tile_path(n)
        if p and p.is_file() and D.is_contained(p, c.dir):
            ents.append((f"{prefix}tiles/{n}", p))
    for m in c.montages:
        p = c.montage_path(m["name"])
        if p and p.is_file() and D.is_contained(p, c.dir):
            ents.append((f"{prefix}montages/{m['file']}", p))
    for name, arc in (("tiles/MANIFEST.csv", "MANIFEST.csv"), ("acquisition.json", "acquisition.json"),
                      ("montages/layout.json", "montages/layout.json")):
        p = c.dir / name
        if p.is_file() and D.is_contained(p, c.dir):
            ents.append((prefix + arc, p))
    ents.append((prefix + "gallery_verdicts.csv", verdicts_csv(c, state)))
    return ents


# ---------------------------------------------------------------- pages
def counts_line(t) -> str:
    return (f'<span class="cj-gallery-k">{t["keep"]} keep</span> ' + chr(0xB7) + f' <span class="cj-gallery-m">{t["maybe"]} maybe</span> '
            + chr(0xB7) + f' <span class="cj-gallery-r">{t["reject"]} reject</span> ' + chr(0xB7) + f' {t["unmarked"]} unmarked of {t["n"]}')


def flag_attrs(e) -> str:
    e = e or blank()
    return f' data-flag="{esc(e["flag"])}" data-stars="{e["stars"]}"'


def tile_html(c: Conv, t: dict, box, state) -> str:
    base = f"{PREFIX}/{c.cid}"
    nm = f' data-nm="{t["nm"]:.6g}"' if t["nm"] else ""
    return (f'<a class="cj-gallery-tile" data-item data-kind="t" data-cid="{c.cid}" data-k="{esc(t["name"])}" '
            f'data-m="{esc(t["montage"])}" href="{base}/t/{u(t["name"])}"{nm}{flag_attrs(state.get(t["name"]))} '
            f'title="{esc(t["name"])}"><img src="{base}/ts/{u(t["name"])}" alt="" loading="lazy" decoding="async">'
            f'<span class="cj-gallery-badge" aria-hidden="true"></span>'
            f'<span class="cj-gallery-tname">{esc(label_of(t, box))}</span></a>')


def montage_html(c: Conv, m: dict, state) -> str:
    base = f"{PREFIX}/{c.cid}"
    tl = tally([n for n in m["tiles"] if n in c.tiles], state)
    out = [f'<div class="cj-gallery-mont" data-cid="{c.cid}" data-m="{esc(m["name"])}">'
           f'<h4 class="cj-gallery-mh"><span class="cj-wrap">{esc(m["name"])}</span> '
           f'<span class="cj-gallery-counts">{counts_line(tl)}</span>']
    if m["file"]:                      # the tile boxes cover the image, so full size is a link of its own
        nm = f' data-nm="{m["nm"]:.6g}"' if m["nm"] else ""
        out.append(f' <a class="cj-gallery-full" data-item data-kind="m" data-cid="{c.cid}" data-k="{esc(m["name"])}" '
                   f'href="{base}/m/{u(m["name"])}"{nm}{" data-sb" if m["sb"] else ""}'
                   f'{flag_attrs(state.get(m["name"]))} title="Montage {esc(m["name"])}">Full size</a>')
    out.append("</h4>")
    boxes = {b["k"]: b for b in m["boxes"]}
    if m["file"] and m["w"] and m["h"]:
        out.append(f'<div class="cj-gallery-stage" style="aspect-ratio:{m["w"]:.0f}/{m["h"]:.0f}">'
                   f'<a class="cj-gallery-mimg" data-kind="m" data-cid="{c.cid}" data-k="{esc(m["name"])}" '
                   f'href="{base}/m/{u(m["name"])}"{flag_attrs(state.get(m["name"]))} title="Montage {esc(m["name"])}">'
                   f'<img src="{base}/mp/{u(m["name"])}" alt="Montage {esc(m["name"])}" loading="lazy" decoding="async"></a>')
        for b in m["boxes"]:
            t = c.tiles[b["k"]]
            tnm = f' data-nm="{t["nm"]:.6g}"' if t["nm"] else ""
            out.append(f'<a class="cj-gallery-box" data-kind="t" data-cid="{c.cid}" data-k="{esc(b["k"])}" href="{base}/t/{u(b["k"])}"{tnm} '
                       f'style="left:{100 * b["x"] / m["w"]:.3f}%;top:{100 * b["y"] / m["h"]:.3f}%;'
                       f'width:{100 * b["w"] / m["w"]:.3f}%;height:{100 * b["h"] / m["h"]:.3f}%"'
                       f'{flag_attrs(state.get(b["k"]))} title="{esc(label_of(t, b))}" aria-label="Tile {esc(label_of(t, b))}"></a>')
        out.append("</div>")
    out.append('<div class="cj-gallery-strip">')
    for n in m["tiles"]:
        if n in c.tiles:
            out.append(tile_html(c, c.tiles[n], boxes.get(n), state))
    out.append("</div></div>")
    return "".join(out)


def conv_gallery_html(c: Conv, state, heading: bool) -> str:
    out = [f'<section class="cj-gallery-conv" data-cid="{c.cid}">']
    if heading:
        out.append(f'<h2 class="cj-gallery-ch">Session {esc(c.session or c.cid)} <small class="cj-mut">{esc(c.cid)}</small></h2>')
    if not c.tiles:
        out.append('<p class="cj-mut">This conversion has no tiles yet.</p>')
    elif c.stitch and c.stitch.get("state") == "failed":
        out.append('<p class="cj-mut">Montages could not be stitched for this conversion; the tiles are shown by montage name.</p>')
    for grid, sample, ms in c.groups():
        out.append(f'<section class="cj-gallery-group" data-g="{esc(grid)}" data-s="{esc(sample)}">'
                   f'<h3 class="cj-gallery-gh">{esc(grid_title(grid, sample))}</h3>')
        out += [montage_html(c, m, state) for m in ms]
        out.append("</section>")
    out.append("</section>")
    return "".join(out)


def options(values, label_all) -> str:
    return f'<option value="">{esc(label_all)}</option>' + "".join(
        f'<option value="{esc(v)}">{esc(v)}</option>' for v in values)


def analyse_kept_html(c: Conv, state, src_key) -> str:
    """Owner / admin (the page is only served to them): the form to Analyse kept tiles (N), N >= 1 flagged keep.
    Goes to the session form, which re-derives the list from events.jsonl on the server. Hidden otherwise."""
    t = tally(list(c.tiles), state)
    if not src_key or t["keep"] < 1:
        return ""
    return (f'<form class="cj-gallery-acts cj-gallery-analyse" method="get" action="/predict/new/session/{esc(src_key)}/">'
            f'<input type="hidden" name="which" value="kept"><input type="hidden" name="from" value="{esc(c.cid)}">'
            f'<button class="cj-gallery-btn" type="submit">Analyse kept tiles ({t["keep"]})</button>'
            + (f' <label class="cj-mut"><input type="checkbox" name="kept_maybe" value="1"> also include maybe '
               f'({t["maybe"]})</label>' if t["maybe"] else "") + '</form>')


def page_gallery(convs, states, who, cids_q: str, compare: bool, analyse_key=None) -> str:
    grids = sorted({g for c in convs for g, _, _ in c.groups() if g}, key=natkey)
    samples = sorted({s for c in convs for _, s, _ in c.groups() if s}, key=natkey)
    tot = merge_tally(tally(list(c.tiles), states[c.cid]) for c in convs)
    names = ", ".join(c.session or c.cid for c in convs)
    qs = "&".join(f"c={c.cid}" for c in convs)
    base = f"{PREFIX}/{convs[0].cid}" if len(convs) == 1 else f"{PREFIX}/compare"
    suffix = "" if len(convs) == 1 else f"?{qs}"
    rep = f"{base}/report{suffix}"
    zipq = f"{base}/images.zip{suffix}"
    cids = " ".join(c.cid for c in convs)
    head = (f'<div class="cj-head"><div><h1>{"Compare: " if compare else "Gallery: "}{esc(names)}</h1>'
            f'<p class="cj-mut">Click a tile for the full-size view. Keys there: &larr; &rarr; next / previous, K keep, M maybe, '
            f'R reject, 0&ndash;5 stars, T reasons, N note, Esc close.</p></div>'
            f'<p class="cj-gallery-acts"><a class="cj-gallery-btn" href="{esc(rep)}">Report</a> '
            f'<a class="cj-gallery-btn" href="{esc(zipq)}">Download images (zip)</a> '
            f'<a class="cj-gallery-btn" href="{PREFIX}/">All conversions</a></p></div>')
    head += analyse_kept_html(convs[0], states[convs[0].cid], analyse_key) if len(convs) == 1 else ""
    bar = ('<div class="cj-gallery-bar"><div class="cj-gallery-filters" role="group" aria-label="Filters">'
           '<label>Mark <select data-filter="flag"><option value="">All</option><option value="keep">Keep</option>'
           '<option value="maybe">Maybe</option><option value="reject">Reject</option>'
           '<option value="none">Unmarked</option></select></label>'
           f'<label>Sample <select data-filter="sample">{options(samples, "All")}</select></label>'
           f'<label>Grid <select data-filter="grid">{options(grids, "All")}</select></label>'
           f'<label>Reason <select data-filter="tag">{options(TAGS, "All")}</select></label>'
           '<button type="button" class="cj-gallery-btn" data-filter-reset hidden>Clear filters</button></div>'
           f'<p class="cj-gallery-sum" id="cj-gallery-sum" aria-live="polite">{counts_line(tot)}</p></div>')
    body = "".join(conv_gallery_html(c, states[c.cid], len(convs) > 1) for c in convs)
    cls = "cj-gallery-cols" if len(convs) > 1 else ""
    return (f'<div class="cj-gallery" id="cj-gallery" data-cids="{esc(cids)}" data-tags="{esc(json.dumps(list(TAGS)))}">'
            f'{head}{bar}<p class="cj-gallery-msg" id="cj-gallery-msg" role="status"></p>'
            f'<div class="{cls}">{body}</div></div><script src="{PREFIX}/gallery.js" defer></script>')


def page_index(rows, admin: bool, show_all: bool) -> str:
    h = ['<div class="cj-gallery"><div class="cj-head"><div><h1>Gallery</h1><p class="cj-mut">Your converted raw sessions: '
         'browse the montages, mark tiles, read the report. Tick two or more to compare them side by side.</p></div>']
    if admin:
        h.append('<p class="cj-gallery-acts"><a class="cj-gallery-btn" href="' + PREFIX + '/"' + (' aria-current="true"' if not show_all else "")
                 + '>Mine</a> <a class="cj-gallery-btn" href="' + PREFIX + '/?all=1"' + (' aria-current="true"' if show_all else "")
                 + ">All</a></p>")
    h.append("</div>")
    if not rows:
        h.append('<p>No conversions yet. Convert raw files under <a href="/predict/new/raw/">Predict</a> first.</p></div>')
        return "".join(h)
    h.append(f'<form method="get" action="{PREFIX}/compare"><div class="cj-gallery-tw"><table class="cj-gallery-tbl">'
             '<thead><tr><th><span class="cj-gallery-sr">Compare</span></th><th>Session</th><th>Converted</th><th>Grids</th>'
             '<th>Montages</th><th>Tiles</th><th>Backup</th><th></th></tr></thead><tbody>')
    for c in rows:
        ready = bool(c.tiles)
        btxt, tone = c.backup()
        state = _text(c.status.get("state"), 20)
        link = (f'<a href="{PREFIX}/{c.cid}/">Gallery</a> &middot; <a href="{PREFIX}/{c.cid}/report">Report</a>' if ready
                else f'<span class="cj-mut">{esc(state or "no tiles yet")}</span>')
        pick = (f'<input type="checkbox" name="c" value="{c.cid}" aria-label="Compare {esc(c.session or c.cid)}">'
                if ready else "")
        owner = f' <span class="cj-mut">{esc(c.owner)}</span>' if admin and show_all else ""
        h.append(f'<tr><td>{pick}</td><td><b>{esc(c.session or c.cid)}</b> <span class="cj-mut cj-wrap">{esc(c.cid)}</span>{owner}</td>'
                 f'<td>{esc(c.created.replace("T", " ")[:16])}</td><td>{c.n_grids()}</td><td>{len(c.montages)}</td>'
                 f'<td>{len(c.tiles)}</td><td class="cj-gallery-b-{tone}">{esc(btxt)}</td><td>{link}</td></tr>')
    h.append('</tbody></table></div><p><button type="submit" class="cj-gallery-btn">Compare selected</button></p></form></div>')
    return "".join(h)


def page_report(convs, data, who, convs_q: str) -> str:
    base = f"{PREFIX}/{convs[0].cid}" if len(convs) == 1 else f"{PREFIX}/compare"
    suffix = "" if len(convs) == 1 else "?" + "&".join(f"c={c.cid}" for c in convs)
    gal = f"{base}/" if len(convs) == 1 else f"{base}{suffix}"

    def url(kind, cid, name):
        return f"{PREFIX}/{cid}/{'rep' if kind == 'rep' else 'ms'}/{u(name)}"
    bar = (f'<p class="cj-gallery-acts cj-gallery-noprint"><a class="cj-gallery-btn" href="{esc(gal)}">Back to the gallery</a> '
           f'<a class="cj-gallery-btn" href="{esc(base)}/report.zip{suffix}">Download report (zip)</a> '
           f'<a class="cj-gallery-btn" href="{esc(base)}/images.zip{suffix}">Download images (zip)</a> '
           f'<button type="button" class="cj-gallery-btn" data-print>Print</button></p>')
    when = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return (f'<div class="cj-gallery">{bar}{render_report(data, url, False, when, who)}</div>'
            f'<script src="{PREFIX}/gallery.js" defer></script>')


# ---------------------------------------------------------------- zip slots (2 per person, per portal process)
# The counts live in this process: the portal runs as several services, each with its own limit, which is enough to
# keep one person from tying up one service. A slot is released by SlotResponse when the response ends, fails or the
# client goes away; a slot whose stream pulled no chunk for ZIP_STALE_S is ignored (the safety net, not the mechanism).
_ZIPS: dict = {}
_ZIPS_LOCK = threading.Lock()


class Slot:
    """One running download of a person."""

    def __init__(self, who):
        self.who, self.t, self.done = who, time.time(), False

    def release(self):
        self.done = True              # lock-free on purpose: take_slot drops done slots, nothing here can deadlock


def take_slot(who):
    now = time.time()
    with _ZIPS_LOCK:
        live = [s for s in _ZIPS.get(who, []) if not s.done and now - s.t < ZIP_STALE_S]
        if len(live) >= MAX_ZIPS:
            _ZIPS[who] = live
            return None
        s = Slot(who)
        _ZIPS[who] = live + [s]
        return s


class Progress:
    """Wraps the zip iterator: every pulled chunk shows the slot is alive."""

    def __init__(self, it, slot):
        self.it, self.slot = it, slot

    def __iter__(self):
        return self

    def __next__(self):
        self.slot.t = time.time()
        return next(self.it)

    def close(self):
        getattr(self.it, "close", lambda: None)()


class SlotResponse(StreamingResponse):
    """A StreamingResponse that frees its slot when the response ends, however it ends (done, error, disconnect, cancel)."""

    def __init__(self, content, slot, **kw):
        super().__init__(content, **kw)
        self.slot = slot

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            self.slot.release()


# ---------------------------------------------------------------- routes
def routes(ctx) -> list:
    cfg = ctx.cfg
    cache_dir = Path(cfg.state_dir) / "gallery_thumbs"

    def gone(request):
        r = ctx.page(request, "Not found", f'<div class="cj-gallery"><h1>Not found</h1><p>{esc(GONE_TEXT)}</p>'
                                           f'<p><a href="{PREFIX}/">My conversions</a></p></div>', active="gallery")
        r.status_code = 404
        return r

    def gone_json():
        return ctx.json({"error": "Not found"}, 404)

    def gone_plain():
        return ctx.no_store(PlainTextResponse("Not found", status_code=404))

    def conv(request, cid):
        return load_conv(cfg, ctx.user(request), ctx.can_see, cid)

    def conv_set(request, cids):
        """The conversions of a compare set, or None when any one is missing or not this person's."""
        out, seen = [], set()
        for cid in cids:
            if cid in seen:
                continue
            seen.add(cid)
            c = conv(request, cid)
            if c is None:
                return None
            out.append(c)
        return out if 1 <= len(out) <= MAX_COMPARE else None

    def over(request):
        return len(set(request.query_params.getlist("c"))) > MAX_COMPARE

    def too_many(request, page=True):
        text = f"Compare at most {MAX_COMPARE} conversions at a time."
        if not page:
            return ctx.no_store(PlainTextResponse(text, status_code=400))
        r = ctx.page(request, "Too many", f'<div class="cj-gallery"><h1>Too many</h1><p>{esc(text)}</p>'
                                          f'<p><a href="{PREFIX}/">My conversions</a></p></div>', active="gallery")
        r.status_code = 400
        return r

    def states_of(convs):
        return {c.cid: state_of(c) for c in convs}

    def who_of(request):
        return str(ctx.user(request).get("id") or "")

    # ---- pages
    def index(request: Request):
        user = ctx.user(request)
        admin = user.get("role") == "admin"
        show_all = admin and request.query_params.get("all") == "1"
        rows = [Conv(d, m, s) for d, m, s in P.list_conversions(cfg, user, ctx.can_see)
                if show_all or m.get("owner") == user.get("id")]
        return ctx.page(request, "Gallery", page_index(rows, admin, show_all), active="gallery")

    def gallery(request: Request):
        convs = conv_set(request, [request.path_params["cid"]])
        if convs is None:
            return gone(request)
        c0 = convs[0]
        key = next((x["key"] for x in P.converted_sources(cfg, ctx.user(request), ctx.can_see)
                    if x.get("conversion") == c0.cid and not x["problem"]), None)
        return ctx.page(request, f"Gallery {c0.session or c0.cid}",
                        page_gallery(convs, states_of(convs), who_of(request), "", False, key), active="gallery")

    def compare(request: Request):
        if over(request):
            return too_many(request)
        convs = conv_set(request, request.query_params.getlist("c"))
        if convs is None:
            return gone(request)
        return ctx.page(request, "Compare", page_gallery(convs, states_of(convs), who_of(request), "", True), active="gallery")

    def report_for(request, cids):
        convs = conv_set(request, cids)
        if convs is None:
            return None, None
        return convs, report_data(convs, states_of(convs))

    def report(request: Request):
        cid = request.path_params.get("cid")
        if not cid and over(request):
            return too_many(request)
        convs, data = report_for(request, [cid] if cid else request.query_params.getlist("c"))
        if convs is None:
            return gone(request)
        return ctx.page(request, "Report " + ", ".join(c.session or c.cid for c in convs),
                        page_report(convs, data, who_of(request), ""), active="gallery")

    # ---- zips
    def zip_response(name, entries_fn, request):
        slot = take_slot(who_of(request))
        if slot is None:
            r = ctx.no_store(PlainTextResponse("Two downloads are already running for you. Wait for one to finish.",
                                               status_code=429))
            r.headers["Retry-After"] = "30"
            return r
        try:
            stream = Progress(zipstream.stream_zip(entries_fn()), slot)
        except BaseException:
            slot.release()
            raise
        r = SlotResponse(stream, slot, media_type="application/zip")
        r.headers["Content-Disposition"] = f'attachment; filename="{name}"'
        r.headers["Cache-Control"] = "no-store"
        return r

    def label_for(convs):
        return convs[0].cid if len(convs) == 1 else "compare_" + "_".join(c.cid for c in convs)[:80]

    def report_zip(request: Request):
        cid = request.path_params.get("cid")
        if not cid and over(request):
            return too_many(request, False)
        convs, data = report_for(request, [cid] if cid else request.query_params.getlist("c"))
        if convs is None:
            return gone_plain()
        return zip_response(f"gallery_report_{label_for(convs)}.zip", lambda: report_entries(cfg, data, who_of(request)),
                            request)

    def images_zip(request: Request):
        cid = request.path_params.get("cid")
        if not cid and over(request):
            return too_many(request, False)
        convs = conv_set(request, [cid] if cid else request.query_params.getlist("c"))
        if convs is None:
            return gone_plain()

        def entries():
            out = []
            for c in convs:
                out += images_entries(c, state_of(c), "" if len(convs) == 1 else f"{c.cid}/")
            return out
        return zip_response(f"gallery_images_{label_for(convs)}.zip", entries, request)

    # ---- images
    def image(kind):
        def handler(request: Request):
            c = conv(request, request.path_params["cid"])
            name = request.path_params["name"]
            if c is None:
                return gone_plain()
            if kind in ("t", "ts", "rep"):
                path = c.tile_path(name)
            else:
                path = c.montage_path(name)
            if path is None or not path.is_file() or not D.is_contained(path, c.dir):
                return gone_plain()
            if kind in ("t", "m"):
                r = FileResponse(path)
            else:
                if kind == "ts":
                    data = D.make_thumb(path, Path(cfg.state_dir) / "data_thumbs")
                elif kind == "rep":
                    data = rep_jpeg(path, c.tiles[name]["nm"])
                else:
                    data = small_jpeg(path, cache_dir, SIZES[kind])
                if data is None:
                    return gone_plain()
                r = Response(data, media_type="image/jpeg")
            r.headers["Cache-Control"] = "private, max-age=300"
            return r
        return handler

    # ---- state and events
    def api_state(request: Request):
        c = conv(request, request.path_params["cid"])
        if c is None:
            return gone_json()
        return ctx.json({"cid": c.cid, "state": state_of(c), "tags": list(TAGS)})

    async def api_event(request: Request):
        if request.headers.get("x-requested-with") != "gallery":
            return ctx.json({"error": "Not allowed"}, 403)
        c = await run_in_threadpool(conv, request, request.path_params["cid"])
        if c is None:
            return gone_json()
        try:
            declared = int(request.headers.get("content-length") or 0)
        except ValueError:
            declared = BODY_MAX + 1
        if declared > BODY_MAX:                                  # refused before a byte of it is read
            return ctx.json({"error": "Too large"}, 413)
        body = b""
        async for part in request.stream():
            body += part
            if len(body) > BODY_MAX:
                return ctx.json({"error": "Too large"}, 413)
        try:
            doc = json.loads(body.decode("utf-8"))
            k, f, v = doc["k"], doc["f"], doc.get("v")
            if not isinstance(doc, dict) or not isinstance(k, str) or not isinstance(f, str):
                raise ValueError("bad request")
        except (ValueError, KeyError, TypeError, UnicodeDecodeError):
            return ctx.json({"error": "That request could not be read."}, 400)
        try:
            entry = await run_in_threadpool(append_event, c, who_of(request), k, f, v)
        except ValueError as e:
            return ctx.json({"error": str(e)}, 400)
        except OSError as e:
            if e.errno == errno.ENOSPC:
                return ctx.json({"error": "This gallery has reached its size limit; ask the admin."}, 507)
            return ctx.json({"error": "Could not save. Try again in a moment."}, 503)
        return ctx.json({"ok": True, "k": k, "entry": entry})

    def script(request: Request):
        r = Response(SCRIPT.read_text(encoding="utf-8"), media_type="text/javascript; charset=utf-8")
        r.headers["Cache-Control"] = "no-cache"
        return r

    return [Route(PREFIX + "/", index, methods=["GET"]),
            Route(PREFIX + "/gallery.js", script, methods=["GET"]),
            Route(PREFIX + "/compare", compare, methods=["GET"]),
            Route(PREFIX + "/compare/report", report, methods=["GET"]),
            Route(PREFIX + "/compare/report.zip", report_zip, methods=["GET"]),
            Route(PREFIX + "/compare/images.zip", images_zip, methods=["GET"]),
            Route(PREFIX + "/{cid}/", gallery, methods=["GET"]),
            Route(PREFIX + "/{cid}/report", report, methods=["GET"]),
            Route(PREFIX + "/{cid}/report.zip", report_zip, methods=["GET"]),
            Route(PREFIX + "/{cid}/images.zip", images_zip, methods=["GET"]),
            Route(PREFIX + "/{cid}/api/state", api_state, methods=["GET"]),
            Route(PREFIX + "/{cid}/api/event", api_event, methods=["POST"]),
            Route(PREFIX + "/{cid}/t/{name}", image("t"), methods=["GET"]),
            Route(PREFIX + "/{cid}/ts/{name}", image("ts"), methods=["GET"]),
            Route(PREFIX + "/{cid}/rep/{name}", image("rep"), methods=["GET"]),
            Route(PREFIX + "/{cid}/m/{name}", image("m"), methods=["GET"]),
            Route(PREFIX + "/{cid}/ms/{name}", image("ms"), methods=["GET"]),
            Route(PREFIX + "/{cid}/mp/{name}", image("mp"), methods=["GET"])]
