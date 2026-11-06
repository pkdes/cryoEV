"""
Predict's session list (CJ portal): every processed imaging session that has model-ready tiles, found by a scan,
plus the admin's hand list (CJ_PREDICT_SOURCES, back-compat), with the admin's per-source overrides (hide, rename).

Discovery (read only; nothing under a scan root is ever written):
  CJ_PREDICT_SCAN_ROOTS   folders to scan, os.pathsep-separated (a Config attribute of the same lower-case name wins;
                          a list works there too). Unset: the processed-data root data_utils/config.py knows
                          (processing_root()). "off" (or "none") scans nothing.
  A session folder is a direct child of a root named YYYYMMDD or YYYYMMDD-N (anything after a space or "_" is
  ignored). Its model-ready tiles are `3. Processed images/3d. Tiles no scalebar*` (no scalebar, 8-bit PNG/JPEG).
  Compatible = that folder holds at least one image and a MANIFEST.csv with an image-name column and a nm_per_px
  column with at least one readable value for those images (and a PNG tile, when there is one, is 8-bit).
  Anything else is listed for the admin with the reason and is never offered to people. Session folders without a
  3d folder are listed as "no model-ready tiles".
  The session is the folder's date (20260827-2 -> session 20260827, label suffix "-2"); the default label is
  "Session 27 Aug 2026" (+ " -2").
  The scan is cached for CACHE_TTL_S, keyed on the roots and their mtimes; one scan at a time (callers that miss
  the cache while a scan runs wait for it). Blocking: call it off the event loop (every caller in predict.py and
  admin.py runs in a worker thread).

Overrides: <state dir>/predict_sources.json
    {"schema": "cj_predict_sources/1", "sources": {"<key>": {"hidden": bool, "label": str, "changed": iso,
                                                             "by": portal id}}}
  key = sha1(normcase(str(dir)))[:10], as predict.sources() always made it. Default: shown, default label. Every
  change is one read-modify-write under a thread lock plus an OS lock (<file>.lock, data_utils/oslock.py), written
  atomically (temp file + os.replace, retried briefly). An unreadable overrides file raises ValueError: Predict then
  offers nothing (never shows a hidden session by accident) and the admin sees why; it is never overwritten.

Merging: the hand list first (its order, its fields win: label, session, sample_column, manifest), then the
discovered sources not on it, newest session first; de-duplicated by key. Overrides apply to both. A hidden source
stays in predict.sources() with problem = HIDDEN, so Predict shows it to the admin only ("Not available") and
never lets anyone start an analysis on it.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from data_utils.oslock import acquire_lock
from jobs import predict_set as PS

SCAN_SETTING = "CJ_PREDICT_SCAN_ROOTS"
LIST_SETTING = "CJ_PREDICT_SOURCES"
SCAN_OFF = ("off", "none")
PROCESSED = "3. Processed images"
TILES_PREFIX = "3d. Tiles no scalebar"
MANIFEST = "MANIFEST.csv"
IMAGE_EXT = (".png", ".jpg", ".jpeg")
SAMPLE_COLUMNS = ("sample_label", "sample", "grid")          # as predict.py
SESSION_DIR_RE = re.compile(r"^(\d{8})(-\d+)?(?:[ _].*)?$")
KEY_RE = re.compile(r"^[0-9a-f]{10}$")
MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
CACHE_TTL_S = 60.0
OVERRIDES = "predict_sources.json"
SCHEMA = "cj_predict_sources/1"
LOCK_WAIT_S = 5.0
REPLACE_TRIES = 20                                          # os.replace / read retries, 50 ms apart (Windows)
LABEL_MAX = 80
LABEL_BAD = re.compile(r"[\x00-\x1f\x7f]")
HIDDEN = "hidden by the admin (Admin, Predict sessions)"

_scan_lock = threading.Lock()
_cache: dict = {}
_write_lock = threading.Lock()


# ---------------------------------------------------------------- settings
def _setting(cfg, name):
    v = getattr(cfg, name.lower(), None)
    return v if v not in (None, "") else os.environ.get(name.upper())


def source_key(d) -> str:
    return hashlib.sha1(os.path.normcase(str(d)).encode("utf-8")).hexdigest()[:10]


def scan_roots(cfg) -> list:
    """-> [Path]: CJ_PREDICT_SCAN_ROOTS, else the processing root data_utils/config.py finds, else []."""
    v = _setting(cfg, SCAN_SETTING)
    if v in (None, ""):
        from data_utils import config
        p = config.processing_root()
        return [p] if p else []
    if isinstance(v, (list, tuple)):
        parts = [str(x) for x in v]
    else:
        s = str(v).strip()
        if s.lower() in SCAN_OFF:
            return []
        parts = s.split(os.pathsep)
    return list(dict.fromkeys(Path(p.strip()) for p in parts if p.strip()))


def session_of(folder: str):
    """'20260827-2' -> ('20260827', '-2'); not a session folder (or not a real date) -> None."""
    m = SESSION_DIR_RE.match(folder)
    if not m:
        return None
    try:
        datetime.strptime(m.group(1), "%Y%m%d")
    except ValueError:
        return None
    return m.group(1), m.group(2) or ""


def default_label(session: str, suffix: str = "") -> str:
    d = datetime.strptime(session, "%Y%m%d").date()
    return f"Session {d.day} {MONTHS[d.month - 1]} {d.year}" + (f" {suffix}" if suffix else "")


# ---------------------------------------------------------------- one tile folder
def _num(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) and x > 0 else None


def png_bit_depth(path) -> int | None:
    """The IHDR bit depth of a PNG (8 for the lab's tiles), None when it can't be read."""
    try:
        with open(path, "rb") as f:
            head = f.read(26)
    except OSError:
        return None
    if len(head) < 26 or head[:8] != b"\x89PNG\r\n\x1a\n" or head[12:16] != b"IHDR":
        return None
    return head[24]


def check_folder(d: Path, manifest: Path | None = None, sample_column=None) -> dict:
    """-> {tiles, samples, nm_min, nm_max, compatible, reason}. Reads only."""
    out = {"tiles": 0, "samples": None, "nm_min": None, "nm_max": None, "compatible": False, "reason": None}
    try:
        names = sorted(e.name for e in os.scandir(d) if e.is_file() and not e.name.startswith(".")
                       and os.path.splitext(e.name)[1].lower() in IMAGE_EXT)
    except OSError as e:
        out["reason"] = f"folder can't be read ({type(e).__name__})"
        return out
    out["tiles"] = len(names)
    man = manifest or (d / MANIFEST)
    if not names:
        out["reason"] = "no PNG or JPEG images"
        return out
    if not man.is_file():
        out["reason"] = "no MANIFEST.csv"
        return out
    try:
        with open(man, newline="", encoding="utf-8-sig") as f:
            rd = csv.DictReader(f)
            head = list(rd.fieldnames or [])
            rows = list(rd)
    except (OSError, UnicodeDecodeError, csv.Error) as e:
        out["reason"] = f"MANIFEST.csv can't be read ({type(e).__name__})"
        return out
    key = next((c for c in PS.NAME_COLUMNS if c in head), None)
    if key is None:
        out["reason"] = "MANIFEST.csv has no image name column"
        return out
    if PS.NM_COLUMN not in head:
        out["reason"] = f"MANIFEST.csv has no {PS.NM_COLUMN} column"
        return out
    have = set(names)
    mine = [r for r in rows if r.get(key) in have]
    nms = [x for x in (_num(r.get(PS.NM_COLUMN)) for r in mine) if x is not None]
    col = sample_column if sample_column in head else next((c for c in SAMPLE_COLUMNS if c in head), None)
    if col:
        out["samples"] = len({str(r.get(col) or "").strip() for r in mine} - {""})
    if not nms:
        out["reason"] = "MANIFEST.csv gives no readable nm/px for these images"
        return out
    out["nm_min"], out["nm_max"] = min(nms), max(nms)
    png = next((n for n in names if n.lower().endswith(".png")), None)
    if png is not None:
        bits = png_bit_depth(d / png)
        if bits is not None and bits != 8:
            out["reason"] = f"tiles are {bits}-bit PNG, not 8-bit"
            return out
    out["compatible"] = True
    return out


# ---------------------------------------------------------------- discovery
def _mtime(p: Path):
    try:
        return p.stat().st_mtime_ns
    except OSError:
        return None


def scan(roots) -> dict:
    """{"found": [source], "skipped": [{folder, root, reason}], "roots": [{root, problem}]}. Never raises."""
    found, skipped, rinfo = [], [], []
    for root in roots:
        root = Path(root)
        try:
            kids = sorted((e.name for e in os.scandir(root) if e.is_dir()), reverse=True)
        except OSError as e:
            rinfo.append({"root": root, "problem": "folder not found" if isinstance(e, FileNotFoundError)
                          else f"can't be read ({type(e).__name__})"})
            continue
        rinfo.append({"root": root, "problem": None})
        for name in kids:
            ses = session_of(name)
            if ses is None:
                continue
            session, suffix = ses
            proc = root / name / PROCESSED
            try:
                tiles = sorted(e.name for e in os.scandir(proc) if e.is_dir() and e.name.startswith(TILES_PREFIX))
            except OSError:
                tiles = []
            if not tiles:
                skipped.append({"folder": name, "root": root, "reason": "no model-ready tiles (no 3d folder)"})
                continue
            for sub in tiles:
                d = proc / sub
                label = default_label(session, suffix)
                if len(tiles) > 1:
                    tail = sub[len(TILES_PREFIX):].strip(" -") or sub
                    label += f" ({tail})"
                chk = check_folder(d)
                man = d / MANIFEST
                found.append({"key": source_key(d), "dir": d, "manifest": man if man.is_file() else None,
                              "session": session, "suffix": suffix, "folder": name, "subfolder": sub, "root": root,
                              "default_label": label, "sample_column": None, **chk})
    found.sort(key=lambda s: (s["session"], s["suffix"], s["subfolder"]), reverse=True)
    return {"found": found, "skipped": skipped, "roots": rinfo}


def discover(cfg, fresh=False) -> dict:
    """scan(scan_roots(cfg)), cached for CACHE_TTL_S per (roots, root mtimes). fresh=True rescans (and caches)."""
    roots = scan_roots(cfg)
    ck = tuple(os.path.normcase(str(r)) for r in roots)
    with _scan_lock:
        sig = tuple(_mtime(r) for r in roots)
        hit = _cache.get(ck)
        if not fresh and hit and hit[1] == sig and 0 <= time.monotonic() - hit[0] < CACHE_TTL_S:
            res = hit[2]
        else:
            res = scan(roots)
            _cache.clear()                               # one config per portal: keep a single entry
            _cache[ck] = (time.monotonic(), sig, res)
    return {"found": [dict(s) for s in res["found"]], "skipped": [dict(s) for s in res["skipped"]],
            "roots": [dict(r) for r in res["roots"]]}


def clear_cache() -> None:
    with _scan_lock:
        _cache.clear()


# ---------------------------------------------------------------- the hand list (CJ_PREDICT_SOURCES)
def listed(cfg) -> list:
    """The admin's hand list exactly as predict.sources() read it before discovery (same fields, same problems).
    Raises OSError / ValueError on an unreadable setting, as before."""
    raw = _setting(cfg, LIST_SETTING)
    if not raw:
        return []
    if isinstance(raw, (list, tuple)):
        items = list(raw)
    else:
        text = str(raw).strip()
        if not text.startswith("["):
            text = Path(text).read_text(encoding="utf-8-sig")
        items = json.loads(text)
    out = []
    for it in items if isinstance(items, list) else []:
        if not isinstance(it, dict) or not it.get("dir"):
            continue
        d = Path(str(it["dir"]))
        man = Path(str(it["manifest"])) if it.get("manifest") else (d / "MANIFEST.csv")
        session = str(it.get("session") or "")
        if not session:
            m = re.search(r"(?<!\d)(\d{8})(?!\d)", d.name) or re.search(r"(?<!\d)(\d{8})(?!\d)", str(d))
            session = m.group(1) if m else ""
        problem = None
        if not d.is_dir():
            problem = "folder not found"
        elif not PS.SESSION_RE.match(session):
            problem = 'no session date: add "session": "YYYYMMDD" to this source'
        out.append({"key": source_key(d), "label": str(it.get("label") or d.name), "dir": d,
                    "manifest": man if man.is_file() else None, "session": session,
                    "sample_column": it.get("sample_column"), "problem": problem})
    return out


# ---------------------------------------------------------------- overrides
def overrides_path(state_dir) -> Path:
    return Path(state_dir) / OVERRIDES


def _read_text(path: Path) -> str:
    for i in range(REPLACE_TRIES):
        try:
            return path.read_text(encoding="utf-8")
        except PermissionError:                          # a writer's os.replace in progress (Windows)
            if i == REPLACE_TRIES - 1:
                raise
            time.sleep(0.05)
    raise OSError("unreachable")


def load_overrides(state_dir) -> dict:
    """-> {key: {hidden, label, ...}}. No state dir or no file: {}. Unreadable or not this shape: ValueError."""
    if state_dir is None:
        return {}
    path = overrides_path(state_dir)
    try:
        doc = json.loads(_read_text(path))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as e:
        raise ValueError(f"{OVERRIDES} can't be read ({type(e).__name__})")
    if not isinstance(doc, dict) or not isinstance(doc.get("sources"), dict):
        raise ValueError(f"{OVERRIDES} has no sources table")
    return {k: v for k, v in doc["sources"].items() if isinstance(v, dict)}


def _os_replace(src, dst):
    os.replace(src, dst)


def _write(path: Path, table: dict) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        tmp.write_text(json.dumps({"schema": SCHEMA, "sources": table}, indent=1, sort_keys=True), encoding="utf-8")
        for i in range(REPLACE_TRIES):
            try:
                return _os_replace(tmp, path)
            except PermissionError:
                if i == REPLACE_TRIES - 1:
                    raise
                time.sleep(0.05)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


@contextmanager
def overrides_lock(state_dir, wait=LOCK_WAIT_S):
    """A thread lock for this process plus an OS lock on <file>.lock for other processes. OSError if not had."""
    path = overrides_path(state_dir)
    with _write_lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        lk = acquire_lock(path.with_name(path.name + ".lock"), wait=wait)
        if lk is None:
            raise OSError("the Predict session settings are being changed elsewhere; try again")
        try:
            yield path
        finally:
            lk.close()


def clean_label(label) -> str:
    """'' = back to the default. ValueError with a plain message when not acceptable."""
    s = " ".join(str(label or "").split())
    if LABEL_BAD.search(str(label or "")):
        raise ValueError("A name can't contain control characters.")
    if len(s) > LABEL_MAX:
        raise ValueError(f"A name is at most {LABEL_MAX} characters.")
    return s


def set_override(state_dir, key: str, *, hidden=None, label=None, actor=None) -> tuple:
    """Change one source's override (hidden and/or label; label '' = the default). -> (old entry, new entry).
    Raises ValueError (bad key or label, unreadable file: nothing changed) or OSError (lock or write)."""
    if not isinstance(key, str) or not KEY_RE.match(key):
        raise ValueError("No such session.")
    lab = None if label is None else clean_label(label)
    with overrides_lock(state_dir) as path:
        table = load_overrides(state_dir)
        old = dict(table.get(key) or {})
        new = dict(old)
        if hidden is not None:
            new["hidden"] = bool(hidden)
        if lab is not None:
            if lab:
                new["label"] = lab
            else:
                new.pop("label", None)
        new["changed"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        new["by"] = str(actor or "-")[:80]
        table[key] = new
        _write(path, table)
    return old, new


# ---------------------------------------------------------------- merged view
def merged(cfg, fresh=False) -> dict:
    """Everything the admin sees: {"sources": [row], "skipped", "roots", "list_error", "overrides_error"}.
    row = {key, label, default_label, dir, manifest, session, suffix, sample_column, origin (found / listed / both),
           tiles, samples, nm_min, nm_max, compatible, reason, hidden, listed_problem}. Never raises."""
    disc = discover(cfg, fresh)
    try:
        hand, list_error = listed(cfg), None
    except (OSError, ValueError) as e:
        hand, list_error = [], f"{LIST_SETTING} unreadable: {type(e).__name__}: {e}"
    try:
        ovr, ovr_error = load_overrides(cfg.state_dir), None
    except ValueError as e:
        ovr, ovr_error = {}, str(e)
    rows, seen = [], {}
    for h in hand:
        if h["key"] in seen:
            continue
        f = next((s for s in disc["found"] if s["key"] == h["key"]), None)
        chk = f if f else (check_folder(h["dir"], h["manifest"], h["sample_column"]) if h["dir"].is_dir() else
                           {"tiles": 0, "samples": None, "nm_min": None, "nm_max": None})
        row = {"key": h["key"], "label": h["label"], "default_label": h["label"], "dir": h["dir"],
               "manifest": h["manifest"], "session": h["session"], "suffix": (f or {}).get("suffix", ""),
               "sample_column": h["sample_column"], "origin": "both" if f else "listed",
               "tiles": chk["tiles"], "samples": chk["samples"], "nm_min": chk["nm_min"], "nm_max": chk["nm_max"],
               "compatible": h["problem"] is None, "reason": h["problem"], "listed_problem": h["problem"]}
        seen[h["key"]] = row
        rows.append(row)
    for s in disc["found"]:
        if s["key"] in seen:
            continue
        row = {"key": s["key"], "label": s["default_label"], "default_label": s["default_label"], "dir": s["dir"],
               "manifest": s["manifest"], "session": s["session"], "suffix": s["suffix"],
               "sample_column": None, "origin": "found", "tiles": s["tiles"], "samples": s["samples"],
               "nm_min": s["nm_min"], "nm_max": s["nm_max"], "compatible": s["compatible"], "reason": s["reason"],
               "listed_problem": None}
        seen[s["key"]] = row
        rows.append(row)
    for row in rows:
        o = ovr.get(row["key"]) or {}
        row["hidden"] = bool(o.get("hidden"))
        if isinstance(o.get("label"), str) and o["label"].strip():
            row["label"] = o["label"]
    return {"sources": rows, "skipped": disc["skipped"], "roots": disc["roots"], "list_error": list_error,
            "overrides_error": ovr_error}


def for_predict(cfg) -> list:
    """predict.sources(): [{key, label, dir, manifest, session, sample_column, problem}]. A hidden source keeps
    its row with problem = HIDDEN; an incompatible discovered one has its reason as the problem. Raises OSError /
    ValueError when the hand list or the overrides file can't be read (predict.sources_safe shows the admin why)."""
    hand = listed(cfg)
    ovr = load_overrides(getattr(cfg, "state_dir", None)) if getattr(cfg, "state_dir", None) else {}
    out, seen = [], set()
    for h in hand:
        if h["key"] not in seen:
            seen.add(h["key"])
            out.append(dict(h))
    for s in discover(cfg)["found"]:
        if s["key"] in seen:
            continue
        seen.add(s["key"])
        out.append({"key": s["key"], "label": s["default_label"], "dir": s["dir"], "manifest": s["manifest"],
                    "session": s["session"], "sample_column": None,
                    "problem": None if s["compatible"] else s["reason"]})
    for it in out:
        o = ovr.get(it["key"]) or {}
        if isinstance(o.get("label"), str) and o["label"].strip():
            it["label"] = o["label"]
        if o.get("hidden"):
            it["problem"] = HIDDEN
    return out
