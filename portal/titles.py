"""
Readable titles and live review-package facts for the Data and Jobs pages (no stored fields: all computed on view).

  label_title(records, lv_id, meta_of)   "<base> + <review package> · <N> images changed · <date>"
  package_title(records, pkg_id, meta)   compare -> "Audit: <set B label> vs <set A>"; lab labeling -> "Lab labeling:
                                         <id>"; prediction -> the caller's display name (pred_name) or a plain word
  package_progress(pkg_dir)              cards decided / total, images reviewed / total, last edit time and who
  scan_packages(root, records)           the lab labeling + compare packages under a review root, with status,
                                         base label version and decisions not applied yet

Reads only review_package.json, data.js and edits/*.json of a package folder, read-only. Never decoys.json (a private
file beside compare packages) and never a prediction package (pred_*). A card is an object flagged `u` in data.js (gt
"g<id>", prediction "p<index>"); decided = its key is in the image's edit record `dec`. Cached per package by the
edits folder's mtime plus data.js's mtime and size (a saved edit replaces a file, which changes the folder mtime).
Callers escape every text they render.
"""
from __future__ import annotations

import copy
import json
import re
import threading
import time
from datetime import datetime
from pathlib import Path

LV_RE = re.compile(r"lv_\d{8}_\d{4}")
PULL_RE = re.compile(r"roboflow_pull_(\d{4})(\d{2})(\d{2})_\d{4}")
MAX_DATA_BYTES = 64_000_000
MAX_DEPTH = 4
MAX_JSON_BYTES = 1_000_000  # review_package.json and friends are tiny; a larger file is skipped, not parsed
SCAN_TTL = 15.0  # seconds a scan_packages result is reused (admin polls, label pages)
SKIP_DIRS = ("pred_", "_", "SANDBOX")
SKIP_KINDS = ("part", "prediction")

_cache: dict = {}
_lock = threading.Lock()


def _dict(x) -> dict:
    return x if isinstance(x, dict) else {}


def _text(x):
    return x.strip() if isinstance(x, str) and x.strip() else None


def read_json(path: Path):
    try:
        path = Path(path)
        if path.stat().st_size > MAX_JSON_BYTES:
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def short_when(s) -> str:
    """'30 Oct' from an ISO date or time; '' when unreadable."""
    try:
        dt = datetime.fromisoformat(str(s)[:19])
    except (TypeError, ValueError):
        return ""
    return f"{dt.day} {dt.strftime('%b')}"


def lv_from_path(text):
    """The label version id inside a path (or the id itself), else None."""
    m = LV_RE.search(str(text or ""))
    return m.group(0) if m else None


def _when(rec) -> str:
    rec = _dict(rec)
    return str(rec.get("created") or rec.get("built") or rec.get("logged") or "")


def _base_ref(record) -> str | None:
    record = _dict(record)
    if _text(record.get("base")):
        return record["base"].strip()
    for i in record.get("inputs") if isinstance(record.get("inputs"), list) else []:
        if isinstance(i, dict) and i.get("role") == "base" and _text(i.get("id")):
            return i["id"].strip()
    return None


def _pull_title(records, ref) -> str:
    pull = _dict(_dict(records.get("pull")).get(ref))
    when = short_when(pull.get("created"))
    if not when:
        m = PULL_RE.fullmatch(ref)
        when = short_when(f"{m.group(1)}-{m.group(2)}-{m.group(3)}") if m else ""
    return "Roboflow pull" + (f" {when}" if when else "")


def _base_title(records, ref, depth) -> str:
    if PULL_RE.fullmatch(ref) or ref in _dict(records.get("pull")):
        return _pull_title(records, ref)
    lv = lv_from_path(ref)
    if lv and lv in _dict(records.get("label_version")) and depth < MAX_DEPTH:
        return label_title(records, lv, None, short=True, _depth=depth + 1)
    return "Earlier labels"  # never a raw path or ref


def label_title(records, lv_id, meta_of=None, short=False, _depth=0, pkg_visible=None) -> str:
    """'<base> + <review package> · <N> images changed · <date>'. short (used for a base or an audit's set A):
    '<base>, edited <date>' (recursive, capped), or 'Labels <date>'. meta_of(pkg_id) -> review_package meta or None;
    pkg_visible(pkg_id) -> False leaves the package out (a private one must not be named)."""
    record = _dict(_dict(records.get("label_version")).get(lv_id))
    when = short_when(_when(record))
    base_ref = _base_ref(record)
    base = _base_title(records, base_ref, _depth) if base_ref else None
    if short:
        if base:
            return f"{base}, edited {when}" if when else base
        return f"Labels {when}" if when else str(lv_id)
    pkg = _text(record.get("pkg"))
    if pkg and pkg_visible is not None and not pkg_visible(pkg):
        pkg = None
    pkg_t = package_title(records, pkg, _dict(meta_of(pkg)) if meta_of else {}) if pkg else None
    head = " + ".join(x for x in (base, pkg_t) if x)
    n = record.get("n_images_applied")
    changed = f"{n} image{'' if n == 1 else 's'} changed" if isinstance(n, int) and not isinstance(n, bool) else None
    out = " · ".join(x for x in (head or (f"Labels {when}" if when else None), changed, when if head else None) if x)
    return out or str(lv_id)


def package_title(records, pkg_id, meta=None, pred_name=None) -> str:
    """By kind: compare -> 'Audit: <B label> vs <A>'; prediction -> pred_name(pkg_id) or 'Prediction review'; else
    'Lab labeling: <id>'. `meta` = review_package.json (merged over the registry record by the caller)."""
    meta = {**_dict(_dict(records.get("review_package")).get(pkg_id)), **_dict(meta)}
    kind = meta.get("kind")
    if kind == "prediction" or str(pkg_id).startswith("pred_"):
        return (pred_name(pkg_id) if pred_name else None) or "Prediction review"
    if kind == "compare":
        cmp = _dict(meta.get("compare"))
        a, b = _dict(cmp.get("a")), _dict(cmp.get("b"))
        a_lv = lv_from_path(a.get("coco") or a.get("id"))
        a_text = (label_title(records, a_lv, None, short=True) if a_lv in _dict(records.get("label_version"))
                  else a_lv or _text(a.get("kind")) or "labels")
        b_text = _text(cmp.get("label")) or _text(b.get("kind")) or "predictions"
        return f"Audit: {b_text} vs {a_text}"
    return f"Lab labeling: {pkg_id}"


# ---------------------------------------------------------------- live facts of a package folder
def _data_js(pdir: Path):
    """{'source', 'cards': {file: set(card keys)}} from data.js, or None."""
    p = Path(pdir) / "data.js"
    try:
        if p.stat().st_size > MAX_DATA_BYTES:
            return None
        text = p.read_text(encoding="utf-8")
        d = json.loads(text[text.index("=") + 1:].strip().rstrip(";"))
    except (OSError, ValueError):
        return None
    if not isinstance(d, dict):
        return None
    cards = {}
    for im in d.get("images") if isinstance(d.get("images"), list) else []:
        if not isinstance(im, dict) or not isinstance(im.get("file"), str):
            continue
        keys = {f"g{o['id']}" for o in im.get("gt") or [] if isinstance(o, dict) and o.get("u") and "id" in o}
        keys |= {f"p{i}" for i, o in enumerate(im.get("pr") or []) if isinstance(o, dict) and o.get("u")}
        cards[im["file"]] = keys
    return {"source": _text(d.get("source")), "cards": cards}


def _sig(pdir: Path):
    """Cache key: data.js plus every edit record's name, size and mtime. The folder's own mtime alone can miss a
    change made within the same clock tick (and never changes when a record is rewritten in place)."""
    try:
        d = (Path(pdir) / "data.js").stat()
    except OSError:
        return None
    try:
        edits = tuple(sorted((f.name, s.st_size, s.st_mtime_ns) for f in (Path(pdir) / "edits").iterdir()
                             if f.suffix == ".json" for s in (f.stat(),)))
    except OSError:
        edits = None
    return (edits, d.st_mtime_ns, d.st_size)


def package_progress(pkg_dir) -> dict | None:
    """{'source', 'cards', 'decided', 'images', 'reviewed', 'last', 'who'} (cached), or None without a data.js."""
    pdir = Path(pkg_dir)
    sig = _sig(pdir)
    if sig is None:
        return None
    key = str(pdir)
    with _lock:
        hit = _cache.get(key)
        if hit and hit[0] == sig:
            return hit[1]
    data = _data_js(pdir)
    if data is None:
        return None
    decided = reviewed = 0
    last, newest_dec, who = None, -1.0, None
    try:
        files = list((pdir / "edits").glob("*.json"))
    except OSError:
        files = []
    for f in files:
        try:
            t = f.stat().st_mtime
        except OSError:
            continue
        last = t if last is None or t > last else last
        rec = _dict(read_json(f))
        edit = _dict(rec.get("edit"))
        dec = _dict(edit.get("dec"))
        mine = data["cards"].get(rec.get("file"), set()) & set(dec)
        decided += len(mine)
        reviewed += 1 if edit.get("reviewed") else 0
        for v in dec.values():
            v = _dict(v)
            ts = v.get("server_t") or v.get("t")
            if isinstance(ts, (int, float)) and ts > newest_dec and _text(v.get("who")):
                newest_dec, who = ts, v["who"]
    out = {"source": data["source"], "cards": sum(len(v) for v in data["cards"].values()), "decided": decided,
           "images": len(data["cards"]), "reviewed": reviewed, "last": last, "who": who}
    with _lock:
        _cache[key] = (sig, out)
    return out


def scan_packages(root, records) -> list:
    """Lab labeling and compare packages under the review root (never pred_*, parts, sandboxes or work folders):
    [{id, title, kind, cards, decided, images, reviewed, last, who, status, applied_by (lv id or None),
    base_lv, waiting (decisions not applied yet, 0 when applied)}]; status ready / in_progress / applied."""
    if root is None:
        return []
    key = ("scan", str(root), hash(repr([_dict(records).get(k) for k in ("label_version", "review_package", "pull")])))
    now = time.monotonic()
    with _lock:
        hit = _cache.get(key)
    if hit and now - hit[0] < SCAN_TTL:
        return copy.deepcopy(hit[1])
    out = _scan_packages(root, records)
    with _lock:
        _cache[key] = (now, copy.deepcopy(out))
    return out


def _scan_packages(root, records) -> list:
    try:
        dirs = sorted(d for d in Path(root).iterdir() if d.is_dir() and not d.name.startswith(SKIP_DIRS))
    except OSError:
        return []
    applied = {}
    for lv, rec in sorted(_dict(records.get("label_version")).items()):  # ids sort by time: the newest wins
        pkg = _text(_dict(rec).get("pkg"))
        if not pkg:
            continue
        if _dict(rec).get("status") != "superseded":
            applied[pkg] = lv          # the current label set built from this package
        else:
            applied.setdefault(pkg, lv)  # only if nothing current exists
    out = []
    for d in dirs:
        prog = package_progress(d)
        if prog is None:
            continue
        meta = {**_dict(_dict(records.get("review_package")).get(d.name)), **_dict(read_json(d / "review_package.json"))}
        if meta.get("kind") in SKIP_KINDS or meta.get("parent"):
            continue
        base_lv = lv_from_path(meta.get("base")) or lv_from_path(prog["source"])
        by = applied.get(d.name)
        has_edits = prog["last"] is not None
        status = "applied" if by else ("in_progress" if has_edits and prog["decided"] else "ready")
        out.append({"id": d.name, "title": package_title(records, d.name, meta), "kind": meta.get("kind") or "review",
                    **{k: prog[k] for k in ("cards", "decided", "images", "reviewed", "last", "who")},
                    "status": status, "applied_by": by, "base_lv": base_lv,
                    "waiting": 0 if by else prog["decided"], "meta": meta})
    return out
