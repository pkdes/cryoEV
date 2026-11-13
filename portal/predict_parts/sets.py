"""
Predict, prediction sets: the models (models.yaml + registry), a session's tiles (sample groups, nm/px), reading
sets, background work (review package, results, spot check list) and a set's status in plain words.

Split out of portal/modules/predict.py (round 7, P7; no behaviour change); predict.py re-exports every name.
Names the tests patch are used from other parts only as <PART>.<name> (portal/predict_parts/__init__.py).
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import threading
import time
from pathlib import Path

from jobs import predict_local as PL
from jobs import predict_set as PS

from portal.predict_parts.common import (IMAGE_EXT, MODELS_YAML, NM_EPS, NM_TOLERANCE, REPO, SAMPLE_COLUMNS,
                                         SIZE_COLUMN, SPOT_N, STALE_S, _parse_local, _read_json, ago, registry_root,
                                         review_root)
from portal.predict_parts import common as COMMON


# ---------------------------------------------------------------- models
def _models_yaml() -> dict:
    import yaml
    return (yaml.safe_load(MODELS_YAML.read_text(encoding="utf-8")) or {}).get("models") or {}


def quality_line(spec: dict) -> str | None:
    """One plain line on how good a model is, from models.yaml (the EV score on held-out test images)."""
    s = str(spec.get("test_seed_spread") or "")
    m = re.search(r"EV\s+([0-9.]+)\s*\+/-\s*([0-9.]+)", s)
    if m:
        return f"EV score {float(m.group(1)):.2f} ± {float(m.group(2)):.2f} on held-out test images (1 = perfect)"
    m = re.search(r"EV\s+([0-9.]+)", str(spec.get("test_mask_map50_95") or ""))
    if m:
        return f"EV score {float(m.group(1)):.2f} on held-out test images (1 = perfect)"
    return None


def models(cfg) -> dict:
    """{"production": id | None, "choices": [{id, quality, production}]}: models.yaml entries whose weights exist;
    production = the newest registry model record with status production that models.yaml also has."""
    try:
        spec = _models_yaml()
    except (OSError, ValueError, ImportError):
        spec = {}
    usable = [m for m, s in spec.items() if isinstance(s, dict) and (REPO / str(s.get("weights", ""))).is_file()]
    prod = None
    try:
        recs = PS.registry_module().current("model", registry_root(cfg))
        cands = sorted((r for r in recs.values() if r.get("status") == "production" and r.get("id") in usable),
                       key=lambda r: str(r.get("logged") or ""))
        prod = cands[-1]["id"] if cands else None
    except Exception:                                    # registry unreadable: no default, never a guess
        prod = None
    return {"production": prod, "choices": [{"id": m, "quality": quality_line(spec[m]), "production": m == prod}
                                            for m in usable]}


def allowed_models(cfg, user) -> tuple:
    """-> (default id or None, [choices this person may pick])."""
    ms = models(cfg)
    if user.get("role") == "basic":
        return ms["production"], [c for c in ms["choices"] if c["production"]]
    return ms["production"], ms["choices"]


# ---------------------------------------------------------------- a session's tiles
def _num(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) and x > 0 else None


def nm_spread(values) -> float:
    """(max - min) / median of nm/px values, one per tile (0.0 for a single value). = results.compute.nm_spread."""
    vals = sorted(float(v) for v in values)
    n = len(vals)
    med = vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2
    return (vals[-1] - vals[0]) / med


def one_scale(values) -> bool:
    """These nm/px values (one per tile of one sample) count as one scale: spread <= NM_TOLERANCE (+ NM_EPS, so
    exactly 1 % is not lost to float rounding). = results.compute.one_scale and jobs.predict_set.nm_one_scale."""
    return nm_spread(values) <= NM_TOLERANCE + NM_EPS


def nm_range_text(lo, hi) -> str:
    return f"{lo:g}" if lo == hi else f"{lo:g}–{hi:g}"


SIZE_RE = re.compile(r"^\s*(\d+)\s*[x×X]\s*(\d+)\s*$")


def tile_size(path: Path, row: dict):
    """(w, h) from the manifest's output_size ("1440x1023"), else from the image header (no decode); None when
    neither can be read (create() then refuses that tile with the reason)."""
    m = SIZE_RE.match(str(row.get(SIZE_COLUMN) or ""))
    if m:
        return int(m.group(1)), int(m.group(2))
    try:
        from PIL import Image
        with Image.open(path) as im:                 # lazy: reads the header only
            return tuple(im.size)
    except Exception:
        return None


def size_suffix(size) -> str:
    return f" · {size[0]}×{size[1]} tiles" if size else " · tiles of unknown size"


def tiles_of(src: dict) -> dict:
    """{"groups": [{label, base, suffix, size, names, nm, nm_values, nm_of, nm_min, nm_max, nm_missing, nm_range,
    mixed}], "sig", "n", "column"}.
    Groups come from the manifest's sample column (without a manifest: one sample, the source label), split by tile
    size (manifest output_size, else the image header): a sample with more than one tile size becomes one group per
    size, labelled "<sample> · <W>×<H> tiles" (suffix; that label goes into images.csv, so results counts them as
    separate samples: one sample per scale). nm: the one value when every tile has the same; None when missing (any
    tile), several or mixed. nm_of: tile name -> its manifest value (None = missing); nm_missing: tiles without one.
    nm_range: several values, all tiles have one, spread within NM_TOLERANCE (one scale; each tile keeps its own).
    mixed: the values spread more than that (refused)."""
    names = sorted((p.name for p in src["dir"].iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXT
                    and not p.name.startswith(".")), key=str)
    rows, col = {}, None
    if src["manifest"] is not None:
        with open(src["manifest"], newline="", encoding="utf-8-sig") as f:
            data = list(csv.DictReader(f))
        head = data[0].keys() if data else []
        key = next((c for c in PS.NAME_COLUMNS if c in head), None)
        if key:
            rows = {r[key]: r for r in data}
        want = src.get("sample_column")
        col = want if want in head else next((c for c in SAMPLE_COLUMNS if c in head), None)
    groups = {}
    for n in names:
        r = rows.get(n) or {}
        if col:
            v = str(r.get(col) or "").strip()
            label = (f"grid {v}" if col == "grid" and v and not v.lower().startswith("grid") else v) or "no sample"
        else:
            label = src["label"]
        size = tile_size(src["dir"] / n, r)
        g = groups.setdefault((label, size), {"base": label, "size": size, "names": [], "nm_of": {}})
        g["names"].append(n)
        g["nm_of"][n] = _num(r.get(PS.NM_COLUMN))
    n_sizes = {}
    for base, _ in groups:
        n_sizes[base] = n_sizes.get(base, 0) + 1
    for g in groups.values():
        g["suffix"] = size_suffix(g["size"]) if n_sizes[g["base"]] > 1 else ""
        g["label"] = g["base"] + g["suffix"]
    out = []
    for g in sorted(groups.values(), key=lambda g: (str(g["base"]), g["size"] or (0, 0))):
        per_tile = [v for v in g["nm_of"].values() if v is not None]
        missing = len(g["names"]) - len(per_tile)
        vals = sorted(set(per_tile))
        mixed = len(vals) > 1 and not one_scale(per_tile)
        out.append({"label": g["label"], "base": g["base"], "suffix": g["suffix"], "size": g["size"],
                    "names": g["names"], "nm_of": g["nm_of"],
                    "nm": vals[0] if len(vals) == 1 and not missing else None, "nm_values": vals,
                    "nm_min": vals[0] if vals else None, "nm_max": vals[-1] if vals else None,
                    "nm_missing": missing, "nm_range": len(vals) > 1 and not mixed and not missing, "mixed": mixed})
    # the values are in the signature too: a range group's tile values are not sent back by the form, so a manifest
    # changed since the person confirmed the range must be seen again
    sig = hashlib.sha1(json.dumps([[g["label"], len(g["names"]), g["nm_values"]] for g in out]).encode()
                       ).hexdigest()[:12]
    return {"groups": out, "sig": sig, "n": len(names), "column": col}


def pick(groups: list, count) -> list:
    """The tiles to use: every tile of `groups`, or `count` of them spread over the groups in proportion to their
    size (>= 1 each when count >= the number of groups), evenly spaced within each group. Deterministic."""
    total = sum(len(g["names"]) for g in groups)
    if count is None or count >= total:
        return [n for g in groups for n in g["names"]]
    if count < len(groups):
        groups = sorted(groups, key=lambda g: (-len(g["names"]), g["label"]))[:count]
        quota = {g["label"]: 1 for g in groups}
    else:
        exact = {g["label"]: count * len(g["names"]) / total for g in groups}
        quota = {g["label"]: min(len(g["names"]), max(1, math.floor(exact[g["label"]]))) for g in groups}
        size = {g["label"]: len(g["names"]) for g in groups}
        while sum(quota.values()) > count:
            k = max((k for k in quota if quota[k] > 1), key=lambda k: (quota[k] - exact[k], k))
            quota[k] -= 1
        while sum(quota.values()) < count:
            k = max((k for k in quota if quota[k] < size[k]), key=lambda k: (exact[k] - quota[k], k))
            quota[k] += 1
    out = []
    for g in groups:
        n, q = len(g["names"]), quota[g["label"]]
        out += [g["names"][min(n - 1, int((i + 0.5) * n / q))] for i in range(q)]
    return out


# ---------------------------------------------------------------- reading sets
def set_dir(cfg, ps) -> Path | None:
    if not isinstance(ps, str) or not PS.PS_RE.match(ps):
        return None
    d = COMMON.predict_root(cfg) / ps
    return d if (d / PS.REQUEST).is_file() else None


def load_set(cfg, user, can_see, ps) -> tuple:
    """-> (folder, request) when it exists AND this person may see it; else (None, None) (one answer for both)."""
    d = set_dir(cfg, ps)
    req = _read_json(d / PS.REQUEST) if d else None
    if not isinstance(req, dict) or not can_see(user, req.get("owner")):
        return None, None
    return d, req


def list_sets(cfg, user, can_see) -> list:
    root = COMMON.predict_root(cfg)
    try:
        dirs = [p for p in root.iterdir() if p.is_dir() and PS.PS_RE.match(p.name)]
    except OSError:
        return []
    out = []
    for d in dirs:
        req = _read_json(d / PS.REQUEST)
        if not isinstance(req, dict) or not can_see(user, req.get("owner")):
            continue
        st = _read_json(d / PS.STATUS) or {}
        out.append((d, req, st))
    out.sort(key=lambda t: (str(t[1].get("created") or ""), t[0].name), reverse=True)
    return out


def images_rows(d: Path) -> list:
    try:
        return PS.read_images(d)
    except (OSError, ValueError, csv.Error):
        return []


def samples_of(rows) -> list:
    by = {}
    for r in rows:
        s = by.setdefault(r.get("sample_label") or "", {"label": r.get("sample_label") or "", "n": 0, "nm": set(),
                                                         "by": set()})
        s["n"] += 1
        s["nm"].add(r.get("nm_per_px") or "")
        s["by"].add(r.get("nm_confirmed_by") or "")
    def nm_text(vals):
        nums = sorted(_num(v) for v in vals if _num(v))
        if len(nums) > 1 and len(nums) == len(vals):     # several values (within NM_TOLERANCE): each image its own
            return f"{nm_range_text(nums[0], nums[-1])} (each image its own)"
        return ", ".join(f"{float(v):g}" if _num(v) else "?" for v in sorted(vals))
    return [{"label": s["label"], "n": s["n"], "nm_per_px": nm_text(s["nm"]),
             "confirmed_by": ", ".join(sorted(b for b in s["by"] if b)) or None} for s in by.values()]


def package_dir(cfg, ps) -> Path | None:
    try:
        p = review_root(cfg) / f"pred_{ps}"
    except Exception:
        return None
    return p if (p / "review_package.json").is_file() else None


def results_file(d: Path) -> Path | None:
    p = d / "results" / "results.json"
    return p if p.is_file() else None


def edits_mtime(pkg: Path | None) -> float:
    if pkg is None or not (pkg / "edits").is_dir():
        return 0.0
    return max((p.stat().st_mtime for p in (pkg / "edits").glob("*.json")), default=0.0)


# ---------------------------------------------------------------- background work (review package, results)
_TASKS: dict = {}
_TASK_LOCK = threading.Lock()


def task(kind, ps) -> dict | None:
    with _TASK_LOCK:
        t = _TASKS.get((kind, ps))
        return dict(t) if t else None


def start_task(kind, ps, fn) -> bool:
    """Run fn() in a daemon thread unless the same (kind, ps) is running. A failure is kept for the page."""
    with _TASK_LOCK:
        cur = _TASKS.get((kind, ps))
        if cur and cur["state"] == "running":
            return False
        _TASKS.pop((kind, ps), None)
        _TASKS[(kind, ps)] = {"state": "running", "error": None, "started": time.time()}
        for k in [k for k, v in _TASKS.items() if v["state"] != "running"][:max(0, len(_TASKS) - COMMON.CACHE_MAX)]:
            _TASKS.pop(k, None)                          # oldest finished entries first: the table stays bounded

    def run():
        try:
            fn()
            with _TASK_LOCK:
                _TASKS.pop((kind, ps), None)
        except BaseException as e:          # noqa: BLE001 - kept and shown, never lost in a thread
            with _TASK_LOCK:
                _TASKS[(kind, ps)] = {"state": "failed", "error": f"{type(e).__name__}: {e}", "started": time.time()}

    threading.Thread(target=run, name=f"cj-predict-{kind}-{ps}", daemon=True).start()
    return True


def build_review(cfg, d: Path):
    from annotation import prediction_package as PP
    root = review_root(cfg)
    root.mkdir(parents=True, exist_ok=True)
    pkg = PP.build(d, root)
    PP.spot_check(pkg, n=SPOT_N, seed=0, write=True, mode="representative")
    req = _read_json(d / PS.REQUEST) or {}
    register_package(cfg, pkg, owner_via=req.get("owner_via") if isinstance(req, dict) else None)
    return pkg


REVIEW_PACKAGE_LOCK = ".review_package.lock"     # in the registry folder: read + append, as predict_set's lock


def register_package(cfg, pkg: Path, owner_via=None):
    """Append the package's registry line (its own review_package.json, prediction_package.registry_record) unless
    the registry already has that id, under a registry-folder lock (check + append as one step). owner_via: the
    agent that made the set for its owner (request.json, round 11 #65), copied onto the line so the Jobs board shows
    "<agent> for <owner>" on the package too."""
    reg = registry_root(cfg)
    if reg is None:
        return None
    rec = _read_json(pkg / "review_package.json")
    if not isinstance(rec, dict) or not rec.get("id"):
        raise ValueError(f"{pkg.name}: review_package.json missing or unreadable")
    if isinstance(owner_via, str) and owner_via:
        rec["owner_via"] = owner_via
    regm = PS.registry_module()
    with PS.locked(Path(reg) / REVIEW_PACKAGE_LOCK):
        if rec["id"] in regm.current("review_package", reg):
            return None
        return regm.append(rec, reg)


def build_results(cfg, d: Path, ps: str):
    from results import compute_and_write
    pkg = package_dir(cfg, ps)
    reviewed = None
    if pkg is not None and edits_mtime(pkg):
        from annotation import prediction_package as PP
        reviewed = PP.reviewed_polygons(pkg, write=True)
    compute_and_write(d, reviewed)


_SPOT_CACHE: dict = {}
_SPOT_LOCK = threading.Lock()


def spot_list(cfg, d: Path, ps: str, rows) -> tuple:
    """-> (rows to check first [{image, file, sample_label, orig_name}], mode text). From the package's
    spotcheck.json when there is one, else spot_check() on the set (cached per set)."""
    by_image = {r.get("image"): r for r in rows}
    pkg = package_dir(cfg, ps)
    ids = None
    if pkg is not None:
        sc = _read_json(pkg / "spotcheck.json")
        if isinstance(sc, dict) and isinstance(sc.get("images"), list):
            ids = [str(i.get("image")) for i in sc["images"] if isinstance(i, dict)]
    if ids is None:
        key = (str(d), (d / PS.STATUS).stat().st_mtime if (d / PS.STATUS).exists() else 0)
        with _SPOT_LOCK:
            ids = _SPOT_CACHE.get(key)
        if ids is None:
            from annotation import prediction_package as PP
            try:
                ids = PP.spot_check(d, n=SPOT_N, seed=0, write=False, mode="representative")
            except Exception:
                ids = []
            with _SPOT_LOCK:
                _SPOT_CACHE[key] = ids
                while len(_SPOT_CACHE) > COMMON.CACHE_MAX:
                    _SPOT_CACHE.pop(next(iter(_SPOT_CACHE)))
    out = []
    from annotation import prediction_package as PP
    for i in ids:
        r = by_image.get(i) or {}
        if pkg is not None and PP._rejected_file(pkg, Path(r.get("file") or "").name or None):
            continue                                     # rejected after spotcheck.json was written: nothing to check
        out.append({"image": i, "file": Path(r.get("file") or "").name or None,
                    "sample_label": r.get("sample_label"), "orig_name": r.get("orig_name")})
    return out


# ---------------------------------------------------------------- status in plain words
RESTART_REQUESTED = "restart requested"


def waiting_text(waiting) -> str:
    """predict_local's `waiting` reason in plain words (admins also get the raw reason on the tech line)."""
    w = str(waiting or "").lower()
    if w.startswith(RESTART_REQUESTED):
        return "Starting again…"
    if "gpu busy" in w:
        return "Waiting for the GPU: another job is using it."
    if any(k in w for k in ("lock", "another", "queue", "after")):
        return "Waiting for the GPU: another analysis is running."
    return "Waiting for the computer to be free."


def age_s(st: dict, now) -> float:
    return now - (_parse_local(st.get("updated")) or now)


def status_view(st: dict, admin: bool, now=None) -> dict:
    """status.json -> {state, tone, label, detail, tech (admin only), final, n_done, n_total, progress}. Queued or
    running with `updated` older than STALE_S (predict_local keeps a heartbeat while it runs or waits) = stale."""
    now = time.time() if now is None else now
    state = st.get("state")
    n_done, n_total = st.get("n_done"), st.get("n_total")
    out = {"state": state, "tone": "wait", "label": "Unknown", "detail": None, "tech": None, "final": False,
           "n_done": n_done, "n_total": n_total, "progress": False}
    stale = state in ("queued", "running") and not st.get("job") and age_s(st, now) > STALE_S
    if state == "queued":
        out["label"] = "Waiting to start"
        if st.get("waiting"):
            out["detail"] = waiting_text(st["waiting"])
        elif st.get("job"):
            out["detail"] = "Queued on the GPU computer."
        else:
            out["detail"] = "It starts in a moment."
    elif state == "running":
        out.update(tone="run", label="Running", progress=bool(n_total))
        out["detail"] = f"{int(n_done or 0)} of {int(n_total or 0)} images done"
    elif state == "done":
        out.update(tone="ok", label="Done", final=True, detail=f"{int(n_total or 0)} images analysed")
    elif state == "failed":
        out.update(tone="warn", label="Stopped with an error", final=True,
                   detail="Try “Start again”; if it fails again, ask the admin (they can see the details).")
    if stale:                                          # no heartbeat: the run (or the wait) may have stopped
        out.update(tone="warn", label="No progress")
        out["detail"] = (out["detail"] + ". " if out["detail"] else "") + \
            f"No news for {ago(age_s(st, now))}: it may have stopped. Try “Start again” or ask the admin."
    if admin:
        bits = [f"where {st.get('where')}"] + [f"{k} {st.get(k)}" for k in ("job", "pid", "waiting", "error")
                                                if st.get(k)] + [f"updated {st.get('updated')}"]
        out["tech"] = "; ".join(bits)
    return out


def can_restart(st: dict, now=None) -> bool:
    """Failed, or queued / running with no heartbeat for STALE_S. A set waiting with a fresh heartbeat is left to
    predict_local (it waits for the lock / GPU and chains the next set); a desktop-queue set is never started here."""
    now = time.time() if now is None else now
    state = st.get("state")
    if state == "failed":
        return True
    if st.get("job") and state in ("queued", "running"):
        return False
    return state in ("queued", "running") and age_s(st, now) > STALE_S


def restart(cfg, d: Path) -> bool:
    """"Start again": mark the set queued with waiting "restart requested" and a fresh `updated` (so a double click
    is not a second restart), then spawn predict_local (its own lock keeps one run per root)."""
    st = _read_json(d / PS.STATUS) or {}
    if not can_restart(st):
        return False
    PS.write_status(d, state="queued", error=None, pid=None, waiting=RESTART_REQUESTED)
    try:
        PL.start_detached(d, root=COMMON.predict_root(cfg), registry_root=registry_root(cfg))
    except Exception as e:                       # shown to the admin; goes stale and offers Start again later
        PS.write_status(d, waiting=f"could not start the run: {type(e).__name__}: {e}")
    return True
