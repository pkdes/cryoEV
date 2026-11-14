"""
Start a review from the portal (round 13, B4): build a label-set revision or a blind audit with
annotation/compare_package.py. Logic only: the routes are portal/modules/reviewstart.py, the detached runner is
jobs/review_build_run.py (which calls run() here). It lives outside portal/modules/ because modules.discover()
imports every module there.

Inputs from a label set (an `lv` id):
  - labels = <labels root>/<lv>/labels.coco.json (labels root = <review root>/../labels, i.e. <CryoAI-laptop>/labels);
  - images and splits from the EXACT link, never a best guess (QC 12): a registered roboflow_export whose
    `label_version` is the lv; else a registered dataset made from the lv (`label_version`, or source.kind
    label_version + source.id; data_utils/dataset_from_lv.py) whose `export` is a registered roboflow_export; else the
    same two checks on the lv's base (version.json `base`, else the registry line's base input), and so on down the
    chain of label versions. Two different exports at one step is refused (ambiguous). The split is the export's own
    `split_file`, else the dataset's split file, else the folder (train / valid / test) each image was found in;
  - every label-set image must be found in the export (check_images): by its own key with the COCO width:height
    (1 % tolerance), else through its registry alias (image records' `duplicate_of`, both ways: the importer's link
    between a Roboflow name and the session-prefixed name of the same image; a 2-name cluster only) with EXACTLY the
    COCO pixel size. A key that is not a plain file name, a stem found twice in the export or a bigger alias cluster
    is refused too. Any refusal stops the build, with the counts; "Images found N of M (k through duplicate_of)" is
    shown before starting. The runner stages the checked images under their label-set keys (hard links, else
    copies) in the state folder and passes that folder and a split CSV to compare_package.

A build (start() in the portal, run() in the runner):
  - admin only (the routes check); ONE build at a time across both portal doors: the runner holds the OS lock
    <review root>/.building/build.lock for its whole life, and start() refuses while that lock is held or a status
    is still starting (under <review root>/.building/start.lock);
  - the name is lab_revise_<lv>_<yyyymmdd_hhmm>_<4 hex> / lab_audit_..., inside portal/app.py PRED_ID_RE; a name whose
    package, _work_ or status folder exists is never reused;
  - status: <review root>/.building/<name>/status.json (both doors read it), the allowlisted phases PHASES and reason
    codes REASONS only; never decoy numbers, paths or the raw log;
  - the spec and the raw log of compare_package: <portal state>/review_build/<name>/ (spec.json, build.log,
    runner.out), never under the review root and never shown on a page; a revise's scratch folder is
    <portal state>/review_build/<name>/work (compare_package --work-dir, deleted after a successful build);
  - compare_package.build() runs IN the runner process (QC 1): when B is a model, YOLO runs there too, so a killed
    runner can't leave an orphaned GPU child. The runner takes <predict root>/predict.lock itself (as
    jobs/predict_local.py does) and waits for jobs.predict_local.gpu_check (run_job.gpu_state) to find the GPU free,
    up to WAIT_S; after the build it releases the lock and starts the next queued prediction set
    (predict_local.next_queued + start_detached, as its --chain does). B is a label set (`--b-coco`): no GPU, no lock;
  - the package is built in <review root>/.building/<short_id>/p/<name> and an audit's work folder (its answer key,
    a fresh --seed per build) in .../<short_id>/w (short_id = sha1(name)[:8]: short paths, so 79-char image keys
    stay under Windows MAX_PATH), and renamed into the review root (as <name> and _work_<name>) only once complete
    and checked (_work_ first), so a killed build never leaves a listable package. On start, under the build lock,
    the runner marks every other non-final status failed "stopped" and removes other builds' staging under
    .building only; nothing in the review root is deleted (an unfinished <name> folder is only logged) and no
    _work_* folder is ever touched;
  - after a successful build the runner appends an allowlisted review_package registry line (ALLOWED_LINE_KEYS: no
    `cards`, no `path`, no decoy counts) under <registry>/.review_package.lock, recorded_by "portal/review_build.py"
    (as portal/predict_parts/sets.py register_package); the registry importer skips the package (built_by "portal").
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from data_utils.oslock import acquire_lock

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
TOOL = "annotation/compare_package.py"
RECORDED_BY = "portal/review_build.py"
PURPOSES = ("revise", "audit")
LV_RE = re.compile(r"^lv_\d{8}_\d{4}\Z")
MODEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_]{0,80}\Z")
PID_RE = re.compile(r"^[a-z][a-z0-9_-]{0,31}\Z")          # a portal id (compare_package REVIEWER_RE)
NAME_RE = re.compile(r"^lab_(?:revise|audit)_lv_\d{8}_\d{4}_\d{8}_\d{4}_[0-9a-f]{4}\Z")
MAX_REVIEWERS = 12
BUILDING = ".building"                 # <review root>/.building/<name>/status.json; a dot folder: never a package
BUILD_LOCK = "build.lock"              # held by the runner for its whole life: one build at a time
START_LOCK = "start.lock"              # held by start() while it checks and starts
STATE_SUB = "review_build"             # <portal state>/review_build/<name>/
REVIEW_PACKAGE_LOCK = ".review_package.lock"   # in the registry folder (portal/predict_parts/sets.py)
RATIO_TOL = 0.01
KEY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*\Z")    # a label-set image key: it names the staged file
SPLIT_DIRS = ("train", "valid", "val", "test")
IMG_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
WAIT_S = 6 * 3600                      # a model build waits this long for the predict lock / a free GPU
WAIT_POLL_S = 30.0
START_GRACE_S = 120.0                  # a status still without a runner pid after this is a failed start
COVERAGE_TTL = 60.0
PHASES = ("checking", "waiting", "predicting", "writing", "done", "failed")
TERMINAL = ("done", "failed")
PHASE_TEXT = {"checking": "Checking images", "waiting": "Waiting for GPU", "predicting": "Predicting",
              "writing": "Writing package", "done": "Done", "failed": "Failed"}
REASONS = {                            # the only reasons a status may carry (code -> text)
    "images": "Some images are missing or don't match their labels",
    "inputs": "The label set's images can't be found",
    "busy": "Another review was being built",
    "lock": "Another analysis is running; waiting for it to finish",
    "gpu_busy": "The GPU is busy; waiting for it",
    "gpu_missing": "No GPU found on this computer",
    "gpu_timeout": "Gave up waiting for the GPU",
    "build": "The build failed (the technical log is kept on the server)",
    "check": "The finished package could not be checked",
    "registry": "Built, but its registry line was not written",
    "stopped": "The build stopped (the server restarted or the process ended)",
    "start": "The build could not be started",
}
STATUS_KEYS = ("name", "purpose", "lv", "b_kind", "b_id", "phase", "failed_at", "reason", "detail", "started",
               "updated", "finished", "pid", "by")
ALLOWED_LINE_KEYS = ("schema", "id", "kind", "purpose", "base", "base_kind", "n_images", "built", "status", "owner",
                     "built_by", "model", "b_label_version", "flag_conf", "tool", "recorded_by")

_cov_cache: dict = {}
_cov_lock = threading.Lock()


class BuildError(Exception):
    """A refusal shown to the admin (short, no paths)."""


# ---------------------------------------------------------------- roots
def review_root(cfg) -> Path | None:
    from portal.modules import results as RES
    return RES.review_root(cfg)


def labels_root(cfg) -> Path | None:
    v = getattr(cfg, "labels_root", None) or os.environ.get("CJ_LABELS_ROOT")
    if v:
        return Path(v)
    rr = review_root(cfg)
    return Path(rr).parent / "labels" if rr else None


def registry_root(cfg) -> Path | None:
    v = getattr(cfg, "registry", None)
    return Path(v) if v else None


def predict_root(cfg) -> Path:
    from portal.predict_parts import common as COMMON
    return COMMON.predict_root(cfg)


def state_root(cfg) -> Path:
    return Path(cfg.state_dir) / STATE_SUB


def _read_json(p):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------- label sets and models
def label_sets(cfg) -> list:
    """[{id, n_annotations, pkg, current}] for <labels root>/lv_* with labels.coco.json, newest first. current = no
    other label set supersedes it (version.json `supersedes`, or a SUPERSEDED.md)."""
    root = labels_root(cfg)
    try:
        dirs = sorted((d for d in Path(root).iterdir() if d.is_dir() and LV_RE.match(d.name)
                       and (d / "labels.coco.json").is_file()), key=lambda d: d.name, reverse=True)
    except (OSError, TypeError):
        return []
    vers = {d.name: _read_json(d / "version.json") or {} for d in dirs}
    named = {v.get("supersedes") for v in vers.values() if isinstance(v.get("supersedes"), str)}
    out = []
    for d in dirs:
        v = vers[d.name]
        n = v.get("n_annotations_out")
        out.append({"id": d.name, "n_annotations": n if isinstance(n, int) else None,
                    "pkg": v.get("pkg") if isinstance(v.get("pkg"), str) else None,
                    "current": d.name not in named and not (d / "SUPERSEDED.md").exists()})
    return out


def models(cfg) -> list:
    """[{id, production, weights, imgsz, classes}] the models Predict offers (models.yaml entries whose weights exist)."""
    from portal.predict_parts import sets as SETS
    try:
        ms = SETS.models(cfg)
        spec = SETS._models_yaml()
    except Exception:
        return []
    out = []
    for c in ms.get("choices") or []:
        s = spec.get(c["id"]) or {}
        cls = s.get("classes")
        names = ",".join(str(cls[k]) for k in sorted(cls, key=lambda k: int(k))) if isinstance(cls, dict) and cls \
            else "EV,non-EV"
        out.append({"id": c["id"], "production": bool(c.get("production")), "weights": str(REPO / str(s.get("weights"))),
                    "imgsz": int(s.get("imgsz") or 1440), "classes": names})
    return out


# ---------------------------------------------------------------- inputs: the exact export link
def _registry(reg):
    from registry import store
    return store.current(root=Path(reg))


def lv_base(lv, lroot, recs) -> str | None:
    """The label version an lv was made on (its version.json `base`, else the registry line's base input), if any."""
    v = _read_json(Path(lroot) / lv / "version.json") if lroot else None
    base = v.get("base") if isinstance(v, dict) else None
    if not isinstance(base, str):
        rec = recs.get(lv) or {}
        base = next((i.get("id") for i in rec.get("inputs") or [] if isinstance(i, dict) and i.get("role") == "base"),
                    None)
    return base if isinstance(base, str) else None


def find_export(lv, lroot, reg, recs=None) -> dict:
    """-> {export: record, split_file: Path or None, how: short text, chain: [lv, ...]}. Raises BuildError with
    exactly what is missing when no registered export is linked to the lv or its base chain."""
    if recs is None:
        recs = registry_records(reg)
    exports = {k: r for k, r in recs.items() if r.get("schema") == "roboflow_export/1"}
    datasets = [r for r in recs.values() if r.get("schema") == "dataset/1"]
    chain, cur = [], lv
    while isinstance(cur, str) and LV_RE.match(cur) and cur not in chain:
        chain.append(cur)
        hit = sorted(k for k, r in exports.items() if r.get("label_version") == cur)
        if len(hit) > 1:
            raise BuildError(f"{len(hit)} registered Roboflow exports name {cur}; refusing to guess")
        if hit:
            e = exports[hit[0]]
            return {"export": e, "split_file": _split_of(e, None), "how": f"registered for {cur}", "chain": chain}
        ds = [d for d in datasets if (d.get("label_version") == cur or (
            isinstance(d.get("source"), dict) and d["source"].get("kind") == "label_version"
            and d["source"].get("id") == cur)) and d.get("export") in exports]
        exps = sorted({d["export"] for d in ds})
        if len(exps) > 1:
            raise BuildError(f"datasets made from {cur} name {len(exps)} different exports; refusing to guess")
        if exps:
            d = sorted(ds, key=lambda d: d["id"])[0]
            e = exports[exps[0]]
            return {"export": e, "split_file": _split_of(e, d), "how": f"dataset {d['id']} made from {cur}",
                    "chain": chain}
        cur = lv_base(cur, lroot, recs)
    end = f"; its base chain ends at {cur}" if isinstance(cur, str) and cur not in chain else ""
    raise BuildError(f"no registered Roboflow export is linked to {' -> '.join(chain) or lv} (no roboflow_export "
                     f"with label_version, no dataset made from it with an export){end}")


def registry_records(reg) -> dict:
    if reg is None or not Path(reg).is_dir():
        raise BuildError("the registry is not configured on this portal, so the label set's images can't be found")
    try:
        return _registry(reg)
    except Exception as e:
        raise BuildError(f"the registry can't be read ({type(e).__name__})")


def _split_of(export, dataset) -> Path | None:
    p = export.get("path")
    sf = export.get("split_file")
    if p and isinstance(sf, str) and sf:
        f = Path(sf) if Path(sf).is_absolute() else Path(p) / sf
        if f.is_file():
            return f
    split = (dataset or {}).get("split")
    if isinstance(split, dict) and isinstance(split.get("file"), str) and split["file"]:
        f = Path(split["file"])
        if f.is_file():
            return f
    return None


def export_images(export_path: Path) -> tuple:
    """-> ({image key: (path, split folder or "")}, {keys found more than once}): the export folder's own images and
    those in its split folders. A key seen twice is ambiguous: left out of the first dict, never matched. Keys are
    compared case-folded (a.png and A.png stage to one file on Windows): every spelling of such a key is ambiguous."""
    from annotation import build_review_package as B
    out, dups, seen = {}, set(), {}            # seen: case-folded key -> the spellings found
    roots = [(Path(export_path), "")] + [(Path(export_path) / s, s) for s in SPLIT_DIRS]
    for root, split in roots:
        try:
            files = sorted(root.iterdir())
        except OSError:
            continue
        for p in files:
            if p.is_file() and p.suffix.lower() in IMG_SUFFIXES:
                k = B.stem_of(p.name)
                spell = seen.setdefault(k.casefold(), [])
                if spell:
                    for x in spell + [k]:
                        dups.add(x)
                        out.pop(x, None)
                else:
                    out[k] = (p, split)
                spell.append(k)
    return out, dups


def image_aliases(recs) -> dict:
    """{image key: set of every key of the same image, itself included} from the registry's image records'
    `duplicate_of` (the importer's link between a Roboflow name and the session-prefixed name of one image), joined
    both ways into clusters. Exact, never a name guess. Keys without a link are absent."""
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for k, r in (recs or {}).items():
        d = r.get("duplicate_of") if isinstance(r, dict) and r.get("schema") == "image/1" else None
        if isinstance(d, str) and d and d != k:
            parent[find(k)] = find(d)
    groups = {}
    for k in list(parent):
        groups.setdefault(find(k), set()).add(k)
    return {k: g for g in groups.values() for k in g}


def split_map(split_file) -> dict:
    """{image key: split} from a split CSV (columns image, split), else {}."""
    if not split_file:
        return {}
    try:
        with open(split_file, newline="", encoding="utf-8") as fh:
            return {r["image"]: r["split"] for r in csv.DictReader(fh) if r.get("image") and r.get("split")}
    except (OSError, KeyError, csv.Error):
        return {}


def check_images(labels: Path, export_path: Path, aliases=None, splits=None) -> dict:
    """Every label-set image found in the export with its COCO size. -> {total, found, missing, mismatched, ambiguous,
    bad_key, aliased, *_ex (first 5 keys each), files: {label-set key: [path, split]}}; split = the split file's, else
    the export folder's. Per image:
      - its key must be a plain file name (KEY_RE, no ".."): else bad_key (it names the staged file);
      - found under its own key: the width:height ratio must match the COCO entry within RATIO_TOL;
      - else through its registry alias cluster (image_aliases): only a cluster of exactly 2 keys, and only when the
        other key is on disk; then the pixel size must match the COCO width and height EXACTLY (a different crop or
        scale of the "same" image would put the outlines in the wrong place);
      - a key (own or alias) found more than once in the export, or a cluster of more than 2: ambiguous."""
    from PIL import Image
    from annotation import build_review_package as B
    coco = json.loads(Path(labels).read_text(encoding="utf-8"))
    (have, dups), aliases, splits = export_images(export_path), aliases or {}, splits or {}
    bad = {"missing": [], "mismatched": [], "ambiguous": [], "bad_key": []}
    files, aliased = {}, 0
    for im in coco.get("images") or []:
        key = B.image_key(im)
        if not isinstance(key, str) or not KEY_RE.match(key) or ".." in key:
            bad["bad_key"].append(str(key)[:80])
            continue
        via_alias = False
        if key in dups:
            bad["ambiguous"].append(key)
            continue
        got = key if key in have else None
        if got is None:
            cluster = aliases.get(key) or {key}
            if len(cluster) > 2 or any(a in dups for a in cluster):
                bad["ambiguous" if len(cluster) > 1 else "missing"].append(key)
                continue
            other = next((a for a in cluster if a != key), None)
            if other is None or other not in have:
                bad["missing"].append(key)
                continue
            got, via_alias = other, True
        p, folder = have[got]
        try:
            with Image.open(p) as img:
                w, h = img.size
            cw, ch = float(im["width"]), float(im["height"])
            ok = (w, h) == (cw, ch) if via_alias else (
                w > 0 and h > 0 and cw > 0 and ch > 0 and abs((w / h) / (cw / ch) - 1) <= RATIO_TOL)
        except (OSError, ValueError, ZeroDivisionError, TypeError, KeyError):
            ok = False
        if not ok:
            bad["mismatched"].append(key)
            continue
        aliased += via_alias
        sp = splits.get(got) or splits.get(key) or folder
        files[key] = [str(p), {"val": "valid"}.get(sp, sp)]
    total = len(coco.get("images") or [])
    out = {"total": total, "found": len(files), "aliased": aliased, "files": files}
    for k, v in bad.items():
        out[k], out[f"{k}_ex"] = len(v), v[:5]
    return out


def coverage(cfg, lv, use_cache=True) -> dict:
    """What a build of this label set would use, checked: {ok, lv, labels, total, found, missing, mismatched,
    aliased, export, export_path, how, split_file, files, problem (short text when not ok)}. Cached COVERAGE_TTL s
    per label set (the registry and the label set's file mtime in the key)."""
    if not isinstance(lv, str) or not LV_RE.match(lv):
        return {"ok": False, "lv": lv, "problem": "pick a label set"}
    lroot = labels_root(cfg)
    labels = Path(lroot) / lv / "labels.coco.json" if lroot else None
    if labels is None or not labels.is_file():
        return {"ok": False, "lv": lv, "problem": "that label set has no labels.coco.json here"}
    key = (lv, str(lroot), str(registry_root(cfg)))
    try:
        sig = labels.stat().st_mtime_ns
    except OSError:
        sig = None
    now = time.monotonic()
    with _cov_lock:
        hit = _cov_cache.get(key)
    if use_cache and hit and hit[0] == sig and now - hit[1] < COVERAGE_TTL:
        return dict(hit[2])
    out = {"ok": False, "lv": lv, "labels": str(labels)}
    try:
        recs = registry_records(registry_root(cfg))
        link = find_export(lv, lroot, registry_root(cfg), recs)
        e = link["export"]
        if not e.get("path") or not Path(e["path"]).is_dir():
            raise BuildError(f"the linked export {e.get('id')} has no folder on this computer")
        chk = check_images(labels, Path(e["path"]), image_aliases(recs), split_map(link["split_file"]))
        out.update({k: v for k, v in chk.items()})
        out.update({"export": e.get("id"), "export_path": e["path"], "how": link["how"],
                    "split_file": str(link["split_file"]) if link["split_file"] else None})
        if not chk["total"]:
            out["problem"] = "the label set has no images"
        elif chk["found"] < chk["total"]:
            out["problem"] = f"Images found {chk['found']} of {chk['total']}: " + refused_text(chk)
        else:
            out["ok"] = True
    except BuildError as e:
        out["problem"] = str(e)
    except (OSError, ValueError) as e:
        out["problem"] = f"the label set or its export can't be read ({type(e).__name__})"
    with _cov_lock:
        _cov_cache[key] = (sig, now, dict(out))
    return out


REFUSED_TEXT = (("missing", "missing"), ("mismatched", "with a different size than their labels"),
                ("ambiguous", "ambiguous (in the export more than once, or more than 2 linked names)"),
                ("bad_key", "with a name that can't be used"))


def refused_text(chk) -> str:
    """"2 missing, 1 ambiguous ...": the non-zero refusal counts."""
    return ", ".join(f"{chk[k]} {t}" for k, t in REFUSED_TEXT if chk.get(k))


def found_text(cov) -> str:
    """"Images found 3 of 3 (1 through the registry's duplicate_of link)"."""
    al = cov.get("aliased") or 0
    return f"Images found {cov['found']} of {cov['total']}" + (
        f" ({al} through the registry's duplicate_of link)" if al else "")


def coverage_line(cov: dict) -> str:
    """The one line the Start page shows (plain text; the page escapes it)."""
    if cov.get("ok"):
        return f"{found_text(cov)} · splits from {cov.get('export')}"
    return cov.get("problem") or "not ready"


# ---------------------------------------------------------------- status (both doors read it)
def building_dir(rr) -> Path:
    return Path(rr) / BUILDING


def status_path(rr, name) -> Path:
    return building_dir(rr) / name / "status.json"


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def write_status(rr, name, **fields) -> dict:
    """Merge allowlisted fields into the status (temp file + os.replace, retried) -> the new status."""
    from jobs import predict_set as PS
    p = status_path(rr, name)
    p.parent.mkdir(parents=True, exist_ok=True)
    cur = _read_json(p) or {}
    st = {k: cur.get(k) for k in STATUS_KEYS if k in cur}
    for k, v in fields.items():
        if k not in STATUS_KEYS:
            raise ValueError(f"status field {k!r} is not allowlisted")
        st[k] = v
    if fields.get("phase") == "failed" and cur.get("phase") not in TERMINAL and "failed_at" not in fields:
        st["failed_at"] = cur.get("phase")                      # the step it failed in, for the progress page
    if st.get("phase") not in PHASES:
        raise ValueError(f"phase {st.get('phase')!r} is not allowlisted")
    if st.get("reason") is not None and st["reason"] not in REASONS:
        raise ValueError(f"reason {st.get('reason')!r} is not allowlisted")
    st["name"], st["updated"] = name, _now()
    PS.write_json_atomic(p, st)
    return st


def _pid_alive(pid) -> bool:
    from jobs.predict_local import pid_alive
    try:
        return bool(pid) and pid_alive(int(pid))
    except (TypeError, ValueError):
        return False


def _age_s(iso) -> float:
    try:
        return time.time() - datetime.fromisoformat(str(iso)).timestamp()
    except (TypeError, ValueError):
        return 1e9


def read_status(rr, name) -> dict | None:
    """The status (allowlisted keys only), or None. A non-final status reads as failed "stopped" when its runner is
    gone (runner_alive: the runner holds the build lock all its life, so a free lock means no runner; a pid that is
    no longer alive too) or never started (no pid after START_GRACE_S)."""
    if not isinstance(name, str) or not NAME_RE.match(name) or rr is None:
        return None
    raw = _read_json(status_path(rr, name))
    if not isinstance(raw, dict) or raw.get("phase") not in PHASES:
        return None
    st = {k: raw.get(k) for k in STATUS_KEYS}
    if st["reason"] not in REASONS:
        st["reason"] = None
    if st["failed_at"] not in PHASES:
        st["failed_at"] = None
    if st["phase"] not in TERMINAL and _stale(rr, st):
        st.update(phase="failed", failed_at=st["phase"], reason="stopped")
    return st


def runner_alive(rr) -> bool:
    """A runner holds <review root>/.building/build.lock now (it can't be taken)."""
    f = building_dir(rr) / BUILD_LOCK
    if not f.exists():
        return False
    held = acquire_lock(f, 0)
    if held is None:
        return True
    held.close()
    return False


def _stale(rr, st) -> bool:
    if st.get("pid"):
        return not (runner_alive(rr) and _pid_alive(st["pid"]))
    return _age_s(st.get("started")) > START_GRACE_S


def active_build(rr) -> dict | None:
    """The build running now (its status), else None: a runner holds the build lock, or a status is still starting
    (no runner pid yet, within START_GRACE_S)."""
    b = building_dir(rr)
    if not b.is_dir():
        return None
    try:
        names = sorted(d.name for d in b.iterdir() if d.is_dir() and NAME_RE.match(d.name))
    except OSError:
        names = []
    live = None
    for n in names:
        st = read_status(rr, n)
        if st and st["phase"] not in TERMINAL:
            live = st
    if live:
        return live
    return {"name": None, "phase": "checking"} if runner_alive(rr) else None


# ---------------------------------------------------------------- start (the portal)
def new_name(purpose, lv, now=None) -> str:
    t = (now or datetime.now()).strftime("%Y%m%d_%H%M")
    return f"lab_{purpose}_{lv}_{t}_{secrets.token_hex(2)}"


def validate(cfg, purpose, lv, b, reviewers, owner) -> tuple:
    """-> (b dict with what the runner needs, reviewers). Raises BuildError."""
    if purpose not in PURPOSES:
        raise BuildError("pick revise or audit")
    sets = {s["id"]: s for s in label_sets(cfg)}
    if not isinstance(lv, str) or lv not in sets:
        raise BuildError("pick a label set")
    if not isinstance(b, dict) or b.get("kind") not in ("model", "label_version"):
        raise BuildError("pick a model or a label set to compare with")
    if b["kind"] == "model":
        m = next((m for m in models(cfg) if m["id"] == b.get("id")), None)
        if m is None or not MODEL_RE.match(m["id"]):
            raise BuildError("pick a model from the list")
        bd = {"kind": "model", "id": m["id"], "weights": m["weights"], "imgsz": m["imgsz"], "classes": m["classes"]}
    else:
        if b.get("id") not in sets or b.get("id") == lv:
            raise BuildError("pick a different label set to compare with")
        bd = {"kind": "label_version", "id": b["id"],
              "coco": str(Path(labels_root(cfg)) / b["id"] / "labels.coco.json")}
    revs = list(dict.fromkeys(r for r in (reviewers or []) if r))
    if len(revs) > MAX_REVIEWERS or not all(isinstance(r, str) and PID_RE.match(r) for r in revs):
        raise BuildError("reviewers must be people from the roster")
    if purpose == "audit":
        if not revs:
            raise BuildError("an audit needs at least one reviewer")
        if owner in revs:
            raise BuildError("you own this audit, so you can't review it")
    return bd, revs


def start(cfg, owner, purpose, lv, b, reviewers=(), *, roster_ids=None, spawn=True, now=None) -> str:
    """Check, then start the detached runner. -> the build name. Raises BuildError (nothing started).
    roster_ids: the portal ids reviewers must come from (the routes pass the roster's)."""
    if not isinstance(owner, str) or not PID_RE.match(owner):
        raise BuildError("your portal id can't own a review package")
    bd, revs = validate(cfg, purpose, lv, b, reviewers, owner)
    if roster_ids is not None and any(r not in roster_ids for r in revs):
        raise BuildError("reviewers must be people from the roster")
    rr = review_root(cfg)
    if rr is None or not Path(rr).is_dir():
        raise BuildError("the review folder is not configured on this portal")
    cov = coverage(cfg, lv, use_cache=False)
    if not cov.get("ok"):
        raise BuildError(cov.get("problem") or "the label set's images can't be checked")
    bdir = building_dir(rr)
    bdir.mkdir(exist_ok=True)
    lock = acquire_lock(bdir / START_LOCK, 5.0)
    if lock is None:
        raise BuildError("another build is starting; try again in a moment")
    try:
        if active_build(rr):
            raise BuildError("another review is being built; one at a time (see its progress page)")
        for _ in range(8):
            name = new_name(purpose, lv, now)
            if not any(p.exists() for p in (Path(rr) / name, Path(rr) / f"_work_{name}", bdir / name,
                                            state_root(cfg) / name)):
                break
        else:
            raise BuildError("no free name for this build; try again")
        sdir = state_root(cfg) / name
        sdir.mkdir(parents=True)
        spec = {"schema": "review_build_spec/1", "name": name, "purpose": purpose, "lv": lv, "owner": owner,
                "reviewers": revs, "b": bd, "labels": cov["labels"], "export": cov["export"],
                "review_root": str(rr), "labels_root": str(labels_root(cfg)),
                "registry": str(registry_root(cfg)) if registry_root(cfg) else None,
                "predict_root": str(predict_root(cfg)), "state": str(sdir), "wait_s": WAIT_S,
                "wait_poll_s": WAIT_POLL_S}
        (sdir / "spec.json").write_text(json.dumps(spec, indent=1), encoding="utf-8")
        write_status(rr, name, purpose=purpose, lv=lv, b_kind=bd["kind"], b_id=bd["id"], phase="checking",
                     reason=None, detail=None, started=_now(), finished=None, pid=None, by=owner)
        if spawn:                    # the runner writes its own pid; none within START_GRACE_S reads as stopped
            try:
                spawn_runner(sdir / "spec.json")
            except OSError:
                write_status(rr, name, phase="failed", reason="start", finished=_now())
                raise BuildError("the build could not be started")
    finally:
        lock.close()
    return name


def spawn_runner(spec_path: Path) -> int:
    """python -m jobs.review_build_run --spec <spec> detached (no console window; survives a portal restart). Its
    console output goes to <state>/review_build/<name>/runner.out (never shown)."""
    argv = [sys.executable, "-m", "jobs.review_build_run", "--spec", str(spec_path)]
    kw = {}
    if os.name == "nt":
        kw["creationflags"] = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kw["start_new_session"] = True
    env = {k: v for k, v in os.environ.items() if k not in ("CRYOEV_VIA", "CRYOEV_FOR")}
    with open(Path(spec_path).parent / "runner.out", "ab") as f:
        p = subprocess.Popen(argv, cwd=REPO, stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT,
                             env={**env, "PYTHONIOENCODING": "utf-8"}, close_fds=True, **kw)
    return p.pid


# ---------------------------------------------------------------- run (the detached runner)
STAGE_RE = re.compile(r"[0-9a-f]{8}")   # <review root>/.building/<short_id>: a build's staging (fullmatch)


def short_id(name) -> str:
    """The build's 8-hex staging folder name: sha1(name)[:8]. Deterministic, so the status folder .building/<name>/
    maps to it without storing it. Short on purpose (fixer 12): the long portal name twice in one path (the
    package folder and _work_<name>) plus a 79-char image key passed Windows MAX_PATH on the real review root."""
    return hashlib.sha1(str(name).encode("utf-8")).hexdigest()[:8]


def staging_root(rr, name) -> Path:
    """<review root>/.building/<short_id>: everything a build stages (removed whole after the build)."""
    return building_dir(rr) / short_id(name)


def staging_dir(rr, name) -> Path:
    """<review root>/.building/<short_id>/p: the package is built in p/<name>, under a dot folder nothing lists or
    serves, and renamed into the review root only once it is complete and checked."""
    return staging_root(rr, name) / "p"


def work_staging_dir(rr, name) -> Path:
    """<review root>/.building/<short_id>/w: an audit's work folder (the answer key); published as _work_<name>."""
    return staging_root(rr, name) / "w"


def compare_args(spec, staged: Path, split_csv: Path) -> list:
    """compare_package.py's arguments for this build (images: the staged folder; splits: the staged CSV; out: the
    staging folder)."""
    b = spec["b"]
    args = ["--labels", spec["labels"], "--purpose", spec["purpose"], "--owner", spec["owner"],
            "--out", str(staging_dir(spec["review_root"], spec["name"]) / spec["name"]), "--built-by", "portal",
            "--source-title", spec["lv"], "--images-root", str(staged), "--split-csv", str(split_csv)]
    if b["kind"] == "model":
        args += ["--b-model", b["weights"], "--imgsz", str(b["imgsz"]), "--pred-classes", b["classes"],
                 "--b-label", b["id"]]
    else:
        args += ["--b-coco", b["coco"], "--b-label", b["id"]]
    if spec["purpose"] == "revise":
        args += ["--work-dir", str(Path(spec["state"]) / "work")]
    else:                                   # an audit: a short staging path; a fresh decoy seed per build (fixer 14),
        args += ["--work-dir", str(work_staging_dir(spec["review_root"], spec["name"])),    # kept only in _work_
                 "--seed", str(secrets.randbelow(2 ** 31))]
    for r in spec.get("reviewers") or []:
        args += ["--reviewer", r]
    return args


def stage(spec, cov) -> tuple:
    """The checked images under their label-set keys in <state>/review_build/<name>/images (hard links, else copies),
    and their splits in <state>/.../splits.csv. -> (images folder, split CSV). Never under the review root.
    Two keys that stage to one file (case-folded names) raise FileExistsError: a refusal, never a copy over it."""
    sdir = Path(spec["state"])
    flat = sdir / "images"
    flat.mkdir(parents=True, exist_ok=False)
    rows = []
    for key, (src, split) in sorted(cov["files"].items()):
        src = Path(src)
        dst = flat / f"{key}.rf.0{src.suffix.lower()}"      # compare_package reads the key back (stem_of)
        try:
            os.link(src, dst)
        except FileExistsError:
            raise
        except OSError:
            with open(src, "rb") as fi, open(dst, "xb") as fo:     # "x": an existing file is FileExistsError
                shutil.copyfileobj(fi, fo)
            shutil.copystat(src, dst)
        rows.append((key, split or "train"))
    sc = sdir / "splits.csv"
    with open(sc, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "split"])
        w.writerows(rows)
    return flat, sc


class _Cfg:
    """The few settings coverage() reads, from a spec (the runner has no portal config)."""

    def __init__(self, spec):
        self.review_root = spec["review_root"]
        self.labels_root = spec["labels_root"]
        self.registry = spec.get("registry")
        self.state_dir = str(Path(spec["state"]).parent.parent)


def wait_for_gpu(spec, set_status, sleep=time.sleep, clock=time.time):
    """Take <predict root>/predict.lock, then wait for a free GPU (jobs.predict_local.gpu_check), up to the spec's
    wait_s. -> (the lock file, None) to run, or (None, reason code) to fail."""
    from jobs import predict_local as PL
    lock_path = Path(spec["predict_root"]) / PL.LOCK_FILE
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = clock() + float(spec.get("wait_s", WAIT_S))
    poll = float(spec.get("wait_poll_s", WAIT_POLL_S))
    lock, said = None, None
    while True:
        if lock is None:
            lock = acquire_lock(lock_path, 0)
        if lock is None:
            if said != "lock":
                set_status(phase="waiting", reason="lock")
                said = "lock"
        else:
            device, why, kind = PL.gpu_check(False)
            if device:
                return lock, None
            if kind == "missing":
                lock.close()
                return None, "gpu_missing"
            if said != "gpu_busy":
                set_status(phase="waiting", reason="gpu_busy")
                said = "gpu_busy"
            lock.close()                           # don't hold the queue while someone else's job has the GPU
            lock = None
        if clock() >= deadline:
            if lock is not None:
                lock.close()
            return None, "gpu_timeout"
        sleep(max(0.05, min(poll, deadline - clock())))


def finished(folder: Path, name: str) -> bool:
    """folder holds a finished portal build of `name` (its review_package.json: id and built_by "portal")."""
    meta = _read_json(Path(folder) / "review_package.json")
    return isinstance(meta, dict) and meta.get("id") == name and meta.get("built_by") == "portal"


def sweep(rr: Path, own: str, warn=None) -> dict:
    """Run by the runner while it holds the build lock (so no other runner lives): every other non-final status is
    rewritten failed "stopped", and the staging left by killed builds is removed: ONLY <review root>/.building/
    <short_id> folders (STAGE_RE), never own's. Nothing in the review root itself is ever deleted: a <name> folder
    without a finished review_package.json is only reported (warn(text)), and a _work_* folder is never touched
    (or opened). -> counts."""
    n = {"stopped": 0, "staging": 0, "unfinished": 0}
    b = building_dir(rr)
    try:
        for d in sorted(b.iterdir()) if b.is_dir() else []:
            if not d.is_dir() or not NAME_RE.match(d.name) or d.name == own:
                continue
            raw = _read_json(d / "status.json")
            if isinstance(raw, dict) and raw.get("phase") in PHASES and raw["phase"] not in TERMINAL:
                write_status(rr, d.name, phase="failed", failed_at=raw["phase"], reason="stopped", finished=_now())
                n["stopped"] += 1
            if (d / "pkg").is_dir():                # the staging of builds before the short layout (fixer 12)
                shutil.rmtree(d / "pkg", ignore_errors=True)
                n["staging"] += 1
        mine = short_id(own)
        for d in sorted(b.iterdir()) if b.is_dir() else []:
            if d.is_dir() and STAGE_RE.fullmatch(d.name) and d.name != mine:
                shutil.rmtree(d, ignore_errors=True)
                n["staging"] += 1
        for d in sorted(Path(rr).iterdir()):
            if d.name != own and NAME_RE.match(d.name) and d.is_dir() and not finished(d, d.name):
                n["unfinished"] += 1
                if warn is not None:
                    warn(f"left alone: review-root folder {d.name} has no finished review_package.json")
    except OSError:
        pass
    return n


class _LogLines:
    """sys.stdout / sys.stderr for compare_package run in this process: every line to the raw log, its
    "[phase] ..." markers to the status."""

    def __init__(self, fh, say):
        self.fh, self.say, self.buf = fh, say, ""

    def write(self, text):
        self.fh.write(text)
        self.fh.flush()
        self.buf += text
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            s = line.strip()
            if s == "[phase] predicting":
                self.say(phase="predicting")
            elif s == "[phase] writing":
                self.say(phase="writing")
        return len(text)

    def flush(self):
        self.fh.flush()

    def isatty(self):
        return False


def run_compare(args, log: Path, say) -> int:
    """compare_package.build() IN this process (QC 1): YOLO, when B is a model, runs here too, so killing the runner
    kills the GPU work with it: no orphaned child under a dead runner. Output -> the raw log; markers -> the status.
    -> 0 built, 1 refused or failed."""
    import contextlib
    import traceback
    from annotation import compare_package as CP
    with open(log, "a", encoding="utf-8", errors="replace") as fh:
        out = _LogLines(fh, say)
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            try:
                CP.build(CP.parser().parse_args(args))
                return 0
            except SystemExit as e:                  # a refusal (or argparse): its message goes to the log
                print(f"[refused] {e.code}")
                return 1
            except Exception:
                traceback.print_exc()
                return 1


def publish(rr: Path, name: str, purpose: str) -> None:
    """Move the checked build from its staging folder into the review root: an audit's _work_ first (a package is
    never there without its key), then the package. Renames retried (antivirus / indexer hold folders briefly)."""
    from jobs.predict_local import rename_retry
    st = staging_dir(rr, name)
    if (Path(rr) / name).exists() or (Path(rr) / f"_work_{name}").exists():
        raise OSError("the package or its work folder already exists in the review root")
    if purpose == "audit":
        rename_retry(work_staging_dir(rr, name), Path(rr) / f"_work_{name}")
    rename_retry(st / name, Path(rr) / name)


def chain_next_prediction(spec, log_note):
    """After predict.lock is released (a model build): start the oldest queued laptop prediction set of that root,
    one step down the queue, as jobs/predict_local.py run(chain=True) does. Never fails the build."""
    try:
        from jobs import predict_local as PL
        nxt = PL.next_queued(Path(spec["predict_root"]))
        if nxt is not None:
            PL.start_detached(nxt, Path(spec["predict_root"]), spec.get("registry"), chain=True)
            log_note(f"chained: started the next queued prediction set {nxt.name}")
    except Exception as e:
        log_note(f"next queued prediction not started: {type(e).__name__}: {e}")


def run(spec_path) -> int:
    """The runner's body (jobs/review_build_run.py). -> exit code (0 done, 1 failed, 3 another build holds the lock)."""
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    name, rr = spec["name"], Path(spec["review_root"])
    if not NAME_RE.match(name):
        return 1
    log = Path(spec["state"]) / "build.log"

    def say(**f):
        return write_status(rr, name, **f)

    def note(text):
        with open(log, "a", encoding="utf-8") as fh:
            fh.write(f"[{_now()}] {text}\n")

    lock = acquire_lock(building_dir(rr) / BUILD_LOCK, 30.0)
    if lock is None:
        say(phase="failed", reason="busy", finished=_now(), pid=os.getpid())
        return 3
    gpu_lock = None
    stage_dir, stage_root = staging_dir(rr, name), staging_root(rr, name)
    try:
        say(phase="checking", pid=os.getpid(), reason=None)
        swept = sweep(rr, name, warn=note)
        if any(swept.values()):
            note(f"swept leftovers of earlier builds: {swept}")
        cov = coverage(_Cfg(spec), spec["lv"], use_cache=False)
        if not cov.get("ok") or cov.get("export") != spec.get("export"):
            note(f"inputs refused: {cov.get('problem') or 'the linked export changed since the build was started'}")
            detail = (f"{found_text(cov)}; " + refused_text(cov)) if cov.get("total") else None
            say(phase="failed", reason="images" if cov.get("total") else "inputs", detail=detail, finished=_now())
            return 1
        say(detail=f"{found_text(cov)} · sizes match")
        try:
            staged, splits = stage(spec, cov)
        except FileExistsError:
            note("refused: two images stage to one file name (names that differ only in case)")
            say(phase="failed", reason="images", finished=_now())
            return 1
        if (rr / name).exists() or (rr / f"_work_{name}").exists() or stage_root.exists():
            note("refused: the package, its work folder or its staging folder already exists")
            say(phase="failed", reason="build", finished=_now())
            return 1
        if spec["b"]["kind"] == "model":
            gpu_lock, why = wait_for_gpu(spec, lambda **f: say(**f))
            if gpu_lock is None:
                say(phase="failed", reason=why, finished=_now())
                return 1
            say(phase="predicting", reason=None)
        else:
            say(phase="writing", reason=None)
        stage_dir.mkdir(parents=True)
        rc = run_compare(compare_args(spec, staged, splits), log, say)
        if gpu_lock is not None:                # the GPU work is over: free the queue before writing the rest
            gpu_lock.close()
            gpu_lock = None
            chain_next_prediction(spec, note)
        if rc != 0:
            note(f"compare_package failed ({rc})")
            say(phase="failed", reason="build", finished=_now())
            return 1
        pkg = stage_dir / name
        meta = _read_json(pkg / "review_package.json")
        if not isinstance(meta, dict) or meta.get("id") != name or meta.get("built_by") != "portal" \
                or meta.get("purpose") != spec["purpose"] or meta.get("n_images") != cov["found"]:
            note("the finished package's review_package.json is missing or not this build's: removed")
            say(phase="failed", reason="check", finished=_now())
            return 1
        meta["path"] = str((rr / name).resolve())          # where it will live (it was built in the staging folder)
        from annotation.compare_package import write_json_atomic
        write_json_atomic(pkg / "review_package.json", meta)
        publish(rr, name, spec["purpose"])
        try:
            register(spec, meta)
        except Exception as e:                  # the package is usable; the line can be added by hand
            note(f"registry line not written: {type(e).__name__}: {e}")
            say(phase="done", reason="registry", finished=_now())
            return 0
        say(phase="done", reason=None, finished=_now())
        return 0
    except BaseException as e:
        try:
            note(f"runner error: {type(e).__name__}: {e}")
            say(phase="failed", reason="build", finished=_now())
        except Exception:
            pass
        return 1
    finally:
        if gpu_lock is not None:
            gpu_lock.close()
            chain_next_prediction(spec, note)
        shutil.rmtree(stage_root, ignore_errors=True)       # gone once published; a failed build's leftovers too
        shutil.rmtree(Path(spec["state"]) / "images", ignore_errors=True)    # staged links / copies
        lock.close()


def registry_line(spec, meta) -> dict:
    """The allowlisted review_package line (ALLOWED_LINE_KEYS): never `cards`, `path`, reviewers or decoy counts."""
    cmp = meta.get("compare") if isinstance(meta.get("compare"), dict) else {}
    line = {"schema": "review_package/1", "id": spec["name"], "kind": "compare", "purpose": spec["purpose"],
            "base": spec["lv"], "base_kind": "label_version", "n_images": int(meta.get("n_images") or 0),
            "built": meta.get("built"), "status": "built", "owner": spec["owner"], "built_by": "portal",
            "flag_conf": cmp.get("flag_conf"), "tool": TOOL, "recorded_by": RECORDED_BY}
    if spec["b"]["kind"] == "model":
        line["model"] = spec["b"]["id"]
    else:
        line["b_label_version"] = spec["b"]["id"]
    return {k: v for k, v in line.items() if k in ALLOWED_LINE_KEYS and v is not None}


def register(spec, meta):
    """Append the line under <registry>/.review_package.lock unless the id is already there. None registry: skipped."""
    reg = spec.get("registry")
    if not reg:
        return None
    from jobs import predict_set as PS
    from registry import store
    line = registry_line(spec, meta)
    with PS.locked(Path(reg) / REVIEW_PACKAGE_LOCK):
        if line["id"] in store.current("review_package", Path(reg)):
            return None
        return store.append(line, Path(reg), via=None)


# ---------------------------------------------------------------- object sets (round 14, contract C4)
# Start a review -> Object set (admin): an items CSV (one row per object), a label schema, the prediction set the
# objects come from, the CSV columns to hide from reviewers (blind), the outlines on the cards and the reviewers.
# start_objects() checks everything in the request, takes the one-at-a-time lock (<review root>/.building/objects.lock),
# saves the CSV under <portal state>/review_build/objects/<job>/items.csv (never under the review root), writes that
# job's status.json there and starts a background thread (QC 3: the POST never waits for a build). The thread calls
# annotation/object_set.build (B3) with the prediction set's folder (<predict root>/<ps>), the CSV's own image column
# (image | image_id | stem) and key column (line, else particle_id), then object_set.register into the portal's
# registry (so the Reviews hub and the closed rule see it). The CSV is deleted after the build, success or not
# (items.json keeps its sha256). The progress page /reviews/start/<job> reads the status: the allowlisted phases,
# the package id when done, and at most 3 problem lines of a refusal with paths scrubbed.
OBJ_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _.()+-]{0,59}\Z")
OBJ_SCHEMA_RE = re.compile(r"^ls_[a-z0-9][a-z0-9_]{0,60}_v\d{1,4}\Z")
OBJ_PS_RE = re.compile(r"^ps_\d{8}_[0-9a-f]{6}\Z")
OBJ_COL_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.+-]{0,63}\Z")
OBJ_ID_RE = re.compile(r"^pred_obj_[A-Za-z0-9][A-Za-z0-9_.-]{0,110}\Z")
OBJ_JOB_RE = re.compile(r"^obj_\d{8}_\d{6}_[0-9a-f]{4}\Z")
OUTLINE_MODES = ("target", "target+contained")
ITEMS_MAX_BYTES = 2_000_000
ITEMS_MAX_ROWS = 2000                  # = annotation/object_set.MAX_ITEMS (a test keeps them equal)
ITEMS_MAX_COLS = 80
SLUG_MAX = 30                          # annotation/object_set.SLUG_RE: [a-z0-9][a-z0-9_]{0,29}
IMAGE_COLS = ("image", "image_id", "stem")      # the items CSV's image column, the first one present
# columns that name the object (shown to reviewers as the item's image / object; never offered for blinding)
OBJ_KEY_COLS = ("id", "item", "image", "image_id", "stem", "obj", "pkg_key", "line", "particle_id", "crop")
OBJ_LOCK = "objects.lock"
OBJ_SUB = "objects"
OBJ_PHASES = ("checking", "building", "registering", "done", "failed")
OBJ_PHASE_TEXT = {"checking": "Checking the items", "building": "Building the set", "registering": "Recording it",
                  "done": "Done", "failed": "Failed"}
OBJ_STATUS_KEYS = ("name", "phase", "failed_at", "pkg", "problems", "n_items", "title", "by", "started", "updated",
                   "finished")
OBJ_PROBLEMS_MAX = 3
_PATHISH = re.compile(r"[A-Za-z]:[\\/]\S*|(?:/[^\s/]+){2,}")
_obj_running: set = set()              # job names whose thread runs in this process (a restart: none)
_obj_lock = threading.Lock()


def parse_items_csv(data: bytes) -> dict:
    """{"columns", "rows": n, "blindable": [non-key columns], "image_col", "line_col", "particle_col"} of an items
    CSV, or BuildError. UTF-8 (a BOM is fine), read with the csv module (a quoted cell may hold a newline), a header
    row of plain column names, 1..ITEMS_MAX_ROWS rows, an image column and a line or particle_id column."""
    if not isinstance(data, (bytes, bytearray)) or not bytes(data).strip():
        raise BuildError("the items CSV is empty")
    if len(data) > ITEMS_MAX_BYTES:
        raise BuildError(f"the items CSV is larger than {ITEMS_MAX_BYTES // 1_000_000} MB")
    try:
        text = bytes(data).decode("utf-8-sig")
    except UnicodeDecodeError:
        raise BuildError("the items CSV is not UTF-8 text") from None
    try:
        rows = list(csv.reader(io.StringIO(text, newline="")))
    except csv.Error:
        raise BuildError("the items CSV could not be read") from None
    rows = [r for r in rows if any(c.strip() for c in r)]
    if len(rows) < 2:
        raise BuildError("the items CSV needs a header row and at least one item")
    cols = [c.strip() for c in rows[0]]
    if len(cols) > ITEMS_MAX_COLS or any(not OBJ_COL_RE.match(c) for c in cols) or len(set(cols)) != len(cols):
        raise BuildError("the items CSV's header must be plain, distinct column names")
    if len(rows) - 1 > ITEMS_MAX_ROWS:
        raise BuildError(f"the items CSV has more than {ITEMS_MAX_ROWS} items")
    low = {c.lower(): c for c in cols}
    image_col = next((low[c] for c in IMAGE_COLS if c in low), None)
    if image_col is None:
        raise BuildError("the items CSV needs an image column (image, image_id or stem)")
    line_col = low.get("line")
    particle_col = None if line_col else low.get("particle_id")
    if line_col is None and particle_col is None:
        raise BuildError("the items CSV needs a line or particle_id column (which object of the image)")
    return {"columns": cols, "rows": len(rows) - 1, "blindable": [c for c in cols if c.lower() not in OBJ_KEY_COLS],
            "image_col": image_col, "line_col": line_col, "particle_col": particle_col}


def object_slug(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(name or "").lower()).strip("_")[:SLUG_MAX].strip("_")
    return slug or "set"


def schema_choices(cfg) -> list:
    """[{id, name, version}] of the registered / shipped label schemas (portal/modules/data/labelmeta.py)."""
    from portal.modules.data import labelmeta as LM
    out = []
    for sid, s in LM.list_schemas(registry_root(cfg)).items():
        v = s.get("version")
        out.append({"id": sid, "name": str(s.get("name") or sid),
                    "version": v if isinstance(v, int) and not isinstance(v, bool) else None})
    return out


def ps_choices(cfg, owner) -> list:
    """[{id, name, images}] of the admin's OWN finished prediction sets (newest first) an object set can be built
    from: the Start page's rule for (a); someone else's set is never listed or accepted."""
    from portal import names
    from portal.predict_parts import sets as SETS
    out = []
    for d, req, st in SETS.list_sets(cfg, {"id": owner}, lambda u, o: bool(owner) and o == owner):
        if st.get("state") != "done" or not OBJ_PS_RE.match(d.name):
            continue
        n = st.get("n_total")
        out.append({"id": d.name, "name": names.display_name(req) or d.name,
                    "images": n if isinstance(n, int) else None})
    return out


def _object_set_build(**kw):
    """annotation/object_set.build (B3, contract C4), called with its own keyword arguments: ps_dir, items_csv,
    schema, review_root, image_col, line_col, particle_col, blind_cols, outline_mode, slug, owner, reviewers,
    registry_root, built_by, title. -> the package folder."""
    from annotation import object_set as OS
    return OS.build(**kw)


def _object_set_register(pkg, reg):
    """annotation/object_set.register: the set's review_package line into the portal's registry."""
    from annotation import object_set as OS
    return OS.register(pkg, reg)


def problem_lines(exc) -> list:
    """At most OBJ_PROBLEMS_MAX short lines of a builder refusal, paths scrubbed: its "  - ..." lines, else its
    first line."""
    text = str(exc).strip()
    lines = [ln.strip()[2:].strip() for ln in text.splitlines() if ln.strip().startswith("- ")]
    if not lines:
        lines = (text.splitlines() or [type(exc).__name__])[:1]
    return [_PATHISH.sub("…", ln)[:200] for ln in lines[:OBJ_PROBLEMS_MAX]]


def obj_status_path(cfg, job) -> Path:
    return state_root(cfg) / OBJ_SUB / job / "status.json"


def write_obj_status(cfg, job, **fields) -> dict:
    from jobs import predict_set as PS
    p = obj_status_path(cfg, job)
    cur = _read_json(p) or {}
    st = {k: cur.get(k) for k in OBJ_STATUS_KEYS if k in cur}
    for k, v in fields.items():
        if k not in OBJ_STATUS_KEYS:
            raise ValueError(f"status field {k!r} is not allowlisted")
        st[k] = v
    if fields.get("phase") == "failed" and cur.get("phase") not in ("done", "failed") and "failed_at" not in fields:
        st["failed_at"] = cur.get("phase")
    if st.get("phase") not in OBJ_PHASES:
        raise ValueError(f"phase {st.get('phase')!r} is not allowlisted")
    st["name"], st["updated"] = job, _now()
    PS.write_json_atomic(p, st)
    return st


def read_obj_status(cfg, job) -> dict | None:
    """An object set build's status (allowlisted keys), or None. A non-final status whose thread is not running in
    this process (the portal restarted) reads as failed."""
    if not isinstance(job, str) or not OBJ_JOB_RE.match(job):
        return None
    raw = _read_json(obj_status_path(cfg, job))
    if not isinstance(raw, dict) or raw.get("phase") not in OBJ_PHASES:
        return None
    st = {k: raw.get(k) for k in OBJ_STATUS_KEYS}
    st["problems"] = [str(x)[:200] for x in st["problems"] if isinstance(x, str)][:OBJ_PROBLEMS_MAX] \
        if isinstance(st["problems"], list) else []
    if st["phase"] not in ("done", "failed"):
        with _obj_lock:
            alive = job in _obj_running
        if not alive:
            st.update(phase="failed", failed_at=st["phase"], problems=["the build stopped (the portal restarted)"])
    return st


def start_objects(cfg, owner, *, name, schema, ps, items_csv: bytes, blind_cols=(), outline_mode="target",
                  reviewers=(), roster_ids=None, schemas=None, sets=None, build=None, register=None, now=None,
                  background=True) -> str:
    """Check, then start building an object set (a background thread unless background=False). -> the job name
    (obj_<stamp>_<4hex>; its progress page is /reviews/start/<job>). Raises BuildError (nothing started)."""
    if not isinstance(owner, str) or not PID_RE.match(owner):
        raise BuildError("your portal id can't own a review package")
    if not isinstance(name, str) or not OBJ_NAME_RE.match(name.strip()):
        raise BuildError("give the set a name (letters, digits, spaces, at most 60)")
    schemas = schema_choices(cfg) if schemas is None else schemas
    if not isinstance(schema, str) or not OBJ_SCHEMA_RE.match(schema) or schema not in {s["id"] for s in schemas}:
        raise BuildError("pick a registered label schema")
    sets = ps_choices(cfg, owner) if sets is None else sets
    if not isinstance(ps, str) or not OBJ_PS_RE.match(ps) or ps not in {s["id"] for s in sets}:
        raise BuildError("pick a finished prediction set")
    if outline_mode not in OUTLINE_MODES:
        raise BuildError("pick which outlines the cards show")
    info = parse_items_csv(items_csv)
    blind = list(dict.fromkeys(str(c) for c in blind_cols or ()))
    if any(c not in info["blindable"] for c in blind):
        raise BuildError("only the CSV's own non-key columns can be hidden")
    revs = list(dict.fromkeys(str(r) for r in reviewers or ()))
    if len(revs) > MAX_REVIEWERS or any(not PID_RE.match(r) for r in revs):
        raise BuildError(f"reviewers must be portal ids (at most {MAX_REVIEWERS})")
    if roster_ids is not None and any(r not in roster_ids for r in revs):
        raise BuildError("reviewers must be people from the roster")
    rr = review_root(cfg)
    if rr is None or not Path(rr).is_dir():
        raise BuildError("the review folder is not configured on this portal")
    ps_dir = Path(predict_root(cfg)) / ps
    if not ps_dir.is_dir():
        raise BuildError("that prediction set's folder is not on this computer")
    bdir = building_dir(rr)
    bdir.mkdir(exist_ok=True)
    lock = acquire_lock(bdir / OBJ_LOCK, 2.0)
    if lock is None:
        raise BuildError("another object set is being built; one at a time")
    try:
        job = f"obj_{(now or datetime.now()).strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(2)}"
        sdir = state_root(cfg) / OBJ_SUB / job
        sdir.mkdir(parents=True)
        items = sdir / "items.csv"
        items.write_bytes(bytes(items_csv))
        write_obj_status(cfg, job, phase="checking", pkg=None, problems=[], n_items=info["rows"],
                         title=name.strip(), by=owner, started=_now(), finished=None, failed_at=None)
        kw = dict(ps_dir=ps_dir, items_csv=items, schema=schema, review_root=Path(rr), image_col=info["image_col"],
                  line_col=info["line_col"], particle_col=info["particle_col"], blind_cols=blind,
                  outline_mode=outline_mode, slug=object_slug(name), owner=owner, reviewers=revs,
                  registry_root=registry_root(cfg), built_by="portal", title=name.strip())
        with _obj_lock:
            _obj_running.add(job)
    except BaseException:
        lock.close()
        raise
    args = (cfg, job, kw, items, lock, build or _object_set_build, register or _object_set_register)
    if background:
        threading.Thread(target=_run_objects, args=args, name=f"objset-{job}", daemon=True).start()
    else:
        _run_objects(*args)
    return job


def _run_objects(cfg, job, kw, items: Path, lock, build, register):
    """The build thread: object_set.build, then register; the status after each step; the CSV deleted at the end
    (success or not); the lock released last."""
    pkg = None
    try:
        write_obj_status(cfg, job, phase="building")
        try:
            got = build(**kw)
            pkg = Path(got) if isinstance(got, (str, Path)) else Path(kw["review_root"]) / str((got or {}).get("id"))
            if not OBJ_ID_RE.match(pkg.name):
                raise BuildError("the object set was built but its id could not be read")
        except Exception as e:                          # noqa: BLE001  the builder's refusal: short lines, no paths
            write_obj_status(cfg, job, phase="failed", problems=problem_lines(e), finished=_now())
            return
        write_obj_status(cfg, job, phase="registering", pkg=pkg.name)
        reg = kw.get("registry_root")
        if reg is not None:
            try:
                register(pkg, reg)
            except Exception as e:                      # noqa: BLE001  not recorded: never leave an unlisted set
                shutil.rmtree(pkg, ignore_errors=True)
                write_obj_status(cfg, job, phase="failed", pkg=None, finished=_now(),
                                 problems=["built, but its registry line was not written, so the set was removed"]
                                 + problem_lines(e)[:OBJ_PROBLEMS_MAX - 1])
                return
        write_obj_status(cfg, job, phase="done", finished=_now())
    except Exception:                                   # noqa: BLE001  a status write failed: the page reads stopped
        pass
    finally:
        try:
            items.unlink()
        except OSError:
            pass
        with _obj_lock:
            _obj_running.discard(job)
        lock.close()
