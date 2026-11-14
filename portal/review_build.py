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
  - every label-set image must be found in the export, by its image key or else by the one registry alias of it
    (image records' `duplicate_of`, both ways: the importer's link between a Roboflow name and the session-prefixed
    name of the same image), with the same width:height ratio as its COCO entry (1 % tolerance). Any missing or
    mismatched image refuses the build, with the counts; "Images found N of M" is shown before starting. The runner
    stages the checked images under their label-set keys (hard links, else copies) in the state folder and passes
    that folder and a split CSV to compare_package.

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
  - B is a model (`--b-model`, YOLO on the GPU): the runner takes <predict root>/predict.lock itself (as
    jobs/predict_local.py does) and waits for jobs.predict_local.gpu_check (run_job.gpu_state) to find the GPU free,
    up to WAIT_S; B is a label set (`--b-coco`): no GPU, no lock;
  - after a successful build the runner appends an allowlisted review_package registry line (ALLOWED_LINE_KEYS: no
    `cards`, no `path`, no decoy counts) under <registry>/.review_package.lock, recorded_by "portal/review_build.py"
    (as portal/predict_parts/sets.py register_package); the registry importer skips the package (built_by "portal").
"""
from __future__ import annotations

import csv
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
COMPARE = REPO / "annotation" / "compare_package.py"
TOOL = "annotation/compare_package.py"
RECORDED_BY = "portal/review_build.py"
PURPOSES = ("revise", "audit")
LV_RE = re.compile(r"^lv_\d{8}_\d{4}$")
MODEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_]{0,80}$")
PID_RE = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")          # a portal id (compare_package REVIEWER_RE)
NAME_RE = re.compile(r"^lab_(?:revise|audit)_lv_\d{8}_\d{4}_\d{8}_\d{4}_[0-9a-f]{4}$")
MAX_REVIEWERS = 12
BUILDING = ".building"                 # <review root>/.building/<name>/status.json; a dot folder: never a package
BUILD_LOCK = "build.lock"              # held by the runner for its whole life: one build at a time
START_LOCK = "start.lock"              # held by start() while it checks and starts
STATE_SUB = "review_build"             # <portal state>/review_build/<name>/
REVIEW_PACKAGE_LOCK = ".review_package.lock"   # in the registry folder (portal/predict_parts/sets.py)
RATIO_TOL = 0.01
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


def export_images(export_path: Path) -> dict:
    """{image key: (path, split folder or "")}: the export folder's own images and those in its split folders."""
    from annotation import build_review_package as B
    out = {}
    roots = [(Path(export_path), "")] + [(Path(export_path) / s, s) for s in SPLIT_DIRS]
    for root, split in roots:
        try:
            files = sorted(root.iterdir())
        except OSError:
            continue
        for p in files:
            if p.is_file() and p.suffix.lower() in IMG_SUFFIXES:
                out.setdefault(B.stem_of(p.name), (p, split))
    return out


def image_aliases(recs) -> dict:
    """{image key: [other keys of the same image]} from the registry's image records: `duplicate_of` both ways (the
    importer's link between a Roboflow name and the session-prefixed name of one image). Exact, never a name guess."""
    out = {}
    for k, r in (recs or {}).items():
        d = r.get("duplicate_of") if r.get("schema") == "image/1" else None
        if isinstance(d, str) and d and d != k:
            out.setdefault(k, []).append(d)
            out.setdefault(d, []).append(k)
    return out


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
    """Every label-set image found in the export (by its key, else by one registry alias of it: image_aliases) with
    its COCO width:height ratio (RATIO_TOL). -> {total, found, missing, mismatched, aliased, missing_ex,
    mismatched_ex, files: {label-set key: [path, split]}}; split = the split file's, else the export folder's."""
    from PIL import Image
    from annotation import build_review_package as B
    coco = json.loads(Path(labels).read_text(encoding="utf-8"))
    have, aliases, splits = export_images(export_path), aliases or {}, splits or {}
    missing, mismatched, files, aliased = [], [], {}, 0
    for im in coco.get("images") or []:
        key = B.image_key(im)
        got = key if key in have else None
        if got is None:
            alts = sorted({a for a in aliases.get(key, []) if a in have})
            got = alts[0] if len(alts) == 1 else None          # two different aliases on disk: not a match
            aliased += got is not None
        if got is None:
            missing.append(key)
            continue
        p, folder = have[got]
        try:
            with Image.open(p) as img:
                w, h = img.size
            ok = w > 0 and h > 0 and im.get("width") and im.get("height") and \
                abs((w / h) / (float(im["width"]) / float(im["height"])) - 1) <= RATIO_TOL
        except (OSError, ValueError, ZeroDivisionError, TypeError):
            ok = False
        if not ok:
            mismatched.append(key)
            continue
        sp = splits.get(got) or splits.get(key) or folder
        files[key] = [str(p), {"val": "valid"}.get(sp, sp)]
    total = len(coco.get("images") or [])
    return {"total": total, "found": len(files), "missing": len(missing), "mismatched": len(mismatched),
            "aliased": aliased, "missing_ex": missing[:5], "mismatched_ex": mismatched[:5], "files": files}


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
        out.update({k: chk[k] for k in ("total", "found", "missing", "mismatched", "aliased", "missing_ex",
                                        "mismatched_ex", "files")})
        out.update({"export": e.get("id"), "export_path": e["path"], "how": link["how"],
                    "split_file": str(link["split_file"]) if link["split_file"] else None})
        if not chk["total"]:
            out["problem"] = "the label set has no images"
        elif chk["missing"] or chk["mismatched"]:
            out["problem"] = (f"Images found {chk['found']} of {chk['total']}: {chk['missing']} missing, "
                              f"{chk['mismatched']} with a different width:height than their labels")
        else:
            out["ok"] = True
    except BuildError as e:
        out["problem"] = str(e)
    except (OSError, ValueError) as e:
        out["problem"] = f"the label set or its export can't be read ({type(e).__name__})"
    with _cov_lock:
        _cov_cache[key] = (sig, now, dict(out))
    return out


def coverage_line(cov: dict) -> str:
    """The one line the Start page shows (plain text; the page escapes it)."""
    if cov.get("ok"):
        return f"Images found {cov['found']} of {cov['total']} · splits from {cov.get('export')}"
    if cov.get("total"):
        return cov.get("problem") or "not ready"
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
    """The status (allowlisted keys only), or None. A non-final status whose runner is gone (or never started) reads
    as failed "stopped"."""
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
    if st["phase"] not in TERMINAL and _stale(st):
        st.update(phase="failed", failed_at=st["phase"], reason="stopped")
    return st


def _stale(st) -> bool:
    if st.get("pid"):
        return not _pid_alive(st["pid"])
    return _age_s(st.get("started")) > START_GRACE_S


def active_build(rr) -> dict | None:
    """The build running now (its status), else None: the runner's lock is held, or a status is not final and its
    runner is alive (or still starting)."""
    b = building_dir(rr)
    if not b.is_dir():
        return None
    held = acquire_lock(b / BUILD_LOCK, 0) if (b / BUILD_LOCK).exists() else None
    busy = (b / BUILD_LOCK).exists() and held is None
    if held is not None:
        held.close()
    live = None
    try:
        names = sorted(d.name for d in b.iterdir() if d.is_dir() and NAME_RE.match(d.name))
    except OSError:
        names = []
    for n in names:
        st = read_status(rr, n)
        if st and st["phase"] not in TERMINAL:
            live = st
    if live:
        return live
    return {"name": None, "phase": "checking"} if busy else None


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
def compare_argv(spec, staged: Path, split_csv: Path) -> list:
    """compare_package.py's command line for this build (images: the staged folder; splits: the staged CSV)."""
    rr, b = Path(spec["review_root"]), spec["b"]
    argv = [sys.executable, "-u", str(COMPARE), "--labels", spec["labels"], "--purpose", spec["purpose"],
            "--owner", spec["owner"], "--out", str(rr / spec["name"]), "--built-by", "portal",
            "--source-title", spec["lv"], "--images-root", str(staged), "--split-csv", str(split_csv)]
    if b["kind"] == "model":
        argv += ["--b-model", b["weights"], "--imgsz", str(b["imgsz"]), "--pred-classes", b["classes"],
                 "--b-label", b["id"]]
    else:
        argv += ["--b-coco", b["coco"], "--b-label", b["id"]]
    if spec["purpose"] == "revise":
        argv += ["--work-dir", str(Path(spec["state"]) / "work")]
    for r in spec.get("reviewers") or []:
        argv += ["--reviewer", r]
    return argv


def stage(spec, cov) -> tuple:
    """The checked images under their label-set keys in <state>/review_build/<name>/images (hard links, else copies),
    and their splits in <state>/.../splits.csv. -> (images folder, split CSV). Never under the review root."""
    sdir = Path(spec["state"])
    flat = sdir / "images"
    flat.mkdir(parents=True, exist_ok=False)
    rows = []
    for key, (src, split) in sorted(cov["files"].items()):
        src = Path(src)
        dst = flat / f"{key}.rf.0{src.suffix.lower()}"      # compare_package reads the key back (stem_of)
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
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
    out = rr / name
    try:
        say(phase="checking", pid=os.getpid(), reason=None)
        cov = coverage(_Cfg(spec), spec["lv"], use_cache=False)
        if not cov.get("ok") or cov.get("export") != spec.get("export"):
            note(f"inputs refused: {cov.get('problem') or 'the linked export changed since the build was started'}")
            detail = (f"Images found {cov.get('found', 0)} of {cov.get('total', 0)}; {cov.get('missing', 0)} missing, "
                      f"{cov.get('mismatched', 0)} mismatched") if cov.get("total") else None
            say(phase="failed", reason="images" if cov.get("total") else "inputs", detail=detail, finished=_now())
            return 1
        say(detail=f"Images found {cov['found']} of {cov['total']} · aspect ratios match")
        staged, splits = stage(spec, cov)
        if out.exists() or (rr / f"_work_{name}").exists():
            note("refused: the package or its work folder already exists")
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
        rc = run_compare(compare_argv(spec, staged, splits), log, say)
        if rc != 0:
            note(f"compare_package exited {rc}")
            _remove_unfinished(out, name)
            say(phase="failed", reason="build", finished=_now())
            return 1
        meta = _read_json(out / "review_package.json")
        if not isinstance(meta, dict) or meta.get("id") != name or meta.get("built_by") != "portal" \
                or meta.get("purpose") != spec["purpose"] or meta.get("n_images") != cov["found"]:
            note("the finished package's review_package.json is missing or not this build's")
            say(phase="failed", reason="check", finished=_now())
            return 1
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
            _remove_unfinished(out, name)
            say(phase="failed", reason="build", finished=_now())
        except Exception:
            pass
        return 1
    finally:
        if gpu_lock is not None:
            gpu_lock.close()
        shutil.rmtree(Path(spec["state"]) / "images", ignore_errors=True)    # staged links / copies
        lock.close()


def _remove_unfinished(out: Path, name: str):
    """A package folder this build left without a finished review_package.json (compare_package removes its own on an
    error; this covers a killed child). Only a folder of this build's own name; never a _work_ folder."""
    meta = _read_json(out / "review_package.json")
    if out.is_dir() and not (isinstance(meta, dict) and meta.get("id") == name and meta.get("built_by") == "portal"):
        shutil.rmtree(out, ignore_errors=True)


def run_compare(argv, log: Path, say) -> int:
    """compare_package as a child: its output appended to the raw log; its "[phase] ..." markers move the status."""
    env = {k: v for k, v in os.environ.items() if k not in ("CRYOEV_VIA", "CRYOEV_FOR")}
    with open(log, "a", encoding="utf-8", errors="replace") as fh,             subprocess.Popen(argv, cwd=REPO, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             env={**env, "PYTHONIOENCODING": "utf-8"}, text=True, encoding="utf-8",
                             errors="replace") as p:
        try:
            for line in p.stdout:
                fh.write(line)
                fh.flush()
                s = line.strip()
                if s == "[phase] predicting":
                    say(phase="predicting")
                elif s == "[phase] writing":
                    say(phase="writing")
            return p.wait()
        except BaseException:
            p.kill()
            p.wait()
            raise


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
