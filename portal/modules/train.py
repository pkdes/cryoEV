"""
Train module (CJ portal): a new training job, the live training run, and past runs. Roles: admin, advanced.

    GET  /train/                 the page: New training job (form), Live training, Past runs
    GET  /train/train.js         the form's helper (placeholders follow the preset); the page works without it
    POST /train/check            validate the form -> the same page with a summary and the exact job JSON
    POST /train/queue            CJ_TRAIN_QUEUE on: create the job; off: the same preview, nothing written (409)
    GET  /train/api/options      datasets, base models, presets, the queue setting (schema cj_train/1)
    GET  /train/api/runs         past training jobs and runs with their metrics and models
    GET  /train/api/live.json    live/live.json for this page: training only; no paths, machines or raw reasons
                                 for non-admins (advanced users can't open /history/live/live.json)

What "queue" creates. Today a training job is three steps by hand (docs/JOBS.md): (1) `registry.py job new --type
train ...` (registry.new_job), (2) `training/make_job_bundle.py` (a CLI main() only: no callable function), (3)
`jobs/sync.py push` (needs step 2's zip). This module does step 1 only, through registry.new_job, the very function
the CLI calls; the settings (preset, base model, run_experiments arguments, max hours) go into the job's `note`. The
result page gives admins the exact commands for steps 2 and 3. Nothing is ever written to a queue share here.

The setting CJ_TRAIN_QUEUE (the Config attribute `train_queue` if it has one, else the environment variable; default
"off"; only the exact word "on", any case, turns it on; read on every request):
  - off: /train/queue answers 409 with "Preview only: this is exactly what would be queued" and writes NOTHING (no
    registry line, no list file, no audit line);
  - on: re-validates on the server, writes an audit line through portal/audit.py BEFORE the registry (write-ahead:
    no line, nothing done), calls registry.new_job, then a second audit line with the result.

The preview is built from the same pieces as new_job (read_image_list, ids.job_id, JOB_UNITS) and checked with the
registry's own schema check; portal/tests/test_train_module.py proves that it equals what the CLI writes.
Defaults: the F5 preset takes epochs, imgsz, batch, patience, rect and seed from the production model's run
(run.jsonl config; seed 42 = run_experiments.py's default when the run has none), else the documented values.
Only what existing jobs varied is exposed: epochs, imgsz, seed and max hours (the job's time cap).

Plain words first; registry paths, weights paths, commands and raw errors only for admins.
"""
from __future__ import annotations

import hashlib
import html
import importlib.util
import json
import logging
import os
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qs

from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route

from portal import audit as PA

PREFIX = "/train"
ROLES = ("admin", "advanced")

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
TEMPLATE = HERE.parent / "templates" / "train.html"
MODELS_YAML = REPO / "models" / "models.yaml"
SCHEMA = "cj_train/1"
SETTING = "CJ_TRAIN_QUEUE"
REGISTRY_TTL_S = 10.0          # one read of the registry files serves the page for 10 s (writes re-read fresh)
STALE_FACTOR = 3               # live.json older than 3 x interval_s is stale (as home and Jobs)
MAX_BODY = 8192
MAX_JOB_ID = 48                # jobs/common.py MAX_JOB_ID (sync.py push refuses longer ids)
NAME_MAX = 35                  # job_<yyyymmdd>_ is 13 characters: 13 + 35 = MAX_JOB_ID
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
N_RUNS = 40
YOLO = "yolo"                  # base "model": run_experiments.py without --weights (COCO-pretrained YOLO weights)
FIELDS = ("dataset", "base", "preset", "name", "epochs", "imgsz", "seed", "max_hours")
SIZES = {"n": "nano (y8n)", "s": "small (y8s)", "m": "medium (y8m)", "l": "large (y8l)", "x": "extra large (y8x)"}
LAPTOP_SIZES = ("n", "s")      # docs/JOBS.md: y8m/l/x or imgsz > 640 are "desktop only" on the laptop
LAPTOP_MAX_IMGSZ = 640

# Presets = the recipes existing jobs used (docs/JOBS.md, run.jsonl). `fixed` never shows as a form field.
PRESETS = {
    "f5": {"label": "Desktop GPU: the production recipe (F5)", "where": "desktop", "size": "m",
           "epochs": 300, "imgsz": 1440, "seed": 42, "max_hours": 48.0,
           "fixed": {"batch": 8, "patience": 300, "rect": 0, "workers": 4},
           "about": "y8m at 1440 px, batch 8, rect off: v5 F5 and its seed repeats (docs/JOBS.md, run.jsonl)."},
    "laptop": {"label": "Laptop GPU: a small model at 640 px", "where": "laptop", "size": "s",
               "epochs": 300, "imgsz": 640, "seed": 42, "max_hours": 4.0,
               "fixed": {"batch": 4, "patience": None, "rect": 0, "workers": 2},
               "about": "y8s at 640 px, batch 4, 2 workers (docs/JOBS.md, \"Training on the laptop\")."},
}
WHERE_WORD = {"desktop": "the desktop GPU computer", "laptop": "the laptop's GPU"}
LIMITS = {"epochs": (1, 1000), "imgsz": (320, 1440), "seed": (0, 2 ** 31 - 1), "max_hours": (0.25, 48.0)}

esc = html.escape
log = logging.getLogger("cj.train")
_reg_mod = None
_reg_lock = threading.Lock()
_reg_cache: dict = {}
_queue_lock = threading.Lock()          # one queue at a time in this process (new_job checks the id, then appends)
_yaml_cache: dict = {}


# ---------------------------------------------------------------- small helpers
def _d(x) -> dict:
    return x if isinstance(x, dict) else {}


def _l(x) -> list:
    return x if isinstance(x, list) else []


def _s(x) -> str | None:
    return x if isinstance(x, str) and x else None


def _num(x):
    return None if isinstance(x, bool) or not isinstance(x, (int, float)) else x


def f3(v) -> str:
    v = _num(v)
    return "?" if v is None else f"{v:.3f}"


def dur(s) -> str:
    v = _num(s)
    if v is None:
        return "?"
    v = max(0, round(v))
    if v < 60:
        return f"{v} s"
    m = round(v / 60)
    if m < 60:
        return f"{m} min"
    h, m = divmod(m, 60)
    if h < 48:
        return f"{h} h" + (f" {m} min" if m else "")
    return f"{h // 24} d {h % 24} h"


def script_json(obj) -> str:
    """JSON for a <script type="application/json"> block: no "<", ">" or "&" can end the element."""
    return (json.dumps(obj, ensure_ascii=True).replace("&", "\\u0026").replace("<", "\\u003c")
            .replace(">", "\\u003e"))


def queue_on(cfg) -> bool:
    v = getattr(cfg, "train_queue", None)
    if v is None:
        v = os.environ.get(SETTING, "off")
    return str(v).strip().lower() == "on"


def is_admin(user) -> bool:
    return (user or {}).get("role") == "admin"


# ---------------------------------------------------------------- registry (read; the one write is new_job)
def registry_module():
    """registry/registry.py, loaded once by path (registry/ is not a package; it puts its own folder on sys.path)."""
    global _reg_mod
    if _reg_mod is None:
        spec = importlib.util.spec_from_file_location("cj_train_registry", REPO / "registry" / "registry.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _reg_mod = mod
    return _reg_mod


def read_registry(root, fresh=False) -> tuple:
    """-> ({entity: {id: record}} for dataset, model, run, job; error or None). Cached REGISTRY_TTL_S per root."""
    if root is None:
        return {}, "no registry folder configured"
    key, now = str(root), time.monotonic()
    with _reg_lock:
        hit = _reg_cache.get(key)
        if not fresh and hit is not None and 0 <= now - hit[0] < REGISTRY_TTL_S:
            return hit[1], hit[2]
    if not Path(root).is_dir():
        data, err = {}, "registry folder not found"
    else:
        try:
            R = registry_module()
            data = {e: R.current(e, Path(root)) for e in ("dataset", "model", "run", "job")}
            err = None
        except Exception as e:                          # an unreadable registry must not take the page down
            data, err = {}, f"registry not readable ({type(e).__name__}: {str(e)[:160]})"
    with _reg_lock:
        _reg_cache[key] = (time.monotonic(), data, err)
    return data, err


def models_yaml() -> dict:
    """models/models.yaml `models:` (read once per file change); {} when unreadable."""
    try:
        st = MODELS_YAML.stat()
    except OSError:
        return {}
    hit = _yaml_cache.get("m")
    if hit and hit[0] == st.st_mtime_ns:
        return hit[1]
    try:
        import yaml
        doc = _d(_d(yaml.safe_load(MODELS_YAML.read_text(encoding="utf-8"))).get("models"))
    except Exception:
        doc = {}
    _yaml_cache["m"] = (st.st_mtime_ns, doc)
    return doc


ARCH_RE = re.compile(r"yolov?8([nsmlx])-seg")


def model_size(model: dict, runs: dict) -> str | None:
    """n/s/m/l/x from the model's run config arch, else models.yaml `arch:` (by id or alias)."""
    arch = _s(_d(_d(runs.get(model.get("run"))).get("config")).get("arch"))
    if not arch:
        y = models_yaml()
        for k in [model.get("id")] + _l(model.get("aliases")):
            if isinstance(k, str) and _s(_d(y.get(k)).get("arch")):
                arch = y[k]["arch"]
                break
    m = ARCH_RE.search(arch or "")
    return m.group(1) if m else None


def weights_ok(w) -> Path | None:
    """The base model's weights file inside the repo (make_job_bundle uses the desktop's own copy), else None."""
    if not _s(w):
        return None
    try:
        p = (REPO / w).resolve()
        p.relative_to(REPO.resolve())
    except (OSError, ValueError):
        return None
    return p if p.is_file() else None


def list_path(root, ds: dict) -> Path | None:
    """The dataset's image list (<registry>/lists/...), only inside the registry folder."""
    f = _s(_d(ds.get("image_list")).get("file"))
    if not f or root is None:
        return None
    try:
        base = Path(root).resolve()
        p = (base / f).resolve()
        p.relative_to(base)
    except (OSError, ValueError):
        return None
    return p


def split_counts(ds: dict) -> dict:
    """{train, valid, test: images} as trained (counts_trained), else the export's counts."""
    ct = _d(ds.get("counts_trained"))
    if ct:
        return {k: v for k, v in ct.items() if _num(v) is not None}
    return {k: _d(v).get("images") for k, v in _d(ds.get("counts")).items() if _num(_d(v).get("images")) is not None}


def source_text(ds: dict) -> str:
    src = _d(ds.get("source"))
    if src.get("kind") == "label_version" or _s(ds.get("label_version")):
        return f"label version {_s(ds.get('label_version')) or src.get('id') or '?'}"
    if src.get("kind") == "roboflow_export":
        return f"Roboflow export “{src.get('id') or ds.get('export') or '?'}”"
    return "source unknown"


def counts_text(c: dict) -> str:
    if not c:
        return "image counts unknown"
    names = {"train": "train", "valid": "val", "val": "val", "test": "test"}
    return " · ".join(f"{names.get(k, k)} {v}" for k, v in c.items())


def dataset_rows(data: dict, root, admin: bool) -> list:
    out = []
    for i, ds in _d(data.get("dataset")).items():
        lp = list_path(root, ds)
        usable = bool(lp and lp.is_file())
        row = {"id": i, "source": source_text(ds), "counts": split_counts(ds),
               "n_list": _d(ds.get("image_list")).get("n"), "status": _s(ds.get("status")), "usable": usable,
               "created": _s(ds.get("created")) or "", "label_version": _s(ds.get("label_version")),
               "export": _s(ds.get("export")) or _s(_d(ds.get("source")).get("id"))}
        if admin:
            row["image_list"] = str(lp) if lp else None
            row["dataset_dir"] = _s(ds.get("dataset_dir"))
        out.append(row)
    usable = sorted([r for r in out if r["usable"]], key=lambda r: r["created"], reverse=True)
    rest = sorted([r for r in out if not r["usable"]], key=lambda r: r["created"], reverse=True)
    return usable + rest


def model_rows(data: dict, admin: bool) -> list:
    runs = _d(data.get("run"))
    out = []
    for i, m in _d(data.get("model")).items():
        if m.get("status") == "retired":
            continue
        row = {"id": i, "status": _s(m.get("status")) or "registered", "dataset": _s(m.get("dataset")),
               "run": _s(m.get("run")), "size": model_size(m, runs), "weights_ok": weights_ok(m.get("weights")) is not None,
               "logged": _s(m.get("logged")) or ""}
        if admin:
            row["weights"] = _s(m.get("weights"))
        out.append(row)
    out.sort(key=lambda r: r["logged"], reverse=True)
    out.sort(key=lambda r: r["status"] != "production")
    return out


def production(data: dict) -> tuple:
    """-> (production model record or None, its run record or None)."""
    for i, m in _d(data.get("model")).items():
        if m.get("status") == "production":
            return m, _d(data.get("run")).get(m.get("run"))
    return None, None


def presets(data: dict) -> dict:
    """PRESETS with F5's numbers from the production model's run config where it has them."""
    out = {k: {**v, "fixed": dict(v["fixed"])} for k, v in PRESETS.items()}
    m, run = production(data)
    cfg = _d(_d(run).get("config"))
    if cfg:
        f5 = out["f5"]
        for k in ("epochs", "imgsz", "seed"):
            if _num(cfg.get(k)) is not None:
                f5[k] = int(cfg[k])
        if _num(cfg.get("batch")) is not None:
            f5["fixed"]["batch"] = int(cfg["batch"])
        if _num(cfg.get("patience")) is not None:
            f5["fixed"]["patience"] = int(cfg["patience"])
        if isinstance(cfg.get("rect"), bool):
            f5["fixed"]["rect"] = int(cfg["rect"])
        f5["from_run"] = run.get("id")
        f5["from_model"] = m.get("id")
    return out


# ---------------------------------------------------------------- the plan (validation + preview)
def _int(form, k, errs, label, default):
    v = (form.get(k) or "").strip()
    if not v:
        return default
    try:
        n = int(v)
    except ValueError:
        errs.append(f"{label} must be a whole number.")
        return None
    lo, hi = LIMITS[k]
    if not lo <= n <= hi:
        errs.append(f"{label} must be between {lo} and {hi}.")
        return None
    return n


def plan(cfg, form: dict, user: dict, fresh: bool = False) -> tuple:
    """Validate the form against the registry. -> (errors [str], plan dict or None). Writes nothing."""
    admin, uid = is_admin(user), (user or {}).get("id")
    root = getattr(cfg, "registry", None)
    data, err = read_registry(root, fresh=fresh)
    if err:
        return [("The registry can't be read, so no job can be made." + (f" ({err})" if admin else ""))], None
    R = registry_module()
    errs = []
    P = presets(data)
    pk = (form.get("preset") or "").strip()
    preset = P.get(pk)
    if preset is None:
        errs.append("Pick a preset.")
    ds_id = (form.get("dataset") or "").strip()
    ds = _d(data.get("dataset")).get(ds_id)
    lp = None
    if not ds_id:
        errs.append("Pick a dataset.")
    elif ds is None:
        errs.append("That dataset is not in the registry.")
    else:
        lp = list_path(root, ds)
        if lp is None or not lp.is_file():
            errs.append("That dataset has no image list in the registry, so a job can't be made from it.")
            lp = None
        else:
            want = _s(_d(ds.get("image_list")).get("sha256"))
            if want and hashlib.sha256(lp.read_bytes()).hexdigest() != want:
                errs.append("That dataset's image list has changed since it was recorded, so it can't be used."
                            + (f" ({lp})" if admin else ""))
                lp = None
    base = (form.get("base") or "").strip()
    model, size = None, None
    if not base:
        errs.append("Pick a base model.")
    elif base != YOLO:
        model = _d(data.get("model")).get(base)
        if model is None or model.get("status") == "retired":
            errs.append("That base model is not in the registry.")
            model = None
        elif weights_ok(model.get("weights")) is None:
            errs.append("That base model's weights file is not in the repo." +
                        (f" ({model.get('weights')})" if admin else ""))
            model = None
        else:
            size = model_size(model, _d(data.get("run")))
            if size is None:
                errs.append("That base model's architecture is unknown, so its size can't be matched.")
    if preset is not None and size is None and base == YOLO:
        size = preset["size"]
    epochs = _int(form, "epochs", errs, "Epochs", preset and preset["epochs"])
    imgsz = _int(form, "imgsz", errs, "Image size", preset and preset["imgsz"])
    seed = _int(form, "seed", errs, "Seed", preset and preset["seed"])
    if imgsz is not None and imgsz % 32:
        errs.append("Image size must be a multiple of 32 (e.g. 640, 1024, 1440).")
    mh_raw = (form.get("max_hours") or "").strip()
    max_hours = preset and preset["max_hours"]
    if mh_raw:
        try:
            max_hours = float(mh_raw)
            if max_hours != max_hours or not LIMITS["max_hours"][0] <= max_hours <= LIMITS["max_hours"][1]:
                raise ValueError
        except ValueError:
            errs.append(f"Max hours must be a number between {LIMITS['max_hours'][0]:g} and "
                        f"{LIMITS['max_hours'][1]:g}.")
            max_hours = None
    if preset is not None and preset["where"] == "laptop":
        if size is not None and size not in LAPTOP_SIZES:
            errs.append(f"A {SIZES.get(size, size)} model is too big for the laptop's GPU: use the desktop preset, "
                        "or start from the YOLO base weights.")
        if imgsz is not None and imgsz > LAPTOP_MAX_IMGSZ:
            errs.append(f"Image size {imgsz} is too big for the laptop's GPU (at most {LAPTOP_MAX_IMGSZ}).")
    name = (form.get("name") or "").strip()
    if not name and preset is not None and size and imgsz and seed is not None:
        name = f"{pk}_{size}{imgsz}_seed{seed}"
    today = datetime.now().strftime("%Y%m%d")
    jid = None
    if not name:
        errs.append("Give the job a name.")
    elif len(name) > NAME_MAX or not NAME_RE.match(name):
        errs.append(f"The name may use letters, digits, - and _ (at most {NAME_MAX} characters).")
    else:
        jid = R.ids.job_id(today, name)
        if not R.ids.is_id("job", jid) or len(jid) > MAX_JOB_ID:
            errs.append("That name doesn't make a valid job id; try a shorter, plainer name.")
            jid = None
    if errs:
        return errs, None
    if jid in _d(data.get("job")) or jid in R.current(root=Path(root)):
        return [f"A job called {jid} already exists today: pick another name."], None
    stems = R.read_image_list(lp)
    if not stems:
        return ["That dataset's image list is empty."], None
    fx = preset["fixed"]
    args = ["--single-run", "--model-size", size, "--batch", str(fx["batch"]), "--imgsz", str(imgsz),
            "--epochs", str(epochs)]
    if fx.get("patience") is not None:
        args += ["--patience", str(fx["patience"])]
    args += ["--rect", str(fx["rect"]), "--workers", str(fx["workers"]), "--seed", str(seed)]
    if model is not None:
        args += ["--weights", model["weights"]]
    who = uid if uid in R.roster() else None
    note = (f"portal Train, preset {pk}: dataset {ds_id}; base {base}; where {preset['where']}; "
            f"max-hours {max_hours:g}; run_experiments {' '.join(args)}; requested by {uid}")
    text = "".join(s + "\n" for s in stems)
    rec = {"schema": "job/1", "id": jid, "logged": R.now(), "recorded_by": "registry.py job new", "name": name,
           "type": "train", "where": preset["where"], "status": "ready", "n_images": len(stems),
           "image_list": {"file": f"{R.LISTS}/{jid}.txt", "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()},
           "progress": {"done": 0, "total": len(stems), "unit": R.JOB_UNITS["train"]},
           "who": who, "inputs": [], "outputs": [], "note": note}
    bad = R.check_record(rec, R.schemas())
    if bad:
        return ["The job would not pass the registry's checks." + (f" ({'; '.join(bad[:3])})" if admin else "")], None
    record = dict(rec)
    record.pop("logged")                                # set by new_job when the line is written
    counts = split_counts(ds)
    return [], {"job": jid, "name": name, "preset": pk, "preset_label": preset["label"], "where": preset["where"],
                "dataset": ds_id, "dataset_source": source_text(ds), "counts": counts, "n_images": len(stems),
                "base": base, "base_status": _s(model.get("status")) if model else None, "size": size,
                "epochs": epochs, "imgsz": imgsz, "seed": seed, "max_hours": max_hours, "fixed": fx,
                "run_experiments_args": args, "note": note, "who": who, "record": record,
                "list_file": lp, "dataset_dir": _s(ds.get("dataset_dir"))}


def form_values(p: dict) -> dict:
    """The resolved plan as form fields (the queue form's hidden fields: the server validates them again)."""
    return {"dataset": p["dataset"], "base": p["base"], "preset": p["preset"], "name": p["name"],
            "epochs": str(p["epochs"]), "imgsz": str(p["imgsz"]), "seed": str(p["seed"]),
            "max_hours": f"{p['max_hours']:g}"}


def next_steps(cfg, p: dict) -> list:
    """Admin only: the exact commands for steps 2 and 3 (docs/JOBS.md)."""
    ddir = p.get("dataset_dir")
    if not ddir:
        try:
            from data_utils import config as DC
            c = DC.cryoai_root(required=False)
            ddir = str(Path(c) / "training outputs" / p["dataset"]) if c else f"<CryoAI>/training outputs/{p['dataset']}"
        except Exception:
            ddir = f"<CryoAI>/training outputs/{p['dataset']}"
    args = " ".join(f'"{a}"' if " " in a else a for a in p["run_experiments_args"])
    out = [f'python training/make_job_bundle.py --dataset-dir "{ddir}" -- {args}']
    if p["where"] == "laptop":
        out.append(f"python jobs/sync.py local {p['job']} --bundle <zip from the line above> "
                   f"--max-hours {p['max_hours']:g}")
    else:
        share = None
        live, _ = read_live(getattr(cfg, "registry", None))
        for q in _l(_d(live).get("queues")):
            share = _s(_d(q).get("share"))
            if share:
                break
        out.append(f"python jobs/sync.py push {p['job']} --bundle <zip from the line above> "
                   f"--share {share or '<desktop share>'} --max-hours {p['max_hours']:g}")
    return out


# ---------------------------------------------------------------- queue (the one write)
def audit(cfg, uid, job, p: dict, phase: str, result=None):
    PA.audit(cfg.state_dir, uid, "train_queue", job,
             {"dataset": p["dataset"], "base": p["base"], "preset": p["preset"], "where": p["where"],
              "epochs": p["epochs"], "imgsz": p["imgsz"], "seed": p["seed"], "max_hours": p["max_hours"],
              "phase": phase, "result": result})


def queue(cfg, form: dict, user: dict) -> tuple:
    """Runs in a worker thread. -> (http status, errors, plan or None, created record or None, message)."""
    if not queue_on(cfg):
        errs, p = plan(cfg, form, user)
        return 409, errs, p, None, None
    admin, uid = is_admin(user), (user or {}).get("id")
    with _queue_lock:
        errs, p = plan(cfg, form, user, fresh=True)
        if errs:
            return 400, errs, None, None, None
        try:
            audit(cfg, uid, p["job"], p, "requested")       # write-ahead: no audit line, no job
        except (OSError, TypeError, AttributeError) as e:
            log.error("train audit not writable (%s): nothing queued", type(e).__name__)
            return 500, ["The audit log can't be written, so nothing was queued."], p, None, None
        R = registry_module()
        try:
            line = R.new_job("train", p["where"], p["name"], images=p["list_file"], who=p["who"],
                             root=Path(cfg.registry), note=p["note"])
            result, out = "ok", (200, [], p, line, None)
        except Exception as e:                           # RegistryError, or the disk failing mid-write
            result = f"error: {type(e).__name__}"
            out = (409, ["The job was not created." + (f" ({e})" if admin else " Ask an admin to check the registry.")],
                   p, None, None)
        try:
            audit(cfg, uid, p["job"], p, "result", result)
        except (OSError, TypeError, AttributeError) as e:
            log.warning("train audit result line not written (%s): %s %s", type(e).__name__, p["job"], result)
        with _reg_lock:
            _reg_cache.pop(str(cfg.registry), None)
        return out


# ---------------------------------------------------------------- live
def read_live(registry) -> tuple:
    """-> (live.json dict or None, problem or None); a read that meets monitor.py's atomic replace is retried."""
    if registry is None:
        return None, "no registry folder configured"
    p = Path(registry) / "live" / "live.json"
    err = None
    for i in range(3):
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
            return (doc, None) if isinstance(doc, dict) else (None, "live.json is not a JSON object")
        except FileNotFoundError:
            return None, "no live snapshot (is jobs/monitor.py running?)"
        except (OSError, ValueError) as e:
            err = f"live.json not readable ({type(e).__name__})"
            time.sleep(0.05 * (i + 1))
    return None, err


def live_for(live, admin: bool):
    """Training only (a running predict job is left out); for non-admins no share path, machine, detail or raw
    reason (they may contain paths)."""
    if not isinstance(live, dict):
        return None
    out = {k: live.get(k) for k in ("schema", "generated", "generated_utc", "interval_s")}
    out["monitor"] = {"machine": _d(live.get("monitor")).get("machine")} if admin else {}
    qs = []
    for n, q in enumerate(_l(live.get("queues")), 1):
        q = _d(q)
        run = _d(q.get("running"))
        run = None if not run or run.get("kind") == "predict" else dict(run)
        qq = {"share": q.get("share") if admin else f"GPU computer {n}", "reach": q.get("reach"),
              "detail": q.get("detail") if admin else None, "running": run,
              "watcher": dict(_d(q.get("watcher"))), "jobs": [dict(_d(j)) for j in _l(q.get("jobs"))],
              "finished": [dict(_d(f)) for f in _l(q.get("finished"))]}
        if not admin:
            for k in ("machine", "detail"):
                qq["watcher"].pop(k, None)
            for j in qq["jobs"] + qq["finished"]:
                if j.get("reason"):
                    j["reason"] = "stopped early (an admin can see why)"
        qs.append(qq)
    out["queues"] = qs
    return out


def live_stale(live) -> bool:
    try:
        t = datetime.fromisoformat(str(live.get("generated_utc")).replace("Z", "+00:00")).timestamp()
        every = float(live.get("interval_s") or 30)
    except (TypeError, ValueError, AttributeError):
        return True
    return time.time() - t > STALE_FACTOR * every


# ---------------------------------------------------------------- past runs
RUN_NOTE = re.compile(r"\brun (\S+)")


def run_rows(data: dict, admin: bool) -> list:
    """Training jobs (registry) and runs, newest first: job -> run (the pull's status note "run <id> (...)"),
    run -> dataset -> source (label version or export), run -> models made from it."""
    runs, jobs, models, dss = (_d(data.get(k)) for k in ("run", "job", "model", "dataset"))
    by_run = {}
    for mid, m in models.items():
        if _s(m.get("run")):
            by_run.setdefault(m["run"], []).append({"id": mid, "status": _s(m.get("status")) or "registered"})
    rows, linked = [], set()

    def row(job, run_id):
        run = _d(runs.get(run_id)) if run_id else {}
        cfg = _d(run.get("config"))
        dsid = _s(run.get("dataset"))
        if not dsid and job:
            m = re.search(r"\bdataset (\S+?);", str(job.get("note") or ""))
            dsid = m.group(1) if m else None
        ds = _d(dss.get(dsid)) if dsid else {}
        met = _d(run.get("metrics"))
        size = ARCH_RE.search(str(cfg.get("arch") or ""))
        return {"when": _s(run.get("started")) or _s((job or {}).get("logged")) or _s(run.get("logged")) or "",
                "job": (job or {}).get("id"), "job_status": (job or {}).get("status"),
                "where": (job or {}).get("where"), "run": run_id if run else None,
                "run_status": _s(run.get("status")), "dataset": dsid,
                "dataset_source": source_text(ds) if ds else None,
                "label_version": _s(ds.get("label_version")) or (
                    _s(_d(ds.get("source")).get("id")) if _d(ds.get("source")).get("kind") == "label_version" else None),
                "size": size.group(1) if size else None, "imgsz": cfg.get("imgsz"), "batch": cfg.get("batch"),
                "epochs": cfg.get("epochs"), "seed": cfg.get("seed"),
                "init": _s(cfg.get("weights_init")) if admin else None,
                "mask_map50_95": _num(met.get("mask_map50_95")), "mask_map50": _num(met.get("mask_map50")),
                "models": by_run.get(run_id, []) if run_id else [],
                "reason": (_s((job or {}).get("blocked_reason")) if admin else
                           ("stopped early" if (job or {}).get("blocked_reason") else None)),
                "machine": _s(run.get("machine")) if admin else None}
    for jid, j in jobs.items():
        if j.get("type") != "train":
            continue
        m = RUN_NOTE.search(str(j.get("note") or ""))
        rid = m.group(1) if m and m.group(1) in runs else None
        if rid:
            linked.add(rid)
        rows.append(row(j, rid))
    for rid in runs:
        if rid not in linked:
            rows.append(row(None, rid))
    rows.sort(key=lambda r: r["when"], reverse=True)
    return rows[:N_RUNS]


# ---------------------------------------------------------------- rendering (every value escaped here)
def chip(text, tone="") -> str:
    return f'<span class="cj-train-chip{(" cj-train-t-" + tone) if tone else ""}">{esc(text)}</span>'


def options_html(rows, selected, label_fn, value_fn, disabled_fn=lambda r: False) -> str:
    out = []
    for r in rows:
        v = value_fn(r)
        dis = disabled_fn(r)
        out.append(f'<option value="{esc(v)}"{" selected" if v == selected else ""}{" disabled" if dis else ""}>'
                   f'{esc(label_fn(r))}</option>')
    return "".join(out)


def form_html(data, form: dict, admin: bool, root) -> str:
    ds = dataset_rows(data, root, admin)
    ms = model_rows(data, admin)
    P = presets(data)
    sel_ds = form.get("dataset") or next((r["id"] for r in ds if r["usable"]), "")
    sel_base = form.get("base") or next((r["id"] for r in ms if r["status"] == "production"), YOLO)
    sel_p = form.get("preset") or "f5"

    def ds_label(r):
        bad = "" if r["usable"] else " (no image list: can't be used)"
        return f"{r['id']}{bad}"

    ds_opts = options_html(ds, sel_ds, ds_label, lambda r: r["id"], lambda r: not r["usable"])
    ms_opts = options_html(ms, sel_base, lambda r: f"{r['id']}" + (" (production)" if r["status"] == "production"
                                                                   else f" ({r['status']})")
                           + (f" · y8{r['size']}" if r["size"] else ""), lambda r: r["id"],
                           lambda r: not r["weights_ok"])
    ms_opts += f'<option value="{YOLO}"{" selected" if sel_base == YOLO else ""}>None: start from the YOLO base weights</option>'
    p_opts = "".join(f'<option value="{esc(k)}"{" selected" if k == sel_p else ""}>{esc(v["label"])}</option>'
                     for k, v in P.items())
    ds_info = "".join(
        f'<li data-ds="{esc(r["id"])}"><b class="cj-train-wrap">{esc(r["id"])}</b>'
        f'<span class="cj-mut">From {esc(r["source"])} · {esc(counts_text(r["counts"]))}'
        + (f' · {esc(str(r["n_list"]))} images in its list' if _num(r.get("n_list")) is not None else "")
        + ("" if r["usable"] else " · no image list: can't be used") + "</span></li>" for r in ds)
    p_info = "".join(
        f'<li><b>{esc(v["label"])}</b><span class="cj-mut">{esc(v["about"])} Defaults: {v["epochs"]} epochs, '
        f'{v["imgsz"]} px, seed {v["seed"]}, at most {v["max_hours"]:g} h.'
        + (f' From the production model’s run{(" " + esc(v["from_run"])) if admin else ""}.'
           if v.get("from_run") else "") + "</span></li>" for v in P.values())
    pre = P.get(sel_p) or P["f5"]
    defaults = script_json({k: {f: v[f] for f in ("epochs", "imgsz", "seed", "max_hours")} for k, v in P.items()})

    def num(k, label, step="1", hint=""):
        val = form.get(k) or ""
        return (f'<label class="cj-train-lbl" for="cj-train-{k}">{esc(label)}'
                f'<input class="cj-train-in" id="cj-train-{k}" name="{k}" inputmode="decimal" value="{esc(val)}" '
                f'placeholder="{esc(str(pre[k]) if k != "max_hours" else format(pre[k], "g"))}" autocomplete="off">'
                f'<span class="cj-mut">{esc(hint)}</span></label>')
    return (
        f'<form class="cj-card cj-train-form" method="post" action="{PREFIX}/check" id="cj-train-form">'
        f'<script type="application/json" id="cj-train-defaults">{defaults}</script>'
        f'<label class="cj-train-lbl" for="cj-train-dataset">Dataset'
        f'<select class="cj-train-in" id="cj-train-dataset" name="dataset">{ds_opts}</select></label>'
        f'<details class="cj-train-more"><summary>About the datasets ({len(ds)})</summary>'
        f'<ul class="cj-train-info">{ds_info or "<li class=cj-mut>None in the registry.</li>"}</ul></details>'
        f'<label class="cj-train-lbl" for="cj-train-base">Start from (base model)'
        f'<select class="cj-train-in" id="cj-train-base" name="base">{ms_opts}</select></label>'
        f'<label class="cj-train-lbl" for="cj-train-preset">Preset'
        f'<select class="cj-train-in" id="cj-train-preset" name="preset">{p_opts}</select></label>'
        f'<details class="cj-train-more"><summary>About the presets</summary>'
        f'<ul class="cj-train-info">{p_info}</ul></details>'
        f'<div class="cj-train-grid">'
        + num("epochs", "Epochs", hint="Blank = the preset's value")
        + num("imgsz", "Image size (px)", hint="A multiple of 32")
        + num("seed", "Seed", hint="Change it to repeat a run")
        + num("max_hours", "Max hours", hint="The run is stopped after this")
        + '</div>'
        f'<label class="cj-train-lbl" for="cj-train-name">Job name'
        f'<input class="cj-train-in" id="cj-train-name" name="name" value="{esc(form.get("name") or "")}" '
        f'maxlength="{NAME_MAX}" autocomplete="off" placeholder="made from the preset, size and seed">'
        f'<span class="cj-mut">Letters, digits, - and _. The job id becomes job_&lt;today&gt;_&lt;name&gt;.</span></label>'
        f'<div class="cj-train-btns"><button class="cj-train-btn cj-train-primary" type="submit">Check</button></div>'
        f'</form>')


def plan_html(cfg, p: dict, admin: bool, on: bool, state: str, created=None) -> str:
    """state: preview (after Check) | refused (Queue while off) | created."""
    c = p["counts"]
    base = ("the YOLO base weights" if p["base"] == YOLO else
            f"{p['base']}" + (" (production)" if p["base_status"] == "production" else ""))
    kv = [("Dataset", f"{p['dataset']}"), ("Its images", f"{counts_text(c)} ({p['n_images']} in the job's list)"),
          ("From", p["dataset_source"]), ("Start from", base), ("Model", SIZES.get(p["size"], p["size"])),
          ("Runs on", WHERE_WORD[p["where"]]),
          ("Settings", f"{p['epochs']} epochs · {p['imgsz']} px · seed {p['seed']} · batch "
                       f"{p['fixed']['batch']}"), ("Time cap", f"stopped after {p['max_hours']:g} h at most"),
          ("Job id", p["job"]), ("Owner", p["who"] or "none (your portal id is not a registry person)")]
    dl = "".join(f"<div><dt>{esc(k)}</dt><dd>{esc(str(v))}</dd></div>" for k, v in kv)
    rec = created if created is not None else p["record"]
    shown = dict(rec)
    if created is None:
        shown = {"logged": "(the time it is written)", **shown}
        shown = {k: shown[k] for k in ["schema", "id", "logged"] + [k for k in rec if k not in ("schema", "id")]}
    js = esc(json.dumps(shown, indent=1, ensure_ascii=False))
    if state == "created":
        head = (f'<h2 class="cj-sec">Queued</h2><p class="cj-train-ok" role="status">Job <b>{esc(p["job"])}</b> was '
                f'created in the registry (status: ready). An audit line was written.</p>')
        foot = ""
    else:
        head = ('<h2 class="cj-sec">Summary</h2>'
                + ('<p class="cj-train-warn" role="status">Queueing is off, so nothing was written.</p>'
                   if state == "refused" else ""))
        if on:
            hidden = "".join(f'<input type="hidden" name="{k}" value="{esc(v)}">' for k, v in form_values(p).items())
            foot = (f'<form method="post" action="{PREFIX}/queue" class="cj-train-btns">{hidden}'
                    f'<button class="cj-train-btn cj-train-primary" type="submit">Queue this job</button></form>'
                    f'<p class="cj-mut">Queue writes exactly the job below to the registry and an audit line.</p>')
        else:
            foot = ('<p class="cj-train-preview"><b>Preview only: this is exactly what would be queued.</b> '
                    'Queueing is turned off on this portal, so nothing is written. Only the platform owner turns '
                    'it on.</p>')
    steps = ""
    if admin:
        cmds = "\n".join(next_steps(cfg, p))
        steps = (f'<details class="cj-train-more"{" open" if state == "created" else ""}><summary>Then, to start it '
                 f'on the GPU (admin)</summary><p class="cj-mut">The portal makes the registry job only. Make the '
                 f'bundle, then send it:</p><pre class="cj-train-pre">{esc(cmds)}</pre></details>')
    elif state == "created":
        steps = '<p class="cj-mut">An admin sends it to the GPU computer; it then shows under Live training.</p>'
    return (f'<section class="cj-card cj-train-sum" id="cj-train-sum" aria-labelledby="cj-train-sum-h">'
            f'<span id="cj-train-sum-h" hidden>Summary</span>{head}<dl class="cj-train-kv">{dl}</dl>{foot}'
            f'<details class="cj-train-more" open><summary>Job JSON</summary><pre class="cj-train-pre" '
            f'id="cj-train-json">{js}</pre></details>{steps}</section>')


def errors_html(errs) -> str:
    return ('<div class="cj-train-errs" role="alert"><b>Please fix:</b><ul>'
            + "".join(f"<li>{esc(e)}</li>" for e in errs) + "</ul></div>")


def live_html(cfg, admin: bool) -> str:
    live, err = read_live(getattr(cfg, "registry", None))
    lv = live_for(live, admin)
    lines = []
    if lv is None:
        lines.append(f'<p class="cj-mut">No live status from the GPU computers{(": " + esc(err)) if admin and err else "."}</p>')
    else:
        stale = live_stale(lv)
        runs = [q for q in lv["queues"] if q.get("reach") == "ok" and q.get("running")]
        if stale:
            lines.append('<p class="cj-mut">The status updates have stopped, so nothing is shown as running.</p>')
        elif runs:
            for q in runs:
                r = q["running"]
                ep = (f"epoch {r.get('epoch')} of {r.get('epochs')}" if r.get("epoch") is not None and r.get("epochs")
                      else str(r.get("phase") or "starting"))
                lines.append(f'<p><b>{esc(str(r.get("job") or "?"))}</b> <span class="cj-mut">{esc(ep)}, about '
                             f'{esc(dur(r.get("eta_s")))} left</span></p>')
        else:
            lines.append('<p class="cj-mut">No training is running.</p>')
        fin = [f for q in lv["queues"] for f in _l(q.get("finished"))]
        fin.sort(key=lambda f: str(f.get("ended") or ""), reverse=True)
        if fin:
            f = fin[0]
            m = _d(f.get("metrics"))
            lines.append(f'<p class="cj-mut">Last finished: <b>{esc(str(f.get("job") or "?"))}</b> '
                         f'({esc(str(f.get("status") or "?"))}'
                         + (f", val mask mAP50-95 {esc(f3(m.get('mask_map50_95')))}" if m else "") + ")</p>")
    text = "".join(lines)
    return (f'<script type="application/json" id="cj-train-live-data">{script_json(lv)}</script>'
            f'<div class="cj-train-live" id="cj-train-live" data-cj-live="full" data-cj-live-json="cj-train-live-data" '
            f'data-cj-live-src="{PREFIX}/api/live.json" data-cj-live-every="30">{text}</div>')


def runs_html(data, err, admin: bool) -> str:
    if err:
        return f'<p class="cj-mut">The registry can’t be read{(": " + esc(err)) if admin else "."}</p>'
    rows = run_rows(data, admin)
    if not rows:
        return '<p class="cj-mut">No training runs in the registry yet.</p>'
    tone = {"done": "ok", "finished": "ok", "ready": "wait", "in_progress": "run", "blocked": "warn",
            "cancelled": "off", "failed": "warn", "time_cap": "warn"}
    trs = []
    for r in rows:
        st = r["job_status"] or r["run_status"] or "?"
        who = []
        if r["job"]:
            who.append(f'<b class="cj-train-wrap">{esc(r["job"])}</b>')
        if r["run"]:
            who.append(f'<span class="cj-mut cj-train-wrap">run {esc(r["run"])}</span>')
        ds = (f'<span class="cj-train-wrap">{esc(r["dataset"])}</span>' if r["dataset"] else '<span class="cj-mut">unknown</span>')
        if r["dataset_source"]:
            ds += f'<span class="cj-mut cj-train-wrap">from {esc(r["dataset_source"])}</span>'
        st_bits = [f"y8{r['size']}" if r["size"] else None, f"{r['imgsz']} px" if r["imgsz"] else None,
                   f"batch {r['batch']}" if r["batch"] else None, f"{r['epochs']} ep" if r["epochs"] else None,
                   f"seed {r['seed']}" if r["seed"] is not None else None]
        sets = " · ".join(x for x in st_bits if x) or "–"
        met = (f'<b>{esc(f3(r["mask_map50_95"]))}</b><span class="cj-mut">mAP50 {esc(f3(r["mask_map50"]))}</span>'
               if r["mask_map50_95"] is not None else '<span class="cj-mut">–</span>')
        mods = "".join(f'<span class="cj-train-wrap">{esc(m["id"])} '
                       f'{chip("production", "prod") if m["status"] == "production" else ""}</span>'
                       for m in r["models"]) or '<span class="cj-mut">–</span>'
        extra = ""
        if r["reason"]:
            extra += f'<span class="cj-mut cj-train-wrap">{esc(r["reason"])}</span>'
        if admin and r["machine"]:
            extra += f'<span class="cj-mut cj-train-wrap">on {esc(r["machine"])}</span>'
        trs.append(f'<tr><td data-k="When">{esc(r["when"][:16].replace("T", " ") or "?")}</td>'
                   f'<td data-k="Job / run">{"".join(who)}{chip(st, tone.get(st, ""))}{extra}</td>'
                   f'<td data-k="Dataset">{ds}</td><td data-k="Settings">{esc(sets)}</td>'
                   f'<td data-k="Val mask mAP50-95">{met}</td><td data-k="Model">{mods}</td></tr>')
    return ('<table class="cj-train-runs"><thead><tr><th>When</th><th>Job / run</th><th>Dataset</th><th>Settings</th>'
            '<th>Val mask mAP50-95</th><th>Model</th></tr></thead><tbody>' + "".join(trs) + "</tbody></table>"
            '<p class="cj-mut">Val scores come from each run’s own validation set: compare runs on the same '
            'dataset only.</p>')


def page_body(cfg, user, form=None, result="", status_note="") -> str:
    admin = is_admin(user)
    root = getattr(cfg, "registry", None)
    data, err = read_registry(root)
    on = queue_on(cfg)
    banner = ('<p class="cj-train-banner cj-train-on">Queueing is on: Queue writes the job to the registry.</p>' if on
              else '<p class="cj-train-banner">Queueing is off: Check shows exactly what would be queued, and '
                   'nothing is written.</p>')
    form_part = (form_html(data, form or {}, admin, root) if not err else
                 f'<p class="cj-mut">The registry can’t be read, so no job can be made'
                 f'{(": " + esc(err)) if admin else "."}</p>')
    tpl = TEMPLATE.read_text(encoding="utf-8")
    parts = {"banner": banner + status_note, "result": result, "form": form_part, "live": live_html(cfg, admin),
             "runs": runs_html(data, err, admin)}
    return re.sub(r"\{\{(banner|result|form|live|runs)\}\}", lambda m: parts[m.group(1)], tpl)


JS = r"""// Train page (portal/modules/train.py): the number fields' placeholders follow the chosen preset.
(function () {
  'use strict';
  var d = document.getElementById('cj-train-defaults'), sel = document.getElementById('cj-train-preset');
  if (!d || !sel) return;
  var defs = {};
  try { defs = JSON.parse(d.textContent); } catch (e) { return; }
  function upd() {
    var p = defs[sel.value]; if (!p) return;
    ['epochs', 'imgsz', 'seed', 'max_hours'].forEach(function (k) {
      var i = document.getElementById('cj-train-' + k); if (i && p[k] != null) i.placeholder = String(p[k]);
    });
  }
  sel.addEventListener('change', upd);
  upd();
})();
"""


# ---------------------------------------------------------------- routes
async def read_form(request: Request):
    """application/x-www-form-urlencoded body, at most MAX_BODY bytes. -> {field: value} or None."""
    ctype = (request.headers.get("content-type") or "").split(";")[0].strip().lower()
    if ctype != "application/x-www-form-urlencoded":
        return None
    buf = b""
    async for chunk in request.stream():
        buf += chunk
        if len(buf) > MAX_BODY:
            return None
    try:
        q = parse_qs(buf.decode("utf-8"), keep_blank_values=True, max_num_fields=40)
    except (UnicodeDecodeError, ValueError):
        return None
    return {k: (q.get(k) or [""])[0][:200] for k in FIELDS}


def routes(ctx) -> list:
    def page(request: Request):                 # sync: Starlette runs it in a worker thread (file reads)
        return ctx.page(request, "Train", page_body(ctx.cfg, ctx.user(request)), active="train")

    def script(request: Request):
        r = Response(JS, media_type="text/javascript; charset=utf-8")
        r.headers["Cache-Control"] = "no-cache"
        return r

    def options(request: Request):
        user = ctx.user(request)
        admin = is_admin(user)
        root = getattr(ctx.cfg, "registry", None)
        data, err = read_registry(root)
        P = presets(data)
        return ctx.json({"schema": SCHEMA, "queue": "on" if queue_on(ctx.cfg) else "off",
                         "problem": (err if admin else "registry not readable") if err else None,
                         "datasets": dataset_rows(data, root, admin), "models": model_rows(data, admin),
                         "presets": {k: {f: v.get(f) for f in ("label", "where", "size", "epochs", "imgsz", "seed",
                                                                "max_hours", "fixed", "about", "from_run")}
                                     for k, v in P.items()}})

    def runs(request: Request):
        user = ctx.user(request)
        data, err = read_registry(getattr(ctx.cfg, "registry", None))
        return ctx.json({"schema": SCHEMA, "problem": (err if is_admin(user) else "registry not readable")
                         if err else None, "runs": [] if err else run_rows(data, is_admin(user))})

    def live(request: Request):
        doc, _ = read_live(getattr(ctx.cfg, "registry", None))
        lv = live_for(doc, is_admin(ctx.user(request)))
        if lv is None:
            return ctx.json({"problem": "no live snapshot"}, 404)
        return ctx.json(lv)

    def bad_request(request):
        body = page_body(ctx.cfg, ctx.user(request), {}, errors_html(["That form was not understood. Reload the "
                                                                      "page and try again."]))
        r = ctx.page(request, "Train", body, active="train")
        r.status_code = 400
        return r

    async def check(request: Request):
        form = await read_form(request)
        if form is None:
            return await run_in_threadpool(bad_request, request)
        user = ctx.user(request)

        def work():
            errs, p = plan(ctx.cfg, form, user)
            admin, on = is_admin(user), queue_on(ctx.cfg)
            result = errors_html(errs) if errs else plan_html(ctx.cfg, p, admin, on, "preview")
            r = ctx.page(request, "Train", page_body(ctx.cfg, user, form, result), active="train")
            r.status_code = 400 if errs else 200
            return r
        return await run_in_threadpool(work)

    async def do_queue(request: Request):
        form = await read_form(request)
        if form is None:
            return await run_in_threadpool(bad_request, request)
        user = ctx.user(request)

        def work():
            code, errs, p, created, _ = queue(ctx.cfg, form, user)
            admin, on = is_admin(user), queue_on(ctx.cfg)
            if errs:
                result = errors_html(errs) + (plan_html(ctx.cfg, p, admin, on, "preview") if p and code != 400 else "")
            elif created is not None:
                result = plan_html(ctx.cfg, p, admin, on, "created", created)
            else:
                result = plan_html(ctx.cfg, p, admin, on, "refused")
            r = ctx.page(request, "Train", page_body(ctx.cfg, user, form, result), active="train")
            r.status_code = code
            return r
        return await run_in_threadpool(work)

    return [Route(PREFIX + "/", page, methods=["GET"]),
            Route(PREFIX + "/train.js", script, methods=["GET"]),
            Route(PREFIX + "/check", check, methods=["POST"]),
            Route(PREFIX + "/queue", do_queue, methods=["POST"]),
            Route(PREFIX + "/api/options", options, methods=["GET"]),
            Route(PREFIX + "/api/runs", runs, methods=["GET"]),
            Route(PREFIX + "/api/live.json", live, methods=["GET"])]
