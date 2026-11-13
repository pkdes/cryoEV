"""
Apply module (CryoEV portal, admin only; round 10 V4 preview, round 11 #67 the Apply button): what applying a lab
review package's decisions to its label set would change, from a scratch apply, and then making that label set.

    GET  /apply/                        the review packages built on a label set: eligible ones with "Preview apply",
                                        their recent previews and the exact CLI command; the others with the reason
    POST /apply/preview                 form pkg + t (form token): in-process apply into scratch, then the change report
                                        -> 303 to the preview's page
    GET  /apply/run/<pkg>/<ts>          one preview: counts, the apply's log, the report link, the CLI command, and the
                                        Apply section (type the package name to confirm) when it can be applied
    POST /apply/run                     form t + op=apply (pkg, ts, name = the typed package name) or op=retry (lv)
                                        -> 303 to /apply/done/<lv>; a refusal page otherwise. Never bearer.
    GET  /apply/done/<lv>               the result: "Applied" or "created, not yet registered" with Retry
    GET  /apply/view/<pkg>/<ts>/<path>  a file of that preview (report/index.html, its pictures, the scratch lv files)

The Apply button (round 11, run_apply), under ONE global OS lock (<state>/apply.lock; a second press is refused):
refused unless the preview exists and succeeded, its frozen decisions copy and the package's decisions now both hash
to the preview's edits_sha256, the typed name is the package's, the package is still eligible, these (pkg, sha) were
never applied (<state>/apply_runs.jsonl), and no lv_* under labels/ has base or supersedes == the package's base. Then:
leftover labels/.partial_lv_* folders removed; the frozen copy copied to <state>/apply_runs/<lv>/edits (kept) and
re-hashed; apply_review_edits.make_version (written to .partial_lv_<stamp>, renamed when complete; who = the admin's
people.json id, via None); registry/importers.import_label_versions (this label set only: no tile scan, no other
entity, no production restatement), verified with current(); the history page; a line in apply_runs.jsonl and one
audit line (PA.event). If the import or history step fails the label set stays and the result page offers Retry
(import + history again; both idempotent). The package's "applied" status line comes from the next `cryoev register`.

Eligible (all of):
  - the package's base resolves to a label version: portal.titles.lv_from_path(review_package.json base, else the
    data.js source) (NOT base_kind), and data.js's source is that label version's own file (<...>/lv_<id>/<file>) on
    this computer (apply_review_edits refuses any other base);
  - at least one image marked reviewed (the apply is reviewed-only);
  - not named by any annotation/review_server.py `--package` in the services config (read-only: env
    CRYOEV_SERVICES_CONFIG, else ~/.cryoev/services.json; an unreadable config makes nothing eligible);
  - not a prediction package (pred_*: never scanned).
Excluded ENTIRELY (not even listed): compare packages and any package with a <review root>/_work_<pkg> folder (a blind
audit's answer key; only its existence is checked, it is never listed or read).

Preview: the package's edits/*.json are copied into <state>/apply_preview/<pkg>/<UTC ts>/edits (frozen: the report's
"decisions copy sha256" is theirs), then apply_review_edits.make_version (reviewed_only, out_root
<ts>/labels, who = the admin's people.json id) and change_report.build (<ts>/report). The who is the portal id mapped
through annotation/people.json (id, name or alias; the rule of apply_review_edits.roster_lookup); an unmapped id is
refused before anything is written. Each preview gets its own folder; folders older than KEEP_DAYS are pruned on every
visit. A preview writes nothing outside <state>/apply_preview (and the portal audit line): never the label sets
(CryoAI-laptop/labels), the package or the registry (only the Apply button does, above). The page also shows the CLI
command (`--out-root` explicit, because the CLI's default <base>/../../labels is wrong for a label-set base).
Serve time re-checks: /run and /view answer 404 once the package has a _work_ folder or is a compare package; /run
warns "decisions changed since this preview" when the package's edits no longer match the frozen copy's sha256 (and
then hides the CLI command). The edits copy takes regular files only (no symlinks / junctions).
Served files: only under that preview's folder, by a strict name pattern per path part, real path checked, a fixed
list of types; the report's HTML gets a CSP with no scripts. Every text is escaped.
"""
from __future__ import annotations

import hashlib
import hmac
import html
import importlib.util
import json
import os
import re
import secrets
import shutil
import stat
import sys
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, quote

from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import FileResponse, RedirectResponse, Response
from starlette.routing import Route

from data_utils.oslock import acquire_lock
from portal import audit as PA
from portal import titles as TT
from portal.modules import admin as AD
from portal.modules import results as RES

PREFIX = "/apply"
ROLES = ("admin",)
REPO = Path(__file__).resolve().parents[2]
PEOPLE = REPO / "annotation" / "people.json"
PREVIEW_DIR = "apply_preview"
KEEP_DAYS = 7
MAX_FORM = 4096
PKG_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.\-]{0,120}")
TS_RE = re.compile(r"(\d{8}T\d{6}Z)(?:-(\d{1,3}))?")
PART_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.\-]{0,200}")
SERVE = {".html": "text/html; charset=utf-8", ".jpg": "image/jpeg", ".json": "application/json",
         ".jsonl": "text/plain; charset=utf-8", ".csv": "text/plain; charset=utf-8"}
REPORT_CSP = ("default-src 'none'; img-src 'self'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'; "
              "frame-ancestors 'none'; sandbox")
APPLY_DIR = "apply_runs"           # <state>/apply_runs/<lv>/edits: the frozen decisions each apply used (kept)
APPLY_LOG = "apply_runs.jsonl"     # <state>/apply_runs.jsonl: one line per apply / retry
APPLY_LOCK = "apply.lock"          # <state>/apply.lock: ONE global OS lock (stamps are per minute; supersedes scans
                                   # all of labels/)
LV_RE = re.compile(r"lv_\d{8}_\d{4}")
PARTIAL_RE = re.compile(r"\.partial_lv_\d{8}_\d{4}")   # = apply_review_edits.PARTIAL + lv: a run that never finished
MAX_RUN_FORM = 8192
esc = html.escape
_busy = threading.Lock()   # one preview at a time (taken by the worker itself: a client hang-up can't start a 2nd)
_applying = threading.Lock()   # one apply / retry per process (and APPLY_LOCK across processes)
_reg_mods = None
_reg_lock = threading.Lock()


# ---------------------------------------------------------------- inputs (read-only)
def services_config() -> Path:
    return Path(os.environ.get("CRYOEV_SERVICES_CONFIG") or Path.home() / ".cryoev" / "services.json")


def _is_review_server(argv) -> bool:
    """annotation/review_server.py (any path), or `-m annotation.review_server` / `-m review_server`."""
    mods = ("annotation.review_server", "review_server")
    for i, x in enumerate(argv):
        x = str(x)
        if re.split(r"[\\/]", x)[-1] == "review_server.py":
            return True
        if x == "-m" and i + 1 < len(argv) and str(argv[i + 1]) in mods:
            return True
        if x.startswith("-m") and x[2:] in mods:
            return True
    return False


def _package_arg(argv) -> list:
    """Values of --package, --package=..., and the prefixes argparse accepts for it (--pa, --pac, ... ; review_server's
    only other --p option is --port, so --pa is already unique)."""
    out = []
    for i, x in enumerate(argv):
        opt, eq, val = str(x).partition("=")
        if len(opt) >= 4 and "--package".startswith(opt):
            if eq:
                out.append(val)
            elif i + 1 < len(argv):
                out.append(str(argv[i + 1]))
    return out


def served_packages(path=None):
    """-> (names, problem). Package folder names given to the review server (annotation/review_server.py or
    -m annotation.review_server) as --package by any services config entry (enabled or not). No config file:
    (set(), None). Anything unreadable, malformed or failing while checking: (None, why): fail closed."""
    p = Path(path) if path else services_config()
    try:
        if not p.exists():
            return set(), None
        doc = json.loads(p.read_text(encoding="utf-8-sig"))
        svcs = doc.get("services")
        if not isinstance(svcs, list):
            raise ValueError("no services list")
        names = set()
        for s in svcs:
            argv = s.get("argv") if isinstance(s, dict) else None
            if isinstance(argv, str):
                argv = argv.split()
            if not isinstance(argv, list) or not _is_review_server(argv):
                continue
            for v in _package_arg(argv):
                name = re.split(r"[\\/]+", v.strip().rstrip("\\/ "))[-1]
                if name:
                    names.add(name)
        return names, None
    except Exception as e:                               # PermissionError, a folder, bad JSON ...: nothing eligible
        return None, f"the services config can't be read ({type(e).__name__})"


def load_people(path=None) -> list:
    try:
        people = json.loads(Path(path or PEOPLE).read_text(encoding="utf-8"))["people"]
        return [p for p in people if isinstance(p, dict) and isinstance(p.get("id"), str)]
    except (OSError, ValueError, KeyError, TypeError):
        return []


def people_id(portal_id, people=None):
    """The people.json id for a portal user id (its id, name or an alias; case- and space-insensitive: the rule of
    apply_review_edits.roster_lookup), else None."""
    if not isinstance(portal_id, str) or not portal_id.strip():
        return None
    for p in load_people() if people is None else people:
        for name in [p.get("id"), p.get("name"), *(p.get("aliases") or [])]:
            if isinstance(name, str) and name.strip().casefold() == portal_id.strip().casefold():
                return p["id"]
    return None


def base_file(row) -> Path | None:
    """data.js's source when it is the base label version's own file on this computer (<...>/lv_<id>/<file>)."""
    src, lv = row.get("source"), row.get("base_lv")
    if not (src and lv):
        return None
    p = Path(src)
    return p if p.parent.name == lv and p.is_file() else None


def packages(cfg, records, served, served_problem=None) -> list:
    """[{id, title, base_lv, base, reviewed, images, waiting, applied_by, last, eligible, why}] for the lab review
    packages under the review root whose base is a label version, compare / _work_ ones left out entirely."""
    root = RES.review_root(cfg)
    if root is None:
        return []
    out = []
    for p in TT.scan_packages(root, records):
        pkg = p["id"]
        if p["kind"] == "compare" or pkg.startswith("pred_") or not PKG_RE.fullmatch(pkg):
            continue
        if has_work(root, pkg):                          # a blind audit's answer key: exists() only, never read
            continue
        if not p["base_lv"]:
            continue                                     # a Roboflow export / pull base: not a label set
        row = {"id": pkg, "title": p["title"], "base_lv": p["base_lv"], "source": None, "base": None,
               "reviewed": p["reviewed"], "images": p["images"], "waiting": p["waiting"], "applied_by": p["applied_by"],
               "last": p["last"], "dir": Path(root) / pkg}
        try:
            row["source"] = (TT.package_progress(Path(root) / pkg) or {}).get("source")
            row["base"] = base_file(row)
        except Exception as e:                           # fail closed: listed, not eligible
            row["why"], row["eligible"] = [f"it can't be checked ({type(e).__name__})"], False
            out.append(row)
            continue
        why = []
        if row["base"] is None:
            why.append("its data.js does not name the label set's own file on this computer")
        if not p["reviewed"]:
            why.append("no image is marked reviewed yet (the apply takes reviewed images only)")
        if served is None:
            why.append(served_problem or "the services config can't be read")
        elif pkg in served:
            why.append("it is being reviewed live (a review server in the services config serves it)")
        row["why"], row["eligible"] = why, not why
        out.append(row)
    return sorted(out, key=lambda r: r["id"])


def has_work(root, pkg) -> bool:
    """<review root>/_work_<pkg> exists (a blind audit's answer key; never listed or read). Any error: True."""
    try:
        return root is None or (Path(root) / f"_work_{pkg}").exists()
    except Exception:
        return True


def excluded(cfg, pkg, records=None) -> bool:
    """Serve-time re-check: a _work_ folder, or a compare package (its review_package.json or registry record)."""
    root = RES.review_root(cfg)
    if has_work(root, pkg):
        return True
    try:
        meta = {**TT._dict(TT._dict(TT._dict(records).get("review_package")).get(pkg)),
                **TT._dict(TT.read_json(Path(root) / pkg / "review_package.json"))}
        return meta.get("kind") == "compare"
    except Exception:
        return True


def cli_command(row, who) -> str:
    base = Path(row["base"]) if row.get("base") else None
    q = lambda s: '"' + str(s).replace('"', '') + '"'  # noqa: E731
    return ("python annotation/apply_review_edits.py --base " + (q(base) if base else "<label set file>")
            + " --package " + q(row["dir"]) + " --out-root " + (q(base.parent.parent) if base else "<labels folder>")
            + " --who " + (who or "<your people.json id>"))


# ---------------------------------------------------------------- the scratch folder
def scratch_root(cfg) -> Path:
    return Path(cfg.state_dir) / PREVIEW_DIR


def ts_time(ts):
    m = TS_RE.fullmatch(ts or "")
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)


def prune(root: Path, now=None, keep_days=KEEP_DAYS) -> list:
    """Remove <root>/<pkg>/<ts>/ folders older than keep_days (by the ts in the name); -> removed. Only names this
    module makes are touched."""
    now = now or datetime.now(timezone.utc)
    gone = []
    try:
        pkgs = [d for d in Path(root).iterdir() if d.is_dir() and PKG_RE.fullmatch(d.name)]
    except OSError:
        return gone
    for d in pkgs:
        try:
            kids = list(d.iterdir())
        except OSError:
            continue
        for t in kids:
            when = ts_time(t.name) if t.is_dir() else None
            if when is not None and now - when > timedelta(days=keep_days):
                shutil.rmtree(t, ignore_errors=True)
                gone.append(t)
        try:
            next(d.iterdir())
        except StopIteration:
            d.rmdir()
        except OSError:
            pass
    return gone


def previews(cfg, pkg) -> list:
    """[{ts, info}] newest first, for one package."""
    d = scratch_root(cfg) / pkg
    out = []
    try:
        for t in d.iterdir():
            if t.is_dir() and TS_RE.fullmatch(t.name):
                try:
                    info = json.loads((t / "preview.json").read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    info = {}
                out.append({"ts": t.name, "info": info if isinstance(info, dict) else {}})
    except OSError:
        pass
    return sorted(out, key=lambda x: x["ts"], reverse=True)


def is_regular(path) -> bool:
    """A plain file: not a symlink, junction or other reparse point (lstat, never followed)."""
    try:
        st = os.lstat(path)
    except OSError:
        return False
    return stat.S_ISREG(st.st_mode) and not getattr(st, "st_file_attributes", 0) & 0x400  # REPARSE_POINT


def is_plain_dir(path) -> bool:
    try:
        st = os.lstat(path)
    except OSError:
        return False
    return stat.S_ISDIR(st.st_mode) and not getattr(st, "st_file_attributes", 0) & 0x400


def edit_files(edits_dir) -> list:
    """The edit records a preview copies: *.json regular files of a plain (not linked) folder, sorted."""
    if not is_plain_dir(edits_dir):
        return []
    return sorted(f for f in Path(edits_dir).glob("*.json") if is_regular(f))


def edits_sha256(edits_dir: Path) -> str:
    """The rule of annotation/compare_package.edits_sha256 (sha256 over the sorted names and their bytes), over
    edit_files() only."""
    h = hashlib.sha256()
    for f in edit_files(edits_dir):
        h.update(f.name.encode("utf-8") + b"\x00" + f.read_bytes() + b"\x00")
    return h.hexdigest()


def new_folder(root: Path, pkg: str, now) -> Path:
    """A fresh <root>/<pkg>/<ts>[-n]/. Another visit's prune may remove an empty <pkg> between the parent and the
    child mkdir: that is retried once; a second failure raises OSError (the caller shows a clean page)."""
    base = now.strftime("%Y%m%dT%H%M%SZ")
    for n in range(0, 1000):
        d = root / pkg / (base if n == 0 else f"{base}-{n}")
        for attempt in (0, 1):
            try:
                d.mkdir(parents=True, exist_ok=False)
                return d
            except FileExistsError:
                break
            except OSError:
                if attempt:
                    raise
    raise OSError("no free preview folder name")


def run_preview(cfg, row, who, now=None) -> tuple:
    """-> (ts, info), or (None, None) when another preview is running. The worker takes and releases the lock itself,
    so a request whose client hung up can't let a second preview start while this one still runs. Writes only under
    <state>/apply_preview/<pkg>/<ts>/. Raises OSError only when the folder can't be made."""
    if not _busy.acquire(blocking=False):
        return None, None
    try:
        return _run_preview(cfg, row, who, now)
    finally:
        _busy.release()


def _run_preview(cfg, row, who, now=None) -> tuple:
    from annotation import apply_review_edits as AR   # lazy: cv2 / numpy only when a preview runs
    from annotation import change_report as CR
    now = now or datetime.now(timezone.utc)
    root = scratch_root(cfg)
    prune(root, now)
    folder = new_folder(root, row["id"], now)
    info = {"pkg": row["id"], "who": who, "base_lv": row["base_lv"], "base": str(row["base"]),
            "started": now.isoformat(timespec="seconds"), "scope": "reviewed_only", "log": []}
    try:
        edits = folder / "edits"                       # frozen copy: the package may get new edits meanwhile
        edits.mkdir()
        if not is_plain_dir(row["dir"] / "edits"):
            raise ValueError("the package's edits folder is missing or a link")
        for f in edit_files(row["dir"] / "edits"):    # regular files only: no symlinks / junctions
            (edits / f.name).write_bytes(f.read_bytes())
        info["edits_sha256"] = edits_sha256(edits)
        ver = AR.make_version(row["base"], row["dir"], edits, out_root=folder / "labels", who=who,
                              stamp=now.strftime("%Y%m%d_%H%M"), scope="reviewed_only", via=None,
                              log=lambda s: info["log"].append(str(s)))
        info["lv"] = ver["id"]
        info["counts"] = {k: ver.get(k) for k in ("n_package_images", "n_reviewed", "n_held_back", "n_images_applied",
                                                  "n_annotations_base", "n_annotations_out", "n_conflicts", "n_dropped",
                                                  "n_warnings", "n_rejected")}
        info["counts_by_op"] = ver.get("counts_by_op")
        res = CR.build(folder / "labels" / ver["id"], row["dir"] / "images", folder / "report", None, 900,
                       info["edits_sha256"])
        info["report"] = {"images": res["images"], "changed_images": res["changed_images"], "missing": res["missing"]}
    except AR.ApplyError as e:
        info["error"] = str(e)
    except Exception as e:                               # a preview must never take the page down
        info["error"] = f"the preview failed ({type(e).__name__}: {e})"
    info["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (folder / "preview.json").write_text(json.dumps(info, indent=1) + "\n", encoding="utf-8")
    return folder.name, info


def served_file(cfg, pkg, ts, rel) -> Path | None:
    """The file <state>/apply_preview/<pkg>/<ts>/<rel>, or None: strict names, no '..', a known type, real path
    inside the preview folder."""
    if not (PKG_RE.fullmatch(pkg or "") and TS_RE.fullmatch(ts or "")):
        return None
    parts = (rel or "").split("/")
    if not parts or any(not PART_RE.fullmatch(x) or ".." in x for x in parts):
        return None
    if Path(parts[-1]).suffix.lower() not in SERVE:
        return None
    folder = scratch_root(cfg) / pkg / ts
    p = folder.joinpath(*parts)
    try:
        real, top = os.path.realpath(p), os.path.realpath(folder)
        if os.path.commonpath([os.path.normcase(real), os.path.normcase(top)]) != os.path.normcase(top):
            return None
    except (OSError, ValueError):
        return None
    return p if p.is_file() else None


# ---------------------------------------------------------------- the apply (round 11, #67)
def registry_modules():
    """-> namespace(importers, registry, history): registry/importers.py (and the registry.py module it imports) and
    registry/history_page.py. THE ONE place this module reaches the registry code: registry/ is not a package yet, so
    they are loaded by path (as jobs/convert_raw.load_importers does), sys.path restored afterwards. When registry/
    becomes a package (round 11 refactor #1) this becomes `from registry import importers, store, history_page`."""
    global _reg_mods
    with _reg_lock:
        if _reg_mods is not None:
            return _reg_mods
        reg_dir = REPO / "registry"
        saved_path, saved_reg = list(sys.path), sys.modules.get("registry")
        try:
            sys.path.insert(0, str(reg_dir))
            if saved_reg is not None and not hasattr(saved_reg, "current"):   # `registry` = the folder (namespace)
                sys.modules.pop("registry", None)     # importers' `import registry` must get registry/registry.py
            imp = sys.modules.get("importers")
            if imp is None or Path(getattr(imp, "__file__", "") or ".").resolve() != reg_dir / "importers.py":
                imp = _load_path("importers", reg_dir / "importers.py")
            hp = _load_path("cj_apply_history_page", reg_dir / "history_page.py")
        finally:                                      # the global name `registry` is left as it was
            sys.path[:] = saved_path
            if saved_reg is None:
                sys.modules.pop("registry", None)
            else:
                sys.modules["registry"] = saved_reg
        _reg_mods = SimpleNamespace(importers=imp, registry=imp.registry, history=hp)
        return _reg_mods


def _load_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return mod


def runs_log(cfg) -> Path:
    return Path(cfg.state_dir) / APPLY_LOG


def read_runs(cfg) -> list:
    """apply_runs.jsonl lines, oldest first ([] when there is none; a cut line is skipped). Raises OSError when the
    file is there but can't be read (an apply is then refused)."""
    try:
        text = runs_log(cfg).read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    out = []
    for line in text.splitlines():
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if isinstance(rec, dict):
            out.append(rec)
    return out


def run_state(cfg, lv) -> dict | None:
    """The newest apply_runs.jsonl line for one label set (its apply, or a later retry), else None."""
    try:
        return next((r for r in reversed(read_runs(cfg)) if r.get("lv") == lv), None)
    except OSError:
        return None


def recorded(rec) -> bool:
    steps = (rec or {}).get("steps") or {}
    return steps.get("import") == "ok" and steps.get("history") == "ok"


def lvs_on_base(labels: Path, base_lv: str) -> list:
    """lv_* folders under labels/ whose version.json has base == base_lv or supersedes == base_lv (the scan of
    importers.label_versions). A version.json that can't be read counts as one (it can't be ruled out)."""
    hits = []
    for d in sorted(Path(labels).glob("lv_*")):
        vj = d / "version.json"
        if d.name == base_lv or not LV_RE.fullmatch(d.name) or not vj.exists():
            continue
        try:
            v = json.loads(vj.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            hits.append(d.name)
            continue
        if not isinstance(v, dict) or base_lv in (v.get("base"), v.get("supersedes")):
            hits.append(d.name)
    return hits


def clean_partials(labels: Path) -> list:
    """Remove <labels>/.partial_lv_<stamp>/ folders (a make_version that never finished). Only under APPLY_LOCK."""
    gone = []
    try:
        kids = [d for d in Path(labels).iterdir() if PARTIAL_RE.fullmatch(d.name) and is_plain_dir(d)]
    except OSError:
        return gone
    for d in kids:
        shutil.rmtree(d, ignore_errors=True)
        gone.append(d.name)
    return gone


def apply_refusal(cfg, row, pkg, ts, typed, who, records=None) -> tuple:
    """-> (why or None, {info, sha, labels}). Every precondition of an apply, each a clear refusal: the preview exists
    and succeeded for this package and label set; its frozen decisions copy still hashes to the preview's sha and so do
    the package's decisions now; the typed name is the package's; the package is eligible (and not a compare / _work_
    package); these decisions were not applied before; no other label set builds on (or replaces) the same base."""
    if who is None:
        return "your portal id is not in annotation/people.json (as an id, name or alias)", {}
    if row is None or not row.get("eligible") or excluded(cfg, pkg, records):
        why = "; ".join((row or {}).get("why") or []) or "it is not a lab review package on a label set"
        return f"the package is not ready to apply ({why})", {}
    if (typed or "").strip() != pkg:
        return "the typed name is not the package's name", {}
    p = served_file(cfg, pkg, ts, "preview.json")
    try:
        info = json.loads(p.read_text(encoding="utf-8")) if p else None
    except (OSError, ValueError):
        info = None
    if not isinstance(info, dict):
        return f"no such preview (they are kept {KEEP_DAYS} days): preview again", {}
    if info.get("error") or not info.get("lv"):
        return "that preview failed: preview again", {}
    if (info.get("pkg"), info.get("base_lv"), info.get("base"), info.get("scope")) != \
            (pkg, row["base_lv"], str(row["base"]), "reviewed_only"):
        return "that preview was made for another package, label set or scope: preview again", {}
    sha = info.get("edits_sha256")
    if not sha or edits_sha256(p.parent / "edits") != sha:
        return "the preview's frozen decisions copy is missing or changed: preview again", {}
    if not is_plain_dir(row["dir"] / "edits") or edits_sha256(row["dir"] / "edits") != sha:
        return "the decisions changed since this preview: preview again, check the report, then apply", {}
    try:
        runs = read_runs(cfg)
    except OSError as e:
        return f"the apply log can't be read ({type(e).__name__})", {}
    done = next((r for r in runs if r.get("op") == "apply" and r.get("pkg") == pkg and r.get("edits_sha256") == sha),
                None)
    if done:
        return f"these decisions were already applied as {done.get('lv')}", {}
    labels = Path(row["base"]).parent.parent
    if labels.name != "labels":
        return f"the label set is not in a labels folder ({labels})", {}
    hits = lvs_on_base(labels, row["base_lv"])
    if hits:
        return (f"{', '.join(hits)} already builds on (or replaces) {row['base_lv']}: a second label set from the same "
                f"base would fork it"), {}
    return None, {"info": info, "sha": sha, "labels": labels, "frozen": p.parent / "edits"}


def record_steps(cfg, lv, labels: Path, supersedes=None) -> dict:
    """The registry import of ONLY this label set (and the one it supersedes, whose status changes), verified with
    current(), then the history page. Both idempotent (Retry runs them again). -> {import, history, appended}."""
    steps = {}
    root = getattr(cfg, "registry", None)
    try:
        if not root:
            raise RuntimeError("the portal has no registry folder configured")
        mods = registry_modules()
        res = mods.importers.import_label_versions(root, via=None, laptop=Path(labels).parent,
                                                   only=[x for x in (lv, supersedes) if x])
        steps["appended"] = sum(res["appended"].values())
        if lv not in mods.registry.current("label_version", root=root):
            raise RuntimeError("not in the registry after the import (refused: "
                               + (", ".join(f"{k}: {v}" for k, v in res["refused"].items()) or "none") + ")")
        steps["import"] = "ok"
    except Exception as e:                               # shown with a Retry button; the label set stays
        steps["import"] = f"failed ({type(e).__name__}: {e})"[:400]
        steps["history"] = "not run (the import failed)"
        return steps
    try:
        mods.history.build([str(root)], Path(root) / "history.html")
        steps["history"] = "ok"
    except (Exception, SystemExit) as e:                 # build() refuses a protected path with SystemExit
        steps["history"] = f"failed ({type(e).__name__}: {e})"[:400]
    return steps


def append_run(cfg, rec) -> None:
    path = runs_log(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _locked(cfg, fn):
    """fn() holding the per-process lock and THE global OS lock (<state>/apply.lock), or ("busy", {}) at once when
    either is held: a second press is refused, never queued."""
    if not _applying.acquire(blocking=False):
        return "busy", {}
    try:
        state = Path(cfg.state_dir)
        state.mkdir(parents=True, exist_ok=True)
        lk = acquire_lock(state / APPLY_LOCK, wait=0)
        if lk is None:
            return "busy", {}
        try:
            return fn()
        finally:
            lk.close()
    finally:
        _applying.release()


def run_apply(cfg, row, pkg, ts, typed, who, records=None, now=None) -> tuple:
    """The Apply button's work, in process, under the lock. `row`: the package's packages() row, or a callable
    returning it (then evaluated under the lock, so eligibility is re-checked at press time). -> (status, detail):
      ("busy", {})                    another apply / retry holds the lock;
      ("refused", {"why"})            a precondition failed (apply_refusal) or make_version refused: nothing written;
      ("failed", {"why"})             make_version failed mid-way: no lv_ folder (its partial folder is removed);
      ("done", rec)                   the label set exists; rec["steps"] says whether the import / history ran.
    Steps: leftover .partial_lv_* folders removed; the preview's frozen decisions copied to
    <state>/apply_runs/<lv>/edits (kept) and re-hashed; apply_review_edits.make_version into <labels> (via None: a
    person pressed it); record_steps (registry import of this label set only + history page); a line in
    <state>/apply_runs.jsonl."""
    return _locked(cfg, lambda: _run_apply(cfg, row, pkg, ts, typed, who, records, now))


def _run_apply(cfg, row, pkg, ts, typed, who, records, now):
    from annotation import apply_review_edits as AR   # lazy: cv2 / numpy only when an apply runs
    row = row() if callable(row) else row
    why, got = apply_refusal(cfg, row, pkg, ts, typed, who, records)
    if why:
        return "refused", {"why": why}
    labels, sha = got["labels"], got["sha"]
    clean_partials(labels)
    stamp = (now or datetime.now()).strftime("%Y%m%d_%H%M")   # local time, as the CLI's default stamp
    lv = f"lv_{stamp}"
    if (labels / lv).exists():
        return "refused", {"why": f"{lv} already exists (label sets are stamped by the minute): try again in a minute"}
    run_dir = Path(cfg.state_dir) / APPLY_DIR / lv
    if run_dir.exists():                                 # an earlier attempt that made no label set: not referenced
        shutil.rmtree(run_dir)
    edits = run_dir / "edits"
    edits.mkdir(parents=True)
    for f in edit_files(got["frozen"]):
        (edits / f.name).write_bytes(f.read_bytes())
    if edits_sha256(edits) != sha:
        shutil.rmtree(run_dir, ignore_errors=True)
        return "refused", {"why": "the preview's frozen decisions changed while copying: preview again"}
    log = []
    try:
        ver = AR.make_version(row["base"], row["dir"], edits, out_root=labels, who=who, stamp=stamp,
                              scope="reviewed_only", via=None, log=lambda s: log.append(str(s)))
    except AR.ApplyError as e:
        shutil.rmtree(run_dir, ignore_errors=True)
        return "refused", {"why": str(e)}
    except Exception as e:                               # the partial folder is removed by make_version
        shutil.rmtree(run_dir, ignore_errors=True)
        return "failed", {"why": f"the apply failed ({type(e).__name__}: {e}); no label set was made"}
    try:
        (run_dir / "apply.log").write_text("\n".join(log) + "\n", encoding="utf-8")
    except OSError:
        pass
    rec = {"t": datetime.now(timezone.utc).isoformat(timespec="seconds"), "op": "apply", "pkg": pkg,
           "edits_sha256": sha, "lv": lv, "base_lv": row["base_lv"], "who": who, "preview": ts,
           "labels": str(labels), "supersedes": ver.get("supersedes"), "edits": str(edits),
           "steps": {"apply": "ok", **record_steps(cfg, lv, labels, ver.get("supersedes"))}}
    try:
        append_run(cfg, rec)
    except OSError as e:                                 # the label set stands; the base scan still blocks a 2nd one
        rec["log_error"] = f"{type(e).__name__}: {e}"
    return "done", rec


def run_retry(cfg, lv) -> tuple:
    """Retry = the registry import + history page again for a label set this page made (both idempotent). -> as
    run_apply ("refused" when there is no such apply, its label set is gone, or nothing is left to do)."""
    return _locked(cfg, lambda: _run_retry(cfg, lv))


def _run_retry(cfg, lv):
    last = run_state(cfg, lv) if LV_RE.fullmatch(lv or "") else None
    if last is None:
        return "refused", {"why": "no apply of that label set is recorded here"}
    if recorded(last):
        return "refused", {"why": f"{lv} is already recorded and the history page updated: nothing to retry"}
    labels = Path(last.get("labels") or "")
    if not (labels / lv / "version.json").is_file():
        return "refused", {"why": f"{lv} is not in {labels} any more"}
    rec = {**{k: last.get(k) for k in ("pkg", "edits_sha256", "lv", "base_lv", "who", "preview", "labels",
                                         "supersedes", "edits")},
           "t": datetime.now(timezone.utc).isoformat(timespec="seconds"), "op": "retry",
           "steps": {"apply": "ok", **record_steps(cfg, lv, labels, last.get("supersedes"))}}
    try:
        append_run(cfg, rec)
    except OSError as e:
        rec["log_error"] = f"{type(e).__name__}: {e}"
    return "done", rec
STYLE = ("<style>.cj-apply table{border-collapse:collapse;width:100%;font-size:.9rem}"
         ".cj-apply td,.cj-apply th{padding:4px 6px;border-bottom:1px solid var(--line);text-align:left;"
         "vertical-align:top}.cj-apply code,.cj-apply pre{font-size:.8rem;overflow-wrap:anywhere;white-space:pre-wrap}"
         ".cj-apply .cj-card{margin:0 0 12px}.cj-apply form{display:inline}</style>")


def _when(iso) -> str:
    return esc(str(iso or "")[:19].replace("T", " ")) + (" UTC" if iso else "")


def render_index(rows, who, portal_id, tok, cfg, problem=None) -> str:
    out = [STYLE, '<div class="cj-apply"><h1>Apply preview</h1>',
           '<p class="cj-mut">What applying a review package\'s decisions (reviewed images only) would change in its '
           'label set. A preview is a scratch apply: nothing is written to the label sets or the registry. To make the '
           'new label set, open a preview and confirm it there (Apply and register).</p>']
    if problem:
        out.append(f'<p class="cj-card" role="alert">{esc(problem)}</p>')
    if who is None:
        out.append(f'<p class="cj-card" role="alert">Your portal id <b>{esc(str(portal_id))}</b> is not in '
                   f'annotation/people.json (as an id, name or alias), so a preview can\'t record who made it. Add it '
                   f'there first.</p>')
    ok = [r for r in rows if r["eligible"]]
    no = [r for r in rows if not r["eligible"]]
    if not ok:
        out.append('<p class="cj-card">No review package is ready for a preview.</p>')
    for r in ok:
        prev = previews(cfg, r["id"])[:5]
        plist = "".join(
            f'<li><a href="{PREFIX}/run/{quote(r["id"])}/{quote(p["ts"])}">{_when(p["info"].get("started"))}</a>'
            f'{" (failed)" if p["info"].get("error") else ""}</li>' for p in prev)
        applied = (f'<p class="cj-mut">Already applied as {esc(r["applied_by"])}: a new label set from it would '
                   f'supersede that one.</p>' if r["applied_by"] else "")
        waiting = "" if r["applied_by"] else f'; {r["waiting"]} decisions not applied'
        button = ("" if who is None else
                  f'<form method="post" action="{PREFIX}/preview">{AD.hidden_token(tok)}'
                  f'<input type="hidden" name="pkg" value="{esc(r["id"])}"><button type="submit">Preview apply</button>'
                  f'</form>')
        out.append(
            f'<section class="cj-card" id="p-{esc(r["id"])}"><h2>{esc(r["title"])}</h2>'
            f'<p>Package <code>{esc(r["id"])}</code> on label set <code>{esc(r["base_lv"])}</code>: '
            f'{r["reviewed"]} of {r["images"]} images reviewed{waiting}.</p>{applied}'
            f'{button}'
            + (f'<p class="cj-mut">Recent previews (kept {KEEP_DAYS} days):</p><ul>{plist}</ul>' if plist else "")
            + f'<p class="cj-mut">To make the label set (run in the cryoEV folder):</p>'
              f'<pre>{esc(cli_command(r, who))}</pre></section>')
    if no:
        body = "".join(f'<tr><td><code>{esc(r["id"])}</code></td><td>{esc(r["base_lv"])}</td>'
                       f'<td>{esc("; ".join(r["why"]))}</td></tr>' for r in no)
        out.append(f'<section class="cj-card"><h2>Not ready</h2><table><thead><tr><th>Package</th><th>Label set</th>'
                   f'<th>Why</th></tr></thead><tbody>{body}</tbody></table></section>')
    out.append("</div>")
    return "".join(out)


def change_line(info) -> str:
    """The mockup's "Changes: +12 boxes · −5 · 31 edited · 2 conflicts" from a preview's counts."""
    ops, c = info.get("counts_by_op") or {}, info.get("counts") or {}
    n = lambda k: ops.get(k) if isinstance(ops.get(k), int) else 0  # noqa: E731
    return (f'+{n("add") + n("new")} boxes &middot; &minus;{n("delete")} &middot; '
            f'{n("reclass") + n("reshape") + n("reclass+reshape")} edited &middot; '
            f'{esc(str(c.get("n_conflicts") or 0))} conflicts')


def render_confirm(pkg, ts, info, row, who, tok, why) -> str:
    """The Apply section of a preview's page: the typed-name confirm, or why it can't be applied."""
    if why:
        return (f'<section class="cj-card" id="apply"><h2>Apply</h2><p class="cj-mut">Not available: {esc(why)}.</p>'
                f'</section>')
    report = f'{PREFIX}/view/{quote(pkg)}/{quote(ts)}/report/index.html'
    base = esc(str(row["base_lv"]))
    return (f'<section class="cj-card" id="apply"><h2>Apply</h2>'
            f'<p><code>{esc(pkg)}</code> &nbsp; base <code>{base}</code> &middot; {esc(str(row["reviewed"]))} images '
            f'reviewed</p>'
            f'<p>Preview {_when(info.get("started"))} &middot; decisions unchanged since &#10003; &nbsp; '
            f'<a href="{report}">View change report</a></p>'
            f'<p>Changes: {change_line(info)}</p>'
            f'<p>Will create <code>labels/lv_&lt;date_time of the press&gt;</code> (built on <code>{base}</code>), by '
            f'{esc(str(who))}; only that label set is recorded in the registry, then the history page is updated.</p>'
            f'<form method="post" action="{PREFIX}/run">{AD.hidden_token(tok)}'
            f'<input type="hidden" name="op" value="apply"><input type="hidden" name="pkg" value="{esc(pkg)}">'
            f'<input type="hidden" name="ts" value="{esc(ts)}">'
            f'<label>Type the package name to confirm: <input name="name" autocomplete="off" spellcheck="false" '
            f'required></label> <button type="submit">Apply and register</button></form>'
            f'<p class="cj-mut">The package\'s "applied" status line comes from the next normal '
            f'<code>cryoev register</code>.</p></section>')


def render_done(lv, rec, tok, run_log="") -> str:
    """The result of an apply / retry (from apply_runs.jsonl): Applied, or created-not-recorded with Retry."""
    steps = rec.get("steps") or {}
    out = [STYLE, '<div class="cj-apply">', f'<p><a href="{PREFIX}/">Back to apply previews</a></p>',
           f'<h1>Apply: {esc(str(rec.get("pkg")))}</h1>']
    if recorded(rec):
        out.append(f'<p class="cj-card" role="status"><b>Applied &#10003;</b> <code>{esc(lv)}</code> created '
                   f'&middot; registered &middot; history updated</p>')
    else:
        what = ("created, not yet registered" if steps.get("import") != "ok"
                else "created and registered; the history page was not updated")
        out.append(f'<section class="cj-card" role="alert"><p><b><code>{esc(lv)}</code> {what}</b></p>'
                   f'<p class="cj-mut">Registry import: {esc(str(steps.get("import")))}<br>History page: '
                   f'{esc(str(steps.get("history")))}</p>'
                   f'<form method="post" action="{PREFIX}/run">{AD.hidden_token(tok)}'
                   f'<input type="hidden" name="op" value="retry"><input type="hidden" name="lv" value="{esc(lv)}">'
                   f'<button type="submit">Retry register</button></form></section>')
    if rec.get("log_error"):
        out.append(f'<p class="cj-card" role="alert">The apply log could not be written ({esc(rec["log_error"])}), so '
                   f'Retry is not available here: run <code>cryoev register</code>.</p>')
    rows = [("Label set", f'<code>{esc(lv)}</code> in {esc(str(rec.get("labels")))}'),
            ("Built on", f'<code>{esc(str(rec.get("base_lv")))}</code>'),
            ("Supersedes", esc(str(rec.get("supersedes") or "none"))),
            ("By", esc(str(rec.get("who")))), ("Preview", esc(str(rec.get("preview")))),
            ("Decisions copy", f'{esc(str(rec.get("edits")))}<br><code>sha256 {esc(str(rec.get("edits_sha256")))}'
                               f'</code>'),
            ("Last step", f'{esc(str(rec.get("op")))} {_when(rec.get("t"))}')]
    out.append('<section class="cj-card"><table>' + "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in rows)
               + '</table><p class="cj-mut">The package\'s "applied" status line comes from the next normal '
                 '<code>cryoev register</code>.</p></section>')
    if run_log:
        out.append(f'<section class="cj-card"><h2>Apply log</h2><pre>{esc(run_log)}</pre></section>')
    out.append("</div>")
    return "".join(out)


def render_run(pkg, ts, info, row, who, changed=False, confirm="") -> str:
    out = [STYLE, '<div class="cj-apply">', f'<p><a href="{PREFIX}/">Back to apply previews</a></p>',
           f'<h1>Preview: {esc(pkg)}</h1>',
           f'<p class="cj-mut">Started {_when(info.get("started"))} by {esc(str(info.get("who")))}; scratch only, '
           f'reviewed images only. Label set <code>{esc(str(info.get("base_lv")))}</code>.</p>']
    if changed:
        out.append('<p class="cj-card" role="alert"><b>Decisions changed since this preview</b>: preview again before '
                   'applying.</p>')
    out.append(confirm)
    if info.get("error"):
        out.append(f'<section class="cj-card" role="alert"><h2>Refused</h2><pre>{esc(info["error"])}</pre></section>')
    else:
        c = info.get("counts") or {}
        ops = info.get("counts_by_op") or {}
        rows = "".join(f"<tr><td>{esc(k)}</td><td>{esc(str(v))}</td></tr>" for k, v in c.items())
        orow = "".join(f"<tr><td>{esc(k)}</td><td>{esc(str(v))}</td></tr>" for k, v in ops.items() if v)
        rep = info.get("report") or {}
        base = f'{PREFIX}/view/{quote(pkg)}/{quote(ts)}'
        out.append(
            f'<section class="cj-card"><h2>Change report</h2><p><a href="{base}/report/index.html">Open the change '
            f'report</a> ({esc(str(rep.get("images")))} of {esc(str(rep.get("changed_images")))} changed images '
            f'drawn).</p><p class="cj-mut">Scratch files: <a href="{base}/labels/{quote(str(info.get("lv")))}/version.json">'
            f'version.json</a> · <a href="{base}/labels/{quote(str(info.get("lv")))}/changes.jsonl">changes.jsonl</a> · '
            f'<a href="{base}/labels/{quote(str(info.get("lv")))}/conflicts.csv">conflicts.csv</a></p></section>'
            f'<section class="cj-card"><h2>Counts</h2><table>{rows}</table></section>'
            f'<section class="cj-card"><h2>Applied by kind</h2><table>{orow or "<tr><td>none</td></tr>"}</table>'
            f'</section>')
    out.append(f'<section class="cj-card"><h2>Apply log</h2><pre>{esc(chr(10).join(info.get("log") or []))}</pre>'
               f'<p class="cj-mut">Decisions copy sha256 <code>{esc(str(info.get("edits_sha256") or ""))}</code></p>'
               f'</section>')
    if row is not None and not changed:
        out.append(f'<section class="cj-card"><h2>Or by hand</h2><p class="cj-mut">The same label set from the command '
                   f'line (run in the cryoEV folder; it does not register it: run <code>cryoev register</code> after):'
                   f'</p><pre>{esc(cli_command(row, who))}</pre></section>')
    out.append("</div>")
    return "".join(out)


# ---------------------------------------------------------------- routes
def routes(ctx) -> list:
    secret = secrets.token_bytes(32)                     # per process, like admin's: a restart makes open forms stale

    def token(request: Request) -> str:
        return hmac.new(secret, hashlib.sha256(AD.binding(request).encode()).digest(), hashlib.sha256).hexdigest()[:40]

    def show(request, title, body, status=200):
        r = ctx.page(request, title, body, active="data")
        r.status_code = status
        return r

    def admin_only(request):
        if ctx.user(request).get("role") != "admin":      # the Gate already refuses; defence in depth
            return show(request, "Apply preview", '<p class="cj-card">Admins only.</p>', 403)
        if request.scope.get("cj_auth") == "bearer":      # never an agent token (the Gate never opens these to one)
            return show(request, "Apply preview", '<p class="cj-card">Not with an agent token.</p>', 403)
        return None

    def records():
        from portal.modules import data as DATA
        return DATA.read_registry(getattr(ctx.cfg, "registry", None))[0]

    def state(request):
        served, problem = served_packages()
        return packages(ctx.cfg, records(), served, problem), problem

    def gone(request):
        return show(request, "Apply preview", '<p class="cj-card">No such preview (they are kept '
                    f'{KEEP_DAYS} days).</p><p><a href="{PREFIX}/">Back</a></p>', 404)

    def index(request: Request):
        if (deny := admin_only(request)) is not None:
            return deny
        prune(scratch_root(ctx.cfg))
        rows, problem = state(request)
        pid = ctx.user(request).get("id")
        return show(request, "Apply preview", render_index(rows, people_id(pid), pid, token(request), ctx.cfg, problem))

    async def preview(request: Request):
        if (deny := admin_only(request)) is not None:
            return deny
        ctype = (request.headers.get("content-type") or "").split(";")[0].strip().lower()
        buf = bytearray()
        async for chunk in request.stream():
            buf += chunk
            if len(buf) > MAX_FORM:
                return show(request, "Apply preview", '<p class="cj-card">That form is too large.</p>', 413)
        try:
            f = {k: v[0] for k, v in parse_qs(bytes(buf).decode("utf-8"), max_num_fields=4).items() if v}
        except (UnicodeDecodeError, ValueError):
            f = None
        if ctype != "application/x-www-form-urlencoded" or f is None:
            return show(request, "Apply preview", '<p class="cj-card">Send the form from the Apply preview page.</p>',
                        400)
        if not hmac.compare_digest(f.get("t", "").encode(), token(request).encode()):
            return show(request, "Apply preview", '<p class="cj-card" role="alert">This form has expired. Reload the '
                        f'page and try again.</p><p><a href="{PREFIX}/">Back</a></p>', 403)
        pid = ctx.user(request).get("id")
        who = people_id(pid)
        if who is None:
            return show(request, "Apply preview", f'<p class="cj-card" role="alert">Refused: your portal id '
                        f'<b>{esc(str(pid))}</b> is not in annotation/people.json (as an id, name or alias). Add it there '
                        f'first; nothing was run.</p><p><a href="{PREFIX}/">Back</a></p>', 403)
        rows, _ = await run_in_threadpool(state, request)
        row = next((r for r in rows if r["id"] == f.get("pkg")), None)
        if row is None or not row["eligible"]:
            return show(request, "Apply preview", '<p class="cj-card" role="alert">That package is not ready for a '
                        f'preview (or does not exist).</p><p><a href="{PREFIX}/">Back</a></p>', 404)
        try:
            ts, info = await run_in_threadpool(run_preview, ctx.cfg, row, who)
        except OSError:
            return show(request, "Apply preview", '<p class="cj-card" role="alert">The preview folder could not be '
                        f'made. Try again.</p><p><a href="{PREFIX}/">Back</a></p>', 503)
        if ts is None:
            return show(request, "Apply preview", '<p class="cj-card">Another preview is running. Try again in a '
                        f'minute.</p><p><a href="{PREFIX}/">Back</a></p>', 409)
        PA.event(ctx.cfg.state_dir, request, "apply preview", row["id"],
                 {"ts": ts, "lv": info.get("lv"), "ok": not info.get("error")})
        return RedirectResponse(f"{PREFIX}/run/{quote(row['id'])}/{quote(ts)}", status_code=303)

    def run_page(request: Request):
        if (deny := admin_only(request)) is not None:
            return deny
        pkg, ts = request.path_params["pkg"], request.path_params["ts"]
        p = served_file(ctx.cfg, pkg, ts, "preview.json")
        if p is None or excluded(ctx.cfg, pkg, records()):
            return gone(request)
        try:
            info = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            info = {"error": "this preview's record can't be read"}
        recs = records()
        served, problem = served_packages()
        rows = packages(ctx.cfg, recs, served, problem)
        any_row = next((r for r in rows if r["id"] == pkg), None)
        row = any_row if any_row is not None and any_row["eligible"] else None
        root = RES.review_root(ctx.cfg)
        try:
            now_sha = edits_sha256(Path(root) / pkg / "edits")
        except Exception:
            now_sha = None
        changed = not info.get("error") and info.get("edits_sha256") != now_sha
        pid = ctx.user(request).get("id")
        who = people_id(pid)
        confirm = ""
        if not info.get("error"):
            try:
                why, _ = apply_refusal(ctx.cfg, any_row, pkg, ts, pkg, who, recs)   # the typed name is checked on press
            except Exception as e:                                                # never takes the page down
                why = f"it can't be checked ({type(e).__name__})"
            confirm = render_confirm(pkg, ts, info, any_row, who, token(request), why)
        return show(request, f"Preview {pkg}", render_run(pkg, ts, info, row, who, changed, confirm))

    async def read_form(request, title):
        """-> (fields, None) or (None, the refusal page): urlencoded, at most MAX_RUN_FORM bytes, a valid token."""
        ctype = (request.headers.get("content-type") or "").split(";")[0].strip().lower()
        buf = bytearray()
        async for chunk in request.stream():
            buf += chunk
            if len(buf) > MAX_RUN_FORM:
                return None, show(request, title, '<p class="cj-card">That form is too large.</p>', 413)
        try:
            f = {k: v[0] for k, v in parse_qs(bytes(buf).decode("utf-8"), max_num_fields=8).items() if v}
        except (UnicodeDecodeError, ValueError):
            f = None
        if ctype != "application/x-www-form-urlencoded" or f is None:
            return None, show(request, title, '<p class="cj-card">Send the form from the Apply page.</p>', 400)
        if not hmac.compare_digest(f.get("t", "").encode(), token(request).encode()):
            return None, show(request, title, '<p class="cj-card" role="alert">This form has expired. Reload the '
                              f'page and try again.</p><p><a href="{PREFIX}/">Back</a></p>', 403)
        return f, None

    async def run(request: Request):
        """POST /apply/run: op=apply (pkg, ts, name: the typed package name) or op=retry (lv). Admin, form token, the
        Gate's Origin check; never bearer. Success -> 303 to /apply/done/<lv> and one audit line (PA.event: the route
        is in app.py SELF_LOGGED); a refusal -> its page (4xx), which the Gate's catch-all logs."""
        if (deny := admin_only(request)) is not None:
            return deny
        f, err = await read_form(request, "Apply")
        if err is not None:
            return err
        op = f.get("op")
        pkg, ts = f.get("pkg", ""), f.get("ts", "")
        back = (f'{PREFIX}/run/{quote(pkg)}/{quote(ts)}' if PKG_RE.fullmatch(pkg) and TS_RE.fullmatch(ts)
                else f"{PREFIX}/")
        if op == "apply":
            pid = ctx.user(request).get("id")
            who = people_id(pid)
            if who is None:
                return show(request, "Apply", f'<p class="cj-card" role="alert">Refused: your portal id <b>'
                            f'{esc(str(pid))}</b> is not in annotation/people.json (as an id, name or alias). Nothing '
                            f'was run.</p><p><a href="{PREFIX}/">Back</a></p>', 403)
            if not (PKG_RE.fullmatch(pkg) and TS_RE.fullmatch(ts)):
                return gone(request)

            recs = await run_in_threadpool(records)

            def find_row():                              # evaluated under the apply lock
                served, problem = served_packages()
                return next((r for r in packages(ctx.cfg, recs, served, problem) if r["id"] == pkg), None)
            status, d = await run_in_threadpool(run_apply, ctx.cfg, find_row, pkg, ts, f.get("name"), who, recs)
        elif op == "retry":
            status, d = await run_in_threadpool(run_retry, ctx.cfg, f.get("lv", ""))
            back = f'{PREFIX}/done/{quote(f.get("lv", ""))}' if LV_RE.fullmatch(f.get("lv", "")) else f"{PREFIX}/"
        else:
            return show(request, "Apply", '<p class="cj-card">Send the form from the Apply page.</p>', 400)
        if status == "busy":
            return show(request, "Apply", '<p class="cj-card" role="alert">Another apply is running. Nothing was done; '
                        f'try again in a minute.</p><p><a href="{esc(back)}">Back</a></p>', 409)
        if status != "done":
            return show(request, "Apply", f'<section class="cj-card" role="alert"><h2>Refused</h2><p>{esc(d["why"])}'
                        f'.</p><p class="cj-mut">Nothing was written.</p></section>'
                        f'<p><a href="{esc(back)}">Back</a></p>', 500 if status == "failed" else 409)
        steps = d["steps"]
        PA.event(ctx.cfg.state_dir, request, "apply" if op == "apply" else "apply retry", d.get("pkg"),
                 {"lv": d["lv"], "edits_sha256": d.get("edits_sha256"), "preview": d.get("preview"),
                  "import": steps.get("import"), "history": steps.get("history")})
        if d.get("log_error"):                           # no apply_runs line to show the result page from
            return show(request, "Apply", render_done(d["lv"], d, token(request)))
        return RedirectResponse(f"{PREFIX}/done/{quote(d['lv'])}", status_code=303)

    def done(request: Request):
        if (deny := admin_only(request)) is not None:
            return deny
        lv = request.path_params["lv"]
        rec = run_state(ctx.cfg, lv) if LV_RE.fullmatch(lv) else None
        if rec is None:
            return show(request, "Apply", '<p class="cj-card">No apply of that label set is recorded here.</p>'
                        f'<p><a href="{PREFIX}/">Back</a></p>', 404)
        try:
            run_log = (Path(ctx.cfg.state_dir) / APPLY_DIR / lv / "apply.log").read_text(encoding="utf-8")
        except OSError:
            run_log = ""
        return show(request, f"Applied {lv}", render_done(lv, rec, token(request), run_log))

    def view(request: Request):
        if (deny := admin_only(request)) is not None:
            return deny
        pkg = request.path_params["pkg"]
        p = served_file(ctx.cfg, pkg, request.path_params["ts"], request.path_params["rel"])
        if p is None or excluded(ctx.cfg, pkg, records()):
            return ctx.no_store(Response("Not found", status_code=404, media_type="text/plain"))
        r = FileResponse(p, media_type=SERVE[p.suffix.lower()])
        r.headers["Cache-Control"] = "no-store"
        if p.suffix.lower() == ".html":
            r.headers["Content-Security-Policy"] = REPORT_CSP
        return r

    return [Route(PREFIX + "/", index, methods=["GET"]),
            Route(PREFIX + "/preview", preview, methods=["POST"]),
            Route(PREFIX + "/run", run, methods=["POST"]),
            Route(PREFIX + "/run/{pkg}/{ts}", run_page, methods=["GET"]),
            Route(PREFIX + "/done/{lv}", done, methods=["GET"]),
            Route(PREFIX + "/view/{pkg}/{ts}/{rel:path}", view, methods=["GET"])]
