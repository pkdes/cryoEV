"""
Predict, raw conversion records and the runner: conversion folders, the converter / stitcher child processes, one
at a time (queued -> running -> done / failed), registration of upload sessions, the input sweep, the watcher
thread, and a conversion's status in plain words.

Split out of portal/modules/predict.py (round 7, P7; no behaviour change); predict.py re-exports every name.
Names the tests patch are used from other parts only as <PART>.<name> (portal/predict_parts/__init__.py).
"""
from __future__ import annotations

import contextlib
import errno
import json
import math
import os
import secrets
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from jobs import convert_raw as CR
from jobs import predict_local as PL
from jobs import predict_set as PS
from portal import convert_settings as CS
from portal import predict_sources as PSRC

from portal.predict_parts.common import (CONVERT_INPUT, CONVERT_JSON, CONVERT_LOCK, CONVERT_LOG, CONVERT_OUT,
                                         CONVERT_POLL_S, CONVERT_TILES, DISK_FACTOR, KEEP_DEFAULT_DAYS, KEEP_SETTING,
                                         MAX_CONVERT_RUNNING, MAX_CONV_PER_USER, PID_START_SLACK_S, PORTAL_TMP,
                                         RAW_ID_RE, RAW_TILES, REPO, Refusal, SCHEMA_CONVERT, _parse_local, _plain,
                                         _read_json, _setting, now_iso, when)
from portal.predict_parts.raw import (check_gpu, image_ref_problem, processing_repo, raw_fields, raw_montages,
                                      sidecar_images, sidecar_problems, test_hooks)
from portal.predict_parts.disk import (_RES_LOCK, _existing)
from portal.predict_parts import common as COMMON
from portal.predict_parts import disk as DISK


# ---------------------------------------------------------------- raw conversion: records and the runner
# Round 7 (BUILD_CONTRACTS.md section 13): a conversion also writes acquisition.json; its montages/ come from a
# "stitch" job (convert_raw --stitch-only) queued when it is done, run by this runner one job at a time with the
# conversions; the admin can queue one again while the raw files are there. Finished upload conversions are
# registered as sessions (portal.upload_sessions, when it is on) by the watcher thread, outside the runner lock.
CONVERT_MONTAGES, CONVERT_ACQ, STITCH_OUT = "montages", "acquisition.json", "stitch.out"
LEGACY_LIST_SETTING = "CJ_RAW_LEGACY_LIST"             # off: <predict root>/raw_tiles not listed (preview portals)
MIN_FREE_SETTING, MIN_FREE_DEFAULT_GB = "CJ_RAW_MIN_FREE_GB", 25   # below: oldest ended raw inputs removed early
UPLOAD_SESSIONS_SETTING = "CJ_UPLOAD_SESSIONS"         # on: register finished upload conversions (live only)
REGISTER_BACKOFF_MAX_S = 300                           # a busy registry: tried again after 2, 4, 8 ... s, at most this


def raw_base(cfg) -> Path:
    """The folder holding the whole raw pipeline (.portal staging of raw uploads + raw_tiles/): cfg.raw_root, else
    CJ_RAW_ROOT, else the predict root (today's behaviour)."""
    v = getattr(cfg, "raw_root", None) or os.environ.get("CJ_RAW_ROOT")
    return Path(v) if v else COMMON.predict_root(cfg)


def raw_root(cfg) -> Path:
    return raw_base(cfg) / RAW_TILES


def _same_path(a: Path, b: Path) -> bool:
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def _off(v) -> bool:
    return str(v if v is not None else "").strip().lower() in ("off", "0", "no", "false")


def _on(v) -> bool:
    return str(v if v is not None else "").strip().lower() in ("on", "1", "yes", "true")


def raw_roots(cfg) -> list:
    """Where conversions are listed from: the raw root, then (when CJ_RAW_ROOT moved it) the old
    <predict root>/raw_tiles, whose conversions stay listed, swept and usable -- unless CJ_RAW_LEGACY_LIST=off (a
    preview portal with its own CJ_RAW_ROOT never lists, sweeps or converts the live raw_tiles)."""
    new, old = raw_root(cfg), COMMON.predict_root(cfg) / RAW_TILES
    if _same_path(new, old) or _off(_setting(cfg, LEGACY_LIST_SETTING)):
        return [new]
    return [new, old]


def tmp_dirs(cfg) -> list:
    """The .portal scratch folders: the predict root's, and the raw root's when that is somewhere else."""
    a, b = COMMON.predict_root(cfg) / PORTAL_TMP, raw_base(cfg) / PORTAL_TMP
    return [a] if _same_path(a, b) else [a, b]


def conv_dir(cfg, cid) -> Path | None:
    if not isinstance(cid, str) or not RAW_ID_RE.match(cid):
        return None
    for root in raw_roots(cfg):
        d = root / cid
        if (d / CONVERT_JSON).is_file():
            return d
    return None


def load_conversion(cfg, user, can_see, cid) -> tuple:
    """-> (folder, convert.json, status) when it exists AND this person may see it; else (None, None, None)."""
    d = conv_dir(cfg, cid)
    meta = _read_json(d / CONVERT_JSON) if d else None
    if not isinstance(meta, dict) or not can_see(user, meta.get("owner")):
        return None, None, None
    return d, meta, _read_json(d / PS.STATUS) or {}


def all_conversions(cfg) -> list:
    """[(folder, convert.json, status)] of every conversion, oldest first."""
    dirs = []
    for root in raw_roots(cfg):
        try:
            dirs += [p for p in root.iterdir() if p.is_dir() and RAW_ID_RE.match(p.name)]
        except OSError:
            continue
    out = []
    for d in dirs:
        meta = _read_json(d / CONVERT_JSON)
        if isinstance(meta, dict):
            out.append((d, meta, _read_json(d / PS.STATUS) or {}))
    out.sort(key=lambda t: (str(t[1].get("created") or ""), t[0].name))
    return out


def list_conversions(cfg, user, can_see) -> list:
    """This person's conversions (admin: all), newest first."""
    return [t for t in reversed(all_conversions(cfg)) if can_see(user, t[1].get("owner"))]


def write_conv_status(d: Path, **fields) -> dict:
    """Merge fields into the conversion's status.json (atomic). Callers hold the runner lock, or own the folder
    (just made)."""
    st = _read_json(d / PS.STATUS) or {}
    st.update(fields, updated=now_iso())
    PS.write_json_atomic(d / PS.STATUS, st)
    return st


def keep_days(cfg) -> float:
    """CJ_RAW_KEEP_DAYS: days an uploaded conversion keeps its raw input/ after it ends (done or failed); 0 = removed
    as soon as it ends. Default KEEP_DEFAULT_DAYS; an unreadable value = the default."""
    v = _setting(cfg, KEEP_SETTING)
    try:
        x = float(v) if v not in (None, "") else KEEP_DEFAULT_DAYS
    except (TypeError, ValueError):
        return float(KEEP_DEFAULT_DAYS)
    return max(0.0, x) if math.isfinite(x) else float(KEEP_DEFAULT_DAYS)


def input_bytes(meta: dict) -> int:
    """The raw bytes a conversion reads: recorded at creation, else estimated from its sidecars (the MRC beside an
    .mrc.mdoc; for an .idoc, the tiles it names that are plain files beside it)."""
    if isinstance(meta.get("input_bytes"), int):
        return meta["input_bytes"]
    total = 0
    for s in meta.get("inputs") or []:
        side = Path(s)
        try:
            total += side.stat().st_size
            if side.name.lower().endswith(".mrc.mdoc"):
                mrc = side.with_name(side.name[:-len(".mdoc")])
                total += mrc.stat().st_size if mrc.is_file() else 0
            else:
                for n in sidecar_images(side):
                    if image_ref_problem(n) is None and (side.parent / n).is_file():
                        total += (side.parent / n).stat().st_size
        except OSError:
            continue
    return total


def check_disk_for(where: Path, nbytes: int, factor=DISK_FACTOR, what="to convert these",
                   tail="Ask the admin."):
    """A conversion writes its work copies and tiles next to the predict root: refuse (507) unless about
    `factor` x its input is free on top of FREE_MARGIN_BYTES and the uploads in flight (a set made from copies:
    DISK_NEED_FACTOR)."""
    free = DISK.disk_free(_existing(where))
    dev = DISK._dev(where)
    with _RES_LOCK:
        held = DISK._held(dev)
    need = int(factor * int(nbytes or 0)) + COMMON.FREE_MARGIN_BYTES
    if free - held < need:
        raise Refusal(f"Not enough disk space on the server {what} (about "
                      f"{factor * int(nbytes or 0) / 2 ** 30:.1f} GB needed). {tail}", 507,
                      tech=f"{free} bytes free, {held} held by uploads, need {need}")


def check_conv_cap(cfg, user):
    """At most MAX_CONV_PER_USER conversions of one person queued or running."""
    mine = [1 for _, meta, st in all_conversions(cfg)
            if meta.get("owner") == user.get("id") and st.get("state") in ("queued", "running")]
    if len(mine) >= MAX_CONV_PER_USER:
        raise Refusal(f"You already have {MAX_CONV_PER_USER} conversions waiting or running. Start another when "
                      f"one of them is done.", 429)


def move_tree(src: Path, dst: Path):
    """Move a folder: os.replace on one drive; across drives (EXDEV, e.g. an upload staged before CJ_RAW_ROOT was set)
    copy, verify every file's size, then remove the source. A failure leaves the source whole and no dst."""
    try:
        os.replace(src, dst)
        return
    except OSError as e:
        if e.errno != errno.EXDEV and not (os.name == "nt" and getattr(e, "winerror", None) in (17, 5)):
            raise
        if Path(dst).exists() and any(Path(dst).iterdir()):
            raise
    try:
        shutil.copytree(src, dst, dirs_exist_ok=True)
        for a in Path(src).rglob("*"):
            if a.is_file() and (Path(dst) / a.relative_to(src)).stat().st_size != a.stat().st_size:
                raise OSError(f"copy of {a.name} has a different size")
    except BaseException:
        shutil.rmtree(dst, ignore_errors=True)
        raise
    shutil.rmtree(src, ignore_errors=True)


def new_conversion(cfg, user, fields: dict, *, source: dict, inputs=None, upload_files: Path | None = None,
                   sample_key=None, nbytes=None, skipped=None, params=None, derived_from=None, owner=None,
                   made_by=None) -> Path:
    """Make <root>/raw_tiles/<session>_<6 hex>/ with convert.json + status queued (uploaded files moved into
    input/), then start it when nothing else runs. -> the folder."""
    root = raw_root(cfg)
    root.mkdir(parents=True, exist_ok=True)
    for _ in range(50):
        name = f"{fields['session']}_{secrets.token_hex(3)}"
        if any((r / name).exists() for r in raw_roots(cfg)):      # free in BOTH roots
            continue
        d = root / name
        try:
            d.mkdir()
            break
        except FileExistsError:
            continue
    else:
        raise Refusal("Could not make a folder for the conversion. Try again.", 500)
    try:
        if upload_files is not None:
            inp = d / CONVERT_INPUT
            # same drive: staging (.portal) and raw_tiles/ are both under the raw root; an upload staged under the
            # predict root before CJ_RAW_ROOT was set is copied across (move_tree), never lost
            move_tree(upload_files, inp)
            names = sorted(p.name for p in inp.iterdir() if p.is_file())
            inputs = [str(inp / n) for n in names if n.lower().endswith((".mrc.mdoc", ".idoc"))]
            keys = [n for n in names if n.lower().endswith(".csv")]
            sample_key = str(inp / keys[0]) if keys else None
        meta = {"schema": SCHEMA_CONVERT, "id": d.name, "owner": owner or user["id"], "session": fields["session"],
                "recipe": fields["recipe"], "gpu": bool(fields["gpu"]), "source": source,
                "inputs": [str(x) for x in inputs or []], "sample_key": str(sample_key) if sample_key else None,
                "created": now_iso()}
        if params:                                       # settings overrides (convert_raw --params)
            meta["params"] = CS.clean_params(params)
        if derived_from:                                 # a re-conversion of another one (made_by: who asked, if not the owner)
            meta["derived_from"] = str(derived_from)
            if made_by and made_by != meta["owner"]:
                meta["made_by"] = str(made_by)
        meta["input_bytes"] = int(nbytes) if nbytes is not None else input_bytes(meta)
        if skipped:                                      # upload files left out: [{name, why}] (the page lists them)
            meta["skipped"] = [{"name": str(x.get("name"))[:200], "why": str(x.get("why"))[:300]} for x in skipped]
        PS.write_json_atomic(d / CONVERT_JSON, meta)
        write_conv_status(d, state="queued", pid=None, pid_created=None, started=None, finished=None, error=None,
                          kind=None, warnings=[], result=None, n_tiles=None)
    except BaseException:
        shutil.rmtree(d, ignore_errors=True)
        raise
    kick(cfg)
    ensure_watcher(cfg)
    return d


def _converter_cmd() -> list:
    hook = test_hooks().get("convert_cmd")
    return [str(x) for x in hook] if isinstance(hook, list) and hook else [sys.executable, "-m", "jobs.convert_raw"]


def convert_argv(d: Path, meta: dict, repo: Path) -> list:
    """`python -m jobs.convert_raw --in ... --out <d>/tiles --recipe <r> [--sample-key k] [--gpu]
    --acquisition-out <d>/acquisition.json --processing-repo <repo>` (tests: the hooks' convert_cmd in place of
    `python -m jobs.convert_raw`). The montages are a separate stitch job, queued when this one is done: a stitch
    that runs long, crashes or is stopped never fails the conversion."""
    argv = _converter_cmd() + ["--in", *meta["inputs"], "--out", str(d / CONVERT_TILES), "--recipe",
                               str(meta["recipe"])]
    if meta.get("sample_key"):
        argv += ["--sample-key", str(meta["sample_key"])]
    if meta.get("gpu"):
        argv.append("--gpu")
    if meta.get("params"):                      # settings overrides of a re-conversion (already validated: CS.clean_params)
        argv += ["--params", CS.params_json(CS.clean_params(meta["params"]))]
    return argv + ["--acquisition-out", str(d / CONVERT_ACQ), "--processing-repo", str(repo)]


def stitch_argv(d: Path, repo: Path) -> list:
    """`python -m jobs.convert_raw --stitch-only <d> --processing-repo <repo>`: the backfill job."""
    return _converter_cmd() + ["--stitch-only", str(d), "--processing-repo", str(repo)]


# -- the converter's process: identity by pid AND start time (a pid can be reused by another program)
def proc_created(pid) -> float | None:
    """The start time (epoch s) of process pid, or None when it can't be read (no psutil, gone, no access)."""
    try:
        import psutil
        return float(psutil.Process(int(pid)).create_time())
    except Exception:
        return None


def proc_alive(st: dict) -> bool:
    """The converter recorded in status.json is still running: its pid is alive AND that process started when the
    converter did (pid_created; else within PID_START_SLACK_S of `started`). A reused pid is not it."""
    pid = st.get("pid")
    if not pid or not PL.pid_alive(pid):
        return False
    t = proc_created(pid)
    if t is None:
        return True                                     # can't tell: trust the pid (the max runtime still applies)
    if st.get("pid_created") is not None:
        try:
            return abs(t - float(st["pid_created"])) <= 2.0
        except (TypeError, ValueError):
            pass
    started = _parse_local(st.get("started"))
    return started is None or abs(t - started) <= PID_START_SLACK_S


def kill_proc(st: dict) -> bool:
    """Kill the converter (and anything it started) when proc_alive(st). -> killed."""
    if not proc_alive(st):
        return False
    pid = int(st["pid"])
    try:
        import psutil
        p = psutil.Process(pid)
        for c in p.children(recursive=True):
            try:
                c.kill()
            except psutil.Error:
                pass
        p.kill()
        p.wait(10)
        return True
    except ImportError:
        pass
    except Exception:
        return not PL.pid_alive(pid)
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, timeout=30,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        else:
            os.kill(pid, 9)
    except (OSError, subprocess.SubprocessError):
        pass
    return not PL.pid_alive(pid)


def overdue(d: Path, st: dict, now=None) -> str | None:
    """Why a running conversion should be stopped: over CONVERT_MAX_S since it started, or its log silent for
    CONVERT_STALE_S. None while it is fine."""
    now = time.time() if now is None else now
    started = _parse_local(st.get("started"))
    if started is None:
        return None
    if now - started > COMMON.CONVERT_MAX_S:
        return f"It ran for over {COMMON.CONVERT_MAX_S / 3600:g} h, so it was stopped."
    try:
        last = max(started, (d / CONVERT_LOG).stat().st_mtime)
    except OSError:
        last = started
    if now - last > COMMON.CONVERT_STALE_S:
        return f"No progress for {COMMON.CONVERT_STALE_S // 60} min (its log stopped), so it was stopped."
    return None


def start_conversion(cfg, d: Path, meta: dict) -> dict:
    """Spawn the converter detached (no console window; it outlives the portal): stdout -> convert.out, stderr ->
    convert.log. -> the new status. Refused (failed) without a processing repo, a montage or the disk it needs."""
    repo, why = processing_repo(cfg)
    if repo is None:
        return write_conv_status(d, state="failed", kind="refused", finished=now_iso(),
                                 error=f"Raw conversion is not set up: {why}")
    if not meta.get("inputs"):
        return write_conv_status(d, state="failed", kind="refused", finished=now_iso(), error="No montage to convert.")
    try:
        check_disk_for(d, input_bytes(meta))
    except Refusal as e:
        return write_conv_status(d, state="failed", kind="refused", finished=now_iso(), error=e.msg)
    try:
        p = spawn_converter(convert_argv(d, meta, repo), repo, d / CONVERT_OUT, d / CONVERT_LOG)
    except CS.ParamError as e:
        return write_conv_status(d, state="failed", kind="refused", finished=now_iso(),
                                 error=f"The settings of this conversion are not valid: {e}")
    except OSError as e:
        return write_conv_status(d, state="failed", kind="failed", finished=now_iso(),
                                 error=f"could not start the converter: {type(e).__name__}: {e}")
    return write_conv_status(d, state="running", pid=p.pid, pid_created=proc_created(p.pid), started=now_iso())


def spawn_converter(argv: list, repo: Path, out: Path, log: Path):
    """Popen detached (no console window; it outlives the portal): stdout -> out, stderr appended to log."""
    kw = {}
    if os.name == "nt":
        kw["creationflags"] = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kw["start_new_session"] = True
    env = {**os.environ, CR.REPO_ENV: str(repo), "PYTHONIOENCODING": "utf-8"}
    with open(out, "wb") as fo, open(log, "ab") as fl:
        return subprocess.Popen(argv, cwd=REPO, stdin=subprocess.DEVNULL, stdout=fo, stderr=fl, env=env,
                                close_fds=True, **kw)


def converter_result(d: Path, name=CONVERT_OUT):
    """The converter's one JSON object from convert.out (or stitch.out), or None (not finished, nothing printed)."""
    try:
        text = (d / name).read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                doc = json.loads(line)
            except ValueError:
                return None
            return doc if isinstance(doc, dict) and "ok" in doc else None
    return None


def log_tail(d: Path, n=4000) -> str:
    try:
        with open(d / CONVERT_LOG, "rb") as f:
            f.seek(max(0, os.fstat(f.fileno()).st_size - n))
            return f.read().decode("utf-8", "replace")
    except OSError:
        return ""


def finalize(d: Path, meta: dict, st: dict) -> dict:
    """A running conversion: its result once printed (done / failed); failed when its process is gone without one;
    stopped (killed, failed "timeout") when overdue(); else unchanged."""
    res = converter_result(d)
    if res is None:
        if proc_alive(st):
            why = overdue(d, st)
            if why is None:
                return st
            kill_proc(st)
            return write_conv_status(d, state="failed", kind="timeout", pid=None, finished=now_iso(), error=why)
        res = converter_result(d)                       # it may have printed just before it ended
        if res is None:
            return write_conv_status(d, state="failed", kind="failed", pid=None, finished=now_iso(),
                                     error="The converter stopped without a result.")
    warnings = [str(w) for w in (res.get("warnings") or [])]
    if res.get("ok"):
        extra = {"registered": False} if is_upload(meta) else {}
        return write_conv_status(d, state="done", pid=None, finished=now_iso(), error=None, kind=None,
                                 warnings=warnings, result=res, n_tiles=res.get("n_tiles"),
                                 stitch=queued_stitch("convert"), **extra)
    return write_conv_status(d, state="failed", pid=None, finished=now_iso(), kind=res.get("kind") or "failed",
                             error=str(res.get("error") or "unknown error"), warnings=warnings, result=res)


def is_upload(meta: dict) -> bool:
    return (meta.get("source") or {}).get("kind") == "upload"


# -- montages: the conversion's own stitch pass, or a "stitch" job (convert_raw --stitch-only) through this runner
def stitch_state(st: dict) -> str | None:
    """status.json stitch.state: queued | running (a stitch job) | done | failed | skipped; None = never stitched."""
    s = st.get("stitch")
    return s.get("state") if isinstance(s, dict) else None


def stitch_busy(st: dict) -> bool:
    return stitch_state(st) in ("queued", "running")


def queued_stitch(job: str) -> dict:
    """A stitch job waiting for the runner: job "convert" (queued when the conversion is done) or "backfill"."""
    return {"state": "queued", "job": job, "queued": now_iso(), "pid": None, "started": None, "finished": None,
            "error": None, "kind": None}


def montages_done(d: Path) -> bool:
    """montages/ holds a finished set (stitch.json done; convert_raw never replaces one with a failed pass)."""
    doc = _read_json(d / CONVERT_MONTAGES / CR.STITCH_JSON)
    return isinstance(doc, dict) and doc.get("state") == "done"


def stitch_summary(d: Path, job: str, doc=None) -> dict:
    """status.json's stitch record from the stitcher's result (its stitch doc), else montages/stitch.json."""
    if not isinstance(doc, dict):
        doc = _read_json(d / CONVERT_MONTAGES / CR.STITCH_JSON)
    if not isinstance(doc, dict) or doc.get("state") not in ("done", "failed", "skipped"):
        return {"state": "skipped", "error": "The stitcher made no montages.", "n": 0, "job": job}
    return {"state": doc["state"], "error": doc.get("error"), "n": doc.get("n"), "job": job,
            "kept_previous": bool(doc.get("kept_previous"))}


def inputs_present(meta: dict, d: Path) -> bool:
    """The raw files a conversion read are all still there (an upload's input/, or the picked session files)."""
    paths = [Path(p) for p in meta.get("inputs") or []]
    if is_upload(meta) and not (d / CONVERT_INPUT).is_dir():
        return False
    return bool(paths) and all(p.is_file() for p in paths)


def can_stitch(meta: dict, d: Path, st: dict) -> bool:
    return (st.get("state") == "done" and not stitch_busy(st) and (d / CONVERT_TILES / PSRC.MANIFEST).is_file()
            and inputs_present(meta, d))


def restore_montages(d: Path) -> bool:
    """A stitcher killed between its two renames leaves montages/ missing and .montages_old there: put it back.
    -> restored."""
    dest = d / CONVERT_MONTAGES
    _work, old = CR.stitch_dirs(dest)
    if dest.exists() or not old.is_dir():
        return False
    try:
        os.replace(old, dest)
    except OSError as e:
        print(f"[predict] {d.name}: could not put the previous montages back: {e}", file=sys.stderr, flush=True)
        return False
    return True


def _stitch_end(d: Path, st: dict, meta: dict, acquisition=False, **fields) -> dict:
    """Record a stitch job's end (fields: state, error, n, kind ...). An upload conversion is (re)registered when
    a backfill wrote its acquisition.json, or when it was never registered (made before round 7): registered false
    and the register_* bookkeeping reset, so the watcher registers it."""
    restore_montages(d)
    s = {**(st.get("stitch") or {}), "pid": None, "finished": now_iso(), **fields}
    extra = {}
    if is_upload(meta) and ("registered" not in st or (acquisition and s.get("job") == "backfill")):
        extra = {"registered": False, "register_next": 0, "register_since": None, "register_tries": 0,
                 "register_wait": None}
    return write_conv_status(d, stitch=s, **extra)


def start_stitch(cfg, d: Path, meta: dict, st: dict) -> dict:
    """Spawn `convert_raw --stitch-only <d>` (stdout -> stitch.out, stderr appended to convert.log). -> status."""
    repo, why = processing_repo(cfg)
    if repo is None:
        return _stitch_end(d, st, meta, state="failed", kind="refused", error=f"Raw conversion is not set up: {why}")
    if not inputs_present(meta, d):
        return _stitch_end(d, st, meta, state="failed", kind="refused",
                           error="The raw files are gone: the montages can't be made.")
    try:
        p = spawn_converter(stitch_argv(d, repo), repo, d / STITCH_OUT, d / CONVERT_LOG)
    except OSError as e:
        return _stitch_end(d, st, meta, state="failed", kind="failed",
                           error=f"could not start the stitcher: {type(e).__name__}: {e}")
    s = {**(st.get("stitch") or {}), "state": "running", "pid": p.pid, "pid_created": proc_created(p.pid),
         "started": now_iso(), "finished": None, "error": None, "kind": None}
    return write_conv_status(d, stitch=s)


def finalize_stitch(d: Path, meta: dict, st: dict) -> dict:
    """A running stitch job: recorded once it printed its result (stitch.json says done / failed), failed when its
    process is gone without one, stopped when overdue() (same limits as a conversion); else unchanged."""
    s = st.get("stitch") or {}
    res = converter_result(d, STITCH_OUT)
    if res is None:
        if proc_alive(s):
            why = overdue(d, s)
            if why is None:
                return st
            kill_proc(s)
            return _stitch_end(d, st, meta, state="failed", kind="timeout", error=why)
        res = converter_result(d, STITCH_OUT)
        if res is None:
            return _stitch_end(d, st, meta, state="failed", kind="failed",
                               error="The stitcher stopped without a result.")
    if res.get("ok"):
        got = stitch_summary(d, s.get("job") or "backfill", res.get("stitch"))
        return _stitch_end(d, st, meta, acquisition=bool(res.get("acquisition")), state=got["state"],
                           error=got["error"], n=got["n"], kept_previous=got["kept_previous"], kind=None,
                           warnings=[str(w) for w in res.get("warnings") or []])
    return _stitch_end(d, st, meta, state="failed", kind=res.get("kind") or "failed",
                       error=str(res.get("error") or "unknown error"))


def queue_stitch(cfg, d: Path, meta: dict) -> str:
    """Admin "Make the montages": a done conversion whose raw files are still there gets a stitch job, run by the
    runner when nothing else runs (never next to a conversion)."""
    with runner_lock(cfg):
        st = _read_json(d / PS.STATUS) or {}
        if not can_stitch(meta, d, st):
            raise Refusal("The montages can’t be made for this conversion (it is not done, a stitch is already "
                          "waiting or running, or its raw files were removed).", 409)
        write_conv_status(d, stitch=queued_stitch("backfill"))
    kick(cfg)
    ensure_watcher(cfg)
    return "queued"


# -- registration as a session (portal.upload_sessions, S), after the runner lock is released
def needs_registration(meta: dict, st: dict) -> bool:
    """A done upload conversion not registered yet that has not been given up on (register_next None = a refusal
    that says not to try again)."""
    return (st.get("state") == "done" and st.get("registered") is False and is_upload(meta)
            and st.get("register_next", 0) is not None)


def can_register_again(cfg, meta: dict, st: dict) -> bool:
    """A done upload not registered yet, with upload sessions on (also while waiting: it retries now, the backoff
    reset)."""
    return (is_upload(meta) and st.get("state") == "done" and st.get("registered") is not True
            and upload_sessions_module(cfg) is not None)


def register_again(cfg, d: Path, meta: dict) -> str:
    """Admin "Register again": a done upload conversion goes back to registered false with the backoff reset; the
    watcher registers it (idempotent: an unchanged record is not appended twice)."""
    with runner_lock(cfg):
        st = _read_json(d / PS.STATUS) or {}
        if not (is_upload(meta) and st.get("state") == "done"):
            raise Refusal("Only a finished upload can be added to Data → Sessions.", 409)
        if upload_sessions_module(cfg) is None:
            raise Refusal("Upload sessions are not switched on on this server.", 409)
        write_conv_status(d, registered=False, register_next=0, register_since=None, register_tries=0,
                          register_wait=None)
    kick(cfg)
    ensure_watcher(cfg)
    return "queued"


def registration_line(cfg, meta: dict, st: dict) -> str | None:
    """Where the upload's registration as a session stands, in a few words (None: not an upload, not done)."""
    if not is_upload(meta) or st.get("state") != "done":
        return None
    reg = st.get("registered")
    if reg is True:
        return "Listed under Data → Sessions."
    if isinstance(reg, str):
        return f"Not added to Data → Sessions ({reg})."
    if upload_sessions_module(cfg) is None:
        return "Not added to Data → Sessions (upload sessions are off on this server)."
    if reg is False and st.get("register_next", 0) is None:
        return f"Not added to Data → Sessions ({st.get('register_wait') or 'skipped'})."
    if reg is False:
        wait = st.get("register_wait")
        return ("Being added to Data → Sessions" + (f" ({wait}; tried {int(st.get('register_tries') or 0)} time(s) "
                                                     f"since {when(st.get('register_since'))})" if wait else "") + ".")
    return None


def registration_due(st: dict, now=None) -> bool:
    return (time.time() if now is None else now) >= float(st.get("register_next") or 0)


def upload_sessions_module(cfg):
    """portal.upload_sessions when it is installed AND switched on (its enabled(cfg) is the one predicate; without
    one, CJ_UPLOAD_SESSIONS on/1/yes/true); else None."""
    try:
        import importlib
        US = importlib.import_module("portal.upload_sessions")
    except ImportError:
        return None                                     # not installed here: registered stays false
    fn = getattr(US, "enabled", None)
    try:
        on = fn(cfg) if callable(fn) else _on(_setting(cfg, UPLOAD_SESSIONS_SETTING))
    except Exception:                                   # noqa: BLE001 - a broken flag reads as off
        on = False
    return US if on else None


def register_conversions(cfg, todo: list, US=None):
    """portal.upload_sessions.register_conversion for each due (folder, meta), from the watcher thread only and
    never holding the runner lock; the outcome is written under it:
      {"ok": true}                                    -> registered true
      {"ok": false, "retry": true} (busy, unreadable)  -> stays false; register_next = now + 2, 4, 8 ... (at most
                                                         REGISTER_BACKOFF_MAX_S) s, register_since = the first
                                                         try, register_wait = "waiting: registry busy" (or why)
      an exception                                     -> the same as retry
      {"ok": false, "error", "retry": false}           -> "excluded: <msg>" for an excluded session, else
                                                         "error: <msg>" (given up)
      {"ok": false, "skipped", "retry": false}         -> stays false, not tried again (register_next null)
    Off, or the module not there: nothing is called."""
    US = US or upload_sessions_module(cfg)
    if US is None:
        return
    for d, _meta in todo:
        try:
            out = US.register_conversion(d, cfg)
            out = out if isinstance(out, dict) else {}
        except Exception as e:                          # noqa: BLE001 - tried again with the backoff
            out = {"ok": False, "error": f"{type(e).__name__}: {e}", "retry": True}
        with runner_lock(cfg):
            st = _read_json(d / PS.STATUS) or {}
            now = time.time()
            msg = str(out.get("error") or out.get("skipped") or out.get("excluded") or "")[:400]
            if out.get("ok"):
                f = {"registered": True, "register_wait": None, "register_next": 0}
            elif out.get("retry"):
                tries = int(st.get("register_tries") or 0) + 1
                f = {"register_tries": tries, "register_since": st.get("register_since") or now_iso(),
                     "register_next": now + min(REGISTER_BACKOFF_MAX_S, 2 ** tries),
                     "register_wait": "waiting: registry busy" if out.get("busy") or "busy" in msg.lower()
                     else f"waiting: {msg or 'try again'}"}
                print(f"[predict] registering {d.name}: {msg} (again in {f['register_next'] - now:.0f} s)",
                      file=sys.stderr, flush=True)
            elif out.get("excluded") or "excluded" in msg.lower():
                f = {"registered": f"excluded: {msg}", "register_wait": None}
            elif out.get("error"):
                f = {"registered": f"error: {msg}", "register_wait": None}
            else:
                f = {"register_next": None, "register_wait": f"not registered: {msg}" if msg else None}
            write_conv_status(d, **f)


def inputs_kept_until(cfg, d: Path, st: dict) -> float | None:
    """When an ended conversion's uploaded input/ goes (epoch s), or None (none kept, or not ended)."""
    if st.get("state") not in ("done", "failed") or not (d / CONVERT_INPUT).exists():
        return None
    fin = _parse_local(st.get("finished")) or time.time()
    return fin + keep_days(cfg) * 86400


def remove_inputs(d: Path) -> bool:
    """Delete a conversion's uploaded raw files (input/). Callers hold the runner lock."""
    if not (d / CONVERT_INPUT).exists():
        return False
    shutil.rmtree(d / CONVERT_INPUT, ignore_errors=True)
    write_conv_status(d, inputs_removed=now_iso())
    return True


def min_free_bytes(cfg) -> int:
    """CJ_RAW_MIN_FREE_GB (default MIN_FREE_DEFAULT_GB; 0 = no floor) in bytes; an unreadable value = the default."""
    v = _setting(cfg, MIN_FREE_SETTING)
    try:
        x = float(v) if v not in (None, "") else MIN_FREE_DEFAULT_GB
    except (TypeError, ValueError):
        x = MIN_FREE_DEFAULT_GB
    return int(max(0.0, x) * 1024 ** 3) if math.isfinite(x) else int(MIN_FREE_DEFAULT_GB * 1024 ** 3)


def inputs_linked(d: Path) -> bool:
    """True when a file of d/input is also another conversion's (hard link, st_nlink > 1): removing it frees no space."""
    try:
        return any(p.is_file() and p.stat().st_nlink > 1 for p in (d / CONVERT_INPUT).iterdir())
    except OSError:
        return False


def sweep_inputs(cfg, items) -> int:
    """Remove input/ of ended conversions past CJ_RAW_KEEP_DAYS (0: at once); then, while a drive holding kept
    inputs has less than CJ_RAW_MIN_FREE_GB free, the oldest-ended ones on it go early (logged). A conversion with a
    stitch job queued or running keeps its inputs; the floor also spares a done conversion whose montages were never
    made (its inputs are what a backfill needs). Callers hold the runner lock."""
    n, kept = 0, []
    for d, _meta, st in items:
        if stitch_busy(st):
            continue
        until = inputs_kept_until(cfg, d, st)
        if until is None:
            continue
        if time.time() >= until:
            n += remove_inputs(d)
        elif not (st.get("state") == "done" and not montages_done(d)):
            kept.append((until, d))
    floor = min_free_bytes(cfg)
    for until, d in sorted(kept, key=lambda t: (t[0], t[1].name)) if floor else ():
        try:
            free = DISK.disk_free(_existing(d))
        except OSError:
            continue
        if free >= floor or inputs_linked(d):          # a hard-linked input (a re-conversion's) frees nothing
            continue
        if remove_inputs(d):
            n += 1
            write_conv_status(d, inputs_removed_reason="low disk space")
            print(f"[predict] raw inputs of {d.name} removed early: {free / 2 ** 30:.1f} GB free < "
                  f"{MIN_FREE_SETTING}={floor / 2 ** 30:g} GB (they were kept until "
                  f"{datetime.fromtimestamp(until).isoformat(timespec='seconds')})", file=sys.stderr, flush=True)
    return n


@contextlib.contextmanager
def runner_lock(cfg):
    """The OS lock of every listed raw root that exists (the raw root is made if need be), taken in one sorted order,
    so two portals sharing a root (live and legacy) never run the queue at once."""
    roots = raw_roots(cfg)
    try:
        roots[0].mkdir(parents=True, exist_ok=True)
    except OSError:                                       # raw drive away: old conversions still need a lock
        pass
    held = [r for r in roots if r.is_dir()] or [roots[-1]]
    with contextlib.ExitStack() as stack:
        for r in sorted(held, key=lambda p: os.path.normcase(os.path.abspath(p))):
            stack.enter_context(PS.locked(r / CONVERT_LOCK))
        yield


def kick(cfg, register=False) -> int:
    """Record finished (or overdue) conversions and stitch jobs, start queued ones (conversions first, then stitch
    jobs; at most MAX_CONVERT_RUNNING of either at once) and sweep old inputs, under the runner lock; then, with the
    lock released and only when `register` (the watcher thread), register finished upload conversions that are due.
    -> how many are still queued or running, plus the registrations still waiting (so the watcher keeps going)."""
    if not any(r.is_dir() for r in raw_roots(cfg)):
        return 0
    with runner_lock(cfg):
        items = all_conversions(cfg)
        running = 0
        for i, (d, meta, st) in enumerate(items):
            if st.get("state") == "running":
                st = finalize(d, meta, st)
                running += st.get("state") == "running"
            if stitch_state(st) == "running":
                st = finalize_stitch(d, meta, st)
                running += stitch_state(st) == "running"
            items[i] = (d, meta, st)
        pending = running
        for i, (d, meta, st) in enumerate(items):
            if st.get("state") != "queued":
                continue
            if running < MAX_CONVERT_RUNNING:
                st = start_conversion(cfg, d, meta)
                items[i] = (d, meta, st)
                running += st.get("state") == "running"
                pending += st.get("state") == "running"
            else:
                pending += 1
        for i, (d, meta, st) in enumerate(items):
            if stitch_state(st) != "queued":
                continue
            if running < MAX_CONVERT_RUNNING:
                st = start_stitch(cfg, d, meta, st)
                items[i] = (d, meta, st)
                running += stitch_state(st) == "running"
                pending += stitch_state(st) == "running"
            else:
                pending += 1
        sweep_inputs(cfg, items)
        todo = [(d, meta, st) for d, meta, st in items if needs_registration(meta, st)]
    US = upload_sessions_module(cfg) if todo else None
    if US is None:
        return pending
    if register:
        register_conversions(cfg, [(d, meta) for d, meta, st in todo if registration_due(st)], US)
    return pending + len(todo)


# -- the admin's actions on one conversion
def stop_conversion(cfg, d: Path) -> str:
    """Admin "Stop": kill a running converter (or drop a queued one) and mark it failed ("stopped")."""
    with runner_lock(cfg):
        st = _read_json(d / PS.STATUS) or {}
        if st.get("state") == "running":
            kill_proc(st)
            write_conv_status(d, state="failed", kind="stopped", pid=None, finished=now_iso(),
                              error="Stopped by the admin.")
            return "stopped"
        if st.get("state") == "queued":
            write_conv_status(d, state="failed", kind="stopped", pid=None, finished=now_iso(),
                              error="Stopped by the admin before it started.")
            return "stopped"
        if stitch_busy(st):                             # a stitch job: stopped, the conversion stays done
            if stitch_state(st) == "running":
                kill_proc(st.get("stitch") or {})
            restore_montages(d)
            s = {**st["stitch"], "state": "failed", "kind": "stopped", "pid": None, "finished": now_iso(),
                 "error": "Stopped by the admin."}
            write_conv_status(d, stitch=s)
            return "stopped"
    return "not running"


def can_retry(meta: dict, d: Path, st: dict) -> bool:
    if st.get("state") != "failed":
        return False
    if (meta.get("source") or {}).get("kind") == "upload":
        return (d / CONVERT_INPUT).is_dir()
    return all(Path(p).is_file() for p in meta.get("inputs") or [])


def retry_conversion(cfg, d: Path, meta: dict) -> str:
    """Admin "Retry": a failed conversion whose inputs are still there goes back to the queue (its tiles/,
    montages/, acquisition.json and result cleared; the log is kept and appended to)."""
    with runner_lock(cfg):
        st = _read_json(d / PS.STATUS) or {}
        if not can_retry(meta, d, st) or stitch_busy(st):
            raise Refusal("This conversion can’t be tried again (it is not stopped, or its raw files were removed: "
                          "upload them again).", 409)
        for sub in (d / CONVERT_TILES, d / CONVERT_MONTAGES, *CR.stitch_dirs(d / CONVERT_MONTAGES)):
            shutil.rmtree(sub, ignore_errors=True)
        for name in (CONVERT_OUT, STITCH_OUT, CONVERT_ACQ):
            try:
                (d / name).unlink()
            except OSError:
                pass
        write_conv_status(d, state="queued", pid=None, pid_created=None, started=None, finished=None, error=None,
                          kind=None, warnings=[], result=None, n_tiles=None, retries=int(st.get("retries") or 0) + 1,
                          stitch=None)
    kick(cfg)
    ensure_watcher(cfg)
    return "queued"


def delete_inputs_now(cfg, d: Path) -> bool:
    """Admin "Delete raw files now" on an ended conversion."""
    with runner_lock(cfg):
        st = _read_json(d / PS.STATUS) or {}
        if st.get("state") in ("queued", "running"):
            raise Refusal("Stop the conversion first.", 409)
        if stitch_busy(st):
            raise Refusal("The montages are being made from these raw files: stop that first, or wait.", 409)
        return remove_inputs(d)


_WATCH: dict = {}                     # raw root -> {"dirty": bool}: one watcher thread per root
_WATCH_LOCK = threading.Lock()


def ensure_watcher(cfg):
    """A daemon thread that calls kick(register=True) every CONVERT_POLL_S while anything is queued or running or a
    session registration is waiting, then ends."""
    key = os.path.normcase(str(raw_root(cfg)))
    with _WATCH_LOCK:
        if key in _WATCH:
            _WATCH[key]["dirty"] = True
            return
        _WATCH[key] = {"dirty": True}

    def loop():
        while True:
            with _WATCH_LOCK:
                _WATCH[key]["dirty"] = False
            try:
                pending = kick(cfg, register=True)    # the only caller that registers sessions
            except Exception as e:                     # noqa: BLE001 - logged; the watcher keeps going
                print(f"[predict] conversion watcher: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
                pending = 1
            if not pending:
                with _WATCH_LOCK:
                    if not _WATCH[key]["dirty"]:
                        _WATCH.pop(key, None)
                        return
                continue
            time.sleep(CONVERT_POLL_S)

    threading.Thread(target=loop, name="cj-predict-convert-watch", daemon=True).start()


def session_convert(cfg, user, s: dict, form: dict) -> Path:
    """The admin's pick of montages in a raw session folder -> a conversion (inputs = the sidecars, read in place)."""
    if user.get("role") != "admin":
        raise Refusal("Picking a raw session is for the admin.", 403)
    fields = raw_fields(cfg, user, form, session=s["session"])
    listing = raw_montages(s)
    by_rel = {m["rel"]: m for m in listing["montages"]}
    picked = [r for r in form.get("montages") or [] if r in by_rel]
    if not picked:
        raise Refusal("Tick at least one montage.")
    if len(picked) != len(set(form.get("montages") or [])):
        raise Refusal("The session’s files changed since this page was opened. Check the montages again.")
    check_conv_cap(cfg, user)
    problems = []                         # a sidecar naming a path (absolute, UNC, ..): never handed to the converter
    for r in picked:
        try:
            problems += sidecar_problems(s["dir"] / r)
        except OSError as e:
            problems.append(f"{r}: can't be read ({type(e).__name__})")
    if problems:
        raise Refusal(f"Not converted ({len(problems)} problem(s)):\n" + "\n".join(problems[:12]))
    inputs = [str(s["dir"] / r) for r in picked]
    nbytes = input_bytes({"inputs": inputs})
    check_disk_for(raw_root(cfg), nbytes)
    check_gpu(fields)
    return new_conversion(cfg, user, fields, source={"kind": "session", "folder": s["folder"], "montages": picked},
                          inputs=inputs, sample_key=listing["sample_key"], nbytes=nbytes)


# ---------------------------------------------------------------- a conversion's status in plain words
CONV_TONE = {"queued": "wait", "running": "run", "done": "ok", "failed": "warn"}


def conv_status_view(meta: dict, st: dict, admin: bool, paths=()) -> dict:
    state = st.get("state")
    out = {"state": state, "tone": CONV_TONE.get(state, "wait"), "label": "Unknown", "detail": None, "tech": None,
           "final": state in ("done", "failed")}
    if state == "queued":
        out.update(label="Waiting to start", detail="Another conversion is running; this one starts after it.")
    elif state == "running":
        out.update(label="Converting", detail="About half a minute per montage on the CPU. This page updates by "
                                              "itself; you can close it.")
    elif state == "done":
        out.update(label="Done", detail=f"{int(st.get('n_tiles') or 0)} tiles made.")
    elif state == "failed":
        msg = _plain(str(st.get("error") or ""), paths)
        out.update(label="Stopped", detail=(f"Not converted: {msg}" if st.get("kind") == "refused" else
                                            f"Stopped: {msg}" if st.get("kind") in ("stopped", "timeout") else
                                            "The conversion failed. Ask the admin (they can see the details)."))
    if admin:
        bits = [f"{k} {st.get(k)}" for k in ("pid", "kind", "error", "started", "finished") if st.get(k)]
        out["tech"] = "; ".join(bits + [f"updated {st.get('updated')}"])
    return out


def conv_paths(d: Path, meta: dict) -> list:
    """Folders whose spelling is left out of what people read (they never typed them)."""
    out = [d / CONVERT_INPUT, d / CONVERT_TILES, d]
    for p in meta.get("inputs") or []:
        out.append(Path(p).parent)
    if meta.get("sample_key"):
        out.append(Path(meta["sample_key"]).parent)
    return list(dict.fromkeys(out))
