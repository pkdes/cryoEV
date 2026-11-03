#!/usr/bin/env python3
"""Run the oldest ready training job from a queue folder (on the GPU desktop; on the laptop for smoke tests).

    python jobs/run_job.py --queue-dir C:\\cj [--job <job id>] [--wait SECONDS] [--max-hours H] [--allow-big]

Picks the oldest <queue>/q/<job>/ that has READY and no <queue>/d/<job>/, then:
  1. checks: zip md5 (job.json and the .md5 sidecar), repo commit == JOB.json commit, Ultralytics 8.4.x,
     longest extracted path < 240 characters, free disk >= 10 GB + 3x the zip, GPU not busy; on a GPU < 8 GB
     (the laptop): --workers <= 2, and no y8m/l/x or imgsz > 640 unless --allow-big (with --max-hours).
     The laptop also refuses (exit 2, queue untouched) a network queue folder and running on battery;
  2. unzips into <queue>/work/<job>/ and runs training/run_experiments.py --single-run --no-notify with the
     bundle's args (CRYOEV_CRYOAI_ROOT = the work folder);
  3. writes <queue>/d/.tmp_<job>/ (run.json, runs/<run>/, status.json, train.log), renames it to d/<job>/,
     then writes DONE.json last. The status comes from run.json, never from the exit code.
A failed check or run ends `blocked` with a reason. One ntfy push at the end ($CRYOEV_NTFY_TOPIC).
Holds an OS lock on <queue>/runner.lock for the whole job (released if the process dies). Never writes the
registry, and never deletes anything outside <queue>/work/<job>/. Keeps the machine awake while a job runs.
Exit: 0 done or nothing to do, 1 blocked, 2 refused at start, 3 lock held by another runner.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "training"))
import common as C  # noqa: E402
sys.path.insert(0, str(REPO))
from data_utils.oslock import acquire_lock  # noqa: E402,F401  (run_job.acquire_lock in tests)

LAPTOP_GPU_MB = 8192            # below this: small train jobs only
BIG_SIZES = ("m", "l", "x")
MAX_LAPTOP_IMGSZ = 640
SMALL_MAX_WORKERS = 2           # small GPU (= the laptop, 15 GB RAM): more dataloader workers exhaust the page file
MIN_FREE_GB = 10                # free disk needed: this + 3x the zip (unzip, results, the pulled copy)
BUSY_MB = 256                   # a compute app using this much (or any python process) means the GPU is busy
HOOKS_ENV = "CRYOEV_JOBS_TEST_HOOKS"   # tests only: JSON {train_cmd, gpu: {total_mb, apps, used_mb}, ultralytics,
TESTING_ENV = "CRYOEV_JOBS_TESTING"    # commit, on_battery, free_gb}; read only when this is "1" (jobs/tests set both)


class Blocked(Exception):
    pass


def hooks() -> dict:
    if os.environ.get(TESTING_ENV) != "1":
        return {}
    try:
        return json.loads(os.environ.get(HOOKS_ENV) or "{}")
    except ValueError:
        return {}


# ---------------------------------------------------------------- checks
def repo_commit():
    h = hooks().get("commit")
    if h:
        return h
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True,
                              text=True, timeout=10).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def same_commit(a, b) -> bool:
    """Short and long shas of one commit match (prefix, at least 7 characters)."""
    if not a or not b or a == "unknown" or b == "unknown":
        return False
    n = min(len(a), len(b))
    return n >= 7 and a[:n].lower() == b[:n].lower()


def ultralytics_version():
    h = hooks().get("ultralytics")
    if h:
        return h
    try:
        from importlib.metadata import version
        return version("ultralytics")
    except Exception:
        return None


def gpu_state() -> dict:
    """{total_mb, busy: [(pid, name, used_mb)]}. Busy = a compute app using >= 256 MB or named python*.
    Graphics apps listed with [N/A] memory under WDDM (Edge, Claude, ...) are not busy."""
    h = hooks().get("gpu")
    if h is not None:
        return {"total_mb": h.get("total_mb", 0), "used_mb": h.get("used_mb", 0),
                "busy": [tuple(a) for a in h.get("apps", [])]}
    from monitor_run import get_gpu_info
    info = get_gpu_info()
    mb = {k: int(v[:-2]) if v.endswith("MB") and v[:-2].isdigit() else 0
          for k, v in ((k, info.get(k, "N/A")) for k in ("mem_total", "mem_used"))}
    total_mb = mb["mem_total"]
    busy = []
    try:
        r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
        for line in r.stdout.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3 or not parts[0].isdigit() or int(parts[0]) == os.getpid():
                continue
            used = int(parts[2]) if parts[2].isdigit() else 0
            if used >= BUSY_MB or Path(parts[1]).name.lower().startswith("python"):
                busy.append((int(parts[0]), parts[1], used))
    except (OSError, subprocess.SubprocessError):
        pass
    return {"total_mb": total_mb, "used_mb": mb["mem_used"], "busy": busy}


def is_network_path(p) -> bool:
    """A UNC path or a mapped network drive."""
    s = str(p)
    if s.startswith(("\\\\", "//")):
        return True
    if os.name == "nt" and len(s) >= 2 and s[1] == ":":
        import ctypes
        return ctypes.windll.kernel32.GetDriveTypeW(s[:2] + "\\") == 4      # DRIVE_REMOTE
    return False


def on_battery() -> bool:
    h = hooks().get("on_battery")
    if h is not None:
        return bool(h)
    if os.name != "nt":
        return False
    import ctypes

    class PowerStatus(ctypes.Structure):
        _fields_ = [("ac", ctypes.c_ubyte), ("flag", ctypes.c_ubyte), ("pct", ctypes.c_ubyte),
                    ("saver", ctypes.c_ubyte), ("life", ctypes.c_ulong), ("full", ctypes.c_ulong)]
    s = PowerStatus()
    return bool(ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(s))) and s.ac == 0   # 0 offline, 255 unknown


def free_gb(p) -> float:
    h = hooks().get("free_gb")
    return float(h) if h is not None else shutil.disk_usage(p).free / 1e9


def keep_awake(on: bool):
    """No system sleep while a job runs (the display may still turn off)."""
    if os.name == "nt":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | (0x00000001 if on else 0))   # CONTINUOUS|SYSTEM


def best_config_defaults() -> dict:
    """run_experiments.BEST_CONFIG, read without importing it (no torch)."""
    tree = ast.parse((REPO / "training" / "run_experiments.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "BEST_CONFIG" for t in node.targets):
            return ast.literal_eval(node.value)
    return {"model_size": "n", "imgsz": 1440}


def arg_value(args, flag, default=None):
    return args[args.index(flag) + 1] if flag in args and args.index(flag) + 1 < len(args) else default


def size_and_imgsz(args):
    d = best_config_defaults()
    return arg_value(args, "--model-size", d["model_size"]), int(arg_value(args, "--imgsz", d["imgsz"]))


def bundle_job_json(z: zipfile.ZipFile):
    names = [n for n in z.namelist() if n.count("/") == 1 and n.endswith("/JOB.json")]
    if len(names) != 1:
        raise Blocked("bundle has no <dataset>/JOB.json")
    return json.loads(z.read(names[0]))


def check_job(qdir: Path, work: Path, st: dict, allow_big=False) -> tuple:
    """All pre-run checks. -> (spec, JOB.json, zip path, zip member names). Raises Blocked.
    Records bundle_id / dataset / commit in `st` as soon as JOB.json is read."""
    try:
        spec = C.read_json(qdir / C.JOB_FILE)
        zp = qdir / spec["zip"]
    except (OSError, ValueError, KeyError) as e:
        raise Blocked(f"job.json missing or unreadable: {e}")
    if not zp.is_file():
        raise Blocked(f"bundle zip {spec['zip']} missing")
    got = C.md5_file(zp)
    side = C.sidecar_md5(zp)
    if got != str(spec.get("zip_md5", "")).lower() or (side and side != got):
        raise Blocked(f"md5 mismatch: zip {got}, job.json {spec.get('zip_md5')}, sidecar {side}")
    try:
        with zipfile.ZipFile(zp) as z:
            job = bundle_job_json(z)
            names = [n for n in z.namelist() if not n.endswith("/")]
    except zipfile.BadZipFile as e:
        raise Blocked(f"bad zip: {e}")
    st.update(bundle_id=job.get("bundle_id") or spec.get("bundle_id"), dataset=job.get("dataset_name"),
              cryoEV_commit=job.get("cryoEV_commit"))
    commit = repo_commit()
    if not same_commit(job.get("cryoEV_commit"), commit):
        raise Blocked(f"commit mismatch: bundle {job.get('cryoEV_commit')}, repo {commit} "
                      f"(check out the bundle's commit in the runner's repo)")
    ul = ultralytics_version()
    if not (ul or "").startswith("8.4."):
        raise Blocked(f"Ultralytics {ul} (need 8.4.x)")
    base = work / "training outputs"
    longest = max(names, key=lambda n: len(str(base / n)))
    n_long = len(str(base / longest))
    if n_long >= C.MAX_PATH:
        raise Blocked(f"path too long: {n_long} characters (limit {C.MAX_PATH}) for {longest}; "
                      f"use a shorter --queue-dir (e.g. C:\\cj) or a shorter dataset/image name")
    args = job.get("run_experiments_args") or []
    if "--queue" in args:
        raise Blocked("only --single-run jobs are run here")
    gpu = gpu_state()
    st["gpu_used_mb_at_start"] = gpu.get("used_mb")
    size, imgsz = size_and_imgsz(args)
    if gpu["total_mb"] < LAPTOP_GPU_MB:
        if size in BIG_SIZES or imgsz > MAX_LAPTOP_IMGSZ:
            if not allow_big:
                raise Blocked(f"desktop only: y8{size} at imgsz {imgsz} on a {gpu['total_mb'] / 1024:.0f} GB GPU "
                              f"(here: y8n/s at imgsz <= {MAX_LAPTOP_IMGSZ}, or run_job.py --allow-big)")
            st["allow_big"] = True                     # main() made --allow-big come with --max-hours
        w = arg_value(args, "--workers")
        if w is None or not str(w).isdigit() or int(w) > SMALL_MAX_WORKERS:
            raise Blocked(f"--workers {w or '8 (default)'} on a {gpu['total_mb'] / 1024:.0f} GB GPU machine: "
                          f"use --workers <= {SMALL_MAX_WORKERS} (page file)")
    need = MIN_FREE_GB + 3 * zp.stat().st_size / 1e9
    have = free_gb(work.parent if work.parent.exists() else qdir)
    if have < need:
        raise Blocked(f"disk: {have:.1f} GB free, need {need:.1f} GB")
    if gpu["busy"]:
        raise Blocked("GPU busy: " + ", ".join(f"pid {p} {Path(n).name} {m} MB" for p, n, m in gpu["busy"][:3]))
    return spec, job, zp, names


def train_argv(job: dict, work: Path) -> list:
    """run_experiments argv from the bundle: --single-run --no-notify; a packed --weights re-pointed into work/."""
    args = list(job.get("run_experiments_args") or [])
    if "--weights" in args:
        i = args.index("--weights") + 1
        packed = "../CryoAI/training outputs/"
        if i < len(args) and args[i].startswith(packed):
            args[i] = str(work / "training outputs" / args[i][len(packed):])
    for flag in ("--single-run", "--no-notify"):
        if flag not in args:
            args.append(flag)
    cmd = hooks().get("train_cmd") or [sys.executable, "-u", str(REPO / "training" / "run_experiments.py")]
    return [*cmd, *args]


# ---------------------------------------------------------------- one job
def pick(root: Path, only=None):
    """Oldest q/<job>/ with READY and no d/<job>/ (READY mtime, then name)."""
    qroot = root / C.Q
    if not qroot.is_dir():
        return None
    ready = [d for d in qroot.iterdir() if d.is_dir() and not d.name.startswith(".")
             and (d / C.READY).is_file() and not (root / C.D / d.name).exists() and (only in (None, d.name))]
    return min(ready, key=lambda d: ((d / C.READY).stat().st_mtime, d.name)) if ready else None


def rename_retry(src: Path, dst: Path, tries=6):
    """Path.rename, retried with backoff: antivirus, the indexer or an open Explorer window can hold a folder
    for a moment on Windows. Gives up with SystemExit (results stay in `src`), never deletes."""
    for i in range(tries):
        try:
            return src.rename(dst)
        except OSError as e:
            if i == tries - 1:
                raise SystemExit(f"[jobs] could not rename {src} -> {dst}: {e}. Results are kept in {src}; "
                                 f"rename it by hand (see docs/JOBS.md, recovery).")
            time.sleep(2 ** i)


def kill_tree(p):
    """The trainer and its dataloader workers."""
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"], capture_output=True)
    else:
        p.kill()


def run_one(root: Path, qdir: Path, allow_big=False, max_hours=None) -> dict:
    job_id = qdir.name
    work, dd = root / C.WORK / job_id, root / C.D
    tmp, final = dd / f".tmp_{job_id}", dd / job_id
    dd.mkdir(parents=True, exist_ok=True)
    if tmp.exists():                                   # a killed earlier attempt: set aside, never deleted
        stale = dd / f".tmp_{job_id}.stale_{time.strftime('%Y%m%d-%H%M%S')}"
        rename_retry(tmp, stale)
        print(f"[jobs] stale {tmp.name} renamed to {stale.name}")
    tmp.mkdir()
    st = {"schema": "job_status/1", "job": job_id, "status": "running", "machine": platform.node(),
          "started": C.now(), "queue_dir": str(root), "max_hours": max_hours}
    C.write_json(tmp / C.STATUS, st)
    log = work / "train.log"
    runs = []
    try:
        spec, job, zp, names = check_job(qdir, work, st, allow_big)   # GPU-busy first: an orphan may own work/
        if work.exists():                              # our own scratch from an earlier attempt
            shutil.rmtree(work)
        work.mkdir(parents=True)
        print(f"[jobs] {job_id}: checks passed; unzipping {zp.name}")
        with zipfile.ZipFile(zp) as z:
            z.extractall(work / "training outputs")
        got = sum(1 for p in (work / "training outputs").rglob("*") if p.is_file())
        if got != len(names):
            raise Blocked(f"unzip wrote {got} of {len(names)} files (path length?)")
        argv = train_argv(job, work)
        st["argv"] = argv
        print(f"[jobs] {job_id}: training: {' '.join(argv[2:])}  (log {log})", flush=True)
        env = {**os.environ, "CRYOEV_CRYOAI_ROOT": str(work), "PYTHONIOENCODING": "utf-8"}
        with open(log, "wb") as lf:
            p = subprocess.Popen(argv, cwd=REPO, env=env, stdout=lf, stderr=subprocess.STDOUT)
            try:
                rc = p.wait(timeout=max_hours * 3600 if max_hours else None)
            except subprocess.TimeoutExpired:
                kill_tree(p)
                rc = p.wait()
                st["time_capped"] = True
        st["exit_code"] = rc
        run_root = work / "training outputs" / st["dataset"] / "runs"
        if st.get("time_capped"):                      # the trainer never wrote run.json: say why, keep last.pt
            for rd in (d for d in run_root.glob("*") if d.is_dir() and not (d / "run.json").exists()):
                C.write_json(rd / "run.json", {
                    "schema": "run/1", "id": rd.name, "dataset": st["dataset"], "status": "time_cap",
                    "error": f"stopped by run_job.py after --max-hours {max_hours:g} (partial run: weights/last.pt)",
                    "bundle_id": st.get("bundle_id"), "job_bundle": st.get("bundle_id"),
                    "cryoEV_commit": st.get("cryoEV_commit"), "machine": platform.node(), "ended": C.now()})
        rjs = sorted(run_root.glob("*/run.json"), key=lambda p: p.stat().st_mtime)
        if not rjs:
            raise Blocked(f"time cap: stopped after --max-hours {max_hours:g}, before any run folder" if
                          st.get("time_capped") else f"no run.json (run_experiments exit {rc}); see train.log")
        runs = [p.parent for p in rjs]
        rj = C.read_json(rjs[-1])
        st["run"], st["run_status"] = rjs[-1].parent.name, rj.get("status")
        if rj.get("status") != "finished":
            raise Blocked(f"run {rj.get('status')}: {str(rj.get('error') or '')[:300]}")
        if rc != 0:                                    # e.g. the summary step failed after run.json was written
            raise Blocked(f"exit {rc} after a finished run.json; see train.log")
        st.update(status="done", reason=None)
    except Blocked as e:
        st.update(status="blocked", reason=str(e))
    except Exception as e:                              # a runner bug must still end the job, not hang it
        st.update(status="blocked", reason=f"runner error: {type(e).__name__}: {e}")
    try:
        for rd in runs:
            shutil.copytree(rd, tmp / "runs" / rd.name)
            if log.exists() and not (tmp / "runs" / rd.name / "train.log").exists():
                shutil.copy2(log, tmp / "runs" / rd.name / "train.log")
        if runs:
            shutil.copy2(runs[-1] / "run.json", tmp / "run.json")
        if log.exists():
            shutil.copy2(log, tmp / "train.log")
    except OSError as e:
        st.update(status="blocked", reason=f"result copy failed: {e}")
    st.update(ended=C.now(), runs=[rd.name for rd in runs])
    C.write_json(tmp / C.STATUS, st)
    rename_retry(tmp, final)
    done = {k: st.get(k) for k in ("job", "status", "reason", "dataset", "bundle_id", "run", "run_status",
                                   "runs", "machine", "ended", "allow_big", "max_hours", "time_capped")}
    C.write_json(final / C.DONE, done)                  # last: the laptop pulls only folders with DONE.json
    if st["status"] == "done":
        shutil.rmtree(work, ignore_errors=True)
    return done


def notify(done):
    try:
        from notify import _send_ntfy
        title = f"Job {done['status']}: {done['job']}"
        _send_ntfy(title, done.get("reason") or f"{done.get('run')} {done.get('run_status')}",
                   priority=3 if done["status"] == "done" else 4)
    except Exception as e:
        print(f"[jobs] no push: {e}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queue-dir", required=True, type=Path, help="queue root (holds q/, d/, work/)")
    ap.add_argument("--job", help="run this job id only (default: the oldest ready one)")
    ap.add_argument("--wait", type=float, default=0, help="seconds to wait for another runner's lock (default 0)")
    ap.add_argument("--allow-big", action="store_true",
                    help=f"GPU < {LAPTOP_GPU_MB // 1024} GB: also run y8m/l/x or imgsz > {MAX_LAPTOP_IMGSZ} "
                         f"(needs --max-hours)")
    ap.add_argument("--max-hours", type=float, help="stop the trainer after this many hours: blocked 'time cap', "
                    "the partial run (weights/last.pt) is kept")
    a = ap.parse_args(argv)
    if a.allow_big and not a.max_hours:
        print("[jobs] --allow-big needs --max-hours: a model too big for the GPU can spill into system RAM and "
              "run 10-50x slower instead of failing", file=sys.stderr)
        return 2
    root = a.queue_dir.resolve()
    if not (root / C.Q).is_dir():
        print(f"[jobs] {root} has no {C.Q}/ folder", file=sys.stderr)
        return 2
    if gpu_state()["total_mb"] < LAPTOP_GPU_MB:        # the laptop: refuse before touching the queue
        if is_network_path(a.queue_dir) or is_network_path(root):
            print(f"[jobs] {a.queue_dir} is a network folder: the laptop runs only its own local queue "
                  f"(sync.py local), never the desktop's", file=sys.stderr)
            return 2
        if on_battery():
            print("[jobs] on battery: plug in, then start again", file=sys.stderr)
            return 2
    lock = acquire_lock(root / C.LOCK, a.wait)
    if lock is None:
        print("[jobs] another runner holds the lock; not starting", file=sys.stderr)
        return 3
    try:
        qdir = pick(root, a.job)
        if qdir is None:
            print("[jobs] nothing ready")
            return 0
        (root / (C.LOCK + ".info")).write_text(f"pid {os.getpid()} {platform.node()} {qdir.name} {C.now()}\n",
                                               encoding="utf-8")
        print(f"[jobs] running {qdir.name}", flush=True)
        keep_awake(True)
        try:
            done = run_one(root, qdir, a.allow_big, a.max_hours)
        finally:
            keep_awake(False)
        print(f"[jobs] {done['job']}: {done['status']}" + (f" ({done['reason']})" if done.get("reason") else ""))
        notify(done)
        return 0 if done["status"] == "done" else 1
    finally:
        lock.close()


if __name__ == "__main__":
    sys.exit(main())
