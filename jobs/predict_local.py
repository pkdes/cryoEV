#!/usr/bin/env python3
"""Laptop path for a prediction set (BUILD_CONTRACTS.md sections 2 and 5): run predict_models on it here.

    python -m jobs.predict_local --ps <ps id | folder> [--root <dir>] [--registry <dir>] [--allow-cpu] [--wait S]

Runs inference/predict_models.py as a subprocess (--images <ps>/images --out <ps>/.p_tmp --models <model> --conf
--max-det [--imgsz] --device cuda --no-overlays; the same argv as the desktop runner, run_job.predict_argv) and keeps
<ps>/status.json current: running with the predictor's pid; every poll (2 s) n_done = prediction files so far and a
fresh `updated` (a stale `updated` with state running = a crash, not a model still loading); then done, or failed
with the error. On success .p_tmp/summary.csv moves to <model>/summary.csv and .p_tmp is renamed to p/ (retried with
backoff; failed if it never succeeds), so p/ exists only for a complete set. The registry record follows.
  - Any error while the predictor runs (a status write, Ctrl+C, a bug) kills the predictor and its children and marks
    the set failed with the reason: never an orphaned predictor under a "running" status.
  - GPU busy (run_job.gpu_state, this process and its parent excluded: any python process, >= 256 MB, or > 2 GB in
    total): the set stays queued with waiting "gpu busy: ...", exit 4. No NVIDIA GPU at all: failed, exit 4.
    --allow-cpu runs on the CPU instead (slow); never silently.
  - One run at a time per root: <root>/predict.lock (a second run exits 3 and changes nothing).
  - Runs a queued or failed set (a failed one's .p_tmp is cleared first). A done set, or one queued on the desktop
    (status.json job set), is refused, exit 2.
  - The predictor's output is appended to <ps>/predict.log; the elapsed time and peak GPU memory are printed.
Exit: 0 done, 1 failed, 2 refused, 3 lock held, 4 GPU busy or unavailable.

Python API (the portal): start_detached(ps_dir) -> pid starts this as a detached process (no console window) and
returns at once; status.json says the rest. run(ps, ...) runs it in this process.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if __package__:                                  # jobs.predict_local (the portal, python -m): sys.path untouched
    from . import common as C
    from . import predict_set as PS
else:
    for _p in (HERE, REPO):
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
    import common as C  # noqa: E402
    import predict_set as PS  # noqa: E402
from data_utils import config  # noqa: E402
from data_utils.oslock import acquire_lock  # noqa: E402

LOCK_FILE = "predict.lock"
PRED_TMP = ".p_tmp"                     # <ps>/.p_tmp while predicting; renamed to p/ when complete
POLL_S = 2.0
RENAME_TRIES, RENAME_BACKOFF_S = 6, 0.5  # .p_tmp -> p: antivirus / indexer / Explorer can hold a folder briefly
EXIT_DONE, EXIT_FAILED, EXIT_REFUSED, EXIT_LOCKED, EXIT_GPU = 0, 1, 2, 3, 4
LOG_DIR = ".logs"                       # <root>/.logs/<ps>.out: a detached run's own console output


class Refused(Exception):
    def __init__(self, msg, code=EXIT_REFUSED):
        super().__init__(msg)
        self.code = code


def _run_job():
    """jobs/run_job.py, imported only when a run starts (it puts jobs/ and training/ on sys.path for itself)."""
    if __package__:
        from . import run_job
    else:
        import run_job
    return run_job


def start_detached(ps_dir, root=None, registry_root=None, allow_cpu=False) -> int:
    """Start `python -m jobs.predict_local --ps <ps_dir>` detached (no console window; survives the caller) and
    return its pid at once. Its console output goes to <root>/.logs/<ps>.out; progress and errors to status.json."""
    ps_dir = Path(ps_dir)
    argv = [sys.executable, "-m", "jobs.predict_local", "--ps", str(ps_dir)]
    if root:
        argv += ["--root", str(root)]
    if registry_root:
        argv += ["--registry", str(registry_root)]
    if allow_cpu:
        argv.append("--allow-cpu")
    out = ps_dir.parent / LOG_DIR / f"{ps_dir.name}.out"
    out.parent.mkdir(parents=True, exist_ok=True)
    kw = {}
    if os.name == "nt":
        kw["creationflags"] = (subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
                               | subprocess.CREATE_NO_WINDOW)
    else:
        kw["start_new_session"] = True
    with open(out, "ab") as f:
        p = subprocess.Popen(argv, cwd=REPO, stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT,
                             env={**os.environ, "PYTHONIOENCODING": "utf-8"}, close_fds=True, **kw)
    return p.pid


def gpu_memory_mb():
    """(used, total) MB of the first NVIDIA GPU from nvidia-smi, or None (tests: None, no real GPU read)."""
    if _run_job().hooks().get("gpu") is not None:
        return None
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=5)
        used, total = (int(x) for x in r.stdout.splitlines()[0].split(","))
        return used, total
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def gpu_check(allow_cpu: bool) -> tuple:
    """-> (device, note, None) to run, or (None, why, "busy" | "missing") when the GPU can't be used and allow_cpu
    is off. This process and its parent (the portal, a shell) never count as busy."""
    rj = _run_job()
    st = rj.gpu_state()
    mine = {os.getpid(), os.getppid()}
    busy = [b for b in st["busy"] if b[0] not in mine]
    if not st.get("total_mb"):
        why, kind = "no NVIDIA GPU found (nvidia-smi reports none): CUDA unavailable", "missing"
    elif busy:
        why, kind = "gpu busy: " + ", ".join(f"pid {p} {Path(n).name} {m} MB" for p, n, m in busy[:3]), "busy"
    else:
        return "cuda", f"GPU {st['total_mb']} MB, {st.get('used_mb') or 0} MB in use", None
    if allow_cpu:
        return "cpu", f"{why}; running on the CPU (--allow-cpu)", None
    return None, why, kind


def fail(ps_dir, ps_id, reg, error, **fields) -> dict:
    st = PS.write_status(ps_dir, state="failed", error=error, pid=None, waiting=None, **fields)
    record(ps_id, reg, "failed")
    return st


def record(ps_id, reg, status, **fields):
    """The registry line for a status change; a registry problem is printed, never hides the run's result."""
    try:
        PS.update_record(ps_id, reg, recorded_by="jobs/predict_local.py", status=status, **fields)
    except Exception as e:
        print(f"[predict_local] registry not updated ({ps_id} -> {status}): {e}", file=sys.stderr)


def rename_retry(src: Path, dst: Path, tries=None, backoff=None):
    """src.rename(dst), retried with backoff (0.5, 1, 2, ... s). Raises the last OSError."""
    tries = RENAME_TRIES if tries is None else tries
    backoff = RENAME_BACKOFF_S if backoff is None else backoff
    for i in range(tries):
        try:
            return src.rename(dst)
        except OSError:
            if i == tries - 1:
                raise
            time.sleep(backoff * 2 ** i)


def run(ps, root=None, registry_root=None, allow_cpu=False, wait=0.0, poll_s=POLL_S) -> dict:
    """Predict one set here. -> {"exit", "status", "elapsed_s", "device", "gpu_mb": (before, peak, total) or None}.
    Raises Refused: nothing changed, except a GPU refusal (queued + waiting when busy, failed when there is none)."""
    reg = Path(registry_root) if registry_root else config.registry_root()
    try:
        ps_dir = PS.find(ps, root, reg)
    except PS.PredictSetError as e:
        raise Refused(str(e))
    lock = acquire_lock(ps_dir.parent / LOCK_FILE, wait)
    if lock is None:
        raise Refused(f"another prediction run holds {ps_dir.parent / LOCK_FILE}; not starting", EXIT_LOCKED)
    try:
        return _run_locked(ps_dir, reg, allow_cpu, poll_s)
    finally:
        lock.close()


def _run_locked(ps_dir: Path, reg: Path, allow_cpu: bool, poll_s: float) -> dict:
    rj = _run_job()
    try:
        req, st, rows = PS.read_request(ps_dir), PS.read_status(ps_dir), PS.read_images(ps_dir)
    except (OSError, ValueError) as e:
        raise Refused(f"{ps_dir}: request.json, status.json or images.csv unreadable: {e}")
    ps_id, model = req["ps"], req["model"]
    state = st.get("state")
    if state == "done":
        raise Refused(f"{ps_id} is already done")
    if st.get("job") and state in ("queued", "running"):
        raise Refused(f"{ps_id} is {state} on the {st.get('where')} as {st['job']}: pull it, or cancel that job first")
    if state not in ("queued", "failed", "running"):
        raise Refused(f"{ps_id}: status {state!r} is not queued or failed")
    if state == "running":                             # we hold the lock, so that run died: run it again
        print(f"[predict_local] {ps_id}: status says running but no run holds the lock; running it again")
    if (ps_dir / C.PS_PRED).exists():
        raise Refused(f"{ps_dir / C.PS_PRED} exists but the status is {state}; check the set by hand")
    stems = [Path(r["file"]).stem for r in rows]
    missing = [r["file"] for r in rows if not (ps_dir / r["file"]).is_file()]
    if not rows or missing:
        raise Refused(f"{ps_id}: {len(missing)} image(s) listed in images.csv are missing, e.g. {missing[:3]}"
                      if missing else f"{ps_id}: images.csv lists no image")
    try:
        PS.model_spec(model)
    except PS.PredictSetError as e:
        raise Refused(f"{ps_id}: {e}")
    device, note, refused = gpu_check(allow_cpu)
    if refused == "busy":                              # someone else's work: stays queued, try again later
        PS.write_status(ps_dir, state="queued", waiting=note, pid=None, error=None)
        if state != "queued":
            record(ps_id, reg, "queued")
        raise Refused(f"{note}. The set stays queued; start it again when the GPU is free, or --allow-cpu (slow)",
                      EXIT_GPU)
    if refused == "missing":
        fail(ps_dir, ps_id, reg, f"{note}; use --allow-cpu (slow)", n_done=0)
        raise Refused(f"{note}; use --allow-cpu (slow)", EXIT_GPU)
    out = ps_dir / PRED_TMP
    if out.exists():                                   # this tool's own scratch from a failed attempt
        shutil.rmtree(out)
    pred = out / model / "predictions"
    log = ps_dir / C.PREDICT_LOG
    argv = rj.predict_argv(req, ps_dir / PS.IMAGES, out, device)
    n_total = len(rows)
    PS.write_status(ps_dir, state="running", n_done=0, n_total=n_total, where="laptop", job=None, error=None,
                    pid=None, waiting=None)
    record(ps_id, reg, "running", where="laptop", job=None)
    print(f"[predict_local] {ps_id}: {n_total} image(s), model {model}, device {device} ({note})", flush=True)
    mem0 = gpu_memory_mb()
    peak = mem0[0] if mem0 else None
    t0, n_done, rc, error = time.time(), 0, None, None
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    p = None
    try:
        with open(log, "ab") as lf:
            lf.write(f"\n=== {C.now()} predict_local {ps_id} on {device}: {' '.join(argv)}\n".encode("utf-8"))
            lf.flush()
            p = subprocess.Popen(argv, cwd=REPO, env=env, stdout=lf, stderr=subprocess.STDOUT)
            PS.write_status(ps_dir, pid=p.pid)
            while True:
                try:
                    rc = p.wait(timeout=poll_s)
                    break
                except subprocess.TimeoutExpired:
                    n_done = C.count_predictions(pred)
                    PS.write_status(ps_dir, n_done=n_done)       # every poll: a fresh `updated` is the heartbeat
                    m = gpu_memory_mb()
                    if m and (peak is None or m[0] > peak):
                        peak = m[0]
    except KeyboardInterrupt:
        error = "interrupted (Ctrl+C)"
    except Exception as e:                              # a status write, the log, a bug: never orphan the predictor
        error = f"predict_local error while running: {type(e).__name__}: {e}"
    finally:
        if p is not None and rc is None:
            rj.kill_tree(p)
            p.wait()
    elapsed = round(time.time() - t0, 1)
    n_done = C.count_predictions(pred)
    gpu = (mem0[0], peak, mem0[1]) if mem0 else None
    try:
        if error is None and rc != 0:
            error = f"predict_models exit {rc}: {rj.log_tail(log)}"
        if error is None:
            lost = [s for s in stems if not (pred / f"{s}.txt").is_file()]
            if lost:
                error = f"{len(lost)} of {n_total} images have no prediction file, e.g. {lost[:3]}"
            elif not (out / "summary.csv").is_file():
                error = f"predict_models wrote no summary.csv; see {C.PREDICT_LOG}"
        if error is None:
            (out / "summary.csv").replace(out / model / "summary.csv")
            try:
                rename_retry(out, ps_dir / C.PS_PRED)
            except OSError as e:
                error = f"predictions complete but {PRED_TMP} could not be renamed to p/: {e}"
        if error is None:
            status = PS.write_status(ps_dir, state="done", n_done=n_done, error=None, pid=None, waiting=None)
            record(ps_id, reg, "done")
        else:
            status = fail(ps_dir, ps_id, reg, error, n_done=n_done)
    except Exception as e:                              # finishing failed: still never a "running" set
        error = f"predict_local error while finishing: {type(e).__name__}: {e}"
        status = fail(ps_dir, ps_id, reg, error, n_done=n_done)
    stats = f"{elapsed:g} s, {n_done}/{n_total} images, device {device}" + (
        f", GPU memory {gpu[0]} MB before, peak {gpu[1]} MB of {gpu[2]} MB" if gpu else "")
    try:
        with open(log, "a", encoding="utf-8") as f:
            f.write(f"=== {C.now()} {status['state']}: {stats}" + (f"; {error}" if error else "") + "\n")
    except OSError:
        pass
    print(f"[predict_local] {ps_id}: {status['state']} ({stats})" + (f": {error}" if error else ""), flush=True)
    return {"exit": EXIT_DONE if error is None else EXIT_FAILED, "status": status, "elapsed_s": elapsed,
            "device": device, "gpu_mb": gpu, "ps_dir": str(ps_dir)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ps", required=True, help="prediction set id (ps_<YYYYMMDD>_<hex>) or its folder")
    ap.add_argument("--root", type=Path, help="where sets live (default <CryoAI-laptop>/predict)")
    ap.add_argument("--registry", type=Path, help="registry folder (default <CryoAI-laptop>/registry)")
    ap.add_argument("--allow-cpu", action="store_true", help="run on the CPU when the GPU is busy or missing (slow)")
    ap.add_argument("--wait", type=float, default=0, help="seconds to wait for another run's lock (default 0)")
    ap.add_argument("--poll-s", type=float, default=POLL_S, help=f"seconds between status.json updates ({POLL_S:g})")
    a = ap.parse_args(argv)
    try:
        return run(a.ps, a.root, a.registry, a.allow_cpu, a.wait, a.poll_s)["exit"]
    except Refused as e:
        print(f"[predict_local] {e}", file=sys.stderr)
        return e.code
    except config.ConfigError as e:
        print(f"[predict_local] {e} (or pass --root / --registry)", file=sys.stderr)
        return EXIT_REFUSED


if __name__ == "__main__":
    sys.exit(main())
