#!/usr/bin/env python3
"""Laptop path for a prediction set (BUILD_CONTRACTS.md sections 2 and 5): run predict_models on it here.

    python -m jobs.predict_local --ps <ps id | folder> [--root <dir>] [--registry <dir>] [--allow-cpu] [--wait S]

Runs inference/predict_models.py as a subprocess (--images <ps>/images --out <ps>/.p_tmp --models <model> --conf
--max-det [--imgsz] --device cuda --no-overlays; the same argv as the desktop runner, run_job.predict_argv) and keeps
<ps>/status.json current: running, n_done = prediction files written so far (checked every 2 s), then done, or failed
with the error. On success .p_tmp/summary.csv moves to <model>/summary.csv and .p_tmp is renamed to p/, so p/ exists
only for a complete set. The registry record follows: running, then done or failed.
  - GPU: no NVIDIA GPU, or one in use (run_job.gpu_state: any python process, >= 256 MB, or > 2 GB in total) ->
    failed with that error, exit 4. --allow-cpu runs on the CPU instead (slow); never silently.
  - One run at a time per root: <root>/predict.lock (a second run exits 3 and changes nothing).
  - Runs a queued or failed set (a failed one's .p_tmp is cleared first). A done set, one running elsewhere, or one
    queued on the desktop (status.json job set) is refused, exit 2.
  - The predictor's output is appended to <ps>/predict.log; the elapsed time and peak GPU memory are printed.
Exit: 0 done, 1 failed, 2 refused, 3 lock held, 4 GPU busy or unavailable.
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
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
import common as C  # noqa: E402
import predict_set as PS  # noqa: E402
import run_job  # noqa: E402
from data_utils import config  # noqa: E402
from data_utils.oslock import acquire_lock  # noqa: E402

LOCK_FILE = "predict.lock"
PRED_TMP = ".p_tmp"                     # <ps>/.p_tmp while predicting; renamed to p/ when complete
POLL_S = 2.0
EXIT_DONE, EXIT_FAILED, EXIT_REFUSED, EXIT_LOCKED, EXIT_GPU = 0, 1, 2, 3, 4


class Refused(Exception):
    def __init__(self, msg, code=EXIT_REFUSED):
        super().__init__(msg)
        self.code = code


def gpu_memory_mb():
    """(used, total) MB of the first NVIDIA GPU from nvidia-smi, or None (tests: None, no real GPU read)."""
    if run_job.hooks().get("gpu") is not None:
        return None
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=5)
        used, total = (int(x) for x in r.stdout.splitlines()[0].split(","))
        return used, total
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def gpu_check(allow_cpu: bool) -> tuple:
    """-> (device, note). Raises Refused(code 4) when the GPU is missing or busy and allow_cpu is off."""
    st = run_job.gpu_state()
    if not st.get("total_mb"):
        why = "no NVIDIA GPU found (nvidia-smi reports none): CUDA unavailable"
    elif st["busy"]:
        why = "GPU busy: " + ", ".join(f"pid {p} {Path(n).name} {m} MB" for p, n, m in st["busy"][:3])
    else:
        return "cuda", f"GPU {st['total_mb']} MB, {st.get('used_mb') or 0} MB in use"
    if allow_cpu:
        return "cpu", f"{why}; running on the CPU (--allow-cpu)"
    raise Refused(f"{why}. Try again when it is free, or --allow-cpu (slow)", EXIT_GPU)


def fail(ps_dir, ps_id, reg, error, n_done=None) -> dict:
    st = PS.write_status(ps_dir, state="failed", error=error, **({} if n_done is None else {"n_done": n_done}))
    record(ps_id, reg, "failed")
    return st


def record(ps_id, reg, status, **fields):
    """The registry line for a status change; a registry problem is printed, never hides the run's result."""
    try:
        PS.update_record(ps_id, reg, recorded_by="jobs/predict_local.py", status=status, **fields)
    except Exception as e:
        print(f"[predict_local] registry not updated ({ps_id} -> {status}): {e}", file=sys.stderr)


def run(ps, root=None, registry_root=None, allow_cpu=False, wait=0.0, poll_s=POLL_S) -> dict:
    """Predict one set here. -> {"exit", "status", "elapsed_s", "device", "gpu_mb": (before, peak, total) or None}.
    Raises Refused (nothing changed, except a GPU refusal, which marks the set failed with that error)."""
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
    try:
        device, note = gpu_check(allow_cpu)
    except Refused as e:
        fail(ps_dir, ps_id, reg, str(e), n_done=0)
        raise
    out = ps_dir / PRED_TMP
    if out.exists():                                   # this tool's own scratch from a failed attempt
        shutil.rmtree(out)
    pred = out / model / "predictions"
    log = ps_dir / C.PREDICT_LOG
    argv = run_job.predict_argv(req, ps_dir / PS.IMAGES, out, device)
    n_total = len(rows)
    PS.write_status(ps_dir, state="running", n_done=0, n_total=n_total, where="laptop", job=None, error=None)
    record(ps_id, reg, "running", where="laptop", job=None)
    print(f"[predict_local] {ps_id}: {n_total} image(s), model {model}, device {device} ({note})", flush=True)
    mem0 = gpu_memory_mb()
    peak = mem0[0] if mem0 else None
    t0, n_done, interrupted = time.time(), 0, False
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    with open(log, "ab") as lf:
        lf.write(f"\n=== {C.now()} predict_local {ps_id} on {device}: {' '.join(argv)}\n".encode("utf-8"))
        lf.flush()
        p = subprocess.Popen(argv, cwd=REPO, env=env, stdout=lf, stderr=subprocess.STDOUT)
        try:
            while True:
                try:
                    rc = p.wait(timeout=poll_s)
                    break
                except subprocess.TimeoutExpired:
                    n = C.count_predictions(pred)
                    if n != n_done:
                        n_done = n
                        PS.write_status(ps_dir, n_done=n_done)
                    m = gpu_memory_mb()
                    if m and (peak is None or m[0] > peak):
                        peak = m[0]
        except KeyboardInterrupt:
            interrupted = True
            run_job.kill_tree(p)
            rc = p.wait()
    elapsed = round(time.time() - t0, 1)
    n_done = C.count_predictions(pred)
    gpu = (mem0[0], peak, mem0[1]) if mem0 else None
    error = None
    if interrupted:
        error = "interrupted (Ctrl+C)"
    elif rc != 0:
        error = f"predict_models exit {rc}: {run_job.log_tail(log)}"
    else:
        lost = [s for s in stems if not (pred / f"{s}.txt").is_file()]
        if lost:
            error = f"{len(lost)} of {n_total} images have no prediction file, e.g. {lost[:3]}"
        elif not (out / "summary.csv").is_file():
            error = f"predict_models wrote no summary.csv; see {C.PREDICT_LOG}"
    if error is None:
        (out / "summary.csv").replace(out / model / "summary.csv")
        out.rename(ps_dir / C.PS_PRED)
        status = PS.write_status(ps_dir, state="done", n_done=n_done, error=None)
        record(ps_id, reg, "done")
    else:
        status = fail(ps_dir, ps_id, reg, error, n_done=n_done)
    stats = f"{elapsed:g} s, {n_done}/{n_total} images, device {device}" + (
        f", GPU memory {gpu[0]} MB before, peak {gpu[1]} MB of {gpu[2]} MB" if gpu else "")
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"=== {C.now()} {status['state']}: {stats}" + (f"; {error}" if error else "") + "\n")
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
