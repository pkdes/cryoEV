"""Shared helpers for the training-job queue (docs/JOBS.md). Standard library only.

Queue folder (on the desktop, shared to the laptop over Tailscale/SMB):

    <queue>/q/.incoming/<job>/   push in progress (laptop)            never read by the runner
    <queue>/q/<job>/             bundle zip + .md5 + job.json, READY   written by the laptop (READY last)
    <queue>/work/<job>/          unzipped bundle + training output     the runner's only scratch (deleted on done)
    <queue>/d/.tmp_<job>/        results being written                 runner
    <queue>/d/<job>/             run.json, runs/<run>/, status.json, train.log, DONE.json (DONE.json last)
    <queue>/d/.retired/          earlier attempts moved aside by `sync.py retry` (never deleted)
    <queue>/runner.lock          OS lock held by the one running runner
    <queue>/ds/<sha>/            dataset pack shared by thin bundles, READY last (laptop; dataset_cache.py)
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

Q, D, WORK, INCOMING, RETIRED = "q", "d", "work", ".incoming", ".retired"
READY, DONE, JOB_FILE, STATUS, LOCK = "READY", "DONE.json", "job.json", "status.json", "runner.lock"
STOP, WATCH_JSON, WATCH_LOCK, WATCH_LOG = "STOP", "watch.json", "watch.lock", "watch.log"   # watch.py
CANCEL = "CANCEL"                 # q/<job>/CANCEL: sync.py cancel; run_job.py stops or skips the job
MAX_PATH = 240                   # Windows MAX_PATH is 260; Expand-Archive dropped files silently past it
MAX_JOB_ID = 48                  # keeps <queue>\work\<job>\... short
ROBOCOPY_OK_MAX = 7              # robocopy exit codes >= 8 are failures
ROBOCOPY_FLAGS = ["/E", "/Z", "/R:2", "/W:5", "/NP", "/NFL", "/NDL"]
FORBIDDEN_FLAGS = ("/MIR", "/PURGE", "/MOV")     # /MOV and /MOVE delete the source
PREDICT_ENV = "CRYOEV_PREDICT_JOBS"   # "1" turns on the desktop path for predict jobs (push-predict; run_job picks them)
PS_PRED, PREDICT_LOG = "p", "predict.log"   # <prediction set>/p/<model>/predictions/<nnnn>.txt; the predictor's log


def now():
    return datetime.now().isoformat(timespec="seconds")


def predict_enabled() -> bool:
    """Predict jobs on the queue (docs/JOBS.md, "Predict jobs") are off unless $CRYOEV_PREDICT_JOBS is "1"."""
    return os.environ.get(PREDICT_ENV) == "1"


def count_predictions(pred_dir) -> int:
    """Prediction files (<stem>.txt) in pred_dir so far: predict_models writes one per image as it goes."""
    try:
        return sum(1 for n in os.listdir(pred_dir) if n.endswith(".txt"))
    except OSError:
        return 0


def md5_file(path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sidecar_md5(zip_path: Path):
    """md5 from `<zip>.md5` (make_job_bundle.py), or None."""
    p = Path(str(zip_path) + ".md5")
    try:
        return p.read_text(encoding="utf-8").split()[0].lower()
    except (OSError, IndexError):
        return None


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj):
    """Write via a .tmp file and rename, so a reader never sees half a file."""
    path = Path(path)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")   # per process: two writers never share a tmp file
    tmp.write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8")
    for i in range(6):                     # a reader holding the target open (e.g. the laptop monitor over SMB, for a
        try:                               # few ms) makes the replace fail on Windows: retry, ~3 s in all
            return os.replace(tmp, path)
        except PermissionError:
            if i == 5:
                raise
            time.sleep(0.1 * 2 ** i)


def now_utc():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_shared(path, limit=None) -> bytes:
    """Read a file another machine may be replacing or deleting, without getting in its way.

    Python's open() on Windows denies delete sharing, so while a laptop read holds a desktop file open over SMB, the
    desktop's os.replace / rmtree on it fails. This opens with read+write+delete sharing, reads (the last `limit`
    bytes when given) and closes at once. Raises OSError like open()."""
    if os.name != "nt":
        with open(path, "rb") as f:
            if limit:
                f.seek(max(0, os.fstat(f.fileno()).st_size - limit))
            return f.read()
    import ctypes
    from ctypes import wintypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateFileW.restype = wintypes.HANDLE
    k32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID, wintypes.DWORD,
                                wintypes.DWORD, wintypes.HANDLE]
    GENERIC_READ, SHARE_ALL, OPEN_EXISTING = 0x80000000, 0x1 | 0x2 | 0x4, 3
    h = k32.CreateFileW(str(path), GENERIC_READ, SHARE_ALL, None, OPEN_EXISTING, 0x80, None)
    if h == wintypes.HANDLE(-1).value or h is None:
        err = ctypes.get_last_error()
        raise FileNotFoundError(err, f"cannot open {path}", str(path)) if err in (2, 3) else \
            OSError(err, f"cannot open {path} (Windows error {err})", str(path))
    try:
        import msvcrt
        fd = msvcrt.open_osfhandle(h, os.O_RDONLY | os.O_BINARY)
    except OSError:
        k32.CloseHandle(h)
        raise
    with os.fdopen(fd, "rb") as f:                   # closes the handle
        if limit:
            f.seek(max(0, os.fstat(fd).st_size - limit))
        return f.read()


def read_json_shared(path):
    return json.loads(read_shared(path).decode("utf-8"))


def robocopy_argv(src, dst, files=()) -> list:
    """robocopy argv: copy only (/E /Z /R:2 /W:5); never /MIR, /PURGE or /MOV*."""
    argv = ["robocopy", str(src), str(dst), *files, *ROBOCOPY_FLAGS]
    bad = [a for a in argv if a.upper().startswith(FORBIDDEN_FLAGS)]
    if bad:
        raise ValueError(f"refused robocopy flags {bad}")
    return argv


def use_robocopy(copier, *paths) -> bool:
    """auto = robocopy on Windows when a path is a UNC share (\\\\host\\share), else shutil."""
    if copier == "auto":
        return os.name == "nt" and any(str(p).startswith("\\\\") for p in paths)
    return copier == "robocopy"


def copy_tree(src, dst, copier="auto"):
    """Copy folder src into dst (created; existing files there are kept). Raises OSError on failure."""
    src, dst = Path(src), Path(dst)
    if use_robocopy(copier, src, dst):
        r = subprocess.run(robocopy_argv(src, dst), capture_output=True, text=True)
        if r.returncode > ROBOCOPY_OK_MAX:
            raise OSError(f"robocopy {src} -> {dst} failed (exit {r.returncode}): {r.stdout[-400:]}")
        return
    shutil.copytree(src, dst, dirs_exist_ok=True)


def copy_files(src_dir, dst_dir, names, copier="auto"):
    """Copy named files from src_dir into dst_dir (created)."""
    src_dir, dst_dir = Path(src_dir), Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    if use_robocopy(copier, src_dir, dst_dir):
        r = subprocess.run(robocopy_argv(src_dir, dst_dir, names), capture_output=True, text=True)
        if r.returncode > ROBOCOPY_OK_MAX:
            raise OSError(f"robocopy {src_dir} -> {dst_dir} failed (exit {r.returncode}): {r.stdout[-400:]}")
        return
    for n in names:
        shutil.copy2(src_dir / n, dst_dir / n)
