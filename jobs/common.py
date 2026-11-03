"""Shared helpers for the training-job queue (docs/JOBS.md). Standard library only.

Queue folder (on the desktop, shared to the laptop over Tailscale/SMB):

    <queue>/q/.incoming/<job>/   push in progress (laptop)            never read by the runner
    <queue>/q/<job>/             bundle zip + .md5 + job.json, READY   written by the laptop (READY last)
    <queue>/work/<job>/          unzipped bundle + training output     the runner's only scratch (deleted on done)
    <queue>/d/.tmp_<job>/        results being written                 runner
    <queue>/d/<job>/             run.json, runs/<run>/, status.json, train.log, DONE.json (DONE.json last)
    <queue>/d/.retired/          earlier attempts moved aside by `sync.py retry` (never deleted)
    <queue>/runner.lock          OS lock held by the one running runner
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

Q, D, WORK, INCOMING, RETIRED = "q", "d", "work", ".incoming", ".retired"
READY, DONE, JOB_FILE, STATUS, LOCK = "READY", "DONE.json", "job.json", "status.json", "runner.lock"
MAX_PATH = 240                   # Windows MAX_PATH is 260; Expand-Archive dropped files silently past it
MAX_JOB_ID = 48                  # keeps <queue>\work\<job>\... short
ROBOCOPY_OK_MAX = 7              # robocopy exit codes >= 8 are failures
ROBOCOPY_FLAGS = ["/E", "/Z", "/R:2", "/W:5", "/NP", "/NFL", "/NDL"]
FORBIDDEN_FLAGS = ("/MIR", "/PURGE", "/MOV")     # /MOV and /MOVE delete the source


def now():
    return datetime.now().isoformat(timespec="seconds")


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
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8")
    os.replace(tmp, path)


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
