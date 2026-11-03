"""Shared fixtures for the job tests: temp queue roots, tiny bundles, the fake trainer, env hooks."""
import hashlib
import json
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from unittest import mock

TESTS = Path(__file__).resolve().parent
JOBS = TESTS.parent
REPO = JOBS.parent
sys.path.insert(0, str(JOBS))
sys.path.insert(0, str(REPO))
import common as C  # noqa: E402
import run_job  # noqa: E402

FAKE = [sys.executable, str(TESTS / "fake_trainer.py")]
DESKTOP_GPU = {"total_mb": 49140, "apps": []}
LAPTOP_GPU = {"total_mb": 4096, "apps": []}
YAML = "names:\n  0: EV\ntrain: images/train\nval: images/val\n"


def hooks_env(train_cmd=FAKE, gpu=DESKTOP_GPU, ultralytics="8.4.94", on_battery=False, free_gb=100, **extra):
    h = {"gpu": gpu, "ultralytics": ultralytics, "on_battery": on_battery, "free_gb": free_gb, **extra}
    if train_cmd:
        h["train_cmd"] = train_cmd
    return {run_job.HOOKS_ENV: json.dumps(h), run_job.TESTING_ENV: "1"}


def make_bundle(out_dir, name="ds_tiny", args=(), commit=None, yaml=True, files=None, stamp="20261102-120000"):
    """A bundle zip shaped like make_job_bundle.py's (+ .md5 sidecar), written directly."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    zp = out_dir / f"{name}__job_{stamp}.zip"
    files = files if files is not None else {"images/train/a.png": b"png", "labels/train/a.txt": b"0 0 0 1 1 1 0",
                                              "images/val/b.png": b"png", "labels/val/b.txt": b"0 0 0 1 1 1 0"}
    job = {"bundle_id": zp.stem, "dataset_name": name, "cryoEV_commit": commit or run_job.repo_commit(),
           "run_experiments_args": ["--dataset", name, "--single-run", *args]}
    with zipfile.ZipFile(zp, "w") as z:
        if yaml:
            z.writestr(f"{name}/dataset.yaml", YAML)
        for rel, data in files.items():
            z.writestr(f"{name}/{rel}", data)
        z.writestr(f"{name}/JOB.json", json.dumps(job))
    Path(str(zp) + ".md5").write_text(f"{hashlib.md5(zp.read_bytes()).hexdigest()}  {zp.name}\n", encoding="utf-8")
    return zp


def queue(share, job_id, zp):
    """Put a bundle in the queue the way `sync.py push` does (no registry check)."""
    import sync
    return sync.push(job_id, zp, share, copier="shutil", check_registry=False)


class TempRoots:
    """A temp folder with a queue share and a bundle folder; env cleaned (no ntfy topic)."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="cj_"))
        self.share = self.tmp / "s"
        (self.share / C.Q).mkdir(parents=True)
        self.bundles = self.tmp / "b"
        env = {k: v for k, v in os.environ.items() if k not in ("CRYOEV_NTFY_TOPIC", run_job.HOOKS_ENV, run_job.TESTING_ENV)}
        self.env = mock.patch.dict(os.environ, env, clear=True)
        self.env.start()

    def tearDown(self):
        self.env.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_job(self, *extra, **hook_kw):
        with mock.patch.dict(os.environ, hooks_env(**hook_kw)):
            return run_job.main(["--queue-dir", str(self.share), *extra])

    def done(self, job_id):
        return C.read_json(self.share / C.D / job_id / C.DONE)
