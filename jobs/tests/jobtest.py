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


def hooks_env(train_cmd=FAKE, gpu=DESKTOP_GPU, ultralytics="8.4.94", on_battery=False, free_gb=100, free_commit_gb=100,
              **extra):
    h = {"gpu": gpu, "ultralytics": ultralytics, "on_battery": on_battery, "free_gb": free_gb,
         "free_commit_gb": free_commit_gb, **extra}
    if train_cmd:
        h["train_cmd"] = train_cmd
    return {run_job.HOOKS_ENV: json.dumps(h), run_job.TESTING_ENV: "1"}


def make_bundle(out_dir, name="ds_tiny", args=(), commit=None, yaml=True, files=None, stamp="20261102-120000",
                yaml_text=YAML):
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
            z.writestr(f"{name}/dataset.yaml", yaml_text)
        for rel, data in files.items():
            z.writestr(f"{name}/{rel}", data)
        z.writestr(f"{name}/JOB.json", json.dumps(job))
    Path(str(zp) + ".md5").write_text(f"{hashlib.md5(zp.read_bytes()).hexdigest()}  {zp.name}\n", encoding="utf-8")
    return zp


def queue(share, job_id, zp):
    """Put a bundle in the queue the way `sync.py push` does (no registry check)."""
    import sync
    return sync.push(job_id, zp, share, copier="shutil", check_registry=False, check_origin=False)


LOSS_COLS = ("train/box_loss", "train/seg_loss", "train/cls_loss", "train/dfl_loss")
CSV_COLS = ("epoch", "time", *LOSS_COLS, "train/sem_loss", "metrics/precision(B)", "metrics/recall(B)",
            "metrics/mAP50(B)", "metrics/mAP50-95(B)", "metrics/precision(M)", "metrics/recall(M)", "metrics/mAP50(M)",
            "metrics/mAP50-95(M)", "val/box_loss", "val/seg_loss", "lr/pg0")


def running_fixture(share, job="job_20261104_live", times=(30, 42.5, 55, 67.5), epochs=300, progress=None,
                    patience=100, batch=16, max_hours=None, age_s=0, latest_run=True, csv=True, tail=""):
    """A queue with one job mid-training, built in code (no real data): q/<job>/ (job.json, READY),
    d/.tmp_<job>/ (so queue_rows says running), work/<job>/LATEST_RUN.txt naming the run with a desktop path
    (C:\\cj\\work\\...), and the run folder: results.csv (one row per cumulative time in `times`), args.yaml and,
    when `progress` is a dict, progress.json (merged over defaults). The last epoch is `age_s` seconds old.
    `tail` is appended to results.csv as is (a half-written line). -> the run folder."""
    import time as _t
    share = Path(share)
    q = share / C.Q / job
    q.mkdir(parents=True, exist_ok=True)
    (q / C.JOB_FILE).write_text(json.dumps({"schema": "queue_job/1", "job": job, "max_hours": max_hours}),
                                encoding="utf-8")
    (q / C.READY).write_text("x\n", encoding="utf-8")
    (share / C.D / f".tmp_{job}").mkdir(parents=True, exist_ok=True)
    name = f"2026-11-04_120000_y8m_1440_{batch}b_{epochs}ep_Run_1"
    run = share / C.WORK / job / "training outputs" / "ds_live" / "runs" / name
    run.mkdir(parents=True, exist_ok=True)
    if latest_run:
        (share / C.WORK / job / "LATEST_RUN.txt").write_text(
            f"experiment_name: {name}\ntraining_dir: C:\\cj\\work\\{job}\\training outputs\\ds_live\\runs\\{name}\n"
            f"started: 2026-11-04T12:00:00\n", encoding="utf-8")
    (run / "args.yaml").write_text(f"task: segment\nepochs: {epochs}\ntime: null\npatience: {patience}\n"
                                   f"batch: {batch}\nimgsz: 1440\n", encoding="utf-8")
    when = _t.time() - age_s
    if csv:
        lines = [",".join(CSV_COLS)]
        for i, t in enumerate(times, 1):
            vals = [i, t, 2.0 / i, 3.0 / i, 1.0 / i, 1.5 / i, 0, .5, .5, .4, .2, .5, .5, round(.1 * i, 4),
                    round(.05 * i, 4), 1, 1, .001]
            lines.append(",".join(str(v) for v in vals))
        (run / "results.csv").write_text("\n".join(lines) + "\n" + tail, encoding="utf-8")
        os.utime(run / "results.csv", (when, when))
    if progress is not None:
        n = len(times)
        doc = {"schema": "progress/1", "phase": "training", "epoch": n, "epochs": epochs, "batch": batch,
               "batch_asked": batch, "epoch_s": 12.5, "elapsed_s": times[-1] if times else 0, "best_epoch": n,
               "patience": patience, "time_cap_h": None, "gpu_reserved_gb": 31.4,
               "metrics": {"metrics/mAP50(M)": round(.1 * n, 4), "metrics/mAP50-95(M)": round(.05 * n, 4)},
               "updated": _t.strftime("%Y-%m-%dT%H:%M:%S", _t.localtime(when)), **progress}
        (run / "progress.json").write_text(json.dumps(doc), encoding="utf-8")
    return run


def watch_fixture(share, state="running", job=None, age_s=10, detail=None):
    """watch.json as watch.py writes it, its heartbeat `age_s` seconds old."""
    from datetime import datetime, timedelta, timezone
    hb = datetime.now(timezone.utc) - timedelta(seconds=age_s)
    (Path(share) / C.WATCH_JSON).write_text(json.dumps({
        "schema": "watch/1", "state": state, "detail": detail, "job": job,
        "heartbeat": (datetime.now() - timedelta(seconds=age_s)).isoformat(timespec="seconds"),
        "heartbeat_utc": hb.isoformat(timespec="seconds"), "machine": "fake-desktop"}), encoding="utf-8")


def done_fixture(share, job, status="done", reason=None, ended="2026-11-04T10:00:00", metrics=None):
    """q/<job>/ + d/<job>/ with run.json and DONE.json, as run_job.py leaves a finished job."""
    share = Path(share)
    (share / C.Q / job).mkdir(parents=True, exist_ok=True)
    (share / C.Q / job / C.JOB_FILE).write_text(json.dumps({"job": job}), encoding="utf-8")
    (share / C.Q / job / C.READY).write_text("x\n", encoding="utf-8")
    d = share / C.D / job
    d.mkdir(parents=True, exist_ok=True)
    (d / "run.json").write_text(json.dumps({"schema": "run/1", "id": "r1", "status": "finished",
                                            "metrics": metrics or {"mask_map50": .61, "mask_map50_95": .32},
                                            "best_epoch": 140}), encoding="utf-8")
    (d / C.DONE).write_text(json.dumps({"job": job, "status": status, "reason": reason, "run": "r1",
                                        "ended": ended}), encoding="utf-8")
    return d


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
