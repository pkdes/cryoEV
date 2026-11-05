"""Stand-in for training/run_experiments.py in the job tests (no GPU). Same CLI and output layout.

FAKE_TRAINER = finished (default) | error (run.json error, exit 1) | error_exit0 (run.json error, exit 0)
               | finished_exit1 (run.json finished, exit 1: a later step failed)
               | hang (a run folder with weights/last.pt, then sleeps 600 s: for the time cap)
FAKE_SLEEP   = seconds to sleep after writing <work>/fake_started.txt (pid), before any output
FAKE_EPOCHS  = n: first LATEST_RUN.txt and args.yaml, then per epoch (FAKE_EPOCH_S apart) a results.csv row and
               progress.json by tmp + os.replace with train_yolo.py's 5 tries; a refused try (a reader had the file
               open) is logged to <run>/fake_replace_retries.txt, a lost write to fake_replace_errors.txt
               (jobs/monitor.py's concurrency test)
Like the real script: no dataset.yaml -> exit 1 with no run.json.
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

ap = argparse.ArgumentParser()
for f in ("--dataset", "--model-size", "--imgsz", "--batch", "--epochs", "--run-num", "--weights"):
    ap.add_argument(f)
a, _ = ap.parse_known_args()
root = Path(os.environ["CRYOEV_CRYOAI_ROOT"])
ds = root / "training outputs" / a.dataset
(root / "fake_started.txt").write_text(str(os.getpid()), encoding="utf-8")
time.sleep(float(os.environ.get("FAKE_SLEEP") or 0))
if not (ds / "dataset.yaml").exists():
    print(f"FileNotFoundError: {ds / 'dataset.yaml'}")
    sys.exit(1)
mode = os.environ.get("FAKE_TRAINER", "finished")
name = (f"{datetime.now().strftime('%Y-%m-%d_%H%M%S')}_y8{a.model_size or 'n'}_{a.imgsz or 1440}_"
        f"{a.batch or 24}b_{a.epochs or 300}ep_Run_{a.run_num or 1}")
run = ds / "runs" / name
(run / "weights").mkdir(parents=True)
n_epochs = int(os.environ.get("FAKE_EPOCHS") or 0)
if n_epochs:
    (root / "LATEST_RUN.txt").write_text(f"experiment_name: {name}\ntraining_dir: {run}\n", encoding="utf-8")
    (run / "args.yaml").write_text(f"epochs: {n_epochs}\npatience: 50\nbatch: 16\ntime: null\n", encoding="utf-8")
    (run / "results.csv").write_text("epoch,time,train/box_loss,train/seg_loss,train/cls_loss,train/dfl_loss,"
                                     "metrics/mAP50(M),metrics/mAP50-95(M)\n", encoding="utf-8")
    t0 = time.time()
    for e in range(1, n_epochs + 1):
        time.sleep(float(os.environ.get("FAKE_EPOCH_S") or 0.3))
        with open(run / "results.csv", "a", encoding="utf-8") as f:
            f.write(f"{e},{time.time() - t0:.3f},{2 / e:.4f},{3 / e:.4f},{1 / e:.4f},{1.5 / e:.4f},"
                    f"{0.1 * e:.4f},{0.05 * e:.4f}\n")
        tmp = run / f"progress.json.{os.getpid()}.tmp"
        tmp.write_text(json.dumps({"schema": "progress/1", "phase": "training", "epoch": e, "epochs": n_epochs,
                                   "batch": 8, "batch_asked": 16, "elapsed_s": round(time.time() - t0, 1),
                                   "best_epoch": e, "patience": 50,
                                   "updated": time.strftime("%Y-%m-%dT%H:%M:%S")}), encoding="utf-8")
        for i in range(5):                             # train_yolo._write_progress's retry
            try:
                os.replace(tmp, run / "progress.json")
                break
            except PermissionError as err:
                name = "fake_replace_errors.txt" if i == 4 else "fake_replace_retries.txt"
                with open(run / name, "a", encoding="utf-8") as f:
                    f.write(f"epoch {e} try {i + 1}: {err}\n")
                time.sleep(0.2 * (i + 1))
if mode == "hang":
    (run / "weights" / "last.pt").write_bytes(b"fake partial weights")
    print(f"fake run {name}: training...", flush=True)
    time.sleep(600)
(run / "weights" / "best.pt").write_bytes(b"fake weights")
if not n_epochs:
    (run / "results.csv").write_text("epoch,metrics/mAP50-95(M)\n1,0.1\n", encoding="utf-8")
try:
    bundle = json.loads((ds / "JOB.json").read_text(encoding="utf-8")).get("bundle_id")
except OSError:
    bundle = None
status = "finished" if mode.startswith("finished") else "error"
started = datetime.now().isoformat()
(run / "run.json").write_text(json.dumps({
    "schema": "run/1", "id": name, "dataset": a.dataset, "bundle_id": bundle, "job_bundle": bundle,
    "cryoEV_commit": "fake", "machine": "fake",
    "config": {"arch": f"yolov8{a.model_size or 'n'}-seg", "imgsz": int(a.imgsz or 1440),
               "batch": int(a.batch or 24), "epochs": int(a.epochs or 300)},
    "status": status, "error": None if status == "finished" else "fake failure",
    "started": started, "ended": datetime.now().isoformat(),
    "metrics": {"mask_map50_95": 0.1}}), encoding="utf-8")
print(f"fake run {name}: {status}")
sys.exit(1 if mode in ("error", "finished_exit1") else 0)
