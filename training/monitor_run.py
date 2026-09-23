from __future__ import annotations

import argparse
import csv
import os
import time
from datetime import datetime
from pathlib import Path

import yaml
import subprocess


DEFAULT_OUTPUT_ROOT = Path(r"C:\Users\ML-2619\Desktop\Pujan Cryo\cryo-ev pipeline\data\experiments")


def get_gpu_info():
    """Get GPU utilization and memory from nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split("\n")[0].split(", ")
            if len(parts) >= 4:
                return {
                    "gpu_util": parts[0].strip() + "%",
                    "mem_used": parts[1].strip() + "MB",
                    "mem_total": parts[2].strip() + "MB",
                    "temp": parts[3].strip() + "°C",
                }
    except Exception:
        pass
    return {"gpu_util": "N/A", "mem_used": "N/A", "mem_total": "N/A", "temp": "N/A"}


def run_activity_time(run_dir: Path) -> float:
    """Return the freshest timestamp associated with a run directory."""
    candidates = [
        run_dir / "results.csv",
        run_dir / "args.yaml",
        run_dir / "weights" / "last.pt",
        run_dir,
    ]
    mtimes = [p.stat().st_mtime for p in candidates if p.exists()]
    return max(mtimes) if mtimes else 0.0


def find_run_dirs(output_root: Path) -> list[Path]:
    """Find all run directories under common output layouts."""
    run_dirs: list[Path] = []

    for runs_dir in output_root.glob("**/runs"):
        if runs_dir.is_dir():
            run_dirs.extend([p for p in runs_dir.iterdir() if p.is_dir()])

    # Backward compatibility with the older layout.
    legacy_root = output_root / "training"
    if legacy_root.exists():
        run_dirs.extend([p for p in legacy_root.iterdir() if p.is_dir()])

    # Direct-output layout (e.g., Ultralytics project=<output_root>).
    if output_root.exists() and output_root.is_dir():
        for p in output_root.iterdir():
            if not p.is_dir():
                continue
            if (p / "args.yaml").exists() or (p / "results.csv").exists() or (p / "weights").exists():
                run_dirs.append(p)

    # Deduplicate while preserving order.
    deduped: list[Path] = []
    seen = set()
    for p in run_dirs:
        rp = str(p.resolve())
        if rp in seen:
            continue
        seen.add(rp)
        deduped.append(p)

    return deduped


def parse_latest_run(output_root: Path) -> Path:
    """Pick the freshest run, preferring an actively updating run over stale LATEST_RUN.txt."""
    run_dirs = find_run_dirs(output_root)
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found under {output_root}")

    freshest_run = max(run_dirs, key=run_activity_time)

    latest_path = output_root / "LATEST_RUN.txt"
    if latest_path.exists():
        info = {}
        for line in latest_path.read_text(encoding="utf-8").splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                info[key.strip()] = value.strip()
        training_dir = Path(info.get("training_dir", ""))
        if training_dir.exists() and run_activity_time(training_dir) >= run_activity_time(freshest_run):
            return training_dir

    return freshest_run


def read_args(run_dir: Path) -> dict:
    args_path = run_dir / "args.yaml"
    if not args_path.exists():
        return {}
    with open(args_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def read_results(results_csv: Path) -> list[dict]:
    if not results_csv.exists():
        return []
    with open(results_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            rows.append({(k or "").strip(): (v or "") for k, v in row.items()})
        return rows


def fmt_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def to_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def clear_screen() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def print_status(run_dir: Path, total_epochs: int, rows: list[dict], started_at: float) -> None:
    clear_screen()
    gpu = get_gpu_info()
    print("=" * 72)
    print("YOLO TRAINING MONITOR")
    print("=" * 72)
    print(f"Run folder : {run_dir}")
    print(f"Updated    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"GPU        : {gpu['gpu_util']} | Mem: {gpu['mem_used']}/{gpu['mem_total']} | {gpu['temp']}")

    if not rows:
        print("\nWaiting for `results.csv` to receive its first epoch...\n")
        return

    latest = rows[-1]
    completed_epochs = len(rows)
    progress = (completed_epochs / total_epochs * 100.0) if total_epochs else 0.0

    elapsed = to_float(latest.get("time", "0"), default=0.0)
    if elapsed <= 0:
        elapsed = time.time() - started_at
    avg_epoch_time = elapsed / completed_epochs if completed_epochs else 0
    eta = avg_epoch_time * max(total_epochs - completed_epochs, 0)

    best_mask_map50 = max(to_float(r.get("metrics/mAP50(M)", "0")) for r in rows)
    best_box_map50 = max(to_float(r.get("metrics/mAP50(B)", "0")) for r in rows)

    print(f"\nEpoch      : {completed_epochs}/{total_epochs} ({progress:.1f}%)")
    print(f"Elapsed    : {fmt_seconds(elapsed)}")
    print(f"ETA        : {fmt_seconds(eta)}")
    print(f"Best so far: Box mAP50={best_box_map50:.4f} | Mask mAP50={best_mask_map50:.4f}")

    print("\nLatest metrics")
    print("-" * 72)
    print(f"Precision(B): {to_float(latest.get('metrics/precision(B)')):.4f}")
    print(f"Recall(B)   : {to_float(latest.get('metrics/recall(B)')):.4f}")
    print(f"mAP50(B)    : {to_float(latest.get('metrics/mAP50(B)')):.4f}")
    print(f"mAP50-95(B) : {to_float(latest.get('metrics/mAP50-95(B)')):.4f}")
    print(f"Precision(M): {to_float(latest.get('metrics/precision(M)')):.4f}")
    print(f"Recall(M)   : {to_float(latest.get('metrics/recall(M)')):.4f}")
    print(f"mAP50(M)    : {to_float(latest.get('metrics/mAP50(M)')):.4f}")
    print(f"mAP50-95(M) : {to_float(latest.get('metrics/mAP50-95(M)')):.4f}")
    print("-" * 72)

    print("\nLatest losses (train / val)")
    print("-" * 72)
    for label, train_key, val_key in [
        ("box_loss", "train/box_loss", "val/box_loss"),
        ("seg_loss", "train/seg_loss", "val/seg_loss"),
        ("cls_loss", "train/cls_loss", "val/cls_loss"),
        ("dfl_loss", "train/dfl_loss", "val/dfl_loss"),
        ("sem_loss", "train/sem_loss", "val/sem_loss"),
    ]:
        train_val = latest.get(train_key)
        val_val = latest.get(val_key)
        if train_val is None and val_val is None:
            continue
        print(f"{label:<9s}   : {to_float(train_val):.4f} / {to_float(val_val):.4f}")
    print("-" * 72)


def monitor(output_root: Path, interval: int, run_dir: Path | None = None, continuous: bool = True) -> None:
    target_run_dir = run_dir
    active_run_dir: Path | None = None
    results_csv: Path | None = None
    total_epochs = 0
    started_at = time.time()
    waiting_for_next = False

    while True:
        latest_run = target_run_dir if target_run_dir is not None else parse_latest_run(output_root)
        if target_run_dir is not None and not latest_run.exists():
            raise FileNotFoundError(f"Run directory not found: {latest_run}")
        if active_run_dir != latest_run:
            active_run_dir = latest_run
            args = read_args(active_run_dir)
            results_csv = active_run_dir / "results.csv"
            total_epochs = int(args.get("epochs", 0) or 0)
            started_at = results_csv.stat().st_mtime if results_csv.exists() else time.time()
            waiting_for_next = False

        rows = read_results(results_csv) if results_csv is not None else []
        if rows and results_csv is not None and results_csv.exists():
            started_at = min(started_at, results_csv.stat().st_mtime)

        if active_run_dir is None:
            raise FileNotFoundError("No run directory detected to monitor.")

        print_status(active_run_dir, total_epochs, rows, started_at)

        newer_run_exists = (parse_latest_run(output_root) != active_run_dir) if target_run_dir is None else False
        finished = (
            rows
            and total_epochs
            and len(rows) >= total_epochs
            and (active_run_dir / "weights" / "best.pt").exists()
        )
        if finished and not newer_run_exists:
            if not continuous or target_run_dir is not None:
                print("\nTraining appears complete. Monitor exiting.\n")
                return
            # Sweep mode: this run is done, but a next run may still be starting
            # (per-class metrics / overlays / next training call take a few
            # seconds). Keep polling instead of exiting so the whole campaign
            # can be watched continuously; only Ctrl+C stops the monitor.
            if not waiting_for_next:
                print(f"\nRun '{active_run_dir.name}' complete. Waiting for the next run to start... (Ctrl+C to stop)\n")
                waiting_for_next = True

        time.sleep(interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live monitor for YOLO training runs (follows an entire sweep by default).")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Training output root containing LATEST_RUN.txt")
    parser.add_argument("--run-dir", type=Path, default=None, help="Optional explicit run directory to monitor (implies --once)")
    parser.add_argument("--interval", type=int, default=5, help="Refresh interval in seconds")
    parser.add_argument("--once", action="store_true", help="Exit when the current run finishes instead of waiting for the next run in a sweep")
    args = parser.parse_args()
    try:
        monitor(args.output_root, args.interval, args.run_dir, continuous=not args.once)
    except KeyboardInterrupt:
        print("\nMonitor stopped.\n")
