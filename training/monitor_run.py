from __future__ import annotations

import argparse
import csv
import os
import time
from datetime import datetime
from pathlib import Path

import yaml


DEFAULT_OUTPUT_ROOT = Path(r"C:\Users\ML-2619\Desktop\Pujan Cryo\cryo-ev pipeline\data\experiments")


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
    """Find all run directories under the experiments tree."""
    run_dirs: list[Path] = []

    for runs_dir in output_root.glob("**/runs"):
        if runs_dir.is_dir():
            run_dirs.extend([p for p in runs_dir.iterdir() if p.is_dir()])

    # Backward compatibility with the older layout.
    legacy_root = output_root / "training"
    if legacy_root.exists():
        run_dirs.extend([p for p in legacy_root.iterdir() if p.is_dir()])

    return run_dirs


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
    print("=" * 72)
    print("YOLO TRAINING MONITOR")
    print("=" * 72)
    print(f"Run folder : {run_dir}")
    print(f"Updated    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if not rows:
        print("\nWaiting for `results.csv` to receive its first epoch...\n")
        return

    latest = rows[-1]
    completed_epochs = len(rows)
    progress = (completed_epochs / total_epochs * 100.0) if total_epochs else 0.0

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


def monitor(output_root: Path, interval: int) -> None:
    run_dir: Path | None = None
    results_csv: Path | None = None
    total_epochs = 0
    started_at = time.time()

    while True:
        latest_run = parse_latest_run(output_root)
        if run_dir != latest_run:
            run_dir = latest_run
            args = read_args(run_dir)
            results_csv = run_dir / "results.csv"
            total_epochs = int(args.get("epochs", 0) or 0)
            started_at = results_csv.stat().st_mtime if results_csv.exists() else time.time()

        rows = read_results(results_csv) if results_csv is not None else []
        if rows and results_csv is not None and results_csv.exists():
            started_at = min(started_at, results_csv.stat().st_mtime)

        print_status(run_dir, total_epochs, rows, started_at)

        newer_run_exists = parse_latest_run(output_root) != run_dir
        finished = (
            rows
            and total_epochs
            and len(rows) >= total_epochs
            and (run_dir / "weights" / "best.pt").exists()
        )
        if finished and not newer_run_exists:
            print("\nTraining appears complete. Monitor exiting.\n")
            return

        time.sleep(interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live monitor for the latest YOLO training run.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Training output root containing LATEST_RUN.txt")
    parser.add_argument("--interval", type=int, default=5, help="Refresh interval in seconds")
    args = parser.parse_args()
    monitor(args.output_root, args.interval)
