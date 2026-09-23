"""Terminal progress + GPU monitor for the embedding workflow.

Run this in a second terminal for the whole session:

    .venv-new/Scripts/python.exe embedding/monitor.py --watch

It reads 'embedding outputs/progress.json', which each step rewrites as it goes, so
it works whether the step is in the foreground, backgrounded, or already finished.

    --once     print one snapshot and exit
    --watch    refresh every --interval seconds (default 5)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.monitor_run import get_gpu_info   # nvidia-smi: util / mem / temp
from embedding.paths import OUT

PROGRESS = OUT / "progress.json"


def fmt_eta(seconds: float) -> str:
    if seconds < 0 or seconds != seconds or seconds > 86400:
        return "--:--"
    return "%02d:%02d" % (int(seconds) // 60, int(seconds) % 60)


def snapshot() -> str:
    g = get_gpu_info()
    try:
        mem = "%.1f/%.1fGB" % (int(g["mem_used"].rstrip("MB")) / 1024,
                               int(g["mem_total"].rstrip("MB")) / 1024)
    except Exception:
        mem = "%s/%s" % (g["mem_used"], g["mem_total"])
    gpu = "GPU %s %s %s" % (g["gpu_util"], mem, g["temp"])
    now = datetime.now().strftime("%H:%M:%S")

    if not PROGRESS.exists():
        return "%s | no step has started yet | %s" % (now, gpu)

    try:
        p = json.loads(PROGRESS.read_text(encoding="utf-8"))
    except Exception:
        return "%s | progress.json unreadable (mid-write) | %s" % (now, gpu)

    done, total = p.get("done", 0), max(1, p.get("total", 1))
    elapsed = max(1e-6, time.time() - p.get("started_at", time.time()))
    rate = done / elapsed
    eta = (total - done) / rate if rate > 0 else -1
    stale = time.time() - p.get("updated_at", 0)

    state = p.get("state", "?")
    if state == "running" and stale > 120:
        state = "stalled?(%ds)" % int(stale)

    return "%s | step=%-9s | %5d/%-5d (%3d%%) | %6.1f obj/s | ETA %s | %s | %s" % (
        now, p.get("step", "?"), done, total, 100 * done // total,
        rate, fmt_eta(eta), gpu, state,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--interval", type=float, default=5.0)
    args = ap.parse_args()

    if args.once or not args.watch:
        print(snapshot())
        return

    print("Watching %s  (Ctrl+C to stop)\n" % PROGRESS)
    try:
        while True:
            print(snapshot(), flush=True)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
