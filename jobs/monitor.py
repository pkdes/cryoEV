#!/usr/bin/env python3
"""Laptop: a live snapshot of the desktop's training queue for the history page's Live tab (docs/JOBS.md).

    python jobs/monitor.py --share \\\\gpu-desktop\\cj [--share ...] [--every 30] [--once] [--out <file>]
        [--servers review=8766,sandbox=8765,history=8768]

Every --every seconds it reads each queue (in a thread, given up after 20 s) and writes one JSON file, by default
<registry>/live/live.json (the history page fetches live/live.json when served over http):
  - reach: ok | offline (path not found or no answer) | access denied;
  - the watcher (watch.json + heartbeat age; stale after 5 min without a heartbeat);
  - every job in q/ (sync.queue_rows), and the 10 most recent finished ones with their final metrics;
  - the running job: phase, epoch / epochs, elapsed, ETA, batch used vs asked, GPU memory, the latest val mask mAP
    and the per-epoch series, from work/<job>/.../runs/<run>/ (progress.json, else results.csv + args.yaml);
  - servers: this laptop's review / sandbox / history servers (--servers), each probed with GET /api/ping on
    127.0.0.1 (any HTTP answer, 403 and 404 included, is up; refused or no answer in 1 s is down), all at once,
    in their own threads, given up after 3 s in total. The history page's Servers bar shows them.
Read only on the share: every file is read with common.read_shared (full sharing, closed at once), so a desktop
delete/rmtree is never blocked by it; an os.replace onto a file being read can still fail for those milliseconds,
which the desktop's writers retry (common.write_json, train_yolo._write_progress). Nothing under d/.tmp_* is
opened. Never writes the registry
*.jsonl. One monitor per output file (monitor.lock next to it; a second one exits 3).
"""
from __future__ import annotations

import argparse
import csv
import http.client
import io
import json
import os
import platform
import statistics
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
import common as C  # noqa: E402
import sync  # noqa: E402
from data_utils import config  # noqa: E402
from data_utils.oslock import acquire_lock  # noqa: E402

READ_TIMEOUT_S = 20          # a queue that doesn't answer in this long is reported offline
STALE_HEARTBEAT_S = 300      # watch.py writes watch.json every minute; sync.py status warns after 5 min
HUNG_MIN_S = 600             # running, no new epoch for max(5 x median epoch, this): hung
HUNG_EPOCHS = 5
ETA_WINDOW = 5               # median of the last <= 5 epoch durations
DEFAULT_HOURS = 48.0         # watch.py's --max-hours when job.json has none
N_FINISHED = 10
SERVERS = "review=8766,sandbox=8765,history=8768"   # name=port on 127.0.0.1 (docs/JOBS.md, Live monitor)
PROBE_TIMEOUT_S = 1.0        # per server: connect + answer
PROBE_TOTAL_S = 3.0          # all servers together (they run in parallel)
LIVE_FILE, LOCK_FILE = "live.json", "monitor.lock"
LOSSES = (("box", "train/box_loss"), ("seg", "train/seg_loss"), ("cls", "train/cls_loss"), ("dfl", "train/dfl_loss"))
MAP50, MAP50_95 = "metrics/mAP50(M)", "metrics/mAP50-95(M)"


# ---------------------------------------------------------------- run folder
def read_results(raw: bytes) -> list:
    """results.csv -> [{column: float}] per epoch. A last line without a newline is still being written: ignored."""
    text = raw.decode("utf-8", errors="replace")
    if text and not text.endswith("\n"):
        text = text[:text.rfind("\n") + 1]
    rows = []
    for r in csv.DictReader(io.StringIO(text)):
        row = {}
        for k, v in r.items():
            if k is None:
                continue
            try:
                row[k.strip()] = float(v)
            except (TypeError, ValueError):
                row[k.strip()] = None
        if row.get("epoch") is not None:
            rows.append(row)
    return rows


def read_args(raw: bytes) -> dict:
    """The top-level `key: value` lines of Ultralytics' args.yaml (no yaml import: standard library only)."""
    out = {}
    for line in raw.decode("utf-8", errors="replace").splitlines():
        if not line or line[0] in " #-" or ":" not in line:
            continue
        k, v = (s.strip() for s in line.split(":", 1))
        if v in ("null", "~", ""):
            out[k] = None
            continue
        try:
            out[k] = int(v)
        except ValueError:
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v.strip("'\"")
    return out


def epoch_median(rows) -> float | None:
    """Median of the last <= 5 epoch durations from the cumulative time column. Epoch 1 (setup + warmup) has no
    previous row so it never counts; a non-positive step (a resumed run, a clock jump) is skipped."""
    steps = []
    for a, b in zip(rows, rows[1:]):
        if a.get("time") is None or b.get("time") is None:
            continue
        d = b["time"] - a["time"]
        if d > 0:
            steps.append(d)
    return statistics.median(steps[-ETA_WINDOW:]) if steps else None


def eta(epoch, epochs, median, best_epoch=None, patience=None, max_hours=None, elapsed_s=None) -> dict:
    """-> {eta_s, eta_patience_bound_s, eta_capped}. Remaining = epochs left x median. The patience bound
    (best_epoch + patience - epoch) x median is reported when smaller: Ultralytics stops there unless val improves.
    Both are capped at the job's time limit (max_hours - elapsed): run_job.py stops the trainer then."""
    out = {"eta_s": None, "eta_patience_bound_s": None, "eta_capped": False}
    if median is None or epoch is None or not epochs:
        return out
    full = max(0, epochs - epoch) * median
    bound = None
    if best_epoch is not None and patience:
        bound = max(0, best_epoch + patience - epoch) * median
        if bound >= full:
            bound = None
    cap = None
    if max_hours and elapsed_s is not None:
        cap = max(0.0, max_hours * 3600 - elapsed_s)
    if cap is not None and full > cap:
        full, out["eta_capped"] = cap, True
    if bound is not None and cap is not None:
        bound = min(bound, cap)
    out["eta_s"] = round(full, 1)
    out["eta_patience_bound_s"] = None if bound is None else round(bound, 1)
    return out


def parse_time(s) -> float | None:
    """ISO time -> epoch seconds; a time without a zone is the desktop's local time (the same zone as the laptop)."""
    try:
        return datetime.fromisoformat(str(s)).timestamp()
    except (TypeError, ValueError):
        return None


def map_run_path(text: str, share: Path, job: str) -> Path | None:
    """LATEST_RUN.txt names the run with the desktop's path (C:\\cj\\work\\<job>\\...): the same folder on the share."""
    for line in text.splitlines():
        val = line.split(":", 1)[1].strip() if line.startswith("training_dir:") else line.strip()
        parts = val.replace("/", "\\").split("\\")
        for i in range(len(parts) - 1):
            if parts[i].lower() == C.WORK and parts[i + 1].lower() == job.lower():
                return share.joinpath(C.WORK, job, *[p for p in parts[i + 2:] if p])
    return None


def find_run(share: Path, job: str) -> Path | None:
    """work/<job>/LATEST_RUN.txt mapped to the share, else the newest run folder (names start with the time)."""
    work = share / C.WORK / job
    try:
        p = map_run_path(C.read_shared(work / "LATEST_RUN.txt").decode("utf-8", errors="replace"), share, job)
        if p is not None and p.is_dir():
            return p
    except OSError:
        pass
    runs = [r for ds in (work / "training outputs").glob("*") if ds.is_dir()
            for r in (ds / "runs").glob("*") if r.is_dir()]
    return max(runs, key=lambda r: r.name) if runs else None


def series(rows) -> dict:
    """Column arrays for the page's charts: epoch, train losses, val mask mAP50 and mAP50-95."""
    out = {"epoch": [int(r["epoch"]) for r in rows]}
    for name, col in LOSSES:
        out[name] = [r.get(col) for r in rows]
    out["map50"] = [r.get(MAP50) for r in rows]
    out["map50_95"] = [r.get(MAP50_95) for r in rows]
    return out


def running_job(share: Path, job: str, now=None) -> dict:
    """What the running job's run folder says. Never raises for a missing file: a field is None when unknown."""
    now = time.time() if now is None else now
    out = {"job": job, "run": None, "phase": "setting up", "epoch": None, "epochs": None, "elapsed_s": None,
           "batch": None, "batch_asked": None, "batch_halved": None, "batch_note": None, "gpu_reserved_gb": None,
           "map50": None, "map50_95": None, "best_epoch": None, "patience": None, "max_hours": None,
           "epoch_median_s": None, "last_epoch_age_s": None, "hung": False, "source": None, "series": None,
           "eta_s": None, "eta_patience_bound_s": None, "eta_capped": False}
    try:
        out["max_hours"] = C.read_json_shared(share / C.Q / job / C.JOB_FILE).get("max_hours") or DEFAULT_HOURS
    except (OSError, ValueError, AttributeError):
        out["max_hours"] = DEFAULT_HOURS
    run = find_run(share, job)
    if run is None:
        return out
    out["run"] = run.name

    def shared(name, parse):
        try:
            return parse(C.read_shared(run / name))
        except (OSError, ValueError):
            return None
    prog = shared("progress.json", lambda b: json.loads(b.decode("utf-8")))
    args = shared("args.yaml", read_args) or {}
    rows = shared("results.csv", read_results) or []
    try:
        csv_mtime = (run / "results.csv").stat().st_mtime if rows else None
    except OSError:
        csv_mtime = None
    if not isinstance(prog, dict):
        prog = None
    if prog is None and not rows:
        return out                                     # trainer starting: no epoch yet
    out["source"] = "progress.json" if prog else "results.csv + args.yaml (older run)"
    last = rows[-1] if rows else {}
    out["epoch"] = (prog or {}).get("epoch") or (int(last["epoch"]) if last else None)
    out["epochs"] = (prog or {}).get("epochs") or args.get("epochs")
    out["patience"] = (prog or {}).get("patience") or args.get("patience")
    out["best_epoch"] = (prog or {}).get("best_epoch")
    if prog:
        out["batch"], out["batch_asked"] = prog.get("batch"), prog.get("batch_asked") or args.get("batch")
        if out["batch"] is not None and out["batch_asked"] is not None:
            out["batch_halved"] = out["batch"] < out["batch_asked"]
        out["gpu_reserved_gb"] = prog.get("gpu_reserved_gb")
    else:
        out["batch_asked"], out["batch_note"] = args.get("batch"), "unknown (older run)"
    m = (prog or {}).get("metrics") or {}
    out["map50"] = last.get(MAP50) if last.get(MAP50) is not None else m.get(MAP50)
    out["map50_95"] = last.get(MAP50_95) if last.get(MAP50_95) is not None else m.get(MAP50_95)
    # elapsed: the trainer's clock at the last epoch, plus the time since that epoch was written
    updated = max([t for t in (parse_time((prog or {}).get("updated")), csv_mtime) if t is not None], default=None)
    at_last = (prog or {}).get("elapsed_s") if prog and prog.get("elapsed_s") is not None else last.get("time")
    age = max(0.0, now - updated) if updated is not None else None
    out["last_epoch_age_s"] = None if age is None else round(age)
    out["elapsed_s"] = None if at_last is None else round(at_last + (age or 0))
    med = epoch_median(rows)
    out["epoch_median_s"] = None if med is None else round(med, 2)
    out.update(eta(out["epoch"], out["epochs"], med, out["best_epoch"], out["patience"], out["max_hours"],
                   out["elapsed_s"]))
    # phase: final_eval from progress.json; or the last epoch is done (all epochs, or patience ran out) and no DONE yet
    early_stop = (out["best_epoch"] is not None and out["patience"] and out["epoch"] is not None
                  and out["epoch"] - out["best_epoch"] >= out["patience"])
    if (prog or {}).get("phase") == "final_eval" or (out["epochs"] and out["epoch"] and out["epoch"] >= out["epochs"]) \
            or early_stop:
        out["phase"] = "finishing"
    else:
        out["phase"] = "training"
    # hung: no new epoch for max(5 x median, 10 min); with one epoch only, epoch 1's own time stands in for the median
    ref = med if med is not None else (rows[0].get("time") if rows else (prog or {}).get("epoch_s"))
    if age is not None and ref:
        out["hung_after_s"] = round(max(HUNG_EPOCHS * ref, HUNG_MIN_S))
        out["hung"] = out["phase"] == "training" and age > out["hung_after_s"]
    out["series"] = series(rows)
    return out


# ---------------------------------------------------------------- one queue
def finished_jobs(share: Path, rows) -> list:
    """The most recent finished jobs (DONE.json), newest first, with run.json's final metrics."""
    done = [r for r in rows if r.get("done")]
    done.sort(key=lambda r: str(r["done"].get("ended") or ""), reverse=True)
    out = []
    for r in done[:N_FINISHED]:
        d = r["done"]
        item = {"job": r["job"], "status": d.get("status"), "reason": d.get("reason"), "ended": d.get("ended"),
                "run": d.get("run"), "run_status": d.get("run_status"), "pulled": r.get("pulled"), "metrics": None,
                "best_epoch": None}
        try:
            rj = C.read_json_shared(share / C.D / r["job"] / "run.json")
            item["metrics"], item["best_epoch"] = rj.get("metrics"), rj.get("best_epoch")
        except (OSError, ValueError, AttributeError):
            pass
        out.append(item)
    return out


def reach(share: Path):
    """-> (reach, detail): ok, offline (path not found) or access denied."""
    try:
        os.stat(share)
        os.listdir(share / C.Q)
    except PermissionError as e:
        return "access denied", str(e)
    except FileNotFoundError as e:
        return "offline", f"not found: {e.filename or share}"
    except OSError as e:
        if getattr(e, "winerror", None) == 5:
            return "access denied", str(e)
        return "offline", str(e)
    return "ok", None


def watcher_view(w) -> dict:
    """watch.json -> what the page shows: alive / paused / waiting for GPU / waiting / stopped / stale / no watcher."""
    if not w:
        return {"display": "no watcher", "state": None, "detail": "no watch.json (watch.py never ran here)",
                "job": None, "age_s": None, "stale": True}
    age = w.get("age_s")
    stale = age is None or age > STALE_HEARTBEAT_S
    st, detail = w.get("state"), w.get("detail")
    if stale:
        display = "stale"
    elif st in ("running", "idle"):
        display = "alive"
    elif st == "waiting":
        display = "waiting for GPU" if "GPU" in str(detail or "") else "waiting"
    else:
        display = st or "unknown"
    return {"display": display, "state": st, "detail": detail, "job": w.get("job"), "age_s": age, "stale": stale,
            "machine": w.get("machine"), "heartbeat": w.get("heartbeat")}


def read_queue(share: Path, now=None) -> dict:
    """Everything the page shows for one queue. Runs in a worker thread (see snapshot())."""
    out = {"share": str(share), "reach": None, "detail": None, "watcher": None, "jobs": [], "running": None,
           "finished": []}
    out["reach"], out["detail"] = reach(share)
    if out["reach"] != "ok":
        return out
    w = watcher_view(sync.watch_state(share))
    rows = sync.queue_rows(share)
    out["watcher"] = w
    out["jobs"] = [{"job": r["job"], "state": r["state"], "reason": r["reason"], "pulled": r["pulled"],
                    "cancel_requested": r["cancel_requested"]} for r in rows]
    running = [r for r in rows if r["state"] == "running"]
    if running:
        r = next((x for x in running if x["job"] == w.get("job")), running[0])
        out["running"] = running_job(share, r["job"], now)
        out["running"]["cancel_requested"] = r["cancel_requested"]
        out["running"]["watcher_on_it"] = w.get("job") == r["job"] and w.get("state") == "running" and not w["stale"]
    out["finished"] = finished_jobs(share, rows)
    return out


class Reader:
    """Reads each queue in its own daemon thread and gives up after READ_TIMEOUT_S. A read still stuck on a dead
    SMB connection is never started twice: the queue stays offline until that thread returns."""

    def __init__(self, timeout=READ_TIMEOUT_S):
        self.timeout = timeout
        self.busy: dict = {}

    def read(self, share: Path) -> dict:
        key = str(share)
        t = self.busy.get(key)
        if t is not None and t.is_alive():
            return {"share": key, "reach": "offline", "detail": "the previous read has not returned (network?)"}
        box = {}

        def work():
            try:
                box["v"] = read_queue(share)
            except Exception as e:                     # the share went away mid-read, a bad file: report, go on
                box["v"] = {"share": key, "reach": "offline", "detail": f"{type(e).__name__}: {e}"}
        t = threading.Thread(target=work, daemon=True, name=f"monitor {key}")
        t.start()
        t.join(self.timeout)
        if t.is_alive():
            self.busy[key] = t
            return {"share": key, "reach": "offline", "detail": f"no answer in {self.timeout:g} s"}
        self.busy.pop(key, None)
        return box["v"]


# ---------------------------------------------------------------- local servers
def parse_servers(text: str) -> list:
    """"review=8766,sandbox=8765" -> [("review", 8766), ("sandbox", 8765)]; "" -> [] (no probes)."""
    out = []
    for part in (text or "").split(","):
        part = part.strip()
        if not part:
            continue
        name, sep, port = part.partition("=")
        if not sep or not name.strip() or not port.strip().isdigit() or not 0 < int(port) < 65536:
            raise ValueError(f"bad server {part!r}: expected name=port, e.g. review=8766")
        out.append((name.strip(), int(port)))
    return out


def probe(port: int, timeout=PROBE_TIMEOUT_S) -> tuple:
    """GET /api/ping on 127.0.0.1:port -> (up, ms). Any HTTP status is up: the review servers answer 403 without
    the key and the history server (http.server) 404; a refused connection, a timeout or garbage is down."""
    t0 = time.perf_counter()
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        c.request("GET", "/api/ping")
        c.getresponse().read(64)
        up = True
    except (OSError, http.client.HTTPException):
        up = False
    finally:
        c.close()
    return up, round((time.perf_counter() - t0) * 1000)


def probe_servers(servers, timeout=PROBE_TIMEOUT_S, total=PROBE_TOTAL_S) -> list:
    """[{name, port, up, ms}] for each (name, port), probed in parallel; one that has not answered by `total`
    seconds is down. Separate from the queue reader threads, so a dead share never delays these."""
    box = [None] * len(servers)

    def work(i, port):
        box[i] = probe(port, timeout)
    threads = [threading.Thread(target=work, args=(i, port), daemon=True, name=f"probe {port}")
               for i, (_, port) in enumerate(servers)]
    for t in threads:
        t.start()
    end = time.monotonic() + total
    for t in threads:
        t.join(max(0.0, end - time.monotonic()))
    out = []
    for (name, port), r in zip(servers, box):
        up, ms = r if r is not None else (False, None)
        out.append({"name": name, "port": port, "up": up, "ms": ms})
    return out


def snapshot(shares, reader: Reader, every, servers=()) -> dict:
    return {"schema": "live/1", "generated": C.now(), "generated_utc": C.now_utc(), "interval_s": every,
            "monitor": {"machine": platform.node(), "pid": os.getpid()},
            "servers": probe_servers(list(servers)),
            "queues": [reader.read(Path(s)) for s in shares]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--share", action="append", required=True, type=Path,
                    help="queue root, e.g. \\\\gpu-desktop\\cj (repeatable)")
    ap.add_argument("--every", type=float, default=30.0, help="seconds between snapshots (default 30)")
    ap.add_argument("--once", action="store_true", help="write one snapshot, then exit")
    ap.add_argument("--out", type=Path, help=f"default <registry>/live/{LIVE_FILE}")
    ap.add_argument("--servers", default=SERVERS,
                    help=f"local servers to probe each cycle, name=port,... on 127.0.0.1 (default {SERVERS}; "
                         "'' = none)")
    a = ap.parse_args(argv)
    try:
        servers = parse_servers(a.servers)
    except ValueError as e:
        ap.error(str(e))
    try:
        out = a.out or config.registry_root() / "live" / LIVE_FILE
    except config.ConfigError as e:
        print(f"[monitor] {e} (or pass --out)", file=sys.stderr)
        return 2
    out.parent.mkdir(parents=True, exist_ok=True)
    lock = acquire_lock(out.parent / LOCK_FILE, 0)
    if lock is None:
        print(f"[monitor] another monitor holds {out.parent / LOCK_FILE}; not starting", file=sys.stderr)
        return 3
    reader = Reader()
    print(f"[monitor] {', '.join(map(str, a.share))} -> {out}" + ("" if a.once else f" every {a.every:g} s"),
          flush=True)
    print("[monitor] servers: " + (", ".join(f"{n}=127.0.0.1:{p}" for n, p in servers) or "none"), flush=True)
    try:
        while True:
            t0 = time.time()
            snap = snapshot(a.share, reader, a.every, servers)
            try:
                C.write_json(out, snap)
            except OSError as e:                       # e.g. the page server's reader holds it: next cycle
                print(f"[monitor] {out} not written: {e}", flush=True)
            if a.once:
                break
            time.sleep(max(1.0, a.every - (time.time() - t0)))
    except KeyboardInterrupt:
        print("[monitor] stopped (Ctrl+C)")
    finally:
        lock.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
