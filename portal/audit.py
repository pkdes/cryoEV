"""
The portal's one audit log: <state dir>/audit/portal.jsonl (outside the registry), append only, one JSON object per
line:
    {"t": UTC ISO time, "actor": portal id, "action": "person_added" | "cancel" | ..., "target": what it acted on,
     "detail": {...} or null}
Every writer (Admin's people actions, Jobs' cancel / release, later modules) goes through `audit()`: a thread lock
for this process plus an OS file lock (<file>.lock, data_utils/oslock.py) for other processes, so lines from
concurrent writers never interleave. A lock that can't be had in LOCK_WAIT_S raises OSError: never an unlocked
append. Never put a key, hash or session id in a line.

`read_audit()` returns the newest lines first, normalised to that schema; lines written before it (Jobs' first
format {time, actor, action, job, reason, result}) are read too: `time` -> t, `job` -> target, reason / result ->
detail.

`Writer` is the Gate's catch-all (portal/app.py, round 9 #17): one line per portal write, written by one worker
thread so a slow or locked log never blocks the event loop and a failed write never fails the request (logged).
Frequent routes (upload chunks, review saves and claims, gallery marks) are grouped per actor x action x target
(x status x via) in WINDOW_S windows: the first event is written at once, then one counter line with detail {from, to, n}
(n = every event in the window, the first included; only when n > 1; plus status and via) when the window closes, on
the next sweep, or at shutdown. A crash loses only the count, never the fact. Sign-ins: the first per person per day
per door per agent (in memory: a restart may log one more).

Agent tokens (round 10 #60): a request the Gate verified by bearer token carries detail.via = the token's agent on
its lines (the window key and the counter line too) and signs in on API_DOOR, so Admin shows "<agent> for
<person>" with the AI tag and never "unverified" (the portal checked the token).
"""
from __future__ import annotations

import importlib.util
import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from data_utils.oslock import acquire_lock  # noqa: E402

LOCK_WAIT_S = 5.0
SHOW = 200
TAIL_BYTES = 2 * 1024 * 1024
WINDOW_S = 600                   # frequent routes: one counter line per actor x action x target per 10 min
MAX_PENDING = 2000               # queued writes beyond this are dropped (logged): a stuck lock never eats memory
MAX_WINDOWS = 5000               # open counting windows; beyond this, new keys fold into OVERFLOW_KEY (still counted)
OVERFLOW_KEY = ("-", "other frequent actions (window table full)", None, None, None)
CLI_FILE = "agent_actions.jsonl"     # the laptop CLI's own log (self-declared identity), read via CJ_AUDIT_EXTRA
CLI_DOOR = "laptop CLI (unverified)"
CLI_ACTOR = "cli"                    # the actor of portal.auth CLI lines (a reserved person id: never a person)
DOOR_NAMES = {"key": "tailnet key", "access": "access"}   # this portal's door, by its --auth mode
API_DOOR = "API (verified)"                               # bearer-token requests (portal/app.py Gate)
_lock = threading.Lock()
log = logging.getLogger("cj.audit")


def actor_module():
    """The attribution helpers (data_utils/actor.py: who / via, label, strip_env). A normal package import since
    round 10; before, registry/actor.py was loaded by file path because `registry` could be registry/registry.py."""
    from data_utils import actor
    return actor


def audit_path(state_dir) -> Path:
    return Path(state_dir) / "audit" / "portal.jsonl"


def _clip(x, n=300):
    if isinstance(x, str):
        return x[:n]
    if isinstance(x, dict):
        return {str(k)[:60]: _clip(v, n) for k, v in list(x.items())[:20]}
    if isinstance(x, (list, tuple)):
        return [_clip(v, n) for v in list(x)[:20]]
    if x is None or isinstance(x, (bool, int, float)):
        return x
    return str(x)[:n]


def audit(state_dir, actor, action, target, detail=None) -> dict:
    """Append one line. -> the record written. Raises OSError when it can't be written (folder, file or lock):
    the caller decides; a write-ahead caller does nothing then."""
    rec = {"t": datetime.now(timezone.utc).isoformat(timespec="seconds"), "actor": _clip(str(actor or "-"), 80),
           "action": _clip(str(action), 60), "target": None if target is None else _clip(str(target), 120),
           "detail": _clip(detail)}
    line = json.dumps(rec, ensure_ascii=False) + "\n"
    path = audit_path(state_dir)
    with _lock:     # held across the OS-lock wait: callers run in worker threads and traffic is low (QC r9, overruled)
        path.parent.mkdir(parents=True, exist_ok=True)
        lk = acquire_lock(path.with_name(path.name + ".lock"), wait=LOCK_WAIT_S)
        if lk is None:
            raise OSError("the audit log is locked by another process")
        try:
            with open(path, "a", encoding="utf-8", newline="\n") as f:
                f.write(line)
                f.flush()
        finally:
            lk.close()
    return rec


def normalise(rec: dict) -> dict:
    """Any line -> {t, actor, action, target, detail} (the old Jobs format included). A top-level "via" (an agent
    acting for the actor) is kept as detail["via"], where the CLI writes it."""
    if "t" in rec and "target" in rec:
        out = {k: rec.get(k) for k in ("t", "actor", "action", "target", "detail")}
        if rec.get("via") and (out["detail"] is None or isinstance(out["detail"], dict)):
            out["detail"] = {**(out["detail"] or {}), "via": rec["via"]}
        return out
    detail = rec.get("detail")
    if detail is None and ("reason" in rec or "result" in rec):
        detail = {k: rec.get(k) for k in ("reason", "result") if rec.get(k) is not None}
    return {"t": rec.get("t") or rec.get("time"), "actor": rec.get("actor"), "action": rec.get("action"),
            "target": rec.get("target") if rec.get("target") is not None else rec.get("job"), "detail": detail}


def read_audit(state_dir, n=SHOW) -> tuple:
    """-> (the last n entries, newest first, normalised; problem or None). Reads the file's tail only; lines that
    are not JSON objects are skipped."""
    return read_file(audit_path(state_dir), n)


EXTRA_ENV = "CJ_AUDIT_EXTRA"


def extra_paths(env=None) -> list:
    """The other doors' audit files named in $CJ_AUDIT_EXTRA: absolute paths separated by ';'. Read only."""
    raw = (os.environ if env is None else env).get(EXTRA_ENV) or ""
    out = []
    for part in raw.split(";"):
        part = part.strip().strip('"')
        if part and Path(part).is_absolute():
            out.append(Path(part))
    return out


def door_label(path) -> str:
    """A label for an extra file without showing a path: its state folder's name (<state>/audit/portal.jsonl); the
    laptop CLI's file (CLI_FILE) is CLI_DOOR: its actor is whatever the caller declared."""
    p = Path(path)
    if p.name == CLI_FILE:
        return CLI_DOOR
    name = p.parent.parent.name if p.parent.name == "audit" else p.parent.name
    return (name or "other")[:40]


def _t_key(e) -> str:
    try:
        return datetime.fromisoformat(str(e.get("t"))).astimezone(timezone.utc).isoformat()
    except (TypeError, ValueError):
        return ""


def read_merged(state_dir, extras=None, n=SHOW) -> tuple:
    """-> (the last n entries of this door and the extra doors, newest first, each with a "door" key: "this door" or
    the extra file's label; list of problem notes). An unreadable extra file is a note, never an error."""
    own = audit_path(state_dir)
    entries, problems = [], []
    rows, prob = read_file(own, n)
    if prob:
        problems.append(prob)
    entries += [dict(e, door="this door") for e in rows]
    seen = {str(own.resolve())}
    for x in (extra_paths() if extras is None else [Path(e) for e in extras]):
        try:
            key = str(x.resolve())
        except OSError:
            key = str(x)
        if key in seen:
            continue
        seen.add(key)
        label = door_label(x)
        rows, prob = read_file(x, n)
        if not prob and not x.exists():
            prob = "missing"
        if prob:
            problems.append(f"The {label} log can't be read.")
        entries += [dict(e, door=label) for e in rows]
    entries.sort(key=_t_key, reverse=True)
    return entries[:n], problems


def read_file(path, n=SHOW) -> tuple:
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - TAIL_BYTES))
            data = f.read()
    except FileNotFoundError:
        return [], None
    except OSError as e:
        return [], f"The log can't be read ({type(e).__name__})."
    lines = data.decode("utf-8", "replace").splitlines()
    if size > TAIL_BYTES and lines:
        lines = lines[1:]                               # the first line may be cut
    out = []
    for ln in reversed(lines):
        try:
            rec = json.loads(ln)
        except ValueError:
            continue
        if isinstance(rec, dict):
            out.append(normalise(rec))
            if len(out) >= n:
                break
    return out, None


# ---------------------------------------------------------------- the Gate's writer (round 9 #17)
def _iso(ts) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).isoformat(timespec="seconds")


class Writer:
    """Audit lines from the event loop without blocking it: event() / signin() only touch memory and queue the write
    on one worker thread (never anyio's pool); the write is audit() inside try/except, a failure is logged and
    counted (self.failed), never raised. sweep() closes due windows (the portal calls it every minute), close()
    flushes every open window and waits for the queue (lifespan shutdown). drain() waits for the queue (tests)."""

    def __init__(self, state_dir, window_s=WINDOW_S, clock=time.time, max_pending=MAX_PENDING, write=None,
                 max_windows=MAX_WINDOWS):
        self.state_dir = state_dir
        self.window_s = float(window_s)
        self.clock = clock
        self.max_pending = int(max_pending)
        self.max_windows = int(max_windows)
        self._write_fn = write or audit
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cj-audit")
        self._mu = threading.Lock()
        self._pending = 0
        self._windows = {}           # (actor, action, target, status, via) -> {"start", "last", "n"}
        self._signins = {}           # (actor, door, via) -> "YYYY-MM-DD" (UTC) of the last sign-in line
        self.failed = 0
        self.dropped = 0
        self.closed = False

    # -- queue
    def _run(self, actor, action, target, detail):
        try:
            self._write_fn(self.state_dir, actor, action, target, detail)
        except Exception as e:                            # noqa: BLE001  never into the request
            self.failed += 1
            log.error("audit line not written (%s): %s %s", type(e).__name__, action, target)
        finally:
            with self._mu:
                self._pending -= 1

    def submit(self, actor, action, target, detail=None) -> bool:
        """Queue one line. -> False when it was dropped (queue full or writer closed)."""
        with self._mu:
            why = ("writer closed" if self.closed else
                   "queue full" if self._pending >= self.max_pending else None)
            if why:
                self.dropped += 1
            else:
                self._pending += 1
        if why:
            log.error("audit line dropped (%s): %s %s", why, action, target)
            return False
        try:
            self._pool.submit(self._run, actor, action, target, detail)
        except RuntimeError as e:                         # the pool was shut down under us
            with self._mu:
                self._pending -= 1
                self.dropped += 1
            log.error("audit line dropped (%s): %s %s", e, action, target)
            return False
        return True

    # -- events
    def event(self, actor, action, target=None, detail=None, frequent=False) -> None:
        """One portal action. frequent=True: the first per actor x action x target window is written at once, the
        rest are counted. An anonymous actor ("-") never has a target (a flood over many addresses is one window);
        past max_windows open windows, a new key is folded into OVERFLOW_KEY, which still counts."""
        if actor == "-":
            target = None
        if not frequent:
            self.submit(actor, action, target, detail)
            return
        now = self.clock()
        status = detail.get("status") if isinstance(detail, dict) else None
        via = detail.get("via") if isinstance(detail, dict) else None
        key = (actor, action, target, status, via)     # done and refused (and each agent) are counted apart
        closed = None
        with self._mu:
            if key not in self._windows and len(self._windows) >= self.max_windows:
                key = OVERFLOW_KEY                      # bounded memory; the fact and the count survive
                actor, action, target, detail = key[0], key[1], None, None
            w = self._windows.get(key)
            if w is not None and now - w["start"] < self.window_s:
                w["n"] += 1
                w["last"] = now
                return
            if w is not None:
                closed = self._windows.pop(key)
            self._windows[key] = {"start": now, "last": now, "n": 1}
        if closed:
            self._count(key, closed)
        self.submit(actor, action, target, detail)

    def _count(self, key, w) -> None:
        if w["n"] > 1:
            actor, action, target, status, via = key
            d = {"from": _iso(w["start"]), "to": _iso(w["last"]), "n": w["n"]}
            if status is not None:
                d["status"] = status
            if via is not None:
                d["via"] = via
            self.submit(actor, action, target, d)

    def sweep(self, now=None) -> int:
        """Close the windows that are due. -> how many closed."""
        now = self.clock() if now is None else now
        with self._mu:
            due = [(k, w) for k, w in self._windows.items() if now - w["start"] >= self.window_s]
            for k, _ in due:
                self._windows.pop(k, None)
        for k, w in due:
            self._count(k, w)
        return len(due)

    def flush(self) -> int:
        """Close every open window (shutdown). -> how many."""
        with self._mu:
            due = list(self._windows.items())
            self._windows.clear()
        for k, w in due:
            self._count(k, w)
        return len(due)

    def signin(self, actor, door, how=None, via=None) -> bool:
        """The first sign-in per person per (UTC) day per door per agent (via: the bearer token's agent). -> True
        when a line was queued."""
        if not actor:
            return False
        day = _iso(self.clock())[:10]
        with self._mu:
            if self._signins.get((actor, door, via)) == day:
                return False
            self._signins[(actor, door, via)] = day
        detail = {"door": door}
        if how:
            detail["how"] = how
        if via:
            detail["via"] = via
        return self.submit(actor, "signed in", None, detail)

    # -- life
    def drain(self, timeout=15.0) -> None:
        """Wait until every line queued so far is written (or failed)."""
        if not self.closed:
            self._pool.submit(lambda: None).result(timeout)

    def close(self, timeout=15.0) -> None:
        """Shutdown: flush the open windows, write the queue (up to `timeout` s), stop the worker; lines still queued
        then are cancelled (logged). Never raises, never waits on a stuck lock beyond `timeout`."""
        try:
            self.flush()
            self.drain(timeout)
        except Exception as e:                            # noqa: BLE001  e.g. TimeoutError: the log is locked
            log.error("audit flush at shutdown incomplete (%s): %d line(s) not written", type(e).__name__,
                      self._pending)
        with self._mu:
            self.closed = True
        self._pool.shutdown(wait=False, cancel_futures=True)
