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
"""
from __future__ import annotations

import json
import os
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from data_utils.oslock import acquire_lock  # noqa: E402

LOCK_WAIT_S = 5.0
SHOW = 200
TAIL_BYTES = 2 * 1024 * 1024
_lock = threading.Lock()


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
    with _lock:
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
    """Any line -> {t, actor, action, target, detail} (the old Jobs format included)."""
    if "t" in rec and "target" in rec:
        return {k: rec.get(k) for k in ("t", "actor", "action", "target", "detail")}
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
    """A label for an extra file without showing a path: its state folder's name (<state>/audit/portal.jsonl)."""
    p = Path(path)
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
