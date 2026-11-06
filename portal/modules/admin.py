"""
Admin module (CryoEV portal, admin only): people and their key links, health at a glance, and the audit log.

    GET  /admin/                 People: everyone on the roster (id, role, active / revoked, added; never keys or
                                 hashes), the add form, a role form and a "Revoke..." link per person
    POST /admin/people/add       id + role -> a new key link, shown ONCE on the answer page (no-store). A refresh
                                 re-sends the form, which is refused ("already on the list"): the key is never shown
                                 again, never logged and never stored (the roster keeps its sha256 only, auth.py)
    POST /admin/people/role      id + role -> the roster entry's role (same key; the Gate re-reads the roster on
                                 every request, so it takes effect on that person's next request)
    GET  /admin/people/revoke    ?id= -> the confirmation page (a form; no script, no confirm())
    POST /admin/people/revoke    id + confirm=yes -> revoked: that person's sessions end at once
    GET  /admin/predict          Predict sessions: the sessions Predict offers (found by the scan of processed
                                 data, plus CJ_PREDICT_SOURCES), each with tiles, samples, usable yes / no + why,
                                 the name people see (editable) and shown / hidden (portal/predict_sources.py)
    POST /admin/predict/label    key + label ('' = the default name)
    POST /admin/predict/show     key + hidden=yes|no
    GET  /admin/health           services (tools/services.py status --json, in a subprocess), the GPU computers'
                                 watchers (<registry>/live/live.json), the backup (last commit of the backup clone,
                                 the scheduled task's last result)
    GET  /admin/audit            Recent actions: the last AUDIT_SHOW audit lines (every module), newest first
    GET  /admin/api/people | /admin/api/predict | /admin/api/health | /admin/api/audit     the same as JSON
    GET  /admin/admin.js         the key page's Copy button (a file: no inline script)

Rules:
  - Every POST: the Gate checks Origin (portal/app.py); here a form token bound to the session cookie
    (HMAC with a per-process secret) is checked as well. A portal restart makes open forms stale (reload).
  - Never leave the portal without an admin: demoting or revoking the last active admin is refused (yourself or
    anyone). Demoting or revoking yourself while other admins remain is allowed (you are signed out / sent home).
  - The roster is the private file auth.py reads (cfg.roster), in auth.py's format. Add, role change and revoke
    are each one read-modify-write inside auth.roster_lock (a thread lock + an OS lock on <roster>.lock, which the
    `python -m portal.auth` tool takes too), written atomically with os.replace retried briefly; the file's
    (mtime, size) must change (the Gate's Roster cache keys on it) or it raises. Any OSError (lock or write):
    "Nothing was changed", no temp file left.
  - After a change the page shows a one-time message bound to the session (?f=<token>); a bare URL shows none.
  - The key link points at the first --origin, else the tailnet address (auth.default_base), else the request's
    Origin reduced to scheme://host[:port].
  - Audit: portal/audit.py (shared with Jobs), <state dir>/audit/portal.jsonl. `audit(ctx, actor, action, target,
    detail)` here is a thin wrapper. The Recent actions page shows the last AUDIT_SHOW lines of every module.
  - Predict sessions: overrides in <state dir>/predict_sources.json (predict_sources.set_override: its own thread +
    OS lock, atomic write); one audit line per change (predict_source_renamed / _hidden / _shown, target = the
    source key). The page rescans the folders (in a worker thread); Predict uses the scan cached for a minute.
  - Health never blocks the event loop (threads), and every source may be missing: "unknown", never an error page.
    Backup clone: env CJ_BACKUP_REPO, else "backup_repo" in <state dir>/admin.json. Task: BACKUP_TASK.
"""
from __future__ import annotations

import asyncio
import csv
import hashlib
import hmac
import html
import io
import json
import logging
import os
import re
import secrets
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, quote, urlsplit

from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import RedirectResponse, Response
from starlette.routing import Route

from portal import audit as PA
from portal import auth as A
from portal import predict_sources as PSRC

PREFIX = "/admin"
ROLES = ("admin",)

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
TEMPLATE = HERE.parent / "templates" / "admin.html"
SERVICES_SCRIPT = REPO / "tools" / "services.py"
SERVICES_TIMEOUT_S = 25.0
GIT = "git"
SCHTASKS = "schtasks"
BACKUP_TASK = "cryoai-state-backup"
SUBPROC_TIMEOUT_S = 10.0
HEALTH_TTL_S = 10.0           # one health read serves every request for 10 s
BACKUP_OLD_S = 36 * 3600      # a daily backup older than this is late
STALE_FACTOR = 3              # live.json older than 3 x interval_s is stale (as home and Jobs)
AUDIT_SHOW = PA.SHOW
REPLACE_TRIES = 20          # os.replace retries, 50 ms apart
SIG_TRIES = 50              # roster rewrites until its (mtime, size) changes, 20 ms apart
FLASH_S = 120               # a message for the page after a POST, taken once
MAX_FORM = 8 * 1024
ROLE_WORD = {"admin": "Admin", "advanced": "Advanced", "basic": "Basic"}
ROLE_DESC = {"admin": "everything, including people and health",
             "advanced": "jobs, history, training (when it opens)",
             "basic": "their own uploads, predictions and results"}
ACTION_WORD = {"person_added": "added", "role_changed": "changed the role of", "person_revoked": "revoked",
               "cancel": "cancel", "release": "release", "predict_source_renamed": "renamed the Predict session",
               "predict_source_hidden": "hid the Predict session", "predict_source_shown": "showed the Predict session"}

esc = html.escape
log = logging.getLogger("cj.admin")


# ---------------------------------------------------------------- small helpers
def _d(x) -> dict:
    return x if isinstance(x, dict) else {}


def _s(x) -> str | None:
    return x if isinstance(x, str) and x else None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def dur(v) -> str:
    """Seconds -> '40 s' / '12 min' / '3 h 5 min' / '2 d 4 h' (as Jobs and the home page)."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return "?"
    v = max(0, round(v))
    if v < 60:
        return f"{v} s"
    m = round(v / 60)
    if m < 60:
        return f"{m} min"
    h, m = divmod(m, 60)
    if h < 48:
        return f"{h} h" + (f" {m} min" if m else "")
    return f"{h // 24} d {h % 24} h"


def parse_time(s) -> float | None:
    """ISO time -> epoch seconds; a time without a zone is this machine's local time (roster `added`)."""
    if not isinstance(s, str) or not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def parse_utc(s) -> float | None:
    """live.json generated_utc -> epoch seconds (a time without a zone is UTC)."""
    try:
        t = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return (t if t.tzinfo else t.replace(tzinfo=timezone.utc)).timestamp()


def when(ts, now=None) -> str:
    """'2026-11-05 14:03 (3 h ago)' in this machine's local time."""
    if ts is None:
        return "unknown"
    now = time.time() if now is None else now
    out = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
    return out + (f" ({dur(now - ts)} ago)" if 0 <= now - ts < 14 * 86400 else "")


def no_window() -> dict:
    return {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}


def run(argv, timeout=SUBPROC_TIMEOUT_S, env=None) -> tuple:
    """-> (exit code or None, stdout, problem or None). Never raises (missing tool, timeout, bad bytes)."""
    try:
        p = subprocess.run(argv, capture_output=True, timeout=timeout, cwd=str(REPO), env=env,
                           stdin=subprocess.DEVNULL, **no_window())
    except FileNotFoundError:
        return None, "", "not found"
    except subprocess.TimeoutExpired:
        return None, "", f"no answer in {timeout:.0f} s"
    except OSError as e:
        return None, "", f"could not run ({type(e).__name__})"
    out = p.stdout.decode("utf-8", "replace") if p.stdout else ""
    return p.returncode, out, None


# ---------------------------------------------------------------- audit log (portal/audit.py, shared)
def audit_path(cfg) -> Path:
    return PA.audit_path(cfg.state_dir)


def audit(ctx, actor, action, target, detail=None) -> dict:
    """portal.audit.audit for a module ctx (or anything with .cfg.state_dir, or a Config). Raises OSError."""
    cfg = getattr(ctx, "cfg", ctx)
    return PA.audit(cfg.state_dir, actor, action, target, detail)


def read_audit(cfg, n=AUDIT_SHOW) -> tuple:
    return PA.read_audit(cfg.state_dir, n)


# ---------------------------------------------------------------- roster (auth.py's file format and lock)
roster_lock = A.roster_lock          # module attribute so a test can show the race it prevents


def load_roster(path) -> dict:
    """As auth._load: a missing roster is empty; an unreadable one raises ValueError (never overwritten)."""
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"schema": "cj_roster/1", "people": []}
    except (OSError, ValueError) as e:
        raise ValueError(f"roster not readable ({type(e).__name__})")
    if not isinstance(doc, dict) or not isinstance(doc.get("people"), list):
        raise ValueError("roster has no people list")
    return doc


def _os_replace(src, dst):
    os.replace(src, dst)


def _replace(tmp: Path, path: Path) -> None:
    """os.replace, retried briefly: on Windows a reader holding the roster makes it fail for a moment."""
    for i in range(REPLACE_TRIES):
        try:
            return _os_replace(tmp, path)
        except PermissionError:
            if i == REPLACE_TRIES - 1:
                raise
            time.sleep(0.05)


def write_roster(path, doc: dict) -> None:
    """Atomic write (temp file + os.replace, retried) that is made sure to change the file's (mtime, size)
    signature: the Roster cache re-reads on it, and a role change can keep the size (admin <-> basic) inside one
    clock tick. Raises OSError after its tries; the temp file never stays behind."""
    path = Path(path)
    try:
        st = path.stat()
        before = (st.st_mtime_ns, st.st_size)
    except OSError:
        before = None
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    data = json.dumps(doc, indent=1)
    try:
        for _ in range(SIG_TRIES):
            tmp.write_text(data, encoding="utf-8")
            _replace(tmp, path)
            st = path.stat()
            if (st.st_mtime_ns, st.st_size) != before:
                return
            time.sleep(0.02)
        raise OSError("the roster file did not change on disk")
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def is_admin(p) -> bool:
    return A.Roster.usable(p) and p.get("role") == "admin"


def public_person(p: dict) -> dict:
    """The only fields that ever leave this module: never key_sha256 (or anything else in the entry)."""
    return {"id": str(p.get("id")), "role": p.get("role") if p.get("role") in A.ROLES else None,
            "state": "revoked" if p.get("revoked") else ("active" if A.Roster.usable(p) else "not usable"),
            "added": _s(p.get("added")), "revoked": _s(p.get("revoked")), "role_changed": _s(p.get("role_changed"))}


def people(path) -> list:
    doc = load_roster(path)
    rows = [public_person(p) for p in doc["people"] if isinstance(p, dict) and _s(p.get("id"))]
    rows.sort(key=lambda r: (r["state"] != "active", r["id"]))
    return rows


class Refused(Exception):
    """A request the roster rules refuse; the message is shown to the admin as is."""

    def __init__(self, msg, status=409):
        super().__init__(msg)
        self.status = status


BAD_ID = "An id is 1 to 40 lower-case letters, digits, '_', '.' or '-', starting with a letter or digit."


def _check_id(pid) -> None:
    if not isinstance(pid, str) or not A.ID_RE.fullmatch(pid):
        raise Refused(BAD_ID, 400)


def _admins_left(doc, pid) -> int:
    return sum(1 for p in doc["people"] if isinstance(p, dict) and is_admin(p) and p.get("id") != pid)


def _active(doc, pid) -> list:
    return [p for p in doc["people"] if isinstance(p, dict) and p.get("id") == pid and not p.get("revoked")]


def add_person(path, pid: str, role: str) -> str:
    """-> the new key (the caller shows it once and forgets it). Same entry as auth.add_key; refuses an id that is
    already active. Raises Refused, ValueError (unreadable roster) or OSError (lock or write: nothing changed)."""
    _check_id(pid)
    if role not in A.ROLES:
        raise Refused("Pick a role: Admin, Advanced or Basic.", 400)
    with roster_lock(path):
        doc = load_roster(path)
        if _active(doc, pid):
            raise Refused(f"{pid} is already on the list. To give them a new key link, revoke them first, then add "
                          "them again.")
        key = secrets.token_urlsafe(A.KEY_BYTES)
        entry = {"id": pid, "role": role, "key_sha256": A.sha256_hex(key),
                 "added": datetime.now().isoformat(timespec="seconds")}
        doc["people"] = [p for p in doc["people"] if not (isinstance(p, dict) and p.get("id") == pid)] + [entry]
        write_roster(path, doc)
        return key


def set_role(path, pid: str, role: str) -> str:
    """Change an active person's role (same key, sessions kept). -> the old role."""
    _check_id(pid)
    if role not in A.ROLES:
        raise Refused("Pick a role: Admin, Advanced or Basic.", 400)
    with roster_lock(path):
        doc = load_roster(path)
        hits = [p for p in doc["people"] if isinstance(p, dict) and p.get("id") == pid]
        if not hits:
            raise Refused(f"{pid} is not on the list.", 404)
        p = hits[-1]
        if p.get("revoked"):
            raise Refused(f"{pid} is revoked. Add them again to give them a role.")
        old = p.get("role")
        if old == role:
            raise Refused(f"{pid} is already {ROLE_WORD[role]}.")
        if is_admin(p) and role != "admin" and _admins_left(doc, pid) == 0:
            raise Refused(f"{pid} is the last admin. Make someone else an admin first.")
        p["role"] = role
        p["role_changed"] = datetime.now().isoformat(timespec="seconds")
        write_roster(path, doc)
        return old


def revoke_person(path, pid: str) -> str:
    """Revoke an active person (as auth.revoke). -> their role. Refuses the last active admin."""
    _check_id(pid)
    with roster_lock(path):
        doc = load_roster(path)
        hits = _active(doc, pid)
        if not hits:
            raise Refused(f"{pid} is not on the list, or already revoked.", 404)
        if any(is_admin(p) for p in hits) and _admins_left(doc, pid) == 0:
            raise Refused(f"{pid} is the last admin. Make someone else an admin first.")
        stamp = datetime.now().isoformat(timespec="seconds")
        for p in hits:
            p["revoked"] = stamp
        write_roster(path, doc)
        return str(hits[-1].get("role"))


def revoke_check(path, pid: str) -> str | None:
    """-> why revoking `pid` would be refused, or None."""
    if not isinstance(pid, str) or not A.ID_RE.fullmatch(pid):
        return BAD_ID
    try:
        doc = load_roster(path)
    except ValueError as e:
        return str(e)
    hits = _active(doc, pid)
    if not hits:
        return f"{pid} is not on the list, or already revoked."
    if any(is_admin(p) for p in hits) and _admins_left(doc, pid) == 0:
        return f"{pid} is the last admin. Make someone else an admin first."
    return None


def why_not_written(e: OSError) -> str:
    if "another process" in str(e):
        return "Nothing was changed: the roster is being changed elsewhere right now. Try again."
    return f"Nothing was changed: the roster could not be written ({type(e).__name__}). Try again."


# ---------------------------------------------------------------- key link address
_tailnet: dict = {}


def tailnet_base() -> str | None:
    """auth.default_base() when it finds this PC's tailnet name (https://<name>:8446), else None. Asked once."""
    if "base" not in _tailnet:
        try:
            b = A.default_base()
        except Exception:
            b = None
        _tailnet["base"] = b if b and b.startswith("https://") else None
    return _tailnet["base"]


def link_base(cfg, request_origin) -> str:
    """Where the key link points: the first configured --origin, else the tailnet address, else the request's
    Origin (which the Gate checked) reduced to scheme://host[:port]; '' if none (a relative link)."""
    origins = tuple(getattr(cfg, "origins", ()) or ())
    if origins and _s(origins[0]):
        return origins[0].rstrip("/")
    t = tailnet_base()
    if t:
        return t.rstrip("/")
    try:
        u = urlsplit(request_origin or "")
        host, port = u.hostname, u.port
    except ValueError:
        return ""
    if u.scheme not in ("http", "https") or not host:
        return ""
    host = f"[{host}]" if ":" in host else host
    return f"{u.scheme}://{host}" + (f":{port}" if port else "")


# ---------------------------------------------------------------- flash messages (only after a real POST)
_flash: dict = {}
_flash_lock = threading.Lock()


def flash_put(session: str, text: str) -> str:
    tok = secrets.token_urlsafe(12)
    now = time.monotonic()
    with _flash_lock:
        for k in [k for k, v in _flash.items() if v[2] < now]:
            del _flash[k]
        _flash[tok] = (hashlib.sha256(session.encode()).hexdigest(), text, now + FLASH_S)
    return tok


def flash_take(session: str, tok) -> str | None:
    if not isinstance(tok, str) or not tok:
        return None
    with _flash_lock:
        rec = _flash.get(tok)
        if not rec or rec[2] < time.monotonic():
            return None
        if not hmac.compare_digest(rec[0], hashlib.sha256(session.encode()).hexdigest()):
            return None
        del _flash[tok]
        return rec[1]


# ---------------------------------------------------------------- health
def services_status() -> dict:
    """tools/services.py status --json in a subprocess (its own config: CRYOEV_SERVICES_CONFIG or the default)."""
    script = Path(SERVICES_SCRIPT)
    if not script.is_file():
        return {"ok": False, "rows": [], "problem": "The service tool is not on this computer."}
    code, out, prob = run([sys.executable, str(script), "status", "--json"], timeout=SERVICES_TIMEOUT_S)
    if prob:
        return {"ok": False, "rows": [], "problem": f"The service tool did not answer ({prob})."}
    if code == 2 and "no config" in out:
        return {"ok": False, "rows": [], "problem": "Services are not set up on this computer (no services config)."}
    if code != 0:
        return {"ok": False, "rows": [], "problem": f"The service tool failed (exit {code})."}
    try:
        rows = json.loads(out)
        if not isinstance(rows, list):
            raise ValueError
    except ValueError:
        return {"ok": False, "rows": [], "problem": "The service tool's answer could not be read."}
    keep = ("name", "enabled", "state", "port", "listening", "health", "uptime_s")
    return {"ok": True, "rows": [{k: _d(r).get(k) for k in keep} for r in rows if _s(_d(r).get("name"))],
            "problem": None}


SERVICE_WORD = {"running": ("Running", "ok"), "running (not listening)": ("Running, not answering", "warn"),
                "port busy (not ours)": ("Running (started by hand)", "ok"), "stale pid file": ("Stopped", "warn"),
                "stopped": ("Stopped", "warn"), "disabled": ("Turned off", "off")}


def watchers(registry, now=None) -> dict:
    """The GPU computers' watchers from <registry>/live/live.json (jobs/monitor.py)."""
    now = time.time() if now is None else now
    if registry is None:
        return {"ok": False, "queues": [], "age_s": None, "stale": True, "problem": "No registry folder is set."}
    p = Path(registry) / "live" / "live.json"
    doc, prob = None, None
    for i in range(3):                                   # monitor.py replaces the file; a read can meet it
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
            prob = None
            break
        except FileNotFoundError:
            prob = "No status snapshot (the job monitor is not running)."
            break
        except (OSError, ValueError):
            prob = "The status snapshot could not be read."
            time.sleep(0.05 * (i + 1))
    if not isinstance(doc, dict):
        return {"ok": False, "queues": [], "age_s": None, "stale": True,
                "problem": prob or "The status snapshot could not be read."}
    t = parse_utc(doc.get("generated_utc"))
    every = doc.get("interval_s") if isinstance(doc.get("interval_s"), (int, float)) else 30
    age = None if t is None else max(0.0, now - t)
    queues = []
    for qi, q in enumerate(doc.get("queues") or []):
        q, w = _d(q), _d(_d(q).get("watcher"))
        queues.append({"machine": _s(w.get("machine")) or f"GPU computer {qi + 1}",
                       "reach": _s(q.get("reach")) or "unknown", "display": _s(w.get("display")) or "unknown",
                       "age_s": w.get("age_s") if isinstance(w.get("age_s"), (int, float)) else None,
                       "running": _s(_d(q.get("running")).get("job"))})
    return {"ok": True, "queues": queues, "age_s": None if age is None else round(age),
            "stale": age is None or age > STALE_FACTOR * every, "problem": None}


WATCH_WORD = {"alive": ("Working normally", "ok"), "paused": ("Paused", "warn"),
              "waiting for GPU": ("Waiting for the GPU to be free", "ok"), "waiting": ("Waiting for a job", "ok"),
              "stale": ("Status updates have stopped", "warn"), "no watcher": ("Not set up to run jobs", "warn"),
              "stopped": ("Stopped", "warn"), "unknown": ("Status unknown", "warn")}


def backup_repo(cfg) -> Path | None:
    env = os.environ.get("CJ_BACKUP_REPO")
    if env:
        return Path(env)
    try:
        doc = json.loads((Path(cfg.state_dir) / "admin.json").read_text(encoding="utf-8"))
        val = _s(_d(doc).get("backup_repo"))
        return Path(val) if val else None
    except (OSError, ValueError):
        return None


def backup_commit(cfg, now=None) -> dict:
    """The backup clone's last commit: {configured, time, age_s, late, subject, problem}."""
    now = time.time() if now is None else now
    repo = backup_repo(cfg)
    if repo is None:
        return {"configured": False, "time": None, "age_s": None, "late": None, "subject": None,
                "problem": "Unknown: the backup folder is not set (CJ_BACKUP_REPO)."}
    if not repo.is_dir():
        return {"configured": True, "time": None, "age_s": None, "late": None, "subject": None,
                "problem": "Unknown: the backup folder is not on this computer."}
    code, out, prob = run([GIT, "-c", "core.fsmonitor=", "-C", str(repo), "log", "-1", "--format=%cI%x09%s"])
    if prob or code != 0 or not out.strip():
        why = prob or ("no commits yet" if code == 0 else "not a git folder, or git failed")
        return {"configured": True, "time": None, "age_s": None, "late": None, "subject": None,
                "problem": f"Unknown: the backup folder could not be read ({why})."}
    stamp, _, subject = out.strip().partition("\t")
    t = parse_time(stamp)
    age = None if t is None else max(0.0, now - t)
    return {"configured": True, "time": stamp or None, "age_s": None if age is None else round(age),
            "late": None if age is None else age > BACKUP_OLD_S, "subject": subject[:120] or None, "problem": None}


def parse_schtasks_csv(text: str) -> dict | None:
    """`schtasks /query /tn X /v /fo csv` -> {last_run, last_result, next_run, status} (English headers), or None."""
    try:
        rows = list(csv.reader(io.StringIO(text.strip())))
    except csv.Error:
        return None
    if len(rows) < 2:
        return None
    head = [h.strip() for h in rows[0]]
    row = next((r for r in rows[1:] if len(r) == len(head) and r != rows[0]), None)
    if row is None:
        return None
    rec = dict(zip(head, (c.strip() for c in row)))
    if "Last Result" not in rec:
        return None
    return {"last_run": rec.get("Last Run Time") or None, "last_result": rec.get("Last Result") or None,
            "next_run": rec.get("Next Run Time") or None,
            "status": rec.get("Scheduled Task State") or rec.get("Status") or None}


def backup_task() -> dict:
    if os.name != "nt":
        return {"found": False, "problem": "Scheduled tasks are only checked on Windows."}
    code, out, prob = run([SCHTASKS, "/query", "/tn", BACKUP_TASK, "/v", "/fo", "csv"])
    if prob:
        return {"found": False, "problem": f"Could not ask Windows about the backup task ({prob})."}
    if code != 0:
        return {"found": False, "problem": "The backup task is not set up on this computer."}
    rec = parse_schtasks_csv(out)
    if rec is None:
        return {"found": False, "problem": "The backup task's details could not be read."}
    res = rec["last_result"]
    ok = None if res is None else res in ("0", "0x0")
    never = res in ("267011", "0x41303")                 # SCHED_S_TASK_HAS_NOT_RUN
    return {"found": True, **rec, "ok": None if never else ok, "never": never, "problem": None}


_health_lock = threading.Lock()
_health_cache: dict = {}


def health(cfg) -> dict:
    """Everything the Health page shows (schema cj_admin_health/1). Blocking: call it off the event loop.
    One check at a time: requests that miss the cache while a check runs wait for it and share its answer."""
    key = (str(getattr(cfg, "registry", None)), str(cfg.state_dir))
    with _health_lock:
        hit = _health_cache.get(key)
        if hit and 0 <= time.monotonic() - hit[0] < HEALTH_TTL_S:
            return hit[1]
        out = _check_health(cfg)
        _health_cache[key] = (time.monotonic(), out)
        return out


def _check_health(cfg) -> dict:
    now = time.time()
    with ThreadPoolExecutor(3) as ex:                    # the subprocesses run side by side
        fs = ex.submit(services_status)
        fc = ex.submit(backup_commit, cfg, now)
        ft = ex.submit(backup_task)
        out = {"schema": "cj_admin_health/1", "generated_utc": utc_now(),
               "services": _safe(fs, {"ok": False, "rows": [], "problem": "The service check failed."}),
               "watchers": _safe_call(watchers, getattr(cfg, "registry", None), now,
                                      default={"ok": False, "queues": [], "age_s": None, "stale": True,
                                               "problem": "The status snapshot could not be read."}),
               "backup": {"commit": _safe(fc, {"configured": None, "problem": "Unknown: the check failed."}),
                          "task": _safe(ft, {"found": False, "problem": "The backup task check failed."})}}
    return out


def _safe(fut, default):
    try:
        return fut.result()
    except Exception as e:                              # a health source must never take the page down
        log.warning("admin health: %s", type(e).__name__)
        return default


def _safe_call(fn, *a, default):
    try:
        return fn(*a)
    except Exception as e:
        log.warning("admin health: %s", type(e).__name__)
        return default


# ---------------------------------------------------------------- rendering (everything escaped here)
def chip(label, tone) -> str:
    return f'<span class="cj-admin-chip cj-admin-t-{esc(tone)}">{esc(label)}</span>'


def subnav(active: str) -> str:
    items = (("people", "People", PREFIX + "/"), ("predict", "Predict sessions", PREFIX + "/predict"),
             ("health", "Health", PREFIX + "/health"),
             ("audit", "Recent actions", PREFIX + "/audit"))
    cur = ' aria-current="page"'
    links = "".join(f'<a href="{href}"{cur if k == active else ""}>{esc(name)}</a>' for k, name, href in items)
    return f'<nav class="cj-admin-sub" aria-label="Admin sections">{links}</nav>'


def note(text, tone="ok") -> str:
    role = "alert" if tone == "warn" else "status"
    return f'<p class="cj-admin-note cj-admin-t-{esc(tone)}" role="{role}">{esc(text)}</p>'


def role_select(name, selected, idattr) -> str:
    opts = "".join(f'<option value="{r}"{" selected" if r == selected else ""}>{ROLE_WORD[r]}</option>'
                   for r in A.ROLES)
    return f'<select class="cj-admin-in" name="{name}" id="{esc(idattr)}">{opts}</select>'


def hidden_token(tok) -> str:
    return f'<input type="hidden" name="t" value="{esc(tok)}">'


def render_people(rows, me, tok, msg_html="") -> str:
    active = [r for r in rows if r["state"] != "revoked"]
    gone = [r for r in rows if r["state"] == "revoked"]
    n_admin = sum(1 for r in active if r["role"] == "admin" and r["state"] == "active")

    def card(r, i):
        you = r["id"] == me
        name = esc(r["id"]) + (' <span class="cj-mut">(you)</span>' if you else "")
        role = r["role"] or "?"
        tone = {"admin": "acc", "advanced": "ok", "basic": "off"}.get(role, "warn")
        added = parse_time(r["added"])
        meta = f'Added {esc(when(added))}' if added is not None else "Added: unknown"
        if r["state"] == "revoked":
            rt = parse_time(r["revoked"])
            return (f'<li class="cj-admin-person"><div class="cj-admin-top"><b class="cj-wrap">{name}</b>'
                    f'{chip(ROLE_WORD.get(role, role), "off")}</div>'
                    f'<p class="cj-mut cj-admin-meta">Revoked {esc(when(rt))}. {meta}.</p></li>')
        last = role == "admin" and n_admin <= 1
        fid = f"cj-admin-r{i}"
        role_form = (f'<form class="cj-admin-row" method="post" action="{PREFIX}/people/role">'
                     f'{hidden_token(tok)}<input type="hidden" name="id" value="{esc(r["id"])}">'
                     f'<label class="cj-admin-lbl" for="{fid}">Role</label>{role_select("role", role, fid)}'
                     f'<button class="cj-admin-btn" type="submit">Change role</button></form>')
        rev = (f'<a class="cj-admin-link cj-admin-danger" href="{PREFIX}/people/revoke?id={quote(r["id"])}">'
               f'Revoke{" (you)" if you else ""}…</a>')
        lastnote = ('<p class="cj-mut cj-admin-meta">The last admin: can\'t be demoted or revoked until someone '
                    'else is an admin.</p>' if last else "")
        bad = ('<p class="cj-admin-note cj-admin-t-warn">This entry has no usable key or a bad role: it can\'t '
               'sign in.</p>' if r["state"] != "active" else "")
        return (f'<li class="cj-admin-person"><div class="cj-admin-top"><b class="cj-wrap">{name}</b>'
                f'{chip(ROLE_WORD.get(role, role), tone)}</div>'
                f'<p class="cj-mut cj-admin-meta">{meta}.</p>{bad}{lastnote}'
                + ("" if last else f'<div class="cj-admin-acts">{role_form}{rev}</div>') + '</li>')

    roles = "".join(f"<li><b>{ROLE_WORD[r]}</b>: {esc(ROLE_DESC[r])}</li>" for r in A.ROLES)
    add = (f'<section class="cj-card cj-admin-add" aria-labelledby="cj-admin-h-add">'
           f'<h2 id="cj-admin-h-add">Add a person</h2>'
           f'<form class="cj-admin-form" method="post" action="{PREFIX}/people/add">{hidden_token(tok)}'
           f'<label class="cj-admin-lbl" for="cj-admin-id">Id <span class="cj-mut">(lower-case, e.g. '
           f'jsmith)</span></label>'
           f'<input class="cj-admin-in" id="cj-admin-id" name="id" required maxlength="40" '
           f'pattern="[a-z0-9][a-z0-9_.\\-]{{0,39}}" autocomplete="off" autocapitalize="none" spellcheck="false">'
           f'<label class="cj-admin-lbl" for="cj-admin-role">Role</label>{role_select("role", "basic", "cj-admin-role")}'
           f'<button class="cj-admin-btn cj-admin-primary" type="submit">Add and make a key link</button></form>'
           f'<ul class="cj-mut cj-admin-roles">{roles}</ul>'
           f'<p class="cj-mut cj-admin-meta">The key link is shown once, on the next page. Send it to the person '
           f'privately.</p></section>')
    act = ("".join(card(r, i) for i, r in enumerate(active)) or
           '<li class="cj-mut">Nobody yet.</li>')
    out = (f'<h1>People</h1><p class="cj-mut">Who can sign in to the portal, and what they can open.</p>{msg_html}'
           f'<div class="cj-admin-cols">{add}<section aria-labelledby="cj-admin-h-ppl">'
           f'<h2 class="cj-sec" id="cj-admin-h-ppl">On the list <span class="cj-admin-n">{len(active)}</span></h2>'
           f'<ul class="cj-admin-people">{act}</ul>')
    if gone:
        out += (f'<details class="cj-admin-gone"><summary>Revoked ({len(gone)})</summary>'
                f'<ul class="cj-admin-people">{"".join(card(r, 1000 + i) for i, r in enumerate(gone))}</ul>'
                f'</details>')
    return out + "</section></div>"


def render_key(pid, role, link) -> str:
    return (f'<h1>Key link for {esc(pid)}</h1>'
            f'{note(f"{pid} was added as {ROLE_WORD[role]}.")}'
            f'<section class="cj-card cj-admin-key" aria-labelledby="cj-admin-h-key">'
            f'<h2 id="cj-admin-h-key">Send this link to {esc(pid)}, privately</h2>'
            f'<label class="cj-admin-lbl" for="cj-admin-link">Key link</label>'
            f'<textarea class="cj-admin-in cj-admin-linkbox" id="cj-admin-link" readonly rows="3" '
            f'spellcheck="false">{esc(link)}</textarea>'
            f'<div class="cj-admin-acts"><button class="cj-admin-btn cj-admin-primary" type="button" '
            f'data-cj-admin-copy="cj-admin-link" hidden>Copy link</button>'
            f'<span class="cj-mut" id="cj-admin-copied" role="status"></span></div>'
            f'<ul class="cj-admin-warnlist">'
            f'<li><b>Shown only now.</b> It is not stored anywhere; if it is lost, revoke {esc(pid)} and add them '
            f'again.</li>'
            f'<li>Anyone who has this link can sign in as {esc(pid)}. They open it once on each device.</li>'
            f'</ul></section>'
            f'<p><a class="cj-admin-link" href="{PREFIX}/">Back to People</a></p>')


def render_revoke(pid, me, why, tok) -> str:
    you = pid == me
    if why:
        return (f'<h1>Revoke {esc(pid)}</h1>{note(why, "warn")}'
                f'<p><a class="cj-admin-link" href="{PREFIX}/">Back to People</a></p>')
    extra = " That is you: you will be signed out too." if you else ""
    return (f'<h1>Revoke {esc(pid)}?</h1>'
            f'<section class="cj-card cj-admin-confirm" aria-labelledby="cj-admin-h-rev">'
            f'<h2 id="cj-admin-h-rev">What happens</h2>'
            f'<p>{esc(pid)} is signed out on every device at once, and their key link stops working.{esc(extra)}</p>'
            f'<p class="cj-mut">This can\'t be undone. To let them back in, add them again: they get a new key '
            f'link.</p>'
            f'<form class="cj-admin-acts" method="post" action="{PREFIX}/people/revoke">{hidden_token(tok)}'
            f'<input type="hidden" name="id" value="{esc(pid)}"><input type="hidden" name="confirm" value="yes">'
            f'<button class="cj-admin-btn cj-admin-dangerbtn" type="submit">Yes, revoke {esc(pid)}</button>'
            f'<a class="cj-admin-link" href="{PREFIX}/">Cancel</a></form></section>')


def session_name(r) -> str:
    return str(r["session"])


def public_source(r) -> dict:
    """A merged row for the JSON view (admin only: paths included)."""
    return {"key": r["key"], "session": session_name(r), "label": r["label"], "default_label": r["default_label"],
            "origin": r["origin"], "tiles": r["tiles"], "samples": r["samples"], "nm_min": r["nm_min"],
            "nm_max": r["nm_max"], "compatible": r["compatible"], "reason": r["reason"], "hidden": r["hidden"],
            "dir": str(r["dir"])}


def nm_text(r) -> str:
    a, b = r.get("nm_min"), r.get("nm_max")
    if a is None:
        return ""
    return f"{a:.3g} nm/px" if abs(b - a) < 5e-4 else f"{a:.3g}–{b:.3g} nm/px"


def render_predict(m: dict, tok, msg_html="") -> str:
    locked = bool(m.get("overrides_error"))
    top = []
    if not m["roots"]:
        top.append(note("No folder is scanned (CJ_PREDICT_SCAN_ROOTS is off, or the processed-data folder was not "
                        "found): only sessions listed by hand (CJ_PREDICT_SOURCES) are offered.", "warn"))
    for r in m["roots"]:
        if r["problem"]:
            top.append(note(f"Scan folder {r['root']}: {r['problem']}.", "warn"))
    if m.get("list_error"):
        top.append(note(f"The hand list can't be read, so Predict offers no sessions: {m['list_error']}", "warn"))
    if locked:
        top.append(note(f"The Predict session settings can't be read ({m['overrides_error']}), so Predict offers "
                        "no sessions and nothing here can be changed. Fix or remove the file in the portal's state "
                        "folder.", "warn"))

    def row(r, i):
        sid = session_name(r)
        origin = {"found": "found", "listed": "listed by hand", "both": "found, listed by hand"}[r["origin"]]
        nm = nm_text(r)
        where = (f'<td data-l="Session"><b class="cj-wrap">{esc(sid)}</b>'
                 f'<span class="cj-mut cj-admin-psmeta">{esc(origin)}{" · " + esc(nm) if nm else ""}</span>'
                 f'<span class="cj-admin-path">{esc(str(r["dir"]))}</span></td>')
        fid = f"cj-admin-ps{i}"
        dflt = (f'<span class="cj-mut cj-admin-psmeta">Default: {esc(r["default_label"])}</span>'
                if r["label"] != r["default_label"] else "")
        if locked:
            name = f'<td data-l="Name people see"><span class="cj-wrap">{esc(r["label"])}</span>{dflt}</td>'
        else:
            name = (f'<td data-l="Name people see"><form class="cj-admin-psform" method="post" '
                    f'action="{PREFIX}/predict/label">{hidden_token(tok)}'
                    f'<input type="hidden" name="key" value="{esc(r["key"])}">'
                    f'<label class="cj-admin-sr" for="{fid}">Name people see for session {esc(sid)}</label>'
                    f'<input class="cj-admin-in" id="{fid}" name="label" value="{esc(r["label"])}" '
                    f'maxlength="{PSRC.LABEL_MAX}" placeholder="{esc(r["default_label"])}" autocomplete="off">'
                    f'<button class="cj-admin-btn" type="submit">Save</button></form>{dflt}</td>')
        samples = "–" if r["samples"] is None else str(r["samples"])
        usable = (chip("Yes", "ok") if r["compatible"] else
                  chip("No", "warn") + f'<span class="cj-mut cj-admin-psmeta">{esc(r["reason"] or "")}</span>')
        if not r["compatible"]:
            shown = chip("Not offered", "off")
        else:
            shown = chip("Hidden", "warn") if r["hidden"] else chip("Shown", "ok")
            if not locked:
                verb = "Show" if r["hidden"] else "Hide"
                shown += (f'<form method="post" action="{PREFIX}/predict/show">{hidden_token(tok)}'
                          f'<input type="hidden" name="key" value="{esc(r["key"])}">'
                          f'<input type="hidden" name="hidden" value="{"no" if r["hidden"] else "yes"}">'
                          f'<button class="cj-admin-btn" type="submit" aria-label="{verb} session {esc(sid)}">'
                          f'{verb}</button></form>')
        return (f'<tr class="cj-admin-psrow{" cj-admin-psoff" if r["hidden"] or not r["compatible"] else ""}">'
                f'{where}{name}<td data-l="Tiles" class="cj-num">{r["tiles"]}</td>'
                f'<td data-l="Samples" class="cj-num">{esc(samples)}</td>'
                f'<td data-l="Usable"><div class="cj-admin-pscell">{usable}</div></td>'
                f'<td data-l="In Predict"><div class="cj-admin-pscell">{shown}</div></td></tr>')

    rows = m["sources"]
    n_on = sum(1 for r in rows if r["compatible"] and not r["hidden"])
    if rows:
        table = (f'<div class="cj-card cj-admin-pswrap"><table class="cj-admin-ps">'
                 f'<caption class="cj-admin-sr">Predict sessions</caption><thead><tr><th scope="col">Session</th>'
                 f'<th scope="col">Name people see</th><th scope="col">Tiles</th><th scope="col">Samples</th>'
                 f'<th scope="col">Usable</th><th scope="col">In Predict</th></tr></thead>'
                 f'<tbody>{"".join(row(r, i) for i, r in enumerate(rows))}</tbody></table></div>')
    else:
        table = '<p class="cj-mut">No session with model-ready tiles was found.</p>'
    orphans = m.get("orphans") or []
    if orphans:
        items = "".join(
            f'<li class="cj-admin-line"><b class="cj-num">{esc(o["key"])}</b>'
            f'{chip("hidden", "warn") if o["hidden"] else ""}'
            f'<span class="cj-mut cj-admin-meta">{esc(o["label"] or "no name set")}'
            f'{" · set by " + esc(str(o["by"])) if o.get("by") else ""}'
            f'{" · " + esc(str(o["changed"])) if o.get("changed") else ""}</span></li>' for o in orphans)
        table += (f'<details class="cj-admin-gone" open><summary>Settings for sessions no longer found '
                  f'({len(orphans)})</summary><p class="cj-mut">These names or hide settings match no session folder '
                  f'now (moved, renamed or removed). They apply again if the folder comes back under the same session '
                  f'and 3d folder name.</p><ul class="cj-admin-lines">{items}</ul></details>')
    skipped = m.get("skipped") or []
    if skipped:
        names = ", ".join(s["folder"] for s in skipped)
        table += (f'<details class="cj-admin-gone"><summary>Session folders without model-ready tiles '
                  f'({len(skipped)})</summary><p class="cj-mut cj-wrap">{esc(names)}</p><p class="cj-mut">They '
                  f'appear here once their processing writes a “{esc(PSRC.TILES_PREFIX)}…” folder with a '
                  f'MANIFEST.csv.</p></details>')
    return (f'<h1>Predict sessions</h1><p class="cj-mut">Sessions whose tiles are model-ready (a '
            f'“{esc(PSRC.TILES_PREFIX)}…” folder with a MANIFEST.csv giving nm/px) appear in Predict by themselves. '
            f'Hide one, or give it a clearer name (leave the name empty for the default). Predict re-checks the '
            f'folders every minute; this page checks them now.</p>{msg_html}{"".join(top)}'
            f'<h2 class="cj-sec">In the list <span class="cj-admin-n">{len(rows)}</span> '
            f'<span class="cj-mut">· {n_on} offered</span></h2>{table}')

def render_health(h: dict) -> str:
    now = time.time()
    s = h["services"]
    if s.get("ok") and s["rows"]:
        items = []
        for r in s["rows"]:
            word, tone = SERVICE_WORD.get(r.get("state"), (str(r.get("state") or "Unknown"), "warn"))
            bits = []
            if r.get("port"):
                bits.append(f"port {r['port']}")
            if r.get("health") and r.get("health") not in ("-", "down"):
                bits.append("answers" if str(r["health"]).startswith("http") else str(r["health"]))
            if isinstance(r.get("uptime_s"), (int, float)):
                bits.append(f"up {dur(r['uptime_s'])}")
            items.append(f'<li class="cj-admin-line"><b class="cj-wrap">{esc(str(r["name"]))}</b>'
                         f'{chip(word, tone)}<span class="cj-mut cj-admin-meta">{esc(" · ".join(bits))}</span></li>')
        svc = f'<ul class="cj-admin-lines">{"".join(items)}</ul>'
    elif s.get("ok"):
        svc = '<p class="cj-mut">No services are listed in the services config.</p>'
    else:
        svc = f'<p class="cj-mut">{esc(s.get("problem") or "Unknown.")}</p>'

    w = h["watchers"]
    if w.get("ok"):
        age = (f"Status from {dur(w['age_s'])} ago." if w.get("age_s") is not None else "Status time unknown.")
        head = note(age + " Updates have stopped: the job monitor may be down.", "warn") if w.get("stale") \
            else f'<p class="cj-mut">{esc(age)}</p>'
        items = []
        for q in w["queues"]:
            if q["reach"] != "ok":
                word, tone = "Can't reach it", "warn"
            else:
                word, tone = WATCH_WORD.get(q["display"], (q["display"], "warn"))
            bits = []
            if q.get("age_s") is not None and q["reach"] == "ok":
                bits.append(f"last heartbeat {dur(q['age_s'])} ago")
            if q.get("running"):
                bits.append("running a job")
            items.append(f'<li class="cj-admin-line"><b class="cj-wrap">{esc(q["machine"])}</b>{chip(word, tone)}'
                         f'<span class="cj-mut cj-admin-meta">{esc(" · ".join(bits))}</span></li>')
        wat = head + (f'<ul class="cj-admin-lines">{"".join(items)}</ul>' if items
                      else '<p class="cj-mut">No GPU computers in the status snapshot.</p>')
    else:
        wat = f'<p class="cj-mut">{esc(w.get("problem") or "Unknown.")}</p>'

    c, t = h["backup"]["commit"], h["backup"]["task"]
    if c.get("time"):
        tone = "warn" if c.get("late") else "ok"
        word = "Late" if c.get("late") else "OK"
        cl = (f'<li class="cj-admin-line"><b>Last backup</b>{chip(word, tone)}<span class="cj-mut cj-admin-meta">'
              f'{esc(when(parse_time(c["time"]), now))}</span></li>')
    else:
        cl = (f'<li class="cj-admin-line"><b>Last backup</b>{chip("Unknown", "off")}<span class="cj-mut '
              f'cj-admin-meta">{esc(c.get("problem") or "")}</span></li>')
    if t.get("found"):
        if t.get("never"):
            word, tone = "Not run yet", "off"
        elif t.get("ok"):
            word, tone = "Last run OK", "ok"
        elif t.get("ok") is False:
            word, tone = f"Last run failed (code {t.get('last_result')})", "warn"
        else:
            word, tone = "Unknown", "off"
        bits = [b for b in (f"last run {t['last_run']}" if t.get("last_run") else None,
                            f"next {t['next_run']}" if t.get("next_run") else None,
                            (t.get("status") or "").lower() or None) if b]
        tl = (f'<li class="cj-admin-line"><b>Daily backup task</b>{chip(word, tone)}<span class="cj-mut '
              f'cj-admin-meta">{esc(" · ".join(bits))}</span></li>')
    else:
        tl = (f'<li class="cj-admin-line"><b>Daily backup task</b>{chip("Unknown", "off")}<span class="cj-mut '
              f'cj-admin-meta">{esc(t.get("problem") or "")}</span></li>')
    bak = f'<ul class="cj-admin-lines">{cl}{tl}</ul>'

    def panel(key, title, inner):
        return (f'<section class="cj-card cj-admin-panel" aria-labelledby="cj-admin-h-{key}">'
                f'<h2 id="cj-admin-h-{key}">{esc(title)}</h2>{inner}</section>')

    return (f'<h1>Health</h1><p class="cj-mut">Checked {esc(when(parse_time(h["generated_utc"]), now))}. '
            f'Reload the page to check again.</p>'
            f'<div class="cj-admin-panels">{panel("svc", "Servers on this laptop", svc)}'
            f'{panel("gpu", "GPU computers", wat)}{panel("bak", "Backup", bak)}</div>')


def render_audit(entries, problem) -> str:
    if problem:
        body = note(problem, "warn")
    elif not entries:
        body = '<p class="cj-mut">Nothing logged yet.</p>'
    else:
        items = []
        for e in entries:
            e = PA.normalise(_d(e))
            t = parse_time(_s(e.get("t")))
            act = str(e.get("action") or "?")
            detail = e.get("detail")
            word = ACTION_WORD.get(act, act.replace("_", " "))
            if isinstance(detail, dict):
                if detail.get("phase") == "requested":       # write-ahead line of a Jobs action
                    word = f"asked to {word}"
                elif detail.get("phase") == "result":
                    word = f"{word} result for"
                dtext = ", ".join(f"{k}: {v}" for k, v in detail.items() if v is not None and k != "phase")
            else:
                dtext = "" if detail is None else str(detail)
            dspan = f' <span class="cj-mut">({esc(dtext)})</span>' if dtext else ""
            items.append(f'<li class="cj-admin-aud"><span class="cj-mut cj-num">'
                         f'{esc(datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S") if t else "?")}</span>'
                         f'<span><b class="cj-wrap">{esc(str(e.get("actor") or "-"))}</b> '
                         f'{esc(word)} '
                         f'<b class="cj-wrap">{esc(str(e.get("target") or ""))}</b>'
                         f'{dspan}</span></li>')
        body = f'<ol class="cj-card cj-admin-auds">{"".join(items)}</ol>'
    return (f'<h1>Recent actions</h1><p class="cj-mut">The last {AUDIT_SHOW} actions in the portal (people, job '
            f'cancels and releases), newest first. Times are this computer\'s.</p>{body}')


_PH = re.compile(r"\{\{(sub|content)\}\}")


def page_body(active, content) -> str:
    tpl = TEMPLATE.read_text(encoding="utf-8")
    parts = {"sub": subnav(active), "content": content}
    return _PH.sub(lambda m: parts[m.group(1)], tpl)


# ---------------------------------------------------------------- the key page's Copy button (served as a file)
JS = r"""// Admin key page (portal/modules/admin.py): the Copy button. Shown only where the clipboard works.
(function () {
  'use strict';
  var b = document.querySelector('[data-cj-admin-copy]');
  if (!b || !navigator.clipboard || !window.isSecureContext) return;
  var box = document.getElementById(b.getAttribute('data-cj-admin-copy'));
  var out = document.getElementById('cj-admin-copied');
  if (!box) return;
  b.hidden = false;
  b.addEventListener('click', function () {
    navigator.clipboard.writeText(box.value).then(function () {
      if (out) out.textContent = 'Copied.';
    }, function () {
      box.focus(); box.select();
      if (out) out.textContent = 'Select the link and copy it.';
    });
  });
})();
"""


# ---------------------------------------------------------------- routes
def routes(ctx) -> list:
    secret = secrets.token_bytes(32)                     # per process: a restart makes open forms stale

    def token(request: Request) -> str:
        sess = request.cookies.get(A.COOKIE) or ""
        return hmac.new(secret, hashlib.sha256(sess.encode()).digest(), hashlib.sha256).hexdigest()[:40]

    def me(request: Request) -> str:
        return str(ctx.user(request).get("id"))

    def show(request, title, active, content, status=200):
        r = ctx.page(request, title, page_body(active, content), active="admin")
        r.status_code = status
        return r

    async def form(request: Request, tab: str = "people"):
        """-> (fields, None) or (None, error response on `tab`). urlencoded only; the token must match this
        session."""
        back = {"people": ("/", "People"), "predict": ("/predict", "Predict sessions")}[tab]
        ctype = (request.headers.get("content-type") or "").split(";")[0].strip().lower()
        if ctype != "application/x-www-form-urlencoded":
            return None, show(request, "Admin", tab, note("Send the form from the Admin page.", "warn"), 415)
        try:
            n = int(request.headers.get("content-length") or 0)
        except ValueError:
            n = MAX_FORM + 1
        if n > MAX_FORM:
            return None, show(request, "Admin", tab, note("That form is too large.", "warn"), 413)
        body = await request.body()
        if len(body) > MAX_FORM:
            return None, show(request, "Admin", tab, note("That form is too large.", "warn"), 413)
        try:
            q = parse_qs(body.decode("utf-8"), keep_blank_values=True, max_num_fields=10)
        except (UnicodeDecodeError, ValueError):
            return None, show(request, "Admin", tab, note("That form could not be read.", "warn"), 400)
        f = {k: v[0].strip() for k, v in q.items() if v}
        if not hmac.compare_digest(f.get("t", "").encode(), token(request).encode()):
            return None, show(request, "Admin", tab,
                              note("This form has expired. Reload the page and try again.", "warn")
                              + f'<p><a class="cj-admin-link" href="{PREFIX}{back[0]}">Back to {back[1]}</a></p>',
                              403)
        return f, None

    def people_page(request, msg_html="", status=200):
        try:
            rows = people(ctx.cfg.roster)
        except ValueError as e:
            return show(request, "People", "people", "<h1>People</h1>" + note(
                f"The roster can't be read ({e}). Nothing was changed.", "warn"), 500)
        return show(request, "People", "people", render_people(rows, me(request), token(request), msg_html), status)

    def try_audit(actor, action, target, detail) -> str:
        try:
            audit(ctx, actor, action, target, detail)
            return ""
        except OSError as e:
            log.error("admin: audit log not written (%s)", type(e).__name__)
            return note(f"The change was made, but the audit log could not be written ({type(e).__name__}).", "warn")

    def index(request: Request):                         # sync: Starlette runs it in a worker thread
        text = flash_take(request.cookies.get(A.COOKIE) or "", request.query_params.get("f"))
        return people_page(request, note(text) if text else "")

    def done(request: Request, text: str, to: str = PREFIX + "/"):
        """303 to People (or `to`) with a one-time message for this session (a bare URL shows nothing)."""
        tok = flash_put(request.cookies.get(A.COOKIE) or "", text)
        return ctx.no_store(RedirectResponse(f"{to}?f={quote(tok)}", status_code=303))

    async def change(request: Request, fn, *args):
        """-> (result, None) or (None, the People page saying why nothing was changed)."""
        try:
            return await run_in_threadpool(fn, ctx.cfg.roster, *args), None
        except Refused as e:
            msg, status = str(e), e.status
        except ValueError as e:
            msg, status = f"Nothing was changed: {e}.", 500
        except OSError as e:                             # lock not had, or the write failed after its retries
            log.error("admin: roster not written (%s)", type(e).__name__)
            msg, status = why_not_written(e), 503
        return None, await run_in_threadpool(people_page, request, note(msg, "warn"), status)

    async def add(request: Request):
        f, err = await form(request)
        if err:
            return err
        pid, role, actor = f.get("id", "").lower(), f.get("role", ""), me(request)
        key, err = await change(request, add_person, pid, role)
        if err:
            return err
        warn = await run_in_threadpool(try_audit, actor, "person_added", pid, {"role": role})
        base = await run_in_threadpool(link_base, ctx.cfg, request.headers.get("origin"))
        link = f"{base}/login?k={key}"
        del key
        r = show(request, f"Key link for {pid}", "people", warn + render_key(pid, role, link))
        r.headers["Cache-Control"] = "no-store"
        r.headers["Pragma"] = "no-cache"
        return r

    async def role(request: Request):
        f, err = await form(request)
        if err:
            return err
        pid, new, actor = f.get("id", ""), f.get("role", ""), me(request)
        old, err = await change(request, set_role, pid, new)
        if err:
            return err
        warn = await run_in_threadpool(try_audit, actor, "role_changed", pid, {"from": old, "to": new})
        if warn:
            return await run_in_threadpool(people_page, request, warn)
        if pid == actor and new != "admin":              # no longer an admin: /admin/ would answer 403
            return ctx.no_store(RedirectResponse("/", status_code=303))
        return done(request, f"{pid} is now {ROLE_WORD[new]}. It applies from their next click.")

    def revoke_page(request: Request):
        pid = request.query_params.get("id") or ""
        if not A.ID_RE.fullmatch(pid):
            return show(request, "Revoke", "people", note("No such person.", "warn"), 404)
        why = revoke_check(ctx.cfg.roster, pid)
        return show(request, f"Revoke {pid}", "people", render_revoke(pid, me(request), why, token(request)),
                    409 if why else 200)

    async def revoke(request: Request):
        f, err = await form(request)
        if err:
            return err
        pid, actor = f.get("id", ""), me(request)
        if f.get("confirm") != "yes":
            why = await run_in_threadpool(revoke_check, ctx.cfg.roster, pid)
            return show(request, f"Revoke {pid}", "people", render_revoke(pid, actor, why, token(request)))
        old, err = await change(request, revoke_person, pid)
        if err:
            return err
        warn = await run_in_threadpool(try_audit, actor, "person_revoked", pid, {"role": old})
        if warn:
            return await run_in_threadpool(people_page, request, warn)
        if pid == actor:
            return ctx.no_store(RedirectResponse("/", status_code=303))
        return done(request, f"{pid} is revoked and signed out.")

    async def health_page(request: Request):
        h = await asyncio.to_thread(health, ctx.cfg)
        return show(request, "Health", "health", render_health(h))

    async def api_health(request: Request):
        return ctx.json(await asyncio.to_thread(health, ctx.cfg))

    def audit_page(request: Request):
        entries, prob = read_audit(ctx.cfg)
        return show(request, "Recent actions", "audit", render_audit(entries, prob))

    def api_audit(request: Request):
        entries, prob = read_audit(ctx.cfg)
        return ctx.json({"schema": "cj_admin_audit/1", "entries": entries, "problem": prob})

    def api_people(request: Request):
        try:
            return ctx.json({"schema": "cj_admin_people/1", "people": people(ctx.cfg.roster)})
        except ValueError as e:
            return ctx.json({"schema": "cj_admin_people/1", "people": [], "problem": str(e)}, 500)

    # ---- Predict sessions
    def predict_page(request: Request, msg_html="", status=200):
        if not msg_html:
            text = flash_take(request.cookies.get(A.COOKIE) or "", request.query_params.get("f"))
            msg_html = note(text) if text else ""
        m = PSRC.merged(ctx.cfg, fresh=True)
        return show(request, "Predict sessions", "predict", render_predict(m, token(request), msg_html), status)

    def predict_apply(key, hidden, label, actor) -> tuple:
        """-> (flash text, (action, target, detail) for the audit line, or None when nothing changed).
        Raises Refused or OSError (nothing changed)."""
        m = PSRC.merged(ctx.cfg)
        if m.get("overrides_error"):
            raise Refused(f"Nothing was changed: {m['overrides_error']}.", 500)
        r = next((x for x in m["sources"] if x["key"] == key), None)
        if r is None:
            raise Refused("That session is not in the list any more. Nothing was changed.", 404)
        sid = session_name(r)
        if hidden is not None:
            if bool(r["hidden"]) == hidden:
                return f"Session {sid} is already {'hidden' if hidden else 'shown'}.", None
            try:
                PSRC.set_override(ctx.cfg.state_dir, key, hidden=hidden, actor=actor)
            except ValueError as e:
                raise Refused(f"Nothing was changed: {e}", 400)
            word = "hidden from Predict" if hidden else "shown in Predict"
            return (f"Session {sid} ({r['label']}) is now {word}.",
                    ("predict_source_hidden" if hidden else "predict_source_shown", key,
                     {"session": sid, "label": r["label"]}))
        try:
            new = PSRC.clean_label(label)
        except ValueError as e:
            raise Refused(f"Nothing was changed: {e}", 400)
        want = new or r["default_label"]
        if want == r["label"]:
            return f"Session {sid} already has that name.", None
        try:
            PSRC.set_override(ctx.cfg.state_dir, key, label=new, actor=actor)
        except ValueError as e:
            raise Refused(f"Nothing was changed: {e}", 400)
        text = (f"Session {sid} is now called “{want}”." if new else
                f"Session {sid} is back to its default name, “{want}”.")
        return text, ("predict_source_renamed", key, {"session": sid, "from": r["label"], "to": want})

    async def predict_post(request: Request, kind: str):
        f, err = await form(request, "predict")
        if err:
            return err
        actor, key = me(request), f.get("key", "")
        hidden = label = None
        if kind == "show":
            if f.get("hidden") not in ("yes", "no"):
                return await run_in_threadpool(predict_page, request, note("That form could not be read.", "warn"),
                                               400)
            hidden = f["hidden"] == "yes"
        else:
            label = f.get("label", "")
        try:
            text, aud = await run_in_threadpool(predict_apply, key, hidden, label, actor)
        except Refused as e:
            return await run_in_threadpool(predict_page, request, note(str(e), "warn"), e.status)
        except OSError as e:
            log.error("admin: predict sessions not written (%s)", type(e).__name__)
            msg = ("Nothing was changed: the settings are being changed elsewhere right now. Try again."
                   if "elsewhere" in str(e) else
                   f"Nothing was changed: the settings could not be written ({type(e).__name__}). Try again.")
            return await run_in_threadpool(predict_page, request, note(msg, "warn"), 503)
        if aud:
            warn = await run_in_threadpool(try_audit, actor, *aud)
            if warn:
                return await run_in_threadpool(predict_page, request, warn)
        return done(request, text, PREFIX + "/predict")

    async def predict_label(request: Request):
        return await predict_post(request, "label")

    async def predict_show(request: Request):
        return await predict_post(request, "show")

    def api_predict(request: Request):
        m = PSRC.merged(ctx.cfg, fresh=True)
        return ctx.json({"schema": "cj_admin_predict/1", "sources": [public_source(r) for r in m["sources"]],
                         "orphans": m["orphans"],
                         "skipped": [{"folder": s["folder"], "reason": s["reason"]} for s in m["skipped"]],
                         "roots": [{"root": str(r["root"]), "problem": r["problem"]} for r in m["roots"]],
                         "list_error": m["list_error"], "overrides_error": m["overrides_error"]})

    def script(request: Request):
        r = Response(JS, media_type="text/javascript; charset=utf-8")
        r.headers["Cache-Control"] = "no-cache"
        return r

    return [Route(PREFIX + "/", index, methods=["GET"]),
            Route(PREFIX + "/people/add", add, methods=["POST"]),
            Route(PREFIX + "/people/role", role, methods=["POST"]),
            Route(PREFIX + "/people/revoke", revoke_page, methods=["GET"]),
            Route(PREFIX + "/people/revoke", revoke, methods=["POST"]),
            Route(PREFIX + "/predict", predict_page, methods=["GET"]),
            Route(PREFIX + "/predict/label", predict_label, methods=["POST"]),
            Route(PREFIX + "/predict/show", predict_show, methods=["POST"]),
            Route(PREFIX + "/api/predict", api_predict, methods=["GET"]),
            Route(PREFIX + "/health", health_page, methods=["GET"]),
            Route(PREFIX + "/audit", audit_page, methods=["GET"]),
            Route(PREFIX + "/admin.js", script, methods=["GET"]),
            Route(PREFIX + "/api/people", api_people, methods=["GET"]),
            Route(PREFIX + "/api/health", api_health, methods=["GET"]),
            Route(PREFIX + "/api/audit", api_audit, methods=["GET"])]
