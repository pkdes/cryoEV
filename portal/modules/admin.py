"""
Admin module (CryoEV portal, admin only): people and their key links, health at a glance, and the audit log.

    GET  /admin/                 People: everyone on the roster (id, role, active / revoked, added; never keys or
                                 hashes), the add form, a role form and a "Revoke..." link per person
    POST /admin/people/add       id + role -> a new key link, shown ONCE on the answer page (no-store). A refresh
                                 re-sends the form, which is refused ("already on the list"): the key is never shown
                                 again, never logged and never stored (the roster keeps its sha256 only, auth.py)
    POST /admin/people/role      id + role -> the roster entry's role (same key; the Gate re-reads the roster on
                                 every request, so it takes effect on that person's next request)
    POST /admin/people/email     id + email -> set, change or ('') remove an active person's sign-in email (Cloudflare
                                 access mode; lower-cased, unique among active people). The add form takes one too
    GET  /admin/people/revoke    ?id= -> the confirmation page (a form; no script, no confirm())
    POST /admin/people/revoke    id + confirm=yes -> revoked: that person's sessions end at once, and their agent
                                 tokens are revoked (portal/auth.py revoke_person_tokens)
    POST /admin/people/token-revoke   id (tk_...) -> that agent token revoked (each person's card lists their live
                                 tokens: agent, expiry, last used; never the token or its hash)
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
    GET  /admin/admin.js         the key page's Copy button and the Recent actions filters (a file: no inline script)

Rules:
  - Every POST: the Gate checks Origin (portal/app.py); here a form token bound to the sign-in (`binding`: the
    session cookie in key mode, the VERIFIED user id in Cloudflare access mode; HMAC with a per-process secret) is
    checked as well. A portal restart makes open forms stale (reload).
  - Never leave the portal without an admin: demoting or revoking the last active admin is refused (yourself or
    anyone). Demoting or revoking yourself while other admins remain is allowed (you are signed out / sent home).
  - The roster is the private file auth.py reads (cfg.roster), in auth.py's format. Add, role change and revoke
    are each one read-modify-write inside auth.roster_lock (a thread lock + an OS lock on <roster>.lock, which the
    `python -m portal.auth` tool takes too), written atomically with os.replace retried briefly; the file's
    (mtime, size) must change (the Gate's Roster cache keys on it) or it raises. Any OSError (lock or write):
    "Nothing was changed", no temp file left.
  - After a change the page shows a one-time message bound to the same `binding` (?f=<token>); a bare URL shows none.
  - The key link points at the first --origin, else the tailnet address (auth.default_base), else the request's
    Origin reduced to scheme://host[:port] (never in access mode: key links don't work through Cloudflare).
  - Audit: portal/audit.py (shared with Jobs), <state dir>/audit/portal.jsonl. `audit(ctx, actor, action, target,
    detail)` here is a thin wrapper. The Recent actions page shows the last AUDIT_SHOW lines of every door: every
    portal write (the Gate's catch-all, portal/app.py), the first sign-in per person per day, the modules' own lines,
    and the laptop CLI's file (named in CJ_AUDIT_EXTRA; door "laptop CLI (unverified)"). Counter lines (detail
    {from, to, n}) show a time range and "n x action"; agent lines (detail.via) show actor.label + an AI tag. A
    portal line with detail.via came from a bearer token the Gate verified: its door shows as audit.API_DOOR
    ("API (verified)"), never "unverified" (only the laptop CLI's self-declared lines are).
    Person / Action filters run in the browser over those lines only (admin.js).
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

from portal import access_config
from portal import audit as PA
from portal import auth as A
from portal import predict_sources as PSRC

ACTOR = PA.actor_module()                          # registry/actor.py: label() for agent lines

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
               "email_changed": "changed the sign-in email of", "key_made": "made a key link for",
               "cancel": "cancel", "release": "release", "predict_source_renamed": "renamed the Predict session",
               "predict_source_hidden": "hid the Predict session", "predict_source_shown": "showed the Predict session",
               "agent token made": "made an agent token", "agent token revoked": "revoked the agent token"}

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


def usable_any(p) -> bool:
    """Can sign in in either mode: a key (tailnet) or an email (Cloudflare access; no key needed)."""
    return A.Roster.usable(p) or A.Roster.usable_access(p)


def is_admin(p) -> bool:
    """An admin who can sign in in EITHER mode: the last-admin guard must see email-only admins too."""
    return usable_any(p) and p.get("role") == "admin"


def public_person(p: dict, email: bool = False) -> dict:
    """The only fields that ever leave this module: never key_sha256 (or anything else in the entry). `email` adds
    the sign-in email (the People page; /admin/api/people keeps its shape)."""
    out = {"id": str(p.get("id")), "role": p.get("role") if p.get("role") in A.ROLES else None,
           "state": "revoked" if p.get("revoked") else ("active" if usable_any(p) else "not usable"),
           "added": _s(p.get("added")), "revoked": _s(p.get("revoked")), "role_changed": _s(p.get("role_changed"))}
    if email:
        out["email"] = _s(p.get("email"))
        out["has_key"] = isinstance(p.get("key_sha256"), str)
    return out


def people(path, email: bool = False) -> list:
    doc = load_roster(path)
    rows = [public_person(p, email) for p in doc["people"] if isinstance(p, dict) and _s(p.get("id"))]
    rows.sort(key=lambda r: (r["state"] != "active", r["id"]))
    return rows


class Refused(Exception):
    """A request the roster rules refuse; the message is shown to the admin as is."""

    def __init__(self, msg, status=409):
        super().__init__(msg)
        self.status = status


BAD_ID = "An id is 1 to 40 lower-case letters, digits, '_', '.' or '-', starting with a letter or digit."


def _check_id(pid) -> None:
    if not A.id_ok(pid):                                 # ID_RE, and never a reserved audit actor ("cli", "-")
        raise Refused(BAD_ID, 400)


def keyed_admin(p) -> bool:
    """An active admin with a key link: the tailnet door, the break-glass way in (works in every mode)."""
    return A.Roster.usable(p) and p.get("role") == "admin"


def _keyed_admins_left(doc, pid) -> int:
    return sum(1 for p in doc["people"] if isinstance(p, dict) and keyed_admin(p) and p.get("id") != pid)


KEYS_TAILNET_ONLY = "Use the Tailscale admin page to make key links."
LAST_KEYED = ("{pid} is the last admin with a key link. Make a key link for another admin first (the Tailscale "
              "page is the way in if Cloudflare sign-in fails).")


def _admins_left(doc, pid) -> int:
    return sum(1 for p in doc["people"] if isinstance(p, dict) and is_admin(p) and p.get("id") != pid)


def _active(doc, pid) -> list:
    return [p for p in doc["people"] if isinstance(p, dict) and p.get("id") == pid and not p.get("revoked")]


def _email(email) -> str:
    """'' or the address lower-cased (auth.norm_email); Refused (400) for something that isn't an address."""
    try:
        return A.norm_email(email)
    except ValueError:
        raise Refused("That email address doesn't look right (name@example.org).", 400) from None


def _email_free(doc, email, pid) -> None:
    other = A.email_taken(doc["people"], email, pid)
    if other:
        raise Refused(f"{email} is already the email of {other}. Emails are unique among active people.")


def add_person(path, pid: str, role: str, email: str | None = None, with_key: bool = True) -> str | None:
    """-> the new key (the caller shows it once and forgets it). Same entry as auth.add_key (+ `email` when given:
    lower-cased, unique among active people); refuses an id that is already active. `with_key=False` (needs an
    email) makes an email-only person: no key, so Cloudflare sign-in only; -> None. Raises Refused, ValueError
    (unreadable roster) or OSError (lock or write: nothing changed)."""
    _check_id(pid)
    if role not in A.ROLES:
        raise Refused("Pick a role: Admin, Advanced or Basic.", 400)
    em = _email(email)
    if not with_key and not em:
        raise Refused("A person without a key link needs an email to sign in with.", 400)
    with roster_lock(path):
        doc = load_roster(path)
        if _active(doc, pid):
            raise Refused(f"{pid} is already on the list. To give them a new key link, use Make key link on their "
                          "card (or revoke them and add them again).")
        _email_free(doc, em, pid)
        key = secrets.token_urlsafe(A.KEY_BYTES) if with_key else None
        entry = {"id": pid, "role": role, "added": datetime.now().isoformat(timespec="seconds")}
        if key:
            entry["key_sha256"] = A.sha256_hex(key)
        if em:
            entry["email"] = em
        doc["people"] = [p for p in doc["people"] if not (isinstance(p, dict) and p.get("id") == pid)] + [entry]
        write_roster(path, doc)
        return key


def make_key(path, pid: str) -> tuple:
    """Issue a key link for an active person (an email-only person gets their first; otherwise the old key is
    replaced and its sessions end). Role and email are kept. -> (key, had_key). Refuses a revoked or unknown id."""
    _check_id(pid)
    with roster_lock(path):
        doc = load_roster(path)
        hits = _active(doc, pid)
        if not hits:
            raise Refused(f"{pid} is not on the list, or revoked. Add them again to give them a key link.", 404)
        p = hits[-1]
        if p.get("role") not in A.ROLES:
            raise Refused(f"{pid} has no valid role. Set a role first.")
        had = isinstance(p.get("key_sha256"), str)
        key = secrets.token_urlsafe(A.KEY_BYTES)
        p["key_sha256"] = A.sha256_hex(key)
        p["key_issued"] = datetime.now().isoformat(timespec="seconds")
        write_roster(path, doc)
        return key, had


def set_email(path, pid: str, email: str) -> tuple:
    """Set, change or ('') remove an active person's sign-in email. -> (old, new) ('' for none). Refuses a revoked or
    unknown id, a bad address and one another active person has."""
    _check_id(pid)
    em = _email(email)
    with roster_lock(path):
        doc = load_roster(path)
        hits = _active(doc, pid)
        if not hits:
            raise Refused(f"{pid} is not on the list, or revoked.", 404)
        old = str(hits[-1].get("email") or "")
        if old and not em and not any(isinstance(p.get("key_sha256"), str) for p in hits):
            raise Refused(f"{pid} has no key link, so the email is their only way in. Make a key link first.")
        if old == em:
            raise Refused(f"{pid} already has that email." if em else f"{pid} has no email.")
        _email_free(doc, em, pid)
        for p in hits:
            if em:
                p["email"] = em
            else:
                p.pop("email", None)
        write_roster(path, doc)
        return old, em


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
        if keyed_admin(p) and role != "admin" and _keyed_admins_left(doc, pid) == 0:
            raise Refused(LAST_KEYED.format(pid=pid))
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
        if any(keyed_admin(p) for p in hits) and _keyed_admins_left(doc, pid) == 0:
            raise Refused(LAST_KEYED.format(pid=pid))
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
    if any(keyed_admin(p) for p in hits) and _keyed_admins_left(doc, pid) == 0:
        return LAST_KEYED.format(pid=pid)
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
    if getattr(cfg, "auth", "key") == "access":         # this origin answers /login with 404
        return ""
    try:
        u = urlsplit(request_origin or "")
        host, port = u.hostname, u.port
    except ValueError:
        return ""
    if u.scheme not in ("http", "https") or not host:
        return ""
    host = f"[{host}]" if ":" in host else host
    return f"{u.scheme}://{host}" + (f":{port}" if port else "")


# ---------------------------------------------------------------- what forms and messages are bound to
def binding(request) -> str:
    """The sign-in the form token and flash messages are bound to. Key mode: the session cookie (as before).
    Cloudflare access mode (the Gate sets scope["cj_auth"]): the VERIFIED user id; there is no session cookie."""
    if request.scope.get("cj_auth") == "access":
        uid = (request.scope.get("cj_user") or {}).get("id")
        return f"access-user:{uid}" if uid else ""
    return request.cookies.get(A.COOKIE) or ""


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
    paths = _health_paths()
    out = []
    for r in rows:
        if _s(_d(r).get("name")):
            row = {k: _d(r).get(k) for k in keep}
            row["health_path"] = row["name"] in paths
            out.append(row)
    return {"ok": True, "rows": out, "problem": None}


def _health_paths() -> set:
    """Names of services whose services config has a "health" path (the tool then reports ok / http N / bad body;
    without one any HTTP answer is 'http N' and counts as up)."""
    p = os.environ.get("CRYOEV_SERVICES_CONFIG") or str(Path.home() / ".cryoev" / "services.json")
    try:
        doc = json.loads(Path(p).read_text(encoding="utf-8-sig"))
        svcs = doc.get("services", [])
        return {str(x.get("name")) for x in (svcs if isinstance(svcs, list) else [])
                if isinstance(x, dict) and x.get("health")}
    except (OSError, ValueError, AttributeError, TypeError):
        return set()


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


DISK_WARN_GB = 20.0


def disks(cfg) -> dict:
    """Free space on the drive of the predict root and of the raw root (CJ_RAW_ROOT; the same drive when unset)."""
    import shutil
    from portal.modules import predict as P
    rows = []
    for label, path in (("Predict sets", P.predict_root(cfg)), ("Raw images", P.raw_base(cfg))):
        row = {"label": label, "path": str(path), "free_gb": None, "warn": False}
        try:
            row["free_gb"] = round(shutil.disk_usage(P._existing(path)).free / 2 ** 30, 1)
            row["warn"] = row["free_gb"] < DISK_WARN_GB
        except OSError:
            pass
        rows.append(row)
    return {"rows": rows, "warn_gb": DISK_WARN_GB}


DISK_TIMEOUT_S = 5.0


def disks_bounded(cfg) -> dict:
    """disks() in a daemon thread with a short timeout: an unreachable UNC/NAS path can't stall the Health page."""
    box = []
    t = threading.Thread(target=lambda: box.append(_safe_call(disks, cfg, default=None)), daemon=True)
    t.start()
    t.join(DISK_TIMEOUT_S)
    return box[0] if box and box[0] else {"rows": [], "warn_gb": DISK_WARN_GB,
                                           "problem": "The disks did not answer in time."}


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
               "disks": disks_bounded(cfg),
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


def render_tokens(pid, toks, tok) -> str:
    """A person's agent tokens on their card: live ones with Revoke; a count of the expired / revoked rest."""
    toks = toks or []
    live = [t for t in toks if t.get("state") == "active"]
    if not toks:
        return ""
    items = []
    for t in live:
        exp = parse_time(_s(t.get("expires")))
        used = parse_time(_s(t.get("last_used")))
        items.append(f'<li class="cj-admin-tok" data-token-id="{esc(str(t.get("id")))}"><span><b>'
                     f'{esc(ACTOR.label(pid, t.get("agent")))}</b> <span class="cj-mut">{esc(str(t.get("id")))} · '
                     f'{esc(str(t.get("scope")))} · expires {esc(_clock(exp, False))} · last used '
                     f'{esc(_clock(used, False) if used else "never")}</span></span>'
                     f'<form method="post" action="{PREFIX}/people/token-revoke">{hidden_token(tok)}'
                     f'<input type="hidden" name="id" value="{esc(str(t.get("id")))}">'
                     f'<button class="cj-admin-btn cj-admin-dangerbtn" type="submit">Revoke token</button></form></li>')
    rest = len(toks) - len(live)
    more = f'<p class="cj-mut cj-admin-meta">{rest} expired or revoked.</p>' if rest else ""
    return (f'<div class="cj-admin-toks"><p class="cj-admin-lbl">Agent tokens ({len(live)} live)</p>'
            f'<ul class="cj-admin-toklist">{"".join(items)}</ul>{more}</div>')


def render_people(rows, me, tok, msg_html="", keys_ok=True, tokens=None) -> str:
    """tokens: {person id: [public token records]} (auth.list_tokens), shown on each active person's card."""
    tokens = tokens or {}
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
        em = r.get("email")
        mail = (f'<p class="cj-mut cj-admin-meta cj-wrap cj-admin-email">Email: {esc(em)}'
                f'{"" if r.get("has_key", True) else " · no key link (Cloudflare sign-in only)"}</p>' if em else
                '<p class="cj-mut cj-admin-meta cj-admin-email">Email: none (no Cloudflare sign-in)</p>')
        if r["state"] == "revoked":
            rt = parse_time(r["revoked"])
            return (f'<li class="cj-admin-person"><div class="cj-admin-top"><b class="cj-wrap">{name}</b>'
                    f'{chip(ROLE_WORD.get(role, role), "off")}</div>'
                    f'<p class="cj-mut cj-admin-meta">Revoked {esc(when(rt))}. {meta}.</p>{mail if em else ""}</li>')
        last = role == "admin" and n_admin <= 1
        fid = f"cj-admin-r{i}"
        role_form = (f'<form class="cj-admin-row" method="post" action="{PREFIX}/people/role">'
                     f'{hidden_token(tok)}<input type="hidden" name="id" value="{esc(r["id"])}">'
                     f'<label class="cj-admin-lbl" for="{fid}">Role</label>{role_select("role", role, fid)}'
                     f'<button class="cj-admin-btn" type="submit">Change role</button></form>')
        has_key = r.get("has_key", True)
        key_form = (f'<form class="cj-admin-row" method="post" action="{PREFIX}/people/key">'
                    f'{hidden_token(tok)}<input type="hidden" name="id" value="{esc(r["id"])}">'
                    f'<button class="cj-admin-btn" type="submit">'
                    f'{"Make a new key link" if has_key else "Make key link"}</button></form>')
        eid = f"cj-admin-e{i}"
        mail_form = (f'<form class="cj-admin-row" method="post" action="{PREFIX}/people/email">'
                     f'{hidden_token(tok)}<input type="hidden" name="id" value="{esc(r["id"])}">'
                     f'<label class="cj-admin-lbl" for="{eid}">Email</label>'
                     f'<input class="cj-admin-in" id="{eid}" name="email" type="email" maxlength="254" '
                     f'value="{esc(em or "")}" autocomplete="off" autocapitalize="none" spellcheck="false">'
                     f'<button class="cj-admin-btn" type="submit">{"Change email" if em else "Set email"}</button>'
                     f'</form>')
        rev = (f'<a class="cj-admin-link cj-admin-danger" href="{PREFIX}/people/revoke?id={quote(r["id"])}">'
               f'Revoke{" (you)" if you else ""}…</a>')
        lastnote = ('<p class="cj-mut cj-admin-meta">The last admin: can\'t be demoted or revoked until someone '
                    'else is an admin.</p>' if last else "")
        bad = ('<p class="cj-admin-note cj-admin-t-warn">This entry has no usable key or a bad role: it can\'t '
               'sign in.</p>' if r["state"] != "active" else "")
        return (f'<li class="cj-admin-person"><div class="cj-admin-top"><b class="cj-wrap">{name}</b>'
                f'{chip(ROLE_WORD.get(role, role), tone)}</div>'
                f'<p class="cj-mut cj-admin-meta">{meta}.</p>{mail}{bad}{lastnote}'
                + ("" if last else f'<div class="cj-admin-acts">{role_form}{rev}</div>')
                + f'<div class="cj-admin-acts">{mail_form}{key_form if keys_ok else ""}</div>'
                + render_tokens(r["id"], tokens.get(r["id"]), tok) + '</li>')

    roles = "".join(f"<li><b>{ROLE_WORD[r]}</b>: {esc(ROLE_DESC[r])}</li>" for r in A.ROLES)
    add = (f'<section class="cj-card cj-admin-add" aria-labelledby="cj-admin-h-add">'
           f'<h2 id="cj-admin-h-add">Add a person</h2>'
           f'<form class="cj-admin-form" method="post" action="{PREFIX}/people/add">{hidden_token(tok)}'
           f'<label class="cj-admin-lbl" for="cj-admin-id">Id <span class="cj-mut">(lower-case, e.g. '
           f'jsmith)</span></label>'
           f'<input class="cj-admin-in" id="cj-admin-id" name="id" required maxlength="40" '
           f'pattern="[a-z0-9][a-z0-9_.\\-]{{0,39}}" autocomplete="off" autocapitalize="none" spellcheck="false">'
           f'<label class="cj-admin-lbl" for="cj-admin-email">Email <span class="cj-mut">(optional: their '
           f'Cloudflare sign-in; with one they get no key link unless you tick the box)</span></label>'
           f'<input class="cj-admin-in" id="cj-admin-email" name="email" type="email" maxlength="254" '
           f'autocomplete="off" autocapitalize="none" spellcheck="false">'
           f'<label class="cj-admin-lbl" for="cj-admin-role">Role</label>{role_select("role", "basic", "cj-admin-role")}'
           + (f'<label class="cj-admin-lbl"><input type="checkbox" name="also_key" value="1"> Also make a '
              f'Tailscale key link <span class="cj-mut">(only matters when an email is entered)</span></label>'
              if keys_ok else "") +
           f'<button class="cj-admin-btn cj-admin-primary" type="submit">Add person</button></form>'
           f'<ul class="cj-mut cj-admin-roles">{roles}</ul>'
           f'<p class="cj-mut cj-admin-meta">Without an email, a key link is made and shown once, on the next '
           f'page. Send it to the person privately.</p></section>')
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


def render_signin(pid, role, email, host) -> str:
    """The result page for an email-only person: where to send them. No host configured: no URL sentence."""
    where = (f"Send them https://{esc(host)}/ \u2014 they sign in with {esc(email)} (one-time code by email)."
             if host else f"They sign in with {esc(email)} (one-time code by email).")
    return (f'<h1>{esc(pid)} added</h1>'
            f'{note(f"{pid} was added as {ROLE_WORD[role]}.")}'
            f'<section class="cj-card cj-admin-key" aria-labelledby="cj-admin-h-key">'
            f'<h2 id="cj-admin-h-key">How {esc(pid)} signs in</h2>'
            f'<p class="cj-admin-signin">{where}</p>'
            f"<p class=\"cj-mut\">Email-only people can't use the Tailscale link. "
            f'Use Make key link on their card if they need one.</p></section>'
            f'<p><a class="cj-admin-link" href="{PREFIX}/">Back to People</a></p>')


def render_key(pid, role, link, rotated=False) -> str:
    return (f'<h1>Key link for {esc(pid)}</h1>'
            f'{note(f"{pid} was added as {ROLE_WORD[role]}." if not rotated else f"A key link was made for {pid}.")}'
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
                hs = str(r["health"])
                if r.get("health_path"):
                    if hs == "ok":
                        bits.append("ok")
                    else:                           # a health path was asked and it did not say ok
                        bits.append(f"problem: {hs}")
                        if tone == "ok":
                            tone, word = "warn", "Running, unhealthy"
                elif hs.startswith("http"):
                    bits.append("answers")          # no health path: any HTTP answer is up
                else:
                    bits.append(hs)
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

    dk = _d(h.get("disks"))
    dl = []
    for r in dk.get("rows") or []:
        r = _d(r)
        if r.get("free_gb") is None:
            word, tone, meta = "Unknown", "off", "could not be read"
        else:
            low = bool(r.get("warn"))
            word, tone = (f"{r['free_gb']} GB free", "warn" if low else "ok")
            meta = f"under {DISK_WARN_GB:g} GB: free some space" if low else ""
        dl.append(f'<li class="cj-admin-line"><b>{esc(r.get("label"))}</b>{chip(word, tone)}<span class="cj-mut '
                  f'cj-admin-meta">{esc(r.get("path"))}{" · " + esc(meta) if meta else ""}</span></li>')
    dsk = (f'<ul class="cj-admin-lines">{"".join(dl)}</ul>' if dl
           else '<p class="cj-mut">Disk space could not be read.</p>')

    def panel(key, title, inner):
        return (f'<section class="cj-card cj-admin-panel" aria-labelledby="cj-admin-h-{key}">'
                f'<h2 id="cj-admin-h-{key}">{esc(title)}</h2>{inner}</section>')

    return (f'<h1>Health</h1><p class="cj-mut">Checked {esc(when(parse_time(h["generated_utc"]), now))}. '
            f'Reload the page to check again.</p>'
            f'<div class="cj-admin-panels">{panel("svc", "Servers on this laptop", svc)}'
            f'{panel("gpu", "GPU computers", wat)}{panel("bak", "Backup", bak)}'
            f'{panel("dsk", "Disk space", dsk)}</div>')


def read_audit_all(cfg, n=AUDIT_SHOW) -> tuple:
    """This door's log plus the other doors' ($CJ_AUDIT_EXTRA, read only), merged by time. -> (entries with "door",
    list of problem notes)."""
    return PA.read_merged(cfg.state_dir, None, n)


AUDIT_LOGGED = ("Logged: every change made in the portal (uploads, predictions, conversions, results, review saves, "
                "gallery marks, feedback, admin and job actions, training), refused attempts, the first sign-in per "
                "person per day, and actions from the laptop CLI. Frequent saves are counted per 10 minutes.")
AUD_KINDS = (("feedback", "feedback"), ("review", "review saves"), ("predict", "predictions"),
             ("convert", "conversions"), ("results", "results"), ("gallery", "gallery marks"),
             ("sign-in", "sign-ins"), ("admin", "admin"), ("jobs", "jobs and training"), ("refused", "refused"),
             ("other", "other"))
_ADMIN_ACTS = {"person_added", "role_changed", "person_revoked", "email_changed", "key_made", "person added",
               "role changed", "person revoked", "sign-in email changed", "key link made", "agent token made",
               "agent token revoked"}
_JOB_ACTS = {"cancel", "release", "job cancel", "job release", "train_queue", "training queued", "training checked"}
REFUSED = {401: "refused", 403: "refused", 409: "refused"}


def aud_kind(action, status=None) -> str:
    """The Action filter's group for one line (AUD_KINDS)."""
    a = str(action or "").lower()
    if isinstance(status, int) and status in REFUSED:
        return "refused"
    if a == "signed in":
        return "sign-in"
    if a.startswith("feedback"):
        return "feedback"
    if a in _ADMIN_ACTS or a.startswith("predict_source") or a.startswith("predict session"):
        return "admin"
    if a in _JOB_ACTS or a.startswith("train"):
        return "jobs"
    if a.startswith("conversion"):
        return "convert"
    if a.startswith("results"):
        return "results"
    if a.startswith("gallery"):
        return "gallery"
    if a.startswith(("prediction", "predict", "upload")) or a == "review package made":
        return "predict"
    if a.startswith("review"):
        return "review"
    return "other"


def _clock(t, with_s=True) -> str:
    return datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S" if with_s else "%Y-%m-%d %H:%M") if t else "?"


def audit_row(e0, own_door="this door") -> dict:
    """One normalised line -> what the Recent actions table shows: {when, who, ai, what, door, unverified, kind}.
    A counter line (detail {from, to, n}) shows the time range and "n x action"; an agent's line (detail.via) shows
    actor.label(who, via) with the AI tag."""
    e0 = _d(e0)
    e = PA.normalise(e0)
    act = str(e.get("action") or "?")
    detail = e.get("detail")
    d = detail if isinstance(detail, dict) else {}
    word = ACTION_WORD.get(act, act.replace("_", " "))
    via = d.get("via") if isinstance(d.get("via"), str) else None
    status = d.get("status") if isinstance(d.get("status"), int) else None
    counted = isinstance(d.get("n"), int) and d.get("from") and d.get("to")
    if counted:
        t0, t1 = parse_time(_s(d.get("from"))), parse_time(_s(d.get("to")))
        end = datetime.fromtimestamp(t1).strftime("%H:%M") if t1 else "?"
        when = f"{_clock(t0, False)}–{end}"
        word = f"{d['n']} × {word}"
    else:
        when = _clock(parse_time(_s(e.get("t"))))
    if d.get("phase") == "requested":                   # write-ahead line of a Jobs / Train action
        word = f"asked to {word}"
    elif d.get("phase") == "result":
        word = f"{word} result for"
    skip = {"phase", "via", "status"} | ({"from", "to", "n"} if counted else set()) | ({"door"} if act == "signed in" else set())
    if isinstance(detail, dict):
        parts = [f"{k}: {v}" for k, v in detail.items() if v is not None and k not in skip]
    else:
        parts = [] if detail is None else [str(detail)]
    if status is not None and status >= 400:
        parts.insert(0, f"{REFUSED.get(status, 'answered')} {status}")
    door = str(e0.get("door") or "this door")
    door = own_door if door == "this door" else door
    name = ACTOR.label(e.get("actor") or "-", via) or "-"
    if e.get("actor") == PA.CLI_ACTOR:                         # the token CLI: whoever ran it, a declared via
        door = PA.CLI_DOOR
    unv = door == PA.CLI_DOOR or "unverified" in door          # a self-declared identity (the laptop CLI)
    if via and not unv and (act != "signed in" or d.get("door") == PA.API_DOOR):
        door = PA.API_DOOR                                      # a bearer token the portal verified
    return {"when": when, "name": name, "who": f"{name} (unverified)" if unv else name, "ai": bool(via),
            "what": word, "target": "" if e.get("target") is None else str(e.get("target")),
            "detail": ", ".join(parts), "door": door, "unverified": unv, "kind": aud_kind(act, status)}


def _options(name, label, values, all_text) -> str:
    opts = "".join(f'<option value="{esc(v)}">{esc(t)}</option>' for v, t in values)
    return (f'<label class="cj-admin-lbl">{esc(label)} <select class="cj-admin-in" data-cj-aud-filter="{name}">'
            f'<option value="">{esc(all_text)}</option>{opts}</select></label>')


def render_audit(entries, problem, own_door="this door") -> str:
    problems = [problem] if isinstance(problem, str) and problem else list(problem or [])
    warn = "".join(note(p, "warn") for p in problems)
    head = (f'<h1>Recent actions</h1><p class="cj-mut">{esc(AUDIT_LOGGED)} The last {AUDIT_SHOW} lines, newest '
            f'first; times are this computer\'s.</p>')
    if not entries:
        return head + warn + ('<p class="cj-mut">Nothing logged yet.</p>' if not problems else "")
    rows = [audit_row(e, own_door) for e in entries]
    items = []
    for r in rows:
        ai = ' <span class="cj-admin-ai" title="an agent acting for this person">AI</span>' if r["ai"] else ""
        if r["unverified"]:                         # its own Person-filter value: never mixed with portal sign-ins
            ai += (' <span class="cj-admin-ai cj-admin-unvtag" title="identity declared by the caller, not '
                   'checked by the portal">unverified</span>')
        tgt = f' <b class="cj-wrap">{esc(r["target"])}</b>' if r["target"] else ""
        dsp = f' <span class="cj-mut">({esc(r["detail"])})</span>' if r["detail"] else ""
        dcls = "cj-admin-door cj-admin-unv" if r["unverified"] else "cj-admin-door cj-mut"
        items.append(f'<li class="cj-admin-aud" data-who="{esc(r["who"])}" data-kind="{esc(r["kind"])}">'
                     f'<span class="cj-mut cj-num">{esc(r["when"])}</span> '      # spaces: the text reads as a
                     f'<span class="cj-admin-who"><b class="cj-wrap">{esc(r["name"])}</b>{ai}</span> '  # sentence
                     f'<span class="cj-admin-what">{esc(r["what"])}{tgt}{dsp}</span> '
                     f'<span class="{dcls}">{esc(r["door"])}</span></li>')
    people = sorted({r["who"] for r in rows}, key=str.lower)
    present = {r["kind"] for r in rows}
    filters = (f'<div class="cj-admin-audf" id="cj-admin-audf" hidden>'
               f'{_options("who", "Person", [(p, p) for p in people], "everyone")}'
               f'{_options("kind", "Action", [(k, t) for k, t in AUD_KINDS if k in present], "all")}'
               f'<span class="cj-mut">Filters search the last {AUDIT_SHOW} lines.</span>'
               f'<span class="cj-mut" id="cj-aud-n" aria-live="polite"></span></div>')
    return head + warn + filters + f'<ol class="cj-card cj-admin-auds">{"".join(items)}</ol>'


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
// Recent actions: the Person / Action filters, in the browser, over the lines on the page only. Shown only with
// script (the filter row is hidden in the HTML).
(function () {
  'use strict';
  var box = document.getElementById('cj-admin-audf');
  if (!box) return;
  var sels = box.querySelectorAll('[data-cj-aud-filter]');
  var rows = document.querySelectorAll('.cj-admin-aud[data-who]');
  var out = document.getElementById('cj-aud-n');
  function apply() {
    var f = {};
    Array.prototype.forEach.call(sels, function (s) { f[s.getAttribute('data-cj-aud-filter')] = s.value; });
    var n = 0;
    Array.prototype.forEach.call(rows, function (r) {
      var ok = (!f.who || r.getAttribute('data-who') === f.who) && (!f.kind || r.getAttribute('data-kind') === f.kind);
      r.hidden = !ok;
      if (ok) n++;
    });
    if (out) out.textContent = (f.who || f.kind) ? n + ' of ' + rows.length + ' shown' : '';
  }
  Array.prototype.forEach.call(sels, function (s) { s.addEventListener('change', apply); });
  box.hidden = false;
  apply();
})();
"""


# ---------------------------------------------------------------- routes
def routes(ctx) -> list:
    secret = secrets.token_bytes(32)                     # per process: a restart makes open forms stale

    def token(request: Request) -> str:
        sess = binding(request)                      # key: session cookie; access: the verified user
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

    def access_mode() -> bool:
        return getattr(ctx.cfg, "auth", "key") == "access"

    def people_page(request, msg_html="", status=200):
        try:
            rows = people(ctx.cfg.roster, email=True)
        except ValueError as e:
            return show(request, "People", "people", "<h1>People</h1>" + note(
                f"The roster can't be read ({e}). Nothing was changed.", "warn"), 500)
        toks: dict = {}
        try:
            for t in A.list_tokens(A.tokens_path(ctx.cfg)):
                toks.setdefault(t["person"], []).append(t)
        except A.TokenError as e:
            msg_html += note(f"Agent tokens can't be listed: {e}.", "warn")
        return show(request, "People", "people", render_people(rows, me(request), token(request), msg_html,
                                                       keys_ok=not access_mode(), tokens=toks), status)

    def try_audit(actor, action, target, detail) -> str:
        try:
            audit(ctx, actor, action, target, detail)
            return ""
        except OSError as e:
            log.error("admin: audit log not written (%s)", type(e).__name__)
            return note(f"The change was made, but the audit log could not be written ({type(e).__name__}).", "warn")

    def index(request: Request):                         # sync: Starlette runs it in a worker thread
        text = flash_take(binding(request), request.query_params.get("f"))
        return people_page(request, note(text) if text else "")

    def done(request: Request, text: str, to: str = PREFIX + "/"):
        """303 to People (or `to`) with a one-time message for this session (a bare URL shows nothing)."""
        tok = flash_put(binding(request), text)
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
        email = f.get("email", "")
        with_key = not email.strip() or (f.get("also_key") in ("1", "on") and not access_mode())
        if not email.strip() and access_mode():
            return show(request, "Add person", "people", note(KEYS_TAILNET_ONLY, "warn"), 403)
        key, err = await change(request, add_person, pid, role, email, with_key)
        if err:
            return err
        detail = {"role": role, "email": email.lower()} if email else {"role": role}
        if email and not with_key:
            detail["key"] = False
        warn = await run_in_threadpool(try_audit, actor, "person_added", pid, detail)
        if key is None:
            host = getattr(getattr(ctx.cfg, "access", None), "host", None) \
                or await run_in_threadpool(access_config.public_host)
            r = show(request, f"{pid} added", "people", warn + render_signin(pid, role, email.strip().lower(), host))
            r.headers["Cache-Control"] = "no-store"
            return r
        base = await run_in_threadpool(link_base, ctx.cfg, request.headers.get("origin"))
        link = f"{base}/login?k={key}"
        del key
        r = show(request, f"Key link for {pid}", "people", warn + render_key(pid, role, link))
        r.headers["Cache-Control"] = "no-store"
        r.headers["Pragma"] = "no-cache"
        return r

    async def makekey(request: Request):
        if access_mode():
            return show(request, "Key link", "people", note(KEYS_TAILNET_ONLY, "warn"), 403)
        f, err = await form(request)
        if err:
            return err
        pid, actor = f.get("id", ""), me(request)
        res, err = await change(request, make_key, pid)
        if err:
            return err
        key, had = res
        detail = {"rotated": had}
        if had:                                          # a re-key: the person's agent tokens go too
            try:
                n_tok = await run_in_threadpool(A.revoke_person_tokens, A.tokens_path(ctx.cfg), pid, actor)
                if n_tok:
                    detail["tokens_revoked"] = n_tok
            except (OSError, A.TokenError) as e:
                log.error("admin: %s's agent tokens not revoked on re-key (%s)", pid, type(e).__name__)
                detail["tokens_revoked"] = "failed"
        warn = await run_in_threadpool(try_audit, actor, "key_made", pid, detail)
        base = await run_in_threadpool(link_base, ctx.cfg, request.headers.get("origin"))
        link = f"{base}/login?k={key}"
        del key
        rows = await run_in_threadpool(people, ctx.cfg.roster)
        role_ = next((r["role"] for r in rows if r["id"] == pid), None)
        r = show(request, f"Key link for {pid}", "people",
                 warn + render_key(pid, role_ if role_ in ROLE_WORD else "basic", link, rotated=True))
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

    async def email(request: Request):
        f, err = await form(request)
        if err:
            return err
        pid, actor = f.get("id", ""), me(request)
        res, err = await change(request, set_email, pid, f.get("email", ""))
        if err:
            return err
        old, new = res
        warn = await run_in_threadpool(try_audit, actor, "email_changed", pid, {"from": old or None, "to": new or None})
        if warn:
            return await run_in_threadpool(people_page, request, warn)
        return done(request, f"{pid} now signs in through Cloudflare as {new}." if new else
                    f"{pid} has no sign-in email now (no Cloudflare sign-in).")

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
        try:                                             # their agent tokens die with them (the Gate also
            n_tok = await run_in_threadpool(A.revoke_person_tokens,      # refuses a revoked person's tokens)
                                            A.tokens_path(ctx.cfg), pid, actor)
        except (OSError, A.TokenError) as e:
            log.error("admin: %s's agent tokens not marked revoked (%s)", pid, type(e).__name__)
            n_tok = None
        detail = {"role": old}
        if n_tok:
            detail["tokens_revoked"] = n_tok
        warn = await run_in_threadpool(try_audit, actor, "person_revoked", pid, detail)
        if n_tok is None:
            warn += note(f"{pid} is revoked, but their agent tokens could not be marked revoked (they stop working "
                         f"anyway while {pid} is revoked). Revoke them below.", "warn")
        if warn:
            return await run_in_threadpool(people_page, request, warn)
        if pid == actor:
            return ctx.no_store(RedirectResponse("/", status_code=303))
        return done(request, f"{pid} is revoked and signed out.")

    async def token_revoke(request: Request):
        f, err = await form(request)
        if err:
            return err
        actor = me(request)
        try:
            rec = await run_in_threadpool(A.revoke_token, A.tokens_path(ctx.cfg), f.get("id", ""), actor)
        except A.TokenError as e:
            return await run_in_threadpool(people_page, request, note(f"Nothing was revoked: {e}", "warn"), 404)
        except OSError as e:
            log.error("admin: token store not written (%s)", type(e).__name__)
            return await run_in_threadpool(people_page, request, note(
                "Nothing was revoked: the token store is busy or can't be written. Try again.", "warn"), 503)
        warn = await run_in_threadpool(try_audit, actor, "agent token revoked", rec["id"],
                                       {"for": rec["person"], "agent": rec["agent"], "scope": rec["scope"],
                                        "expires": rec["expires"]})
        if warn:
            return await run_in_threadpool(people_page, request, warn)
        return done(request, f"Token {rec['id']} ({ACTOR.label(rec['person'], rec['agent'])}) is revoked.")

    async def health_page(request: Request):
        h = await asyncio.to_thread(health, ctx.cfg)
        return show(request, "Health", "health", render_health(h))

    async def api_health(request: Request):
        return ctx.json(await asyncio.to_thread(health, ctx.cfg))

    def audit_page(request: Request):
        entries, prob = read_audit_all(ctx.cfg)
        own = PA.DOOR_NAMES.get(getattr(ctx.cfg, "auth", "key"), "this door")
        return show(request, "Recent actions", "audit", render_audit(entries, prob, own))

    def api_audit(request: Request):
        entries, prob = read_audit_all(ctx.cfg)
        return ctx.json({"schema": "cj_admin_audit/1", "entries": entries, "problem": prob[0] if prob else None,
                         "problems": prob})

    def api_people(request: Request):
        try:
            return ctx.json({"schema": "cj_admin_people/1", "people": people(ctx.cfg.roster)})
        except ValueError as e:
            return ctx.json({"schema": "cj_admin_people/1", "people": [], "problem": str(e)}, 500)

    # ---- Predict sessions
    def predict_page(request: Request, msg_html="", status=200):
        if not msg_html:
            text = flash_take(binding(request), request.query_params.get("f"))
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
            Route(PREFIX + "/people/email", email, methods=["POST"]),
            Route(PREFIX + "/people/key", makekey, methods=["POST"]),
            Route(PREFIX + "/people/revoke", revoke_page, methods=["GET"]),
            Route(PREFIX + "/people/revoke", revoke, methods=["POST"]),
            Route(PREFIX + "/people/token-revoke", token_revoke, methods=["POST"]),
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
