"""
CJ portal sign-in, tailnet mode (P1): key links -> session cookie. Deny by default; see portal/README.md.

    python -m portal.auth add-key --id <id> --role admin [--roster <path>] [--base <portal address>]
    python -m portal.auth revoke --id <id> [--roster <path>]
    python -m portal.auth list [--roster <path>]

- Roster: a private file OUTSIDE the repo, default %USERPROFILE%\\.cryoev\\portal_roster.json:
  {"people": [{"id": "...", "role": "admin", "key_sha256": "...", "added": "..."}]}. Keys are stored only as
  sha256; `add-key` prints the new key link once and never again. Re-keying an id ends its old sessions.
- Login: /login?k=<key> -> sha256 compared with hmac.compare_digest against every roster entry -> a random
  32-byte session id in the cookie __Host-cj_session (HttpOnly, Secure, SameSite=Lax, Path=/, 30 days).
- Sessions persist in <state dir>/sessions.json (default %USERPROFILE%\\.cryoev\\portal\\), keyed by the sha256 of
  the session id only, each tied to the person's id and key hash. The roster is checked on every request (re-read
  when the file changes), so removing or revoking a person, or re-keying them, ends their sessions at once.
- Neither keys nor session ids are ever logged.
A later `access` mode (Cloudflare, C3) gets its own port and middleware; it is never mixed with key links.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import hmac
import json
import os
import re
import secrets
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from data_utils.oslock import acquire_lock  # noqa: E402

COOKIE = "__Host-cj_session"
SESSION_S = 30 * 86400            # 30 days
ROLES = ("admin", "advanced", "basic")   # admin > advanced > basic; modules list every role they allow
ID_RE = re.compile(r"[a-z0-9][a-z0-9_.-]{0,39}")
KEY_BYTES = 32


def default_dir() -> Path:
    return Path.home() / ".cryoev"


def default_roster() -> Path:
    return default_dir() / "portal_roster.json"


def default_state_dir() -> Path:
    return default_dir() / "portal"


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json_atomic(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def cookie_header(token: str, max_age: int = SESSION_S) -> str:
    return f"{COOKIE}={token}; Path=/; Max-Age={max_age}; HttpOnly; Secure; SameSite=Lax"


class Roster:
    """The private roster, re-read whenever the file's mtime or size changes (checked on every request).
    A missing or unreadable roster denies everyone."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._sig = None
        self._people: dict = {}
        self._lock = threading.Lock()

    def people(self) -> dict:
        try:
            st = self.path.stat()
            sig = (st.st_mtime_ns, st.st_size)
        except OSError:
            return {}
        with self._lock:
            if sig != self._sig:
                try:
                    doc = json.loads(self.path.read_text(encoding="utf-8"))
                    people = {p["id"]: p for p in doc.get("people", []) if isinstance(p, dict) and p.get("id")}
                except (OSError, ValueError, AttributeError, KeyError, TypeError):
                    people = {}
                self._people, self._sig = people, sig
            return self._people

    @staticmethod
    def usable(p) -> bool:
        return bool(p) and not p.get("revoked") and p.get("role") in ROLES and isinstance(p.get("key_sha256"), str)

    def match_key(self, key: str):
        """The person whose key this is, or None. Every entry is compared (no early exit)."""
        h = sha256_hex(key)
        found = None
        for p in self.people().values():
            ok = hmac.compare_digest(h.encode(), str(p.get("key_sha256", "")).encode())
            if ok and self.usable(p):
                found = p
        return found

    def active(self, pid: str, key_sha: str):
        """The roster entry for a session's person, if it is still valid with the same key; else None."""
        p = self.people().get(pid)
        if not self.usable(p):
            return None
        return p if hmac.compare_digest(str(p["key_sha256"]).encode(), str(key_sha).encode()) else None


class Sessions:
    """sha256(session id) -> {id, key_sha256, created, expires}, persisted to <state dir>/sessions.json."""

    def __init__(self, state_dir: Path):
        self.path = Path(state_dir) / "sessions.json"
        self._lock = threading.Lock()
        try:
            self._s = json.loads(self.path.read_text(encoding="utf-8")).get("sessions", {})
        except (OSError, ValueError, AttributeError):
            self._s = {}

    def _save(self):
        write_json_atomic(self.path, {"schema": "cj_sessions/1", "sessions": self._s})

    def create(self, person: dict, now: float | None = None) -> str:
        now = time.time() if now is None else now
        token = secrets.token_urlsafe(KEY_BYTES)
        with self._lock:
            self._s = {h: r for h, r in self._s.items() if r.get("expires", 0) > now}
            self._s[sha256_hex(token)] = {"id": person["id"], "key_sha256": person["key_sha256"],
                                          "created": int(now), "expires": int(now + SESSION_S)}
            self._save()
        return token

    def lookup(self, token: str, now: float | None = None):
        now = time.time() if now is None else now
        rec = self._s.get(sha256_hex(token))
        return rec if rec and rec.get("expires", 0) > now else None


class Auth:
    def __init__(self, roster_path: Path, state_dir: Path):
        self.roster = Roster(roster_path)
        self.sessions = Sessions(state_dir)

    def login(self, key: str):
        """-> (session token, person) for a valid key, else None."""
        if not key or len(key) > 512:
            return None
        p = self.roster.match_key(key)
        if p is None:
            return None
        return self.sessions.create(p), p

    def check(self, token: str | None):
        """-> ("none", None) without a cookie, ("denied", None) for an unknown / expired / revoked session,
        ("ok", person) otherwise."""
        if not token:
            return "none", None
        rec = self.sessions.lookup(token)
        if rec is None:
            return "denied", None
        p = self.roster.active(rec.get("id", ""), rec.get("key_sha256", ""))
        return ("ok", p) if p else ("denied", None)


# ---------------------------------------------------------------- roster tool
ROSTER_LOCK_WAIT_S = 10.0
_roster_rlock = threading.RLock()
_roster_depth = [0]


@contextlib.contextmanager
def roster_lock(path: Path, wait: float = ROSTER_LOCK_WAIT_S):
    """Every roster read-modify-write (this tool, the portal's Admin page) runs inside this: a re-entrant lock in
    this process plus an OS lock on <roster>.lock for other processes. Raises OSError if it can't be had in `wait`
    seconds (nothing is changed then)."""
    with _roster_rlock:
        held = None
        if _roster_depth[0] == 0:
            lp = Path(path).with_name(Path(path).name + ".lock")
            lp.parent.mkdir(parents=True, exist_ok=True)
            held = acquire_lock(lp, wait=wait)
            if held is None:
                raise OSError("the roster is being changed by another process; try again")
        _roster_depth[0] += 1
        try:
            yield
        finally:
            _roster_depth[0] -= 1
            if held is not None:
                held.close()


def _load(path: Path) -> dict:
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"schema": "cj_roster/1", "people": []}
    if not isinstance(doc.get("people"), list):
        raise SystemExit(f"{path}: no people list; not touching it")
    return doc


def add_key(path: Path, pid: str, role: str) -> str:
    """Give `pid` a new key (replacing any old one, which ends its sessions). -> the key, shown once."""
    if not ID_RE.fullmatch(pid):
        raise ValueError(f"bad id {pid!r}: lower-case letters, digits, '_', '.', '-'")
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}")
    with roster_lock(path):
        doc = _load(path)
        key = secrets.token_urlsafe(KEY_BYTES)
        entry = {"id": pid, "role": role, "key_sha256": sha256_hex(key),
                 "added": datetime.now().isoformat(timespec="seconds")}
        doc["people"] = [p for p in doc["people"] if p.get("id") != pid] + [entry]
        write_json_atomic(path, doc)
    return key


def revoke(path: Path, pid: str) -> bool:
    with roster_lock(path):
        doc = _load(path)
        hit = False
        for p in doc["people"]:
            if p.get("id") == pid and not p.get("revoked"):
                p["revoked"] = datetime.now().isoformat(timespec="seconds")
                hit = True
        if hit:
            write_json_atomic(path, doc)
    return hit


def default_base() -> str:
    """https://<this PC's tailnet name>:8446 when Tailscale answers, else the local address."""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "annotation"))
        from review_server import tailscale_hosts
        names = [h for h in tailscale_hosts() if not h[:1].isdigit()]
    except Exception:
        names = []
    return f"https://{names[0]}:8446" if names else "http://127.0.0.1:8770"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add-key", help="new key for an id (prints the key link once)")
    a.add_argument("--id", required=True)
    a.add_argument("--role", required=True, choices=ROLES)
    a.add_argument("--base", default=None, help="portal address for the printed link (default: the tailnet :8446)")
    r = sub.add_parser("revoke", help="revoke an id (its sessions end on their next request)")
    r.add_argument("--id", required=True)
    sub.add_parser("list", help="ids, roles and state (never keys)")
    for p in (a, r, sub.choices["list"]):
        p.add_argument("--roster", type=Path, default=None, help=f"default {default_roster()}")
    args = ap.parse_args(argv)
    path = args.roster or default_roster()
    if args.cmd == "add-key":
        try:
            key = add_key(path, args.id, args.role)
        except ValueError as e:
            ap.error(str(e))
        base = (args.base or default_base()).rstrip("/")
        print(f"[OK] {args.id} ({args.role}) added to {path}. Key link (shown once, keep it private):")
        print(f"{base}/login?k={key}")
        return 0
    if args.cmd == "revoke":
        ok = revoke(path, args.id)
        print(f"[OK] {args.id} revoked" if ok else f"[--] {args.id} not found or already revoked")
        return 0 if ok else 1
    for p in _load(path)["people"]:
        print(f"{p.get('id')}\t{p.get('role')}\t{'revoked ' + p['revoked'] if p.get('revoked') else 'active'}"
              f"\tadded {p.get('added', '?')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
