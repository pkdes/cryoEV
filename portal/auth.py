"""
CryoEV portal sign-in, tailnet mode (P1): key links -> session cookie. Deny by default; see portal/README.md.

    python -m portal.auth add-key --id <id> --role admin [--email <address>] [--roster <path>] [--base <address>]
    python -m portal.auth set-email --id <id> --email <address or ''> [--roster <path>]
    python -m portal.auth revoke --id <id> [--roster <path>]
    python -m portal.auth list [--roster <path>]
    python -m portal.auth mint-token --id <id> --agent <name> [--days 30] [--tokens <path>] [--roster <path>]
    python -m portal.auth list-tokens [--id <id>] [--tokens <path>]
    python -m portal.auth revoke-token --token-id <tk_...> [--tokens <path>]

- Roster: a private file OUTSIDE the repo, default %USERPROFILE%\\.cryoev\\portal_roster.json:
  {"people": [{"id": "...", "role": "admin", "key_sha256": "...", "added": "...", "email": "..."}]}. Keys are stored
  only as sha256; `add-key` prints the new key link once and never again. Re-keying an id ends its old sessions and
  keeps its email. `email` (lower-case, unique among active people) is the sign-in for Cloudflare access mode
  (portal/access.py, Roster.usable_access / match_email); key mode ignores it.
- Login: /login?k=<key> -> sha256 compared with hmac.compare_digest against every roster entry -> a random
  32-byte session id in the cookie __Host-cj_session (HttpOnly, Secure, SameSite=Lax, Path=/, 30 days).
- Sessions persist in <state dir>/sessions.json (default %USERPROFILE%\\.cryoev\\portal\\), keyed by the sha256 of
  the session id only, each tied to the person's id and key hash. The roster is checked on every request (re-read
  when the file changes), so removing or revoking a person, or re-keying them, ends their sessions at once.
- Neither keys nor session ids are ever logged.
- Agent tokens (round 10 #60, docs/API.md): one shared store for every door, default
  %USERPROFILE%\\.cryoev\\agent_tokens.json (env CJ_AGENT_TOKENS), kept OUT of the roster (add-key rebuilds roster
  entries). {"tokens": [{id, person, agent, scope, sha256, created, expires, revoked, revoked_by, last_used}]}: the
  token itself (cev_ + 32 random bytes, base64url) is shown once and stored only as sha256. Every read-modify-write
  takes tokens_lock (thread lock + OS lock on <store>.lock) and re-reads the file; the portal's TokenStore re-reads
  on (mtime, size) like Roster, looks tokens up by sha256, keeps last_used in memory and merges it into the file at
  most every LAST_USED_FLUSH_S (the merge sets last_used only: it never clears a revoke). A token is valid when it is
  not revoked or expired, its scope is "read" ("write" is reserved for the next round and refused), and its person
  is on the roster, not revoked, with a known role (no key needed: email-only people count). Revoking a person
  (this CLI, Admin) revokes their tokens too.
The Cloudflare `access` mode (portal/access.py) runs on its own port; it is never mixed with key links.
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
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from data_utils import actor as ACTOR  # noqa: E402
from data_utils.oslock import acquire_lock  # noqa: E402

COOKIE = "__Host-cj_session"
SESSION_S = 30 * 86400            # 30 days
ROLES = ("admin", "advanced", "basic")   # admin > advanced > basic; modules list every role they allow
ID_RE = re.compile(r"[a-z0-9][a-z0-9_.-]{0,39}")
EMAIL_RE = re.compile(r"(?=.{3,254}\Z)[a-z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-z0-9](?:[a-z0-9-]*[a-z0-9])?(?:\.[a-z0-9](?:[a-z0-9-]*[a-z0-9])?)+")
KEY_BYTES = 32
REPLACE_TRIES = 10                # os.replace retries on PermissionError; backoff totals ~0.23 s
REPLACE_PAUSE_S = 0.005


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
    try:
        tmp.write_text(json.dumps(obj, indent=1), encoding="utf-8")
        # Windows: os.replace raises PermissionError while another process briefly has the file open
        for attempt in range(REPLACE_TRIES):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                if attempt == REPLACE_TRIES - 1:
                    raise
                time.sleep(REPLACE_PAUSE_S * (attempt + 1))
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


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

    @staticmethod
    def usable_access(p) -> bool:
        """Access mode's `usable`: not revoked, a known role and an email (no key needed)."""
        return (bool(p) and not p.get("revoked") and p.get("role") in ROLES
                and isinstance(p.get("email"), str) and bool(p["email"].strip()))

    def match_email(self, email: str):
        """The ONE access-usable person with this email (case-insensitive), else None (none, or a hand-edited
        roster with two: refuse rather than guess)."""
        e = access_email(email)
        if not e:
            return None
        hits = [p for p in self.people().values() if self.usable_access(p) and access_email(p["email"]) == e]
        return hits[0] if len(hits) == 1 else None

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


def norm_email(email) -> str:
    """'' for none, else the address lower-cased. Raises ValueError for something that isn't an address."""
    if email is None:
        return ""
    e = str(email).strip()
    if e and access_email(e) is None:
        raise ValueError(f"{email!r} is not a plain (ASCII) email address")
    return e.lower()


def access_email(email) -> str | None:
    """A plain ASCII address (EMAIL_RE), lower-cased; else None. ASCII is checked BEFORE lower-casing: str.lower()
    maps some non-ASCII letters onto ASCII ones (U+212A KELVIN SIGN -> 'k'), which would let one address pose as
    another."""
    if not isinstance(email, str):
        return None
    e = email.strip()
    if not e or not e.isascii():
        return None
    e = e.lower()
    return e if EMAIL_RE.fullmatch(e) else None


def email_taken(people, email: str, pid: str) -> str | None:
    """The id of another ACTIVE (not revoked) person who has `email` already, else None."""
    if not email:
        return None
    for p in people:
        if (isinstance(p, dict) and p.get("id") != pid and not p.get("revoked")
                and str(p.get("email") or "").strip().lower() == email):
            return str(p.get("id"))
    return None


def add_key(path: Path, pid: str, role: str, email: str | None = None) -> str:
    """Give `pid` a new key (replacing any old one, which ends its sessions). -> the key, shown once.
    The entry's email is kept (or set to `email`; emails are unique among active people)."""
    if not ID_RE.fullmatch(pid):
        raise ValueError(f"bad id {pid!r}: lower-case letters, digits, '_', '.', '-'")
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}")
    new_email = norm_email(email) if email is not None else None
    with roster_lock(path):
        doc = _load(path)
        old = [p for p in doc["people"] if isinstance(p, dict) and p.get("id") == pid]
        keep = new_email if new_email is not None else str((old[-1] if old else {}).get("email") or "")
        other = email_taken(doc["people"], keep, pid)
        if other:
            raise ValueError(f"{keep} is already the email of {other}")
        key = secrets.token_urlsafe(KEY_BYTES)
        entry = {"id": pid, "role": role, "key_sha256": sha256_hex(key),
                 "added": datetime.now().isoformat(timespec="seconds")}
        if keep:
            entry["email"] = keep
        doc["people"] = [p for p in doc["people"] if p.get("id") != pid] + [entry]
        write_json_atomic(path, doc)
    return key


def set_email(path: Path, pid: str, email: str) -> str:
    """Set (or with '' remove) an active person's email; unique among active people. -> the old email ('' if none).
    Raises ValueError (bad address, taken, no such active person)."""
    e = norm_email(email)
    with roster_lock(path):
        doc = _load(path)
        hits = [p for p in doc["people"] if isinstance(p, dict) and p.get("id") == pid and not p.get("revoked")]
        if not hits:
            raise ValueError(f"{pid} is not on the list, or revoked")
        other = email_taken(doc["people"], e, pid)
        if other:
            raise ValueError(f"{e} is already the email of {other}")
        old = str(hits[-1].get("email") or "")
        for p in hits:
            if e:
                p["email"] = e
            else:
                p.pop("email", None)
        write_json_atomic(path, doc)
    return old


def revoke(path: Path, pid: str, tokens: Path | None = None) -> bool:
    """Revoke an id. With `tokens` (a token store path), that person's agent tokens are revoked too."""
    with roster_lock(path):
        doc = _load(path)
        hit = False
        for p in doc["people"]:
            if p.get("id") == pid and not p.get("revoked"):
                p["revoked"] = datetime.now().isoformat(timespec="seconds")
                hit = True
        if hit:
            write_json_atomic(path, doc)
    if hit and tokens is not None:
        try:
            revoke_person_tokens(tokens, pid, by="cli")
        except (OSError, TokenError) as e:          # the person is revoked; their tokens fail the person check
            print(f"[WARN] {pid}'s agent tokens were not marked revoked ({e}); they stop working anyway while "
                  f"{pid} is revoked. Run revoke-token for each.", file=sys.stderr)
    return hit


# ---------------------------------------------------------------- agent tokens (round 10 #60; docs/API.md)
TOKENS_ENV = "CJ_AGENT_TOKENS"
TOKEN_PREFIX = "cev_"
TOKEN_RE = re.compile(r"cev_[A-Za-z0-9_-]{43}\Z")       # tools/prepush_scan.py blocks this shape in pushes
TOKEN_ID_RE = re.compile(r"tk_[0-9a-f]{12}\Z")
TOKEN_MAX_DAYS = 90
TOKEN_DEFAULT_DAYS = 30
TOKENS_PER_PERSON = 10                                  # live (not revoked, not expired) tokens per person
TOKEN_SCOPES = ("read",)                                # what a token may carry now
RESERVED_SCOPES = ("write",)                            # next round (a flag in the store); refused tonight
LAST_USED_FLUSH_S = 300.0
TOKENS_LOCK_WAIT_S = 10.0
_tokens_rlock = threading.RLock()
_tokens_depth = [0]


class TokenError(ValueError):
    """A token request that is refused; the message is shown as is."""


def default_tokens() -> Path:
    return Path(os.environ.get(TOKENS_ENV) or (default_dir() / "agent_tokens.json"))


def tokens_path(cfg) -> Path:
    """The store a portal Config uses: cfg.tokens when set, else default_tokens()."""
    t = getattr(cfg, "tokens", None)
    return Path(t) if t else default_tokens()


def _utc(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).isoformat(timespec="seconds")


def _ts(iso) -> float | None:
    try:
        t = datetime.fromisoformat(str(iso))
    except (TypeError, ValueError):
        return None
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return t.timestamp()


@contextlib.contextmanager
def tokens_lock(path: Path, wait: float = TOKENS_LOCK_WAIT_S):
    """Every token-store read-modify-write runs inside this (as roster_lock): a re-entrant lock in this process plus
    an OS lock on <store>.lock for other processes (the other door, the CLI). Raises OSError after `wait` s."""
    with _tokens_rlock:
        held = None
        if _tokens_depth[0] == 0:
            lp = Path(path).with_name(Path(path).name + ".lock")
            lp.parent.mkdir(parents=True, exist_ok=True)
            held = acquire_lock(lp, wait=wait)
            if held is None:
                raise OSError("the token store is being changed by another process; try again")
        _tokens_depth[0] += 1
        try:
            yield
        finally:
            _tokens_depth[0] -= 1
            if held is not None:
                held.close()


def _load_tokens(path: Path) -> dict:
    """The store's document; a missing file is empty. An unreadable one raises TokenError (never overwritten)."""
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"schema": "cj_agent_tokens/1", "tokens": []}
    except (OSError, ValueError) as e:
        raise TokenError(f"the token store can't be read ({type(e).__name__})") from None
    if not isinstance(doc, dict) or not isinstance(doc.get("tokens"), list):
        raise TokenError("the token store has no token list")
    return doc


def token_state(rec: dict, now: float | None = None) -> str:
    """'revoked', 'expired' or 'active' (the person's own state is not looked at here)."""
    now = time.time() if now is None else now
    if rec.get("revoked"):
        return "revoked"
    exp = _ts(rec.get("expires"))
    return "expired" if exp is None or exp <= now else "active"


def public_token(rec: dict, now: float | None = None) -> dict:
    """The only token fields that leave the store (never sha256)."""
    out = {k: rec.get(k) for k in ("id", "person", "agent", "scope", "created", "expires", "revoked",
                                   "revoked_by", "last_used")}
    out["state"] = token_state(rec, now)
    return out


def person_ok(p) -> bool:
    """A person a token may act for: on the roster, not revoked, a known role. No key needed (QC 3: email-only
    people count), so Roster.usable is NOT used."""
    return isinstance(p, dict) and not p.get("revoked") and p.get("role") in ROLES


def create_token(path: Path, person: str, agent: str, days: int = TOKEN_DEFAULT_DAYS, scope: str = "read",
                 roster: Path | None = None, now: float | None = None) -> tuple:
    """-> (token, public record). The token is returned once and stored only as sha256. Raises TokenError: bad
    agent name / days / scope, a person who isn't active on `roster` (when given), TOKENS_PER_PERSON live tokens
    already; OSError when the store is locked or can't be written."""
    now = time.time() if now is None else now
    if not isinstance(person, str) or not ID_RE.fullmatch(person):
        raise TokenError("bad person id")
    if not isinstance(agent, str) or not ACTOR.VIA_RE.match(agent):
        raise TokenError("The agent name is 1 to 32 lower-case letters, digits, '_' or '-', starting with a letter "
                         "(e.g. claude).")
    if scope in RESERVED_SCOPES:
        raise TokenError(f"The {scope!r} scope is not available yet: tokens are read-only for now.")
    if scope not in TOKEN_SCOPES:
        raise TokenError(f"Unknown scope {scope!r}.")
    try:
        days = int(days)
    except (TypeError, ValueError):
        raise TokenError(f"Expiry is 1 to {TOKEN_MAX_DAYS} days.") from None
    if not 1 <= days <= TOKEN_MAX_DAYS:
        raise TokenError(f"Expiry is 1 to {TOKEN_MAX_DAYS} days.")
    if roster is not None and not person_ok(Roster(roster).people().get(person)):
        raise TokenError(f"{person} is not on the list, or revoked.")
    with tokens_lock(path):
        doc = _load_tokens(path)
        live = [r for r in doc["tokens"] if isinstance(r, dict) and r.get("person") == person
                and token_state(r, now) == "active"]
        if len(live) >= TOKENS_PER_PERSON:
            raise TokenError(f"{person} has {TOKENS_PER_PERSON} live tokens already. Revoke one first.")
        token = TOKEN_PREFIX + secrets.token_urlsafe(32)
        ids = {r.get("id") for r in doc["tokens"] if isinstance(r, dict)}
        tid = "tk_" + secrets.token_hex(6)
        while tid in ids:
            tid = "tk_" + secrets.token_hex(6)
        rec = {"id": tid, "person": person, "agent": agent, "scope": scope, "sha256": sha256_hex(token),
               "created": _utc(now), "expires": _utc(now + days * 86400), "revoked": None, "revoked_by": None,
               "last_used": None}
        doc["tokens"].append(rec)
        write_json_atomic(path, doc)
    return token, public_token(rec, now)


def list_tokens(path: Path, person: str | None = None, now: float | None = None) -> list:
    """Public records (no sha256), newest first; only `person`'s when given. Raises TokenError when unreadable."""
    doc = _load_tokens(path)
    rows = [public_token(r, now) for r in doc["tokens"] if isinstance(r, dict) and r.get("id")
            and (person is None or r.get("person") == person)]
    rows.sort(key=lambda r: str(r.get("created") or ""), reverse=True)
    return rows


def revoke_token(path: Path, tid: str, by: str, person: str | None = None, now: float | None = None) -> dict:
    """Revoke one token. `person` set: only that person's own token (My account). -> the public record. Raises
    TokenError (unknown, not theirs, already revoked) or OSError (locked / not written)."""
    now = time.time() if now is None else now
    if not isinstance(tid, str) or not TOKEN_ID_RE.fullmatch(tid):
        raise TokenError("No such token.")
    with tokens_lock(path):
        doc = _load_tokens(path)
        hit = next((r for r in doc["tokens"] if isinstance(r, dict) and r.get("id") == tid), None)
        if hit is None or (person is not None and hit.get("person") != person):
            raise TokenError("No such token.")
        if hit.get("revoked"):
            raise TokenError(f"Token {tid} is already revoked.")
        hit["revoked"], hit["revoked_by"] = _utc(now), str(by or "-")[:80]
        write_json_atomic(path, doc)
    return public_token(hit, now)


def revoke_person_tokens(path: Path, person: str, by: str, now: float | None = None) -> int:
    """Revoke every unrevoked token of `person` (the person was revoked). -> how many. A missing store: 0, nothing
    created (not even the lock file)."""
    if not Path(path).exists():
        return 0
    now = time.time() if now is None else now
    with tokens_lock(path):
        doc = _load_tokens(path)
        hits = [r for r in doc["tokens"] if isinstance(r, dict) and r.get("person") == person and not r.get("revoked")]
        for r in hits:
            r["revoked"], r["revoked_by"] = _utc(now), str(by or "-")[:80]
        if hits:
            write_json_atomic(path, doc)
    return len(hits)


class TokenStore:
    """The portal's view of the shared store: re-read when the file's (mtime, size) change (one stat per bearer
    request, as Roster), looked up by sha256 (a dict). last_used is kept in memory and merged into the file by
    flush() at most every flush_s (the portal's sweeper; shutdown flushes once more). A missing or unreadable store
    refuses every token."""

    def __init__(self, path: Path, flush_s: float = LAST_USED_FLUSH_S, clock=time.time):
        self.path = Path(path)
        self.flush_s = float(flush_s)
        self.clock = clock
        self._sig = None
        self._by_sha: dict = {}
        self._lock = threading.Lock()
        self._used: dict = {}                 # token id -> last use (epoch s), not yet in the file
        self._flushed = clock()

    def _index(self) -> dict:
        try:
            st = self.path.stat()
            sig = (st.st_mtime_ns, st.st_size)
        except OSError:
            return {}
        with self._lock:
            if sig != self._sig:
                try:
                    doc = _load_tokens(self.path)
                    idx = {r["sha256"]: r for r in doc["tokens"]
                           if isinstance(r, dict) and isinstance(r.get("sha256"), str) and r.get("id")}
                except (TokenError, KeyError, TypeError):
                    idx = {}
                self._by_sha, self._sig = idx, sig
            return self._by_sha

    def check(self, token, roster: Roster, now: float | None = None) -> tuple:
        """-> (person entry, token record) for a valid token, else (None, reason). reason: "malformed", "unknown",
        "revoked", "expired", "scope" (anything but read) or "person" (not on the roster, revoked, no valid role)."""
        now = self.clock() if now is None else now
        if not isinstance(token, str) or not TOKEN_RE.fullmatch(token):
            return None, "malformed"
        rec = self._index().get(sha256_hex(token))
        if rec is None:
            return None, "unknown"
        state = token_state(rec, now)
        if state != "active":
            return None, state
        made = _ts(rec.get("created"))
        if made is None or _ts(rec["expires"]) - made > TOKEN_MAX_DAYS * 86400 + 60:   # a hand-edited lifetime
            return None, "expired"
        if rec.get("scope") not in TOKEN_SCOPES or not isinstance(rec.get("agent"), str) \
                or not ACTOR.VIA_RE.match(rec["agent"]):
            return None, "scope"
        p = roster.people().get(rec.get("person"))
        if not person_ok(p):
            return None, "person"
        return p, rec

    def used(self, tid: str, now: float | None = None) -> None:
        with self._lock:
            self._used[tid] = self.clock() if now is None else now

    def flush(self, force: bool = False) -> int:
        """Merge last_used into the file (at most every flush_s unless force). Only last_used is set, and only when
        newer: revoked and everything else stay as the file has them. -> how many records changed. Never raises
        (a locked store keeps the times for the next flush)."""
        now = self.clock()
        with self._lock:
            if not self._used or (not force and now - self._flushed < self.flush_s):
                return 0
            used, self._used = self._used, {}
            self._flushed = now
        n = 0
        try:
            with tokens_lock(self.path, wait=2.0):
                doc = _load_tokens(self.path)
                for r in doc["tokens"]:
                    t = used.get(r.get("id")) if isinstance(r, dict) else None
                    if t is not None and (_ts(r.get("last_used")) or 0) < int(t):
                        r["last_used"] = _utc(t)
                        n += 1
                if n:
                    write_json_atomic(self.path, doc)
        except (OSError, TokenError):
            with self._lock:                          # keep them (newer marks win) for the next flush
                for k, t in used.items():
                    self._used[k] = max(t, self._used.get(k, 0))
            return 0
        return n


def default_base() -> str:
    """https://<this PC's tailnet name>:8446 when Tailscale answers, else the local address."""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "annotation"))
        from review_server import tailscale_hosts
        names = [h for h in tailscale_hosts() if not h[:1].isdigit()]
    except Exception:
        names = []
    return f"https://{names[0]}:8446" if names else "http://127.0.0.1:8770"


def token_audit(state_dir, action, rec) -> None:
    """The audit line for a CLI token change (portal/audit.py; never the token or its hash). The actor is "cli"; an
    agent session's CRYOEV_VIA rides along as detail.via. A failed write is a warning, never an error."""
    from portal import audit as PA
    detail = {"for": rec.get("person"), "agent": rec.get("agent"), "scope": rec.get("scope"),
              "expires": rec.get("expires")}
    try:
        v = ACTOR.via()
    except ACTOR.ActorError:
        v = None
    if v:
        detail["via"] = v
    try:
        PA.audit(state_dir or default_state_dir(), "cli", action, rec.get("id"), detail)
    except OSError as e:
        print(f"[WARN] audit line not written ({type(e).__name__})", file=sys.stderr)


def token_cli(args) -> int:
    path = args.tokens or default_tokens()
    try:
        if args.cmd == "mint-token":
            token, rec = create_token(path, args.id, args.agent, args.days, args.scope,
                                      roster=args.roster or default_roster())
            token_audit(args.state_dir, "agent token made", rec)
            print(f"[OK] token {rec['id']} for {args.agent} acting for {args.id} (read-only, expires "
                  f"{rec['expires']}). Shown once, keep it private:")
            print(token)
            return 0
        if args.cmd == "revoke-token":
            rec = revoke_token(path, args.token_id, by="cli")
            token_audit(args.state_dir, "agent token revoked", rec)
            print(f"[OK] {rec['id']} ({rec['agent']} for {rec['person']}) revoked")
            return 0
        rows = list_tokens(path, args.id)
    except (TokenError, OSError) as e:
        print(f"[--] {e}")
        return 1
    for t in rows:
        print(f"{t['id']}\t{t['person']}\t{t['agent']}\t{t['scope']}\t{t['state']}\tcreated {t['created']}"
              f"\texpires {t['expires']}\tlast used {t['last_used'] or '-'}")
    if not rows:
        print("(no agent tokens)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add-key", help="new key for an id (prints the key link once)")
    a.add_argument("--id", required=True)
    a.add_argument("--role", required=True, choices=ROLES)
    a.add_argument("--base", default=None, help="portal address for the printed link (default: the tailnet :8446)")
    a.add_argument("--email", default=None, help="sign-in email for Cloudflare access mode (kept if not given)")
    r = sub.add_parser("revoke", help="revoke an id (its sessions end on their next request)")
    r.add_argument("--id", required=True)
    e = sub.add_parser("set-email", help="set an active id's sign-in email for access mode ('' removes it)")
    e.add_argument("--id", required=True)
    e.add_argument("--email", required=True)
    sub.add_parser("list", help="ids, roles, emails and state (never keys)")
    m = sub.add_parser("mint-token", help="a read-only agent token for a person (printed once); docs/API.md")
    m.add_argument("--id", required=True, help="the person the agent acts for")
    m.add_argument("--agent", required=True, help="the agent's name, e.g. claude (shown as '<agent> for <id>')")
    m.add_argument("--days", type=int, default=TOKEN_DEFAULT_DAYS, help=f"1 to {TOKEN_MAX_DAYS} (default "
                                                                        f"{TOKEN_DEFAULT_DAYS})")
    m.add_argument("--scope", default="read", help="read (write is not available yet)")
    lt = sub.add_parser("list-tokens", help="agent tokens: id, person, agent, dates, state (never the token)")
    lt.add_argument("--id", default=None, help="only this person's")
    rt = sub.add_parser("revoke-token", help="revoke one agent token (it stops working on its next request)")
    rt.add_argument("--token-id", required=True, help="tk_... (from list-tokens)")
    for p in (a, r, e, sub.choices["list"], m):
        p.add_argument("--roster", type=Path, default=None, help=f"default {default_roster()}")
    for p in (r, m, lt, rt):
        p.add_argument("--tokens", type=Path, default=None,
                       help=f"the agent token store (default ${TOKENS_ENV}, else {default_dir() / 'agent_tokens.json'})")
    for p in (m, rt):
        p.add_argument("--state-dir", type=Path, default=None,
                       help=f"the portal state dir whose audit log gets the line (default {default_state_dir()})")
    args = ap.parse_args(argv)
    if args.cmd in ("mint-token", "list-tokens", "revoke-token"):
        return token_cli(args)
    path = args.roster or default_roster()
    if args.cmd == "set-email":
        try:
            set_email(path, args.id, args.email)
        except ValueError as err:
            print(f"[--] {err}")
            return 1
        print(f"[OK] {args.id}: email {'set' if args.email.strip() else 'removed'}")
        return 0
    if args.cmd == "add-key":
        try:
            key = add_key(path, args.id, args.role, args.email)
        except ValueError as e:
            ap.error(str(e))
        base = (args.base or default_base()).rstrip("/")
        print(f"[OK] {args.id} ({args.role}) added to {path}. Key link (shown once, keep it private):")
        print(f"{base}/login?k={key}")
        return 0
    if args.cmd == "revoke":
        # the person's agent tokens too: the store named (or $CJ_AGENT_TOKENS), else the default store only with the
        # default roster (a --roster elsewhere, e.g. a test's, never touches the real store)
        tokens = args.tokens or (Path(os.environ[TOKENS_ENV]) if os.environ.get(TOKENS_ENV) else
                                 default_tokens() if args.roster is None else None)
        ok = revoke(path, args.id, tokens=tokens)
        print(f"[OK] {args.id} revoked" if ok else f"[--] {args.id} not found or already revoked")
        return 0 if ok else 1
    for p in _load(path)["people"]:
        print(f"{p.get('id')}\t{p.get('role')}\t{'revoked ' + p['revoked'] if p.get('revoked') else 'active'}"
              f"\tadded {p.get('added', '?')}\t{p.get('email') or '-'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
