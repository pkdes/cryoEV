"""
CryoEV portal sign-in, Cloudflare Access mode (C4): `python -m portal.app --auth access --port 8448`.

Lab members reach the portal from the internet through a Cloudflare Tunnel guarded by Cloudflare Access (one-time PIN
email login). Access adds a signed JWT to every request in the `Cf-Access-Jwt-Assertion` header. This mode trusts
ONLY that verified JWT; key links and session cookies are off on its port (portal/app.py).

- Header `Cf-Access-Jwt-Assertion` only (no cookie fallback). PyJWT, RS256 only; `aud` = the configured AUD, `iss` =
  https://<team>.cloudflareaccess.com, `exp` / `nbf` with LEEWAY_S.
- Keys: https://<team>.cloudflareaccess.com/cdn-cgi/access/certs, fetched in a worker thread with a timeout, cached,
  refetched on an unknown `kid` (or after KEYS_TTL_S) at most once per REFETCH_S. No usable keys = everyone refused.
- Config: ~/.cryoev/cloudflare_access.json {"team", "aud", "host"} (or --access-config <path>); env CJ_ACCESS_TEAM,
  CJ_ACCESS_AUD, CJ_ACCESS_HOST override it. Missing = access mode refuses to start.
PyJWT is imported here only, so key mode runs without it. Tokens are never logged.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import jwt                      # PyJWT (requirements-portal.txt); access mode only
from jwt.algorithms import RSAAlgorithm

HEADER = b"cf-access-jwt-assertion"
LOGOUT = "/cdn-cgi/access/logout"
ALG = "RS256"
LEEWAY_S = 60
REFETCH_S = 60.0                # at most one key fetch per minute (unknown kid, TTL refresh or failure)
KEYS_TTL_S = 3600.0             # refresh the cached keys after an hour (rotation removes old ones)
KEYS_MAX_AGE_S = 6 * 3600.0     # no successful fetch for this long: the cached keys are dropped (everyone refused)
FETCH_TIMEOUT_S = 5.0
MAX_CERTS = 256 * 1024          # bytes read from the certs URL
MAX_TOKEN = 16 * 1024
TEAM_RE = re.compile(r"[a-z0-9][a-z0-9-]{0,62}")
AUD_RE = re.compile(r"[A-Za-z0-9_-]{8,128}")
HOST_RE = re.compile(r"(?=.{1,253}\Z)[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?(\.[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?)+")
IP_RE = re.compile(r"[0-9A-Fa-f:.]{1,45}")
log = logging.getLogger("cj.access_mode")


class ConfigError(ValueError):
    pass


def default_config_path() -> Path:
    return Path.home() / ".cryoev" / "cloudflare_access.json"


@dataclass(frozen=True)
class AccessConfig:
    team: str           # <team>.cloudflareaccess.com
    aud: str            # the Access application's AUD tag
    host: str           # the public host name, e.g. portal.example.org

    @property
    def issuer(self) -> str:
        return f"https://{self.team}.cloudflareaccess.com"

    @property
    def certs_url(self) -> str:
        return f"{self.issuer}/cdn-cgi/access/certs"

    @property
    def origin(self) -> str:
        return f"https://{self.host}"


def load_config(path: Path | None = None, env=None) -> AccessConfig:
    """The file (default ~/.cryoev/cloudflare_access.json), then env CJ_ACCESS_TEAM / _AUD / _HOST on top.
    Raises ConfigError (with what is missing) when a value is missing or malformed."""
    env = os.environ if env is None else env
    path = Path(path) if path else default_config_path()
    doc, problem = {}, None
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(doc, dict):
            doc, problem = {}, f"{path} is not a JSON object"
    except FileNotFoundError:
        problem = f"{path} not found"
    except (OSError, ValueError) as e:
        problem = f"{path} not readable ({type(e).__name__})"
    vals = {}
    for k in ("team", "aud", "host"):
        v = env.get(f"CJ_ACCESS_{k.upper()}") or doc.get(k)
        vals[k] = v.strip() if isinstance(v, str) else ""
    vals["team"] = vals["team"].lower()
    vals["host"] = vals["host"].lower()
    missing = [k for k, v in vals.items() if not v]
    if missing:
        raise ConfigError(f"Cloudflare access config incomplete: missing {', '.join(missing)} "
                          f"({problem or path}; env CJ_ACCESS_TEAM / CJ_ACCESS_AUD / CJ_ACCESS_HOST override it)")
    if not TEAM_RE.fullmatch(vals["team"]):
        raise ConfigError("Cloudflare access config: `team` must be the bare team name (<team>.cloudflareaccess.com)")
    if not AUD_RE.fullmatch(vals["aud"]):
        raise ConfigError("Cloudflare access config: `aud` is not an Access AUD tag")
    if not HOST_RE.fullmatch(vals["host"]):
        raise ConfigError("Cloudflare access config: `host` must be a bare host name (no scheme, port or path)")
    return AccessConfig(**vals)


def fetch_certs(url: str, timeout: float = FETCH_TIMEOUT_S) -> dict:
    """GET the certs JSON (blocking: run it in a worker thread). Raises on any failure."""
    req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "cryoev-portal"})
    with urllib.request.urlopen(req, timeout=timeout) as r:    # noqa: S310 (https URL built from the config)
        body = r.read(MAX_CERTS + 1)
    if len(body) > MAX_CERTS:
        raise ValueError("certs response too large")
    return json.loads(body.decode("utf-8"))


def parse_keys(doc) -> dict:
    """{kid: RSA public key} from a JWKS document; only RSA signing keys with a kid (others skipped). Raises
    ValueError when there is none."""
    keys = {}
    for k in (doc.get("keys") if isinstance(doc, dict) else None) or []:
        if not isinstance(k, dict) or k.get("kty") != "RSA" or not isinstance(k.get("kid"), str) or not k["kid"]:
            continue
        if k.get("use", "sig") != "sig" or k.get("alg", ALG) != ALG:
            continue
        try:
            keys[k["kid"]] = RSAAlgorithm.from_jwk(k)
        except (jwt.InvalidKeyError, ValueError, TypeError, KeyError):
            continue
    if not keys:
        raise ValueError("no usable RS256 keys in the certs document")
    return keys


class KeySet:
    """The team's signing keys, cached. `get(kid)` refetches (in a worker thread, with a timeout) when the kid is
    unknown or the cache is older than KEYS_TTL_S, at most once per REFETCH_S; a failed fetch keeps the old keys until
    KEYS_MAX_AGE_S after the last good one. No keys = None for every kid (the caller refuses)."""

    def __init__(self, url: str, fetch=None, timeout: float = FETCH_TIMEOUT_S, clock=time.monotonic):
        self.url = url
        self.fetch = fetch
        self.timeout = timeout
        self.clock = clock
        self.keys: dict = {}
        self.good_at: float | None = None       # last successful fetch
        self.tried_at: float | None = None      # last fetch attempt (good or not)
        self.fetches = 0
        self.problem: str | None = None
        self._lock = asyncio.Lock()

    def _expired(self, now) -> bool:
        return self.good_at is None or now - self.good_at > KEYS_MAX_AGE_S

    async def _refresh(self):
        self.tried_at = self.clock()
        self.fetches += 1
        try:
            fetch = self.fetch or fetch_certs          # looked up per call (a test may replace it)
            doc = await asyncio.wait_for(asyncio.to_thread(fetch, self.url, self.timeout), self.timeout + 1)
            keys = parse_keys(doc)
        except Exception as e:                   # any failure: keep what we had (until KEYS_MAX_AGE_S)
            self.problem = f"key fetch failed ({type(e).__name__})"
            log.warning("access mode: %s from %s", self.problem, self.url)
            return
        self.keys, self.good_at, self.problem = keys, self.clock(), None

    async def get(self, kid):
        if not isinstance(kid, str) or not kid:
            return None
        now = self.clock()
        stale = self.good_at is None or now - self.good_at > KEYS_TTL_S
        if kid not in self.keys or stale:
            async with self._lock:
                now = self.clock()
                stale = self.good_at is None or now - self.good_at > KEYS_TTL_S
                if (kid not in self.keys or stale) and (self.tried_at is None or now - self.tried_at >= REFETCH_S):
                    await self._refresh()
        if self._expired(self.clock()):
            self.keys = {}
            return None
        return self.keys.get(kid)


class Verifier:
    """`await verify(token)` -> (claims, None) or (None, reason). Never raises; the reason is for the log only."""

    def __init__(self, cfg: AccessConfig, keyset: KeySet | None = None):
        self.cfg = cfg
        self.keys = keyset or KeySet(cfg.certs_url)

    async def verify(self, token):
        if not token:
            return None, "no Cf-Access-Jwt-Assertion header"
        if len(token) > MAX_TOKEN:
            return None, "token too large"
        try:
            head = jwt.get_unverified_header(token)
        except jwt.PyJWTError:
            return None, "malformed token"
        if head.get("alg") != ALG:
            return None, f"alg {str(head.get('alg'))[:16]!r} refused (RS256 only)"
        key = await self.keys.get(head.get("kid"))
        if key is None:
            return None, ("no signing keys (" + (self.keys.problem or "not fetched") + ")" if not self.keys.keys
                          else "unknown kid")
        try:
            claims = jwt.decode(token, key, algorithms=[ALG], audience=self.cfg.aud, issuer=self.cfg.issuer,
                                leeway=LEEWAY_S, options={"require": ["exp", "aud", "iss"], "verify_signature": True})
        except jwt.ExpiredSignatureError:
            return None, "expired"
        except jwt.ImmatureSignatureError:
            return None, "not yet valid (nbf / iat)"
        except jwt.InvalidAudienceError:
            return None, "wrong aud"
        except jwt.InvalidIssuerError:
            return None, "wrong iss"
        except jwt.InvalidSignatureError:
            return None, "bad signature"
        except jwt.PyJWTError as e:
            return None, f"invalid token ({type(e).__name__})"
        email = claims.get("email")
        if not isinstance(email, str) or not email.strip():
            return None, "no email claim"
        return claims, None


def client_ip(value) -> str:
    """Cf-Connecting-IP for the log: an IP-looking string, else '-' (never raw header bytes in the log)."""
    v = (value or "").strip()
    return v if IP_RE.fullmatch(v) else "-"


def admin_allowed(env=None) -> bool:
    """CJ_ACCESS_ADMIN: only the word "on" lets admins in through Cloudflare (default off)."""
    env = os.environ if env is None else env
    return str(env.get("CJ_ACCESS_ADMIN", "off")).strip().lower() == "on"
