"""
Cloudflare access config reader (no jwt import, so key-mode code and Admin can use it).

~/.cryoev/cloudflare_access.json {"team", "aud", "host"}, with env CJ_ACCESS_TEAM / _AUD / _HOST on top.
`access.load_config` validates the values; Admin only needs `public_host()` to tell a new person where to sign in.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

KEYS = ("team", "aud", "host")
_HOST_RE = re.compile(r"(?=.{1,253}\Z)[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?(\.[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?)+")


def default_config_path() -> Path:
    return Path.home() / ".cryoev" / "cloudflare_access.json"


def read_file(path: Path | None = None) -> tuple:
    """-> (dict, problem or None). Never raises."""
    path = Path(path) if path else default_config_path()
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(doc, dict):
            return {}, f"{path} is not a JSON object"
        return doc, None
    except FileNotFoundError:
        return {}, f"{path} not found"
    except (OSError, ValueError) as e:
        return {}, f"{path} not readable ({type(e).__name__})"


def read_config(path: Path | None = None, env=None) -> tuple:
    """-> (values, problem): team / aud / host as stripped strings ('' when missing), the env override first.
    `problem` describes the file when it could not be read (None when it was)."""
    env = os.environ if env is None else env
    doc, problem = read_file(path)
    vals = {}
    for k in KEYS:
        v = env.get(f"CJ_ACCESS_{k.upper()}") or doc.get(k)
        vals[k] = v.strip() if isinstance(v, str) else ""
    return vals, problem


def public_host(path: Path | None = None, env=None) -> str | None:
    """The public host name (lower-case, bare) people sign in at, or None when none is configured or malformed."""
    host = read_config(path, env)[0]["host"].lower()
    return host if host and _HOST_RE.fullmatch(host) else None
