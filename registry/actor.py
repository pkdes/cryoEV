"""
Who did it, and through which agent: the "claude for pujan" attribution (round 9).

- `who`: the responsible person, a people.json id (unchanged meaning everywhere it already exists).
- `via`: the agent that acted for them (e.g. "claude"), or absent for a person acting directly.

Where the values come from:
- an explicit flag always wins (`--who randy-carney` stays randy-carney);
- else the env: CRYOEV_FOR (a people.json id) and CRYOEV_VIA (^[a-z][a-z0-9_-]{0,31}$), which Claude Code sets for
  its own sessions (user settings), so scripts an agent runs are stamped and the user's own shell runs are not;
- else the caller's old default.

Only CLI / script entry points read the env. Long-running servers (portal, review servers) and services.py children
must not carry these variables: call `strip_env()` in their main(), so a service restarted from an agent session
never credits lab members' actions to the agent. Never call it at import time: the CLI imports portal modules.

`label()` is the one display form: "claude for pujan", or just "pujan".
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

ENV_VIA, ENV_FOR = "CRYOEV_VIA", "CRYOEV_FOR"
VIA_RE = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")
PEOPLE = Path(__file__).resolve().parent.parent / "annotation" / "people.json"


class ActorError(ValueError):
    pass


def _people(path=PEOPLE) -> set:
    try:
        return {p["id"] for p in json.loads(Path(path).read_text("utf-8"))["people"]}
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise ActorError(f"can't read people.json ({path}): {e}") from e


def via(environ=None):
    """The agent from $CRYOEV_VIA, or None when unset/empty. A malformed value is an error, never silently dropped."""
    v = (os.environ if environ is None else environ).get(ENV_VIA, "").strip()
    if not v:
        return None
    if not VIA_RE.match(v):
        raise ActorError(f"{ENV_VIA}={v!r} is not a short lower-case id (^[a-z][a-z0-9_-]{{0,31}}$)")
    return v


def who(flag=None, default=None, environ=None, people_path=PEOPLE):
    """The responsible person: the explicit flag, else $CRYOEV_FOR (must be a people.json id), else `default`."""
    if flag:
        return flag
    f = (os.environ if environ is None else environ).get(ENV_FOR, "").strip()
    if f:
        if f not in _people(people_path):
            raise ActorError(f"{ENV_FOR}={f!r} is not a people.json id")
        return f
    return default


def current(flag=None, default=None, environ=None, people_path=PEOPLE) -> dict:
    """{"who": ..., "via": ...} for a script entry point; "via" is omitted when no agent is acting."""
    out = {"who": who(flag, default, environ, people_path)}
    v = via(environ)
    if v:
        out["via"] = v
    return out


def label(who_=None, via_=None) -> str:
    """Display form: "claude for pujan", "pujan", "claude" (agent, no person), or "" (nothing known).
    Callers escape it and apply their own privacy mask BEFORE calling (a hidden owner stays hidden)."""
    w, v = (str(who_).strip() if who_ else ""), (str(via_).strip() if via_ else "")
    if v and w:
        return f"{v} for {w}"
    return v or w


def label_of(rec: dict, who_key="who") -> str:
    """label() from a record's own fields: the person from who_key, else status_by, else created_by; the agent that
    goes with that person: status_by -> status_via (who set the status, which may differ from the line's writer
    `via`), created_by -> created_via, who_key -> via, else created_via."""
    if not isinstance(rec, dict):
        return ""
    pair = {"status_by": ("status_via",), "created_by": ("created_via",)}
    for key in (who_key, "status_by", "created_by"):
        if rec.get(key):
            via_keys = pair.get(key, ("via", "created_via"))
            return label(rec[key], next((rec[v] for v in via_keys if rec.get(v)), None))
    return label(None, rec.get("via") or rec.get("created_via"))


def is_agent(rec: dict) -> bool:
    return isinstance(rec, dict) and bool(rec.get("via") or rec.get("created_via") or rec.get("status_via"))


def strip_env(environ=None) -> list:
    """Remove CRYOEV_VIA / CRYOEV_FOR from `environ` (default os.environ); -> the names removed. For server main()s and
    for child environments built by services.py."""
    env = os.environ if environ is None else environ
    gone = [k for k in (ENV_VIA, ENV_FOR) if k in env]
    for k in gone:
        env.pop(k, None)
    return gone
