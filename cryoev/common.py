"""Shared pieces of the cryoev CLI: module loading, data roots, who/via, the dry-run / --do runner and the action log.

Action log: <CryoAI-laptop>/logs/agent_actions.jsonl (= portal/audit.py CLI_FILE; the Admin page reads it through
$CJ_AUDIT_EXTRA as the door "laptop CLI (unverified)"). Each --do appends two lines in the portal's audit shape,
under an OS lock (data_utils/oslock.py, <file>.lock):
    {"t": UTC ISO, "actor": <who>, "action": ..., "target": ..., "detail": {"via", "phase": "requested", "args"}}
    {... "detail": {"via", "phase": "result", "args", "result": {"rc", ...}}}
The "requested" line is written first; if it can't be written nothing is done. `args` are the command's own safe
fields (never secrets, never file contents). The identity comes from --who / $CRYOEV_FOR and $CRYOEV_VIA, declared by
the caller, which is why the Admin page marks this door unverified.
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from data_utils import config  # noqa: E402
from data_utils import path_guard  # noqa: E402
from data_utils.oslock import acquire_lock  # noqa: E402

ACTION_LOG = "agent_actions.jsonl"          # = portal/audit.py CLI_FILE
LOG_WAIT_S = 5.0
GUIDE = "docs/AGENTS.md"
# exit codes (the same in __main__'s help and docs/AGENTS.md): 0 ok; 1 not found (unknown id / package, no match) or
# done but its result line could not be logged; 2 refused or failed (bad arguments, a guard, an error); 3 refused by
# rule (never automated, the blind-audit rule)
EXIT_REFUSED = 3


class CliError(Exception):
    """A refusal or a problem the user must fix: printed, exit 2."""


class Refused(CliError):
    """A refusal by rule (never automated, blind audit): exit 3."""


# ---------------------------------------------------------------- the repo's modules, loaded once
def load(name: str, rel: str):
    """A repo file loaded by its path under a fixed module name (jobs/ and tools/ are not packages; registry is:
    `from registry import ...`)."""
    mod = sys.modules.get(name)
    if mod is None:
        spec = importlib.util.spec_from_file_location(name, REPO / rel)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        try:
            spec.loader.exec_module(mod)
        except BaseException:
            sys.modules.pop(name, None)
            raise
    return mod


def registry():
    """registry.store (the one registry module object; jobs/predict_set.py and the portal use the same)."""
    from registry import store
    return store


def actor():
    """data_utils/actor.py (the attribution helpers; = portal/audit.actor_module())."""
    from data_utils import actor as _actor
    return _actor


def services():
    return load("cj_services", "tools/services.py")


# ---------------------------------------------------------------- roots
def registry_root() -> Path:
    try:
        return Path(config.registry_root())
    except config.ConfigError as e:
        raise CliError(str(e))


def laptop_root() -> Path:
    try:
        return config.laptop_root()
    except config.ConfigError as e:
        raise CliError(str(e))


def action_log_path() -> Path:
    return laptop_root() / "logs" / ACTION_LOG


def services_config_path() -> Path:
    return Path(os.environ.get("CRYOEV_SERVICES_CONFIG") or services().DEFAULT_CONFIG)


# ---------------------------------------------------------------- who / via
def identity(flag=None) -> tuple:
    """(who, via): who = --who, else $CRYOEV_FOR (a people.json id), else None; via = $CRYOEV_VIA or None."""
    A = actor()
    try:
        if flag and flag not in A._people():
            raise CliError(f"--who {flag!r} is not a people.json id")
        return A.who(flag, None), A.via()
    except A.ActorError as e:
        raise CliError(str(e))


def label(who, via) -> str:
    return actor().label(who, via) or "nobody (pass --who or set CRYOEV_FOR)"


# ---------------------------------------------------------------- paths
def under(path, root) -> bool:
    """path is root or inside it, compared as typed AND by real path (links, junctions and 8.3 names resolved on
    the part that exists)."""
    def inside(p, r):
        return p == r or p.startswith(r.rstrip("\\/") + os.sep)
    ps = {path_guard.normalize(path), path_guard.normalize(os.path.realpath(path))}
    rs = {path_guard.normalize(root), path_guard.normalize(os.path.realpath(root))}
    return any(inside(p, r) for p in ps for r in rs)


def box_problem(path) -> str | None:
    """Agents never touch Box (read or write): the guard's Box root and the current home's."""
    roots = [r for r in path_guard.PROTECTED_ROOTS if Path(r).name.lower() == "box"] + [Path.home() / "Box"]
    return path_guard.protected_problem(path, tuple(dict.fromkeys(roots)))


def out_problem(out) -> str | None:
    """Why --out may not be a pull destination: a protected root (path_guard: Box, Cryo-image-processing), or inside
    a data root or the repo (pulled copies would mix with the originals or get committed)."""
    problem = path_guard.protected_problem(out)
    if problem:
        return problem
    roots = [("the CryoAI-laptop data folder", config.laptop_root(required=False)),
             ("the registry folder", config.registry_root(required=False)),
             ("the CryoAI data folder", config.cryoai_root(required=False)), ("the repo", REPO)]
    for name, root in roots:
        if root and under(out, root):
            return f"{out} is inside {name}: pull to a folder outside the data folders and the repo"
    return None


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------- output and wrapped mains
def emit_json(obj) -> None:
    print(json.dumps(obj, indent=1, ensure_ascii=False, default=str))


class _Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
        return len(s)

    def flush(self):
        for st in self.streams:
            st.flush()


def call_main(fn, argv, capture=False) -> tuple:
    """Run an existing main(argv); SystemExit is caught. -> (exit code, what it printed when capture else "")."""
    buf = io.StringIO()
    ctx = contextlib.redirect_stdout(_Tee(sys.stdout, buf)) if capture else contextlib.nullcontext()
    try:
        with ctx:
            rc = fn(list(argv))
    except SystemExit as e:
        if e.code is None or isinstance(e.code, int):
            rc = e.code
        else:
            print(e.code, file=sys.stderr)
            rc = 2
    return (0 if rc is None else int(rc)), buf.getvalue()


# ---------------------------------------------------------------- the action log and the --do runner
def log_action(who, action, target, detail) -> dict:
    """Append one audit-shaped line to the action log under its OS lock. Raises OSError when it can't."""
    rec = {"t": utc_now(), "actor": who, "action": action, "target": target, "detail": detail}
    path = action_log_path()
    problem = path_guard.protected_problem(path)
    if problem:
        raise OSError(problem)
    path.parent.mkdir(parents=True, exist_ok=True)
    lk = acquire_lock(path.with_name(path.name + ".lock"), LOG_WAIT_S)
    if lk is None:
        raise OSError(f"{path} is locked by another process")
    try:
        lead = ""
        if path.exists() and path.stat().st_size:
            with open(path, "rb") as f:
                f.seek(-1, os.SEEK_END)
                lead = "" if f.read(1) == b"\n" else "\n"
        with open(path, "a", encoding="utf-8", newline="\n") as f:
            f.write(lead + json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
    finally:
        lk.close()
    return rec


def write_flags(p) -> None:
    p.add_argument("--do", action="store_true", help="really do it (default: a dry run that writes nothing)")
    p.add_argument("--who", help="people.json id responsible (default $CRYOEV_FOR; required with --do)")


def run_write(a, action, target, args, plan, act) -> int:
    """The one write path. Without --do: plan() prints exactly what would happen and returns 0 (would go ahead) or
    non-zero (refused); nothing is written. With --do: the "requested" line, act(who, via) -> (rc, result dict),
    then the result line."""
    who, via = identity(getattr(a, "who", None))
    if not a.do:
        rc = plan()
        print(f"\nrecorded as: {label(who, via)}")
        print("DRY RUN: nothing was written. Run the same command with --do to do it." if rc == 0
              else "DRY RUN: refused (see above); nothing was written.")
        return rc
    if not who:
        raise CliError("--do needs a person: pass --who <people.json id> or set CRYOEV_FOR")
    try:
        log_action(who, action, target, {"via": via, "phase": "requested", "args": args})
    except OSError as e:
        raise CliError(f"not done: the action log can't be written ({e})")
    try:
        rc, result = act(who, via)
    except CliError as e:
        print(e, file=sys.stderr)
        rc, result = 2, {"error": str(e)[:300]}
    except Exception as e:                                  # logged as the result, then reported
        print(f"failed: {type(e).__name__}: {e}", file=sys.stderr)
        rc, result = 2, {"error": f"{type(e).__name__}: {e}"[:300]}
    try:
        log_action(who, action, target, {"via": via, "phase": "result", "args": args,
                                         "result": {"rc": rc, **(result or {})}})
    except OSError as e:
        print(f"warning: done, but the result line was not logged ({e})", file=sys.stderr)
        rc = rc or 1
    print(f"logged to {action_log_path()} as {label(who, via)}")
    return rc
