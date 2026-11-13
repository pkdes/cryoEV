"""Shared helpers for the test suites (round 10). Add `<repo>/tests` to sys.path to import this.

free_ports(n)   n distinct free ports.
free_port()     a port the OS just handed out. Still racy if you close it and rebind later: prefer binding port 0 in
                the server itself, or use bind_retry().
bind_retry(f)   call f(port) with a fresh free port until it does not raise OSError (the free-port-then-bind race).
clean_env()     a copy of os.environ with the actor variables (CRYOEV_VIA / CRYOEV_FOR) removed; strip_all=True drops
                every CRYOEV_* (data roots, registry), drop=(...) more names, keywords are set last.
strip_environ() the same removal applied to os.environ itself (module level, for the rest of the process).
env_patch()     mock.patch.dict(os.environ, clean_env(...), clear=True), restored afterwards (with / enterContext).
no_actor_env()  context manager that removes the actor variables from os.environ for its body (the gate strips them
                for children; a test that checks the "no actor" default must not depend on the caller's shell).
write_registry(root, records, samples=..., skip=())
                a fixture registry folder: records as {entity: [records]} or a list (entity from `schema`);
                samples=True starts from the repo's registry/samples/*.jsonl (required keyword).
process_lock()  a cross-process lock on a name (a lock directory in the temp dir; stale-safe), for machine-wide
                resources such as `subst` drive letters.
wait_until()    poll a condition instead of sleeping a fixed time.
browser         decorator for a test class or method that drives a real browser: skipped when the gate sets
                CJ_GATE_SKIP_BROWSER=1 (--affected with only non-UI files changed; tools/test_gate.py).
"""
from __future__ import annotations

import contextlib
import json
import os
import shutil
import socket
import tempfile
import time
from pathlib import Path

ACTOR_VARS = ("CRYOEV_VIA", "CRYOEV_FOR")
SKIP_BROWSER_ENV = "CJ_GATE_SKIP_BROWSER"      # = tools/test_gate.py SKIP_BROWSER_ENV
SAMPLES = Path(__file__).resolve().parent.parent / "registry" / "samples"


def browser(obj):
    """@browser on a TestCase class (or one test method) that drives a real browser (Playwright). The test gate sets
    CJ_GATE_SKIP_BROWSER=1 in --affected runs whose changed files are all non-UI; such classes are then skipped, while
    the file's other (unit) classes still run. Decorate the class when its setUpClass / setUp starts the browser."""
    import unittest
    return unittest.skipIf(os.environ.get(SKIP_BROWSER_ENV) == "1", "browser test: no UI file changed (gate)")(obj)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def free_ports(n: int) -> list[int]:
    """n DISTINCT free ports (all n sockets are held open together, so the OS cannot hand one out twice)."""
    socks = [socket.socket() for _ in range(n)]
    try:
        for s in socks:
            s.bind(("127.0.0.1", 0))
        return [s.getsockname()[1] for s in socks]
    finally:
        for s in socks:
            s.close()


def bind_retry(start, attempts: int = 8):
    """start(port) -> the started thing; retried with a new port if the bind lost a race (OSError)."""
    last = None
    for _ in range(attempts):
        try:
            return start(free_port())
        except OSError as e:
            last = e
            time.sleep(0.05)
    raise last


def wait_until(cond, timeout: float = 15.0, step: float = 0.05) -> bool:
    """True as soon as cond() is truthy; False after `timeout` (a generous ceiling, not a delay: it returns early)."""
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(step)
    return bool(cond())


def _drop(key: str, strip_all: bool, drop) -> bool:
    """drop: names, or prefixes written with a trailing * ("CJ_*")."""
    k = key.upper()
    return k in ACTOR_VARS or (strip_all and k.startswith("CRYOEV_")) or any(
        k.startswith(d.upper()[:-1]) if d.endswith("*") else k == d.upper() for d in drop)


def clean_env(base=None, strip_all: bool = False, drop=(), **extra) -> dict:
    """A copy of os.environ (or `base`) without the actor variables (CRYOEV_VIA / CRYOEV_FOR); strip_all=True drops
    every CRYOEV_* (the data roots, the registry, ...); `drop` names more variables (or "CJ_*" prefixes) to remove;
    `extra` is set last."""
    env = {k: v for k, v in dict(os.environ if base is None else base).items() if not _drop(k, strip_all, drop)}
    env.update(extra)
    return env


def strip_environ(strip_all: bool = False, drop=()) -> dict:
    """Remove the same variables from os.environ itself, for the rest of the process (module level in a helper that
    every test of a suite imports). -> what was removed."""
    return {k: os.environ.pop(k) for k in [k for k in os.environ if _drop(k, strip_all, drop)]}


def env_patch(strip_all: bool = False, drop=(), **extra):
    """mock.patch.dict(os.environ, clean_env(...), clear=True): a with-block, start()/stop(), or
    TestCase.enterContext / enterClassContext. os.environ is restored afterwards."""
    from unittest import mock
    return mock.patch.dict(os.environ, clean_env(None, strip_all, drop, **extra), clear=True)


@contextlib.contextmanager
def no_actor_env():
    saved = {k: os.environ.pop(k) for k in ACTOR_VARS if k in os.environ}
    try:
        yield
    finally:
        os.environ.update(saved)


def _alive(pid: int) -> bool:
    if os.name == "nt":                      # os.kill(pid, 0) would terminate the process on Windows
        import ctypes
        k = ctypes.windll.kernel32
        h = k.OpenProcess(0x1000, False, pid)
        if not h:
            return False
        code = ctypes.c_ulong()
        ok = k.GetExitCodeProcess(h, ctypes.byref(code))
        k.CloseHandle(h)
        return bool(ok) and code.value == 259
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


@contextlib.contextmanager
def process_lock(name: str, timeout: float = 120.0, stale_s: float = 300.0):
    """Cross-process mutex: atomic mkdir of <tmp>/cryoev_lock_<name>; a lock whose owner pid is gone, or older than
    stale_s, is taken over."""
    d = Path(tempfile.gettempdir()) / f"cryoev_lock_{name}"
    end = time.time() + timeout
    while True:
        try:
            d.mkdir()
            (d / "pid").write_text(str(os.getpid()), encoding="utf-8")
            break
        except FileExistsError:
            try:
                pid = int((d / "pid").read_text(encoding="utf-8"))
                stale = not _alive(pid) or time.time() - d.stat().st_mtime > stale_s
            except (OSError, ValueError):
                try:
                    stale = time.time() - d.stat().st_mtime > 5     # owner has not written its pid: give it 5 s
                except OSError:
                    stale = False
            if stale:
                for f in d.glob("*"):
                    with contextlib.suppress(OSError):
                        f.unlink()
                with contextlib.suppress(OSError):
                    d.rmdir()
                continue
            if time.time() > end:
                raise TimeoutError(f"process_lock({name}) held by another process")
            time.sleep(0.2)
    try:
        yield
    finally:
        for f in d.glob("*"):
            with contextlib.suppress(OSError):
                f.unlink()
        with contextlib.suppress(OSError):
            d.rmdir()


def write_registry(root, records=None, *, samples: bool, skip=()) -> Path:
    """A fixture registry folder at `root` (created if missing) -> root. `samples` is required at every call site.

    records: {entity: [record, ...]}, or a list of records whose entity is taken from `schema` ("model/1" ->
    model.jsonl); order is kept. samples=True: every registry/samples/<entity>.jsonl (minus the names in `skip`, e.g.
    "label_change.jsonl", whose rows have no id) with that entity's records appended; an entity with no sample file
    gets a file of its own. samples=False: each entity in `records` gets root/<entity>.jsonl rewritten whole.
    Lines are `json.dumps(record, ensure_ascii=False)`, UTF-8, LF line ends (newline="\n" on every OS)."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if isinstance(records, dict):
        extra = {e: list(r) for e, r in records.items()}
    else:
        extra = {}
        for rec in records or []:
            extra.setdefault(str(rec["schema"]).split("/", 1)[0], []).append(rec)

    def write(path, lines):
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write("".join(x + "\n" for x in lines))

    done = set()
    for f in sorted(SAMPLES.glob("*.jsonl")) if samples else []:
        if f.name in skip:
            continue
        lines = f.read_text(encoding="utf-8").splitlines()
        write(root / f.name, lines + [json.dumps(r, ensure_ascii=False) for r in extra.get(f.stem, [])])
        done.add(f.stem)
    for ent, recs in extra.items():
        if ent not in done:
            write(root / f"{ent}.jsonl", [json.dumps(r, ensure_ascii=False) for r in recs])
    return root
    return root
