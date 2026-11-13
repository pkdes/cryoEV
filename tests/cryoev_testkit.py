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
write_registry(root, {entity: [records]}, samples=False)
                a fixture registry folder: <entity>.jsonl, one JSON line per record (samples=True starts from the
                repo's registry/samples/*.jsonl).
process_lock()  a cross-process lock on a name (a lock directory in the temp dir; stale-safe), for machine-wide
                resources such as `subst` drive letters.
wait_until()    poll a condition instead of sleeping a fixed time.
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
SAMPLES = Path(__file__).resolve().parent.parent / "registry" / "samples"
NOT_REGISTRY = {"label_change.jsonl"}          # samples/ rows with no id: not a registry entity file


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


def write_registry(root, records=None, samples: bool = False) -> Path:
    """A fixture registry folder (created if missing) -> root.

    records: {entity: [record, ...]}; each becomes root/<entity>.jsonl, one `json.dumps(record)` line per record
    (written in text mode, as the copies this replaces did). Without samples each entity file is rewritten whole.
    samples=True first copies the repo's registry/samples/*.jsonl byte for byte (label_change.jsonl left out), and
    `records` are then appended after the sample lines of the same entity."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if samples:
        for f in sorted(SAMPLES.glob("*.jsonl")):
            if f.name not in NOT_REGISTRY:
                shutil.copyfile(f, root / f.name)
    for entity, recs in (records or {}).items():
        mode = "a" if samples and (root / f"{entity}.jsonl").exists() else "w"
        with open(root / f"{entity}.jsonl", mode, encoding="utf-8") as fh:
            fh.write("".join(json.dumps(r) + "\n" for r in recs))
    return root
