"""Shared helpers for the test suites (round 10). Add `<repo>/tests` to sys.path to import this.

free_port()     a port the OS just handed out. Still racy if you close it and rebind later: prefer binding port 0 in
                the server itself, or use bind_retry().
bind_retry(f)   call f(port) with a fresh free port until it does not raise OSError (the free-port-then-bind race).
clean_env()     a copy of os.environ with the actor variables (CRYOEV_VIA / CRYOEV_FOR) removed.
no_actor_env()  context manager that removes them from os.environ for its body (the gate strips them for children;
                a test that checks the "no actor" default must not depend on the caller's shell).
process_lock()  a cross-process lock on a name (a lock directory in the temp dir; stale-safe), for machine-wide
                resources such as `subst` drive letters.
wait_until()    poll a condition instead of sleeping a fixed time.
"""
from __future__ import annotations

import contextlib
import os
import socket
import tempfile
import time
from pathlib import Path

ACTOR_VARS = ("CRYOEV_VIA", "CRYOEV_FOR")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


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


def clean_env(base=None) -> dict:
    env = dict(os.environ if base is None else base)
    for k in ACTOR_VARS:
        env.pop(k, None)
    return env


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
