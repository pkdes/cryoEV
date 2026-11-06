"""Keep the laptop's long-running servers (review, sandbox, history, job monitor, portal) up without a shell.

A small service manager, standard library only. The services are listed in a JSON config kept OUTSIDE the
repo, because it holds real paths: --config, else env CRYOEV_SERVICES_CONFIG, else
%USERPROFILE%/.cryoev/services.json. `example-config` prints a template.

  python tools/services.py status [--json]          # name, enabled, state, pid, port, listening, health, uptime, log
  python tools/services.py start [name ...|all]     # default all; enabled entries only
  python tools/services.py stop [name ...|all]      # only processes this tool started
  python tools/services.py restart [name ...|all]
  python tools/services.py example-config [--out FILE]

Start: skipped if its pid file names a live process of ours, or if something else already listens on its
port ("port busy (not ours)": left alone). Otherwise the process is started detached with no console window,
stdout+stderr appended to its log (moved to <log>.1 when over 10 MB), and a pid file written to
<run_dir>/<name>.pid (default %USERPROFILE%/.cryoev/run). Start waits up to --wait seconds for the port.

Stop: kills a process only when the pid file's pid is alive AND its command line and creation time match
the ones recorded at start. Anything else (a hand-started server on the port, a reused pid, the pid of an
unrelated program) is never touched. Children of our process (e.g. the real python behind a venv launcher)
are stopped with it; a child is only counted if it was created after our process.

Exit codes: 0 ok; 1 a service failed to start/stop or its port is busy (not ours); 2 bad config or usage.
"""
import argparse
import http.client
import json
import os
import re
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from data_utils.oslock import acquire_lock  # noqa: E402

HOME_DIR = Path.home() / ".cryoev"
DEFAULT_CONFIG = HOME_DIR / "services.json"
DEFAULT_RUN_DIR = HOME_DIR / "run"
LOG_MAX_BYTES = 10 * 1024 * 1024
NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")

EXAMPLE_CONFIG = {
    "_comment": "Private: real paths live here, never in the repo. {python} = the python running services.py "
                "(python.exe, even when started by pythonw.exe); {repo} = this cryoEV checkout; %VARS% and ~ "
                "are expanded. Optional per service: host (health check, default 127.0.0.1), health (a path such as /healthz: "
                "up = HTTP 200 + JSON {\"ok\": true}; unset = GET /api/ping, any HTTP answer is up), env {}, log.",
    "run_dir": "~/.cryoev/run",
    "services": [
        {"name": "review", "enabled": True, "port": 8766, "cwd": "{repo}",
         "argv": ["{python}", "annotation/review_server.py", "--package", "<CryoAI-laptop>/review/<pkg>",
                  "--port", "8766"],
         "log": "~/.cryoev/logs/review.log"},
        {"name": "sandbox", "enabled": True, "port": 8765, "cwd": "{repo}",
         "argv": ["{python}", "annotation/review_server.py", "--package", "<CryoAI-laptop>/review/SANDBOX_<pkg>",
                  "--port", "8765", "--sandbox"],
         "log": "~/.cryoev/logs/sandbox.log"},
        {"name": "history", "enabled": True, "port": 8768, "cwd": "{repo}",
         "argv": ["{python}", "-m", "http.server", "8768", "--bind", "127.0.0.1",
                  "--directory", "<CryoAI-laptop>/registry"],
         "log": "~/.cryoev/logs/history.log"},
        {"name": "monitor", "enabled": True, "port": None, "cwd": "{repo}",
         "argv": ["{python}", "jobs/monitor.py", "--share", "\\\\gpu-desktop\\cj", "--every", "30"],
         "log": "~/.cryoev/logs/monitor.log"},
        {"name": "portal", "enabled": False, "port": 8770, "cwd": "{repo}",
         "health": "/healthz",
         "argv": ["{python}", "-m", "portal.app", "--host", "127.0.0.1", "--port", "8770"],
         "log": "~/.cryoev/logs/portal.log"},
    ],
}


# ----------------------------------------------------------------------------------------------- config
@dataclass
class Service:
    name: str
    argv: list
    cwd: Path
    port: int = None
    host: str = "127.0.0.1"
    enabled: bool = True
    log: Path = None
    env: dict = field(default_factory=dict)
    health: str = None          # path (e.g. /healthz): up = 200 + JSON ok:true; None = /api/ping, any answer


@dataclass
class Config:
    path: Path
    run_dir: Path
    services: list


class ConfigError(Exception):
    pass


def python_exe() -> str:
    """The console python next to this interpreter: services get python.exe even when we run under pythonw."""
    exe = Path(sys.executable)
    if exe.name.lower() == "pythonw.exe" and (exe.parent / "python.exe").exists():
        return str(exe.parent / "python.exe")
    return str(exe)


def expand(s: str) -> str:
    s = s.replace("{python}", python_exe()).replace("{repo}", str(REPO))
    return os.path.expanduser(os.path.expandvars(s))


def read_text_any(path: Path) -> str:
    """UTF-8 (with or without BOM) or UTF-16 (what PowerShell 5.1's `>` writes)."""
    raw = path.read_bytes()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16")
    return raw.decode("utf-8-sig")


def load_config(path: Path, run_dir: Path = None) -> Config:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"no config at {path}: write one with `services.py example-config --out {path}`")
    try:
        data = json.loads(read_text_any(path))
    except (ValueError, UnicodeError) as e:
        raise ConfigError(f"{path}: not valid JSON ({e})")
    rd = Path(run_dir) if run_dir else Path(expand(data.get("run_dir") or str(DEFAULT_RUN_DIR)))
    services, names, ports = [], set(), {}
    for i, s in enumerate(data.get("services") or []):
        name = s.get("name", "")
        if not NAME_RE.match(name or ""):
            raise ConfigError(f"service #{i + 1}: bad or missing name {name!r} (letters, digits, _ . -)")
        if name in names:
            raise ConfigError(f"service {name}: duplicate name")
        names.add(name)
        argv = s.get("argv")
        if not isinstance(argv, list) or not argv or not all(isinstance(a, str) for a in argv):
            raise ConfigError(f"service {name}: argv must be a non-empty list of strings")
        port = s.get("port")
        if port is not None and (not isinstance(port, int) or not 0 < port < 65536):
            raise ConfigError(f"service {name}: port must be an integer 1-65535 or null")
        health_path = s.get("health")
        if health_path is not None and not (isinstance(health_path, str) and health_path.startswith("/")):
            raise ConfigError(f"service {name}: health must be a path starting with /, e.g. \"/healthz\"")
        enabled = bool(s.get("enabled", True))
        if port and enabled:
            if port in ports:
                raise ConfigError(f"services {ports[port]} and {name}: both enabled on port {port}")
            ports[port] = name
        log = Path(expand(s["log"])) if s.get("log") else rd.parent / "logs" / f"{name}.log"
        services.append(Service(name=name, argv=[expand(a) for a in argv], cwd=Path(expand(s.get("cwd") or "{repo}")),
                                port=port, host=s.get("host") or "127.0.0.1", enabled=enabled, log=log,
                                health=health_path,
                                env={str(k): expand(str(v)) for k, v in (s.get("env") or {}).items()}))
    return Config(path=path, run_dir=rd, services=services)


# --------------------------------------------------------------------------------------- process identity
if os.name == "nt":
    import ctypes
    from ctypes import wintypes

    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _nt = ctypes.WinDLL("ntdll")
    _k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    _k32.OpenProcess.restype = wintypes.HANDLE
    _k32.CloseHandle.argtypes = [wintypes.HANDLE]
    _k32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    _k32.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    _k32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
    _k32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    _k32.WaitForSingleObject.restype = wintypes.DWORD
    _k32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    _k32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    _nt.NtQueryInformationProcess.argtypes = [wintypes.HANDLE, ctypes.c_ulong, ctypes.c_void_p, ctypes.c_ulong,
                                              ctypes.POINTER(ctypes.c_ulong)]
    _nt.NtQueryInformationProcess.restype = ctypes.c_long

    class _PE32(ctypes.Structure):
        _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD), ("th32ProcessID", wintypes.DWORD),
                    ("th32DefaultHeapID", ctypes.c_size_t), ("th32ModuleID", wintypes.DWORD),
                    ("cntThreads", wintypes.DWORD), ("th32ParentProcessID", wintypes.DWORD),
                    ("pcPriClassBase", ctypes.c_long), ("dwFlags", wintypes.DWORD),
                    ("szExeFile", ctypes.c_wchar * 260)]

    class _UStr(ctypes.Structure):
        _fields_ = [("Length", wintypes.USHORT), ("MaximumLength", wintypes.USHORT), ("Buffer", ctypes.c_void_p)]

    _k32.Process32FirstW.argtypes = _k32.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PE32)]

    _QUERY, _TERMINATE, _SYNC, _STILL_ACTIVE = 0x1000, 0x0001, 0x00100000, 259


class Proc:
    """A handle on one process. While the handle is open (Windows) its pid cannot be reused, so the identity
    checked here is the process that gets terminated."""

    def __init__(self, pid: int):
        self.pid = int(pid)
        self.h = None
        if os.name == "nt" and self.pid > 0:
            self.h = _k32.OpenProcess(_QUERY | _TERMINATE | _SYNC, False, self.pid) or \
                _k32.OpenProcess(_QUERY | _SYNC, False, self.pid) or None

    def close(self):
        if self.h:
            _k32.CloseHandle(self.h)
            self.h = None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def alive(self) -> bool:
        if os.name == "nt":
            code = wintypes.DWORD()
            return bool(self.h) and bool(_k32.GetExitCodeProcess(self.h, ctypes.byref(code))) \
                and code.value == _STILL_ACTIVE
        try:
            os.kill(self.pid, 0)
        except OSError:
            return False
        stat = Path(f"/proc/{self.pid}/stat")
        return not (stat.exists() and stat.read_text().rsplit(")", 1)[-1].split()[0] == "Z")

    def ctime(self):
        """Creation time (opaque int, comparable between calls), or None."""
        if os.name == "nt":
            if not self.h:
                return None
            t = [wintypes.FILETIME() for _ in range(4)]
            if not _k32.GetProcessTimes(self.h, *[ctypes.byref(x) for x in t]):
                return None
            return (t[0].dwHighDateTime << 32) | t[0].dwLowDateTime
        try:
            return int(Path(f"/proc/{self.pid}/stat").read_text().rsplit(")", 1)[-1].split()[19])
        except (OSError, IndexError, ValueError):
            return None

    def cmdline(self):
        """The process's command line as one string (Windows) or NUL-joined argv (Linux); None if unreadable."""
        if os.name == "nt":
            if not self.h:
                return None
            size = ctypes.c_ulong(0)
            _nt.NtQueryInformationProcess(self.h, 60, None, 0, ctypes.byref(size))  # ProcessCommandLineInformation
            if not size.value:
                return None
            buf = ctypes.create_string_buffer(size.value)
            if _nt.NtQueryInformationProcess(self.h, 60, buf, size, ctypes.byref(size)) != 0:
                return None
            us = _UStr.from_buffer(buf)
            return ctypes.wstring_at(us.Buffer, us.Length // 2) if us.Buffer else ""
        try:
            return Path(f"/proc/{self.pid}/cmdline").read_bytes().rstrip(b"\0").decode(errors="replace")
        except OSError:
            return None

    def wait(self, timeout: float) -> bool:
        """True once the process has exited."""
        if os.name == "nt":
            return not self.h or _k32.WaitForSingleObject(self.h, int(timeout * 1000)) == 0
        end = time.time() + timeout
        while time.time() < end:
            if not self.alive():
                return True
            time.sleep(0.1)
        return not self.alive()


def cmdline_of(argv) -> str:
    """What the OS will report as the command line of a process we start with `argv`."""
    return subprocess.list2cmdline(argv) if os.name == "nt" else "\0".join(argv)


def identify(rec: dict):
    """(verdict, Proc or None) for a pid-file record: 'ours' (alive, same command line and creation time),
    'dead', or 'foreign' (alive but a different process, or unreadable: never ours)."""
    p = Proc(rec.get("pid", 0))
    if not p.alive():
        p.close()
        return "dead", None
    cl, ct = p.cmdline(), p.ctime()
    if cl is not None and ct is not None and cl == rec.get("cmdline") and ct == rec.get("ctime"):
        return "ours", p
    p.close()
    return "foreign", None


def descendants(root: Proc) -> list:
    """Live descendants of `root` (open Procs), each created after its parent (rules out a reused parent pid)."""
    if os.name != "nt":
        return []
    snap = _k32.CreateToolhelp32Snapshot(0x2, 0)
    if not snap or snap == wintypes.HANDLE(-1).value:
        return []
    kids = {}
    try:
        e = _PE32()
        e.dwSize = ctypes.sizeof(_PE32)
        ok = _k32.Process32FirstW(snap, ctypes.byref(e))
        while ok:
            kids.setdefault(e.th32ParentProcessID, []).append(e.th32ProcessID)
            ok = _k32.Process32NextW(snap, ctypes.byref(e))
    finally:
        _k32.CloseHandle(snap)
    out, todo = [], [root]
    while todo:
        parent = todo.pop()
        pct = parent.ctime()
        for pid in kids.get(parent.pid, []):
            if pid == parent.pid:
                continue
            c = Proc(pid)
            ct = c.ctime()
            if c.alive() and pct is not None and ct is not None and ct >= pct:
                out.append(c)
                todo.append(c)
            else:
                c.close()
    return out


def terminate(p: Proc, timeout: float = 10.0) -> bool:
    """Stop a verified process and its descendants. True once the root has exited."""
    if os.name == "nt":
        kids = descendants(p)
        try:
            _k32.TerminateProcess(p.h, 1)
            for c in kids:
                if c.alive():
                    _k32.TerminateProcess(c.h, 1)
            return p.wait(timeout)
        finally:
            for c in kids:
                c.close()
    import signal
    try:
        os.killpg(p.pid, signal.SIGTERM)  # started with start_new_session: its own group
    except OSError:
        os.kill(p.pid, signal.SIGTERM)
    if p.wait(timeout):
        return True
    os.kill(p.pid, signal.SIGKILL)
    return p.wait(2)


# ----------------------------------------------------------------------------------------- pid files, net
def pid_path(cfg: Config, name: str) -> Path:
    return cfg.run_dir / f"{name}.pid"


def read_pid(cfg: Config, name: str):
    try:
        rec = json.loads(pid_path(cfg, name).read_text(encoding="utf-8"))
        return rec if isinstance(rec, dict) and isinstance(rec.get("pid"), int) else {"pid": 0}
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        return {"pid": 0}  # unreadable: treated as stale, never as ours


def write_pid(cfg: Config, name: str, rec: dict):
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    tmp = pid_path(cfg, name).with_suffix(".pid.tmp")
    tmp.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    os.replace(tmp, pid_path(cfg, name))


def remove_pid(cfg: Config, name: str):
    try:
        pid_path(cfg, name).unlink()
    except FileNotFoundError:
        pass


def listening(host: str, port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def health(host: str, port: int, timeout: float = 1.0, path: str = None) -> str:
    """No `path`: GET /api/ping, any HTTP answer is up ('http 200', 'http 403' for a review server without its key,
    'http 404' for http.server). With `path` (a service's "health" option, e.g. /healthz): up only on HTTP 200 and
    a JSON body with "ok": true -> 'ok'; any other answer -> 'http <status>' / 'bad body' (not up). Accepts but no
    HTTP: 'tcp'. Nothing: 'down'."""
    if not listening(host, port, timeout):
        return "down"
    c = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        c.request("GET", path or "/api/ping")
        r = c.getresponse()
        body = r.read(4096)
        if not path:
            return f"http {r.status}"
        if r.status != 200:
            return f"http {r.status}"
        try:
            return "ok" if json.loads(body).get("ok") is True else "bad body"
        except (ValueError, AttributeError):
            return "bad body"
    except (OSError, http.client.HTTPException):
        return "tcp"
    finally:
        c.close()


def fmt_uptime(sec) -> str:
    if sec is None:
        return "-"
    sec = int(sec)
    d, sec = divmod(sec, 86400)
    h, sec = divmod(sec, 3600)
    m, s = divmod(sec, 60)
    return f"{d}d{h:02d}h" if d else f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


# ------------------------------------------------------------------------------------------------ commands
def service_status(cfg: Config, svc: Service) -> dict:
    rec = read_pid(cfg, svc.name)
    verdict, p = identify(rec) if rec else (None, None)
    if p:
        p.close()
    listen = listening(svc.host, svc.port) if svc.port else None
    if verdict == "ours":
        state = "running" if listen is not False else "running (not listening)"
    elif listen:
        state = "port busy (not ours)"
    elif rec:
        state = "stale pid file"
    else:
        state = "stopped" if svc.enabled else "disabled"
    return {"name": svc.name, "enabled": svc.enabled, "state": state,
            "pid": rec["pid"] if verdict == "ours" else None, "port": svc.port, "listening": listen,
            "health": health(svc.host, svc.port, path=svc.health) if listen else ("-" if svc.port is None else "down"),
            "uptime_s": round(time.time() - rec.get("started", time.time())) if verdict == "ours" else None,
            "log": str(svc.log)}


def print_table(rows: list):
    cols = ["name", "enabled", "state", "pid", "port", "listening", "health", "uptime", "log"]
    def cell(r, c):
        v = fmt_uptime(r["uptime_s"]) if c == "uptime" else r[c]
        return "-" if v is None else ("yes" if v is True else "no" if v is False else str(v))
    table = [cols] + [[cell(r, c) for c in cols] for r in rows]
    w = [max(len(row[i]) for row in table) for i in range(len(cols))]
    for row in table:
        print("  ".join(v.ljust(w[i]) for i, v in enumerate(row)).rstrip())


def rotate_log(log: Path):
    log.parent.mkdir(parents=True, exist_ok=True)
    try:
        if log.stat().st_size > LOG_MAX_BYTES:
            os.replace(log, log.with_name(log.name + ".1"))
    except FileNotFoundError:
        pass


def log_tail(log: Path, n: int = 8) -> str:
    try:
        lines = log.read_text(encoding="utf-8", errors="replace").splitlines()[-n:]
    except OSError:
        return ""
    return "\n".join("    | " + ln for ln in lines)


def spawn(svc: Service) -> subprocess.Popen:
    rotate_log(svc.log)
    logf = open(svc.log, "ab")
    try:
        logf.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} services.py start {svc.name}: "
                   f"{cmdline_of(svc.argv).replace(chr(0), ' ')}\n".encode("utf-8"))
        logf.flush()
        env = dict(os.environ, PYTHONUNBUFFERED="1", **svc.env)
        kw = dict(cwd=str(svc.cwd), env=env, stdin=subprocess.DEVNULL, stdout=logf, stderr=subprocess.STDOUT)
        if os.name != "nt":
            return subprocess.Popen(svc.argv, start_new_session=True, **kw)
        flags = 0x08000000 | 0x00000008 | 0x00000200  # CREATE_NO_WINDOW | DETACHED_PROCESS | NEW_PROCESS_GROUP
        try:  # leave the caller's job object, so closing the shell/session that ran us does not kill it
            return subprocess.Popen(svc.argv, creationflags=flags | 0x01000000, **kw)  # CREATE_BREAKAWAY_FROM_JOB
        except PermissionError:
            return subprocess.Popen(svc.argv, creationflags=flags, **kw)  # the job forbids breakaway
    finally:
        logf.close()


def start_one(cfg: Config, svc: Service, wait: float):
    """-> (ok: bool, message)."""
    if not svc.enabled:
        return True, "skipped: disabled in config"
    rec = read_pid(cfg, svc.name)
    note = ""
    if rec:
        verdict, p = identify(rec)
        if p:
            p.close()
            return True, f"already running (pid {rec['pid']})"
        remove_pid(cfg, svc.name)
        note = f" (stale pid file for pid {rec['pid']} removed)"
    if svc.port and listening(svc.host, svc.port):
        return False, f"port busy (not ours): something else listens on {svc.port}; left alone{note}"
    if not svc.cwd.is_dir():
        return False, f"cwd does not exist: {svc.cwd}"
    try:
        proc = spawn(svc)
    except OSError as e:
        return False, f"could not start: {e}"
    with Proc(proc.pid) as p:
        ct = p.ctime()
    write_pid(cfg, svc.name, {"pid": proc.pid, "ctime": ct, "cmdline": cmdline_of(svc.argv), "argv": svc.argv,
                              "cwd": str(svc.cwd), "port": svc.port, "log": str(svc.log), "started": time.time()})
    t0 = time.time()
    grace = min(2.0, wait)
    while True:
        rc = proc.poll()
        el = time.time() - t0
        if rc is not None:
            remove_pid(cfg, svc.name)
            return False, f"exited with code {rc} after {el:.1f} s; log {svc.log}:\n{log_tail(svc.log)}"
        if svc.port and listening(svc.host, svc.port):
            return True, f"started (pid {proc.pid}), listening on {svc.port} after {el:.1f} s{note}"
        if not svc.port and el >= grace:
            return True, f"started (pid {proc.pid}), alive after {el:.1f} s{note}"
        if el >= wait:
            return True, f"started (pid {proc.pid}) but not listening on {svc.port} after {wait:.0f} s; " \
                         f"check {svc.log}{note}"
        time.sleep(0.2)


def stop_one(cfg: Config, svc: Service, timeout: float = 10.0):
    """-> (ok: bool, message). Kills only a process whose pid, command line and creation time match its pid file."""
    rec = read_pid(cfg, svc.name)
    if not rec:
        if svc.port and listening(svc.host, svc.port):
            return False, f"port busy (not ours): {svc.port} is held by a process services.py did not start; left alone"
        return True, "not running"
    verdict, p = identify(rec)
    if verdict == "dead":
        remove_pid(cfg, svc.name)
        return True, "not running (stale pid file removed)"
    if verdict == "foreign":
        remove_pid(cfg, svc.name)
        return True, f"pid {rec['pid']} is another process now (command line or start time differ): left alone; " \
                     "stale pid file removed"
    with p:
        done = terminate(p, timeout)
    if not done:
        return False, f"pid {rec['pid']} did not exit within {timeout:.0f} s"
    remove_pid(cfg, svc.name)
    return True, f"stopped (pid {rec['pid']})"


def select(cfg: Config, names: list) -> list:
    if not names or names == ["all"]:
        return list(cfg.services)
    by = {s.name: s for s in cfg.services}
    unknown = [n for n in names if n not in by]
    if unknown:
        raise ConfigError(f"not in {cfg.path}: {', '.join(unknown)} (known: {', '.join(by) or 'none'})")
    return [by[n] for n in names]


def run_actions(cfg: Config, svcs: list, action: str, wait: float) -> int:
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    lock = acquire_lock(cfg.run_dir / "services.lock", wait=60)
    if lock is None:
        print("[ERR] another services.py holds services.lock for 60 s", flush=True)
        return 1
    rc = 0
    try:
        for svc in svcs:
            if action in ("stop", "restart"):
                ok, msg = stop_one(cfg, svc)
                print(f"[{'OK' if ok else 'WARN'}] {svc.name}: {msg}", flush=True)
                rc |= not ok
                if not ok and action == "restart":
                    continue
            if action in ("start", "restart"):
                ok, msg = start_one(cfg, svc, wait)
                print(f"[{'OK' if ok else 'ERR'}] {svc.name}: {msg}", flush=True)
                rc |= not ok
    finally:
        lock.close()
    return int(bool(rc))


def main(argv=None) -> int:
    if sys.stdout is None:  # pythonw (the logon task): no console, keep a log instead
        (HOME_DIR / "logs").mkdir(parents=True, exist_ok=True)
        sys.stdout = sys.stderr = open(HOME_DIR / "logs" / "services.log", "a", encoding="utf-8")
        print(f"--- {time.strftime('%Y-%m-%d %H:%M:%S')} services.py {' '.join(argv or sys.argv[1:])}")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", type=Path, default=os.environ.get("CRYOEV_SERVICES_CONFIG") or DEFAULT_CONFIG,
                    help=f"default env CRYOEV_SERVICES_CONFIG, else {DEFAULT_CONFIG}")
    ap.add_argument("--run-dir", type=Path, help="pid files (default: config run_dir, else ~/.cryoev/run)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    st = sub.add_parser("status")
    st.add_argument("--json", action="store_true")
    for name in ("start", "stop", "restart"):
        p = sub.add_parser(name)
        p.add_argument("names", nargs="*", help="service names, or all (default)")
        p.add_argument("--wait", type=float, default=15.0, help="seconds to wait for a port to listen (start)")
    ex = sub.add_parser("example-config")
    ex.add_argument("--out", type=Path, help="write the template here (UTF-8); refuses to overwrite")
    a = ap.parse_args(argv)

    if a.cmd == "example-config":
        text = json.dumps(EXAMPLE_CONFIG, indent=2) + "\n"
        if not a.out:
            print(text, end="")
            return 0
        if a.out.exists():
            print(f"[ERR] {a.out} exists; not overwritten")
            return 2
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(text, encoding="utf-8")
        print(f"[OK] wrote {a.out}: replace the <placeholders>, then `services.py status`")
        return 0
    try:
        cfg = load_config(a.config, a.run_dir)
        if a.cmd == "status":
            rows = [service_status(cfg, s) for s in cfg.services]
            if a.json:
                print(json.dumps(rows, indent=1))
            else:
                print_table(rows)
            return 0
        return run_actions(cfg, select(cfg, a.names), a.cmd, a.wait)
    except ConfigError as e:
        print(f"[ERR] {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
