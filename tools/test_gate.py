"""Run the repo's test suites in parallel, per test file, and record how long each file takes.

  python tools/test_gate.py --affected [--base origin/main]   # only the suites the changed files touch (default)
  python tools/test_gate.py --full                            # every suite (big go-lives)
  python tools/test_gate.py --suites portal,results           # named suites
  python tools/test_gate.py --times                           # slowest test files from past runs, then exit
  python tools/test_gate.py --plan [--files a.py,b.py]        # what a gate would run + estimated wall; runs nothing

Each test file is one unit: `python -m unittest discover -s <suite dir> -p <file>`, the same import setup as the
classic `discover -s <suite dir>` gate. Units run in parallel (--jobs, default CPU count // 2), slowest first (from
past timings). Files that drive a real browser (Playwright, or an import of a browser helper module) share a smaller
lane (--browser-jobs, default 2), because several browsers at once on a loaded laptop is what makes the known
load-sensitive tests flaky.

A failed file is run once more on its own (--retry 1). One that then passes is reported as FLAKY, never as a plain
pass; the gate still exits 0 for it, so say so when reporting. Anything that fails twice fails the gate.
The pre-push scan (tools/prepush_scan.py --range <base>..HEAD) runs at the end unless --no-scan.

Browser skip (round 11 #6, --affected only): when EVERY changed file is on the non-UI allowlist (NON_UI: registry/
except history_page.py, jobs/, training/, inference/, data_utils/, docs/, tools/ except this gate and its test,
cryoev/) and none is unknown, the children run with CJ_GATE_SKIP_BROWSER=1: the test classes (or methods) marked
@browser (tests/cryoev_testkit.py) skip, and every other class in the same file still runs. No file is skipped. A
changed test file runs in full. The gate prints "N browser classes skipped (no UI change)"; --full and --suites never
skip. --plan prints the selection and an estimate from past timings without running anything.

One gate runs at a time per machine (round 10): a lock directory in ~/.cryoev/gate/ is shared by every worktree. A
second gate waits up to --lock-wait minutes (default 45) and then fails "gate busy"; a lock whose owner process is gone,
or whose heartbeat is older than 10 minutes, is stale and is taken over.

Child processes run with CRYOEV_VIA / CRYOEV_FOR removed from the environment (the registry and jobs writers default
their "via/for" actor fields from them, so an agent's own env would leak into test data). --agent-env runs the children
with CRYOEV_VIA=claude CRYOEV_FOR=pujan on purpose, to prove the suites do not depend on the env being unset.

Timings are appended to ~/.cryoev/gate/timings.jsonl (one line per file per run, failures included, with retries, the
retry time and the tail of a failure) for the dev-time audit; slowest-first uses the median of past passing runs.
Exit 0 = every unit passed (flaky ones noted), 1 = a failure or gate busy, 2 = usage error.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SUITES = {                       # name -> test dir (relative to the repo)
    "jobs": "jobs/tests", "tests": "tests", "registry": "registry/tests", "results": "results/tests",
    "tools": "tools/tests", "annotation": "annotation/tests", "data_utils": "data_utils/tests",
    "portal": "portal/tests", "cryoev": "cryoev/tests",
}
# changed path prefix -> suites that exercise it (a file's own suite plus the suites that import it)
AFFECTS = [
    ("portal/", {"portal", "cryoev", "annotation"}),
    ("annotation/", {"annotation", "portal", "results", "cryoev"}),
    ("results/", {"results", "portal", "jobs"}),
    ("jobs/", {"jobs", "portal", "tools", "cryoev"}),
    ("training/", {"jobs", "portal", "annotation", "results"}),
    # dataset_from_lv uses the registry; review/tools code reads it too
    ("registry/", {"registry", "portal", "jobs", "data_utils", "cryoev", "annotation", "tools"}),
    ("data_utils/", set(SUITES)),            # oslock etc. are used everywhere
    ("tools/", {"tools", "cryoev", "portal"}),
    ("client/", {"portal"}),                 # client/cryoev_client.py is served by the portal
    ("cryoev/", {"cryoev"}),                 # the agent CLI (docs/AGENTS.md)
    ("tests/", {"tests"}),
    ("inference/", {"jobs", "portal", "annotation", "results"}),
    ("models/", {"jobs", "portal", "registry", "cryoev"}),
]
DOC_ONLY = re.compile(r"(\.md|\.txt|\.png|\.jpg|\.svg)$", re.I)
# .md/.txt files that are NOT doc-only: they change what runs (CHANGELOG is parsed by tests; requirements pin the env)
NOT_DOC = re.compile(r"(^|/)(CHANGELOG\.md|requirements[^/]*\.txt)$", re.I)
ALWAYS = {"tests"}                           # repo-wide checks, ~1 min
BROWSER_HELPERS = {"test_review_tool_browser"}   # test modules wrapping Playwright; importing one = a browser file
ENV_STRIP = ("CRYOEV_VIA", "CRYOEV_FOR")
# --affected only: the @browser classes skip (CJ_GATE_SKIP_BROWSER=1) when EVERY changed file is on this non-UI
# allowlist (round 11 #6)
NON_UI = ("registry/", "jobs/", "training/", "inference/", "data_utils/", "docs/", "tools/", "cryoev/")
NON_UI_EXCEPT = {"registry/history_page.py", "tools/test_gate.py", "tools/tests/test_test_gate.py"}
SKIP_BROWSER_ENV = "CJ_GATE_SKIP_BROWSER"     # = tests/cryoev_testkit.py SKIP_BROWSER_ENV (@browser)
AGENT_ENV = {"CRYOEV_VIA": "claude", "CRYOEV_FOR": "pujan"}
LOCK_DIR = Path.home() / ".cryoev" / "gate" / "lock"
LOCK_WAIT_MIN = 45.0
STALE_HEARTBEAT_S = 600
HEARTBEAT_S = 30
TIMINGS = Path.home() / ".cryoev" / "gate" / "timings.jsonl"
SUMMARY = re.compile(r"^(Ran \d+ tests? in [\d.]+s|OK.*|FAILED.*)$", re.M)


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout


def changed_files(base: str) -> list[str]:
    files = set(git("diff", "--name-only", f"{base}...HEAD").split())
    files |= set(git("diff", "--name-only", "HEAD").split())          # uncommitted edits count too
    return sorted(files)


def affected_suites(files: list[str]) -> tuple[set[str], list[str]]:
    """-> (suites, files that matched no rule). An unknown code file means: run everything."""
    out, unknown = set(ALWAYS), []
    for f in (x.replace("\\", "/") for x in files):
        if f.startswith(".claude/"):
            continue
        hits = [s for p, s in AFFECTS if f.startswith(p)]
        if hits:
            for s in hits:
                out |= s
        elif NOT_DOC.search(f) or not DOC_ONLY.search(f):
            unknown.append(f)
    if unknown:
        out = set(SUITES)
    return out, unknown


def ui_untouched(files: list[str]) -> bool:
    """True when there is a changed file and every one is on the non-UI allowlist (.claude/ ignored)."""
    files = [f.replace("\\", "/") for f in files if not f.startswith(".claude/")]
    return bool(files) and all(f.startswith(NON_UI) and f not in NON_UI_EXCEPT for f in files)


def browser_marks(path: Path) -> tuple[int, int]:
    """-> (classes, test methods) decorated @browser in a test file (static: the decorator by name)."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except (OSError, SyntaxError):
        return 0, 0
    n_cls = n_fn = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and any(
                (isinstance(d, ast.Name) and d.id == "browser") or (isinstance(d, ast.Attribute) and d.attr == "browser")
                for d in node.decorator_list):
            if isinstance(node, ast.ClassDef):
                n_cls += 1
            else:
                n_fn += 1
    return n_cls, n_fn


def is_browser_file(path: Path) -> bool:
    """True if the file imports playwright or a browser helper module (any import, even inside a function).

    Static imports only: a module loaded with importlib / __import__ / a string is not seen (none of the tests do that).

    Read from the imports, not the text: four browser files only import test_review_tool_browser and never name
    playwright, and a comment that says "playwright" must not make a file a browser file.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return "playwright" in text
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            mods = [node.module or ""]
        else:
            continue
        for m in mods:
            top = m.split(".")[0]
            if top == "playwright" or top in BROWSER_HELPERS:
                return True
    return False


def child_env(agent_env: bool = False, base: dict | None = None, skip_browser: bool = False) -> dict:
    """Environment for every child: CRYOEV_VIA / CRYOEV_FOR removed (or set to the agent values on purpose);
    CJ_GATE_SKIP_BROWSER=1 only when skip_browser (never inherited from the caller's shell)."""
    env = dict(os.environ if base is None else base)
    for k in ENV_STRIP + (SKIP_BROWSER_ENV,):
        env.pop(k, None)
    if agent_env:
        env.update(AGENT_ENV)
    if skip_browser:
        env[SKIP_BROWSER_ENV] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":                      # os.kill(pid, 0) would TERMINATE the process on Windows
        import ctypes
        k = ctypes.windll.kernel32
        h = k.OpenProcess(0x1000, False, pid)            # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        ok = k.GetExitCodeProcess(h, ctypes.byref(code))
        k.CloseHandle(h)
        return bool(ok) and code.value == 259            # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class GateLock:
    """Machine-wide single-gate lock: an atomically created directory holding owner.json (pid) and a heartbeat file.

    Shared by every worktree (it lives under ~/.cryoev/gate/). Stale = the owner pid is gone, or the heartbeat is
    older than `stale_s`; a stale lock is removed and taken over.
    """

    def __init__(self, path: Path = None, wait_min: float = None, stale_s: float = None, poll_s: float = 2.0):
        self.path = Path(path or LOCK_DIR)
        self.wait_s = (LOCK_WAIT_MIN if wait_min is None else wait_min) * 60
        self.stale_s = STALE_HEARTBEAT_S if stale_s is None else stale_s
        self.poll_s = poll_s
        self._stop = threading.Event()

    def owner(self) -> dict:
        try:
            return json.loads((self.path / "owner.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _heartbeat_age(self) -> float:
        for f in (self.path / "heartbeat", self.path):
            try:
                return time.time() - f.stat().st_mtime
            except OSError:
                continue
        return 0.0

    def _stale_dir(self, d: Path) -> bool:
        try:
            pid = json.loads((d / "owner.json").read_text(encoding="utf-8")).get("pid")
        except (OSError, ValueError):
            pid = None
        if isinstance(pid, int) and not _pid_alive(pid):
            return True
        for f in (d / "heartbeat", d):
            try:
                return time.time() - f.stat().st_mtime > self.stale_s
            except OSError:
                continue
        return False

    def is_stale(self) -> bool:
        return self._stale_dir(self.path)

    def try_acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.path.mkdir()
        except FileExistsError:
            if not self.is_stale():
                return False
            # Atomic takeover: rename the stale dir to a unique name first; only the waiter whose rename succeeds goes
            # on (a second waiter's rename finds no dir and fails). The renamed dir is re-checked, so a waiter that
            # raced a fresh lock into place puts it back instead of deleting a live gate's lock.
            grave = self.path.with_name(f"{self.path.name}.stale.{os.getpid()}.{uuid.uuid4().hex[:8]}")
            try:
                os.rename(self.path, grave)
            except OSError:
                return False
            if not self._stale_dir(grave):
                try:
                    os.rename(grave, self.path)
                except OSError:
                    pass
                return False
            shutil.rmtree(grave, ignore_errors=True)
            try:
                self.path.mkdir()
            except OSError:
                return False
        (self.path / "owner.json").write_text(json.dumps({"pid": os.getpid(), "cwd": str(REPO), "t": time.time()}),
                                              encoding="utf-8")
        (self.path / "heartbeat").write_text("x", encoding="utf-8")
        self._stop.clear()
        threading.Thread(target=self._beat, daemon=True).start()
        return True

    def _beat(self):
        every = HEARTBEAT_S if self.stale_s >= 120 else max(0.05, self.stale_s / 4)
        while not self._stop.wait(every):
            try:
                (self.path / "heartbeat").write_text(str(time.time()), encoding="utf-8")
            except OSError:
                pass

    def acquire(self, log=print) -> bool:
        """Wait up to wait_s for the lock. False = gate busy."""
        t0, told = time.time(), False
        while not self.try_acquire():
            if time.time() - t0 >= self.wait_s:
                return False
            if not told:
                o = self.owner()
                log(f"another gate is running (pid {o.get('pid')}, {o.get('cwd')}); waiting up to "
                    f"{self.wait_s / 60:.0f} min")
                told = True
            time.sleep(self.poll_s)
        return True

    def release(self):
        self._stop.set()
        if self.owner().get("pid") != os.getpid():
            return
        for f in self.path.glob("*"):
            try:
                f.unlink()
            except OSError:
                pass
        try:
            self.path.rmdir()
        except OSError:
            pass


def past_times(path: Path = None, keep: int = 9, skip_browser: bool = False) -> dict[str, float]:
    """Median seconds per unit over its last `keep` passing runs (one slow loaded run must not reorder the queue).
    skip_browser: medians from runs with the @browser classes skipped where a unit has any, else from full runs."""
    seen: dict[bool, dict[str, list[float]]] = {False: {}, True: {}}
    try:
        for line in Path(path or TIMINGS).read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
                if r.get("ok", True) is False:
                    continue
                seen[bool(r.get("skip_browser"))].setdefault(r["unit"], []).append(float(r["seconds"]))
            except (ValueError, KeyError, TypeError, AttributeError):
                continue
    except OSError:
        pass
    out = {u: statistics.median(v[-keep:]) for u, v in seen[False].items()}
    if skip_browser:
        out.update({u: statistics.median(v[-keep:]) for u, v in seen[True].items()})
    return out


def units_for(suites: set[str]) -> list[tuple[str, str, bool]]:
    """-> [(suite, test file name, is_browser)]"""
    out = []
    for s in sorted(suites):
        d = REPO / SUITES[s]
        for f in sorted(d.glob("test*.py")):
            out.append((s, f.name, is_browser_file(f)))
    return out


def run_unit(suite: str, name: str, timeout: int, agent_env: bool = False, skip_browser: bool = False) -> dict:
    t0 = time.time()
    cmd = [sys.executable, "-m", "unittest", "discover", "-s", SUITES[suite], "-p", name]
    try:
        p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=timeout,
                           env=child_env(agent_env, skip_browser=skip_browser))
        text, rc = p.stdout + p.stderr, p.returncode
    except subprocess.TimeoutExpired as e:
        text, rc = f"TIMEOUT after {timeout}s\n" + ((e.stdout or "") if isinstance(e.stdout, str) else ""), 124
    m = SUMMARY.findall(text)
    ran = re.search(r"Ran (\d+) test", text)
    return {"unit": f"{suite}/{name}", "suite": suite, "file": name, "rc": rc, "ok": rc == 0,
            "seconds": round(time.time() - t0, 1), "tests": int(ran.group(1)) if ran else 0,
            "summary": " | ".join(m[-2:]) if m else "", "output": text, "retries": 0, "skip_browser": skip_browser}


def record(results: list[dict], mode: str, agent_env: bool = False, path: Path = None) -> None:
    path = Path(path or TIMINGS)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        head = git("rev-parse", "--short", "HEAD").strip()
        with open(path, "a", encoding="utf-8") as f:
            for r in results:
                tail = r.get("first_fail_tail") or ("" if r["ok"] else r["output"][-600:])
                f.write(json.dumps({"t": stamp, "head": head, "mode": mode, "unit": r["unit"],
                                    "seconds": r["seconds"], "tests": r["tests"], "ok": r["ok"],
                                    "flaky": r.get("flaky", False), "retries": r.get("retries", 0),
                                    "retry_seconds": r.get("retry_seconds"), "agent_env": agent_env,
                                    "skip_browser": bool(r.get("skip_browser")),
                                    "fail_tail": tail}) + "\n")
    except (OSError, subprocess.CalledProcessError) as e:
        print(f"(timings not recorded: {e})", file=sys.stderr)


def estimate_wall(units: list, times: dict, jobs: int, browser_jobs: int) -> float:
    """Seconds the gate would take for `units` (already in run order): the median per file (60 s if unknown) run on
    `jobs` workers, browser files also holding one of `browser_jobs` lanes (a worker waits for a lane, as in
    run_gate). No retries, no scan."""
    import heapq
    from collections import deque
    running, blocked = [], deque()
    t, free, lane, i = 0.0, jobs, browser_jobs, 0
    while i < len(units) or running:
        while free > 0 and i < len(units):
            s, n, b = units[i]
            i += 1
            free -= 1
            d = times.get(f"{s}/{n}", 60.0)
            if b and lane <= 0:
                blocked.append(d)
            else:
                lane -= b
                heapq.heappush(running, (t + d, b))
        if not running:
            break
        t, b = heapq.heappop(running)
        free += 1
        if b:
            lane += 1
            if blocked:
                lane -= 1
                heapq.heappush(running, (t + blocked.popleft(), True))
    return t


def skip_mode(mode: str, files: list | None) -> bool:
    """CJ_GATE_SKIP_BROWSER for this run: --affected only, every changed file non-UI, none unknown."""
    return mode == "affected" and files is not None and ui_untouched(files) and not affected_suites(files)[1]


def select(suites: set[str], mode: str, files: list | None, timings: Path = None) -> tuple[list, set, dict]:
    """-> (units in run order, the units run with CJ_GATE_SKIP_BROWSER=1, past medians). Every unit runs; in skip mode
    a unit's browser flag is cleared (its browser classes skip, so it needs no browser lane), except a changed test
    file, which runs in full."""
    units, skip = units_for(suites), set()
    if skip_mode(mode, files):
        changed = {f.replace("\\", "/") for f in files}
        skip = {f"{s}/{n}" for s, n, _ in units if f"{SUITES[s]}/{n}" not in changed}
        units = [(s, n, b and f"{s}/{n}" not in skip) for s, n, b in units]
    order = past_times(timings, skip_browser=bool(skip))
    if skip:
        full = past_times(timings)
        order = {u: (order.get(u, 60.0) if u in skip else full.get(u, 60.0)) for u in set(order) | set(full)}
    units.sort(key=lambda u: -order.get(f"{u[0]}/{u[1]}", 60.0))
    return units, skip, order


def skip_note(units: list, skip: set) -> str:
    if not skip:
        return ""
    n_cls = n_fn = 0
    for s, n, _ in units:
        if f"{s}/{n}" in skip:
            c, f = browser_marks(REPO / SUITES[s] / n)
            n_cls, n_fn = n_cls + c, n_fn + f
    return f"{n_cls} browser classes skipped (no UI change)" + (f", plus {n_fn} browser test methods" if n_fn else "")


def plan(a, suites: set[str], mode: str, files: list | None) -> int:
    """--plan: the selection and an estimate from past timings; nothing runs and no lock is taken."""
    units, skip, order = select(suites, mode, files, a.timings)
    serial = sum(order.get(f"{s}/{n}", 60.0) for s, n, _ in units)
    nb = sum(1 for u in units if u[2])
    print(f"plan ({mode}): suites {', '.join(sorted(suites))}")
    if files is not None:
        print(f"changed files ({len(files)}): {', '.join(files[:8])}" + (" ..." if len(files) > 8 else ""))
    print(f"{len(units)} test files ({nb} browser); estimated {estimate_wall(units, order, a.jobs, a.browser_jobs) / 60:.1f} "
          f"min wall at {a.jobs} jobs / {a.browser_jobs} browser ({serial / 60:.1f} min serial), from past medians")
    if skip:
        print(skip_note(units, skip))
    for s, n, b in units:
        print(f"  {order.get(f'{s}/{n}', 60.0):7.1f}s  {s}/{n}" + ("  [browser]" if b else ""))
    return 0


def show_times(n: int) -> int:
    t = past_times()
    if not t:
        print("No timings yet: run the gate once.")
        return 0
    total = sum(t.values())
    print(f"{len(t)} test files, {total / 60:.1f} min if run one after another (median); slowest {n}:")
    for unit, s in sorted(t.items(), key=lambda kv: -kv[1])[:n]:
        print(f"  {s:7.1f}s  {unit}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--full", action="store_true")
    g.add_argument("--affected", action="store_true", help="default")
    g.add_argument("--suites", help="comma-separated: " + ",".join(SUITES))
    g.add_argument("--times", action="store_true", help="show the slowest test files from past runs")
    ap.add_argument("--base", default="origin/main")
    ap.add_argument("--plan", action="store_true", help="print the selection and estimated wall time; run nothing")
    ap.add_argument("--files", help="comma-separated changed files to use instead of git diff (with --affected)")
    ap.add_argument("--timings", type=Path, default=None, help="--plan: read past timings from this file instead")
    ap.add_argument("--jobs", type=int, default=max(2, (os.cpu_count() or 4) // 2))
    ap.add_argument("--browser-jobs", type=int, default=2)
    ap.add_argument("--retry", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=1800, help="seconds per test file")
    ap.add_argument("--no-scan", action="store_true")
    ap.add_argument("--agent-env", action="store_true",
                    help="run children with CRYOEV_VIA=claude CRYOEV_FOR=pujan (default: both removed)")
    ap.add_argument("--lock-wait", type=float, default=LOCK_WAIT_MIN, help="minutes to wait for another gate")
    a = ap.parse_args(argv)
    if a.times:
        return show_times(25)

    files = None
    if a.full:
        suites, mode = set(SUITES), "full"
    elif a.suites:
        suites = {s.strip() for s in a.suites.split(",") if s.strip()}
        bad = suites - set(SUITES)
        if bad:
            print(f"unknown suite(s): {', '.join(sorted(bad))}", file=sys.stderr)
            return 2
        mode = "suites"
    else:
        files = [f.strip() for f in a.files.split(",") if f.strip()] if a.files else changed_files(a.base)
        suites, unknown = affected_suites(files)
        mode = "affected"
        print(f"{len(files)} changed file(s) vs {a.base}" + (f"; unmatched code files -> full run: {unknown[:5]}"
                                                            if unknown else ""))
    if a.plan:
        return plan(a, suites, mode, files)
    lk = GateLock(wait_min=a.lock_wait)
    if not lk.acquire():
        print(f"gate busy: another gate held the lock for {a.lock_wait:.0f} min ({lk.owner()})", file=sys.stderr)
        return 1
    try:
        return run_gate(a, suites, mode, files)
    finally:
        lk.release()


def run_gate(a, suites: set[str], mode: str, files: list | None = None) -> int:
    units, skip, order = select(suites, mode, files)
    if skip:
        print(skip_note(units, skip))
    print(f"gate ({mode}): suites {', '.join(sorted(suites))}; {len(units)} test files; "
          f"{a.jobs} at once ({a.browser_jobs} browser); child env: "
          + ("CRYOEV_VIA=claude CRYOEV_FOR=pujan (--agent-env)" if a.agent_env else "CRYOEV_VIA/CRYOEV_FOR removed"))

    browser_lane = threading.Semaphore(a.browser_jobs)
    lock = threading.Lock()
    t0 = time.time()

    def job(u):
        suite, name, browser = u
        sk = f"{suite}/{name}" in skip
        if browser:
            with browser_lane:
                return run_unit(suite, name, a.timeout, a.agent_env, sk)
        return run_unit(suite, name, a.timeout, a.agent_env, sk)

    results = []
    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        futs = [ex.submit(job, u) for u in units]
        for f in as_completed(futs):
            r = f.result()
            results.append(r)
            with lock:
                print(f"  {'ok  ' if r['ok'] else 'FAIL'} {r['seconds']:7.1f}s  {r['unit']}  {r['summary']}",
                      flush=True)

    failed = [r for r in results if not r["ok"]]
    for _ in range(a.retry):
        if not failed:
            break
        print(f"retrying {len(failed)} failed file(s) one at a time")
        still = []
        for r in failed:
            r2 = run_unit(r["suite"], r["file"], a.timeout, a.agent_env, r.get("skip_browser", False))
            r["retries"] = r.get("retries", 0) + 1
            r.setdefault("first_fail_tail", r["output"][-600:])
            if r2["ok"]:
                r.update(ok=True, flaky=True, retry_seconds=r2["seconds"])
                print(f"  FLAKY (passed alone) {r['unit']}")
            else:
                r["output"] = r2["output"]
                still.append(r)
        failed = still
    record(results, mode, a.agent_env)

    wall = time.time() - t0
    serial = sum(r["seconds"] for r in results)
    ntests = sum(r["tests"] for r in results)
    flaky = [r["unit"] for r in results if r.get("flaky")]
    print(f"\n{ntests} tests in {len(results)} files: {wall / 60:.1f} min wall "
          f"({serial / 60:.1f} min if run one after another)")
    if flaky:
        print("FLAKY (failed in parallel, passed alone): " + ", ".join(flaky))
    for r in failed:
        print(f"\n===== FAILED {r['unit']} =====\n" + r["output"][-4000:])

    scan_ok = True
    if not a.no_scan:
        p = subprocess.run([sys.executable, "tools/prepush_scan.py", "--range", f"{a.base}..HEAD"],
                           cwd=REPO, capture_output=True, text=True, env=child_env(a.agent_env))
        scan_ok = p.returncode == 0
        print("prepush_scan: " + (p.stdout.strip().splitlines()[-1] if p.stdout.strip() else f"rc {p.returncode}"))
    ok = not failed and scan_ok
    print("GATE " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
