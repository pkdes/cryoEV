"""Run the repo's test suites in parallel, per test file, and record how long each file takes.

  python tools/test_gate.py --affected [--base origin/main]   # only the suites the changed files touch (default)
  python tools/test_gate.py --full                            # every suite (big go-lives)
  python tools/test_gate.py --suites portal,results           # named suites
  python tools/test_gate.py --times                           # slowest test files from past runs, then exit

Each test file is one unit: `python -m unittest discover -s <suite dir> -p <file>`, the same import setup as the
classic `discover -s <suite dir>` gate. Units run in parallel (--jobs, default CPU count // 2), slowest first (from
past timings). Files that drive a real browser (Playwright) share a smaller lane (--browser-jobs, default 2), because
several browsers at once on a loaded laptop is what makes the known load-sensitive tests flaky.

A failed file is run once more on its own (--retry 1). One that then passes is reported as FLAKY, never as a plain
pass; the gate still exits 0 for it, so say so when reporting. Anything that fails twice fails the gate.
The pre-push scan (tools/prepush_scan.py --range <base>..HEAD) runs at the end unless --no-scan.

Timings are appended to ~/.cryoev/gate/timings.jsonl (one line per file per run) for the dev-time audit.
Exit 0 = every unit passed (flaky ones noted), 1 = a failure, 2 = usage error.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SUITES = {                       # name -> test dir (relative to the repo)
    "jobs": "jobs/tests", "tests": "tests", "registry": "registry/tests", "results": "results/tests",
    "tools": "tools/tests", "annotation": "annotation/tests", "data_utils": "data_utils/tests",
    "portal": "portal/tests",
}
# changed path prefix -> suites that exercise it (a file's own suite plus the suites that import it)
AFFECTS = [
    ("portal/", {"portal"}),
    ("annotation/", {"annotation", "portal", "results"}),
    ("results/", {"results", "portal"}),
    ("jobs/", {"jobs", "portal", "tools"}),
    ("training/", {"jobs", "portal"}),
    ("registry/", {"registry", "portal", "jobs"}),
    ("data_utils/", set(SUITES)),            # oslock etc. are used everywhere
    ("tools/", {"tools"}),
    ("tests/", {"tests"}),
    ("inference/", {"jobs", "portal"}),
    ("models/", {"jobs", "portal", "registry"}),
]
DOC_ONLY = re.compile(r"(\.md|\.txt|\.png|\.jpg|\.svg)$", re.I)
ALWAYS = {"tests"}                           # repo-wide checks, ~1 min
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
    for f in files:
        if f.startswith(".claude/"):
            continue
        hits = [s for p, s in AFFECTS if f.startswith(p)]
        if hits:
            for s in hits:
                out |= s
        elif not DOC_ONLY.search(f):
            unknown.append(f)
    if unknown:
        out = set(SUITES)
    return out, unknown


def is_browser_file(path: Path) -> bool:
    try:
        return "playwright" in path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False


def past_times() -> dict[str, float]:
    """Median-ish (last seen) seconds per unit, for slowest-first ordering."""
    t = {}
    try:
        for line in TIMINGS.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
                t[r["unit"]] = float(r["seconds"])
            except (ValueError, KeyError):
                continue
    except OSError:
        pass
    return t


def units_for(suites: set[str]) -> list[tuple[str, str, bool]]:
    """-> [(suite, test file name, is_browser)]"""
    out = []
    for s in sorted(suites):
        d = REPO / SUITES[s]
        for f in sorted(d.glob("test*.py")):
            out.append((s, f.name, is_browser_file(f)))
    return out


def run_unit(suite: str, name: str, timeout: int) -> dict:
    t0 = time.time()
    cmd = [sys.executable, "-m", "unittest", "discover", "-s", SUITES[suite], "-p", name]
    try:
        p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=timeout,
                           env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        text, rc = p.stdout + p.stderr, p.returncode
    except subprocess.TimeoutExpired as e:
        text, rc = f"TIMEOUT after {timeout}s\n" + ((e.stdout or "") if isinstance(e.stdout, str) else ""), 124
    m = SUMMARY.findall(text)
    ran = re.search(r"Ran (\d+) test", text)
    return {"unit": f"{suite}/{name}", "suite": suite, "file": name, "rc": rc, "ok": rc == 0,
            "seconds": round(time.time() - t0, 1), "tests": int(ran.group(1)) if ran else 0,
            "summary": " | ".join(m[-2:]) if m else "", "output": text}


def record(results: list[dict], mode: str) -> None:
    try:
        TIMINGS.parent.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        head = git("rev-parse", "--short", "HEAD").strip()
        with open(TIMINGS, "a", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps({"t": stamp, "head": head, "mode": mode, "unit": r["unit"],
                                    "seconds": r["seconds"], "tests": r["tests"], "ok": r["ok"],
                                    "flaky": r.get("flaky", False)}) + "\n")
    except (OSError, subprocess.CalledProcessError) as e:
        print(f"(timings not recorded: {e})", file=sys.stderr)


def show_times(n: int) -> int:
    t = past_times()
    if not t:
        print("No timings yet: run the gate once.")
        return 0
    total = sum(t.values())
    print(f"{len(t)} test files, {total / 60:.1f} min if run one after another; slowest {n}:")
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
    ap.add_argument("--jobs", type=int, default=max(2, (os.cpu_count() or 4) // 2))
    ap.add_argument("--browser-jobs", type=int, default=2)
    ap.add_argument("--retry", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=1800, help="seconds per test file")
    ap.add_argument("--no-scan", action="store_true")
    a = ap.parse_args(argv)
    if a.times:
        return show_times(25)

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
        files = changed_files(a.base)
        suites, unknown = affected_suites(files)
        mode = "affected"
        print(f"{len(files)} changed file(s) vs {a.base}" + (f"; unmatched code files -> full run: {unknown[:5]}"
                                                            if unknown else ""))
    units = units_for(suites)
    order = past_times()
    units.sort(key=lambda u: -order.get(f"{u[0]}/{u[1]}", 60.0))
    print(f"gate ({mode}): suites {', '.join(sorted(suites))}; {len(units)} test files; "
          f"{a.jobs} at once ({a.browser_jobs} browser)")

    browser_lane = threading.Semaphore(a.browser_jobs)
    lock = threading.Lock()
    t0 = time.time()

    def job(u):
        suite, name, browser = u
        if browser:
            with browser_lane:
                return run_unit(suite, name, a.timeout)
        return run_unit(suite, name, a.timeout)

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
            r2 = run_unit(r["suite"], r["file"], a.timeout)
            if r2["ok"]:
                r.update(ok=True, flaky=True, retry_seconds=r2["seconds"])
                print(f"  FLAKY (passed alone) {r['unit']}")
            else:
                r["output"] = r2["output"]
                still.append(r)
        failed = still
    record(results, mode)

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
                           cwd=REPO, capture_output=True, text=True)
        scan_ok = p.returncode == 0
        print("prepush_scan: " + (p.stdout.strip().splitlines()[-1] if p.stdout.strip() else f"rc {p.returncode}"))
    ok = not failed and scan_ok
    print("GATE " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
