"""
Feedback store (round 7, F): bug reports, feature ideas and questions from portal users.

One append-only JSON-lines file; the current state of every report is the fold of its events:
    {"type": "create", "id": 12, "who": portal id, "t": server UTC, "kind": "bug"|"feature"|"question",
     "text": <=4000, "steps": <=4000, "page": "/path" (no query, no fragment), "status": "New"}
    {"type": "status", "id": 12, "who": admin id, "t": ..., "status": "Planned"}
    {"type": "reply",  "id": 12, "who": admin id, "t": ..., "reply": <=2000}
Any event may carry "via": the agent that acted for `who` (registry/actor.py, round 9: e.g. "claude" for the
`python -m cryoev feedback status|reply` CLI). The portal never passes it; fold() ignores it.

Location: env CJ_FEEDBACK_FILE (an absolute path) else <state dir>/feedback.jsonl. The live portals have two state
dirs (:8770 key mode and :8448 access mode); the live setup sets CJ_FEEDBACK_FILE for both services to ONE file, so
both see the same reports. Writers take a thread lock plus an OS lock (data_utils/oslock.py, as portal/audit.py);
the rate limit (20 creates per user per UTC day) is counted and the create appended under that same lock.

Read-only triage for agents:   python -m portal.feedback_store list --file <path> [--json]
Status / reply from an agent:   python -m cryoev feedback status|reply ... (dry run unless --do; docs/AGENTS.md)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from data_utils.oslock import acquire_lock  # noqa: E402

ENV = "CJ_FEEDBACK_FILE"
KINDS = ("bug", "feature", "question")
STATUSES = ("New", "Planned", "In progress", "Done", "Won't do")
TEXT_MAX, STEPS_MAX, REPLY_MAX, PAGE_MAX = 4000, 4000, 2000, 200
DAILY_MAX = 20
LOCK_WAIT_S = 5.0
VIA_RE = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")    # = registry/actor.py VIA_RE (an agent id, e.g. "claude")
_lock = threading.Lock()


class RateLimited(Exception):
    pass


class Invalid(ValueError):
    pass


def feedback_path(state_dir) -> Path:
    env = os.environ.get(ENV, "").strip()
    if env and Path(env).is_absolute():
        return Path(env)
    return Path(state_dir) / "feedback.jsonl"


def clean_page(p) -> str:
    """A path only: query and fragment dropped, must start with '/', at most PAGE_MAX; else ''."""
    if not isinstance(p, str):
        return ""
    p = p.strip().split("#", 1)[0].split("?", 1)[0]
    if not p.startswith("/") or p.startswith("//") or len(p) > PAGE_MAX or "\\" in p \
            or any(unicodedata.category(c) in ("Cc", "Cf") for c in p):
        return ""
    return p


def norm(x) -> str:
    """Line endings as \n (a browser form sends \r\n) so the limits count what is stored and shown."""
    return (x or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def _via(rec: dict, via) -> dict:
    """rec with "via" set when an agent acted (a short lower-case id, registry/actor.py VIA_RE), else unchanged."""
    if via:
        if not isinstance(via, str) or not VIA_RE.match(via):
            raise Invalid(f"via {via!r} is not a short lower-case id")
        rec["via"] = via
    return rec


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_lines(path: Path) -> list:
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return []
    out = []
    for ln in data.decode("utf-8", "replace").splitlines():
        try:
            rec = json.loads(ln)
        except ValueError:
            continue
        if isinstance(rec, dict):
            out.append(rec)
    return out


def fold(events) -> dict:
    """events -> {id: report}. Unknown or malformed events are ignored; status / reply events for an unknown id too."""
    out: dict = {}
    for e in events:
        t, i = e.get("type"), e.get("id")
        if not isinstance(i, int) or isinstance(i, bool):
            continue
        if t == "create":
            if i in out or e.get("kind") not in KINDS or not isinstance(e.get("who"), str) \
                    or not isinstance(e.get("text"), str):
                continue
            out[i] = {"id": i, "who": e["who"], "t": e.get("t"), "kind": e["kind"], "text": e["text"][:TEXT_MAX],
                      "steps": str(e.get("steps") or "")[:STEPS_MAX], "page": clean_page(e.get("page")),
                      "status": "New", "reply": "", "reply_t": None, "updated": e.get("t")}
        elif i in out and t == "status" and e.get("status") in STATUSES:
            out[i]["status"], out[i]["updated"] = e["status"], e.get("t")
        elif i in out and t == "reply" and isinstance(e.get("reply"), str):
            out[i]["reply"], out[i]["reply_t"], out[i]["updated"] = e["reply"][:REPLY_MAX], e.get("t"), e.get("t")
    return out


def read_all(path) -> dict:
    return fold(_read_lines(Path(path)))


def _append(path: Path, rec: dict, check=None):
    """Append under both locks. `check(events, rec)` runs under the lock and may raise (rate limit, unknown id);
    it returns the record to write (so the id is allocated under the lock)."""
    with _lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        lk = acquire_lock(path.with_name(path.name + ".lock"), wait=LOCK_WAIT_S)
        if lk is None:
            raise OSError("the feedback file is locked by another process")
        try:
            if check is not None:
                rec = check(_read_lines(path), rec) or rec
            lead = ""
            try:                                             # a torn last line (crash mid-write): start a new line
                with open(path, "rb") as rf:
                    if rf.seek(0, os.SEEK_END) > 0:
                        rf.seek(-1, os.SEEK_END)
                        if rf.read(1) != b"\n":
                            lead = "\n"
            except FileNotFoundError:
                pass
            with open(path, "a", encoding="utf-8", newline="\n") as f:
                f.write(lead + json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
        finally:
            lk.close()
    return rec


def create(path, who, kind, text, steps="", page="", via=None) -> dict:
    """-> the create event (with the new id). Raises Invalid, RateLimited (20 per user per UTC day), OSError.
    via: the agent acting for `who`, or None."""
    text, steps = norm(text), norm(steps)
    if kind not in KINDS:
        raise Invalid("Choose what kind of report this is.")
    if not text:
        raise Invalid("Write what happened or what you would like.")
    if len(text) > TEXT_MAX or len(steps) > STEPS_MAX:
        raise Invalid("That is too long.")
    rec = {"type": "create", "id": 0, "who": str(who), "t": _now(), "kind": kind, "text": text, "steps": steps,
           "page": clean_page(page), "status": "New"}
    _via(rec, via)

    def check(events, r):
        day = r["t"][:10]
        mine = sum(1 for e in events if e.get("type") == "create" and e.get("who") == r["who"]
                   and str(e.get("t", ""))[:10] == day)
        if mine >= DAILY_MAX:
            raise RateLimited(f"{DAILY_MAX} reports a day is the limit. Try again tomorrow.")
        ids = [e["id"] for e in events if e.get("type") == "create" and isinstance(e.get("id"), int)]
        r["id"] = max(ids, default=0) + 1
        return r
    return _append(Path(path), rec, check)


def _must_exist(events, r):
    if not any(e.get("type") == "create" and e.get("id") == r["id"] for e in events):
        raise Invalid("No such report.")
    return r


def set_status(path, rid: int, who, status: str, via=None) -> dict:
    if status not in STATUSES:
        raise Invalid("Unknown status.")
    return _append(Path(path), _via({"type": "status", "id": rid, "who": str(who), "t": _now(), "status": status},
                                    via), _must_exist)


def set_reply(path, rid: int, who, reply: str, via=None) -> dict:
    reply = norm(reply)
    if len(reply) > REPLY_MAX:
        raise Invalid("That reply is too long.")
    return _append(Path(path), _via({"type": "reply", "id": rid, "who": str(who), "t": _now(), "reply": reply}, via),
                   _must_exist)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m portal.feedback_store", description="Feedback store (read-only CLI)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ls = sub.add_parser("list", help="read-only: print every report")
    ls.add_argument("--file", type=Path, default=None, help=f"default ${ENV}")
    ls.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    f = a.file or (Path(os.environ[ENV]) if os.environ.get(ENV) else None)
    if f is None:
        ap.error(f"give --file or set {ENV}")
    reports = sorted(read_all(f).values(), key=lambda r: r["id"])
    if a.json:
        print(json.dumps(reports, ensure_ascii=False, indent=2))
    else:
        for r in reports:
            print(f"#{r['id']} [{r['kind']}] {r['status']} {r['t']} {r['who']} {r['page']}\n  {r['text']}")
            if r["steps"]:
                print(f"  steps: {r['steps']}")
            if r["reply"]:
                print(f"  reply: {r['reply']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
