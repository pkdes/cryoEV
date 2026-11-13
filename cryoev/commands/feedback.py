"""feedback list | status | reply: the portal's feedback reports (portal/feedback_store.py).

The file: --file, else $CJ_FEEDBACK_FILE, else the CJ_FEEDBACK_FILE that the services config gives the portals (the
same one file both live portals use). `list` is feedback_store's own read-only CLI. `status` / `reply` are writes:
a dry run unless --do; with --do the file is first copied to ~/.cryoev/backups/feedback/<UTC time>.jsonl, then ONE
event is appended through feedback_store.set_status / set_reply with who (--who / $CRYOEV_FOR) and via
($CRYOEV_VIA); the action log gets its two lines (the reply text is never logged there, only its length).
"""
from __future__ import annotations

import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from cryoev import common as CM


def add(sub):
    p = sub.add_parser("feedback", help="the portal's feedback reports: list (read), status / reply (write)")
    s = p.add_subparsers(dest="fcmd", required=True)
    ls = s.add_parser("list", help="every report (feedback_store's read-only CLI)")
    ls.add_argument("--file", type=Path)
    ls.add_argument("--json", action="store_true")
    ls.set_defaults(func=list_)
    st = s.add_parser("status", help='set a report\'s status: New, Planned, "In progress", Done, "Won\'t do"')
    st.add_argument("id", type=int)
    st.add_argument("status")
    st.add_argument("--file", type=Path)
    CM.write_flags(st)
    st.set_defaults(func=status)
    rp = s.add_parser("reply", help="reply to a report (shown to its author)")
    rp.add_argument("id", type=int)
    rp.add_argument("text")
    rp.add_argument("--file", type=Path)
    CM.write_flags(rp)
    rp.set_defaults(func=reply)


def feedback_file(flag=None) -> Path | None:
    if flag:
        return Path(flag)
    env = os.environ.get("CJ_FEEDBACK_FILE", "").strip()
    if env:
        return Path(env)
    S = CM.services()
    try:
        cfg = S.load_config(CM.services_config_path())
    except S.ConfigError:
        return None
    for svc in cfg.services:
        v = (svc.env or {}).get("CJ_FEEDBACK_FILE")
        if v and Path(v).is_absolute():
            return Path(v)
    return None


def _file(a) -> Path:
    f = feedback_file(a.file)
    if f is None:
        raise CM.CliError("no feedback file: pass --file, or set CJ_FEEDBACK_FILE")
    return f


def _store():
    from portal import feedback_store as FS
    return FS


def list_(a) -> int:
    rc, _ = CM.call_main(_store().main, ["list", "--file", str(_file(a))] + (["--json"] if a.json else []))
    return rc


def backup_dir() -> Path:
    return Path.home() / ".cryoev" / "backups" / "feedback"


def backup(path: Path) -> Path | None:
    """Copy the feedback file to backup_dir()/<UTC time>.jsonl (a new name, never over another backup)."""
    if not path.exists():
        return None
    d = backup_dir()
    d.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dst, n = d / f"{stamp}.jsonl", 1
    while dst.exists():
        n += 1
        dst = d / f"{stamp}_{n}.jsonl"
    shutil.copy2(path, dst)
    if CM.sha256(dst) != CM.sha256(path):
        raise CM.CliError(f"backup {dst} differs from {path}: not written")
    return dst


def _report(FS, path, rid) -> dict:
    rep = FS.read_all(path).get(rid)
    if rep is None:
        raise CM.CliError(f"no report #{rid} in {path}")
    return rep


def _write(a, action, args, describe, do):
    FS = _store()
    path = _file(a)

    def plan():
        rep = _report(FS, path, a.id)
        print(f"#{a.id} [{rep['kind']}] now {rep['status']}: {rep['text'][:120]}")
        print(describe(rep))
        print(f"file: {path}; a backup copy goes to {backup_dir()} first")
        return 0

    def act(who, via):
        _report(FS, path, a.id)
        b = backup(path)
        ev = do(FS, path, who, via)
        print(f"appended to {path}: {ev['type']} #{ev['id']} by {CM.label(who, via)}" + (f"; backup {b}" if b else ""))
        return 0, {"event": ev["type"], "backup": b.name if b else None}

    return CM.run_write(a, action, f"feedback #{a.id}", args, plan, act)


def status(a) -> int:
    FS = _store()
    match = {s.lower(): s for s in FS.STATUSES}
    new = match.get(a.status.strip().lower().replace("’", "'"))
    if new is None:
        raise CM.CliError(f"unknown status {a.status!r}: one of {', '.join(FS.STATUSES)}")
    return _write(a, "feedback status", {"id": a.id, "status": new},
                  lambda rep: f"would append: status #{a.id} -> {new}",
                  lambda FS, path, who, via: FS.set_status(path, a.id, who, new, via=via))


def reply(a) -> int:
    FS = _store()
    text = FS.norm(a.text)
    if len(text) > FS.REPLY_MAX:
        raise CM.CliError(f"the reply is {len(text)} characters; at most {FS.REPLY_MAX}")
    return _write(a, "feedback reply", {"id": a.id, "chars": len(text)},
                  lambda rep: f"would append: reply #{a.id} ({len(text)} characters): {text}",
                  lambda FS, path, who, via: FS.set_reply(path, a.id, who, text, via=via))
