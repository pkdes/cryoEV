"""Analysis display names (BUILD_CONTRACTS section 12). Pure and dependency-light: Results and My reviews import
`display_name` without loading the Predict module.

    display_name(req)  ->  req["name"] when it is a non-empty string, else "<source label> · <date>"
                           (e.g. "Uploaded images · 6 Nov 2026, 14:03", "Session 20261028 (EV prep) · ...").

The source label is the text a person already sees ("Uploaded images", "Session <id>", plus the configured session
label when the caller knows it). It is NEVER built from request.json source.dir (an absolute server path).
Every caller still escapes the result for HTML: a name is free text.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


def when(s) -> str:
    """'6 Nov 2026, 14:03' from an ISO time (local time for an aware one); '?' when it can't be read."""
    try:
        dt = datetime.fromisoformat(str(s))
    except (TypeError, ValueError):
        return "?"
    if dt.tzinfo is not None:
        dt = dt.astimezone()
    return f"{dt.day} {dt.strftime('%b %Y, %H:%M')}"


def source_label(req: dict, label=None) -> str:
    """'Uploaded images' | 'Session <id>' (+ ' (<label>)' when the caller passes the session's configured label)."""
    src = (req or {}).get("source") or {}
    if src.get("kind") == "upload":
        return "Uploaded images"
    return f"Session {src.get('session') or '?'}" + (f" ({label})" if label else "")


def default_name(req: dict, label=None) -> str:
    return f"{source_label(req, label)} · {when((req or {}).get('created'))}"


def display_name(req: dict, label=None) -> str:
    name = (req or {}).get("name")
    if isinstance(name, str) and name.strip():
        return name
    return default_name(req, label)


PEOPLE = Path(__file__).resolve().parent.parent / "annotation" / "people.json"
_people: dict = {}


def person_name(pid, path=PEOPLE) -> str:
    """A portal / people id -> its `name` in annotation/people.json (re-read when the file changes), else the id
    itself ('?' for none). Callers escape it."""
    if not isinstance(pid, str) or not pid:
        return "?"
    try:
        stamp = Path(path).stat().st_mtime_ns
        if _people.get("stamp") != (str(path), stamp):
            doc = json.loads(Path(path).read_text(encoding="utf-8"))
            names = {p["id"]: p["name"] for p in doc.get("people", [])
                     if isinstance(p, dict) and isinstance(p.get("id"), str) and isinstance(p.get("name"), str)}
            _people.clear()
            _people.update(stamp=(str(path), stamp), names=names)
    except (OSError, ValueError, AttributeError):
        return pid
    return _people.get("names", {}).get(pid) or pid
