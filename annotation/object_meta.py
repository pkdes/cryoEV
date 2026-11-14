"""Object review sets (round 14, contracts C3 + C4): reading an object set's attribute decisions, for display (the
portal's progress and listings) and for export (annotation/object_set.py).

New module on purpose: annotation/review_meta.py and review_groups.py are frozen (the live :8766 review imports them
lazily at request time), so object-set display code lives here. Read only; nothing here writes.

  is_object_set(pkg)          data.js kind == "objects"
  public_items(pkg)           data.js D.items: [{id, image, obj, crop[, also]}] (never items.json: blind columns)
  decisions(pkg)              {item id: {"<attr>@<rid>": decision}} from edits/*.json (a:<obj>:<attr>@<rid> keys;
                              the record files only: edits/_groups.json and edits/_attr_log.jsonl are skipped)
  history(pkg)                edits/_attr_log.jsonl rows ({t, who, key, v, file}); a cut-short line is skipped
  progress(pkg, reviewers)    {n_items, n_done, by_reviewer {rid: n items with every required attribute set}}:
                              an item is done when every reviewer (the package's reviewers, else everyone who
                              decided anything) has set every required attribute (C4 "Progress")
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATAJS = "data.js"
ATTR_LOG = "_attr_log.jsonl"         # = review_server.ATTR_LOG
GROUPS_FILE = "_groups.json"         # = review_groups.GROUPS_FILE
RECORD = "review_package.json"


def _rs():
    import sys
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from annotation import review_server
    return review_server


def read_datajs(pkg) -> dict:
    return json.loads((Path(pkg) / DATAJS).read_text(encoding="utf-8").split("=", 1)[1].strip().rstrip(";"))


def is_object_set(pkg) -> bool:
    try:
        return read_datajs(pkg).get("kind") == "objects"
    except (OSError, ValueError, IndexError):
        return False


def public_items(pkg) -> list:
    return list(read_datajs(pkg).get("items") or [])


def record_files(edits_dir) -> list:
    """edits/*.json image records (not _groups.json; _attr_log.jsonl is no *.json)."""
    d = Path(edits_dir)
    return sorted(f for f in d.glob("*.json") if f.name != GROUPS_FILE) if d.is_dir() else []


def _stem(file: str) -> str:
    return Path(file).stem


def decisions(pkg, data=None) -> dict:
    """{item id: {"<attr>@<rid>": {v, t, who, server_t}}} for every item of the set (empty dict = undecided).
    An undone decision (v null) is kept: it says the reviewer cleared it."""
    rs = _rs()
    data = data or read_datajs(pkg)
    by_obj = {(it["image"], it["obj"]): it["id"] for it in data.get("items") or []}
    out = {it["id"]: {} for it in data.get("items") or []}
    for f in record_files(Path(pkg) / "edits"):
        try:
            rec = json.loads(f.read_text(encoding="utf-8"))
            file, dec = rec["file"], (rec.get("edit") or {}).get("dec") or {}
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            continue
        for k, d in dec.items():
            p = rs.parse_attr_key(k)
            if p is None or p[0] is None or not isinstance(d, dict):
                continue
            iid = by_obj.get((_stem(file), p[0]))
            if iid is not None:
                out[iid][f"{p[1]}@{p[2]}"] = d
    return out


def history(pkg) -> list:
    p = Path(pkg) / "edits" / ATTR_LOG
    rows = []
    if not p.is_file():
        return rows
    for ln in p.read_text(encoding="utf-8").splitlines():
        try:
            o = json.loads(ln)
        except ValueError:
            continue
        if isinstance(o, dict):
            rows.append(o)
    return rows


def package_reviewers(pkg) -> list:
    try:
        rec = json.loads((Path(pkg) / RECORD).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    revs = rec.get("reviewers") if isinstance(rec, dict) else None
    return [r for r in revs if isinstance(r, str)] if isinstance(revs, list) else []


def required_attrs(schema: dict) -> list:
    return [a["id"] for a in schema.get("object_attrs") or [] if a.get("required")]


def progress(pkg, reviewers=None) -> dict:
    """C4 progress: items on which every reviewer has set every required attribute (a value, not null)."""
    data = read_datajs(pkg)
    req = required_attrs(data.get("schema") or {})
    dec = decisions(pkg, data)
    revs = list(reviewers if reviewers is not None else package_reviewers(pkg))
    if not revs:
        revs = sorted({k.split("@", 1)[1] for d in dec.values() for k in d})

    def done_by(item, rid):
        return all((dec[item].get(f"{a}@{rid}") or {}).get("v") is not None for a in req)

    items = list(dec)
    by = {r: sum(1 for i in items if done_by(i, r)) for r in revs}
    n_done = sum(1 for i in items if revs and all(done_by(i, r) for r in revs))
    return {"n_items": len(items), "n_done": n_done, "reviewers": revs, "by_reviewer": by}
