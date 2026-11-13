"""
Group actions in the review tool (round 12, feedback #21): reject, undo or comment on a whole sample, or on one montage
within a sample. Served as POST <pkg>/api/group and GET <pkg>/api/groups by review_server.py (so by the multi server
too); docs/REVIEW_TOOL.md "Groups".

Groups
  sample   the images whose data.js `split` is the sample (prediction packages: the sample label, as the page's sample
           switcher shows it)
  montage  the images of one sample whose montage (annotation/review_meta.py: MANIFEST.csv, else the tile's name, else
           a source_manifest parent image) is that montage
  id       group_id(kind, sample[, montage]): a compact JSON array, e.g. ["sample","EV-013-2"]

Reject = the existing image-level reject on every image of the group, nothing new for the readers: each image gets an
i:reject decision {v: "reject:<reason>", who, t, server_t, grp: <group id>} through review_server.merge_edits, as if
that reviewer had rejected it in the image dialog (results/compute.py and apply_review_edits drop / list it as any
rejected image). Images already rejected are left as they are: an individual rejection (no grp) keeps its own reason,
and one rejected by another group keeps that group's. Rejecting a group again with another reason updates the images
it rejected.
Undo = only images whose i:reject still carries this group's id: each goes back to not rejected, unless another group
reject that is still on covers it (a montage inside a rejected sample, or the sample around a rejected montage): then
it is rejected by that group, with its reason. Images rejected individually keep their rejection; an image someone
changed after the group action (their own reject or un-reject) carries no grp any more and is left alone.
Comment = one note per group (text, who, t); an empty text removes it. Every action is also logged (who, when, what).

Concurrency: every action runs under the package's write lock (review_server: one writer process per package, one
write lock per package in it), the same lock every image save takes, so two reviewers' group actions, or a group
action and an image save, never interleave: all of a group's records are read, merged and written before the next
request is handled. Images open on another reviewer's screen (an active claim) are changed too: the reviewer's page
merges per decision key, so the reject reaches it on its next sync (<= 20 s) and a later decision of theirs on that
image wins as usual. The answer lists them (claimed_by_others) so the page can say so.
Crash safety: records are written one by one (each atomically), then _groups.json. Repeating the same action is safe
(a reject skips images already rejected by the group; an undo finds only images still carrying its id).

The groups file: <pkg>/edits/_groups.json (GROUPS_FILE). It is NOT an image record: every reader of edits/*.json goes
through is_record_file() / record_files() (or reads records by name), so counts, apply, progress and results never
see it. It is backed up with the edits (tools/backup_state.py copies review/*/edits/**/*.json).
    {"schema": "review_groups/1", "pkg": <package id>,
     "groups": {<group id>: {"kind", "sample", "montage"?, "note"? {text, who, t},
                             "reject"? {reason, note?, who, t, n_images, n_rejected}}},
     "log": [{t, who, action, group, reason?, n?}, ...]}        (the last LOG_MAX entries)
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

GROUPS_FILE = "_groups.json"
SCHEMA = "review_groups/1"
KINDS = ("sample", "montage")
ACTIONS = ("reject", "undo", "comment")
NOTE_MAX = 1000          # characters, as an image note (review_server.IMAGE_NOTE_MAX)
NAME_MAX = 200           # a sample or montage name
GRP_MAX = 500            # a group id on a decision (review_server validates the grp field with it)
LOG_MAX = 2000
BODY_KEYS = frozenset({"action", "kind", "sample", "montage", "reason", "note"})


# ---------------------------------------------------------------- the edits folder: records vs the groups file
def is_record_file(p) -> bool:
    """An image's edit record (edits/<file>.json), not the groups file."""
    name = Path(p).name
    return name.endswith(".json") and name != GROUPS_FILE


def record_files(edits_dir) -> list:
    """edits/*.json image records, sorted by name, the groups file left out."""
    d = Path(edits_dir)
    return sorted(f for f in d.glob("*.json") if is_record_file(f)) if d.is_dir() else []


# ---------------------------------------------------------------- ids and membership
def group_id(kind: str, sample: str, montage: str | None = None) -> str:
    parts = [kind, sample] + ([montage] if kind == "montage" else [])
    return json.dumps(parts, ensure_ascii=False, separators=(",", ":"))


def parse_group_id(gid):
    """(kind, sample, montage or None), or None for anything else."""
    try:
        v = json.loads(gid) if isinstance(gid, str) else None
    except ValueError:
        return None
    if (isinstance(v, list) and all(isinstance(x, str) for x in v) and len(v) in (2, 3) and v[0] in KINDS
            and len(v) == (3 if v[0] == "montage" else 2)):
        return v[0], v[1], (v[2] if len(v) == 3 else None)
    return None


def members(images: dict, kind: str, sample: str, montage: str | None = None) -> list:
    """The data.js files of a group, in package order. images: review_meta.package_meta()["images"]."""
    return [f for f, m in images.items() if m.get("split") == sample
            and (kind == "sample" or (m.get("montage") is not None and m.get("montage") == montage))]


def groups_of(images: dict, file: str) -> list:
    """The ids of the groups an image belongs to: its sample, and its montage when it has one."""
    m = images.get(file) or {}
    out = [group_id("sample", m.get("split", ""))]
    if m.get("montage"):
        out.append(group_id("montage", m.get("split", ""), m["montage"]))
    return out


# ---------------------------------------------------------------- the groups file
class GroupsUnreadable(Exception):
    """_groups.json exists but can't be read: nothing is written over it."""


def empty(pkg_id: str = "") -> dict:
    return {"schema": SCHEMA, "pkg": pkg_id, "groups": {}, "log": []}


def load(edits_dir, pkg_id: str = "") -> dict:
    p = Path(edits_dir) / GROUPS_FILE
    if not p.exists():
        return empty(pkg_id)
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise GroupsUnreadable(f"{GROUPS_FILE} is unreadable ({type(e).__name__})") from e
    if not isinstance(doc, dict) or not isinstance(doc.get("groups"), dict):
        raise GroupsUnreadable(f"{GROUPS_FILE} is not a groups file")
    doc.setdefault("log", [])
    return doc


def save(edits_dir, doc: dict):
    p = Path(edits_dir) / GROUPS_FILE
    doc["log"] = doc.get("log", [])[-LOG_MAX:]
    tmp = p.with_name(GROUPS_FILE + ".tmp")
    tmp.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, p)


def public(doc: dict, log_tail: int = 50) -> dict:
    """What GET /api/groups answers: the groups and the last log entries."""
    return {"schema": SCHEMA, "groups": doc.get("groups", {}), "log": doc.get("log", [])[-log_tail:]}


# ---------------------------------------------------------------- the request
def check_body(body, reasons) -> str | None:
    """Why a POST /api/group body is refused, else None."""
    if not isinstance(body, dict) or not set(body) <= BODY_KEYS:
        return "bad body"
    if body.get("action") not in ACTIONS or body.get("kind") not in KINDS:
        return "bad action or kind"
    for k in ("sample", "montage"):
        v = body.get(k)
        if v is not None and (not isinstance(v, str) or not v or len(v) > NAME_MAX or not _utf8(v)):
            return f"bad {k}"
    if not isinstance(body.get("sample"), str):
        return "sample required"
    if (body["kind"] == "montage") != isinstance(body.get("montage"), str):
        return "montage required for a montage group (only there)"
    if body["action"] == "reject" and body.get("reason") not in reasons:
        return "reason required (one of the reject reasons)"
    if body["action"] != "reject" and body.get("reason") is not None:
        return "reason only with reject"
    note = body.get("note")
    if note is not None and (not isinstance(note, str) or len(note) > NOTE_MAX or not _utf8(note)):
        return "bad note"
    if body["action"] == "undo" and note is not None:
        return "no note with undo"
    if len(group_id(body["kind"], body["sample"], body.get("montage"))) > GRP_MAX:
        return "group name too long"
    return None


def _utf8(s) -> bool:
    try:
        s.encode("utf-8")
        return True
    except UnicodeEncodeError:
        return False


# ---------------------------------------------------------------- the action (caller holds the package write lock)
def act(rs, edits_dir: Path, pkg_id: str, images: dict, body: dict, who: str, now_ms: int | None = None,
        claims: dict | None = None) -> tuple:
    """Run one checked group action -> (http status, answer). rs: the review_server module (merge_edits, _migrate,
    stamp_received, RECORD_FMT, REJECT_PREFIX); images: package_meta()["images"]; claims: {"<file>|<case>": who}
    active now. The caller holds the package's write lock for the whole call."""
    now = int(time.time() * 1000) if now_ms is None else int(now_ms)
    kind, sample, montage = body["kind"], body["sample"], body.get("montage")
    gid = group_id(kind, sample, montage)
    files = members(images, kind, sample, montage)
    try:
        doc = load(edits_dir, pkg_id)
    except GroupsUnreadable as e:
        return 409, {"error": str(e) + ": fix or move it first; nothing was changed"}
    groups = doc["groups"]
    action = body["action"]
    if not files and not (action == "undo" and gid in groups):
        return 404, {"error": "no images in this group"}
    entry = groups.get(gid) or {"kind": kind, "sample": sample, **({"montage": montage} if montage else {})}
    log = {"t": now, "who": who, "action": action, "group": gid}
    changed, counts = {}, {}

    if action == "comment":
        text = (body.get("note") or "").strip()
        if text:
            entry["note"] = {"text": text, "who": who, "t": now}
        else:
            entry.pop("note", None)
        log["n"] = len(files)
    else:
        try:
            records = _read_records(edits_dir, images if action == "undo" else files)
        except (OSError, ValueError) as e:  # nothing is written: a half-applied group is what the lock prevents
            return 409, {"error": f"an edit record can't be read ({type(e).__name__}): nothing was changed"}
        if action == "reject":
            reason = body["reason"]
            for f in files:
                d = _reject_dec(rs, records[f])
                if _is_rejected(rs, d):
                    if d.get("grp") == gid:
                        if d.get("v") == rs.REJECT_PREFIX + reason:
                            counts["already"] = counts.get("already", 0) + 1
                            continue
                    else:
                        k = "kept_other_group" if d.get("grp") else "kept_individual"
                        counts[k] = counts.get(k, 0) + 1
                        continue
                changed[f] = _set_reject(rs, records[f], rs.REJECT_PREFIX + reason, gid, who, now)
            entry["reject"] = {"reason": reason, "who": who, "t": now, "n_images": len(files),
                               "n_rejected": sum(1 for f in files if f in changed or _grp_of(rs, records[f]) == gid)}
            detail = (body.get("note") or "").strip()
            if detail:  # the reason in the reviewer's words, kept with the group (the images carry the reason only)
                entry["reject"]["note"] = detail
            log.update(reason=reason, n=len(changed))
        else:  # undo
            entry.pop("reject", None)
            groups[gid] = entry  # off before the fallback lookup below
            for f, rec in records.items():
                if _grp_of(rs, rec) != gid:
                    continue
                fb = _fallback(groups, images, f, gid)
                if fb:
                    changed[f] = _set_reject(rs, rec, rs.REJECT_PREFIX + groups[fb]["reject"]["reason"], fb, who, now)
                    counts["kept_other_group"] = counts.get("kept_other_group", 0) + 1
                else:
                    changed[f] = _set_reject(rs, rec, None, None, who, now)
            log["n"] = len(changed)
        for f, merged in changed.items():   # every record is computed before the first write
            _write_record(rs, edits_dir, pkg_id, f, records[f], merged, now)
    if entry.get("note") or entry.get("reject"):
        groups[gid] = entry
    else:
        groups.pop(gid, None)
    doc["log"].append(log)
    doc["pkg"] = doc.get("pkg") or pkg_id
    save(edits_dir, doc)
    others = sorted({(f, w) for k, w in (claims or {}).items() for f in [k.split("|", 1)[0]]
                     if f in changed and w != who})
    return 200, {"ok": True, "action": action, "group": gid, "n_images": len(files), "n_changed": len(changed),
                 **counts, "claimed_by_others": [{"file": f, "who": w} for f, w in others],
                 "edits": {f: m for f, m in changed.items()}, **public(doc)}


def _read_records(edits_dir: Path, files) -> dict:
    out = {}
    for f in files:
        p = Path(edits_dir) / f"{f}.json"
        rec = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        if not isinstance(rec, dict) or not isinstance(rec.get("edit", {}), dict):
            raise ValueError(f"{f}: not an edit record")
        out[f] = rec
    return out


def _reject_dec(rs, saved: dict):
    cur = saved.get("edit")
    return rs._migrate(cur)["dec"].get("i:reject") if cur else None


def _is_rejected(rs, d) -> bool:
    return bool(d) and isinstance(d.get("v"), str) and d["v"].startswith(rs.REJECT_PREFIX)


def _grp_of(rs, saved: dict):
    d = _reject_dec(rs, saved)
    return d.get("grp") if _is_rejected(rs, d) else None


def _fallback(groups: dict, images: dict, file: str, gone: str):
    """Another group reject still on that covers this image: its montage group first, then its sample."""
    for g in reversed(groups_of(images, file)):
        if g != gone and (groups.get(g) or {}).get("reject"):
            return g
    return None


def _set_reject(rs, saved: dict, v, gid, who, now) -> dict:
    """The record with a new i:reject decision, merged as a PUT from this reviewer would be (pending -> server_t)."""
    cur = saved.get("edit")
    prev = _reject_dec(rs, saved)
    t = max(now, int(prev.get("t", 0) or 0) + 1) if prev else now
    d = {"v": v, "t": t, "who": who, "pending": True}
    if v and gid:
        d["grp"] = gid
    return rs.merge_edits(cur, {"dec": {"i:reject": d}, "t": t}, now)


def _write_record(rs, edits_dir: Path, pkg_id: str, file: str, saved: dict, merged: dict, now: int):
    target = Path(edits_dir) / f"{file}.json"
    rx = rs.stamp_received(saved.get("rx") or {}, saved.get("edit"), merged, now)
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps({"file": file, "pkg": pkg_id, "fmt": rs.RECORD_FMT, "edit": merged, "rx": rx}),
                   encoding="utf-8")
    os.replace(tmp, target)
