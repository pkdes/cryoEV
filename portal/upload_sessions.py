"""
Upload sessions (round 7 S, BUILD_CONTRACTS.md section 13): a raw upload's conversion becomes the registry
`session/1` record of its imaging session, so Data -> Sessions shows it (the registry is the one store).

    register_conversion(conv_dir, cfg) -> dict      jobs side: CX's finalize calls it after the runner lock is released
    save_details(session_id, fields, who, cfg, admin) -> dict    the "Session details" form (conversion page, Data)
    enabled(cfg) -> bool                            THE switch (CX calls it too)
    python -m portal.upload_sessions unregister ... admin fix for a mistyped imaging date (below)

Registration
  - Only when enabled(cfg) (cfg.upload_sessions / cfg.cj_upload_sessions, else env CJ_UPLOAD_SESSIONS; on / 1 / yes /
    true) and only for a conversion whose convert.json says source.kind == "upload" and whose status.json says done.
    Nothing is written otherwise (no registry line, no file).
  - The session id is the conversion's session (the imaging date the uploader typed), so it WILL match a lab session:
    read the current line -> merge -> append, under <registry>/.upload_session.lock (registry/importers.py takes the
    same lock while it writes, and carries `conversions` and `details` forward). Lab fields are never replaced:
    status, local_folder, sample_labels, the acquisition summary and grid ids stay as the lab recorded them. A new
    record gets status "processed", its date, the sample labels and the acquisition summary of this upload. Where the
    lab labelled a grid, an upload's other name is shown as "Lab label (upload: X)" (effective_labels).
  - Upload-owned keys (UPLOAD_KEYS): `conversions` [{id, uploaded_by, grids, sample_labels, tiles_per_grid, n_tiles,
    n_montages, acquisition}] (one entry per conversion id; `owner` is NOT used: in the registry it names an owner
    file; `uploaded_via`, the agent that uploaded it for uploaded_by, from convert.json made_via, round 11 #65) and
    `details` (below).
  - The acquisition summary comes from <conv>/acquisition.json (written by jobs/convert_raw.py with the importer's
    file_summary / session_acquisition / scope_model): only the session/1 acquisition keys, each checked against the
    schema, never a raw header, serial or path. A missing file registers without it.
  - Idempotent: a merged record equal to the current one (logged / recorded_by aside) appends nothing.
  - A refusal (an excluded session, a bad record) is returned as {"ok": False, "error"} and written to
    <conv>/session.json. The lock is waited for LOCK_WAIT_S (2 s) only: then {"ok": False, "busy": True, "retry":
    True} and nothing is appended (save_details: "Registry busy, try again in a minute."). Never raises for those.
  - Known limit, a mistyped imaging date: the upload lands in that date's session (perhaps a lab one). Admin fix:
    re-upload under the right date, then `python -m portal.upload_sessions unregister --registry <dir> --session
    <wrong id> --conversion <cid> --who <id>` appends the line without it (a portal-made session left empty becomes
    "excluded"; append-only, nothing deleted).

Session details
  - grid ids (inventory Grid#, free text up to 40 characters), prep notes (up to 2000 characters) and a display
    sample name per grid (the labels baked into tile names and MANIFEST are not changed). Merge-append under the
    same lock; an audit line (portal/audit.py, no user text in it). Who may edit: the admin, and whoever uploaded
    one of the session's conversions (may_edit); the conversion page also lets its owner (can_see) edit. A display
    name: the admin on any grid, an uploader only on the grids of their own conversions. Grid ids and names lose
    control characters and a leading = + - @.

The upload form asks for the sample per grid BEFORE conversion (form_samples / sample_key_text): the labels are baked
into tile names and MANIFEST.csv, so they are written as input/sample_key.csv in the lab's `#,Label` format (what
cryoem_core.metadata.load_sample_key and jobs/convert_raw.py --sample-key read), printable ASCII only (the lab
parser opens the file in the system code page).

All text that reaches a page goes through html.escape here.
"""
from __future__ import annotations

import contextlib
import csv
import html
import io
import json
import os
import re
import sys
import threading
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from jobs import predict_set as PS  # noqa: E402

FLAG = "CJ_UPLOAD_SESSIONS"
LOCK = ".upload_session.lock"            # in the registry folder; registry/importers.py holds it while it writes
UPLOAD_KEYS = ("conversions", "details")  # = registry/importers.py UPLOAD_KEYS (a test ties them)
SESSION_JSON = "session.json"            # <conv>/session.json: this conversion's registration outcome
SCHEMA_LOCAL = "cj_conversion_session/1"
RECORDED_BY = "portal/upload_sessions.py"
ACQ_KEYS = ("instrument", "magnification", "pixel_size_a", "defocus_um", "target_defocus_um", "dose", "voltage_kv",
            "binning", "acq_dates", "n_mdoc", "n_imaging", "n_other", "other_magnification")
LAB_KEYS = ("status", "local_folder", "sample_labels", "session_grids", "date", "date_basis") + ACQ_KEYS
CID_RE = re.compile(r"^(\d{8}(?:-\d+)?)_[0-9a-f]{6}$")
SESSION_RE = re.compile(r"^\d{8}(-\d+)?$")
SLOT_RE = re.compile(r"^\d{1,3}$")
FORM_SLOTS = 12                          # the upload form's grid rows (an autoloader cassette holds 12)
SAMPLE_MAX = 60                          # a sample name typed in the upload form
DISPLAY_MAX = 80                         # a display sample name (details)
GRID_ID_MAX = 40
NOTES_MAX = 2000
MAX_GRIDS = 48                           # grid rows one details form may carry
LABEL_MAX = 200                          # a label read back from MANIFEST.csv (an uploaded key may be anything)
FORMULA_START = ("=", "+", "-", "@")
MANIFEST = "MANIFEST.csv"
FLAG_ON = ("on", "1", "yes", "true")
LOCK_WAIT_S = 2.0                        # the upload-session lock: a short wait, then "busy" (the caller retries)
SCOPE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9 ._-]{0,39}$")   # acquisition.json scope_model: a model name only
BUSY_TEXT = "Registry busy, try again in a minute."
_THREAD_LOCKS: dict = {}
_THREAD_LOCKS_LOCK = threading.Lock()

esc = html.escape


class DetailsError(ValueError):
    """The details form was refused: a message for the person (plain words)."""


# ---------------------------------------------------------------- settings
def enabled(cfg) -> bool:
    """THE switch (CX's finalize / kick call this too): cfg.upload_sessions or cfg.cj_upload_sessions when set,
    else env CJ_UPLOAD_SESSIONS; on / 1 / yes / true (any case) = on, anything else = off."""
    v = None
    for name in ("upload_sessions", "cj_upload_sessions"):
        v = getattr(cfg, name, None)
        if v not in (None, ""):
            break
    if v in (None, ""):
        v = os.environ.get(FLAG, "")
    return str(v).strip().lower() in FLAG_ON


class Busy(Exception):
    """The upload-session lock (or the registry's own) could not be had in time."""


@contextlib.contextmanager
def upload_lock(reg: Path, wait: float = None):
    """<registry>/.upload_session.lock for read -> merge -> append: a per-process thread lock, then the OS lock,
    each waited for at most LOCK_WAIT_S (never the 30 s of a registry write). Raises Busy on timeout. The registry
    folder goes through the registry's path guard before the lock file is made."""
    wait = LOCK_WAIT_S if wait is None else wait
    regm = _regm()
    regm._guard(reg)
    reg.mkdir(parents=True, exist_ok=True)
    key = os.path.normcase(os.path.abspath(reg))
    with _THREAD_LOCKS_LOCK:
        tl = _THREAD_LOCKS.setdefault(key, threading.Lock())
    if not tl.acquire(timeout=wait):
        raise Busy("another request is writing this registry")
    try:
        f = regm.acquire_lock(reg / LOCK, wait)
        if f is None:
            raise Busy(f"{LOCK} is held by another process")
        try:
            yield
        finally:
            f.close()
    finally:
        tl.release()


def registry_root(cfg):
    r = getattr(cfg, "registry", None)
    return Path(r) if r else None


def _regm():
    return PS.registry_module()


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _read_json(p: Path):
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _dict(x) -> dict:
    return x if isinstance(x, dict) else {}


def _list(x) -> list:
    return x if isinstance(x, list) else []


# ---------------------------------------------------------------- the upload form: sample per grid
def sample_problem(label: str) -> str | None:
    if len(label) > SAMPLE_MAX:
        return f"at most {SAMPLE_MAX} characters"
    if any(not (32 <= ord(c) < 127) for c in label):
        return "plain letters, digits, spaces and punctuation only (no accents or symbols such as µ or Å)"
    if label.startswith(FORMULA_START):
        return "must not start with = + - or @"
    return None


def form_samples(fields: dict) -> tuple:
    """The upload form's `sample_<n>` fields -> ({grid number: label}, [problems]). Blank rows are left out."""
    out, problems = {}, []
    rows = [(re.match(r"^sample_(\d{1,3})$", str(k)), v) for k, v in (fields or {}).items()]
    for m, v in sorted(((m, v) for m, v in rows if m), key=lambda mv: int(mv[0].group(1))):
        label = " ".join(str(v or "").split())
        if not label:
            continue
        why = sample_problem(label)
        if why:
            problems.append(f"Sample for grid {int(m.group(1))}: {why}.")
        else:
            out[int(m.group(1))] = label
    return out, problems


def sample_key_text(samples: dict) -> str:
    """{grid: label} -> the lab's sample_key.csv (`#,Label`), as cryoem_core.metadata.load_sample_key reads it."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["#", "Label"])
    for g in sorted(samples):
        w.writerow([int(g), samples[g]])
    return buf.getvalue()


def samples_fieldset(values=None) -> str:
    """The upload form's optional "Sample on each grid" rows (grid 1-12)."""
    values = values or {}
    rows = "".join(
        f'<label class="cj-us-row"><span>Grid {n}</span><input name="sample_{n}" maxlength="{SAMPLE_MAX}" '
        f'autocomplete="off" value="{esc(str(values.get(f"sample_{n}") or ""), quote=True)}"></label>'
        for n in range(1, FORM_SLOTS + 1))
    return ('<fieldset class="cj-card cj-predict-fs"><legend>Sample on each grid (optional)</legend>'
            '<p class="cj-mut">The grid number is the autoloader slot in the file names (grid1_…, grid2_…). The sample '
            'name goes into the tile names, so it is asked now. Leave a row blank to keep “Grid n”. Not needed when '
            'you add the session’s sample_key.csv (use one or the other). Plain text only.</p>'
            f'<div class="cj-us-grid">{rows}</div></fieldset>')


# ---------------------------------------------------------------- what a finished conversion holds
def manifest_info(tiles: Path) -> dict:
    """tiles/MANIFEST.csv -> {"grids": [n], "labels": {slot: label} (not the default "Grid n"), "counts": {slot: n},
    "n_tiles": n}. Missing or unreadable -> empty values."""
    grids, labels, counts, n = set(), {}, {}, 0
    try:
        with open(tiles / MANIFEST, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                n += 1
                g = str(row.get("grid") or "").strip()
                if not g.isdigit():
                    continue
                slot = str(int(g))
                grids.add(int(g))
                counts[slot] = counts.get(slot, 0) + 1
                lab = " ".join(str(row.get("sample_label") or "").split())[:LABEL_MAX]
                if lab and lab != f"Grid {slot}" and slot not in labels:
                    labels[slot] = lab
    except (OSError, csv.Error, UnicodeDecodeError):
        pass
    return {"grids": sorted(grids), "labels": labels, "counts": counts, "n_tiles": n}


def acquisition_of(conv_dir: Path, sid: str) -> tuple:
    """<conv>/acquisition.json -> ({session/1 acquisition key: value} that pass the schema, n_montages or None).
    Only ACQ_KEYS are read: never a raw header, a serial or a path."""
    doc = _read_json(Path(conv_dir) / "acquisition.json")
    if not isinstance(doc, dict):
        return {}, None
    sess = _dict(doc.get("session"))
    out = {}
    regm = _regm()
    for k in ACQ_KEYS:
        v = sess.get(k)
        if v is None:
            continue
        probe = {"schema": "session/1", "id": sid, "logged": now_iso(), "status": "processed", k: v}
        if not regm.check_record(probe, regm.schemas()):
            out[k] = v
    scope = " ".join(str(doc.get("scope_model") or "").split())
    if "instrument" not in out and SCOPE_RE.match(scope):          # a model name only (allow-list)
        out["instrument"] = scope
    mons = doc.get("montages")
    return out, (len(mons) if isinstance(mons, list) else None)


def conversion_entry(conv_dir: Path, meta: dict, sid: str) -> dict:
    """The `conversions` entry for one conversion: deterministic from its files (registering twice gives the same)."""
    info = manifest_info(Path(conv_dir) / "tiles")
    acq, n_mont = acquisition_of(conv_dir, sid)
    entry = {"id": Path(conv_dir).name, "uploaded_by": meta.get("owner") if isinstance(meta.get("owner"), str) else None,
             "grids": info["grids"], "sample_labels": info["labels"] or None, "tiles_per_grid": info["counts"],
             "n_tiles": info["n_tiles"], "n_montages": n_mont, "acquisition": acq or None}
    if isinstance(meta.get("derived_from"), str):       # a re-conversion with other settings: not new tiles
        entry["derived_from"] = meta["derived_from"]
    if isinstance(meta.get("made_via"), str) and meta["made_via"]:   # an agent's upload (convert.json, round 11 #65)
        entry["uploaded_via"] = meta["made_via"]
    return {k: v for k, v in entry.items() if v is not None or k == "uploaded_by"}


def merge_session(cur: dict | None, sid: str, entry: dict) -> dict:
    """The current stored line (or None) + this conversion's entry -> the line to append (logged left to append).
    Lab fields are never replaced; top-level fields are only filled where the record has none."""
    if cur:
        rec = {k: v for k, v in cur.items() if k != "logged"}
    else:
        rec = {"schema": "session/1", "id": sid, "date": f"{sid[:4]}-{sid[4:6]}-{sid[6:8]}", "status": "processed"}
        rec.update(_dict(entry.get("acquisition")))
    convs = [c for c in _list(rec.get("conversions")) if isinstance(c, dict) and c.get("id") != entry["id"]]
    rec["conversions"] = sorted(convs + [entry], key=lambda c: str(c.get("id")))
    if not rec.get("status"):
        rec["status"] = "processed"
    if not cur and entry.get("sample_labels"):        # a session the portal makes; a lab record's labels stay its own
        rec["sample_labels"] = dict(entry["sample_labels"])
    rec["recorded_by"] = RECORDED_BY
    return rec


def _same(a: dict | None, b: dict | None) -> bool:
    ign = ("logged", "recorded_by")
    return a is not None and b is not None and \
        {k: v for k, v in a.items() if k not in ign} == {k: v for k, v in b.items() if k not in ign}


def _latest(regm, reg: Path, sid: str) -> dict | None:
    """The last stored session line for sid (file order), read fresh (no cache)."""
    rec = regm.current("session", reg).get(sid)
    return dict(rec) if isinstance(rec, dict) else None


def _write_local(conv_dir: Path, doc: dict):
    try:
        PS.write_json_atomic(Path(conv_dir) / SESSION_JSON, {"schema": SCHEMA_LOCAL, **doc, "t": now_iso()})
    except OSError as e:
        print(f"[upload_sessions] {Path(conv_dir).name}: session.json not written ({type(e).__name__})",
              file=sys.stderr, flush=True)


def register_conversion(conv_dir: Path, cfg) -> dict:
    """A finished upload conversion -> its session/1 record (read -> merge -> append). -> {"ok": bool, "session",
    "appended": bool} | {"ok": False, "skipped": why, "retry": bool} | {"ok": False, "error": text, "retry": bool}
    | {"ok": False, "busy": True, "retry": True, "error": text} (the lock was not had in LOCK_WAIT_S, or the
    registry's own write lock was busy: nothing appended; not a failure to count). ok True = nothing left to do.
    Never raises for a refusal, a busy lock or an unreadable folder."""
    conv_dir = Path(conv_dir)
    if not enabled(cfg):
        return {"ok": False, "skipped": f"{FLAG} is not on", "retry": False}
    meta = _read_json(conv_dir / "convert.json")
    if not isinstance(meta, dict):
        return {"ok": False, "error": "convert.json missing or unreadable", "retry": False}
    if _dict(meta.get("source")).get("kind") != "upload":
        return {"ok": True, "skipped": "not an upload (only uploads become sessions)", "appended": False}
    st = _read_json(conv_dir / PS.STATUS) or {}
    if st.get("state") != "done":
        return {"ok": False, "skipped": "the conversion is not done", "retry": st.get("state") in ("queued", "running")}
    sid = str(meta.get("session") or "")
    m = CID_RE.match(conv_dir.name)
    if not SESSION_RE.match(sid) or not m or m.group(1) != sid:
        return {"ok": False, "error": "the conversion's session id is not valid", "retry": False}
    reg = registry_root(cfg)
    if reg is None:
        return {"ok": False, "error": "no registry folder is configured", "retry": True}
    regm = _regm()
    try:
        entry = conversion_entry(conv_dir, meta, sid)
        with upload_lock(reg):
            cur = _latest(regm, reg, sid)
            rec = merge_session(cur, sid, entry)
            if _same(rec, cur):
                out = {"ok": True, "session": sid, "appended": False}
            else:
                regm.append(rec, reg)
                out = {"ok": True, "session": sid, "appended": True}
    except Busy as e:
        out = {"ok": False, "session": sid, "busy": True, "retry": True, "error": f"busy: {e}"}
    except regm.RegistryError as e:
        if str(e).startswith("registry busy"):
            out = {"ok": False, "session": sid, "busy": True, "retry": True, "error": str(e)[:300]}
        else:
            out = {"ok": False, "session": sid, "error": str(e)[:300], "retry": False}
    except OSError as e:
        out = {"ok": False, "session": sid, "error": f"{type(e).__name__}: {str(e)[:250]}", "retry": True}
    _write_local(conv_dir, {"session": sid, "conversion": conv_dir.name, "registered": out.get("ok", False),
                            "appended": out.get("appended", False), "busy": bool(out.get("busy")),
                            "error": out.get("error")})
    if out.get("error"):
        print(f"[upload_sessions] {conv_dir.name}: not registered: {out['error']}", file=sys.stderr, flush=True)
    return out


# ---------------------------------------------------------------- reading a record (Data and the conversion page)
def conversions_of(record: dict) -> list:
    return [c for c in _list(_dict(record).get("conversions")) if isinstance(c, dict) and isinstance(c.get("id"), str)]


def is_upload_session(record: dict) -> bool:
    return bool(conversions_of(record))


def uploaders(record: dict) -> set:
    return {c.get("uploaded_by") for c in conversions_of(record) if isinstance(c.get("uploaded_by"), str)}


def may_edit(user: dict, record: dict) -> bool:
    """The admin, or whoever uploaded one of the session's conversions."""
    return bool(user) and is_upload_session(record) and (
        user.get("role") == "admin" or user.get("id") in uploaders(record))


def _slot_key(s):
    return (0, int(s), "") if str(s).isdigit() else (1, 0, str(s))


def session_slots(record: dict) -> list:
    """Every grid slot the session knows of: lab labels / grid ids, the conversions' grids, the details."""
    rec = _dict(record)
    det = _dict(rec.get("details"))
    slots = set(_dict(rec.get("sample_labels"))) | set(_dict(rec.get("session_grids")))
    for c in conversions_of(rec):
        slots |= {str(g) for g in _list(c.get("grids")) if isinstance(g, int)}
        slots |= set(_dict(c.get("sample_labels")))
    slots |= set(_dict(det.get("grid_ids"))) | set(_dict(det.get("sample_display")))
    return sorted((str(s) for s in slots if SLOT_RE.match(str(s))), key=_slot_key)


def _labels(d) -> dict:
    return {str(k): v for k, v in _dict(d).items() if isinstance(v, str) and v}


def lab_labels(record: dict) -> dict:
    """{slot: label} the lab recorded (its sample_key.csv): a record with a lab session folder. A session the portal
    made has no local_folder; its top-level labels are the upload's."""
    rec = _dict(record)
    return _labels(rec.get("sample_labels")) if rec.get("local_folder") else {}


def upload_labels(record: dict) -> dict:
    """{slot: label} from the uploads: the details' display name, else the upload's own label."""
    rec = _dict(record)
    out = {} if rec.get("local_folder") else _labels(rec.get("sample_labels"))
    for c in conversions_of(rec):
        for k, v in _labels(c.get("sample_labels")).items():
            out.setdefault(k, v)
    out.update(_labels(_dict(rec.get("details")).get("sample_display")))
    return out


def recorded_labels(record: dict) -> dict:
    """{slot: label} as recorded (the lab's, else an upload's), before display edits."""
    rec = _dict(record)
    out = {} if rec.get("local_folder") else _labels(rec.get("sample_labels"))
    for c in conversions_of(rec):
        for k, v in _labels(c.get("sample_labels")).items():
            out.setdefault(k, v)
    out.update(lab_labels(rec))
    return out


def effective_labels(record: dict) -> dict:
    """{slot: label} as shown. A slot the lab labelled keeps the lab's label; an upload that names it otherwise is
    added as "Lab label (upload: X)", never in its place. Other slots: the display name, else the upload's label."""
    lab, up = lab_labels(record), upload_labels(record)
    out = dict(up)
    for slot, v in lab.items():
        out[slot] = v if up.get(slot) in (None, v) else f"{v} (upload: {up[slot]})"
    return out


def own_grids(record: dict, who) -> set:
    """Grid slots in the conversions `who` uploaded (where an uploader may set a display name)."""
    return {str(g) for c in conversions_of(record) if c.get("uploaded_by") == who
            for g in _list(c.get("grids")) if isinstance(g, int)}


def upload_acquisition(record: dict) -> dict:
    """The newest conversion's acquisition summary (for a record without one of its own)."""
    for c in sorted(conversions_of(record), key=lambda c: c["id"], reverse=True):
        acq = _dict(c.get("acquisition"))
        if acq:
            return acq
    return {}


def has_acquisition(record: dict) -> bool:
    return any(_dict(record).get(k) not in (None, [], {}) for k in ACQ_KEYS)


# ---------------------------------------------------------------- session details
def _clean_line(v, cap: int, what: str, problems: list) -> str:
    """One line: control characters out, spaces collapsed, no leading = + - @ (a spreadsheet formula)."""
    s = " ".join("".join(c for c in str(v or "") if ord(c) >= 32 and ord(c) != 127).split())
    s = s.lstrip("".join(FORMULA_START)).strip()
    if len(s) > cap:
        problems.append(f"{what}: at most {cap} characters.")
        return ""
    return s


def clean_details(fields: dict, slots: list, display_slots=None) -> dict:
    """The details form -> {"grid_ids": {slot: id}, "sample_display": {slot: name}, "prep_notes": text}. A blank
    value removes the entry. display_slots: where a display name may be set (None = every slot: the admin).
    Raises DetailsError listing every problem."""
    problems, grid_ids, display = [], {}, {}
    allowed = set(slots)
    seen = 0
    for k in sorted(fields or {}):
        m = re.match(r"^(grid_id|display)_(\d{1,3})$", str(k))
        if not m:
            continue
        slot = str(int(m.group(2)))
        if slot not in allowed:
            problems.append(f"Grid {slot} is not a grid of this session.")
            continue
        seen += 1
        if m.group(1) == "grid_id":
            grid_ids[slot] = _clean_line(fields[k], GRID_ID_MAX, f"Grid {slot} id", problems)
        elif display_slots is not None and slot not in display_slots:
            if str(fields[k] or "").strip():
                problems.append(f"Grid {slot}: only the person who uploaded that grid (or the admin) can rename "
                                f"its sample.")
        else:
            display[slot] = _clean_line(fields[k], DISPLAY_MAX, f"Grid {slot} sample name", problems)
    if seen > 2 * MAX_GRIDS:
        problems.append("Too many grid rows.")
    notes = str((fields or {}).get("prep_notes") or "").replace("\r\n", "\n").replace("\r", "\n")
    notes = "".join(c for c in notes if c in "\n\t" or ord(c) >= 32).strip()
    if len(notes) > NOTES_MAX:
        problems.append(f"Prep notes: at most {NOTES_MAX} characters (now {len(notes)}).")
    if problems:
        raise DetailsError(f"Please fix ({len(problems)}):\n" + "\n".join(problems))
    return {"grid_ids": grid_ids, "sample_display": display, "prep_notes": notes if "prep_notes" in (fields or {})
            else None}


def merge_details(cur: dict, upd: dict, who: str) -> dict:
    """The record's details + the form's values (posted rows replace, blank removes, other rows stay)."""
    old = _dict(cur.get("details"))
    out = {"grid_ids": dict(_dict(old.get("grid_ids"))), "sample_display": dict(_dict(old.get("sample_display"))),
           "prep_notes": old.get("prep_notes") if isinstance(old.get("prep_notes"), str) else None}
    for key in ("grid_ids", "sample_display"):
        for slot, v in upd[key].items():
            if v:
                out[key][slot] = v
            else:
                out[key].pop(slot, None)
    if upd.get("prep_notes") is not None:
        out["prep_notes"] = upd["prep_notes"] or None
    out["by"], out["at"] = who, now_iso()
    return out


def save_details(session_id: str, fields: dict, who: str, cfg, admin: bool = False, request=None) -> dict:
    """Merge the form into the session record's `details` and append (under the upload lock). -> {"ok": True,
    "appended": bool} | {"ok": False, "error": text, "busy"?: True}. The caller has checked who may edit (may_edit
    / can_see). A display name: the admin anywhere, an uploader only on the grids of their own conversions. Only a
    session that is already an upload session (has `conversions`) takes details."""
    if not enabled(cfg):
        return {"ok": False, "error": "Session details are not switched on on this server.", "status": 403}
    if not isinstance(session_id, str) or not SESSION_RE.match(session_id):
        return {"ok": False, "error": "Unknown session.", "status": 404}
    reg = registry_root(cfg)
    if reg is None:
        return {"ok": False, "error": "The registry is not set up on this server. Ask the admin."}
    regm = _regm()
    try:
        with upload_lock(reg):
            cur = _latest(regm, reg, session_id)
            if not cur or not is_upload_session(cur):
                return {"ok": False, "error": "This session has no uploaded conversion yet, so it takes no details."}
            upd = clean_details(fields, session_slots(cur), None if admin else own_grids(cur, who))
            rec = {k: v for k, v in cur.items() if k != "logged"}
            new = merge_details(cur, upd, who)
            base = merge_details(cur, {"grid_ids": {}, "sample_display": {}, "prep_notes": None}, who)
            if cur.get("details") is not None and \
                    {k: v for k, v in new.items() if k not in ("by", "at")} == \
                    {k: v for k, v in base.items() if k not in ("by", "at")}:
                return {"ok": True, "appended": False}
            rec["details"] = new
            rec["recorded_by"] = RECORDED_BY + " details"
            regm.append(rec, reg)
    except DetailsError as e:
        return {"ok": False, "error": str(e)}
    except Busy as e:
        return {"ok": False, "busy": True, "error": BUSY_TEXT, "tech": str(e), "status": 503}
    except regm.RegistryError as e:
        if str(e).startswith("registry busy"):
            return {"ok": False, "busy": True, "error": BUSY_TEXT, "tech": str(e)[:300], "status": 503}
        return {"ok": False, "error": "Not saved: the registry refused it.", "tech": str(e)[:300]}
    except OSError as e:
        return {"ok": False, "error": "Not saved: the registry could not be written. Ask the admin.",
                "tech": f"{type(e).__name__}: {str(e)[:250]}"}
    from portal import audit as PA                  # the one audit helper (round 11 #3): never raises
    if not PA.event(getattr(cfg, "state_dir", None), request, "session_details", session_id,
                    {"grids": len(upd["grid_ids"]) + len(upd["sample_display"]),
                     "notes_chars": len(upd["prep_notes"] or "")}, actor=who):
        print(f"[upload_sessions] {session_id}: details saved, audit line not written", file=sys.stderr, flush=True)
    return {"ok": True, "appended": True}


def current_record(cfg, session_id: str) -> dict | None:
    """The session's current registry record (read fresh), or None."""
    reg = registry_root(cfg)
    if reg is None or not isinstance(session_id, str) or not SESSION_RE.match(session_id):
        return None
    try:
        return _latest(_regm(), reg, session_id)
    except (OSError, ValueError):
        return None


def save_conversion_details(cfg, user: dict, conv_dir: Path, meta: dict, fields: dict) -> dict:
    """The conversion page's details form. The caller has checked can_see on the conversion (owner or admin); the
    session record must list this conversion."""
    sid = str(meta.get("session") or "")
    if _dict(meta.get("source")).get("kind") != "upload":
        return {"ok": False, "error": "Only uploaded conversions take session details.", "status": 400}
    rec = current_record(cfg, sid)
    if not rec or Path(conv_dir).name not in {c["id"] for c in conversions_of(rec)}:
        return {"ok": False, "error": "This upload is not in Data → Sessions yet, so it takes no details. Try again "
                                      "in a minute.", "status": 409}
    return save_details(sid, fields, user.get("id"), cfg, admin=user.get("role") == "admin")


def unregister_conversion(session_id: str, cid: str, root, who: str) -> dict:
    """Admin fix for an upload filed under the wrong session (a mistyped imaging date): append the session line
    without that conversion. A session the portal made (no lab folder) and left with no upload is marked
    "excluded". Nothing is deleted (append-only); details stay in the history. Re-upload under the right date.
    -> {"ok", "appended"} | {"ok": False, "error"}."""
    reg = Path(root)
    regm = _regm()
    try:
        with upload_lock(reg, wait=30.0):
            cur = _latest(regm, reg, session_id)
            if not cur or cid not in {c["id"] for c in conversions_of(cur)}:
                return {"ok": False, "error": f"session {session_id} lists no conversion {cid}"}
            rec = {k: v for k, v in cur.items() if k != "logged"}
            rec["conversions"] = [c for c in conversions_of(cur) if c["id"] != cid]
            if not rec["conversions"] and not cur.get("local_folder"):
                rec["status"] = "excluded"
            rec["note"] = f"upload {cid} removed from this session by {who} (filed under the wrong date)"
            rec["recorded_by"] = RECORDED_BY + " unregister"
            regm.append(rec, reg)
    except (Busy, regm.RegistryError, OSError) as e:
        return {"ok": False, "error": str(e)[:300]}
    return {"ok": True, "appended": True}


# ---------------------------------------------------------------- HTML (escaped here)
def conversion_details_html(cfg, user: dict, conv_dir: Path, meta: dict, st: dict, error=None, saved=False) -> str:
    """The conversion page's Session details card: the form once the session record lists this upload, else where
    the registration stands. Empty for a conversion that is not an upload, or when CJ_UPLOAD_SESSIONS is off."""
    if not enabled(cfg) or _dict(meta.get("source")).get("kind") != "upload" or st.get("state") != "done":
        return ""
    sid, cid = str(meta.get("session") or ""), Path(conv_dir).name
    rec = current_record(cfg, sid)
    if rec and cid in {c["id"] for c in conversions_of(rec)}:
        link = (f'<p class="cj-mut">This upload is listed as <a href="/data/session/{esc(sid)}">session '
                f'{esc(sid)}</a> under Data → Sessions.</p>')
        return link + details_form(rec, f"/predict/convert/{cid}/details", error, saved, user)
    reg = st.get("registered")
    local = _dict(_read_json(Path(conv_dir) / SESSION_JSON))
    why = reg[len("error:"):].strip() if isinstance(reg, str) and reg.startswith("error") else \
        (local.get("error") if not local.get("busy") else None)
    if why:
        tech = (f'<p class="cj-mut cj-wrap">(admin) {esc(str(why))}</p>' if user.get("role") == "admin" else "")
        return (f'<div class="cj-card"><b>Not added to Data → Sessions.</b> <span class="cj-mut">Ask the admin if you '
                f'need it there.</span>{tech}</div>')
    if reg is False or local.get("busy"):                 # waiting for (or retrying) the registration
        return ('<div class="cj-card"><p class="cj-mut">Adding this upload to Data → Sessions… Reload this page in a '
                'minute to fill in the session details.</p></div>')
    return ""


def details_view(record: dict) -> str:
    """Read-only: prep notes and who edited them last (grid ids and display names are on the grid cards)."""
    det = _dict(_dict(record).get("details"))
    notes = det.get("prep_notes") if isinstance(det.get("prep_notes"), str) else ""
    who = (f'<p class="cj-mut">Last edited {esc(str(det.get("at") or "?")[:16].replace("T", " "))} by '
           f'{esc(person_name(det.get("by")))}.</p>' if det.get("at") else "")
    body = (f'<p class="cj-us-notes">{esc(notes)}</p>' if notes else '<p class="cj-mut">No prep notes yet.</p>')
    return f'<div class="cj-card"><h3 class="cj-us-h">Prep notes</h3>{body}{who}</div>'


def person_name(pid) -> str:
    """A person's name for the page (portal/names.person_name: annotation/people.json), else the id."""
    from portal import names as NM
    return NM.person_name(pid)


def details_form(record: dict, action: str, error: str | None = None, saved: bool = False, user=None) -> str:
    """The "Session details" form for one session record (posted to `action`). `user`: who sees it; a non-admin
    gets a "Show as" box only on the grids of their own uploads (save_details checks the same)."""
    rec = _dict(record)
    user = _dict(user)
    own = None if user.get("role") == "admin" else own_grids(rec, user.get("id"))
    det = _dict(rec.get("details"))
    gids, disp = _dict(det.get("grid_ids")), _dict(det.get("sample_display"))
    labels = recorded_labels(rec)
    lab_grids = _dict(rec.get("session_grids"))
    rows = []
    for slot in session_slots(rec)[:MAX_GRIDS]:
        lab = labels.get(slot) or f"Grid {slot}"
        lab_id = lab_grids.get(slot)
        show = (f'<input name="display_{esc(slot)}" maxlength="{DISPLAY_MAX}" autocomplete="off" '
                f'value="{esc(str(disp.get(slot) or ""), quote=True)}" aria-label="Grid {esc(slot)} sample name">'
                if own is None or slot in own else
                f'<span class="cj-mut">{esc(str(disp.get(slot) or "")) or "(not your upload)"}</span>')
        rows.append(
            f'<tr><td data-l="Grid">Grid {esc(slot)}<br><span class="cj-mut">{esc(lab)}</span></td>'
            f'<td data-l="Grid id"><input name="grid_id_{esc(slot)}" maxlength="{GRID_ID_MAX}" autocomplete="off" '
            f'placeholder="{esc(str(lab_id) if lab_id else "e.g. 2602", quote=True)}" '
            f'value="{esc(str(gids.get(slot) or ""), quote=True)}" aria-label="Grid {esc(slot)} id"></td>'
            f'<td data-l="Show as">{show}</td></tr>')
    table = ('<table class="cj-us-table"><thead><tr><th>Grid</th><th>Grid id (inventory #)</th><th>Show sample as'
             '</th></tr></thead><tbody>' + "".join(rows) + '</tbody></table>') if rows else \
        '<p class="cj-mut">No grids are known for this session yet.</p>'
    notes = det.get("prep_notes") if isinstance(det.get("prep_notes"), str) else ""
    msg = ""
    if error:
        msg = '<div class="cj-us-msg" role="alert">' + "<br>".join(esc(x) for x in str(error).split("\n")) + "</div>"
    elif saved:
        msg = '<p class="cj-us-ok" role="status">Saved.</p>'
    return (f'<form method="post" action="{esc(action, quote=True)}" class="cj-card cj-us-form" '
            f'id="cj-us-details"><h2 class="cj-us-h">Session details</h2>{msg}'
            '<p class="cj-mut">What the microscope files don’t say. “Show sample as” changes how a grid’s sample is '
            'shown here, not the tile names; where the lab already named the sample, both are shown. Visible to '
            'admins and advanced users.</p>'
            f'{table}<label class="cj-us-field"><span>Prep notes (up to {NOTES_MAX} characters)</span>'
            f'<textarea name="prep_notes" maxlength="{NOTES_MAX}" rows="5">{esc(notes)}</textarea></label>'
            '<p><button class="cj-us-btn" type="submit">Save details</button></p></form>')


# ---------------------------------------------------------------- conversions and analyses (Data)
def conversion_dir(cfg, cid: str):
    """(folder, convert.json) of a conversion id, or (None, None)."""
    if not isinstance(cid, str) or not CID_RE.match(cid):
        return None, None
    from portal.modules import predict as P
    d = P.conv_dir(cfg, cid)
    meta = _read_json(d / "convert.json") if d else None
    return (d, meta) if isinstance(meta, dict) else (None, None)


def visible_conversions(cfg, record: dict, user: dict, can_see) -> list:
    """The record's conversions this person may open (can_see on convert.json's owner: the owner or the admin):
    [{id, dir, href, gallery, n_tiles}]. Others are left out (no link, no id)."""
    out = []
    for c in conversions_of(record):
        d, meta = conversion_dir(cfg, c["id"])
        if d is None or not can_see(user, meta.get("owner")):
            continue
        out.append({"id": c["id"], "dir": d, "href": f"/predict/convert/{c['id']}/",
                    "gallery": f"/gallery/{c['id']}/", "n_tiles": c.get("n_tiles"),
                    "derived_from": c.get("derived_from") if isinstance(c.get("derived_from"), str) else None})
    return out


def conversion_roots(cfg) -> list:
    """Where conversion folders live (the thumbnail containment check adds these)."""
    try:
        from portal.modules import predict as P
        return list(P.raw_roots(cfg))
    except Exception:                                   # noqa: BLE001 - no predict root configured
        return []


def conversion_images(cfg, record: dict, user: dict, can_see) -> list:
    """Image files of the conversions this person may open: stitched montages first, then tiles (each folder's own
    files only; links leading out are dropped by the caller's containment check)."""
    files = []
    for c in visible_conversions(cfg, record, user, can_see):
        for sub, exts in (("montages", (".jpg", ".jpeg", ".png")), ("tiles", (".png", ".jpg", ".jpeg"))):
            folder = c["dir"] / sub
            try:
                names = sorted((e for e in os.scandir(folder) if e.is_file() and not e.name.startswith(".")
                                and e.name.lower().endswith(exts)), key=lambda e: e.name.lower())
            except OSError:
                continue
            files += [Path(e.path) for e in names]
    return files


# ---------------------------------------------------------------- admin CLI
def main(argv=None) -> int:
    """python -m portal.upload_sessions unregister --registry <dir> --session <id> --conversion <cid> --who <id>"""
    import argparse
    ap = argparse.ArgumentParser(description="Upload sessions: admin fixes (see the module docstring).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    un = sub.add_parser("unregister", help="take an upload off a session it was filed under by mistake")
    un.add_argument("--registry", required=True, type=Path)
    un.add_argument("--session", required=True)
    un.add_argument("--conversion", required=True)
    un.add_argument("--who", required=True, help="your portal id (recorded in the line's note)")
    a = ap.parse_args(argv)
    res = unregister_conversion(a.session, a.conversion, a.registry, a.who)
    print(json.dumps(res))
    return 0 if res.get("ok") else 2


if __name__ == "__main__":
    sys.exit(main())
