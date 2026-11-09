"""
Upload sessions (round 7 S, BUILD_CONTRACTS.md section 13): a raw upload's conversion becomes the registry
`session/1` record of its imaging session, so Data -> Sessions shows it (the registry is the one store).

    register_conversion(conv_dir, cfg) -> dict      jobs side: CX's finalize calls it after the runner lock is released
    save_details(session_id, fields, who, cfg) -> dict      the "Session details" form (conversion page, Data page)

Registration
  - Only when CJ_UPLOAD_SESSIONS=on (env, or cfg.upload_sessions) and only for a conversion whose convert.json says
    source.kind == "upload" and whose status.json says done. Nothing is written otherwise (no registry line, no file).
  - The session id is the conversion's session (the imaging date the uploader typed), so it WILL match a lab session:
    read the current line -> merge -> append, under <registry>/.upload_session.lock (registry/importers.py takes the
    same lock while it writes, and carries `conversions` and `details` forward). Lab fields are never replaced:
    status, local_folder, sample_labels, the acquisition summary and grid ids stay as the lab recorded them. A new
    record gets status "processed", its date, the sample labels and the acquisition summary of this upload.
  - Upload-owned keys (UPLOAD_KEYS): `conversions` [{id, uploaded_by, grids, sample_labels, tiles_per_grid, n_tiles,
    n_montages, acquisition}] (one entry per conversion id; `owner` is NOT used: in the registry it names an owner
    file) and `details` (below).
  - The acquisition summary comes from <conv>/acquisition.json (written by jobs/convert_raw.py with the importer's
    file_summary / session_acquisition / scope_model): only the session/1 acquisition keys, each checked against the
    schema, never a raw header, serial or path. A missing file registers without it.
  - Idempotent: a merged record equal to the current one (logged / recorded_by aside) appends nothing.
  - A refusal (an excluded session, a bad record) is returned as {"ok": False, "error"} and written to
    <conv>/session.json; a busy lock or an unreadable registry also says "retry": True. Never raises for those.

Session details
  - grid ids (inventory Grid#, free text up to 40 characters), prep notes (up to 2000 characters) and a display
    sample name per grid (the labels baked into tile names and MANIFEST are not changed). Merge-append under the
    same lock; an audit line (portal/audit.py, no user text in it). Who may edit: the admin, and whoever uploaded
    one of the session's conversions (may_edit); the conversion page also lets its owner (can_see) edit.

The upload form asks for the sample per grid BEFORE conversion (form_samples / sample_key_text): the labels are baked
into tile names and MANIFEST.csv, so they are written as input/sample_key.csv in the lab's `#,Label` format (what
cryoem_core.metadata.load_sample_key and jobs/convert_raw.py --sample-key read), printable ASCII only (the lab
parser opens the file in the system code page).

All text that reaches a page goes through html.escape here.
"""
from __future__ import annotations

import csv
import html
import io
import json
import os
import re
import sys
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

esc = html.escape


class DetailsError(ValueError):
    """The details form was refused: a message for the person (plain words)."""


# ---------------------------------------------------------------- settings
def enabled(cfg) -> bool:
    v = getattr(cfg, "upload_sessions", None)
    v = v if v not in (None, "") else os.environ.get(FLAG, "")
    return str(v).strip().lower() == "on"


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
    if "instrument" not in out and isinstance(doc.get("scope_model"), str) and doc["scope_model"].strip():
        out["instrument"] = doc["scope_model"].strip()[:80]
    mons = doc.get("montages")
    return out, (len(mons) if isinstance(mons, list) else None)


def conversion_entry(conv_dir: Path, meta: dict, sid: str) -> dict:
    """The `conversions` entry for one conversion: deterministic from its files (registering twice gives the same)."""
    info = manifest_info(Path(conv_dir) / "tiles")
    acq, n_mont = acquisition_of(conv_dir, sid)
    entry = {"id": Path(conv_dir).name, "uploaded_by": meta.get("owner") if isinstance(meta.get("owner"), str) else None,
             "grids": info["grids"], "sample_labels": info["labels"] or None, "tiles_per_grid": info["counts"],
             "n_tiles": info["n_tiles"], "n_montages": n_mont, "acquisition": acq or None}
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
    if not rec.get("sample_labels") and entry.get("sample_labels"):
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
    "appended": bool} | {"ok": False, "skipped": why} | {"ok": False, "error": text, "retry": bool}.
    ok True = nothing left to do. Never raises for a refusal, a busy lock or an unreadable folder."""
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
        with PS.locked(reg / LOCK):
            cur = _latest(regm, reg, sid)
            rec = merge_session(cur, sid, entry)
            if _same(rec, cur):
                out = {"ok": True, "session": sid, "appended": False}
            else:
                regm.append(rec, reg)
                out = {"ok": True, "session": sid, "appended": True}
    except regm.RegistryError as e:
        busy = str(e).startswith("registry busy")
        out = {"ok": False, "session": sid, "error": str(e)[:300], "retry": busy}
    except (PS.PredictSetError, OSError) as e:
        out = {"ok": False, "session": sid, "error": f"{type(e).__name__}: {str(e)[:250]}", "retry": True}
    _write_local(conv_dir, {"session": sid, "conversion": conv_dir.name, "registered": out.get("ok", False),
                            "appended": out.get("appended", False), "error": out.get("error")})
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


def recorded_labels(record: dict) -> dict:
    """{slot: label} as recorded (lab sample_labels, else an upload's), before display edits."""
    out = {}
    for c in conversions_of(record):
        for k, v in _dict(c.get("sample_labels")).items():
            if isinstance(v, str) and v:
                out.setdefault(str(k), v)
    out.update({str(k): v for k, v in _dict(_dict(record).get("sample_labels")).items() if isinstance(v, str) and v})
    return out


def effective_labels(record: dict) -> dict:
    """{slot: label} as shown: the details' display name, else the recorded label."""
    out = recorded_labels(record)
    out.update({str(k): v for k, v in _dict(_dict(_dict(record).get("details")).get("sample_display")).items()
                if isinstance(v, str) and v})
    return out


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
    s = " ".join(str(v or "").split())
    if len(s) > cap:
        problems.append(f"{what}: at most {cap} characters.")
        return ""
    return s


def clean_details(fields: dict, slots: list) -> dict:
    """The details form -> {"grid_ids": {slot: id}, "sample_display": {slot: name}, "prep_notes": text}. A blank
    value removes the entry. Raises DetailsError listing every problem."""
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


def save_details(session_id: str, fields: dict, who: str, cfg) -> dict:
    """Merge the form into the session record's `details` and append (under the upload lock). -> {"ok": True,
    "appended": bool} | {"ok": False, "error": text}. The caller has checked who may edit (may_edit / can_see).
    Only a session that is already an upload session (has `conversions`) takes details."""
    if not enabled(cfg):
        return {"ok": False, "error": "Session details are not switched on on this server.", "status": 403}
    if not isinstance(session_id, str) or not SESSION_RE.match(session_id):
        return {"ok": False, "error": "Unknown session.", "status": 404}
    reg = registry_root(cfg)
    if reg is None:
        return {"ok": False, "error": "The registry is not set up on this server. Ask the admin."}
    regm = _regm()
    try:
        with PS.locked(reg / LOCK):
            cur = _latest(regm, reg, session_id)
            if not cur or not is_upload_session(cur):
                return {"ok": False, "error": "This session has no uploaded conversion yet, so it takes no details."}
            upd = clean_details(fields, session_slots(cur))
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
    except regm.RegistryError as e:
        return {"ok": False, "error": "Not saved: the registry refused it.", "tech": str(e)[:300]}
    except (PS.PredictSetError, OSError) as e:
        return {"ok": False, "error": "Not saved: the registry is busy. Try again in a minute.",
                "tech": f"{type(e).__name__}: {str(e)[:250]}"}
    state = getattr(cfg, "state_dir", None)
    try:
        if state is None:
            raise OSError("no state folder")
        from portal import audit as PA
        PA.audit(state, who, "session_details", session_id,
                 {"grids": len(upd["grid_ids"]) + len(upd["sample_display"]),
                  "notes_chars": len(upd["prep_notes"] or "")})
    except (OSError, TypeError):
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
    return save_details(sid, fields, user.get("id"), cfg)


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
        return link + details_form(rec, f"/predict/convert/{cid}/details", error, saved)
    reg = st.get("registered")
    local = _dict(_read_json(Path(conv_dir) / SESSION_JSON))
    why = reg[len("error:"):].strip() if isinstance(reg, str) and reg.startswith("error") else local.get("error")
    if why:
        tech = (f'<p class="cj-mut cj-wrap">(admin) {esc(str(why))}</p>' if user.get("role") == "admin" else "")
        return (f'<div class="cj-card"><b>Not added to Data → Sessions.</b> <span class="cj-mut">Ask the admin if you '
                f'need it there.</span>{tech}</div>')
    return ('<div class="cj-card"><p class="cj-mut">Adding this upload to Data → Sessions… Reload this page in a '
            'minute to fill in the session details.</p></div>')


def details_view(record: dict) -> str:
    """Read-only: prep notes and who edited them last (grid ids and display names are on the grid cards)."""
    det = _dict(_dict(record).get("details"))
    notes = det.get("prep_notes") if isinstance(det.get("prep_notes"), str) else ""
    who = (f'<p class="cj-mut">Last edited {esc(str(det.get("at") or "?")[:16].replace("T", " "))} by '
           f'{esc(str(det.get("by") or "?"))}.</p>' if det.get("at") else "")
    body = (f'<p class="cj-us-notes">{esc(notes)}</p>' if notes else '<p class="cj-mut">No prep notes yet.</p>')
    return f'<div class="cj-card"><h3 class="cj-us-h">Prep notes</h3>{body}{who}</div>'


def details_form(record: dict, action: str, error: str | None = None, saved: bool = False) -> str:
    """The "Session details" form for one session record (posted to `action`)."""
    rec = _dict(record)
    det = _dict(rec.get("details"))
    gids, disp = _dict(det.get("grid_ids")), _dict(det.get("sample_display"))
    labels = recorded_labels(rec)
    lab_grids = _dict(rec.get("session_grids"))
    rows = []
    for slot in session_slots(rec)[:MAX_GRIDS]:
        lab = labels.get(slot) or f"Grid {slot}"
        lab_id = lab_grids.get(slot)
        rows.append(
            f'<tr><td data-l="Grid">Grid {esc(slot)}<br><span class="cj-mut">{esc(lab)}</span></td>'
            f'<td data-l="Grid id"><input name="grid_id_{esc(slot)}" maxlength="{GRID_ID_MAX}" autocomplete="off" '
            f'placeholder="{esc(str(lab_id) if lab_id else "e.g. 2602", quote=True)}" '
            f'value="{esc(str(gids.get(slot) or ""), quote=True)}" aria-label="Grid {esc(slot)} id"></td>'
            f'<td data-l="Show as"><input name="display_{esc(slot)}" maxlength="{DISPLAY_MAX}" autocomplete="off" '
            f'value="{esc(str(disp.get(slot) or ""), quote=True)}" aria-label="Grid {esc(slot)} sample name"></td></tr>')
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
            'shown here, not the tile names. Visible to admins and advanced users.</p>'
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
                    "gallery": f"/gallery/{c['id']}/", "n_tiles": c.get("n_tiles")})
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
