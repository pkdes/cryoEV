"""
Predict, creating sets: the sources people may pick (sessions + their finished conversions), the checks (nm/px,
model, active limit), background create tasks, the kept-tiles scope, the session plan / run / submit, the audit
records and rename.

Split out of portal/modules/predict.py (round 7, P7; no behaviour change); predict.py re-exports every name.
Names the tests patch are used from other parts only as <PART>.<name> (portal/predict_parts/__init__.py).
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import secrets
import shutil
import sys
import threading
import time
from pathlib import Path

from jobs import predict_local as PL
from jobs import predict_set as PS
from portal import predict_sources as PSRC

from portal.predict_parts.common import (CONVERT_JSON, CONVERT_TILES, DISK_NEED_FACTOR, IMAGE_EXT, MAX_ACTIVE_BASIC,
                                         NM_MAX, NM_MIN, NM_TOLERANCE, NO_MODEL, PORTAL_TMP, RAW_ID_RE, RESUBMIT_S,
                                         Refusal, _plain, _read_json, os_error_text, registry_root, when)
from portal.predict_parts.sets import (_num, allowed_models, list_sets, nm_range_text, one_scale, pick)
from portal.predict_parts.runner import (list_conversions, raw_base)
from portal.predict_parts import common as COMMON
from portal.predict_parts import sets as SETS
from portal.predict_parts import runner as RUNNER


# ---------------------------------------------------------------- the sources people may pick
def sources(cfg) -> list:
    """The sessions people may pick: [{key, label, dir, manifest, session, sample_column, problem}]. The admin's
    CJ_PREDICT_SOURCES list (as before) merged with the sessions found by the scan, with the Admin page's overrides
    (portal/predict_sources.py). A hidden or incompatible source has a problem (shown to the admin only). A bad
    setting raises (sources_safe: an empty list, and the admin sees why on the page)."""
    return PSRC.for_predict(cfg)


def sources_safe(cfg, user=None, can_see=None):
    """(sources, problem). With a user: also the finished raw conversions that person may see (owner or admin)."""
    extra = converted_sources(cfg, user, can_see) if user is not None and can_see is not None else []
    try:
        return sources(cfg) + extra, None
    except (OSError, ValueError) as e:
        return extra, f"Session list unreadable: {type(e).__name__}: {e}"


# ---------------------------------------------------------------- the Name field
def chosen_name(f: dict):
    """The name to store: the cleaned Name field, or None when empty or still the prefill that was offered."""
    n = PS.clean_name(f.get("name"))
    return None if n is None or n == PS.clean_name(f.get("name_default")) else n


# ---------------------------------------------------------------- creating sets
def check_nm(text, what) -> tuple:
    v = _num(str(text or "").strip().replace(",", "."))
    if v is None:
        return None, f"{what}: enter the pixel size as a number (nm per pixel), e.g. 1.47"
    if not NM_MIN <= v <= NM_MAX:
        return None, f"{what}: {v:g} nm per pixel is outside {NM_MIN:g}-{NM_MAX:g}; check the value"
    return v, None


def check_model(cfg, user, model) -> str | None:
    _, choices = allowed_models(cfg, user)
    if not choices:
        return NO_MODEL
    if model not in {c["id"] for c in choices}:
        return "Pick a model from the list."
    return None


def scratch(cfg, kind, raw=False) -> Path:
    """A fresh scratch folder under <root>/.portal; raw=True: under the raw root (a raw upload stays on the drive its
    conversion will use, so the move into raw_tiles/<id>/input is one os.replace on one drive)."""
    base = raw_base(cfg) if raw else COMMON.predict_root(cfg)
    d = base / PORTAL_TMP / f"{kind}_{secrets.token_hex(6)}"
    d.mkdir(parents=True)
    return d


# ---------------------------------------------------------------- creating a set in the background
# A set of 1000 tiles takes longer to create than one HTTP request may last (a tunnel gives up at about 100 s), and
# a retried request would make a second set. So the request only validates and starts a task, answers 303 to a
# progress page, and the page polls. A task is keyed: an upload by its upload id, a session pick by owner + session +
# selection. The same key while it runs is the same task (a reload or a double submit starts nothing new).
_CREATES: dict = {}
_CREATE_LOCK = threading.Lock()


def create_task(tid) -> dict | None:
    with _CREATE_LOCK:
        t = _CREATES.get(tid)
        return dict(t) if t else None


def create_running(tid) -> bool:
    return (create_task(tid) or {}).get("state") == "running"


def recently_done(tid) -> bool:
    """The same pick finished within EST_TTL_S: pressing the button again opens that set, it makes no copy."""
    t = create_task(tid) or {}
    return t.get("state") == "done" and time.time() - t.get("finished", 0) < RESUBMIT_S


def running_creates(user, exclude=None) -> int:
    with _CREATE_LOCK:
        return sum(1 for k, t in _CREATES.items()
                   if t["state"] == "running" and t["owner"] == user.get("id") and k != exclude)


def start_create(tid, owner, fn, *, back, upload=None) -> bool:
    """Run fn(progress) -> the new set's folder in a daemon thread, unless this tid is already running (False).
    progress(done, total, phase) is for the page. A failure is kept for the page; fn cleans up after a success."""
    with _CREATE_LOCK:
        cur = _CREATES.get(tid)
        if cur and cur["state"] == "running":
            return False
        _CREATES.pop(tid, None)
        _CREATES[tid] = {"state": "running", "owner": owner, "done": 0, "total": 0, "phase": "", "ps": None,
                         "error": None, "tech": None, "status": None, "back": back, "upload": upload,
                         "started": time.time()}
        for k in [k for k, v in _CREATES.items() if v["state"] != "running"][:max(0, len(_CREATES) - COMMON.CACHE_MAX)]:
            _CREATES.pop(k, None)

    def progress(done, total, phase=""):
        with _CREATE_LOCK:
            t = _CREATES.get(tid)
            if t is not None:
                t.update(done=int(done), total=int(total), phase=phase)

    def run():
        try:
            ps_dir = fn(progress)
            change = {"state": "done", "ps": Path(ps_dir).name, "finished": time.time()}
        except Refusal as e:
            change = {"state": "failed", "error": e.msg, "tech": e.tech, "status": e.status}
        except OSError as e:
            change = {"state": "failed", "error": os_error_text(e), "tech": f"{type(e).__name__}: {e}", "status": 500}
        except BaseException as e:          # noqa: BLE001 - kept and shown, never lost in a thread
            change = {"state": "failed", "error": "Your images could not be prepared. Nothing was started. Try again, "
                                                  "or ask the admin.", "tech": f"{type(e).__name__}: {e}", "status": 500}
        with _CREATE_LOCK:
            t = _CREATES.get(tid)
            if t is not None:
                t.update(change)

    threading.Thread(target=run, name=f"cj-predict-create-{tid}", daemon=True).start()
    return True


def selection_key(user, session, model, rows) -> str:
    """sc_<12 hex>: the same person, session, model and tiles (with their sample and nm/px) = the same task."""
    blob = json.dumps([user.get("id"), str(session), model, sorted(rows)], ensure_ascii=False)
    return "sc_" + hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]


def tile_bytes(d_from: Path, names=None) -> int:
    """Bytes of the tiles a create will place (the listed names, else every image file of the folder)."""
    total = 0
    try:
        paths = [d_from / n for n in names] if names is not None else \
            [p for p in d_from.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXT]
        for p in paths:
            try:
                total += p.stat().st_size
            except OSError:
                pass
    except OSError:
        pass
    return total


def same_volume(a: Path, b: Path) -> bool:
    try:
        return os.stat(a).st_dev == os.stat(b).st_dev and PS.protected_problem(a) is None
    except OSError:
        return False


_LINK_OK: dict = {}                      # predict root -> whether this volume can hard-link (probed once)


def can_link(root: Path) -> bool:
    """Whether hard links work on the volume of `root` (some, e.g. exFAT or a network share, can't): probed with
    two scratch names under <root>/.portal."""
    key = str(root)
    if key not in _LINK_OK:
        probe = root / PORTAL_TMP / f"probe_{secrets.token_hex(4)}"
        try:
            probe.parent.mkdir(parents=True, exist_ok=True)
            probe.write_bytes(b"x")
            os.link(probe, str(probe) + ".l")
            _LINK_OK[key] = True
        except OSError:
            _LINK_OK[key] = False
        finally:
            for q in (probe, Path(str(probe) + ".l")):
                try:
                    q.unlink()
                except OSError:
                    pass
    return _LINK_OK[key]


def check_create_disk(root: Path, d_from: Path, names, link: bool):
    """Refuse (507) when the tiles to be copied (their total x 1.1, plus the free margin and the uploads in flight:
    check_disk_for) do not fit. Hard links need none: only when link is wanted, the tiles are on the set's drive
    AND that drive can link; otherwise _place copies and the copy size is checked."""
    if link and same_volume(d_from, root) and can_link(root):
        return
    RUNNER.check_disk_for(root, tile_bytes(d_from, names), DISK_NEED_FACTOR, "for these images",
                   "Nothing was started. Ask the admin.")


def create_and_start(cfg, d_from, user, model, *, names=None, manifest=None, sample_label=None, nm=None,
                     session=None, name=None, link=False, progress=None) -> Path:
    """create() then start_detached(). Runs in a worker thread. Raises Refusal (plain) on a refusal.
    link=True (uploads): the tiles are hard-linked into the set when they are on the set's drive, else copied.
    progress(done, total, phase): for the progress page."""
    root, reg = COMMON.predict_root(cfg), registry_root(cfg)
    root.mkdir(parents=True, exist_ok=True)
    check_create_disk(root, Path(d_from), names, link)
    kw = dict(model=model, owner=user["id"], nm_confirmed_by=user["id"], registry_root=reg, root=root,
              session=session or None, name=name, link=link, progress=progress)
    if manifest is not None:
        kw.update(manifest=manifest, sample_label_column="manifest:sample_label", nm_from_manifest=True, names=names)
    else:
        kw.update(sample_label=sample_label, nm_per_px=nm)
    try:
        ps_dir = PS.create(d_from, **kw)
    except PS.PredictSetError as e:
        text = re.sub(r"^refused \((\d+) problem\(s\)\)", r"Not started: \1 problem(s)", _plain(str(e), [d_from, root]))
        raise Refusal(re.sub(r"^refused: ", "Not started: ", text), 400, tech=str(e))
    try:
        PL.start_detached(ps_dir, root=root, registry_root=reg)
    except Exception as e:                       # the set exists (queued): its page offers "Start again"
        PS.write_status(ps_dir, waiting=f"could not start the run: {type(e).__name__}: {e}")
    return ps_dir


def read_tiles(src: dict) -> dict:
    """tiles_of(), with an unreadable folder or manifest as a plain refusal (details for the admin)."""
    try:
        return SETS.tiles_of(src)
    except (OSError, csv.Error, UnicodeDecodeError) as e:
        raise Refusal("This session’s tile list can’t be read right now. Try again later, or ask the admin.", 503,
                      tech=f"{type(e).__name__}: {e}")


def active_sets(cfg, user) -> int:
    """This person's own sets that are queued or running."""
    return sum(1 for _, req, st in list_sets(cfg, user, lambda u, owner: owner == u.get("id"))
               if st.get("state") in ("queued", "running"))


def check_active(cfg, user, exclude=None):
    """Basic users may have at most MAX_ACTIVE_BASIC sets queued or running (failed and done sets don't count;
    sets still being prepared do, except the task `exclude` itself)."""
    if user.get("role") == "basic" and active_sets(cfg, user) + running_creates(user, exclude) >= MAX_ACTIVE_BASIC:
        raise Refusal(f"You already have {MAX_ACTIVE_BASIC} analyses waiting or running. Start a new one when one "
                      f"of them is done.", 429)


KEPT_CHANGED = ("The kept tiles changed in the Gallery since this page was opened. The samples and pixel sizes below "
                "are from the manifest again: check them, then confirm.")


def kept_form(q) -> dict | None:
    """The gallery hand-over of a session form's query / posted fields: {which, from, kept_maybe}, else None."""
    if str(q.get("which") or "") != "kept":
        return None
    return {"which": "kept", "from": str(q.get("from") or ""), "kept_maybe": "1" if str(q.get("kept_maybe") or "") == "1" else ""}


def kept_scope(cfg, src: dict, q, t: dict) -> dict:
    """`t` (tiles_of) narrowed to the tiles the person flagged keep (and maybe, with kept_maybe=1) in the Gallery of
    this conversion: derived here from gallery/events.jsonl with the Gallery's own fold, never from a posted list.
    Without which=kept: `t` itself, unchanged. Raises Refusal when nothing is flagged or `from` is not this source."""
    k = kept_form(q)
    if k is None:
        return t
    if not k["from"] or k["from"] != src.get("conversion"):
        raise Refusal("These kept tiles belong to a different conversion. Open the Gallery of this conversion and use "
                      "its “Analyse kept tiles” button.", 400)
    from portal.modules import gallery as G                 # lazily: gallery imports this module
    d = Path(src["dir"]).parent                              # the source's own conversion folder (from == src["conversion"])
    meta = _read_json(d / CONVERT_JSON) if d else None
    if not isinstance(meta, dict):
        raise Refusal("This conversion can’t be read right now. Try again later.", 503)
    c = G.Conv(d, meta, {})
    state = G.state_of(c)
    flags = ("keep", "maybe") if k["kept_maybe"] else ("keep",)
    kept = {n for n in c.tiles if (state.get(n) or {}).get("flag") in flags}
    groups = []
    for i, g in enumerate(t["groups"]):
        names = [n for n in g["names"] if n in kept]
        if not names:
            continue
        nm_of = {n: g["nm_of"][n] for n in names}
        per_tile = [v for v in nm_of.values() if v is not None]
        missing = len(names) - len(per_tile)
        vals = sorted(set(per_tile))
        mixed = len(vals) > 1 and not one_scale(per_tile)
        groups.append({**g, "gid": f"g{i}", "names": names, "nm_of": nm_of,
                       "nm": vals[0] if len(vals) == 1 and not missing else None, "nm_values": vals,
                       "nm_min": vals[0] if vals else None, "nm_max": vals[-1] if vals else None,
                       "nm_missing": missing, "nm_range": len(vals) > 1 and not mixed and not missing, "mixed": mixed})
    n = sum(len(g["names"]) for g in groups)
    if not n:
        raise Refusal("No tiles are marked keep" + (" or maybe" if k["kept_maybe"] else "") +
                      " in the Gallery of this conversion. Mark some there, then try again.", 409)
    sig = hashlib.sha1(json.dumps([t["sig"], sorted(kept), [g["nm_values"] for g in groups]]).encode()).hexdigest()[:12]
    return {**t, "groups": groups, "n": n, "sig": sig, "kept": k}


def session_plan(cfg, user, src: dict, form: dict) -> dict:
    """Every check of a session form, no work: -> what session_run needs (+ `tid`, the task key). Raises Refusal."""
    if form.get("confirm") != "yes":
        raise Refusal("Tick “I confirm the pixel size”: nothing was started.")
    t = kept_scope(cfg, src, form, read_tiles(src))
    if form.get("sig") != t["sig"]:
        raise Refusal(KEPT_CHANGED if t.get("kept") else
                      "The session’s tiles changed since this page was opened. Check the samples again.")
    problems = []
    chosen = []
    for i, g in enumerate(t["groups"]):
        gid = g.get("gid", f"g{i}")
        if gid not in (form.get("samples") or []):
            continue
        if g["mixed"]:
            problems.append(f"{g['label']}: pixel sizes differ by more than {NM_TOLERANCE * 100:g} %; it can't be "
                            f"analysed as one sample")
            continue
        typed = str(form.get(f"nm_{gid}") or "").strip()
        if g["nm_range"] and not typed:          # each tile keeps its own manifest value (confirmed by the tick)
            bad = [v for v in (g["nm_min"], g["nm_max"]) if not NM_MIN <= v <= NM_MAX]
            if bad:
                problems.append(f"{g['label']}: {bad[0]:g} nm per pixel (from the manifest) is outside "
                                f"{NM_MIN:g}-{NM_MAX:g}; check the value, or type the right one")
                continue
            chosen.append({**g, "nm_in": None})
            continue
        nm, err = check_nm(typed, g["label"])
        if err:
            problems.append(err)
        chosen.append({**g, "nm_in": nm})
    if not chosen and not problems:
        problems.append("Tick at least one sample.")
    label = None
    if t["column"] is None:
        label = str(form.get("label") or "").strip()
        if not label or len(label) > 80:
            problems.append("Sample name: 1-80 characters.")
    count = None
    if form.get("how") == "count" and not t.get("kept"):          # kept tiles: exactly those, never a sample of them
        try:
            count = int(str(form.get("count") or "").strip())
            if count < 1:
                raise ValueError
        except ValueError:
            problems.append("Number of tiles: a whole number of 1 or more.")
    m_err = check_model(cfg, user, form.get("model"))
    if m_err:
        problems.append(m_err)
    if problems:
        raise Refusal(f"Please fix ({len(problems)}):\n" + "\n".join(problems))
    names = pick(chosen, count)
    if len(names) > COMMON.MAX_FILES and t.get("kept"):
        raise Refusal(f"More than {COMMON.MAX_FILES} kept tiles ({len(names)}): untick a sample, or reject some "
                      f"tiles in the Gallery.")
    if len(names) > COMMON.MAX_FILES:
        raise Refusal(f"{len(names)} tiles chosen; an analysis takes at most {COMMON.MAX_FILES}. Choose “Only ... "
                      f"tiles” with a number up to {COMMON.MAX_FILES}, or fewer samples.")
    # a typed sample name keeps the tile-size suffix of a split sample (one sample per scale in the results)
    label_of = {n: (label + g["suffix"] if label is not None else g["label"]) for g in chosen for n in g["names"]}
    # a typed value applies to every tile of its sample; a range sample left empty keeps each tile's own value
    nm_of = {n: (g["nm_in"] if g["nm_in"] is not None else g["nm_of"][n]) for g in chosen for n in g["names"]}
    # recorded: a typed value that replaced several manifest values (a range), or manifest values of a sample where
    # some tiles had none
    overrides = [{"sample": label_of[g["names"][0]], "typed_nm_per_px": g["nm_in"],
                  "manifest_nm_per_px": nm_range_text(g["nm_min"], g["nm_max"])
                  + (f" ({g['nm_missing']} tile(s) without a value)" if g["nm_missing"] else ""),
                  "tiles": len(set(names) & set(g["names"]))}
                 for g in chosen if g["nm_in"] is not None and (g["nm_range"] or (g["nm_missing"] and g["nm_values"]))]
    tid = selection_key(user, src["session"], form["model"], [(n, label_of[n], repr(float(nm_of[n]))) for n in names])
    if not create_running(tid) and not recently_done(tid):
        check_active(cfg, user, exclude=tid)
    return {"tid": tid, "names": names, "label_of": label_of, "nm_of": nm_of, "overrides": overrides,
            "model": form["model"], "name": chosen_name(form)}


def session_run(cfg, user, src: dict, plan: dict, progress=None) -> Path:
    """Make the set from a plan (slow: the background task, or session_submit). Tiles are copied."""
    names, label_of, nm_of = plan["names"], plan["label_of"], plan["nm_of"]
    tmp = scratch(cfg, "sel")
    try:
        man = tmp / "MANIFEST.csv"
        with open(man, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["output_name", "sample_label", PS.NM_COLUMN])
            for n in names:
                w.writerow([n, label_of[n], repr(float(nm_of[n]))])
        ps_dir = create_and_start(cfg, src["dir"], user, plan["model"], names=names, manifest=man,
                                  session=src["session"], name=plan["name"], progress=progress)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    for o in plan["overrides"]:
        record_nm_override(cfg, user, ps_dir.name, o)
    return ps_dir




def session_submit(cfg, user, src: dict, form: dict, progress=None) -> Path:
    """session_plan then session_run in this thread (tests, scripts; the portal runs session_run in a task)."""
    return session_run(cfg, user, src, session_plan(cfg, user, src, form), progress)


_AUDIT_FAILED: dict = {}                 # ps id -> why its nm/px override record was not written (admin sees it)
_AUDIT_LOCK = threading.Lock()


def record_nm_override(cfg, user, ps_id: str, detail: dict) -> bool:
    """A person typed one nm/px for a sample whose tiles had other manifest values (a range, or some tiles with none):
    every tile of it got the typed value in images.csv. Recorded in the portal audit log
    (<state dir>/audit/portal.jsonl) as predict.nm_override. -> written. A failure does not undo the analysis (it
    already exists): it is logged to the console and shown to the admin on the set's page (audit_problem)."""
    state = getattr(cfg, "state_dir", None)
    try:
        if not state:
            raise OSError("the portal has no state folder configured")
        from portal import audit as PA
        PA.audit(state, user.get("id"), "predict.nm_override", ps_id, detail)
        return True
    except OSError as e:
        why = (f"nm/px override not written to the audit log: {type(e).__name__}: {e}; "
               f"{json.dumps(detail, ensure_ascii=False)}")
        print(f"[predict] {ps_id}: {why}", file=sys.stderr, flush=True)
        with _AUDIT_LOCK:
            _AUDIT_FAILED[ps_id] = why
            for k in list(_AUDIT_FAILED)[:max(0, len(_AUDIT_FAILED) - COMMON.CACHE_MAX)]:
                _AUDIT_FAILED.pop(k, None)
        return False


def audit_problem(ps_id: str) -> str | None:
    with _AUDIT_LOCK:
        return _AUDIT_FAILED.get(ps_id)


def audit_event(cfg, user, action: str, ps_id: str, detail: dict) -> bool:
    """One line in the portal audit log; a failure is logged to the console and never undoes the action."""
    try:
        state = getattr(cfg, "state_dir", None)
        if not state:
            raise OSError("the portal has no state folder configured")
        from portal import audit as PA
        PA.audit(state, user.get("id"), action, ps_id, detail)
        return True
    except OSError as e:
        print(f"[predict] {ps_id}: {action} not written to the audit log: {type(e).__name__}: {e}",
              file=sys.stderr, flush=True)
        return False


_RENAME_LOCKS: dict = {}
_RENAME_LOCKS_LOCK = threading.Lock()


def rename_lock(ps: str) -> threading.Lock:
    with _RENAME_LOCKS_LOCK:
        return _RENAME_LOCKS.setdefault(ps, threading.Lock())


def rename_set(cfg, user, d: Path, raw_name) -> str | None:
    """Set (or, when empty, clear) the analysis name: read-modify-write of request.json under the set's lock,
    written atomically, then audited as predict.rename. -> the new name (None = back to the default).
    The caller has already checked that this person may see the set (owner or admin)."""
    new = PS.clean_name(raw_name)
    with rename_lock(d.name):
        req = _read_json(d / PS.REQUEST)
        if not isinstance(req, dict):
            raise Refusal("This analysis can’t be renamed right now. Try again later.", 503)
        old = req.get("name")
        if new:
            req["name"] = new
        else:
            req.pop("name", None)
        PS.write_json_atomic(d / PS.REQUEST, req)
    audit_event(cfg, user, "predict.rename", d.name, {"from": old if isinstance(old, str) else None, "to": new})
    return new


# ---------------------------------------------------------------- finished conversions as sources (owner and admin)
_SRC_CACHE: dict = {}                 # (tiles folder, status.json mtime_ns) -> check_folder() of a done conversion
_SRC_LOCK = threading.Lock()


def converted_sources(cfg, user, can_see) -> list:
    """The finished conversions this person may see (owner, admin) as Predict sources: [{key, label, dir, manifest,
    session (from the folder name), sample_column, problem, owner, conversion}]. The folder check is cached per
    status.json mtime (a done conversion's tiles no longer change)."""
    out = []
    for d, meta, st in list_conversions(cfg, user, can_see):
        if st.get("state") != "done":
            continue
        m = RAW_ID_RE.match(d.name)
        tiles = d / CONVERT_TILES
        man = tiles / PSRC.MANIFEST
        try:
            key = (str(tiles), (d / PS.STATUS).stat().st_mtime_ns)
        except OSError:
            continue
        with _SRC_LOCK:
            chk = _SRC_CACHE.get(key)
        if chk is None:
            chk = PSRC.check_folder(tiles)
            with _SRC_LOCK:
                _SRC_CACHE[key] = chk
                while len(_SRC_CACHE) > COMMON.CACHE_MAX:
                    _SRC_CACHE.pop(next(iter(_SRC_CACHE)))
        out.append({"key": PSRC.source_key(tiles),
                    "label": f"{PSRC.default_label(m.group(1))} · converted ({meta.get('recipe')} recipe, "
                             f"{when(meta.get('created'))})",
                    "dir": tiles, "manifest": man if man.is_file() else None, "session": m.group(1),
                    "sample_column": None, "problem": None if chk["compatible"] else chk["reason"],
                    "owner": meta.get("owner"), "conversion": d.name})
    return out
