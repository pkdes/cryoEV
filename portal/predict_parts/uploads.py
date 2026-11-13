"""
Predict, uploads: the multipart stream (no-JavaScript form), the upload sessions API (one file or piece per
request), declared files and raw pairing, start / piece / create, and the start-up scratch sweep.

Split out of portal/modules/predict.py (round 7, P7; no behaviour change); predict.py re-exports every name.
Names the tests patch are used from other parts only as <PART>.<name> (portal/predict_parts/__init__.py).

Agents (round 11 #65): an upload started by a write token's agent carries upload.json "owner_via" (the agent). Open
uploads are per (person, via, kind): a new one gives up only an earlier one with the same three, so an agent never
cancels its person's browser upload (or the reverse). A person has at most AGENT_UPLOADS open agent uploads (any
agent, any kind; 429 otherwise). Everything after the start (create, the background task, a retry) reads the agent
from upload.json, never from a request.
"""
from __future__ import annotations

import os
import re
import shutil
import threading
import time
from pathlib import Path

from starlette.concurrency import run_in_threadpool
from starlette.requests import Request

from jobs import predict_set as PS
from portal import upload_sessions as US

from portal.predict_parts.common import (IMAGE_EXT, MAX_FIELDS, MAX_FIELD_BYTES, NAME_BAD, PART, RAW_EXT,
                                         RAW_MAX_FILES, RAW_MAX_FILE_BYTES, RAW_MAX_TOTAL_BYTES, RAW_ROLES,
                                         RAW_SIDE_MAX_BYTES, Refusal, SCHEMA_UPLOAD, SCRATCH_MAX_AGE_S, UPLOAD_FILES,
                                         UPLOAD_META, UPLOAD_RE, WIN_RESERVED, WRITE_BATCH, _read_json, now_iso)
from portal.predict_parts.raw import (SIDECAR_EXT, check_gpu, file_problem, raw_fields, sidecar_images,
                                      sidecar_problems)
from portal.predict_parts.disk import Reservation
from portal.predict_parts.runner import (check_conv_cap, new_conversion, raw_root, tmp_dirs)
from portal.predict_parts.create import (_CREATES, _CREATE_LOCK, check_nm, chosen_name, create_and_start,
                                         create_running, scratch, start_create)
from portal.predict_parts import common as COMMON
from portal.predict_parts import runner as RUNNER
from portal.predict_parts import create as CREATE


# ---------------------------------------------------------------- upload (multipart, streamed to disk)
def safe_name(raw: str) -> str | None:
    name = re.split(r"[\\/]", raw or "")[-1].strip()
    if (not name or name in (".", "..") or name.startswith(".") or len(name) > 150 or NAME_BAD.search(name)
            or WIN_RESERVED.match(name) or name.endswith((" ", "."))):
        return None
    return name


class UploadSink:
    """Takes the parser's events (in order) and writes file parts to `dest` under their checked names; enforces
    the caps; re-checks the disk at each file start and every DISK_CHECK_BYTES; calls first_file(fields) when the
    first file part starts (the form puts the text fields first, so a refusal comes before any big file lands).
    Runs in a worker thread (file writes)."""

    def __init__(self, dest: Path, reservation: Reservation | None = None, first_file=None):
        self.dest, self.fields, self.files, self.seen = dest, {}, [], set()
        self.cur, self.fh, self.size, self.n_fields = None, None, 0, 0
        self.res, self.first_file, self.since_check = reservation, first_file, 0

    def apply(self, events):
        for ev in events:
            kind = ev[0]
            if kind == "part":
                _, name, filename = ev
                if filename is None:
                    self.n_fields += 1
                    if self.n_fields > MAX_FIELDS:
                        raise Refusal("Too many form fields.", 413)
                    self.cur = {"kind": "field", "name": name, "buf": bytearray()}
                elif filename == "":
                    self.cur = {"kind": "skip"}              # an empty file input
                else:
                    if name != "files":
                        raise Refusal("Unexpected file field.", 400)
                    if not self.files and self.first_file is not None:
                        self.first_file(self.fields)
                    if self.res is not None:
                        self.res.check()
                        self.since_check = 0
                    if len(self.files) >= COMMON.MAX_FILES:
                        raise Refusal(f"More than {COMMON.MAX_FILES} files: an analysis takes at most "
                                      f"{COMMON.MAX_FILES}. Nothing was kept.", 413)
                    safe = safe_name(filename)
                    if safe is None:
                        raise Refusal(f"File name not accepted: {filename[:80]!r}. Rename it (letters, digits, "
                                      f"- _ . and spaces) and upload again.", 400)
                    if safe.lower() in self.seen:
                        raise Refusal(f"Two files are named {safe!r}; each name must be unique.", 400)
                    self.seen.add(safe.lower())
                    self.files.append(safe)
                    self.fh = open(self.dest / safe, "xb")
                    self.size = 0
                    self.cur = {"kind": "file", "name": safe}
            elif kind == "data":
                c = self.cur or {}
                if c.get("kind") == "file":
                    self.size += len(ev[1])
                    if self.size > COMMON.MAX_FILE_BYTES:
                        raise Refusal(f"{c['name']} is over {COMMON.MAX_FILE_BYTES // 2 ** 20} MB. "
                                      f"Nothing was kept.", 413)
                    self.fh.write(ev[1])
                    if self.res is not None:
                        self.res.written(len(ev[1]))
                        self.since_check += len(ev[1])
                        if self.since_check >= COMMON.DISK_CHECK_BYTES:
                            self.res.check()
                            self.since_check = 0
                elif c.get("kind") == "field":
                    c["buf"] += ev[1]
                    if len(c["buf"]) > MAX_FIELD_BYTES:
                        raise Refusal("A form field is too long.", 413)
            elif kind == "end":
                c = self.cur or {}
                if c.get("kind") == "file":
                    self.fh.close()
                    self.fh = None
                elif c.get("kind") == "field":
                    self.fields.setdefault(c["name"], c["buf"].decode("utf-8", "replace"))
                self.cur = None

    def close(self):
        if self.fh is not None:
            self.fh.close()
            self.fh = None


async def receive_upload(request: Request, dest: Path, first_file=None) -> UploadSink:
    """Stream a multipart/form-data body into `dest` (python-multipart's push parser; file writes and checks in a
    worker thread). Raises Refusal on a bad body, a cap, low disk space, or first_file's refusal."""
    from python_multipart.multipart import parse_options_header
    ctype, params = parse_options_header(request.headers.get("content-type"))
    boundary = params.get(b"boundary")
    if ctype != b"multipart/form-data" or not boundary:
        raise Refusal("Expected a file upload (multipart/form-data).", 400)
    try:
        length = int(request.headers.get("content-length") or 0)
    except ValueError:
        length = 0
    limit = min(COMMON.MAX_FILES * (COMMON.MAX_FILE_BYTES + 4096) + MAX_FIELDS * (MAX_FIELD_BYTES + 1024),
                COMMON.NOJS_MAX_BYTES)
    if length > limit:
        raise Refusal(nojs_too_big(), 413)
    res = Reservation(dest, length)
    try:
        await run_in_threadpool(res.check)
        return await _stream(request, boundary, dest, res, limit, first_file)
    finally:
        res.release()


async def _stream(request: Request, boundary: bytes, dest: Path, res: Reservation, limit: int,
                  first_file) -> UploadSink:
    from python_multipart.multipart import MultipartParser, parse_options_header
    events, hdr = [], {"field": b"", "value": b"", "headers": {}}

    def on_part_begin():
        hdr["headers"] = {}

    def on_header_field(data, start, end):
        hdr["field"] += data[start:end]

    def on_header_value(data, start, end):
        hdr["value"] += data[start:end]

    def on_header_end():
        hdr["headers"][hdr["field"].lower()] = hdr["value"]
        hdr["field"], hdr["value"] = b"", b""

    def on_headers_finished():
        _, opts = parse_options_header(hdr["headers"].get(b"content-disposition"))
        name = opts.get(b"name", b"").decode("utf-8", "replace")
        fn = opts.get(b"filename")
        events.append(("part", name, None if fn is None else fn.decode("utf-8", "replace")))

    def on_part_data(data, start, end):
        events.append(("data", bytes(data[start:end])))

    def on_part_end():
        events.append(("end",))

    callbacks = {"on_part_begin": on_part_begin, "on_header_field": on_header_field,
                 "on_header_value": on_header_value, "on_header_end": on_header_end,
                 "on_headers_finished": on_headers_finished, "on_part_data": on_part_data, "on_part_end": on_part_end}
    parser = MultipartParser(boundary, callbacks)
    sink = UploadSink(dest, res, first_file)
    total = 0
    try:
        async for chunk in request.stream():
            total += len(chunk)
            if total > limit:
                raise Refusal(nojs_too_big(), 413)
            try:
                parser.write(chunk)
            except Exception as e:
                raise Refusal("The upload could not be read; try again.", 400, tech=str(e))
            if events:
                batch, events[:] = list(events), []
                await run_in_threadpool(sink.apply, batch)
        parser.finalize()
        if events:
            await run_in_threadpool(sink.apply, list(events))
    finally:
        sink.close()
    return sink


def upload_fields(cfg, user, f: dict, files=True) -> tuple:
    """The upload form's text fields -> (label, nm). Raises Refusal listing every problem. Called when the first
    file part arrives (files=True there: the files are coming) and again at the end."""
    problems = []
    if f.get("confirm") != "yes":
        raise Refusal("Tick “I confirm the pixel size”: nothing was started.")
    if not files:
        problems.append("Choose at least one image.")
    label = str(f.get("sample_label") or "").strip()
    if not label or len(label) > 80:
        problems.append("Sample name: 1-80 characters.")
    nm, err = check_nm(f.get("nm_per_px"), "Pixel size")
    if err:
        problems.append(err)
    m_err = CREATE.check_model(cfg, user, f.get("model"))
    if m_err:
        problems.append(m_err)
    if problems:
        raise Refusal(f"Please fix ({len(problems)}):\n" + "\n".join(problems))
    return label, nm


def upload_submit(cfg, user, dest: Path, sink: UploadSink) -> Path:
    label, nm = upload_fields(cfg, user, sink.fields, bool(sink.files))
    CREATE.check_active(cfg, user)
    return create_and_start(cfg, dest, user, sink.fields["model"], sample_label=label, nm=nm,
                            name=chosen_name(sink.fields), link=True)


def nojs_too_big() -> str:
    return (f"Without JavaScript, one upload is at most {COMMON.NOJS_MAX_BYTES // 2 ** 20} MB in total (the link to "
            f"the server caps one request at about 100 MB). Nothing was kept. Turn JavaScript on (the page then sends "
            f"the files one at a time), or upload fewer files.")


# ---------------------------------------------------------------- upload sessions (one file or piece per request)
_UP_RES: dict = {}                      # upload id -> Reservation (the bytes it may still write)
_UP_RES_LOCK = threading.Lock()
_UP_ALOCKS: dict = {}                   # upload id -> asyncio.Lock (one request at a time per session)


def upload_dir(cfg, uid) -> Path | None:
    if not isinstance(uid, str) or not UPLOAD_RE.match(uid):
        return None
    for tmp in tmp_dirs(cfg):
        d = tmp / uid
        if (d / UPLOAD_META).is_file():
            return d
    return None


def load_upload(cfg, user, uid) -> tuple:
    """-> (folder, upload.json) when it exists and is this person's own (an admin too: uploads are personal)."""
    d = upload_dir(cfg, uid)
    meta = _read_json(d / UPLOAD_META) if d else None
    if not isinstance(meta, dict) or meta.get("owner") != user.get("id"):
        return None, None
    return d, meta


def upload_reservation(uid, d: Path, meta: dict) -> Reservation:
    """The session's disk hold (made again after a portal restart from what is still to come)."""
    with _UP_RES_LOCK:
        res = _UP_RES.get(uid)
        if res is None:
            res = Reservation(d, max(0, sum(meta["declared"].values()) - sum(upload_received(d, meta).values())))
            _UP_RES[uid] = res
        return res


def drop_upload(d: Path, uid: str):
    with _UP_RES_LOCK:
        res = _UP_RES.pop(uid, None)
    if res is not None:
        res.release()
    _UP_ALOCKS.pop(uid, None)
    shutil.rmtree(d, ignore_errors=True)


def upload_received(d: Path, meta: dict) -> dict:
    out = {}
    done = set(meta.get("done") or [])
    for n, size in meta["declared"].items():
        if n in done:
            out[n] = size
        else:
            p = d / UPLOAD_FILES / (n + PART)
            try:
                out[n] = p.stat().st_size
            except OSError:
                out[n] = 0
    return out


def upload_state(d: Path, meta: dict) -> dict:
    return {"schema": SCHEMA_UPLOAD, "upload": meta["id"], "kind": meta["kind"], "chunk": COMMON.CHUNK_BYTES,
            "received": upload_received(d, meta), "done": list(meta.get("done") or []),
            "skipped": list(meta.get("skipped") or []),
            "total": sum(meta["declared"].values())}


def skipped_names(meta: dict) -> set:
    return {x.get("name") for x in meta.get("skipped") or [] if isinstance(x, dict)}


def skip_raw_file(d: Path, meta: dict, name: str, why: str) -> None:
    """Leave one raw upload file out (it must never reach the converter): its staged copy (or partial) deleted, the
    name in skipped {name, why} and in done. An .mrc.mdoc takes its .mrc with it. Callers write meta."""
    why = why.replace(f"{name}: ", "")               # the checks start with the file name: listed once
    names = [(name, why)]
    lo = name.lower()
    if lo.endswith(".mrc.mdoc"):
        mrc = next((n for n in meta["declared"] if n.lower() == lo[:-len(".mdoc")]), None)
        if mrc:
            names.append((mrc, f"its {name} was left out"))
    elif lo.endswith(".mrc"):
        side = next((n for n in meta["declared"] if n.lower() == lo + ".mdoc"), None)
        if side:
            names.append((side, f"its {name} was left out"))
    have = skipped_names(meta)
    for n, w in names:
        for p in (d / UPLOAD_FILES / n, d / UPLOAD_FILES / (n + PART)):
            try:
                p.unlink()
            except FileNotFoundError:
                pass
        if n not in have:
            meta["skipped"] = list(meta.get("skipped") or []) + [{"name": n, "why": w[:300]}]
            have.add(n)
        if n not in (meta.get("done") or []):
            meta["done"] = list(meta.get("done") or []) + [n]


def is_sidecar(name: str) -> bool:
    return name.lower().endswith((".mdoc", ".idoc", ".csv"))


def raw_pairing(names) -> list:
    """Problems with a raw file list: each .mrc with its .mrc.mdoc, .tif tiles with an .idoc, one sample key."""
    low = {n.lower(): n for n in names}
    problems = []
    for lo, n in low.items():
        if lo.endswith(".mrc") and lo + ".mdoc" not in low:
            problems.append(f"{n}: its {n}.mdoc is missing (the pixel size and tile positions come from it)")
        if lo.endswith(".mrc.mdoc") and lo[:-len(".mdoc")] not in low:
            problems.append(f"{n}: its {n[:-len('.mdoc')]} is missing")
    idocs = [n for lo, n in low.items() if lo.endswith(".idoc")]
    tifs = [n for lo, n in low.items() if lo.endswith((".tif", ".tiff"))]
    if tifs and not idocs:
        problems.append("TIFF tiles need the montage’s .idoc file")
    if idocs and not tifs:
        problems.append("an .idoc needs its .tif tiles")
    if sum(1 for lo in low if lo.endswith(".csv")) > 1:
        problems.append("at most one sample key (.csv)")
    if not any(lo.endswith((".mrc.mdoc", ".idoc")) for lo in low) and not problems:
        problems.append("no montage: add an .mrc with its .mrc.mdoc, or an .idoc with its .tif tiles")
    return problems


def declared_files(kind: str, files, skipped=None) -> dict:
    """The file list a page declares at the start -> {name: size}. Raises Refusal on a cap (413) or a bad list.
    Raw: atlas files are left out of it (raw_leave_out), appended to `skipped` when given."""
    if not isinstance(files, list) or not files:
        raise Refusal("Choose at least one image." if kind == "tiles" else "Choose the raw files.")
    cap_n = COMMON.MAX_FILES if kind == "tiles" else RAW_MAX_FILES
    if len(files) > cap_n:
        raise Refusal(f"More than {cap_n} files: " + (f"an analysis takes at most {cap_n}." if kind == "tiles" else
                                                       f"convert at most {cap_n} files at once.") +
                      " Nothing was kept.", 413)
    out, seen = {}, set()
    for it in files:
        name = str((it or {}).get("name") or "") if isinstance(it, dict) else ""
        size = (it or {}).get("size") if isinstance(it, dict) else None
        safe = safe_name(name)
        if safe is None or safe != name:
            raise Refusal(f"File name not accepted: {name[:80]!r}. Rename it (letters, digits, - _ . and spaces) and "
                          f"upload again.")
        if safe.lower() in seen:
            raise Refusal(f"Two files are named {safe!r}; each name must be unique.")
        seen.add(safe.lower())
        if safe.lower().endswith(PART):                  # never the name of another file's partial copy
            raise Refusal(f"File name not accepted: {name[:80]!r} (it ends in {PART}). Rename it and upload again.")
        if kind == "tiles" and not safe.lower().endswith(IMAGE_EXT):
            raise Refusal(f"{safe}: not a PNG or JPEG file name. Upload processed .png / .jpg tiles.")
        if kind == "raw" and not safe.lower().endswith(RAW_EXT):
            raise Refusal(f"{safe}: not a raw montage file. Upload .mrc files with their .mrc.mdoc, or an .idoc with "
                          f"its .tif tiles (and, if you have it, the session’s sample key .csv).")
        if not isinstance(size, int) or isinstance(size, bool) or size < 1:
            raise Refusal(f"{safe} is empty.")
        cap = (COMMON.MAX_FILE_BYTES if kind == "tiles" else RAW_SIDE_MAX_BYTES if is_sidecar(safe)
               else RAW_MAX_FILE_BYTES)
        if size > cap:
            raise Refusal(f"{safe} is over {cap // 2 ** 20} MB. Nothing was kept.", 413)
        out[safe] = size
    if kind == "tiles" and sum(out.values()) > COMMON.TILES_MAX_TOTAL_BYTES:
        raise Refusal(f"Over {COMMON.TILES_MAX_TOTAL_BYTES // 2 ** 30} GB of images in one upload. Upload fewer images "
                      f"at once. Nothing was kept.", 413)
    if kind == "raw":
        if sum(out.values()) > RAW_MAX_TOTAL_BYTES:
            raise Refusal(f"Over {RAW_MAX_TOTAL_BYTES // 2 ** 30} GB in one conversion. Convert fewer montages at "
                          f"once.", 413)
        out, left = raw_leave_out(out)
        if skipped is not None:
            skipped += left
        problems = raw_pairing(out)
        if problems:
            if left:
                problems.append(f"{len(left)} atlas file(s) were left out (whole-grid overviews are never converted)")
            raise Refusal(f"Please fix ({len(problems)}):\n" + "\n".join(problems))
    return out


ATLAS_WHY = "a whole-grid atlas (overview): never converted"


def raw_leave_out(decl: dict) -> tuple:
    """({name: size} kept, [{name, why}] left out) of a raw file list: names with "atlas" in them (whole-grid
    overviews; cryoem_core's discovery never converts them), and an .mrc whose .mrc.mdoc is left out."""
    left = {n: ATLAS_WHY for n in decl if "atlas" in n.lower()}
    low = {n.lower(): n for n in decl}
    for lo, n in low.items():
        side = low.get(lo + ".mdoc")
        if lo.endswith(".mrc") and side in left and n not in left:
            left[n] = f"its {side} was left out"
    return ({n: s for n, s in decl.items() if n not in left},
            [{"name": n, "why": w} for n, w in left.items()])


def clean_fields(raw) -> dict:
    if raw is None:
        return {}
    if not isinstance(raw, dict) or len(raw) > MAX_FIELDS:
        raise Refusal("Too many form fields.", 413)
    out = {}
    for k, v in raw.items():
        if not isinstance(k, str) or not isinstance(v, (str, int, float)) or isinstance(v, bool):
            continue
        v = str(v)
        if len(k) > 64 or len(v.encode("utf-8")) > MAX_FIELD_BYTES:
            raise Refusal("A form field is too long.", 413)
        out[k] = v
    return out


NOJS_REFILL_BYTES = 1024 ** 2                 # a no-JavaScript raw post this small is read to refill the form
SAMPLE_KEY_NAME = "sample_key.csv"            # the upload form's samples, written into input/ before conversion


def check_sample_source(raw: dict, names) -> None:
    """The sample table and an uploaded sample key .csv are either / or (the converter reads one key)."""
    if raw.get("samples") and any(str(n).lower().endswith(".csv") for n in names):
        raise Refusal("Give the samples either in the table or as a sample_key.csv file, not both. Nothing was kept.")


def write_sample_key(files_dir: Path, raw: dict) -> None:
    """The form's {grid: label} -> <files>/sample_key.csv (`#,Label`, as cryoem_core's load_sample_key reads it).
    new_conversion moves it into input/ and passes it to the converter (its one .csv)."""
    if raw.get("samples"):
        tmp = files_dir / (SAMPLE_KEY_NAME + PART)
        tmp.write_text(US.sample_key_text(raw["samples"]), encoding="ascii", newline="")
        os.replace(tmp, files_dir / SAMPLE_KEY_NAME)


def check_upload_fields(cfg, user, kind, fields, has_files=True, via=None) -> dict:
    """Tiles: the label / nm / tick / model checks of the plain form (+ the basic user's and every agent's limit).
    Raw: raw_fields."""
    if kind == "tiles":
        upload_fields(cfg, user, fields, has_files)
        CREATE.check_active(cfg, user, via=via)
        return {}
    return raw_fields(cfg, user, fields)


AGENT_UPLOADS = 1                       # open upload sessions an agent (all of one person's agents together) may have
_START_LOCK = threading.Lock()           # upload_start's check-then-create (one portal process per door)


def open_uploads(cfg) -> list:
    """[(folder, upload.json)] of every upload session still on disk."""
    out = []
    for tmp in tmp_dirs(cfg):
        try:
            names = [p for p in tmp.iterdir() if UPLOAD_RE.match(p.name)]
        except OSError:
            continue
        for p in names:
            m = _read_json(p / UPLOAD_META)
            if isinstance(m, dict):
                out.append((p, m))
    return out


def upload_start(cfg, user, body, via=None) -> dict:
    """Open (or resume) an upload session. Every check that can be made before a byte is sent is made here.
    via: the agent acting for `user` (from the request; None for a browser): stored as upload.json owner_via."""
    if not isinstance(body, dict):
        raise Refusal("Bad request.", 400)
    kind = body.get("kind")
    if kind not in ("tiles", "raw"):
        raise Refusal("Unknown upload.", 400)
    if kind == "raw" and user.get("role") not in RAW_ROLES:
        raise Refusal("Converting raw images is for advanced users and the admin.", 403)
    sweep_scratch(cfg)
    fields = clean_fields(body.get("fields"))
    if kind == "tiles" and fields.get("confirm") != "yes":       # the plain form's first refusal, before the list
        raise Refusal("Tick “I confirm the pixel size”: nothing was started.")
    left = []
    decl = declared_files(kind, body.get("files"), left)
    raw = check_upload_fields(cfg, user, kind, fields, via=via)
    if raw:
        check_sample_source(raw, decl)
        check_conv_cap(cfg, user)
        check_gpu(raw)
    via = via or None
    resume = body.get("resume")
    if isinstance(resume, str):
        d, meta = load_upload(cfg, user, resume)
        if meta and meta.get("kind") == kind and meta.get("declared") == decl and \
                (meta.get("owner_via") or None) == via:
            meta["fields"] = fields
            PS.write_json_atomic(d / UPLOAD_META, meta)
            return upload_state(d, meta)
    # one start at a time per portal process: the scan, the drops, the agent-upload check and the new session's
    # upload.json are one step (concurrent starts by two agents could otherwise both pass the check)
    with _START_LOCK:
        # one open session per person, agent (None: the browser) and kind: an earlier one (other files) is given up.
        # Another agent's, or the browser's, is never touched (round 11 #65: the key was (person, kind) before)
        mine = [(p, m) for p, m in open_uploads(cfg) if m.get("owner") == user.get("id")]
        for p, m in mine:
            if m.get("kind") == kind and (m.get("owner_via") or None) == via and not create_running(p.name):
                drop_upload(p, p.name)
        if via:
            busy = [p for p, m in mine if m.get("owner_via") and p.is_dir()]
            if len(busy) >= AGENT_UPLOADS:
                raise Refusal(f"Your agents already have {len(busy)} upload open (at most {AGENT_UPLOADS} at a time). "
                              f"Finish it (create) or cancel it first.", 429)
        d = scratch(cfg, "us", raw=(kind == "raw"))
        uid = d.name
        res = Reservation(d, sum(decl.values()))
        try:
            res.check()
            (d / UPLOAD_FILES).mkdir()
            # left-out files count as done (an older page sends only what is not done) and are never received
            meta = {"schema": SCHEMA_UPLOAD, "id": uid, "owner": user["id"], "kind": kind, "fields": fields,
                    "declared": decl, "done": [x["name"] for x in left], "skipped": left, "created": now_iso()}
            if via:
                meta["owner_via"] = via
            PS.write_json_atomic(d / UPLOAD_META, meta)
        except BaseException:
            res.release()
            shutil.rmtree(d, ignore_errors=True)
            raise
        with _UP_RES_LOCK:
            _UP_RES[uid] = res
        return upload_state(d, meta)


class Resync(Exception):
    """The piece does not start where the file stands: answer 409 with the session (the page resumes there)."""


async def receive_piece(request: Request, d: Path, uid: str) -> dict:
    """One piece of one file: ?name=&offset=, body = the bytes. Appends to <name>.part; the complete file is renamed
    and content-checked. Raises Refusal (the caller drops the session only when drop is set or the disk is full:
    a size or content violation; a bad request leaves it as it is), Resync, ClientDisconnect."""
    meta = await run_in_threadpool(_read_json, d / UPLOAD_META)
    if not isinstance(meta, dict):
        raise Refusal("This upload is gone. Start again.", 404)
    name = request.query_params.get("name") or ""
    try:
        offset = int(request.query_params.get("offset") or "")
    except ValueError:
        raise Refusal("Bad request (offset).", 400)
    decl = meta["declared"]
    if name not in decl:
        raise Refusal(f"{name[:80]!r} is not one of the files of this upload.", 400)
    if name in (meta.get("done") or []):
        return upload_state(d, meta)                     # already complete: answering again is harmless
    size = decl[name]
    part = d / UPLOAD_FILES / (name + PART)
    have = part.stat().st_size if part.is_file() else 0
    if offset != have:
        raise Resync()
    try:
        length = int(request.headers.get("content-length") or "")
    except ValueError:
        raise Refusal("Each piece must say its length (Content-Length).", 411)
    if length < 1 or length > COMMON.CHUNK_BYTES:
        raise Refusal(f"One request carries at most {COMMON.CHUNK_BYTES // 2 ** 20} MB. Nothing was kept.", 413,
                      drop=True)
    if offset + length > size:
        raise Refusal(f"{name}: more bytes than its size. Nothing was kept.", 400, drop=True)
    res = upload_reservation(uid, d, meta)
    await run_in_threadpool(res.check)
    fh = await run_in_threadpool(open, part, "ab")
    got, buf, since = 0, bytearray(), 0
    try:
        async for piece in request.stream():
            if not piece:
                continue
            got += len(piece)
            if got > length:
                raise Refusal("A piece was longer than it said. Nothing was kept.", 400, drop=True)
            buf += piece
            if len(buf) >= WRITE_BATCH:
                await run_in_threadpool(fh.write, bytes(buf))
                res.written(len(buf))
                since += len(buf)
                buf.clear()
                if since >= COMMON.DISK_CHECK_BYTES:
                    await run_in_threadpool(res.check)
                    since = 0
        if buf:
            await run_in_threadpool(fh.write, bytes(buf))
            res.written(len(buf))
    finally:
        await run_in_threadpool(fh.close)
    if got != length:
        raise Resync()                                   # cut short: the page asks again from what arrived
    try:
        os.utime(d / UPLOAD_META)                        # last activity (the sweep's clock)
    except OSError:
        pass
    if offset + length == size:
        final = d / UPLOAD_FILES / name
        await run_in_threadpool(os.replace, part, final)
        problem = await run_in_threadpool(file_problem, meta["kind"], final)
        if problem and meta["kind"] == "raw":
            # one odd raw file (an atlas mdoc, a sidecar naming a path) is left out, not the whole upload: deleted
            # here, so the converter never sees it; upload_create refuses if nothing convertible is left
            await run_in_threadpool(skip_raw_file, d, meta, name, problem)
            await run_in_threadpool(PS.write_json_atomic, d / UPLOAD_META, meta)
            return upload_state(d, meta)
        if problem:
            raise Refusal(f"{problem}. Nothing was kept.", 400, drop=True)
        meta["done"] = list(meta.get("done") or []) + [name]
        await run_in_threadpool(PS.write_json_atomic, d / UPLOAD_META, meta)
    return upload_state(d, meta)


def upload_create(cfg, user, d: Path, meta: dict) -> str:
    """Every declared file is there -> the prediction set (tiles) or the conversion (raw). -> where to go next.
    Refusal 409 (keep the session) when files are missing."""
    missing = [n for n in meta["declared"] if n not in set(meta.get("done") or [])]
    if missing:
        raise Refusal(f"{len(missing)} file(s) not sent yet ({', '.join(missing[:3])}"
                      f"{', …' if len(missing) > 3 else ''}). Press the button again to send them.", 409)
    fields = meta.get("fields") or {}
    via = meta.get("owner_via") or None                  # the agent that started the upload (data, not the request)
    if meta["kind"] == "tiles":
        uid = meta["id"]
        if not create_running(uid):                      # the same upload pressed again: the same task, no new check
            label, nm = upload_fields(cfg, user, fields, True)
            CREATE.check_active(cfg, user, exclude=uid, via=via)
            name = chosen_name(fields)

            def run(progress):
                # hard links: the staged names are removed after, the set keeps its own; a failure leaves the upload
                try:
                    ps_dir = create_and_start(cfg, d / UPLOAD_FILES, user, fields["model"], sample_label=label,
                                              nm=nm, name=name, link=True, progress=progress, via=via)
                except BaseException:
                    try:                             # kept for the stated hour from now, not from the first request
                        os.utime(d / UPLOAD_META)
                    except OSError:
                        pass
                    raise
                drop_upload(d, uid)
                return ps_dir
            start_create(uid, user["id"], run, back="/predict/new/upload/", upload=uid)
        return f"/predict/preparing/{uid}/"
    raw = raw_fields(cfg, user, fields)
    check_sample_source(raw, meta["declared"])
    check_conv_cap(cfg, user)
    # every sidecar's [Image = ...]: a bare name, and one of the uploaded files (else cryoem_core would open whatever
    # path it names on the server: an absolute path, a UNC share, ../). One that fails is left out and DELETED here,
    # before the conversion, so the converter never reads it; the rest still converts.
    files = d / UPLOAD_FILES
    sent = sorted(n for n in meta["declared"] if n not in skipped_names(meta) and (files / n).is_file())
    for n in sent:
        if n.lower().endswith(SIDECAR_EXT) and (files / n).is_file():
            bad = sidecar_problems(files / n, set(meta["declared"]))
            if bad:
                skip_raw_file(d, meta, n, "; ".join(bad))
                continue
            gone = [t for t in sidecar_images(files / n) if t in skipped_names(meta)]
            if gone:                                 # convert_raw refuses a montage with tiles missing: left out
                skip_raw_file(d, meta, n, f"its tile {gone[0]} was left out"
                              + (f" (and {len(gone) - 1} more)" if len(gone) > 1 else ""))
    named = set()                                    # TIFF tiles go with a surviving .idoc, or not at all
    for n in sent:
        if n.lower().endswith(".idoc") and (files / n).is_file():
            named |= set(sidecar_images(files / n))
    for n in sent:
        if n.lower().endswith((".tif", ".tiff")) and (files / n).is_file() and n not in named:
            skip_raw_file(d, meta, n, "its .idoc was left out")
    for n in sent:                                   # an MRC stack and its .mrc.mdoc go together
        lo = n.lower()
        if not (files / n).is_file():
            continue
        if lo.endswith(".mrc") and not any(p.lower() == lo + ".mdoc" and (files / p).is_file() for p in sent):
            skip_raw_file(d, meta, n, "its .mrc.mdoc was left out")
        elif lo.endswith(".mrc.mdoc") and not any(p.lower() == lo[:-len(".mdoc")] and (files / p).is_file()
                                                  for p in sent):
            skip_raw_file(d, meta, n, "its MRC stack was left out")
    left = [x for x in meta.get("skipped") or [] if isinstance(x, dict)]
    present = {p.name for p in files.iterdir() if p.is_file()} if files.is_dir() else set()
    low = {n.lower() for n in present}
    montages = [n for n in present if n.lower().endswith(".idoc")
                or (n.lower().endswith(".mrc.mdoc") and n.lower()[:-len(".mdoc")] in low)]
    PS.write_json_atomic(d / UPLOAD_META, meta)
    if not montages:
        why = [f"{x.get('name')}: {x.get('why')}" for x in left]
        raise Refusal(f"Not converted: no montage is left to convert ({len(left)} file(s) left out):\n"
                      + "\n".join(why[:12]) + ("\n…" if len(why) > 12 else ""))
    nbytes = sum(s for n, s in meta["declared"].items() if n in present)
    RUNNER.check_disk_for(raw_root(cfg), nbytes)
    check_gpu(raw)
    write_sample_key(files, raw)                     # before conversion: labels are baked into tile names
    conv = new_conversion(cfg, user, raw, upload_files=files,
                          source={"kind": "upload", "files": [n for n in meta["declared"] if n in present]},
                          nbytes=nbytes, skipped=left, made_via=via)
    return f"/predict/convert/{conv.name}/"


# ---------------------------------------------------------------- the start-up sweep
def sweep_scratch(cfg, max_age_s=SCRATCH_MAX_AGE_S) -> int:
    """Remove <root>/.portal/* left by a portal that stopped mid-upload, and upload sessions never finished (no
    "create"), older than max_age_s (a session: since its last request; its disk hold goes too). -> how many."""
    def age(p: Path) -> float:
        t = p.stat().st_mtime
        try:
            t = max(t, (p / UPLOAD_META).stat().st_mtime)
        except OSError:
            pass
        return time.time() - t
    old = []
    for tmp in tmp_dirs(cfg):                        # the predict root's .portal and the raw root's
        try:
            old += [p for p in tmp.iterdir() if p.is_dir() and p.name.startswith(("up_", "sel_", "us_"))
                    and age(p) > max_age_s]
        except Exception:                            # no root configured or not there yet: nothing to sweep
            continue
    try:                                             # a set half built when the portal stopped (no create is running)
        with _CREATE_LOCK:
            busy = any(t["state"] == "running" for t in _CREATES.values())
        inc = COMMON.predict_root(cfg) / PS.INCOMING
        if not busy and inc.is_dir():
            for p in inc.iterdir():
                if time.time() - p.stat().st_mtime > max_age_s:
                    shutil.rmtree(p, ignore_errors=True)
                    old.append(p)
    except Exception:                                # no root configured or not there yet
        pass
    for p in old:
        if UPLOAD_RE.match(p.name):
            if not create_running(p.name):
                drop_upload(p, p.name)
        elif p.exists():
            shutil.rmtree(p, ignore_errors=True)
    return len(old)
