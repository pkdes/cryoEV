"""Backfill importers: existing artifacts -> registry records (docs/TRACKING_CONTRACT.md mapping).

Scanning is validate.Scan's (subclassed only to keep per-session MANIFEST rows). Each record is appended
only when it is new or differs from the current one (ignoring logged / recorded_by), so a rerun appends 0.
Image keys: legacy keys (pull / tracking CSVs) stay; session tiles only in a MANIFEST are `<session>__<key>`;
generic upload copies (`blind_NN`) are `<session>_<key>` with `copy_of` when a *selection_PRIVATE.csv maps it.
An existing image key that arrives with a different session, or a stem claimed by a second Roboflow id, is
reported and not written. Session 20261012 is counted (`excluded`) and gets no records. Reads only, except
the per-dataset image lists it writes next to the registry (`<root>/lists/<dataset>.tsv`, only when changed).

v2 (jobs and inputs, TRACKING_CONTRACT.md addendum):
- dataset membership = the split file's `split` column (NOT `source_split`, NOT Roboflow's split), without
  `DROPPED-*` rows -> `image_list` {file, sha256}, `counts_trained` (all rows, incl. excluded sessions) and
  `counts_listed` (rows whose image is in the registry). No data -> null ("unknown"), never 0.
- `inputs` on label_version (base, edits), dataset (source), model (train_data, init_weights).
- a dataset built from a label version (data_utils/dataset_from_lv.py: `provenance.json` in its folder, or a
  registered record with source.kind label_version) keeps its own record and list; only frozen -> trained is derived.
- one revise `job/1` per top-level review package (no `_partN`, no SANDBOX): progress = edit records with
  `edit.reviewed` true / n_images; status in_progress if any record has decisions or reviewed, else ready.
  A package's `applied` status never makes its job done (only an explicit set-status line does).
- a run folder with its own `run.json` (run_experiments.py, 2026-11-02) gives status (finished; error/oom ->
  failed, with a note), job_bundle and config as written; older run folders are derived as before.
- export membership (2026-11-05, additive): each roboflow_export gets `image_list` {file, sha256, method, n}
  (`<root>/lists/exports/<slug>_<sha8 of id>.tsv`, `stem\tsplit`), counts_listed and not_listed, from its own COCO
  files (exact; excluded-session images counted, not listed), else from tracking/v2 `in: <export>` (tracked images
  only; method says so), else image_list null + image_list_gap. Image ids: the export file name's key, else the
  image whose rf_name is the COCO `extra.name` (a name two images claim is not used). The portal's Lineage follows
  it from an export to its images' grids / sessions.
- list files are written with their record's line, under the registry lock (append files=, replace_files=True), so
  a refused line leaves no list behind; a list whose unchanged record already names it is only repaired if missing
  or different.

Raw imaging (2026-11-05, additive; nothing in Cryo-image-processing is written):
- grid: imaged_grids_master.csv fields (sample_label, project, prep_date, grid_type, replicate, glow_discharge,
  storage, n_montages = sum over the grid's rows, notes / flags prefixed by session when a grid has several rows).
- session: acquisition summary of the session's raw .mdoc/.idoc files (`2. Raw images/`, or RAW_LAYOUT for the
  five older sessions), parsed by the canonical `processing scripts/cryoem_core/metadata.py` parse_metadata (loaded
  by path, no bytecode written). instrument = scope model name only (no serial, no site). Atlas files count in
  n_mdoc, voltage and dates, not in the per-image stats; images below IMAGING_MIN_MAG (low-mag / search maps)
  are left out too and shown as n_other / other_magnification. Parsed summaries are cached per file (mtime + size, and the
  parser's sha256) in `<registry>/cache/mdoc_summaries.json`: derived, safe to delete. Plus local_folder and
  sample_labels (sample_key.csv). No raw folder / no parser -> `acquisition_gap` says why.
- image: montage and source_file from the session MANIFEST row. A `YYYYMMDD-N` session's master rows are the ones
  whose key_source names that folder (validate.master_folder); they belong to that session only.

Upload sessions (round 7, portal/upload_sessions.py): the portal appends session lines for raw uploads (same id as
a lab session of that date) with the upload-owned keys UPLOAD_KEYS (`conversions`, `details`). When this importer
rebuilds a session record it carries them forward from the current line, as it carries `status_by`; everything else
on the record stays the lab's. write() holds <registry>/.upload_session.lock (UPLOAD_LOCK, the portal's
read-merge-append lock) from reading the current records to its last append, so a portal registration can't land in
between and be dropped.

Attribution (round 9): `via` (the agent that ran the import, from `registry.py import` under $CRYOEV_VIA) is set on
the lines this run appends and is IGNOREd by the no-change check, like logged / recorded_by: a record that differs
only in who ran the import is unchanged, so an agent's import followed by anyone's import appends 0 (and the other
way round). Records carried forward from the current line drop it (registry.append drops it too). The production
line restated from PRODUCTION is the user's recorded decision: it is never stamped with via. `status_via` (the agent
that set an explicit status) is carried with `status_by`, so an agent-set status keeps its "claude for pujan".
"""
import hashlib
import importlib.util
import json
import re
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from annotation.review_groups import is_record_file
from registry import ids
from registry import store as registry
from registry.validate import (EXCLUDED_SESSIONS, REPO, Scan, find_sibling, lv_provenance, read_csv, read_datajs,
                               read_json, read_jsonl, sha256_file)
from registry.validate import config  # data_utils.config

PROCESSING = config.processing_root()   # optional: None when not found
GENERIC = re.compile(r"^(blind|label|labeling)_\d+_[A-Za-z0-9]+$")
SUPERSEDED = {"lv_20261030_0846": "lv_20261030_0909"}          # old -> new (SUPERSEDED.md)
LEGACY_MODELS = {"allLayer_run14_20260713": "v0_alllayer_y8n_20260713",   # legacy id -> canonical id
                 "yifei_202512": "v0_yifei_y8n_20251201"}                 # (day unknown: 01)
PRODUCTION = ("v5_evnonev_y8m_20261029", "pujan")
IGNORE = {"logged", "recorded_by", "via"}     # who ran the import is not a change (round 9)
DROPPED = "DROPPED"                      # split_assignment.csv `split` values like DROPPED-duplicate: not trained
SPLIT_ORDER = {"train": 0, "valid": 1, "test": 2}
LISTS = "lists"                          # <registry>/lists/<dataset>.tsv  (stem \t split)
APP_PORT = 8766                          # review app port for the real review (NEXT_ITERATION.md)
RAW_DIR = "2. Raw images"                # <session>/2. Raw images/**/*.mdoc|*.idoc (current layout)
NOTES_DIR = "1. Notes + reports"
PARSER_REL = Path("processing scripts") / "cryoem_core" / "metadata.py"
MDOC_CACHE = Path("cache") / "mdoc_summaries.json"
MDOC_CACHE_SCHEMA = "mdoc_cache/2"       # bump when file_summary() changes: older entries are then re-parsed
ATLAS = re.compile(r"atlas", re.IGNORECASE)
IMAGING_MIN_MAG = 5000                   # below: low-mag (62-210x) / search (2,600x) maps, not imaging
# Older sessions keep their raw .mdoc/.idoc elsewhere; each entry matches the session's processing config
# (processing scripts/datasets/process_<id>.py; test_import_raw greps them so this can't drift):
#   dir: folder under the session ("" = session root), recursive: search below it, subdirs: also these
#   direct subfolders (non-recursive), skip: folder names left out.
RAW_LAYOUT = {
    "20260727": {"dir": "raw", "recursive": True},
    "20260804": {"dir": "1. Raw-mrc", "recursive": True, "skip": ("_excluded",)},   # local copy (SOURCE.md)
    "20260812": {"dir": "", "recursive": False},                                     # FlatGridRegionLayout
    "20260817": {"dir": "", "recursive": False, "subdirs": "grid*"},                 # GridSubfolderLayout
    "20260827": {"dir": "raw images", "recursive": True},                            # raw_subdir='raw images'
}
DEFAULT_LAYOUT = {"dir": RAW_DIR, "recursive": True}
GRID_FIELDS = ("sample_label", "project", "prep_date", "grid_type", "replicate", "glow_discharge", "storage")
HEADER_TIME = re.compile(r"\s+\d{1,2}-[A-Za-z]{3}-\d{2,4}\s+\d{1,2}:\d{2}(:\d{2})?\s*$")
UPLOAD_KEYS = ("conversions", "details")  # session keys the portal owns (= portal/upload_sessions.py UPLOAD_KEYS)
UPLOAD_LOCK, UPLOAD_LOCK_WAIT = registry.UPLOAD_LOCK, 60.0   # = portal/upload_sessions.py LOCK


def split_membership(csv_path):
    """split_assignment.csv -> [(image key, split, batch)] as TRAINED: the `split` column (not `source_split`),
    `DROPPED-*` rows left out."""
    out = []
    for r in read_csv(csv_path):
        split = (r.get("split") or "").strip()
        if not split or split.upper().startswith(DROPPED) or not r.get("image"):
            continue
        out.append((ids.image_key(r["image"]), split, r.get("batch") or ""))
    return out


def list_text(pairs):
    """{(stem, split)} -> sorted `stem\tsplit` lines (split order, then stem)."""
    rows = sorted(set(pairs), key=lambda x: (SPLIT_ORDER.get(x[1], 9), x[1], x[0]))
    return "".join(f"{stem}\t{split}\n" for stem, split in rows)


def export_list_name(export_id):
    """Export id -> its image list file, `lists/exports/<slug>_<sha256(id)[:8]>.tsv` (slug: runs of other characters
    -> `_`). It depends on the id alone, so a later export never renames an existing file, and the hash keeps two ids
    with one slug apart."""
    slug = re.sub(r"[^A-Za-z0-9]+", "_", export_id).strip("_") or "x"
    return f"{LISTS}/exports/{slug}_{hashlib.sha256(export_id.encode('utf-8')).hexdigest()[:8]}.tsv"


def split_counts(pairs):
    c = Counter(split for _, split in set(pairs))
    return {k: c[k] for k in sorted(c, key=lambda k: (SPLIT_ORDER.get(k, 9), k))}


def edit_body(o):
    """Edit record -> the edit itself: wrapper {file, pkg, fmt, edit, rx} or the old bare form."""
    if isinstance(o, dict) and isinstance(o.get("edit"), dict):
        return o["edit"]
    return o if isinstance(o, dict) else {}


def package_progress(edits_dir):
    """edits/*.json -> (n records, n with edit.reviewed true, n with decisions or reviewed, `who` names). The groups
    file (edits/_groups.json, annotation/review_groups.py) is no image record: not counted."""
    n = reviewed = touched = 0
    who = set()
    files = sorted(f for f in Path(edits_dir).glob("*.json") if is_record_file(f)) if Path(edits_dir).is_dir() else []
    for f in files:
        try:
            e = edit_body(read_json(f))
        except (ValueError, OSError):
            continue
        n += 1
        dec = e.get("dec") or e.get("decisions") or {}
        if e.get("reviewed") is True:
            reviewed += 1
        if dec or e.get("reviewed"):
            touched += 1
        for d in (dec.values() if isinstance(dec, dict) else dec if isinstance(dec, list) else []):
            if isinstance(d, dict) and d.get("who"):
                who.add(d["who"])
    return n, reviewed, touched, who


def git_sha():
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, timeout=10).stdout.strip() or "nogit"
    except (OSError, subprocess.SubprocessError):
        return "nogit"


class TileScan(Scan):
    """validate.Scan, also keeping each session MANIFEST row per (session, key)."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.tiles = {}

    def note_stem(self, stem, **info):
        key = super().note_stem(stem, **info)
        if key and info.get("manifest") and info.get("session_date_basis") == "folder":
            d = self.tiles.setdefault((info["session_date"], key), {"name": Path(stem).name})
            for k, v in info.items():
                if v not in (None, ""):
                    d.setdefault(k, v)
        return key


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _iso(t):
    """ISO time to seconds (schema pattern), or None."""
    return str(t)[:19] if t and re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", str(t)) else None


def _date_id(d):
    return (d or "").replace("-", "")[:8] or None


# ---------------------------------------------------------------- raw imaging metadata
def parser_path(proc=None):
    """`<Cryo-image-processing>/processing scripts/cryoem_core/metadata.py` (proc first, then the default root)."""
    return next((p for p in (Path(r) / PARSER_REL for r in (proc, PROCESSING) if r) if p.is_file()), None)


def load_metadata_parser(proc=None):
    """parse_metadata from `<Cryo-image-processing>/processing scripts/cryoem_core/metadata.py`, the canonical
    SerialEM parser, loaded by file path (importing the cryoem_core package would pull in numpy/torch) with
    bytecode writing off, so nothing is written next to it. None when the file is not found."""
    path = parser_path(proc)
    if path is None:
        return None
    spec = importlib.util.spec_from_file_location("_cryoem_core_metadata_readonly", path)
    mod = importlib.util.module_from_spec(spec)
    old, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = old
    return mod.parse_metadata


def scope_name(header):
    """`[T = SerialEM: <scope>   27-Oct-26  16:27:16 ]` header value -> `<scope>` (None if empty)."""
    h = re.sub(r"^SerialEM:\s*", "", str(header or "").strip().rstrip("]").strip())
    return HEADER_TIME.sub("", h).strip() or None


def scope_model(header):
    """Header value -> the scope model only (`Glacios <serial> at <site>` -> `Glacios`): cut at ` at ` and at the
    first all-digit word (the serial). Serial and site are never stored."""
    name = re.split(r"\s+at\s+", scope_name(header) or "", maxsplit=1)[0]
    words = []
    for wd in name.split():
        if wd.isdigit():
            break
        words.append(wd)
    return " ".join(words) or None


def acq_time(text):
    """SerialEM DateTime (`27-Oct-2026  16:27:27`, older `27-Oct-26 ...`) -> ISO seconds, or None."""
    s = " ".join(str(text or "").split())
    for fmt in ("%d-%b-%Y %H:%M:%S", "%d-%b-%y %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt).isoformat(timespec="seconds")
        except ValueError:
            pass
    return None


def _int_if_whole(v):
    return int(v) if float(v).is_integer() else v


def file_summary(global_params, pieces):
    """parse_metadata() output -> the per-file values a session summary needs. Image sections only ([ZValue] in an
    .mdoc, [Image] in an .idoc); [MontSection] repeats the montage's values and would count them twice.
    The per-image values (mag, px, defocus, target, dose, binning) come from imaging images only: magnification
    >= IMAGING_MIN_MAG (or none recorded). Low-mag (62-210x) and search maps (2,600x, defocus -100) are `other`.
    Not ExposureDose > 0: 20260804 and part of 20260817 record dose 0 at 45,000x / 73,000x (uncalibrated)."""
    ims = [p for p in pieces if p.get("_section") in ("ZValue", "Image")]
    img = [p for p in ims if (_num(p.get("Magnification")) or IMAGING_MIN_MAG) >= IMAGING_MIN_MAG]

    def nums(key, src=img):
        return [v for v in (_num(p.get(key)) for p in src) if v is not None]
    px = nums("PixelSpacing") or ([v for v in [_num(global_params.get("PixelSpacing"))] if v is not None] if img else [])
    kv = [v for v in [_num(global_params.get("Voltage")), *nums("Voltage", ims)] if v is not None]
    times = sorted(t for t in (acq_time(p.get("DateTime")) for p in ims) if t)
    other = [p for p in ims if not any(p is q for q in img)]                   # low-mag / search images
    return {"scope": scope_model(global_params.get("[T")), "n_images": len(ims), "n_imaging": len(img),
            "other_mag": sorted({int(v) for v in nums("Magnification", other)}),
            "mag": sorted({int(v) for v in nums("Magnification")}), "px": sorted({round(v, 4) for v in px}),
            "defocus": [round(v, 4) for v in nums("Defocus")],
            "target": sorted({round(v, 3) for v in nums("TargetDefocus")}),
            "dose": [round(v, 4) for v in nums("ExposureDose")], "kv": sorted({round(v, 1) for v in kv}),
            "binning": sorted(set(nums("Binning"))), "first": times[0] if times else None,
            "last": times[-1] if times else None}


def spread(values, nd=3):
    return {"min": round(min(values), nd), "median": round(statistics.median(values), nd),
            "max": round(max(values), nd)} if values else None


def session_acquisition(summaries):
    """[(file name, file_summary)] -> session fields. Atlas files (name contains `atlas`: grid overview maps)
    count in n_mdoc, instrument, voltage and the acquisition dates, not in the per-image stats (which also leave
    out low-mag / search images, see file_summary). Non-atlas files: n_imaging have imaging images, n_other have
    none; other_magnification = the left-out images' magnifications (e.g. 2,600x search maps). Dose stats use
    ExposureDose > 0 only (0 = not calibrated), so a session with no calibrated dose has dose null."""
    acq = [s for name, s in summaries if not ATLAS.search(name)]
    scopes = sorted({s["scope"] for _, s in summaries if s.get("scope")})
    times = sorted(t for _, s in summaries for t in (s.get("first"), s.get("last")) if t)
    union = lambda key, src: sorted({_int_if_whole(v) for s in src for v in s.get(key) or []})   # noqa: E731
    return {"instrument": "; ".join(scopes) or None,
            "magnification": union("mag", acq) or None,
            "pixel_size_a": union("px", acq) or None,
            "defocus_um": spread([v for s in acq for v in s.get("defocus") or []]),
            "target_defocus_um": union("target", acq) or None,
            "dose": spread([v for s in acq for v in s.get("dose") or [] if v > 0]),
            "voltage_kv": union("kv", [s for _, s in summaries]) or None,
            "binning": union("binning", acq) or None,
            "acq_dates": {"first": times[0], "last": times[-1]} if times else None,
            "n_mdoc": len(summaries),
            "n_imaging": sum(1 for s in acq if s.get("n_imaging")),
            "n_other": sum(1 for s in acq if not s.get("n_imaging")),
            "other_magnification": union("other_mag", acq) or None}


def raw_files(session_dir, layout):
    """The session's raw .mdoc/.idoc per its layout (RAW_LAYOUT / DEFAULT_LAYOUT); None when the raw folder is
    missing. Backups (`.mdoc~`) are not matched."""
    base = Path(session_dir) / layout["dir"] if layout["dir"] else Path(session_dir)
    if not base.is_dir():
        return None
    dirs = [base] + (sorted(d for d in base.glob(layout["subdirs"]) if d.is_dir()) if layout.get("subdirs") else [])
    skip = set(layout.get("skip") or ())
    out = set()
    for d in dirs:
        for ext in ("*.mdoc", "*.idoc"):
            for f in (d.rglob(ext) if layout.get("recursive") else d.glob(ext)):
                if f.is_file() and not skip & set(f.relative_to(base).parts[:-1]):
                    out.add(f)
    return sorted(out)


class MdocCache:
    """Parsed per-file summaries keyed by path (relative to the processing root) + [mtime_ns, size]: a file is
    parsed again only when it changed; every file when the parser (metadata.py sha256) or MDOC_CACHE_SCHEMA did.
    Lives next to the registry (derived, safe to delete); written by Importer.write (not on dry runs)."""

    def __init__(self, path, parser_sha256=None):
        self.path, self.files, self.seen = Path(path), {}, set()
        self.parser_sha256 = parser_sha256
        self.parsed = self.hits = 0
        self.dirty = False
        try:
            d = read_json(self.path)
            if d.get("schema") == MDOC_CACHE_SCHEMA and d.get("parser_sha256") == parser_sha256:
                self.files = d.get("files") or {}
            else:
                self.dirty = True
        except (OSError, ValueError, AttributeError):
            pass

    def get(self, key, stat, parse):
        sig = [stat.st_mtime_ns, stat.st_size]
        self.seen.add(key)
        e = self.files.get(key)
        if isinstance(e, dict) and e.get("sig") == sig:
            self.hits += 1
            return e["s"]
        s = parse()
        self.files[key] = {"sig": sig, "s": s}
        self.parsed += 1
        self.dirty = True
        return s

    def text(self):
        files = {k: self.files[k] for k in sorted(self.seen) if k in self.files}   # files gone since: dropped
        return json.dumps({"schema": MDOC_CACHE_SCHEMA, "parser_sha256": self.parser_sha256, "files": files},
                          separators=(",", ":")) + "\n"


def sample_labels(session_dir):
    """`sample_key.csv` (`1. Notes + reports/` or the session root; columns `#`, `Label`, as cryoem_core
    load_sample_key reads it) -> {slot: label}, or None."""
    for f in (Path(session_dir) / NOTES_DIR / "sample_key.csv", Path(session_dir) / "sample_key.csv"):
        if not f.is_file():
            continue
        try:
            rows = read_csv(f)
        except UnicodeDecodeError:
            import csv
            with open(f, newline="", encoding="cp1252") as fh:
                rows = list(csv.DictReader(fh))
        out = {}
        for r in rows:
            try:
                slot = str(int((r.get("#") or "").strip()))
            except ValueError:
                continue
            if (r.get("Label") or "").strip():
                out[slot] = r["Label"].strip()
        return out or None
    return None


def grid_fields(rows):
    """[(session folder, imaged_grids_master.csv row)] of one grid -> grid record fields. Prep fields (label,
    project, prep_date, type, replicate, glow discharge): the earliest session's non-empty value; storage: the latest
    session's (where the grid is now). n_montages: the sum over rows. notes / flags: as written for a grid with one
    row, else `<session>: <text>` per row joined by `; `."""
    def slot(r):
        s = (r.get("session_grid") or "").strip()
        return int(s) if s.isdigit() else 0
    rows = sorted(rows, key=lambda x: (x[0], slot(x[1])))
    val = lambda r, k: str(r.get(k) or "").strip()          # noqa: E731
    out = {k: next((val(r, k) for _, r in (rows[::-1] if k == "storage" else rows) if val(r, k)), None)
           for k in GRID_FIELDS}
    if out["prep_date"] and not re.match(r"^\d{4}-\d{2}-\d{2}$", out["prep_date"]):
        out["prep_date"] = None
    if out["replicate"] and out["replicate"].isdigit():
        out["replicate"] = int(out["replicate"])
    n = [int(val(r, "n_montages")) for _, r in rows if val(r, "n_montages").isdigit()]
    out["n_montages"] = sum(n) if n else None
    for k in ("notes", "flags"):
        vals = [(f, val(r, k)) for f, r in rows if val(r, k)]
        out[k] = (vals[0][1] if len(rows) == 1 else "; ".join(f"{f}: {v}" for f, v in vals)) if vals else None
    return out


class Importer:
    def __init__(self, root, laptop=None, cache=None, processing=PROCESSING, parser=None, via=registry._ENV):
        """`parser`: parse_metadata-compatible callable; default = the canonical one (load_metadata_parser).
        `via`: the agent running the import, stamped on appended lines; not passed = $CRYOEV_VIA, None = a person."""
        self.root = Path(root)
        registry.check_format(self.root)          # a registry written for a newer reader: refuse before scanning
        self.via = registry.resolve_via(via)
        self.laptop = Path(laptop) if laptop else config.laptop_root(required=False)
        self.cache = Path(cache) if cache else config.cryoai_root(required=False)
        self.proc = Path(processing) if processing else None
        self._parser = parser
        pp = parser_path(self.proc)
        self.mdoc_cache = MdocCache(self.root / MDOC_CACHE, sha256_file(pp) if pp else None)
        self.tag = f"backfill:importers {git_sha()}"
        self.out = []                       # (entity, record)
        self.refused = Counter()
        self.examples = defaultdict(list)
        self.excluded = Counter()
        self.lists = {}                     # registry-relative path -> text (written by write())

    def refuse(self, why, ident):
        self.refused[why] += 1
        if len(self.examples[why]) < 3:
            self.examples[why].append(ident)

    def add(self, entity, **rec):
        rec = {"schema": f"{entity}/1", **rec}
        self.out.append((entity, {k: v for k, v in rec.items() if v is not None or k in REQUIRED_NULL.get(entity, ())}))

    # ------------------------------------------------------------ sources
    def scan(self):
        self.s = TileScan(self.laptop, self.cache, self.proc).run()
        self.excluded.update({e: t.excluded for e, t in self.s.t.items() if t.excluded})
        tr = self.laptop / "tracking" / "v2"
        self.tracking = {r["stem"]: r for r in read_csv(tr / "prelipo_tracking.csv")} \
            if (tr / "prelipo_tracking.csv").exists() else {}
        projects = Counter(r.get("roboflow_project") for r in self.tracking.values() if r.get("roboflow_project"))
        self.project = projects.most_common(1)[0][0] if projects else "unknown"
        self.pulls_and_rf_images()
        self.images_sessions_grids()
        self.exports_datasets_runs()
        self.models()
        self.packages()
        self.label_versions()
        self.inference_checks()
        return self

    def registry_image_ids(self):
        if not hasattr(self, "_img_ids"):
            self._img_ids = {r["id"] for e, r in self.out if e == "image"} | set(registry.current("image", self.root))
        return self._img_ids

    def membership(self, did, exp, sf):
        """Dataset -> image_list / counts_trained / counts_listed (null = unknown, never 0)."""
        d = self.cache / "training images" / exp if exp else None
        rows, method = None, None
        if sf:
            rows, method = split_membership(sf), "split file `split` column, DROPPED-* rows left out"
        elif d and any((d / s / "_annotations.coco.json").exists() for s in SPLIT_ORDER):
            rows, method = [], "export split folders"
            for split in SPLIT_ORDER:
                f = d / split / "_annotations.coco.json"
                if f.exists():
                    rows += [(ids.image_key(im["file_name"]), split, "") for im in read_json(f)["images"]]
        if rows is None:
            return {"image_list": None, "counts_trained": None, "counts_listed": None,
                    "image_list_gap": "no split file and no local COCO for the source export (unknown, not 0)"}
        pairs = {(stem, split) for stem, split, _ in rows}
        text = list_text(pairs)
        rel = f"{LISTS}/{did}.tsv"
        self.lists[rel] = text
        known = self.registry_image_ids()
        excl = {stem for stem, _, b in rows if b[:8] in EXCLUDED_SESSIONS or self.s.is_excluded_stem(stem)}
        missing = {stem for stem, _ in pairs if stem not in known}
        why = Counter("excluded_session" if stem in excl else "not_in_registry" for stem in missing)
        return {"image_list": {"file": rel, "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                               "method": method, "n": len(pairs)},
                "counts_trained": split_counts(pairs),
                "counts_listed": split_counts({p for p in pairs if p[0] in known}),
                "not_listed": dict(sorted(why.items())) or None}      # reason -> n (no excluded ids written)

    def image_by_rf_name(self):
        """Roboflow name -> image id, from image.rf_name (names claimed by two images are left out)."""
        if not hasattr(self, "_by_rf_name"):
            seen = defaultdict(set)
            recs = {**registry.current("image", self.root), **{r["id"]: r for e, r in self.out if e == "image"}}
            for iid, r in recs.items():
                if r.get("rf_name"):
                    seen[r["rf_name"]].add(iid)
            self._by_rf_name = {n: next(iter(v)) for n, v in seen.items() if len(v) == 1}
        return self._by_rf_name

    def export_key(self, im):
        """Export COCO image -> registry image id: the file name's key (`<stem>.rf.<hash>.jpg` -> image_key), else
        the image whose rf_name is the Roboflow name (`extra.name`), else the file name's key (not in the registry)."""
        key = ids.image_key(im["file_name"])
        if key in self.registry_image_ids():
            return key
        return self.image_by_rf_name().get((im.get("extra") or {}).get("name"), key)

    def export_membership(self, eid, rel, members):
        """Which images went into an export -> image_list {file, sha256, method, n} (`stem\tsplit`, like datasets),
        counts_listed, not_listed; or image_list null + image_list_gap (unknown, never empty). Sources, in order:
        - the export's own COCO files (members = [(coco image, split, kept)]): exact. Excluded-session images are
          counted in not_listed.excluded_session and left out of the list, as they are out of `counts`.
        - tracking/v2 prelipo_tracking.csv `in: <export>` column, used only when the export has no local COCO and no
          row says `unread`: membership of the TRACKED images only (backfill_tracking.py's pinned set), so the method
          says so; an export image the tracker does not list is not known here.
        """
        known = self.registry_image_ids()
        excluded = 0
        if members is not None:
            pairs = set()
            for im, split, kept in members:
                if kept:
                    pairs.add((self.export_key(im), split))
                else:
                    excluded += 1
            method = "export COCO split folders"
        else:
            col = f"in: {eid}"
            vals = {stem: (r.get(col) or "").strip() for stem, r in self.tracking.items() if col in r}
            if not vals or any(v.lower() == "unread" for v in vals.values()) or not any(vals.values()):
                gap = ("no local COCO for this export, and tracking/v2 does not record its images (unread or absent)")
                return {"image_list": None, "image_list_gap": gap}
            pairs = set()
            for stem, v in vals.items():
                split = v.split(" ")[0]
                if not v or split not in SPLIT_ORDER:
                    continue
                if self.s.is_excluded_stem(stem):
                    excluded += 1
                    continue
                pairs.add((ids.image_key(stem), split))
            method = (f"tracking/v2 prelipo_tracking.csv `{col}` column: tracked images only (no local COCO); "
                      "an export image the tracker does not list is not known")
        text = list_text(pairs)
        self.lists[rel] = text
        missing = sum(1 for stem, _ in pairs if stem not in known)
        why = {k: v for k, v in (("excluded_session", excluded), ("not_in_registry", missing)) if v}
        return {"image_list": {"file": rel, "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                               "method": method, "n": len(pairs)},
                "counts_listed": split_counts({p for p in pairs if p[0] in known}),
                "not_listed": why or None, "image_list_gap": None}

    def pulls_and_rf_images(self):
        for pid, p in sorted(self.s.pulls.items()):
            if not p["coco"]:
                continue
            c = read_json(p["coco"])
            keep = {im["id"] for im in c["images"]
                    if not self.s.is_excluded_stem(ids.image_key(im.get("extra", {}).get("name") or im["file_name"]))}
            self.add("pull", id=pid, path=str(p["path"]), project=self.project,
                     created=f"{pid[14:18]}-{pid[18:20]}-{pid[20:22]}T{pid[23:25]}:{pid[25:27]}",
                     n_images=len(p["stems"]), coco_sha256=sha256_file(p["coco"]),
                     n_annotations=sum(1 for a in c["annotations"] if a["image_id"] in keep),
                     note="project from tracking/v2 (not stored in the pull folder)")
            for im in c["images"]:
                ex = im.get("extra", {})
                stem = ids.image_key(ex.get("name") or im["file_name"])
                if self.s.is_excluded_stem(stem) or not ex.get("roboflow_id"):
                    continue
                t = self.tracking.get(stem, {})
                self.add("roboflow_image", id=ex["roboflow_id"], stem=stem, project=self.project,
                         name=im["file_name"], split=ex.get("split") if ex.get("split") in ("train", "valid", "test") else None,
                         tags=(t.get("roboflow_tags") or "").split(";") if t.get("roboflow_tags") else [],
                         uploaded=t.get("uploaded_at") or None, upload_batch=t.get("roboflow_batch") or None,
                         first_seen_pull=pid, deleted=False)

    def images_sessions_grids(self):
        s = self.s
        sessions = set()
        legacy = set(self.tracking)
        for p in s.pulls.values():
            legacy |= set(p["stems"])
        for key in sorted(legacy):
            d = s.image_info.get(key, {})
            if s.is_excluded_stem(key):
                continue
            basis = (d.get("session_date_basis") or "").split(":")[0] or None
            sess = _date_id(d.get("session_date")) if basis else None
            sess = sess if ids.is_id("session", sess or "") else None
            grid = d.get("grid") if ids.is_id("grid", d.get("grid") or "") else None
            dup = ids.image_key(d["duplicate_of"]) if d.get("duplicate_of") else None
            self.add("image", id=key, session=sess, grid=grid, session_date_basis=basis if sess else None,
                     session_grid=d.get("session_grid") or None, rf_name=d.get("rf_name"),
                     duplicate_of=dup if dup and ids.is_id("stem", dup) else None)
            if sess:
                sessions.add(sess)
        copies = self.copy_map()
        known = {f"{ss}__{k}" for ss, k in s.tiles} | legacy
        for (sess, key), d in sorted(s.tiles.items()):
            if sess[:8] in EXCLUDED_SESSIONS or key in legacy:
                continue
            generic = bool(GENERIC.match(key))
            iid = f"{sess}_{key}" if generic else f"{sess}__{key}"
            src = copies.get((sess, key))
            if generic and d.get("source_file") and ids.image_key(d["source_file"]) != key:
                src = src or f"{sess}__{ids.image_key(d['source_file'])}"
            src = src if src in known else None                 # only link to a tile that is registered
            grid = d.get("grid") if ids.is_id("grid", d.get("grid") or "") else None
            # MANIFEST source_file / montage; a generic copy (blind_NN) whose row names only itself has neither
            real_src = bool(d.get("source_file")) and ids.image_key(d["source_file"]) != key
            self.add("image", id=iid, session=sess, grid=grid, session_date_basis="folder",
                     session_grid=d.get("session_grid") or None, output_name=d.get("name"),
                     nm_per_px=_num(d.get("nm_per_px")), md5=d.get("md5") or None, copy_of=src,
                     montage=d.get("montage") if real_src or not generic else None,
                     source_file=d.get("source_file") if real_src or not generic else None)
            sessions.add(sess)
            if d.get("rf_id") and ids.is_id("rf_id", d["rf_id"]):
                self.add("roboflow_image", id=d["rf_id"], stem=iid, project=self.project, name=d.get("name"),
                         uploaded=None, deleted=False)
        for sid, info in sorted(s.sessions.items()):
            if ids.is_id("session", sid):
                sessions.add(sid)
        for sid in sorted(sessions):
            info = s.sessions.get(sid, {})
            basis = "folder" if info.get("folder") else "tracker" if info.get("tracker") or info.get("master") else "inferred"
            slots = {slot: g for (ss, slot), g in s.slot_grid.items() if ss == sid and slot}
            sdir = self.proc / sid if self.proc and (self.proc / sid).is_dir() else None
            self.add("session", id=sid, date=f"{sid[:4]}-{sid[4:6]}-{sid[6:8]}", date_basis=basis,
                     session_grids=slots or None, status="processed" if info.get("folder") else "imaged",
                     local_folder=f"{self.proc.name}/{sid}" if sdir else None,
                     sample_labels=sample_labels(sdir) if sdir else None, **self.acquisition(sid, sdir))
        for g in sorted(s.grids):
            self.add("grid", id=g, **grid_fields(s.grid_master.get(g, [])))

    @property
    def parser(self):
        if self._parser is None:
            self._parser = load_metadata_parser(self.proc) or False
        return self._parser or None

    def acquisition(self, sid, sdir):
        """Session acquisition summary from its raw .mdoc/.idoc (`2. Raw images/**`, or RAW_LAYOUT; read only),
        or a gap."""
        if sid[:8] in EXCLUDED_SESSIONS:
            return {}
        if sdir is None:
            return {"acquisition_gap": "no local session folder: raw metadata not read"}
        lay = RAW_LAYOUT.get(sid, DEFAULT_LAYOUT)
        where = f"`{lay['dir']}/`" if lay["dir"] else "the session root" + (f" + `{lay['subdirs']}/`" if lay.get("subdirs") else "")
        files = raw_files(sdir, lay)
        if files is None:
            return {"acquisition_gap": f"no {where} folder: raw metadata not read"}
        if not files:
            return {"acquisition_gap": f"no .mdoc/.idoc in {where}"}
        if self.parser is None:
            return {"n_mdoc": len(files), "acquisition_gap": "canonical parser cryoem_core/metadata.py not found"}
        out, bad = [], 0
        for f in files:
            try:
                key = f.relative_to(self.proc).as_posix()
                out.append((f.name, self.mdoc_cache.get(key, f.stat(), lambda f=f: file_summary(*self.parser(str(f))))))
            except (OSError, ValueError, UnicodeDecodeError):
                bad += 1
        res = session_acquisition(out) if out else {}
        res["n_mdoc"] = len(files)
        if bad:
            res["acquisition_gap"] = f"{bad} of {len(files)} .mdoc/.idoc files unreadable (left out)"
        return res

    def copy_map(self):
        """(session, blind key) -> source image id, from <session>/5. Model + labeling/*selection_PRIVATE.csv."""
        out = {}
        if not (self.proc and self.proc.is_dir()):
            return out
        for f in self.proc.glob("*/5. Model + labeling/*selection_PRIVATE.csv"):
            sess = f.parent.parent.name
            if sess[:8] in EXCLUDED_SESSIONS:
                continue
            for r in read_csv(f):
                if r.get("blind_name") and r.get("review_name"):
                    out[(sess, ids.image_key(r["blind_name"]))] = f"{sess}__{ids.image_key(r['review_name'])}"
        return out

    def exports_datasets_runs(self):
        s = self.s
        self.export_stems = {}
        for eid, e in sorted(s.exports.items()):
            d = self.cache / "training images" / eid
            counts, classes, lines, members = {}, [], [], []
            for split, n in e["counts"].items():
                c = read_json(d / split / "_annotations.coco.json")
                keep = {im["id"] for im in c["images"] if not s.is_excluded_stem(ids.stem_of_export(im["file_name"]))}
                counts[split] = {"images": n, "instances": sum(1 for a in c["annotations"] if a["image_id"] in keep)}
                classes = classes or [k["name"] for k in c["categories"] if k.get("supercategory") != "none"]
                lines += [f"{split}\t{ids.image_key(im['file_name'])}\n" for im in c["images"] if im["id"] in keep]
                members += [(im, split, im["id"] in keep) for im in c["images"]]
            self.export_stems[eid] = (hashlib.sha256("".join(sorted(lines)).encode()).hexdigest() if lines else None)
            self.add("roboflow_export", id=eid, path=str(d), counts=counts, classes=classes,
                     derived_from=e["derived_from"], split_file=e["split_file"].name if e["split_file"] else None,
                     note=None if counts else "local placeholder: no COCO in the cache",
                     **self.export_membership(eid, export_list_name(eid), members if counts else None))
        self.exp_counts = {r["id"]: r for ent, r in self.out if ent == "roboflow_export"}
        for did, d in sorted(s.datasets.items()):
            if not did:
                continue
            lv_rec = self.lv_dataset(did, d)
            if lv_rec is not None:
                self.add("dataset", **lv_rec)
                continue
            exp = d["export"]
            er = self.exp_counts.get(exp, {})
            sf = s.exports.get(exp, {}).get("split_file")
            created = None
            m = re.match(r"^roboflow_(\d{4})(\d{2})(\d{2})_", did)
            if m:
                created = f"{m[1]}-{m[2]}-{m[3]}T00:00:00"
            img_sha = self.export_stems.get(exp)
            mem = self.membership(did, exp, sf)
            counts = er.get("counts") or {"train": {"images": None}, "valid": {"images": None}}   # unknown, never 0
            self.add("dataset", id=did, created=created or "1970-01-01T00:00:00",
                     source={"kind": "roboflow_export", "id": exp or "unknown"}, export=exp, label_version=None,
                     inputs=[{"id": exp, "role": "source", "entity": "roboflow_export"}] if exp else None,
                     split={"method": "export split folders" if not sf else "split_assignment.csv",
                            "file": str(sf) if sf else "", "sha256": sha256_file(sf) if sf else None,
                            "sha256_gap": None if sf else "no split file (export split folders)"},
                     counts=counts, counts_gap=None if er.get("counts") else "source export COCO not local (unknown)",
                     classes=er.get("classes") or [], images_sha256=img_sha, **mem,
                     images_sha256_gap=None if img_sha else "source export COCO not local",
                     status="trained" if d["runs"] else "frozen",
                     note="created = date in the dataset name (time unknown)")
        for rid, r in sorted(s.runs.items()):
            rj = self.run_json(r["dataset"], rid) if r["dataset"] else None
            if rj:
                self.add("run", **rj)
                continue
            cfg = ids.parse_run(rid)
            if not cfg or not r["dataset"]:
                self.refuse("run: config not parseable / no dataset", rid)
                continue
            self.add("run", id=rid, dataset=r["dataset"], job_bundle=None, job_bundle_gap="pre-bundle run",
                     config={"arch": f"yolov{cfg['ver']}{cfg['size']}-seg", "imgsz": cfg["imgsz"],
                             "batch": cfg["batch"], "epochs": cfg["epochs"]},
                     status="finished" if r["model"] else "running",
                     note=None if r["local"] else "run folder not in the local cache")

    def lv_dataset(self, did, d):
        """A dataset built from a label version (data_utils/dataset_from_lv.py) keeps its own record: the registered
        one when its source.kind is label_version, else <cache>/training outputs/<did>/provenance.json
        `dataset_record`. Only the status is derived (frozen -> trained once it has runs), so an unchanged record
        compares equal and nothing is appended; membership() is never called for it. Its list is the folder's
        image_list.tsv, used only when it matches the record's image_list.sha256. -> record or None."""
        if not hasattr(self, "_cur_datasets"):
            self._cur_datasets = registry.current("dataset", self.root)
        old = self._cur_datasets.get(did)
        folder = self.cache / "training outputs" / did if self.cache else None
        if old and (old.get("source") or {}).get("kind") == "label_version":
            rec = dict(old)
        else:
            rec = lv_provenance(folder)
        if rec is None:
            return None
        for k in ("schema", "logged", "recorded_by", "via"):
            rec.pop(k, None)
        if d["runs"] and rec.get("status") == "frozen":
            rec["status"] = "trained"
        lst = rec.get("image_list") or {}
        tsv = folder / "image_list.tsv" if folder else None
        if tsv and tsv.is_file() and lst.get("file") and lst.get("sha256") == sha256_file(tsv):
            self.lists[lst["file"]] = tsv.read_bytes().decode("utf-8")
        return rec

    def run_json(self, dataset, rid):
        """Run fields from the run folder's own run.json (written by run_experiments.py), or None.
        Its status finished/error/oom maps to finished/failed; config and job_bundle are taken as written."""
        p = self.cache / "training outputs" / dataset / "runs" / rid / "run.json" if self.cache else None
        if not p or not p.is_file():
            return None
        try:
            j = read_json(p)
        except (ValueError, OSError):
            self.refuse("run: run.json unreadable", rid)
            return None
        bundle = j.get("job_bundle") or j.get("bundle_id")
        bundle = bundle if ids.is_id("bundle", bundle) else None
        st = j.get("status")
        keep = ("arch", "imgsz", "batch", "epochs", "patience", "rect", "weights_init", "seed")
        return {"id": rid, "dataset": dataset, "job_bundle": bundle,
                "job_bundle_gap": None if bundle else "no bundle id in run.json",
                "config": {k: v for k, v in (j.get("config") or {}).items() if k in keep},
                "status": "finished" if st == "finished" else "failed",
                "machine": j.get("machine"), "cryoEV_commit": j.get("cryoEV_commit"),
                "ultralytics": j.get("ultralytics"), "started": _iso(j.get("started")),
                "finished": _iso(j.get("ended")), "experiment_log_timestamp": j.get("experiment_log_timestamp"),
                "metrics": {k: v for k, v in (j.get("metrics") or {}).items() if isinstance(v, (int, float))} or None,
                "note": None if st == "finished" else f"run.json status {st}: {str(j.get('error') or '')[:200]}"}

    def models(self):
        wroot = [REPO / "models", (find_sibling("cryoEV") or REPO) / "models"]
        for mid, m in sorted(self.s.models.items()):
            m = m or {}
            sr = Path(m.get("source_run") or "")
            run = sr.name if ids.parse_run(sr.name) else None
            ds = sr.parts[sr.parts.index("runs") - 1] if "runs" in sr.parts else None
            w = next((p / mid / "best.pt" for p in wroot if (p / mid / "best.pt").exists()), None)
            init = re.search(r"fine-tuned from (\w+)", m.get("arch") or "")
            canon = LEGACY_MODELS.get(mid, mid)
            init_id = LEGACY_MODELS.get(init.group(1), init.group(1)) if init else None
            inputs = ([{"id": ds, "role": "train_data", "entity": "dataset"}] if ds else []) + \
                     ([{"id": init_id, "role": "init_weights", "entity": "model"}] if init_id else [])
            self.add("model", id=canon, aliases=[mid] if canon != mid else None, run=run, dataset=ds,
                     inputs=inputs or None,
                     weights=m.get("weights") or "", imgsz=int(m.get("imgsz") or 0),
                     weights_sha256=sha256_file(w) if w else None,
                     init_from=init.group(1) if init else None, status="registered",
                     note="legacy id kept as alias; models.yaml key unchanged" if canon != mid else None)

    def packages(self):
        s = self.s
        lv_pkgs = {}
        for d in sorted((self.laptop / "labels").glob("lv_*")):
            if (d / "version.json").exists():
                lv_pkgs.setdefault(read_json(d / "version.json").get("pkg"), []).append(d.name)
        run_to_model = {Path((m or {}).get("source_run") or "x").name: k for k, m in s.models.items()}
        for pid, entries in sorted(s.packages.items()):
            for e in entries:
                if e["folder"].upper().startswith("SANDBOX"):
                    self.excluded["review_package (SANDBOX)"] += 1
                    continue
                if e.get("base_kind") == "prediction_set":     # registered by its builder (owner, prediction_set);
                    self.excluded["review_package (prediction, own line)"] += 1    # deriving it here would drop those
                    continue
                d = read_datajs(e["path"] / "data.js")
                run = (d.get("run") or "").split(" ")[0]
                model = run if run in s.models else run_to_model.get(run)
                edits = [f for f in (e["path"] / "edits").glob("*.json") if is_record_file(f)] \
                    if (e["path"] / "edits").is_dir() else []
                parent = ids.package_parent(pid)
                status = "applied" if pid in lv_pkgs else "in_review" if edits else "built"
                base_kind = "pull" if e["base"].startswith("roboflow_pull_") else "export"
                n_images = len(d.get("images") or [])
                self.add("review_package", id=pid, path=str(e["path"]), kind="part" if parent else "review",
                         parent=parent, base=e["base"], base_kind=base_kind,
                         model=model, run=run if ids.is_id("run", run) else None, flag_conf=d.get("flag_conf"),
                         n_images=n_images, built=d.get("created"),
                         n_edit_records=len(edits), status=status)
                if not parent:                                  # parts are worked inside their parent's job
                    self.revise_job(pid, e, d, model, base_kind, n_images, lv_pkgs.get(pid, []))

    def revise_job(self, pid, e, d, model, base_kind, n_images, outputs):
        """One `job/1` (revise, app) per top-level review package. Status from edits/ only: never done from
        the package being `applied` (an explicit set-status line is kept by write())."""
        built = _date_id(d.get("created"))
        if not built:
            self.refuse("job: review package has no built date", pid)
            return
        n, reviewed, touched, names = package_progress(e["path"] / "edits")
        alias = {}
        try:
            for p in read_json(registry.PEOPLE)["people"]:
                for a in [p["id"], *(p.get("aliases") or [])]:
                    alias[a] = p["id"]
        except (OSError, ValueError, KeyError):
            pass
        who = {alias.get(x) for x in names}
        inputs = [{"id": e["base"], "role": "base", "entity": "pull" if base_kind == "pull" else "roboflow_export"}]
        if model:
            inputs.append({"id": LEGACY_MODELS.get(model, model), "role": "predictions", "entity": "model"})
        self.add("job", id=ids.job_id(built, pid), name=f"Revise {pid}", type="revise", where="app",
                 package=pid, n_images=n_images, status="in_progress" if touched else "ready",
                 progress={"done": reviewed, "total": n_images, "unit": "images reviewed"},
                 who=next(iter(who)) if len(who) == 1 and None not in who else None,
                 inputs=inputs, outputs=sorted(outputs), port=APP_PORT, n_edit_records=n)

    def label_versions(self):
        """Review scope (apply_review_edits.py, 2026-11-04): review_scope, n_reviewed, n_package_images, n_held_back
        and supersedes come from version.json (absent on older versions: the page shows those as all edits). An lv
        that a newer version.json names in `supersedes` is status superseded, like SUPERSEDED / SUPERSEDED.md."""
        dirs = [d for d in sorted((self.laptop / "labels").glob("lv_*")) if (d / "version.json").exists()]
        named = set()
        for d in dirs:
            try:
                s = read_json(d / "version.json").get("supersedes")
            except (OSError, ValueError, AttributeError):
                s = None
            if isinstance(s, str) and s != d.name:
                named.add(s)
        for d in sorted((self.laptop / "labels").glob("lv_*")):
            vj = d / "version.json"
            if not vj.exists():
                self.refuse("label_version: version.json missing", d.name)
                continue
            v = read_json(vj)
            sup = next((old for old, new in SUPERSEDED.items() if new == d.name), None) or \
                (v.get("supersedes") if isinstance(v.get("supersedes"), str) else None)
            superseded = d.name in SUPERSEDED or (d / "SUPERSEDED.md").exists() or d.name in named
            inputs = []
            if v.get("base"):
                b = str(v["base"])        # a pull, an earlier label version (a review applied on top), or an export
                inputs.append({"id": v["base"], "role": "base",
                               "entity": "pull" if b.startswith("roboflow_pull_") else
                               "label_version" if re.fullmatch(r"lv_\d{8}_\d{4}", b) else "roboflow_export"})
            if v.get("pkg"):
                inputs.append({"id": v["pkg"], "role": "edits", "entity": "review_package"})
            self.add("label_version", id=d.name, owner=str(vj), supersedes=sup, inputs=inputs or None,
                     review_scope=v.get("review_scope"), n_reviewed=v.get("n_reviewed"),
                     n_package_images=v.get("n_package_images"), n_held_back=v.get("n_held_back"),
                     status="superseded" if superseded else "created")

    def inference_checks(self):
        bj = self.laptop / "inference_checks" / "batches.jsonl"
        if not bj.exists():
            return
        seen = []
        for _, o, _ in read_jsonl(bj):
            if o and o.get("batch_id") and o["batch_id"] not in seen:
                seen.append(o["batch_id"])
        for bid in seen:
            if bid[:8] in EXCLUDED_SESSIONS:
                continue
            self.add("inference_check", id=bid, owner=f"{bj}#{bid}")

    # ------------------------------------------------------------ write
    def list_files(self, rec):
        """{rel: bytes} for the image list a record names, when this import built its text, else None."""
        il = rec.get("image_list")
        rel = il.get("file") if isinstance(il, dict) else None
        return {rel: self.lists[rel].encode("utf-8")} if rel in self.lists else None

    def changed_files(self, files):
        """How many of {rel: bytes} are missing or differ on disk."""
        return sum(1 for rel, data in (files or {}).items()
                   if not (self.root / rel).is_file() or (self.root / rel).read_bytes() != data)

    def write_lists(self, dry_run=False, only=None):
        """Image lists next to the registry (those in `only`); (re)written only when the text changed. -> n files to
        write. Used to repair a list whose record is unchanged; a new or changed record's list goes with its line."""
        n = 0
        for rel, text in sorted(self.lists.items()):
            if only is not None and rel not in only:
                continue
            path = registry.registry_file(self.root, rel)
            if path.exists() and path.read_bytes() == text.encode("utf-8"):
                continue
            n += 1
            if dry_run:
                continue
            registry._guard(self.root)
            registry._guard(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_bytes(text.encode("utf-8"))
            tmp.replace(path)
        return n

    def write_mdoc_cache(self, dry_run=False):
        """The parsed .mdoc/.idoc summaries next to the registry, when anything was parsed or dropped."""
        c = self.mdoc_cache
        if dry_run or not (c.dirty or set(c.files) - c.seen) or not c.seen:
            return False
        registry._guard(self.root)
        registry._guard(c.path)
        c.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = c.path.with_name(c.path.name + ".tmp")
        tmp.write_text(c.text(), encoding="utf-8")
        tmp.replace(c.path)
        return True

    def write(self, dry_run=False, production=True):
        """_write() holding UPLOAD_LOCK (not on a dry run): the portal's upload-session lines are read and carried
        forward with nothing appended in between. production=False skips the PRODUCTION restatement (round 11:
        import_label_versions records label versions and nothing else)."""
        if dry_run:
            return self._write(dry_run, production)
        registry._guard(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        lock = registry.acquire_lock(self.root / UPLOAD_LOCK, UPLOAD_LOCK_WAIT)
        if lock is None:
            raise registry.RegistryError(f"registry busy: {self.root / UPLOAD_LOCK} held for {UPLOAD_LOCK_WAIT:.0f} s "
                                         "(the portal is registering an upload session)")
        try:
            return self._write(dry_run, production)
        finally:
            lock.close()

    def _write(self, dry_run=False, production=True):
        """Append new / changed records. A record's image list is written with its line under the registry lock
        (nothing for a refused line); a list whose record is unchanged is repaired afterwards if missing or changed."""
        lists_written, repair = 0, set()
        self.write_mdoc_cache(dry_run)
        cur = registry.current(root=self.root)
        appended = Counter()
        img_session = {k: v.get("session") for k, v in cur.items() if v.get("schema") == "image/1"}
        stem_rf = {v["stem"]: k for k, v in cur.items() if v.get("schema") == "roboflow_image/1"}
        seen_batch = set()
        for entity, rec in self.out:
            rid = rec["id"]
            if (entity, rid) in seen_batch:
                continue
            seen_batch.add((entity, rid))
            old = cur.get(rid)
            if old and old.get("schema") != rec["schema"]:
                self.refuse(f"{entity}: id already used by {old.get('schema')}", rid)
                continue
            if entity == "image" and rid in img_session and img_session[rid] != rec.get("session"):
                self.refuse("image: existing key arrives with a different session", rid)
                continue
            if entity == "roboflow_image" and stem_rf.get(rec["stem"], rid) != rid:
                self.refuse("roboflow_image: stem already has another Roboflow id", rec["stem"])
                continue
            if old and old.get("status_by"):                      # explicit status lines beat derived ones
                rec = dict(rec, status=old.get("status"), status_by=old["status_by"])
                rec.pop("status_via", None)                       # status_via travels with status_by (round 9)
                if old.get("status_via"):
                    rec["status_via"] = old["status_via"]
                if old.get("note") and not rec.get("note"):
                    rec["note"] = old["note"]
                if old.get("blocked_reason") and not rec.get("blocked_reason"):     # job: blocked by hand
                    rec["blocked_reason"] = old["blocked_reason"]
            if entity == "session" and old:                       # the portal's upload keys (upload_sessions.py)
                rec = dict(rec, **{k: old[k] for k in UPLOAD_KEYS if old.get(k) is not None and rec.get(k) is None})
            files = self.list_files(rec)
            if old is not None:
                full = registry.derive(entity, rec)
                if {k: v for k, v in full.items() if k not in IGNORE} == \
                        {k: v for k, v in old.items() if k not in IGNORE}:
                    repair |= set(files or ())
                    continue
            rec = dict(rec, recorded_by=self.tag)
            try:
                n_files = self.changed_files(files)
                if not dry_run:
                    registry.append(rec, root=self.root, files=files, replace_files=True, via=self.via)
            except registry.RegistryError as e:
                self.refuse(f"{entity}: {str(e)[:90]}", rid)
                continue
            appended[entity] += 1
            lists_written += n_files
            if entity == "image":
                img_session[rid] = rec.get("session")
            if entity == "roboflow_image":
                stem_rf[rec["stem"]] = rid
        lists_written += self.write_lists(dry_run, only=repair)
        if lists_written:
            appended["(image list files)"] += lists_written
        prod_id, who = PRODUCTION
        cur = registry.current(root=self.root)
        if production and not dry_run and self._restate_production(prod_id, who, cur):
            # the user's recorded decision, restated: never via (an agent never promotes; registry._check)
            registry.set_status(prod_id, "production", who, note="user decision 2026-10-30", root=self.root,
                                via=None)
            appended["model (set-status production)"] += 1
        return appended


    def _restate_production(self, prod_id, who, cur) -> bool:
        """PRODUCTION is a seed for an empty or rebuilt registry, never an override (round 10, TODO #62): restate it
        only when the model exists, isn't production now, NO line in its history is a decision by PRODUCTION_WHO in
        person (status_by == who, no status_via: e.g. a later demotion), and no other model is production."""
        rec = cur.get(prod_id)
        if rec is None or rec.get("status") == "production":
            return False
        if any(r.get("status_by") == who and not r.get("status_via")
               for r in registry.history(prod_id, root=self.root)):
            return False                       # the user decided something about this model: follow that
        return not any(m.get("status") == "production" for k, m in cur.items()
                       if k != prod_id and registry.entity_of(m) == "model")

REQUIRED_NULL = {"image": ("session", "grid"), "model": ("run",),
                 "dataset": ("images_sha256", "image_list", "counts_trained", "counts_listed"),
                 "roboflow_export": ("image_list",),
                 "review_package": (), "label_version": ()}


def run(root, dry_run=False, **kw):
    imp = Importer(root, **kw).scan()
    appended = imp.write(dry_run)
    cur = registry.current(root=root)
    by = Counter((v["schema"].split("/")[0], v.get("status") or "-") for v in cur.values())
    return {"appended": appended, "refused": imp.refused, "examples": dict(imp.examples),
            "excluded": imp.excluded, "current": by, "dry_run": dry_run}


def import_label_versions(root, via=None, laptop=None, only=None):
    """Round 11 (#67, the portal's Apply button): record label versions and nothing else pending. Importer ->
    label_versions() -> write(production=False): no tile scan, no other entity, no PRODUCTION restatement. `laptop`:
    the folder holding labels/ (default: config). `only`: the label-version ids to record (None = every lv_* folder),
    so other unregistered work stays for the next full import. -> {appended, refused, examples}."""
    imp = Importer(root, laptop=laptop, processing=None, via=via)
    imp.label_versions()
    if only is not None:
        keep = set(only)
        imp.out = [(e, r) for e, r in imp.out if r["id"] in keep]
    appended = imp.write(production=False)
    return {"appended": appended, "refused": imp.refused, "examples": dict(imp.examples)}


def print_result(res):
    print(f"appended{' (dry run)' if res['dry_run'] else ''}: {sum(res['appended'].values())} "
          + ", ".join(f"{k}={v}" for k, v in sorted(res["appended"].items())))
    for k, v in res["refused"].most_common():
        print(f"refused: {k}: {v} (e.g. {', '.join(res['examples'].get(k, []))})")
    if res["excluded"]:
        print("excluded (counted only): " + ", ".join(f"{k}={v}" for k, v in sorted(res["excluded"].items())))
    print("current by entity/status:")
    for (e, st), n in sorted(res["current"].items()):
        print(f"  {e:16} {st:12} {n}")
