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
- one revise `job/1` per top-level review package (no `_partN`, no SANDBOX): progress = edit records with
  `edit.reviewed` true / n_images; status in_progress if any record has decisions or reviewed, else ready.
  A package's `applied` status never makes its job done (only an explicit set-status line does).
- a run folder with its own `run.json` (run_experiments.py, 2026-11-02) gives status (finished; error/oom ->
  failed, with a note), job_bundle and config as written; older run folders are derived as before.
"""
import hashlib
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import ids
import registry
from validate import (EXCLUDED_SESSIONS, REPO, Scan, find_sibling, read_csv, read_datajs, read_json,
                      read_jsonl, sha256_file)
from validate import config  # data_utils.config (validate puts the repo on sys.path)

PROCESSING = config.processing_root()   # optional: None when not found
GENERIC = re.compile(r"^(blind|label|labeling)_\d+_[A-Za-z0-9]+$")
SUPERSEDED = {"lv_20261030_0846": "lv_20261030_0909"}          # old -> new (SUPERSEDED.md)
LEGACY_MODELS = {"allLayer_run14_20260713": "v0_alllayer_y8n_20260713",   # legacy id -> canonical id
                 "yifei_202512": "v0_yifei_y8n_20251201"}                 # (day unknown: 01)
PRODUCTION = ("v5_evnonev_y8m_20261029", "pujan")
IGNORE = {"logged", "recorded_by"}
DROPPED = "DROPPED"                      # split_assignment.csv `split` values like DROPPED-duplicate: not trained
SPLIT_ORDER = {"train": 0, "valid": 1, "test": 2}
LISTS = "lists"                          # <registry>/lists/<dataset>.tsv  (stem \t split)
APP_PORT = 8766                          # review app port for the real review (NEXT_ITERATION.md)


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


def split_counts(pairs):
    c = Counter(split for _, split in set(pairs))
    return {k: c[k] for k in sorted(c, key=lambda k: (SPLIT_ORDER.get(k, 9), k))}


def edit_body(o):
    """Edit record -> the edit itself: wrapper {file, pkg, fmt, edit, rx} or the old bare form."""
    if isinstance(o, dict) and isinstance(o.get("edit"), dict):
        return o["edit"]
    return o if isinstance(o, dict) else {}


def package_progress(edits_dir):
    """edits/*.json -> (n records, n with edit.reviewed true, n with decisions or reviewed, `who` names)."""
    n = reviewed = touched = 0
    who = set()
    files = sorted(Path(edits_dir).glob("*.json")) if Path(edits_dir).is_dir() else []
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


class Importer:
    def __init__(self, root, laptop=None, cache=None, processing=PROCESSING):
        self.root = Path(root)
        self.laptop = Path(laptop) if laptop else config.laptop_root(required=False)
        self.cache = Path(cache) if cache else config.cryoai_root(required=False)
        self.proc = Path(processing) if processing else None
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
            self.add("image", id=iid, session=sess, grid=grid, session_date_basis="folder",
                     session_grid=d.get("session_grid") or None, output_name=d.get("name"),
                     nm_per_px=_num(d.get("nm_per_px")), md5=d.get("md5") or None, copy_of=src)
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
            self.add("session", id=sid, date=f"{sid[:4]}-{sid[4:6]}-{sid[6:8]}", date_basis=basis,
                     session_grids=slots or None, status="processed" if info.get("folder") else "imaged")
        for g in sorted(s.grids):
            self.add("grid", id=g)

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
            counts, classes, lines = {}, [], []
            for split, n in e["counts"].items():
                c = read_json(d / split / "_annotations.coco.json")
                keep = {im["id"] for im in c["images"] if not s.is_excluded_stem(ids.stem_of_export(im["file_name"]))}
                counts[split] = {"images": n, "instances": sum(1 for a in c["annotations"] if a["image_id"] in keep)}
                classes = classes or [k["name"] for k in c["categories"] if k.get("supercategory") != "none"]
                lines += [f"{split}\t{ids.image_key(im['file_name'])}\n" for im in c["images"] if im["id"] in keep]
            self.export_stems[eid] = (hashlib.sha256("".join(sorted(lines)).encode()).hexdigest() if lines else None)
            self.add("roboflow_export", id=eid, path=str(d), counts=counts, classes=classes,
                     derived_from=e["derived_from"], split_file=e["split_file"].name if e["split_file"] else None,
                     note=None if counts else "local placeholder: no COCO in the cache")
        self.exp_counts = {r["id"]: r for ent, r in self.out if ent == "roboflow_export"}
        for did, d in sorted(s.datasets.items()):
            if not did:
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
        keep = ("arch", "imgsz", "batch", "epochs", "patience", "rect", "weights_init")
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
                d = read_datajs(e["path"] / "data.js")
                run = (d.get("run") or "").split(" ")[0]
                model = run if run in s.models else run_to_model.get(run)
                edits = list((e["path"] / "edits").glob("*.json")) if (e["path"] / "edits").is_dir() else []
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
        for d in sorted((self.laptop / "labels").glob("lv_*")):
            vj = d / "version.json"
            if not vj.exists():
                self.refuse("label_version: version.json missing", d.name)
                continue
            sup = next((old for old, new in SUPERSEDED.items() if new == d.name), None)
            superseded = d.name in SUPERSEDED or (d / "SUPERSEDED.md").exists()
            v = read_json(vj)
            inputs = []
            if v.get("base"):
                inputs.append({"id": v["base"], "role": "base",
                               "entity": "pull" if str(v["base"]).startswith("roboflow_pull_") else "roboflow_export"})
            if v.get("pkg"):
                inputs.append({"id": v["pkg"], "role": "edits", "entity": "review_package"})
            self.add("label_version", id=d.name, owner=str(vj), supersedes=sup, inputs=inputs or None,
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
    def write_lists(self, dry_run=False):
        """Image lists next to the registry; (re)written only when the text changed. -> n files to write."""
        n = 0
        for rel, text in sorted(self.lists.items()):
            path = self.root / rel
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

    def write(self, dry_run=False):
        lists_written = self.write_lists(dry_run)
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
                if old.get("note") and not rec.get("note"):
                    rec["note"] = old["note"]
                if old.get("blocked_reason") and not rec.get("blocked_reason"):     # job: blocked by hand
                    rec["blocked_reason"] = old["blocked_reason"]
            if old is not None:
                full = registry.derive(entity, rec)
                if {k: v for k, v in full.items() if k not in IGNORE} == \
                        {k: v for k, v in old.items() if k not in IGNORE}:
                    continue
            rec = dict(rec, recorded_by=self.tag)
            if not dry_run:
                try:
                    registry.append(rec, root=self.root)
                except registry.RegistryError as e:
                    self.refuse(f"{entity}: {str(e)[:90]}", rid)
                    continue
            appended[entity] += 1
            if entity == "image":
                img_session[rid] = rec.get("session")
            if entity == "roboflow_image":
                stem_rf[rec["stem"]] = rid
        if lists_written:
            appended["(image list files)"] += lists_written
        prod_id, who = PRODUCTION
        cur = registry.current(root=self.root)
        if not dry_run and prod_id in cur and cur[prod_id].get("status") != "production":
            registry.set_status(prod_id, "production", who, note="user decision 2026-10-30", root=self.root)
            appended["model (set-status production)"] += 1
        return appended


REQUIRED_NULL = {"image": ("session", "grid"), "model": ("run",),
                 "dataset": ("images_sha256", "image_list", "counts_trained", "counts_listed"),
                 "review_package": (), "label_version": ()}


def run(root, dry_run=False, **kw):
    imp = Importer(root, **kw).scan()
    appended = imp.write(dry_run)
    cur = registry.current(root=root)
    by = Counter((v["schema"].split("/")[0], v.get("status") or "-") for v in cur.values())
    return {"appended": appended, "refused": imp.refused, "examples": dict(imp.examples),
            "excluded": imp.excluded, "current": by, "dry_run": dry_run}


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
