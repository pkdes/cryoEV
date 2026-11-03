"""
History-tree page for the tracking registry (contract v1, docs/TRACKING_CONTRACT.md). Standard library only.

    python registry/history_page.py [--registry FILE_OR_DIR ...] [--out history.html]

Reads append-only JSONL (one file, or a folder of <entity>.jsonl like registry/samples/) and writes ONE
self-contained static HTML file: inline CSS/JS, no external URLs. The page has
- a Queue: one lane per job type (Annotate, Revise, Train, Check); cards blocked -> in_progress -> ready, with
  done/cancelled collapsed under "Done (n)"; each card shows status, n images, progress, who, where,
  inputs -> outputs, the blocked reason and a one-line next action;
- a collapsible history tree (pull -> label versions -> datasets -> job bundles -> runs -> models, with review
  packages and inference checks hung off what they relate to);
- per-record detail with the status history from repeated lines; for a model, a Model page first:
  production/candidate badge, dataset -> source, per-split trained vs listed counts ("unknown" when missing),
  the image list (read from the dataset's image_list file next to the registry and embedded at build time),
  init model and scores (inference checks);
- filters by entity/status, and id search;
- a Lineage graph tab: columns Roboflow labels (exports, pulls) -> local labels (lv_*, prediction sets) ->
  datasets -> models -> review/revise (packages, revise/annotate jobs); SVG edges parent -> child from `inputs[]`
  (legacy fields when absent); model -> package/job -> label version is the dashed back-edge. Superseded /
  abandoned / cancelled / retired records are hidden until "Show history" (then dimmed); production is starred;
- an Image lookup tab: image name or substring -> datasets + split (lists/*.tsv), label versions that changed it
  (the lv's changes.jsonl next to its version.json, else the registry's label_change rows), review packages
  (their data.js) and jobs (package or image_list), Roboflow image + split, session / grid. ONE embedded index.
Pointer records (`owner` = version.json / batches.jsonl#id) get the owner file's fields underneath, as
registry.current() does.

Semantics: the LAST LINE PER id IN FILE ORDER is current (not by `logged`); every line is kept as history.
Links are child -> parent only; children are derived here (no `superseded_by`, `parts`, `scores` needed).
If a later line is a partial status line, fields from earlier lines are kept (merged view) so the record
stays linked; the status always comes from the last line.

Record text reaches the page only as JSON inside <script type="application/json"> with <, >, & escaped,
and the page script builds the DOM with textContent (never innerHTML with record values).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
if str(HERE) not in sys.path:
    sys.path.insert(1, str(HERE))
from data_utils import config  # noqa: E402
from data_utils.path_guard import protected_problem  # noqa: E402
from ids import image_key  # noqa: E402  (pure helper, no I/O)

# Default registry: config.registry_root() (one <entity>.jsonl per entity; every *.jsonl in it is read),
# resolved in main() so importing this module never needs the data folders. Default page: <registry>/history.html

# Page lanes, from the contract's "Statuses (queue and history view)" table.
LANES = [
    ("queued", {"dataset": {"frozen"}, "job_bundle": {"bundled", "sent"}, "run": {"queued"}}),
    ("staged", {"inference_check": {"staged"}, "review_package": {"built"}}),
    ("in review", {"review_package": {"in_review"}}),
    ("reviewed", {"review_package": {"reviewed"}, "inference_check": {"reviewed"}}),
    ("applied", {"label_version": {"created", "qc_passed"}, "review_package": {"applied"}}),
    ("pushed", {"label_version": {"pushed"}}),
    ("trained", {"run": {"running", "finished", "failed"}, "dataset": {"trained"}, "job_bundle": {"returned"}}),
    ("scored", {"inference_check": {"scored"}, "model": {"candidate", "production", "retired"}}),
    ("closed", {"review_package": {"superseded", "abandoned"}, "label_version": {"superseded"},
                "session": {"excluded"}}),
]
TREE_ENTITIES = ("pull", "roboflow_export", "label_version", "review_package", "inference_check",
                 "dataset", "job_bundle", "run", "model")
WHO_FIELDS = ("who", "reviewer", "reviewers", "recorded_by", "tool")
WHEN_FIELDS = ("logged", "created", "built", "uploaded")
JOB_LANES = (("annotate", "Annotate"), ("revise", "Revise"), ("train", "Train"), ("check", "Check"))
OPEN_ORDER = {"blocked": 0, "in_progress": 1, "ready": 2}       # anything else open sorts last
CLOSED = {"done", "cancelled"}
APP_PORT = 8766                                                 # review app port (NEXT_ITERATION.md)
SPLITS = ("train", "valid", "test")
EXCLUDED_LABEL = {"20261012": "20261012 lipo"}                  # sessions left out of the registry by design


# ---------------------------------------------------------------- reading
# Own small reader with the same semantics as registry/registry.py (piece 4): last line per id wins, plus
# the history list. It can switch to registry.current() / registry.history(id) once that module lands.

def entity_of(rec: dict) -> str | None:
    s = rec.get("schema")
    if isinstance(s, str) and "/" in s:
        return s.split("/", 1)[0]
    if "batch_id" in rec:  # legacy batches.jsonl line (no schema): inference_check
        return "inference_check"
    return None


def input_files(paths) -> list[Path]:
    out = []
    for p in paths:
        p = Path(p)
        out.extend(sorted(p.glob("*.jsonl")) if p.is_dir() else [p])
    return out


def read_registry(paths, changes: list | None = None):
    """-> (current {key: rec}, history {key: [rec, ...]}, stats). key = '<entity>|<id>'.
    current[key] is the merged view (earlier fields overlaid by later lines); its status is the last line's.
    label_change rows are not records; with `changes` given, (label_version, image) of each row is appended."""
    history: dict[str, list[dict]] = {}
    order: list[str] = []
    stats = {"lines": 0, "bad_json": 0, "no_id": 0, "label_change_rows": 0}
    for f in input_files(paths):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                stats["lines"] += 1
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    stats["bad_json"] += 1
                    continue
                if not isinstance(rec, dict):
                    stats["bad_json"] += 1
                    continue
                ent = entity_of(rec)
                if ent == "label_change":
                    stats["label_change_rows"] += 1
                    if changes is not None:
                        changes.append((rec.get("label_version"), rec.get("image")))
                    continue
                rid = rec.get("id", rec.get("batch_id"))
                if rid is None or ent is None:
                    stats["no_id"] += 1
                    continue
                key = f"{ent}|{rid}"
                if key not in history:
                    history[key] = []
                    order.append(key)
                history[key].append(rec)
    current = {}
    for key in order:
        merged: dict = {}
        for rec in history[key]:
            merged.update(rec)
        last = history[key][-1]
        base = owner_fields(key.split("|", 1)[0], merged.get("owner"))
        merged = {**base, **merged}
        merged["status"] = last.get("status", base.get("status"))
        merged["id"] = str(last.get("id", last.get("batch_id")))
        current[key] = merged
    return current, history, stats


def owner_fields(ent: str, owner) -> dict:
    """Pointer records: fields from the owner file (`<version.json>` or `<batches.jsonl>#<batch_id>`), read-only.
    Same rule as registry.derive(); a missing or unreadable owner file gives {}."""
    if not owner or ent not in ("label_version", "inference_check"):
        return {}
    path, _, bid = str(owner).partition("#")
    try:
        if ent == "label_version":
            o = json.loads(Path(path).read_text(encoding="utf-8"))
            return o if isinstance(o, dict) else {}
        last = {}
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    o = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(o, dict) and o.get("batch_id") == bid:
                    last = o
        return {k: v for k, v in last.items() if k != "logged"}
    except (OSError, ValueError):
        return {}


# ---------------------------------------------------------------- model

def lane_of(ent: str, status) -> str | None:
    for lane, m in LANES:
        if status in m.get(ent, ()):
            return lane
    return None


def _ids_in(v):
    if isinstance(v, str):
        yield v
    elif isinstance(v, list):
        for x in v:
            if isinstance(x, str):
                yield x
    elif isinstance(v, dict) and isinstance(v.get("id"), str):  # dataset.source {kind, id}
        yield v["id"]


def build_model(current: dict, history: dict, stats: dict, sources, changes=None) -> dict:
    by_id: dict[str, list[str]] = {}
    for key, rec in current.items():
        by_id.setdefault(rec["id"], []).append(key)

    def find(ent, rid):
        k = f"{ent}|{rid}"
        return k if isinstance(rid, str) and k in current else None

    # derived reverse links
    superseded_by = {}
    for key, rec in current.items():
        if key.startswith("label_version|") and isinstance(rec.get("supersedes"), str):
            superseded_by.setdefault(rec["supersedes"], []).append(rec["id"])

    lv_for_pkg = {}
    for key, rec in current.items():
        if key.startswith("label_version|") and find("review_package", rec.get("pkg")):
            lv_for_pkg.setdefault(rec["pkg"], key)

    def parent_of(key, rec):
        ent = key.split("|", 1)[0]
        if ent == "roboflow_export":
            return find("roboflow_export", rec.get("derived_from"))
        if ent == "label_version":
            return find("pull", rec.get("base"))
        if ent == "review_package":
            return (find("review_package", rec.get("parent")) or lv_for_pkg.get(rec["id"])
                    or find("model", rec.get("model")) or find(rec.get("base_kind") or "pull", rec.get("base"))
                    or find("pull", rec.get("base")) or find("roboflow_export", rec.get("base")))
        if ent == "inference_check":
            return find("model", rec.get("model"))
        if ent == "dataset":
            src = rec.get("source") if isinstance(rec.get("source"), dict) else {}
            return (find(src.get("kind") or "", src.get("id")) or find("label_version", rec.get("label_version"))
                    or find("roboflow_export", rec.get("export")))
        if ent == "job_bundle":
            return find("dataset", rec.get("dataset"))
        if ent == "run":
            return find("job_bundle", rec.get("job_bundle")) or find("dataset", rec.get("dataset"))
        if ent == "model":
            return find("run", rec.get("run")) or find("dataset", rec.get("dataset"))
        return None

    parent = {}
    for key, rec in current.items():
        if key.split("|", 1)[0] in TREE_ENTITIES:
            parent[key] = parent_of(key, rec)
    for key in list(parent):  # break cycles: a node that reaches itself becomes a root
        seen, p = {key}, parent[key]
        while p:
            if p in seen:
                parent[key] = None
                break
            seen.add(p)
            p = parent.get(p)
    children: dict[str, list[str]] = {}
    for key, p in parent.items():
        if p:
            children.setdefault(p, []).append(key)

    def sort_key(k):
        r = current[k]
        return (str(r.get("created") or r.get("built") or r.get("logged") or ""), r["id"])

    def node(k):
        return {"k": k, "c": [node(c) for c in sorted(children.get(k, []), key=sort_key)]}

    roots = [k for k, p in parent.items() if not p]
    sections = []
    for title, ents in (("Roboflow pulls", ("pull",)), ("Roboflow exports", ("roboflow_export",)),
                        ("Unlinked", tuple(e for e in TREE_ENTITIES if e not in ("pull", "roboflow_export")))):
        rs = sorted((k for k in roots if k.split("|", 1)[0] in ents), key=sort_key)
        if rs:
            sections.append({"title": title, "nodes": [node(k) for k in rs]})
    other: dict[str, list[str]] = {}
    for key in current:
        ent = key.split("|", 1)[0]
        if ent not in TREE_ENTITIES:
            other.setdefault(ent, []).append(key)
    for ent in sorted(other):
        sections.append({"title": f"Other: {ent}", "nodes": [{"k": k, "c": []} for k in sorted(other[ent])],
                         "collapsed": True})

    recs = {}
    for key, rec in current.items():
        ent = key.split("|", 1)[0]
        links = []
        for field, v in rec.items():
            if field in ("id", "schema"):
                continue
            for rid in _ids_in(v):
                for tk in by_id.get(rid, []):
                    if tk != key:
                        links.append([field, tk])
        if ent == "label_version":
            for s in superseded_by.get(rec["id"], []):
                tk = find("label_version", s)
                if tk:
                    links.append(["superseded by (derived)", tk])
        for c in children.get(key, []):
            links.append([f"child {c.split('|', 1)[0]}", c])
        hist = []
        for line in history[key]:
            hist.append({"logged": line.get("logged"), "status": line.get("status"),
                         "by": line.get("recorded_by") or line.get("tool"), "fields": len(line)})
        recs[key] = {
            "e": ent, "id": rec["id"], "s": rec.get("status"), "lane": lane_of(ent, rec.get("status")),
            "who": {f: rec[f] for f in WHO_FIELDS if rec.get(f) not in (None, [], "")},
            "when": {f: rec[f] for f in WHEN_FIELDS if rec.get(f)},
            "gaps": {f: rec[f] for f in rec if f.endswith("_gap") and rec[f]},
            "copy_of": rec.get("copy_of"),
            "sup": rec.get("supersedes") if ent == "label_version" else None,
            "supby": superseded_by.get(rec["id"], []) if ent == "label_version" else [],
            "dim": ent == "label_version" and (rec.get("status") == "superseded" or bool(superseded_by.get(rec["id"]))),
            "prod": ent == "model" and rec.get("status") == "production",
            "links": links, "hist": hist, "raw": rec,
        }
    counts: dict[str, int] = {}
    for r in recs.values():
        counts[r["e"]] = counts.get(r["e"], 0) + 1

    # queue: job cards per type lane
    queue = []
    for typ, title in JOB_LANES:
        keys = [k for k in current if k.startswith("job|") and current[k].get("type") == typ]
        for k in keys:
            recs[k]["job"] = job_card(current[k], find)
        opened = sorted((k for k in keys if current[k].get("status") not in CLOSED),
                        key=lambda k: (OPEN_ORDER.get(current[k].get("status"), 3), sort_key(k)))
        closed = sorted((k for k in keys if current[k].get("status") in CLOSED), key=sort_key, reverse=True)
        queue.append({"type": typ, "title": title, "open": opened, "done": closed})
    other_jobs = [k for k in current if k.startswith("job|") and current[k].get("type") not in dict(JOB_LANES)]
    for k in other_jobs:
        recs[k]["job"] = job_card(current[k], find)

    # model pages
    roots = list_roots(sources)
    lists: dict[str, dict] = {}
    for key, rec in current.items():
        if key.startswith("model|"):
            recs[key]["mp"] = model_page(key, rec, current, find, roots, lists)
    rank = {"production": 0, "candidate": 1}
    models = sorted((k for k in current if k.startswith("model|")), key=lambda k: current[k]["id"], reverse=True)
    models.sort(key=lambda k: rank.get(current[k].get("status"), 2))
    return {
        "generated": _dt.datetime.now().isoformat(timespec="seconds"),
        "sources": [str(s) for s in sources],
        "stats": stats, "counts": counts,
        "lanes": [lane for lane, _ in LANES],
        "recs": recs, "sections": sections,
        "production": [k for k, r in recs.items() if r["prod"]],
        "queue": queue, "other_jobs": other_jobs, "models": models, "lists": lists,
        "graph": graph_model(current, recs, find, sort_key),
        "lk": lookup_index(current, roots, changes or []),
    }


# ---------------------------------------------------------------- lineage graph
# Columns label sets -> datasets -> models, plus a revise column. Edges are parent -> child from `inputs[]`
# (legacy fields when a record has none). Review packages and revise/annotate jobs sit right of the models:
# model -> package/job (predictions) and package/job -> label version (edits / outputs) is the back-edge.

GRAPH_COLS = ("Roboflow labels", "Local labels", "Datasets", "Models", "Review / revise")
GRAPH_COL = {"roboflow_export": 0, "pull": 0, "label_version": 1, "prediction_set": 1, "dataset": 2, "model": 3,
             "review_package": 4, "job": 4}
GRAPH_JOBS = ("revise", "annotate")
HIST_STATUSES = {"superseded", "abandoned", "cancelled", "retired"}


def _split_counts(rec) -> str | None:
    ct = rec.get("counts_trained") if isinstance(rec.get("counts_trained"), dict) else None
    if ct is None and isinstance(rec.get("counts"), dict):
        ct = {s: v.get("images") for s, v in rec["counts"].items() if isinstance(v, dict)}
    if not ct:
        return None
    names = [s for s in SPLITS if s in ct] + sorted(set(ct) - set(SPLITS))
    if all(ct[s] is None for s in names):
        return "counts unknown"
    return "/".join("?" if ct[s] is None else str(ct[s]) for s in names) + " " + "/".join(names)


def _graph_sub(ent, rec) -> str:
    if ent == "dataset":
        return _split_counts(rec) or ""
    if ent == "roboflow_export":
        c = _split_counts(rec)
        return c or (rec.get("note") or "")
    if ent in ("pull", "review_package"):
        n = rec.get("n_images")
        return (f"{n} images" if n is not None else "") + (f" · {rec['kind']}" if rec.get("kind") == "part" else "")
    if ent == "label_version":
        n = rec.get("n_images_applied")
        return f"{n} images changed" if n is not None else ""
    if ent == "job":
        p = rec.get("progress") if isinstance(rec.get("progress"), dict) else {}
        return f"{rec.get('type')} · {p.get('done', '?')}/{p.get('total', '?')} {p.get('unit') or ''}".strip()
    if ent == "model":
        return f"init {rec['init_from']}" if rec.get("init_from") else ""
    return ""


def graph_model(current, recs, find, sort_key) -> dict:
    """{cols, nodes: [[key, col, hist 0/1, part_of node index or -1, sub]], edges: [[from, to, role]]}.
    hist = hidden by default (superseded / abandoned / cancelled / retired, or a label version with a successor)."""
    keys = [k for k, r in current.items() if k.split("|", 1)[0] in GRAPH_COL
            and (not k.startswith("job|") or r.get("type") in GRAPH_JOBS)]
    keys.sort(key=lambda k: (GRAPH_COL[k.split("|", 1)[0]], sort_key(k)))
    idx = {k: i for i, k in enumerate(keys)}
    edges: dict = {}

    def add(parent, child, role):
        if parent in idx and child in idx and parent != child:
            edges.setdefault((idx[parent], idx[child]), role)

    def resolve(x):
        rid, ent = (x.get("id"), x.get("entity")) if isinstance(x, dict) else (x, None)
        if not isinstance(rid, str):
            return None
        return (find(ent, rid) if ent else None) or next((f for f in (find(e, rid) for e in ENTITY_GUESS) if f), None)

    for k in keys:
        rec, ent = current[k], k.split("|", 1)[0]
        ins = [x for x in rec.get("inputs") or [] if isinstance(x, dict)]
        for x in ins:
            if GRAPH_COL[ent] == 4 and x.get("role") == "base":
                continue        # the revise column's base (pull/export) would cross every column; the lv shows it
            add(resolve(x), k, x.get("role") or "input")
        if not ins:             # legacy fields (records written before inputs[])
            if ent == "dataset":
                s = rec.get("source") if isinstance(rec.get("source"), dict) else {}
                add(find(s.get("kind") or "", s.get("id")) or find("label_version", rec.get("label_version"))
                    or find("roboflow_export", rec.get("export")), k, "source")
            elif ent == "model":
                add(find("dataset", rec.get("dataset")), k, "train_data")
                add(find("model", rec.get("init_from")), k, "init_weights")
            elif ent == "label_version":
                add(find("pull", rec.get("base")) or find("roboflow_export", rec.get("base")), k, "base")
                add(find("review_package", rec.get("pkg")), k, "edits")
        if ent == "roboflow_export":
            add(find("roboflow_export", rec.get("derived_from")), k, "derived_from")
        if ent == "review_package" and not find("review_package", rec.get("parent")):
            add(find("model", rec.get("model")), k, "predictions")
        if ent == "job":
            for x in rec.get("outputs") or []:
                add(k, resolve(x), "output")
    nodes = []
    for k in keys:
        rec, ent = current[k], k.split("|", 1)[0]
        hist = rec.get("status") in HIST_STATUSES or bool(recs[k].get("dim"))
        part = idx.get(find("review_package", rec.get("parent"))) if ent == "review_package" else None
        nodes.append([k, GRAPH_COL[ent], int(hist), -1 if part is None else part, _graph_sub(ent, rec)])
    return {"cols": list(GRAPH_COLS), "nodes": nodes,
            "edges": [[a, b, role] for (a, b), role in edges.items()]}


# ---------------------------------------------------------------- image lookup
# ONE index: image key -> refs into a shared entity-key list. ref = n (entity index) or [n, note]
# (split for a dataset / Roboflow image, number of change rows for a label version).
# Only images that are records (image/1, or a roboflow_image stem) are indexed, as on the model page;
# other names from lists and logs are counted, never shown.

def read_change_log(owner) -> list | None:
    """<label version folder>/changes.jsonl next to the owner version.json -> [image, ...] (one per row), or None."""
    if not owner:
        return None
    f = Path(str(owner).partition("#")[0]).parent / "changes.jsonl"
    if not f.is_file():
        return None
    out = []
    with open(f, encoding="utf-8") as fh:
        for line in fh:
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(o, dict) and isinstance(o.get("image"), str):
                out.append(o["image"])
    return out


def package_images(rec, roots) -> list | None:
    """Image keys of a review package, from <package path>/data.js (window.REVIEW_DATA = {...}), or None."""
    p = rec.get("path")
    if not p:
        return None
    cands = [Path(p)] if Path(p).is_absolute() else [b / p for r in roots for b in (r, r.parent, r.parent.parent)]
    f = next((c / "data.js" for c in cands if (c / "data.js").is_file()), None)
    if f is None:
        return None
    try:
        text = f.read_text(encoding="utf-8")
        obj, _ = json.JSONDecoder().raw_decode(text, text.index("{"))
    except (OSError, ValueError):
        return None
    return [image_key(im["file"]) for im in obj.get("images") or [] if isinstance(im, dict) and im.get("file")]


def lookup_index(current, roots, changes) -> dict:
    """-> {"e": [entity key, ...], "i": [[image key, [ref, ...]], ...] sorted, "hidden": n, "sources": {...}}."""
    known = {rec["id"] for k, rec in current.items() if k.startswith("image|")}
    rf_of = {}
    for k, rec in current.items():
        if k.startswith("roboflow_image|") and isinstance(rec.get("stem"), str):
            rf_of.setdefault(image_key(rec["stem"]), []).append(k)
    known |= set(rf_of)
    ents, eidx, img, hidden = [], {}, {name: [] for name in known}, set()
    used = {"change_log": [], "label_change_rows": [], "package_data_js": [], "no_image_list": []}

    def add(name, k, note=None):
        if name not in known:
            hidden.add(name)
            return
        if k not in eidx:
            eidx[k] = len(ents)
            ents.append(k)
        ref = eidx[k] if note is None else [eidx[k], note]
        if ref not in img[name]:
            img[name].append(ref)

    for name, ks in rf_of.items():
        for k in ks:
            add(name, k, current[k].get("split") or "")
    for k, rec in current.items():
        if k.startswith("dataset|") and isinstance(rec.get("image_list"), dict):
            for stem, split in read_list_file(rec["image_list"], roots)["rows"] or []:
                add(stem, k, split or "?")
    reg_rows: dict = {}
    for lv, im in changes:
        if isinstance(lv, str) and isinstance(im, str):
            reg_rows.setdefault(lv, []).append(im)
    for k, rec in current.items():
        if not k.startswith("label_version|"):
            continue
        rows = read_change_log(rec.get("owner"))
        used["change_log" if rows else "label_change_rows"].append(rec["id"])
        if not rows:
            rows = reg_rows.get(rec["id"], [])
        n: dict = {}
        for im in rows:
            n[im] = n.get(im, 0) + 1
        for im, c in n.items():
            add(image_key(im), k, c)
    pkg_imgs: dict = {}
    for k, rec in current.items():
        if k.startswith("review_package|"):
            got = package_images(rec, roots)
            if got is None:
                used["no_image_list"].append(rec["id"])
                continue
            used["package_data_js"].append(rec["id"])
            pkg_imgs[rec["id"]] = got
            for name in got:
                add(name, k)
    for k, rec in current.items():
        if not k.startswith("job|"):
            continue
        names = list(pkg_imgs.get(rec.get("package"), []))
        if isinstance(rec.get("image_list"), dict):
            names += [stem for stem, _ in read_list_file(rec["image_list"], roots)["rows"] or []]
        for name in names:
            add(image_key(name), k)
    return {"e": ents, "i": [[name, img[name]] for name in sorted(img)], "hidden": len(hidden), "sources": used}


def next_action(rec: dict) -> str:
    st, typ, where = rec.get("status"), rec.get("type"), rec.get("where")
    if st == "blocked":
        return rec.get("blocked_reason") or "Blocked (no reason recorded)"
    if st == "ready":
        return "Start"
    if st == "in_progress":
        if where == "app":
            port = rec.get("port") or APP_PORT
            return f"{'Review' if typ == 'revise' else 'Label'} in app (port {port})"
        return {"annotate": f"Label in {where}", "revise": f"Review in {where}",
                "train": f"Training on {where}: import the run when it returns",
                "check": f"Score the check on {where}"}.get(typ, f"Continue on {where}")
    if st == "done":
        return "Done"
    if st == "cancelled":
        return "Cancelled"
    return f"Status {st!r}"


def job_card(rec: dict, find) -> dict:
    def ref(x):
        if isinstance(x, dict):
            rid, role, ent = x.get("id"), x.get("role"), x.get("entity")
        else:
            rid, role, ent = x, None, None
        k = None
        if isinstance(rid, str):
            k = find(ent, rid) if ent else None
            if not k:
                k = next((f for f in (find(e, rid) for e in ENTITY_GUESS) if f), None)
        return [role, rid, k]
    prog = rec.get("progress") if isinstance(rec.get("progress"), dict) else None
    return {"name": rec.get("name") or rec["id"], "type": rec.get("type"), "where": rec.get("where"),
            "n": rec.get("n_images"), "progress": prog, "who": rec.get("who"), "package": rec.get("package"),
            "inputs": [ref(x) for x in rec.get("inputs") or []], "outputs": [ref(x) for x in rec.get("outputs") or []],
            "blocked": rec.get("blocked_reason") if rec.get("status") == "blocked" else None,
            "next": next_action(rec)}


ENTITY_GUESS = ("pull", "roboflow_export", "label_version", "review_package", "dataset", "model", "run",
                "inference_check", "job_bundle", "session", "job")


def list_roots(sources) -> list[Path]:
    out = []
    for s in sources:
        p = Path(s)
        out.append(p if p.is_dir() else p.parent)
    return out


def read_list_file(il: dict, roots) -> dict:
    """image_list {file, sha256} -> {"rows": [[stem, split], ...], "file": path, "problem": str|None}."""
    f = str(il.get("file") or "")
    cands = [Path(f)] if Path(f).is_absolute() else [r / f for r in roots]
    path = next((c for c in cands if c.is_file()), None)
    if not f or path is None:
        return {"rows": None, "file": f, "problem": "list file not found next to the registry"}
    data = path.read_bytes()
    problem = None
    if il.get("sha256") and hashlib.sha256(data).hexdigest() != il["sha256"]:
        problem = "list file changed since it was recorded (sha256 differs)"
    rows = []
    for line in data.decode("utf-8-sig").splitlines():
        parts = line.split("\t")
        if parts[0].strip():
            rows.append([parts[0].strip(), parts[1].strip() if len(parts) > 1 else ""])
    return {"rows": rows, "file": str(path), "problem": problem}


def model_page(key, rec, current, find, roots, lists) -> dict:
    """Which labeled images trained / validated / tested this model (dataset image_list), plus links."""
    inputs = {x.get("role"): x.get("id") for x in rec.get("inputs") or [] if isinstance(x, dict)}
    ds_id = inputs.get("train_data") or rec.get("dataset")
    dk = find("dataset", ds_id)
    ds = current.get(dk, {}) if dk else {}
    src = ds.get("source") if isinstance(ds.get("source"), dict) else {}
    src_id = next((x.get("id") for x in ds.get("inputs") or [] if isinstance(x, dict) and x.get("role") == "source"),
                  src.get("id") or ds.get("export") or ds.get("label_version"))
    src_kind = src.get("kind") or ("label_version" if str(src_id).startswith("lv_") else "roboflow_export")
    sk = find(src_kind, src_id) or find("roboflow_export", src_id) or find("label_version", src_id)
    trained = ds.get("counts_trained") if isinstance(ds.get("counts_trained"), dict) else None
    listed = ds.get("counts_listed") if isinstance(ds.get("counts_listed"), dict) else None
    seen = set(trained or {}) | set(listed or {})
    names = list(SPLITS) if not seen else [s for s in SPLITS if s in seen] + sorted(seen - set(SPLITS))
    splits = [{"split": s, "trained": None if trained is None else trained.get(s, 0),
               "listed": None if listed is None else listed.get(s, 0)} for s in names]   # None = unknown
    note = None
    if trained is not None and listed is not None:
        gap = sum(trained.values()) - sum(listed.values())
        if gap:
            nl = ds.get("not_listed") if isinstance(ds.get("not_listed"), dict) else {}
            excl = [EXCLUDED_LABEL.get(s, s) for s in ds.get("excluded_sessions") or []] or \
                (list(EXCLUDED_LABEL.values()) if nl.get("excluded_session") else [])
            parts = []
            if excl:
                parts.append(f"{', '.join(excl)}, excluded from registry")
            if nl.get("not_in_registry") or not excl:
                parts.append(f"{nl.get('not_in_registry') or ''} not in the registry".strip())
            note = f"+{gap} trained on, not listed ({'; '.join(parts)})"
    lk = None
    il = ds.get("image_list") if isinstance(ds.get("image_list"), dict) else None
    if il and dk:
        lk = dk
        if dk not in lists:
            got = read_list_file(il, roots)
            rows, hidden = None, 0
            if got["rows"] is not None:
                rows = []
                for stem, split in got["rows"]:
                    im = current.get(f"image|{stem}")
                    if im is None:                      # not in the registry (e.g. excluded session): count only
                        hidden += 1
                        continue
                    rows.append([stem, split, im.get("session") or ""])
            lists[dk] = {"rows": rows, "hidden": hidden, "file": got["file"], "problem": got["problem"]}
    init_id = inputs.get("init_weights") or rec.get("init_from")
    scores = []
    for k, r in current.items():
        if k.startswith("inference_check|") and r.get("model") in (rec["id"], *(rec.get("aliases") or [])):
            scores.append({"k": k, "id": r["id"], "status": r.get("status"), "session": r.get("session"),
                           "n": r.get("n"), "metrics": r.get("metrics") or r.get("scores")})
    status = rec.get("status")
    return {"badge": status if status in ("production", "candidate", "retired") else None,
            "dataset": [ds_id, dk], "source": [src_kind, src_id, sk] if src_id else None,
            "derived_from": current.get(sk, {}).get("derived_from") if sk else None,
            "splits": splits if ds else [], "note": note, "list": lk,
            "list_gap": None if il else (ds.get("image_list_gap") or ("no dataset" if not ds else "no image list recorded")),
            "init": [init_id, find("model", init_id)] if init_id else None,
            "run": [rec.get("run"), find("run", rec.get("run"))] if rec.get("run") else None,
            "scores": sorted(scores, key=lambda x: x["id"]),
            "metrics": rec.get("metrics") or rec.get("scores")}


# ---------------------------------------------------------------- HTML

def safe_json(obj) -> str:
    """JSON safe inside a <script> element: no '<', '>', '&' (so no '</script>' or '<!--')."""
    s = json.dumps(obj, ensure_ascii=False, default=str)
    return s.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e") \
        .replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")


CSS = r"""
:root{--bg:#fafaf9;--fg:#1c1917;--mut:#6b6560;--card:#fff;--line:#e3e0dc;--acc:#2456c7;--accbg:#e8eefc;
--prod:#0f7a3d;--prodbg:#dcf3e5;--warn:#9a5b00;--warnbg:#fdf0d8;--dim:.5}
@media (prefers-color-scheme:dark){:root{--bg:#141312;--fg:#ece9e5;--mut:#a39e98;--card:#1e1c1b;--line:#35322f;
--acc:#8fb0ff;--accbg:#1f2a44;--prod:#6fdc9c;--prodbg:#163323;--warn:#f2c26b;--warnbg:#3a2c12}}
*{box-sizing:border-box}html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
header,main{max-width:1280px;margin:0 auto;padding:12px 16px}
h1{font-size:1.25rem;margin:4px 0}h2{font-size:1.05rem;margin:18px 0 8px}
.meta{color:var(--mut);font-size:.85rem;overflow-wrap:anywhere}
.prodline{margin:8px 0;padding:8px 10px;border-radius:8px;background:var(--prodbg);color:var(--prod);font-weight:600;overflow-wrap:anywhere}
.filters{display:flex;flex-wrap:wrap;gap:8px;margin:10px 0}
.filters select,.filters input{font:inherit;padding:8px;border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--fg);min-height:40px}
.filters input{flex:1 1 180px;min-width:0}
.models{display:flex;flex-wrap:wrap;gap:4px 8px;align-items:baseline;margin:4px 0}
.queue{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:10px}
.qlane{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:8px;min-width:0}
.qlane h3{margin:0 0 6px;font-size:1rem;display:flex;justify-content:space-between;gap:6px;align-items:baseline}
.qlane .n{color:var(--mut);font-size:.8rem;font-weight:400}
.qlane.empty{opacity:.6}
.qlane>details>summary{cursor:pointer;color:var(--mut);font-size:.85rem;padding:4px 0}
.card{border:1px solid var(--line);border-radius:8px;padding:8px;margin:6px 0;background:var(--bg);min-width:0}
.card .top{display:flex;flex-wrap:wrap;gap:4px 8px;align-items:baseline}
.card .name{font:inherit;font-weight:600;color:var(--fg);background:none;border:0;padding:2px 0;cursor:pointer;text-align:left;overflow-wrap:anywhere;min-width:0}
.card .row{font-size:.82rem;color:var(--mut);overflow-wrap:anywhere;margin-top:3px}
.card .next{font-size:.85rem;margin-top:6px;font-weight:600}
.why{font-size:.82rem;margin-top:4px;padding:3px 6px;border-radius:6px;background:var(--warnbg);color:var(--warn);overflow-wrap:anywhere}
.bar{height:8px;border-radius:999px;background:var(--line);overflow:hidden;margin-top:6px}
.bar>i{display:block;height:100%;background:var(--acc)}
.chip{font-size:.72rem;padding:1px 7px;border-radius:999px;white-space:nowrap;border:1px solid var(--line)}
.chip.blocked{background:var(--warnbg);color:var(--warn);border-color:transparent}
.chip.in_progress{background:var(--accbg);color:var(--acc);border-color:transparent}
.chip.done{background:var(--prodbg);color:var(--prod);border-color:transparent}
.chip.cancelled{opacity:.7;text-decoration:line-through}
.mp{border-bottom:1px solid var(--line);padding-bottom:10px;margin-bottom:10px}
.mp .badge{display:inline-block;font-size:.78rem;font-weight:700;padding:2px 8px;border-radius:999px;margin-left:6px;vertical-align:middle}
.badge.production{background:var(--prodbg);color:var(--prod)}.badge.candidate{background:var(--accbg);color:var(--acc)}
.badge.other{border:1px solid var(--line);color:var(--mut)}
.mp table.splits td,.mp table.splits th{text-align:right}.mp table.splits td:first-child,.mp table.splits th:first-child{text-align:left}
.mp .unk{color:var(--warn)}
.mp .imgs{max-height:50vh;overflow:auto}
.mp .imgs td:first-child{font-family:ui-monospace,Consolas,monospace;font-size:.75rem;word-break:break-all}
.mp .imgs td:not(:first-child),.mp .imgs th{white-space:nowrap;overflow-wrap:normal;word-break:normal}
.layout{display:grid;grid-template-columns:1fr;gap:16px}
@media (min-width:900px){.layout{grid-template-columns:minmax(0,1fr) 400px}#detail{position:sticky;top:12px;max-height:calc(100vh - 24px);overflow:auto}}
@media (max-width:899px){#detail{position:fixed;left:0;right:0;bottom:0;max-height:70vh;overflow:auto;z-index:5;border-radius:14px 14px 0 0;box-shadow:0 -6px 24px rgba(0,0,0,.25)}#detail.hidden{display:none}}
.tree ul{list-style:none;margin:0;padding-left:14px;border-left:1px solid var(--line)}
.tree>ul{padding-left:0;border-left:0}
.tree li{margin:2px 0}
.tree summary{cursor:pointer;padding:2px 0}
.leaf{padding-left:16px}
.rec{font:inherit;color:inherit;background:none;border:0;padding:4px 2px;text-align:left;cursor:pointer;display:inline-flex;flex-wrap:wrap;gap:4px 6px;align-items:baseline;max-width:100%;min-height:32px}
.rec:hover .rid,.rec:focus .rid{text-decoration:underline}
.rid{font-family:ui-monospace,Consolas,monospace;font-size:.85rem;overflow-wrap:anywhere;word-break:break-all}
.tag{font-size:.72rem;padding:1px 6px;border-radius:999px;background:var(--accbg);color:var(--acc);white-space:nowrap}
.st{font-size:.72rem;padding:1px 6px;border-radius:999px;border:1px solid var(--line);color:var(--mut);white-space:nowrap}
.prod{background:var(--prodbg);color:var(--prod);font-weight:700}
.gap{background:var(--warnbg);color:var(--warn)}
.note{font-size:.78rem;color:var(--mut)}
.dim{opacity:var(--dim)}.ctx{opacity:.6}
.sel{outline:2px solid var(--acc);border-radius:6px}
.hide{display:none!important}
section.sec{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:8px 10px;margin-bottom:10px}
section.sec>details>summary{font-weight:600;cursor:pointer}
#detail{background:var(--card);border:1px solid var(--line);padding:12px}
#detail .close{float:right;font:inherit;padding:6px 12px;border:1px solid var(--line);border-radius:8px;background:var(--bg);color:var(--fg);cursor:pointer}
#detail dl{display:grid;grid-template-columns:max-content minmax(0,1fr);gap:4px 10px;margin:8px 0}
#detail dt{color:var(--mut);font-size:.85rem}#detail dd{margin:0;overflow-wrap:anywhere}
#detail table{border-collapse:collapse;width:100%;font-size:.85rem;table-layout:auto}#detail td,#detail th{border-top:1px solid var(--line);padding:3px 4px;text-align:left;vertical-align:top;overflow-wrap:anywhere}
#detail pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:.78rem;background:var(--bg);padding:8px;border-radius:8px}
.lnk{font:inherit;color:var(--acc);background:none;border:0;padding:2px 0;cursor:pointer;text-align:left;overflow-wrap:anywhere;word-break:break-all}
.layout>div{min-width:0}
.tabs{display:flex;flex-wrap:wrap;gap:6px;margin:18px 0 10px}
.tab{font:inherit;font-weight:600;padding:6px 14px;min-height:38px;border:1px solid var(--line);border-radius:999px;background:var(--card);color:var(--fg);cursor:pointer}
.tab.on{background:var(--accbg);color:var(--acc);border-color:transparent}
.gbar{display:flex;flex-wrap:wrap;gap:4px 14px;align-items:center;margin:0 0 8px}
.gbar label{display:inline-flex;gap:6px;align-items:center;min-height:36px;cursor:pointer}
.gwrap{overflow-x:auto;overflow-y:hidden;max-width:100%;border:1px solid var(--line);border-radius:10px;background:var(--card)}
.gcanvas{position:relative}
.gcanvas svg{position:absolute;left:0;top:0}
.ghead{position:absolute;top:6px;font-size:.75rem;font-weight:700;color:var(--mut);text-transform:uppercase;letter-spacing:.03em;white-space:nowrap}
.gnode{position:absolute;font:inherit;color:var(--fg);background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:4px 7px;text-align:left;cursor:pointer;overflow:hidden;display:flex;flex-direction:column;gap:2px}
.gnode:hover,.gnode:focus{border-color:var(--acc)}
.gnode.part{background:var(--card)}
.gnode.prodn{border:2px solid var(--prod)}
.gnode.dim{border-style:dashed}
.gnode .gl1{display:flex;gap:4px;align-items:baseline;white-space:nowrap;overflow:hidden}
.gnode .gid{font-family:ui-monospace,Consolas,monospace;font-size:.72rem;line-height:1.25;word-break:break-all;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
.gnode .gsub{font-size:.7rem;color:var(--mut);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.ge{fill:none;stroke:var(--mut);stroke-width:1.4}
.ge.back{stroke:var(--acc);stroke-dasharray:6 4}
.ga{fill:var(--mut)}.ga.back{fill:var(--acc)}
.gdim{opacity:.35}
.lcard{border:1px solid var(--line);border-radius:8px;padding:8px 10px;margin:8px 0;background:var(--card);min-width:0}
.lcard .lname{font-family:ui-monospace,Consolas,monospace;font-size:.85rem;font-weight:600;word-break:break-all}
.lrow{font-size:.85rem;margin-top:4px;overflow-wrap:anywhere}
.lrow>b{color:var(--mut);font-weight:600;margin-right:4px}
"""

JS = r"""
(function(){
var M=JSON.parse(document.getElementById('data').textContent);
var R=M.recs;
function el(t,cls,txt){var e=document.createElement(t);if(cls)e.className=cls;if(txt!=null)e.textContent=String(txt);return e;}
function recButton(k){
  var r=R[k],b=el('button','rec'+(r.dim?' dim':''));b.type='button';b.dataset.k=k;
  b.appendChild(el('span','tag',r.e));b.appendChild(el('span','rid',r.id));
  if(r.s!=null)b.appendChild(el('span','st'+(r.prod?' prod':''),r.prod?'PRODUCTION':r.s));
  if(r.prod&&r.s!=='production')b.appendChild(el('span','st',r.s));
  if(r.sup)b.appendChild(el('span','note','supersedes '+r.sup));
  if(r.supby.length)b.appendChild(el('span','note','superseded by '+r.supby.join(', ')));
  if(Object.keys(r.gaps).length)b.appendChild(el('span','st gap','gap'));
  if(r.copy_of)b.appendChild(el('span','note','copy of '+r.copy_of));
  b.addEventListener('click',function(ev){ev.preventDefault();show(k);});
  return b;
}
function renderNode(n){
  var li=el('li');var r=R[n.k];li.dataset.k=n.k;
  if(n.c.length){
    var d=el('details');d.open=true;var s=el('summary');s.appendChild(recButton(n.k));
    s.appendChild(el('span','note',' ('+n.c.length+')'));d.appendChild(s);
    var ul=el('ul');n.c.forEach(function(c){ul.appendChild(renderNode(c));});d.appendChild(ul);li.appendChild(d);
  }else{var w=el('div','leaf');w.appendChild(recButton(n.k));li.appendChild(w);}
  return li;
}
// header
document.getElementById('gen').textContent='generated '+M.generated+' from '+M.sources.join(', ');
var cnt=Object.keys(M.counts).sort().map(function(e){return e+' '+M.counts[e];}).join(' \u00b7 ');
var st=M.stats;
document.getElementById('counts').textContent=Object.keys(R).length+' records ('+cnt+') from '+st.lines+' lines'+
  (st.label_change_rows?'; '+st.label_change_rows+' label_change rows not shown':'')+
  (st.bad_json?'; '+st.bad_json+' unreadable lines':'')+(st.no_id?'; '+st.no_id+' lines without id/schema':'');
var pl=document.getElementById('prodline');
if(M.production.length){pl.textContent='Production model: ';M.production.forEach(function(k){pl.appendChild(recButton(k));});}
else{pl.textContent='No model is marked production.';pl.classList.add('note');}
// filters
var fe=document.getElementById('fe'),fs=document.getElementById('fs'),fq=document.getElementById('fq');
var ents={},sts={};Object.keys(R).forEach(function(k){ents[R[k].e]=1;if(R[k].s!=null)sts[R[k].s]=1;});
Object.keys(ents).sort().forEach(function(e){var o=el('option',null,e);o.value=e;fe.appendChild(o);});
Object.keys(sts).sort().forEach(function(s){var o=el('option',null,s);o.value=s;fs.appendChild(o);});
function match(k){var r=R[k],q=fq.value.trim().toLowerCase();
  return (!fe.value||r.e===fe.value)&&(!fs.value||String(r.s)===fs.value)&&(!q||r.id.toLowerCase().indexOf(q)>=0);}
// models strip (tap a model -> model page)
var ms=document.getElementById('models');
if(M.models.length){ms.appendChild(el('span','note','Models:'));M.models.forEach(function(k){ms.appendChild(recButton(k));});}
// queue: one lane per job type
function refBtn(ref){var role=ref[0],id=ref[1],k=ref[2];var w=el('span');
  if(role)w.appendChild(el('span','note',role+' '));
  if(k){var b=el('button','lnk',id);b.type='button';b.addEventListener('click',function(){show(k);});w.appendChild(b);}
  else w.appendChild(el('span',null,id==null?'?':id));return w;}
function card(k){var r=R[k],j=r.job,c=el('div','card');c.dataset.k=k;
  var top=el('div','top');var nb=el('button','name',j.name);nb.type='button';nb.addEventListener('click',function(){show(k);});
  top.appendChild(nb);top.appendChild(el('span','chip '+(r.s||''),String(r.s).replace('_',' ')));c.appendChild(top);
  c.appendChild(el('div','row',(j.n==null?'? images':j.n+' image'+(j.n===1?'':'s'))+' · '+(j.where||'?')+' · who: '+(j.who||'unassigned')));
  if(j.progress){var p=j.progress,tot=p.total;var bar=el('div','bar');var fill=el('i');
    fill.style.width=(tot?Math.min(100,Math.round(100*p.done/tot)):0)+'%';bar.appendChild(fill);c.appendChild(bar);
    c.appendChild(el('div','row',p.done+'/'+(tot==null?'?':tot)+' '+(p.unit||'')));}
  if(j.inputs.length||j.outputs.length){var io=el('div','row');
    j.inputs.forEach(function(x,i){if(i)io.appendChild(document.createTextNode(', '));io.appendChild(refBtn(x));});
    io.appendChild(document.createTextNode(' → '));
    if(j.outputs.length)j.outputs.forEach(function(x,i){if(i)io.appendChild(document.createTextNode(', '));io.appendChild(refBtn(x));});
    else io.appendChild(el('span',null,'(no outputs yet)'));c.appendChild(io);}
  if(j.blocked)c.appendChild(el('div','why','Blocked: '+j.blocked));
  c.appendChild(el('div','next','Next: '+j.next));
  return c;}
var queue=document.getElementById('queue');var cardEls=[];var laneEls=[];
M.queue.forEach(function(L){var d=el('div','qlane');var h=el('h3');h.appendChild(el('span',null,L.title));
  var n=el('span','n','');h.appendChild(n);d.appendChild(h);var openBox=el('div');d.appendChild(openBox);
  L.open.forEach(function(k){var c=card(k);openBox.appendChild(c);cardEls.push(c);});
  if(!L.open.length)openBox.appendChild(el('div','note','Nothing open.'));
  var dd=el('details');var sm=el('summary',null,'Done ('+L.done.length+')');dd.appendChild(sm);
  L.done.forEach(function(k){var c=card(k);dd.appendChild(c);cardEls.push(c);});
  if(L.done.length)d.appendChild(dd);
  queue.appendChild(d);laneEls.push({d:d,n:n,open:L.open,done:L.done,box:openBox});});
if(M.other_jobs.length){var d=el('div','qlane');d.appendChild(el('h3',null,'Other jobs'));
  M.other_jobs.forEach(function(k){var c=card(k);d.appendChild(c);cardEls.push(c);});queue.appendChild(d);}
// tree
var tree=document.getElementById('tree');
M.sections.forEach(function(sec){var s=el('section','sec');var d=el('details');d.open=!sec.collapsed;
  var sm=el('summary',null,sec.title+' ('+sec.nodes.length+')');d.appendChild(sm);
  var w=el('div','tree');var ul=el('ul');sec.nodes.forEach(function(n){ul.appendChild(renderNode(n));});
  w.appendChild(ul);d.appendChild(w);s.appendChild(d);tree.appendChild(s);});
function apply(){
  var active=!!(fe.value||fs.value||fq.value.trim());
  cardEls.forEach(function(c){c.classList.toggle('hide',!match(c.dataset.k));});
  laneEls.forEach(function(L){var o=L.open.filter(match).length,dn=L.done.filter(match).length;
    L.n.textContent=o+' open · '+dn+' done';L.d.classList.toggle('empty',!o);});
  function walk(li){var m=match(li.dataset.k),any=false;var sub=li.querySelector(':scope>details>ul');
    if(sub)Array.prototype.forEach.call(sub.children,function(c){if(walk(c))any=true;});
    li.classList.toggle('hide',active&&!m&&!any);li.classList.toggle('ctx',active&&!m&&any);
    if(active&&any){var d=li.querySelector(':scope>details');if(d)d.open=true;}
    return m||any;}
  Array.prototype.forEach.call(tree.querySelectorAll('section.sec'),function(sec){var any=false;
    Array.prototype.forEach.call(sec.querySelector('.tree>ul').children,function(li){if(walk(li))any=true;});
    sec.classList.toggle('hide',active&&!any);if(active&&any)sec.querySelector('details').open=true;});
  if(gboxes)markGraph();
}
[fe,fs].forEach(function(x){x.addEventListener('change',apply);});fq.addEventListener('input',apply);
// detail
var D=document.getElementById('detail');
function kv(dl,k,v){dl.appendChild(el('dt',null,k));var dd=el('dd');
  if(v instanceof Node)dd.appendChild(v);else dd.textContent=(typeof v==='object')?JSON.stringify(v):String(v);dl.appendChild(dd);}
function link(id,k){if(!k)return el('span',null,id==null?'?':id);var b=el('button','lnk',id);b.type='button';
  b.addEventListener('click',function(){show(k);});return b;}
function num(v){return v==null?el('span','unk','unknown'):el('span',null,v);}
function modelPage(r){var P=r.mp,box=el('section','mp');
  var h=el('h3',null,'Model page');if(P.badge)h.appendChild(el('span','badge '+(P.badge==='production'||P.badge==='candidate'?P.badge:'other'),P.badge));
  box.appendChild(h);var dl=el('dl');
  kv(dl,'trained on',P.dataset[0]?link(P.dataset[0],P.dataset[1]):el('span','unk','unknown (no dataset)'));
  if(P.source){var s=el('span');s.appendChild(el('span','note',P.source[0]+' '));s.appendChild(link(P.source[1],P.source[2]));
    if(P.derived_from)s.appendChild(el('span','note',' (re-split of '+P.derived_from+')'));kv(dl,'source',s);}
  if(P.run)kv(dl,'run',link(P.run[0],P.run[1]));
  kv(dl,'init model',P.init?link(P.init[0],P.init[1]):'none (from scratch / unknown)');
  box.appendChild(dl);
  if(P.splits.length){var t=el('table','splits');var hr=el('tr');['split','trained on','listed','not listed'].forEach(function(x){hr.appendChild(el('th',null,x));});t.appendChild(hr);
    var tt=0,tl=0,ut=false,ul=false;
    P.splits.forEach(function(x){var tr=el('tr');tr.appendChild(el('td',null,x.split));tr.appendChild(el('td',null,'')).appendChild(num(x.trained));
      tr.appendChild(el('td',null,'')).appendChild(num(x.listed));
      tr.appendChild(el('td',null,'')).appendChild(num(x.trained==null||x.listed==null?null:x.trained-x.listed));t.appendChild(tr);
      if(x.trained==null)ut=true;else tt+=x.trained;if(x.listed==null)ul=true;else tl+=x.listed;});
    var fr=el('tr');fr.appendChild(el('th',null,'total'));[ut?null:tt,ul?null:tl,(ut||ul)?null:tt-tl].forEach(function(v){fr.appendChild(el('th',null,'')).appendChild(num(v));});t.appendChild(fr);
    box.appendChild(t);}
  if(P.note)box.appendChild(el('div','note',P.note));
  if(P.list){var L=M.lists[P.list];
    if(L.problem)box.appendChild(el('div','why',L.problem));
    if(L.rows){var d=el('details','imgs');d.appendChild(el('summary',null,'Images ('+L.rows.length+' listed'+(L.hidden?', +'+L.hidden+' not in the registry, not shown':'')+')'));
      var it=el('table');var ih=el('tr');['image','split','session'].forEach(function(x){ih.appendChild(el('th',null,x));});it.appendChild(ih);
      L.rows.forEach(function(x){var tr=el('tr');tr.appendChild(el('td',null,x[0]));tr.appendChild(el('td',null,x[1]));tr.appendChild(el('td',null,x[2]||'?'));it.appendChild(tr);});
      d.appendChild(it);d.appendChild(el('div','note','from '+L.file));box.appendChild(d);}
  }else box.appendChild(el('div','note','Image list: unknown ('+(P.list_gap||'not recorded')+')'));
  var sc=el('div');sc.appendChild(el('strong',null,'Scores: '));
  if(P.scores.length)P.scores.forEach(function(x,i){if(i)sc.appendChild(document.createTextNode('; '));sc.appendChild(link(x.id,x.k));
    sc.appendChild(el('span','note',' '+(x.status||'')+(x.n!=null?', n='+x.n:'')+(x.metrics?', '+JSON.stringify(x.metrics):'')));});
  else sc.appendChild(el('span','note','no inference checks for this model yet'));
  if(P.metrics)sc.appendChild(el('div','note',JSON.stringify(P.metrics)));
  box.appendChild(sc);D.appendChild(box);}
function show(k){var r=R[k];D.textContent='';D.classList.remove('hidden');
  var c=el('button','close','Close');c.type='button';c.addEventListener('click',function(){D.classList.add('hidden');});D.appendChild(c);
  D.appendChild(el('h2',null,r.e+(r.prod?' (PRODUCTION)':'')));D.appendChild(el('div','rid',r.id));
  if(r.mp)modelPage(r);
  if(r.job){var jc=card(k);jc.querySelector('.name').disabled=true;D.appendChild(jc);}
  var dl=el('dl');kv(dl,'status',r.s==null?'(none)':r.s);if(r.lane)kv(dl,'lane',r.lane);
  Object.keys(r.who).forEach(function(f){kv(dl,f,r.who[f]);});Object.keys(r.when).forEach(function(f){kv(dl,f,r.when[f]);});
  Object.keys(r.gaps).forEach(function(f){kv(dl,f,r.gaps[f]);});if(r.copy_of)kv(dl,'copy_of',r.copy_of);
  if(r.sup)kv(dl,'supersedes',r.sup);if(r.supby.length)kv(dl,'superseded by',r.supby.join(', '));
  D.appendChild(dl);
  D.appendChild(el('h3',null,'Links ('+r.links.length+')'));
  if(r.links.length){var t=el('table');r.links.forEach(function(L){var tr=el('tr');tr.appendChild(el('td',null,L[0]));
    var td=el('td');var b=el('button','lnk',R[L[1]].e+': '+R[L[1]].id);b.type='button';b.addEventListener('click',function(){show(L[1]);});
    td.appendChild(b);tr.appendChild(td);t.appendChild(tr);});D.appendChild(t);}
  else D.appendChild(el('div','note','No links to other records in this registry.'));
  D.appendChild(el('h3',null,'Status history ('+r.hist.length+' line'+(r.hist.length===1?'':'s')+')'));
  var h=el('table');var hr=el('tr');['logged','status','by','fields'].forEach(function(x){hr.appendChild(el('th',null,x));});h.appendChild(hr);
  r.hist.forEach(function(x){var tr=el('tr');[x.logged,x.status,x.by,x.fields].forEach(function(v){tr.appendChild(el('td',null,v==null?'':v));});h.appendChild(tr);});
  D.appendChild(h);
  var dd=el('details');dd.appendChild(el('summary',null,'All fields (merged current record)'));dd.appendChild(el('pre',null,JSON.stringify(r.raw,null,1)));D.appendChild(dd);
  Array.prototype.forEach.call(document.querySelectorAll('.sel'),function(x){x.classList.remove('sel');});
  Array.prototype.forEach.call(document.querySelectorAll('.rec,.gnode'),function(x){if(x.dataset.k===k)x.classList.add('sel');});
  if(window.matchMedia('(min-width:900px)').matches)D.scrollTop=0;
}
// tabs
var panes=['tree','graph','lookup'];
function tab(p){panes.forEach(function(n){document.getElementById(n).classList.toggle('hide',n!==p);
  var t=document.getElementById('t-'+n);t.classList.toggle('on',n===p);t.setAttribute('aria-selected',String(n===p));});}
panes.forEach(function(n){document.getElementById('t-'+n).addEventListener('click',function(){tab(n);});});
// lineage graph: columns, boxes as buttons over an SVG edge layer (positions computed here)
var G=M.graph,gc=document.getElementById('gcanvas'),gsvg=document.getElementById('gsvg'),NS=gsvg.namespaceURI;
var gh=document.getElementById('ghist'),gboxes=[];
var BW=184,BH=74,GAP=60,VG=12,PAD=30,HEAD=30,LANE=7;
var SHORT={roboflow_export:'export',review_package:'package',label_version:'label version',prediction_set:'predictions'};
var nHist=G.nodes.filter(function(n){return n[2];}).length;
document.getElementById('ghistl').textContent='Show history (+'+nHist+' superseded/cancelled)';
function gbox(i,p){var n=G.nodes[i],k=n[0],r=R[k];
  var b=el('button','gnode'+(n[3]>=0?' part':'')+(r.prod?' prodn':'')+(n[2]?' dim':''));b.type='button';b.dataset.k=k;b.title=r.e+': '+r.id;
  b.style.left=p.x+'px';b.style.top=p.y+'px';b.style.width=p.w+'px';b.style.height=BH+'px';
  var l1=el('div','gl1');l1.appendChild(el('span','tag',SHORT[r.e]||r.e));
  if(r.prod)l1.appendChild(el('span','st prod','★ production'));else if(r.s!=null)l1.appendChild(el('span','st',String(r.s).replace('_',' ')));
  b.appendChild(l1);b.appendChild(el('div','gid',r.id));if(n[4])b.appendChild(el('div','gsub',n[4]));
  b.addEventListener('click',function(){show(k);});return b;}
function svgEl(t,cls){var e=document.createElementNS(NS,t);if(cls)e.setAttribute('class',cls);return e;}
function arrow(x,y,dir){return 'M'+x+','+y+'L'+(x-8*dir)+','+(y-4.5)+'L'+(x-8*dir)+','+(y+4.5)+'Z';}
function drawGraph(){
  var hist=gh.checked,ncol=G.cols.length;
  gboxes.forEach(function(b){b.remove();});gboxes=[];while(gsvg.firstChild)gsvg.removeChild(gsvg.firstChild);
  var vis=G.nodes.map(function(n){return hist||!n[2];});
  var cols=[],kids={},parents={},pos={},maxr=0;for(var c=0;c<ncol;c++)cols.push([]);
  G.nodes.forEach(function(n,i){if(!vis[i])return;if(n[3]>=0&&vis[n[3]]){(kids[n[3]]=kids[n[3]]||[]).push(i);return;}cols[n[1]].push(i);});
  G.edges.forEach(function(e){(parents[e[1]]=parents[e[1]]||[]).push(e[0]);});
  var back=G.edges.filter(function(e){return vis[e[0]]&&vis[e[1]]&&G.nodes[e[0]][1]>G.nodes[e[1]][1];});
  var top=HEAD+back.length*LANE+(back.length?10:0);
  var ch={};G.edges.forEach(function(e){(ch[e[0]]=ch[e[0]]||[]).push(e[1]);});
  function mean(ks){return ks.length?ks.reduce(function(a,k){return a+pos[k].r;},0)/ks.length:1e9;}
  function byParents(c){return function(i){return mean((parents[i]||[]).filter(function(p){return pos[p]&&pos[p].c<c;}));};}
  function place(c,score){   // order by score (mean row of linked boxes), then build order; parts follow their package
    var o=cols[c].map(function(i,j){return [score(i),j,i];});o.sort(function(a,b){return a[0]-b[0]||a[1]-b[1];});
    var r=0,x=PAD+c*(BW+GAP);
    o.forEach(function(t){pos[t[2]]={c:c,r:r,x:x,w:BW,y:top+r*(BH+VG)};r++;
      (kids[t[2]]||[]).forEach(function(j){pos[j]={c:c,r:r,x:x+18,w:BW-18,y:top+r*(BH+VG)};r++;});});
    return r;}
  cols.forEach(function(list,c){place(c,byParents(c));});
  place(0,function(i){return mean((ch[i]||[]).filter(function(k){return pos[k]&&pos[k].c>0;}));});  // 2nd pass: sources near their children
  for(var c2=1;c2<ncol;c2++)place(c2,byParents(c2));
  cols.forEach(function(list,c){maxr=Math.max(maxr,list.length+list.reduce(function(a,i){return a+(kids[i]||[]).length;},0));});
  var W=2*PAD+ncol*BW+(ncol-1)*GAP+back.length*LANE+10,H=top+maxr*(BH+VG)+PAD;
  gc.style.width=W+'px';gc.style.height=H+'px';gsvg.setAttribute('width',W);gsvg.setAttribute('height',H);
  G.cols.forEach(function(t,c){var h=el('div','ghead',t);h.style.left=(PAD+c*(BW+GAP))+'px';gc.appendChild(h);gboxes.push(h);});
  var same={},bl=0;
  G.edges.forEach(function(e){var a=pos[e[0]],b=pos[e[1]];if(!a||!b)return;
    var ay=a.y+BH/2,by=b.y+BH/2,d,dir,ex,cls='ge';
    if(a.c<b.c){var x1=a.x+a.w,x2=b.x,m=(x1+x2)/2;d='M'+x1+','+ay+'C'+m+','+ay+' '+m+','+by+' '+x2+','+by;dir=1;ex=x2;}
    else if(a.c===b.c){var n=same[a.c]=(same[a.c]||0)+1,lx=PAD+a.c*(BW+GAP)-8-n*LANE;
      d='M'+a.x+','+ay+'H'+lx+'V'+by+'H'+b.x;dir=1;ex=b.x;}
    else{var lane=bl++,band=HEAD+4+lane*LANE,rx=PAD+ncol*BW+(ncol-1)*GAP+6+lane*LANE,tx=b.x+b.w+10+lane*LANE;
      d='M'+(a.x+a.w)+','+ay+'H'+rx+'V'+band+'H'+tx+'V'+by+'H'+(b.x+b.w);dir=-1;ex=b.x+b.w;cls+=' back';}
    var g=svgEl('g',(G.nodes[e[0]][2]||G.nodes[e[1]][2])?'gdim':null);
    var t=svgEl('title');t.textContent=R[G.nodes[e[0]][0]].id+' → '+R[G.nodes[e[1]][0]].id+' ('+e[2]+')';g.appendChild(t);
    var p=svgEl('path',cls);p.setAttribute('d',d);g.appendChild(p);
    var h=svgEl('path',cls.replace('ge','ga'));h.setAttribute('d',arrow(ex,by,dir));g.appendChild(h);gsvg.appendChild(g);});
  G.nodes.forEach(function(n,i){if(pos[i]){var b=gbox(i,pos[i]);gc.appendChild(b);gboxes.push(b);}});
  markGraph();}
function markGraph(){var active=!!(fe.value||fs.value||fq.value.trim());
  gboxes.forEach(function(b){if(b.dataset.k)b.classList.toggle('ctx',active&&!match(b.dataset.k));});}
gh.addEventListener('change',drawGraph);
// image lookup: one embedded index (image -> refs into M.lk.e)
var LK=M.lk,lq=document.getElementById('lq'),lres=document.getElementById('lres');
var GROUPS=[['dataset','Datasets (split)'],['label_version','Label versions that changed it (change rows)'],
  ['review_package','Review packages'],['job','Jobs'],['roboflow_image','Roboflow image (split)']];
function norm(q){return q.replace(/\.(png|jpe?g|tiff?)$/i,'_$1').replace(/[\s-]+/g,'-').toLowerCase();}
function imgCard(x){var name=x[0],c=el('div','lcard');c.dataset.name=name;var ik=R['image|'+name]?'image|'+name:null;
  var h=el('div','lname');h.appendChild(link(name,ik));c.appendChild(h);
  var im=ik?R[ik].raw:null,row=el('div','lrow');row.appendChild(el('b',null,'Session'));
  row.appendChild(document.createTextNode(im?((im.session||'?')+' · grid '+(im.grid||'?')+(im.session_grid?' (session grid '+im.session_grid+')':'')):'no image record (Roboflow only)'));
  c.appendChild(row);
  GROUPS.forEach(function(g){var refs=x[1].filter(function(f){return LK.e[Array.isArray(f)?f[0]:f].split('|')[0]===g[0];});
    if(!refs.length&&g[0]==='roboflow_image')return;
    var r=el('div','lrow');r.appendChild(el('b',null,g[1]));
    if(!refs.length)r.appendChild(el('span','note','none'));
    refs.forEach(function(f,i){var k=LK.e[Array.isArray(f)?f[0]:f];if(i)r.appendChild(document.createTextNode(', '));
      r.appendChild(link(R[k].id,k));if(R[k].dim||R[k].s==='superseded'||R[k].s==='cancelled')r.appendChild(el('span','note',' ['+R[k].s+']'));
      if(Array.isArray(f))r.appendChild(el('span','note',' ('+f[1]+')'));});
    c.appendChild(r);});
  return c;}
function lookup(){var q=lq.value.trim();lres.textContent='';
  if(!q){lres.appendChild(el('div','note',LK.i.length+' images indexed'+(LK.hidden?' (+'+LK.hidden+' names in lists/logs not in the registry, not shown)':'')+'. Type an image name or part of it.'));return;}
  var ql=q.toLowerCase(),qn=norm(q);
  var hits=LK.i.filter(function(x){var n=x[0].toLowerCase();return n.indexOf(ql)>=0||n.indexOf(qn)>=0;});
  if(!hits.length){lres.appendChild(el('div','why','No match for “'+q+'”.'));return;}
  lres.appendChild(el('div','note',hits.length+' match'+(hits.length===1?'':'es')+(hits.length>50?', showing the first 50':'')));
  hits.slice(0,50).forEach(function(x){lres.appendChild(imgCard(x));});}
lq.addEventListener('input',lookup);
drawGraph();lookup();
apply();
})();
"""


def render_html(model: dict) -> str:
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Registry history</title>
<style>{CSS}</style></head>
<body>
<header><h1>Registry history</h1>
<div class="meta" id="gen"></div><div class="meta" id="counts"></div>
<div class="prodline" id="prodline"></div>
<div class="models" id="models"></div>
<div class="filters">
<select id="fe" aria-label="Entity"><option value="">All entities</option></select>
<select id="fs" aria-label="Status"><option value="">All statuses</option></select>
<input id="fq" type="search" placeholder="Search id" aria-label="Search id">
</div></header>
<main>
<div class="layout">
<div><h2>Queue</h2>
<div class="queue" id="queue"></div>
<div class="tabs" role="tablist">
<button type="button" class="tab on" role="tab" id="t-tree" data-p="tree" aria-selected="true">History tree</button>
<button type="button" class="tab" role="tab" id="t-graph" data-p="graph" aria-selected="false">Lineage graph</button>
<button type="button" class="tab" role="tab" id="t-lookup" data-p="lookup" aria-selected="false">Image lookup</button>
</div>
<div id="tree" class="pane"></div>
<div id="graph" class="pane hide">
<div class="gbar"><label><input type="checkbox" id="ghist"> <span id="ghistl">Show history</span></label>
<span class="note">Arrows: parent → child (inputs). Dashed blue: model predictions → review/revise → label version. ★ = production.</span></div>
<div class="gwrap" id="gwrap"><div class="gcanvas" id="gcanvas"><svg id="gsvg" aria-hidden="true"></svg></div></div>
</div>
<div id="lookup" class="pane hide">
<div class="filters"><input id="lq" type="search" placeholder="Image name or part of it" aria-label="Image name"></div>
<div id="lres"></div>
</div></div>
<aside id="detail" class="hidden"><div class="note">Tap a record for its detail.</div></aside>
</div>
</main>
<noscript>This page needs JavaScript (inline, no network).</noscript>
<script type="application/json" id="data">{safe_json(model)}</script>
<script>{JS}</script>
</body></html>
"""


def build(paths, out) -> dict:
    """Read registry paths, write the page to `out`. -> model."""
    out = Path(out)
    problem = protected_problem(out)
    if problem:
        raise SystemExit(f"refusing to write: {problem}")
    files = input_files(paths)
    if any(out.resolve() == f.resolve() for f in files):
        raise SystemExit(f"refusing to write: {out} is an input file")
    changes: list = []
    current, history, stats = read_registry(files, changes)
    model = build_model(current, history, stats, [str(p) for p in paths], changes)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(render_html(model), encoding="utf-8")
    tmp.replace(out)
    return model


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--registry", action="append", help="JSONL file or folder of *.jsonl (repeatable). "
                    "Default: $CRYOEV_REGISTRY, else <CryoAI-laptop>/registry")
    ap.add_argument("--out", help="default <registry>/history.html")
    a = ap.parse_args(argv)
    if not a.registry or not a.out:
        try:
            reg = config.registry_root()
        except config.ConfigError as e:
            raise SystemExit(f"{e} (or pass --registry and --out)")
    paths = a.registry or [str(reg)]
    a.out = a.out or str(reg / "history.html")
    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        raise SystemExit(f"not found: {', '.join(missing)} (try --registry registry/samples)")
    m = build(paths, a.out)
    print(f"wrote {a.out}: {len(m['recs'])} records, {m['stats']['lines']} lines")


if __name__ == "__main__":
    main()
