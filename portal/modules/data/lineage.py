"""
Lineage: the chain back to the raw imaging sessions (the Lineage walker and lineage_tree).

Part of the Data module (portal/modules/data/__init__.py has the rules); moved from data.py unchanged.
"""
from __future__ import annotations

from pathlib import Path

from portal.modules.data.records import (
    MAX_JSON_BYTES, SCHEMA, SET_KIND_NAMES, class_names, _dict, entity_href, find_set, format_number, grid_sort_key,
    image_count_of, is_contained, lineage_href, _list, _number, package_images_for, plural, ref_hidden, review_note,
    session_images, set_images, set_is_visible, short_date, split_counts, status_tags, status_word, _text, _when)
from portal.modules.data.details import (
    acquisition_fields, dataset_images, evenly_spaced, field, grid_images, hide_private, present, replaced_by,
    thumbnails)


# ---------------------------------------------------------------- lineage (the chain back to the raw sessions)
LINEAGE_MAX_DEPTH = 10      # steps from the item you asked about
LINEAGE_MAX_NODES = 80      # steps in one lineage, however the registry links them
LINEAGE_FANOUT = 8          # grids / sessions listed under one step before "and N more"
LINEAGE_THUMBS = 4
REVIEW_OPEN = ("built", "in_review", "reviewed")     # a review package not applied yet: "in progress"
KIND_TO_NODE = {"session": "session", "grid": "grid", "images": "images", "labels": "label", "dataset": "dataset",
                "model": "model"}
STEP_NAMES = {"model": "Model", "run": "Training run", "dataset": "Dataset", "label": "Label version",
              "export": "Roboflow export", "pull": "Roboflow download", "images": "Image set", "grid": "Grid",
              "session": "Imaging session", "source": "Source", "hidden": "Step", "schema": "Label schema"}
NODE_THUMB_KIND = {"session": "session", "grid": "grid", "images": "images", "label": "labels", "dataset": "dataset"}
HINT_TO_NODE = {"roboflow_export": "export", "export": "export", "label_version": "label", "lv": "label",
                "pull": "pull", "review_package": "images", "package": "images", "prediction_set": "images",
                "dataset": "dataset", "model": "model", "session": "session", "run": "run", "grid": "grid"}
NODE_RECORDS = {"model": "model", "run": "run", "dataset": "dataset", "label": "label_version",
                "export": "roboflow_export", "pull": "pull", "grid": "grid", "session": "session"}


def locate(records, ref_id, hint=None):
    """Which node type a registry id is: the hinted one if it is there, else the first record that has the id; None
    when no record has it (never guessed)."""
    def has(ntype):
        if ntype == "images":
            return find_set(records, ref_id)[1] is not None
        return ntype in NODE_RECORDS and ref_id in records[NODE_RECORDS[ntype]]
    first = HINT_TO_NODE.get(str(hint))
    for ntype in ([first] if first else []) + ["export", "label", "pull", "images", "dataset", "model", "run",
                                                        "grid", "session"]:
        if has(ntype):
            return ntype
    return None


def ref_spec(records, viewer, ref_id, hint=None):
    """A parent link for a recorded id: ('node', type, id) when something has it, ('hidden',) when it is a private
    set the viewer may not see, else ('unknown', type, why, id)."""
    if not _text(ref_id):
        return None
    ntype = locate(records, ref_id, hint)
    if ntype:
        return ("node", ntype, ref_id)
    if ref_hidden(records, viewer, ref_id, hint):
        return ("hidden",)
    return ("unknown", HINT_TO_NODE.get(str(hint), "source"),
            "It is recorded here but is not in the registry.", ref_id)


def unknown_spec(ntype, why, title=None):
    return ("unknown", ntype, why, None, title)


def input_specs(records, viewer, record, seen) -> list:
    """Parent links from the record's generic `inputs` list (child -> parent), skipping ids already linked."""
    specs = []
    for item in _list(record.get("inputs")):
        item = _dict(item)
        ref_id = _text(item.get("id"))
        if ref_id and ref_id not in seen and item.get("role") not in ("init_weights", "replaces"):
            seen.add(ref_id)
            specs.append(ref_spec(records, viewer, ref_id, item.get("entity")))
    return capped(specs, lambda spec: spec, "sources")


def image_sources(records, match) -> dict:
    """Imaging sessions of the registry's image records that `match` picks (those naming a session), with the grids
    where known: {"sessions": {session: {"n": images, "grids": {grid: images}}}, "all_sessions": set, "n": traced}.
    A grid is the image's own, else its session's session_grids slot."""
    sessions: dict = {}
    n = 0
    for image in records["image"].values():
        if not match(image) or not _text(image.get("session")):
            continue
        n += 1
        sid, slot = image["session"], image.get("session_grid")
        grid = _text(image.get("grid")) or _dict(_dict(records["session"].get(sid)).get("session_grids")).get(
            str(slot) if slot is not None else "")
        entry = sessions.setdefault(sid, {"n": 0, "grids": {}})
        entry["n"] += 1
        if grid:
            entry["grids"][str(grid)] = entry["grids"].get(str(grid), 0) + 1
    return {"sessions": sessions, "all_sessions": set(sessions), "n": n}


def pull_sources(records, pull_id) -> tuple:
    """Sessions of a download's images, from the registry only: the Roboflow image records first seen in this
    download, matched by image name to the image records that name a session. -> (image_sources result, number of
    those names with no session)."""
    names = {r.get("name") for r in records["roboflow_image"].values() if r.get("first_seen_pull") == pull_id}
    names.discard(None)
    found = image_sources(records, lambda image: image.get("rf_name") in names)
    traced = {im.get("rf_name") for im in records["image"].values()
              if im.get("rf_name") in names and _text(im.get("session"))}
    return found, len(names - traced)


def read_image_list(root, record):
    """The stems in a dataset's image list (first column of registry/<image_list.file>), or None when there is no
    readable list. The file must sit under the registry folder."""
    name = _text(_dict(record.get("image_list")).get("file"))
    if not name or root is None:
        return None
    path = Path(root) / name
    if path.suffix.lower() not in (".tsv", ".txt", ".csv") or not is_contained(path, root):
        return None
    try:
        if path.stat().st_size > MAX_JSON_BYTES:
            return None
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, ValueError):
        return None
    return [ln.split("\t")[0].strip() for ln in lines if ln.strip()]


def list_trace(records, root, record):
    """A dataset's / export's image list -> (listed stems, image_sources result, stems with no session in the
    registry), or None when there is no readable list."""
    stems = read_image_list(root, record)
    if stems is None:
        return None
    wanted = set(stems)
    found = image_sources(records, lambda image: image.get("id") in wanted)
    untraced = {s for s in wanted if not _text(_dict(records["image"].get(s)).get("session"))}
    return wanted, found, untraced


def traced_note(wanted, found) -> str:
    sessions = sorted(found["all_sessions"])
    shown = ", ".join(sessions[:6]) + (f" and {len(sessions) - 6} more" if len(sessions) > 6 else "")
    return (f"{found['n']} of {len(wanted)} images traced to sessions {shown}." if sessions else
            f"0 of {len(wanted)} images traced to a session.")


def grids_text(entry):
    """'Grid 2642 (16), Grid 2602 (9); 19 with no grid recorded' for one session's share of a list, or None."""
    grids = sorted(entry["grids"].items(), key=lambda kv: (-kv[1], grid_sort_key(kv[0])))
    if not grids:
        return None
    rest = entry["n"] - sum(n for _, n in grids)
    return (", ".join(f"Grid {g} ({n})" for g, n in grids[:LINEAGE_FANOUT])
            + (f" and {len(grids) - LINEAGE_FANOUT} more" if len(grids) > LINEAGE_FANOUT else "")
            + (f"; {rest} with no grid recorded" if rest else ""))


def list_compare_note(own, other, other_id, records) -> str:
    """A dataset's image list against the export it took its images from (counts only, never names)."""
    both, mine, theirs = own & other, own - other, other - own
    in_reg = sum(1 for s in mine if s in records["image"])
    text = f"{len(own)} images in this dataset: {len(both)} are in the image list of export {other_id}"
    if mine:
        text += (f", {len(mine)} are not" + (" (none of them is in the registry's image records)" if not in_reg else
                                             f" ({in_reg} of them in the registry's image records)"))
    if theirs:
        text += f"; {len(theirs)} of the export's listed images are not in this dataset"
    return text + ". Its imaging sessions are traced through that export."


def capped(specs: list, make, label: str) -> list:
    """The first LINEAGE_FANOUT specs, then a ('more', n, label) marker."""
    out = [make(s) for s in specs[:LINEAGE_FANOUT]]
    if len(specs) > LINEAGE_FANOUT:
        out.append(("more", len(specs) - LINEAGE_FANOUT, label))
    return out


class Lineage:
    """One walk over the registry for one viewer. Every object passes the viewer's visibility test before any of its
    facts are read."""

    def __init__(self, records, viewer, root=None):
        self.records, self.viewer, self.root, self.count = records, viewer, root, 0
        self.admin = _dict(viewer.user).get("role") == "admin"
        self.untraced = set()       # listed images (datasets / exports shown) with no session in the registry

    # ---- lookup and visibility
    def lookup(self, ntype, nid):
        """-> (ntype, record, entity) | ('hidden', None, None) | None when there is no such record."""
        if ntype == "images":
            entity, record = find_set(self.records, nid)
            if record is None:
                return None
            if not set_is_visible(self.viewer, entity, record):
                return ("hidden", None, None)
            return ("export", record, entity) if entity == "roboflow_export" else ("images", record, entity)
        record = self.records[NODE_RECORDS[ntype]].get(nid) if ntype in NODE_RECORDS else None
        if record and ntype == "export" and not set_is_visible(self.viewer, "roboflow_export", record):
            return ("hidden", None, None)
        return (ntype, record, None) if record else None

    # ---- the walk
    def node(self, ntype, nid, status="ok", title=None, **extra) -> dict:
        node = {"type": ntype, "step": STEP_NAMES[ntype], "status": status, "id": nid, "title": title or STEP_NAMES[ntype],
                "chips": [], "fields": [], "note": None, "url": None, "lineage": None, "thumbs": None,
                "parents": []}
        node.update(extra)
        self.count += 1                                     # every step shown counts toward the cap
        return node

    def walk(self, ntype, nid, path, depth, ctx=None) -> dict:
        node = self._walk(ntype, nid, path, depth)
        if ctx and node["type"] == "session" and node["status"] in ("ok", "cut"):
            self.with_count(node, ctx)
        return node

    @staticmethod
    def with_count(node, ctx):
        """A session reached from an image list: how many of the list's images it gave, and their grids."""
        node["images"] = ctx["n"]
        node["chips"] = [(plural(ctx["n"], "image"), "acc")] + list(node["chips"])
        node["fields"] = present(field("Images in this list", str(ctx["n"])),
                                 field("Grids", grids_text(ctx))) + list(node["fields"])

    def _walk(self, ntype, nid, path, depth) -> dict:
        found = self.lookup(ntype, nid)
        if found is None:
            if ref_hidden(self.records, self.viewer, nid):
                return self.hidden_node()
            return self.node(ntype, None, "unknown", note="Recorded as a source but not found in the registry.",
                             chips=[("Unknown", "warn")], recorded=nid)
        ntype, record, entity = found
        if ntype == "hidden":
            return self.hidden_node()
        key = (ntype, nid)
        if key in path:                                     # bad data: this step leads back to itself
            return self.node(ntype, nid, "loop", title=self.title(ntype, nid), chips=[("Loop in the registry", "warn")],
                             note="The registry links this step back to one above it, so the chain stops here.")
        if depth >= LINEAGE_MAX_DEPTH or self.count >= LINEAGE_MAX_NODES:
            return self.node(ntype, nid, "cut", title=self.title(ntype, nid),
                             note="Not shown: limit reached." + (" Use \"Lineage from here\" to continue."
                                                                  if self.lineage_url(ntype, nid, entity) else ""),
                             url=self.url(ntype, nid, entity), lineage=self.lineage_url(ntype, nid, entity))
        title, chips, fields, specs, note = self.describe(ntype, nid, record, entity)
        specs = [sp for sp in specs if sp]
        room = max(0, LINEAGE_MAX_NODES - self.count - 1)
        if len(specs) > room + 1:                           # keep the whole tree under the cap
            specs = specs[:room] + [("more", len(specs) - room, "steps")]
        node = self.node(ntype, nid, title=title, chips=chips, fields=fields, note=note,
                         url=self.url(ntype, nid, entity), lineage=self.lineage_url(ntype, nid, entity))
        if entity:
            node["set_kind"] = SET_KIND_NAMES.get(entity)
        node["thumbs"] = self.thumbs(ntype, nid, record, depth)
        child_path = path | {key}
        node["parents"] = [self.child(spec, child_path, depth + 1) for spec in specs]
        return node

    def hidden_node(self) -> dict:
        return self.node("hidden", None, "hidden", "Not visible to you", chips=[("Not visible to you", "warn")],
                         note="This step exists but belongs to someone else, so its details are not shown.")

    def child(self, spec, path, depth) -> dict:
        if spec[0] == "hidden":
            return self.hidden_node()
        if spec[0] == "node":
            return self.walk(spec[1], spec[2], path, depth, spec[3] if len(spec) > 3 else None)
        if spec[0] == "more":
            return self.node("source", None, "more", f"and {spec[1]} more {spec[2]}", note=None)
        if spec[0] == "schema":                             # round 14: an object set's label schema (a leaf)
            from portal.modules.data import labelmeta as LM
            from portal.modules.data.labelpages import schema_href
            sid = spec[1]
            sch = LM.load_schema(sid, getattr(self.viewer.cfg, "registry", None) if self.viewer.cfg else None)
            if not sch:
                return self.node("schema", None, "unknown", sid, note="This label schema is not registered.",
                                 chips=[("Unknown", "warn")], recorded=sid)
            v = sch.get("version")
            return self.node("schema", sid, title=str(sch.get("name") or sid), url=schema_href(sid),
                             chips=[(f"v{v}", "acc")] if isinstance(v, int) and not isinstance(v, bool) else [],
                             fields=present(field("Made by", _text(sch.get("maker"))),
                                            field("Attributes", ", ".join(str(_dict(a).get("label") or _dict(a).get("id"))
                                                                          for a in _list(sch.get("object_attrs"))
                                                                          if _dict(a).get("id")) or None)))
        if spec[0] == "untraced":                           # listed images no image record places in a session
            n = spec[1]
            return self.node("session", None, "unknown", "Session unknown", images=n,
                             chips=[(plural(n, "image"), "warn")],
                             note=f"{plural(n, 'listed image')} {'is' if n == 1 else 'are'} not matched to an imaging "
                                  "session in the registry's image records.")
        _, ntype, why, recorded, *rest = spec
        node = self.node(ntype, None, "unknown", rest[0] if rest else None, note=why, chips=[("Unknown", "warn")],
                         recorded=recorded)
        if len(rest) > 1 and rest[1]:
            self.with_count(node, rest[1])
        return node

    # ---- addresses
    @staticmethod
    def catalog_kind(ntype, entity):
        return {"label": "labels", "images": "images", "export": "images", "pull": None, "run": None}.get(ntype, ntype)

    def url(self, ntype, nid, entity=None):
        kind = self.catalog_kind(ntype, entity)
        return entity_href(kind, nid) if kind else None

    def lineage_url(self, ntype, nid, entity=None):
        kind = self.catalog_kind(ntype, entity)
        return lineage_href(kind, nid) if kind else None

    @staticmethod
    def title(ntype, nid):
        return f"Session {nid}" if ntype == "session" else f"Grid {nid}" if ntype == "grid" else str(nid)

    # ---- thumbnails (only where the catalog already has them)
    def thumbs(self, ntype, nid, record, depth):
        kind = NODE_THUMB_KIND.get(ntype)
        if kind is None or (ntype == "grid" and depth > 0):
            return None
        if ntype == "session":
            files = session_images(nid)
        elif ntype == "grid":
            files = grid_images(self.records, nid)
        elif ntype == "images":
            files = set_images(self.records, nid, self.viewer)
        elif ntype == "label":
            package = _text(record.get("pkg"))
            files = package_images_for(self.records, package, self.viewer) if package else []
        else:
            files = dataset_images(self.records, record, self.viewer)
        if not files:
            return None
        part = thumbnails(kind, nid, files, indexes=evenly_spaced(len(files), LINEAGE_THUMBS))
        return {"kind": kind, "id": nid, **part}

    # ---- each kind: title, chips, key facts, parent links, note
    def link(self, ntype, nid, entity=None):
        """Catalog address for a field value, only when that object exists and may be seen."""
        found = self.lookup(ntype, nid) if nid else None
        return self.url(ntype, nid, entity) if found and found[0] != "hidden" else None

    def describe(self, ntype, nid, record, entity):
        fn = getattr(self, "describe_" + ntype)
        return fn(nid, record, entity) if ntype == "images" else fn(nid, record)

    def describe_model(self, nid, rec):
        run, dataset = _text(rec.get("run")), _text(rec.get("dataset"))
        run_ok = run in self.records["run"]
        specs = []
        if run_ok and _text(self.records["run"][run].get("dataset")):
            specs.append(("node", "run", run))
        else:
            specs.append(("node", "run", run) if run_ok else (
                ("unknown", "run", "It is recorded here but is not in the registry.", run) if run else
                unknown_spec("run", "The registry does not say which training run made this model.")))
            specs.append(ref_spec(self.records, self.viewer, dataset, "dataset") or
                         unknown_spec("dataset", "The registry does not say which dataset this model was trained on."))
        specs += input_specs(self.records, self.viewer, rec, {nid, run, dataset})
        production = rec.get("status") == "production"
        fields = present(
            field("Status", status_word(rec["status"]) if _text(rec.get("status")) else None),
            field("Classes", class_names(rec)),
            field("Image size", f"{rec['imgsz']} px" if rec.get("imgsz") else None),
            field("Started from", _text(rec.get("init_from")), self.link("model", _text(rec.get("init_from")))),
            field("Registered", short_date(rec.get("logged"))),
            field("Weights file", _text(rec.get("weights")) if self.admin else None))
        return nid, [("Production", "prod")] if production else status_tags(rec), fields, specs, None

    def describe_run(self, nid, rec):
        config, metrics = _dict(rec.get("config")), _dict(rec.get("metrics"))
        dataset = _text(rec.get("dataset"))
        specs = [ref_spec(self.records, self.viewer, dataset, "dataset") or
                 unknown_spec("dataset", "The registry does not say which dataset this run trained on.")]
        fields = present(
            field("Status", status_word(rec["status"]) if _text(rec.get("status")) else None),
            field("Model type", _text(config.get("arch"))), field("Epochs", str(config["epochs"]) if
                                                                   _number(config.get("epochs")) is not None else None),
            field("Image size", f"{config['imgsz']} px" if _number(config.get("imgsz")) is not None else None),
            field("Machine", _text(rec.get("machine"))),
            field("Started", (rec.get("started") or "").replace("T", " ")[:16]),
            field("Mask mAP50", format_number(metrics.get("mask_map50"), 3)
                  if _number(metrics.get("mask_map50")) is not None else None),
            field("Job bundle", _text(rec.get("job_bundle"))))
        gap = _text(rec.get("job_bundle_gap"))
        return nid, status_tags(rec), fields, specs, f"No job bundle: {gap}." if gap and not rec.get("job_bundle") else None

    def describe_dataset(self, nid, rec):
        label, source = _text(rec.get("label_version")), _dict(rec.get("source"))
        seen = {nid}
        specs = []
        for ref_id, hint in ((label, "label_version"), (_text(source.get("id")), source.get("kind")),
                             (_text(rec.get("export")), "roboflow_export")):
            if ref_id and ref_id not in seen:
                seen.add(ref_id)
                specs.append(ref_spec(self.records, self.viewer, ref_id, hint))
        specs += input_specs(self.records, self.viewer, rec, seen)
        note = None
        trace = list_trace(self.records, self.root, rec)
        if trace and trace[0]:
            wanted, found, untraced = trace
            self.untraced |= untraced
            via = self.listed_export(rec)
            if via:                     # its sessions show under the export it took its images from (#15)
                note = list_compare_note(wanted, via[1], via[0], self.records)
            else:
                note = traced_note(wanted, found)
                specs += self.source_specs(found, len(untraced))
        if not specs:
            specs = [unknown_spec("source", "The registry does not say what this dataset was built from.")]
        fields = present(
            field("Status", status_word(rec["status"]) if _text(rec.get("status")) else None),
            field("Images per split", split_counts(rec)),
            field("Classes", class_names(rec)),
            field("Created", short_date(_when(rec))),
            field("Note", _text(rec.get("note"))))
        return nid, status_tags(rec), fields, specs, note

    def listed_export(self, rec):
        """(id, listed stems) of the first Roboflow export this dataset links to (export / source / inputs) that the
        viewer may see and that has a readable image list, else None."""
        source = _dict(rec.get("source"))
        ids = [_text(rec.get("export"))] + ([_text(source.get("id"))] if source.get("kind") == "roboflow_export"
                                            else [])
        ids += [_text(_dict(i).get("id")) for i in _list(rec.get("inputs"))
                if _dict(i).get("entity") == "roboflow_export"]
        for eid in ids:
            got = self.lookup("export", eid) if eid else None
            if got and got[0] == "export":
                stems = read_image_list(self.root, got[1])
                if stems:
                    return eid, set(stems)
        return None

    def source_specs(self, found, n_untraced=0) -> list:
        """Session parents for listed images, by image count (each with its count and grids), then one "Session
        unknown" step for the images no image record places in a session."""
        ordered = sorted(found["sessions"].items(), key=lambda kv: (-kv[1]["n"], kv[0]))
        specs = capped(ordered, lambda kv: ("node", "session", kv[0], kv[1]) if kv[0] in self.records["session"] else
                       ("unknown", "session", "The session is recorded on its images but is not in the registry.",
                        kv[0], None, kv[1]), "sessions")
        if n_untraced:
            specs.append(("untraced", n_untraced))
        return specs

    def describe_label(self, nid, rec):
        seen = {nid}
        specs = []
        base = _text(rec.get("base"))
        base_hint = next((_dict(i).get("entity") for i in _list(rec.get("inputs")) if _dict(i).get("id") == base), None)
        for ref_id, hint in ((base, base_hint), (_text(rec.get("pkg")), "review_package")):
            if ref_id and ref_id not in seen:
                seen.add(ref_id)
                specs.append(ref_spec(self.records, self.viewer, ref_id, hint))
        specs += input_specs(self.records, self.viewer, rec, seen)
        if not specs:
            specs = [unknown_spec("source", "The registry does not say what these labels were made from.")]
        note, tone = review_note(rec)
        replaces = _text(rec.get("supersedes"))
        fields = present(
            field("Status", status_word(rec["status"]) if _text(rec.get("status")) else None),
            field("Checking", note),
            field("Images changed", str(rec["n_images_applied"]) if rec.get("n_images_applied") is not None else None),
            field("Replaces", replaces, self.link("label", replaces)),
            *[field("Replaced by", newer, self.link("label", newer)) for newer in replaced_by(self.records, _text(rec.get("id")))],
            field("Created", short_date(_when(rec))))
        maker = "Labels: revised" if _text(rec.get("pkg")) or any(
            _dict(i).get("role") == "edits" for i in _list(rec.get("inputs"))) else "Labels: human"
        return nid, [(note, tone), (maker, "")] + status_tags(rec), fields, specs, None

    def describe_export(self, nid, rec, entity=None):
        parent, label = _text(rec.get("derived_from")), _text(rec.get("label_version"))
        seen = {nid}
        specs = []
        for ref_id, hint in ((parent, "roboflow_export"), (label, "label_version")):
            if ref_id and ref_id not in seen:
                seen.add(ref_id)
                specs.append(ref_spec(self.records, self.viewer, ref_id, hint))
        specs += input_specs(self.records, self.viewer, rec, seen)
        note, listed = None, None
        trace = list_trace(self.records, self.root, rec)
        if trace is not None:
            wanted, found, untraced = trace
            listed = len(wanted)
            self.untraced |= untraced
            note = traced_note(wanted, found)
            if str(_dict(rec.get("image_list")).get("method", "")).startswith("tracking"):
                note += " The image list comes from the image tracker, which lists only the images it tracks."
            # a re-split shows its sessions under the export it was made from only when that one is visible, lists
            # its images and they give exactly the same sessions, counts and grids; otherwise it shows its own
            if not (parent and self.same_sources_as_parent(parent, found, len(untraced))):
                specs += self.source_specs(found, len(untraced))
            if not specs:
                specs = [unknown_spec("session", "No image in this export is matched to an imaging session in the "
                                                 "registry.")]
        elif not parent:
            specs.append(unknown_spec("session", "The registry has no list of the images in this export, so the "
                                                 "imaging sessions they came from are not known.",
                                      "Image list not recorded"))
        if not specs:
            specs = [unknown_spec("source", "The registry does not record which images went into this export.")]
        fields = present(field("Classes", class_names(rec)),
                         field("Images per split", split_counts(rec)),
                         field("Image list", plural(listed, "image") if listed is not None else "not recorded"),
                         field("Made from export", hide_private(self.records, self.viewer, parent, "roboflow_export")
                               if parent else None, self.link("images", parent) if parent else None))
        return nid, [("Roboflow export", "acc"), ("Labels: human, via Roboflow", "")], fields, specs, note

    def same_sources_as_parent(self, parent, found, n_untraced) -> bool:
        """True when the parent export is visible to this viewer, has a readable image list, and its images give the
        same sessions with the same counts and grids, and as many untraced images, as this export's."""
        got = self.lookup("export", parent)
        if not got or got[0] != "export":
            return False
        trace = list_trace(self.records, self.root, got[1])
        return trace is not None and trace[1]["sessions"] == found["sessions"] and len(trace[2]) == n_untraced

    def describe_pull(self, nid, rec):
        found, n_untraced = pull_sources(self.records, nid)
        specs = self.source_specs(found, n_untraced) if found["n"] else []
        note = ("Sessions and grids come from the registry's image records, matched by image name to the images "
                "first seen in this download.")
        if not specs:
            specs = [unknown_spec("session", "No image in this download is matched to an imaging session in the "
                                             "registry.")]
            note = None
        fields = present(field("Images", str(rec["n_images"]) if rec.get("n_images") is not None else None),
                         field("Annotations", str(rec["n_annotations"]) if rec.get("n_annotations") is not None
                               else None),
                         field("Downloaded", short_date(_when(rec) or rec.get("created"))),
                         field("Image group", _text(rec.get("image_set"))))
        return nid, [], fields, specs, note

    def describe_images(self, nid, rec, entity):
        specs, note = [], None
        if entity == "prediction_set":
            source = _dict(rec.get("source"))
            session = _text(source.get("session"))
            if session:
                specs.append(ref_spec(self.records, self.viewer, session, "session"))
            elif source.get("kind") == "upload":
                note = "Uploaded by its owner. It does not come from a recorded imaging session."
            else:
                specs.append(unknown_spec("session", "The registry does not say which images this was run on."))
        else:
            seen = {nid}
            base = _text(rec.get("base"))
            for ref_id, hint in ((_text(rec.get("prediction_set")), "prediction_set"), (_text(rec.get("parent")),
                                 "review_package"), (base, rec.get("base_kind"))):
                if ref_id and ref_id not in seen:
                    seen.add(ref_id)
                    specs.append(ref_spec(self.records, self.viewer, ref_id, hint))
            specs += input_specs(self.records, self.viewer, rec, seen)
            if entity == "review_package" and rec.get("kind") == "objects":     # round 14: its label schema
                from portal.modules.data.labelpages import object_set_schema
                sid = object_set_schema(rec)
                if sid:
                    specs.append(("schema", sid))
            if not specs:
                specs = [unknown_spec("source", "The registry does not say what this package was made from.")]
        model, n = _text(rec.get("model")), image_count_of(rec)
        fields = present(
            field("Images", str(n) if n is not None else None),
            field("Status", status_word(rec["status"]) if _text(rec.get("status")) else None),
            field("Created", short_date(_when(rec))),
            field("Model", model, self.link("model", model) if model else None),
            field("Folder", _text(rec.get("path")) if self.admin and entity == "review_package" else None))
        if entity == "review_package" and rec.get("kind") == "objects":         # round 14: who answers it
            revs = [r for r in _list(rec.get("reviewers")) if isinstance(r, str)]
            n_items = rec.get("n_items")
            fields += present(field("Items", str(n_items) if isinstance(n_items, int) and not isinstance(n_items, bool)
                                    else None),
                              field("Reviewers", ", ".join(revs) or None))
        chips = [(SET_KIND_NAMES[entity], "acc")]
        if entity == "review_package" and rec.get("kind") == "objects":
            chips.append(("Object set", ""))
        if entity == "prediction_set":
            chips.append(("Labels: automatic", ""))
        elif entity == "review_package":            # #16: a review sits between label sets until it is applied
            applied = sorted(lid for lid, lv in self.records["label_version"].items() if lv.get("pkg") == nid)
            if applied:
                fields.append(("Applied as", applied[0], self.link("label", applied[0])))
                chips.append(("Applied", ""))
            elif rec.get("status") in REVIEW_OPEN:
                chips.append(("In progress", "warn"))
        return nid, chips, fields, specs, note

    def describe_grid(self, nid, rec):
        sessions = sorted({sid for sid, s in self.records["session"].items()
                           if nid in _dict(s.get("session_grids")).values()}
                          | {im["session"] for im in self.records["image"].values()
                             if im.get("grid") == nid and im.get("session")})
        specs = capped(sessions, lambda sid: ("node", "session", sid) if sid in self.records["session"] else
                       ("unknown", "session", "The session is recorded here but is not in the registry.", sid),
                       "sessions")
        if not specs:
            specs = [unknown_spec("session", "No imaging session is recorded for this grid.")]
        fields = present(field("Sample", _text(rec.get("sample_label"))), field("Grid type", _text(rec.get("grid_type"))),
                         field("Prepared", short_date(rec.get("prep_date"))),
                         field("Replicate", str(rec["replicate"]) if rec.get("replicate") is not None else None))
        return f"Grid {nid}", [("Replaced", "warn")] if _text(rec.get("replaced_by")) else [], fields, specs, None

    def describe_session(self, nid, rec):
        files = session_images(nid)
        keep = ("Date", "Instrument", "Magnification", "Pixel size", "Defocus", "Dose", "Voltage")
        fields = [f for f in acquisition_fields(rec, files) if f[0] in keep]
        if self.admin:
            fields += present(field("Folder", _text(rec.get("local_folder"))))
        gap = _text(rec.get("acquisition_gap"))
        note = f"Acquisition details are not available: {gap}" if gap else None
        return f"Session {nid}", status_tags(rec), fields, [], note


def lineage_tree(records, viewer, kind, entity_id, root=None):
    """-> the lineage dict for a catalog item, or None when it is unknown or hidden from this viewer."""
    ntype = KIND_TO_NODE.get(kind)
    if ntype is None or not _text(entity_id):
        return None
    walker = Lineage(records, viewer, root)
    found = walker.lookup(ntype, entity_id)
    if found is None or found[0] == "hidden":
        return None
    root = walker.walk(ntype, entity_id, frozenset(), 0)
    counts = {"steps": 0, "known": 0, "unknown": 0, "hidden": 0, "cut": 0, "loop": 0}

    def tally(node):
        if node["status"] == "more":
            return
        counts["steps"] += 1
        counts["known" if node["status"] == "ok" else node["status"] if node["status"] in counts else "unknown"] += 1
        for parent in node["parents"]:
            tally(parent)
    tally(root)
    counts["untraced_images"] = len(walker.untraced)        # a number only: never the ids
    return {"schema": SCHEMA, "kind": kind, "id": entity_id, "title": root["title"], "summary": counts,
            "limits": {"depth": LINEAGE_MAX_DEPTH, "steps": LINEAGE_MAX_NODES}, "root": root}
