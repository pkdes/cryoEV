"""
The detail dicts (one source for each page and its JSON) and the thumbnails.

Part of the Data module (portal/modules/data/__init__.py has the rules); moved from data.py unchanged.
"""
from __future__ import annotations

import hashlib
import io
import os
import threading
from pathlib import Path
from urllib.parse import quote

from portal import titles as TT
from portal import upload_sessions as US
from portal.modules.data.records import (
    PAGE_N, PREDICTION_PACKAGE_ID, PREFIX, SCHEMA, SET_KIND_NAMES, STRIP_N, THUMB_MAX_PIXELS, THUMB_PX, cached,
    class_mapping, class_names, _decoders, _dict, entity_href, _files_cache, format_number, image_count_of,
    label_title, _list, magnification_text, metric_name, _number, package_images_for, plural, ref_hidden,
    review_folder, review_note, session_files, session_images, session_tags, set_images, set_is_visible, set_title,
    short_date, slot_sort_key, split_counts, status_tags, status_word, _text, thumb_version, upload_images,
    visible_set, _when, who_label, _actor_module)


# ---------------------------------------------------------------- detail dicts (one source for page and API)
def make_detail(kind, entity_id, title, chips=(), fields=(), **extra) -> dict:
    detail = {"schema": SCHEMA, "kind": kind, "id": entity_id, "title": title, "chips": [list(c) for c in chips],
              "fields": [list(f) for f in fields], "n_images": 0, "thumbs": [], "page": 1, "pages": 1,
              "view": "paged"}
    detail.update(extra)
    return detail


def field(label, value, href=None):
    return (label, value, href) if value not in (None, "", []) else None


def present(*fields):
    return [f for f in fields if f]


def thumbnails(kind, entity_id, files, page=None, indexes=None) -> dict:
    """The thumbnail part of a detail: a page of 24, or the given file indexes (a strip)."""
    n = len(files)
    pages = max(1, -(-n // PAGE_N))
    if indexes is None:
        page = min(max(page or 1, 1), pages)
        indexes = range((page - 1) * PAGE_N, min(n, page * PAGE_N))
    else:
        page = 1
    base = f"{PREFIX}/thumb/{kind}/{quote(str(entity_id), safe='')}"
    thumbs = []
    for i in indexes:
        version = thumb_version(files[i])
        thumbs.append({"n": i, "url": f"{base}/{i}.jpg" + (f"?v={version}" if version else "")})
    return {"n_images": n, "page": page, "pages": pages, "thumbs": thumbs}


def evenly_spaced(n, k):
    if n <= k:
        return list(range(n))
    return sorted({round(i * (n - 1) / (k - 1)) for i in range(k)})


def acquisition_fields(record, files) -> list:
    defocus, dose = _dict(record.get("defocus_um")), _dict(record.get("dose"))
    pixel_sizes = [x for x in _list(record.get("pixel_size_a")) if _number(x) is not None]
    other_mags = [x for x in _list(record.get("other_magnification")) if _number(x) is not None]
    dates = _dict(record.get("acq_dates"))
    n_other = record.get("n_other")
    other_text = None
    if _number(n_other):
        other_text = f"{plural(n_other, 'file')} below 5,000×" + (
            " (" + ", ".join(f"{x:,}×" for x in other_mags) + ")" if other_mags else "")
    elif other_mags:
        other_text = ", ".join(f"{x:,}×" for x in other_mags)
    voltages = [f"{format_number(x, 0)} kV" for x in _list(record.get("voltage_kv")) if _number(x) is not None]
    imaged = None
    if dates.get("first") and dates.get("last"):
        imaged = f"{dates['first'].replace('T', ' ')[:16]} to {dates['last'].replace('T', ' ')[:16]}"
    n_imaging = record.get("n_imaging")
    return present(
        field("Date", short_date(record.get("date"))),
        field("Instrument", _text(record.get("instrument"))),
        field("Magnification", magnification_text(record)),
        field("Pixel size", " and ".join(f"{format_number(x, 3)} Å/px" for x in pixel_sizes)),
        field("Defocus", f"median {format_number(defocus.get('median'))} µm "
                         f"(range {format_number(defocus.get('min'))} to {format_number(defocus.get('max'))})"
              if defocus else None),
        field("Dose", f"median {format_number(dose.get('median'))} e/Å² "
                      f"(range {format_number(dose.get('min'))} to {format_number(dose.get('max'))})"
              if dose else None),
        field("Voltage", " and ".join(voltages)),
        field("Imaged", imaged),
        field("Imaging files", str(n_imaging) if _number(n_imaging) is not None else None),
        field("Other files", other_text),
        field("Processed images", str(len(files)) if files else None))


def session_grid_cards(records, record, files) -> list:
    sid = record["id"]
    slots = sorted(set(_dict(record.get("sample_labels"))) | set(_dict(record.get("session_grids")))
                   | set(US.session_slots(record)), key=slot_sort_key)
    labels = US.effective_labels(record)
    typed = _dict(_dict(record.get("details")).get("grid_ids"))
    registry_counts: dict = {}
    for image in records["image"].values():
        if image.get("session") == sid and image.get("session_grid") is not None:
            slot = str(image["session_grid"])
            registry_counts[slot] = registry_counts.get(slot, 0) + 1
    uploaded: dict = {}
    for c in US.conversions_of(record):
        if c.get("derived_from"):                               # a re-conversion of another: the same tiles, not new ones
            continue
        for slot, n in _dict(c.get("tiles_per_grid")).items():
            if _number(n) is not None:
                uploaded[str(slot)] = uploaded.get(str(slot), 0) + int(n)
    cards = []
    for slot in slots:
        grid_id = _dict(record.get("session_grids")).get(slot)
        grid = records["grid"].get(grid_id) if grid_id else None
        processed = len(session_images(sid, f"grid{slot}_".lower())) if files else 0
        n, source = ((processed, "processed files") if processed else
                     (registry_counts[slot], "registry records") if registry_counts.get(slot) else
                     (uploaded[slot], "uploaded tiles") if uploaded.get(slot) else (0, "registry records"))
        cards.append({"slot": slot, "label": _text(labels.get(slot)) or (grid or {}).get("sample_label"),
                      "grid": grid_id if grid else None, "grid_id_recorded": grid_id,
                      "grid_id_typed": _text(typed.get(slot)),
                      "grid_type": (grid or {}).get("grid_type"),
                      "n_images": n, "count_source": source})
    return cards


def session_analyses(records, sid, viewer) -> list:
    """Analyses (prediction sets) whose source session is `sid` that the viewer may see: the registry's records,
    plus the Predict folder's request.json files (cached FILES_TTL_S per viewer). [{id, title, url, status}]"""
    found = {}
    for record in records["prediction_set"].values():
        if _dict(record.get("source")).get("session") == sid and set_is_visible(viewer, "prediction_set", record):
            found[record["id"]] = {"id": record["id"], "title": _text(record.get("name")) or record["id"],
                                   "url": f"/predict/{quote(record['id'], safe='')}/",
                                   "status": status_word(record["status"]) if _text(record.get("status")) else None}
    cfg = getattr(viewer, "cfg", None)
    if cfg is not None:
        user = viewer.user or {}

        def scan():
            try:
                from portal.modules import predict as P
                return [(d.name, req, st) for d, req, st in P.list_sets(cfg, user, viewer.can_see)]
            except Exception:                            # noqa: BLE001 - no predict root: the registry's only
                return []
        for ps, req, st in cached(_files_cache, ("analyses", str(user.get("id")), str(user.get("role"))), scan):
            src = _dict(req.get("source"))
            if src.get("session") == sid and ps not in found:
                found[ps] = {"id": ps, "title": _text(req.get("name")) or ps, "url": f"/predict/{quote(ps, safe='')}/",
                             "status": status_word(st["state"]) if _text(_dict(st).get("state")) else None}
    return sorted(found.values(), key=lambda a: a["id"], reverse=True)


def detail_session(records, record, page=None, viewer=None) -> dict:
    sid = record["id"]
    lab_files = session_images(sid)
    files = lab_files or upload_images(record, viewer)
    acq, from_upload = record, False
    if not US.has_acquisition(record) and US.upload_acquisition(record):
        acq, from_upload = {**record, **US.upload_acquisition(record)}, True
    elif US.is_upload_session(record) and not record.get("local_folder") and US.has_acquisition(record):
        from_upload = True                               # a session the portal made: its summary is the upload's
    fields = acquisition_fields(acq, files)
    if from_upload:
        fields.append(("Acquisition read from", "the uploaded raw files", None))
    upload = None
    if US.is_upload_session(record):
        det = _dict(record.get("details"))
        convs = (US.visible_conversions(viewer.cfg, record, viewer.user or {}, viewer.can_see)
                 if viewer is not None and viewer.cfg is not None else [])
        upload = {"conversions": [{"id": c["id"], "url": c["href"], "gallery": c["gallery"], "n_tiles": c["n_tiles"],
                                   "derived_from": c.get("derived_from")}
                                  for c in convs],
                  "details": {"grid_ids": _dict(det.get("grid_ids")), "sample_display": _dict(det.get("sample_display")),
                              "prep_notes": _text(det.get("prep_notes")),
                              "by": who_label(det.get("by"), det.get("via")) or None,
                              "by_ai": bool(_text(det.get("via")) and _text(det.get("by"))),
                              "at": _text(det.get("at"))},
                  "can_edit": bool(viewer is not None and US.enabled(viewer.cfg) and US.may_edit(viewer.user, record))}
    detail = make_detail(
        "session", sid, f"Session {sid}", session_tags(record), fields,
        grids=session_grid_cards(records, record, files),
        acquisition={k: acq.get(k) for k in (
            "instrument", "magnification", "pixel_size_a", "defocus_um", "dose", "voltage_kv", "binning",
            "acq_dates", "n_mdoc", "n_imaging", "n_other", "other_magnification")},
        gap=None if from_upload else _text(record.get("acquisition_gap")), upload=upload,
        images_word="processed" if lab_files or not files else "from the upload",
        analyses=session_analyses(records, sid, viewer) if viewer is not None else [])
    if page is None:
        detail.update(thumbnails("session", sid, files, indexes=evenly_spaced(len(files), STRIP_N)))
        detail["view"] = "strip"
    else:
        detail.update(thumbnails("session", sid, files, page))
    return detail


def grid_images(records, grid_id) -> list:
    pairs = set()
    for sid, session in records["session"].items():
        for slot, gid in _dict(session.get("session_grids")).items():
            if gid == grid_id:
                pairs.add((sid, str(slot)))
    for image in records["image"].values():
        if image.get("grid") == grid_id and image.get("session") and image.get("session_grid") is not None:
            pairs.add((str(image["session"]), str(image["session_grid"])))
    files = []
    for sid, slot in sorted(pairs):
        files += session_images(sid, f"grid{slot}_".lower())
    return files


def detail_grid(records, record, page) -> dict:
    gid = record["id"]
    sessions = sorted({sid for sid, s in records["session"].items() if gid in _dict(s.get("session_grids")).values()}
                      | {im["session"] for im in records["image"].values()
                         if im.get("grid") == gid and im.get("session")})
    replaced_by = record.get("replaced_by")
    fields = present(
        field("Sample", _text(record.get("sample_label"))), field("Grid type", _text(record.get("grid_type"))),
        field("Project", _text(record.get("project"))), field("Prepared", short_date(record.get("prep_date"))),
        field("Replicate", str(record["replicate"]) if record.get("replicate") is not None else None),
        field("Glow discharge", _text(record.get("glow_discharge"))), field("Storage", _text(record.get("storage"))),
        field("Montages", str(record["n_montages"]) if record.get("n_montages") is not None else None),
        field("Notes", _text(record.get("notes"))),
        field("Flags", _text(record.get("flags")) if isinstance(record.get("flags"), str) else None),
        field("Replaced by", f"Grid {replaced_by}" if _text(replaced_by) else None,
              entity_href("grid", replaced_by) if _text(replaced_by) and replaced_by in records["grid"] else None))
    detail = make_detail("grid", gid, f"Grid {gid}", [("Replaced", "warn")] if _text(replaced_by) else [], fields,
                         sessions=[{"id": s, "url": entity_href("session", s)} for s in sessions])
    detail.update(thumbnails("grid", gid, grid_images(records, gid), page))
    return detail


def metrics_of(record) -> dict:
    metrics = record.get("metrics") or record.get("scores")
    return metrics if isinstance(metrics, dict) else {}


def metric_rows(metrics: dict) -> list:
    """-> [[plain name, value text]]; one level of nesting becomes 'Name (sub)'."""
    rows = []
    for key, value in metrics.items():
        if isinstance(value, dict):
            rows += [[metric_name(key, sub), format_number(v, 3) if _number(v) is not None else str(v)]
                     for sub, v in value.items()]
        else:
            rows.append([metric_name(key), format_number(value, 3) if _number(value) is not None else str(value)])
    return rows


def detail_model(records, record) -> dict:
    model_id = record["id"]
    names = {model_id, *[a for a in _list(record.get("aliases")) if isinstance(a, str)]}
    checks = [{"id": c["id"], "session": c.get("session"), "n": c.get("n"),
               "status": status_word(c["status"]) if c.get("status") else None, "metrics": metric_rows(metrics_of(c))}
              for c in sorted(records["inference_check"].values(), key=lambda c: c["id"]) if c.get("model") in names]
    production = record.get("status") == "production"
    dataset, init = _text(record.get("dataset")), _text(record.get("init_from"))
    fields = present(
        field("Status", status_word(record["status"]) if _text(record.get("status")) else None),
        field("Trained on", dataset, entity_href("dataset", dataset) if dataset in records["dataset"] else None),
        field("Started from", init, entity_href("model", init) if init in records["model"] else None),
        field("Classes", class_names(record)),
        field("Image size", f"{record['imgsz']} px" if record.get("imgsz") else None),
        field("Detection threshold", format_number(record.get("conf")) if _number(record.get("conf")) is not None
              else None),
        field("Registered", short_date(record.get("logged"))),
        field("Weights fingerprint", (record.get("weights_sha256") or "")[:12]),
        field("Note", _text(record.get("note"))))
    chips = [("Production", "prod")] if production else status_tags(record)
    return make_detail("model", model_id, model_id, chips, fields, metrics=metric_rows(metrics_of(record)),
                       checks=checks, production=production)


def hide_private(records, viewer, ref_id, hint=None):
    """An id to quote in a field, or "Not visible to you" when it names a private set this viewer may not see."""
    return "Not visible to you" if ref_hidden(records, viewer, ref_id, hint) else ref_id


# provenance.derived_from kind -> (records table, href kind, word); "other" has no table and is never linked
DERIVED_KIND = {"dataset": ("dataset", "dataset", "dataset"),
                "label_version": ("label_version", "labels", "label version"),
                "session": ("session", "session", "session"),
                "prediction_set": ("prediction_set", "images", "prediction set"), "model": ("model", "model", "model")}


def tiling_text(tiling) -> str | None:
    """A prediction set's provenance.tiling as one line: 'tiles 1440 px, overlap 20%, 0.368 nm/px, no downsampling'."""
    t = _dict(tiling)
    parts = []
    if _number(t.get("tile_px")) is not None:
        parts.append(f"tiles {t['tile_px']:g} px")
    px, frac = _number(t.get("overlap_px")), _number(t.get("overlap_frac"))
    if px is not None or frac is not None:
        pct = f"{format_number(frac * 100, 1)}%" if frac is not None else None
        parts.append("overlap " + (f"{px:g} px ({pct})" if px is not None and pct
                                   else f"{px:g} px" if px is not None else pct))
    if _number(t.get("nm_per_px")) is not None:
        parts.append(f"{t['nm_per_px']:.6g} nm/px")
    if _number(t.get("downsample")) is not None:
        parts.append("no downsampling" if t["downsample"] == 1 else f"downsampled {format_number(t['downsample'], 3)}x")
    return ", ".join(parts) or None


def made_how(provenance) -> list:
    """'How the images were made': provenance.steps, then the tiling line."""
    p = _dict(provenance)
    lines = [s for s in _list(p.get("steps")) if _text(s)]
    tiling = tiling_text(p.get("tiling"))
    return lines + ([tiling] if tiling else [])


def derived_from(records, viewer, provenance) -> list:
    """'Derived from': [text, href or None] per provenance.derived_from entry; linked only when that record exists
    (and, for a private set, only when this viewer may see it; otherwise its id is never quoted)."""
    out = []
    for entry in _list(_dict(provenance).get("derived_from")):
        kind, rid = _dict(entry).get("kind"), _text(_dict(entry).get("id"))
        if not rid:
            continue
        table, href_kind, word = DERIVED_KIND.get(kind, (None, None, _text(kind) or "other"))
        shown = hide_private(records, viewer, rid)
        if shown != rid:
            out.append([shown, None])
            continue
        known = table is not None and rid in records.get(table, {})
        out.append([f"{rid} ({word})", entity_href(href_kind, rid) if known else None])
    return out


def detail_images(records, entity, record, viewer, page) -> dict:
    set_id = record["id"]
    model, source = _text(record.get("model")), _dict(record.get("source"))
    n = image_count_of(record)
    package_of_set = _text(record.get("review_package"))
    package_shown = package_of_set if package_of_set and (
        package_of_set not in records["review_package"]
        or set_is_visible(viewer, "review_package", records["review_package"][package_of_set])) else None
    fields = present(
        field("Kind", SET_KIND_NAMES[entity]), field("Images", str(n) if n is not None else None),
        field("Status", status_word(record["status"]) if _text(record.get("status")) else None),
        field("Created", short_date(_when(record))),
        field("Made by", (who_label(record.get("owner"), _actor_module().via_of(record, "owner")) or None)
              if entity == "prediction_set" else None),   # set_is_visible already requires viewer.may_see(owner)
        *((field("Note", _text(record.get("note"))),
           field("How the images were made", made_how(record.get("provenance"))),
           field("Derived from", derived_from(records, viewer, record.get("provenance"))))
          if entity == "prediction_set" else ()),         # round 11b: absent fields show nothing
        field("Model", model, entity_href("model", model) if model in records["model"] else None),
        field("Detection threshold", format_number(record.get("conf")) if _number(record.get("conf")) is not None
              else None),
        field("Source", (f"session {source['session']}" if source.get("session") else _text(source.get("kind")))
              if source else None,
              entity_href("session", source["session"]) if source.get("session") in records["session"] else None),
        field("Reviewers", ", ".join(map(str, _list(record.get("reviewers"))))),
        field("Edits recorded", str(record["n_edit_records"]) if record.get("n_edit_records") is not None else None),
        field("Starts from", hide_private(records, viewer, _text(record.get("base"))
                                          or _text(record.get("derived_from")), record.get("base_kind"))),
        field("Splits", split_counts(record)), field("Classes", class_names(record)),
        field("Review package", package_shown,
              entity_href("images", package_shown) if package_shown in records["review_package"] else None))
    open_review = None
    if entity == "review_package" and PREDICTION_PACKAGE_ID.match(set_id):
        open_review = f"/review/{quote(set_id, safe='')}/"
    elif package_shown and PREDICTION_PACKAGE_ID.match(package_shown):
        open_review = f"/review/{quote(package_shown, safe='')}/"
    detail = make_detail("images", set_id, set_title(records, entity, record), [(SET_KIND_NAMES[entity], "acc")], fields,
                         set_kind=entity, review_url=open_review)
    if entity == "roboflow_export":
        detail["note"] = "Thumbnails are not shown for Roboflow exports."
    detail.update(thumbnails("images", set_id, set_images(records, set_id, viewer), page))
    return detail


def waiting_on(records, label_id, viewer) -> list:
    """Review packages whose base is this label version and that hold decisions not applied yet (applied = some label
    version record has that package as its pkg): [{id, title, n, url or None}]. Read-only; the package folders are
    scanned through portal.titles (cached by the edits folder's mtime)."""
    root = review_folder()
    out = []
    for p in TT.scan_packages(root, records) if root is not None else []:
        if p["base_lv"] != label_id or not p["waiting"]:
            continue
        if not set_is_visible(viewer, "review_package", {**p["meta"], "id": p["id"]}):
            continue
        known = p["id"] in records["review_package"]
        admin = _dict(getattr(viewer, "user", None)).get("role") == "admin"
        preview = (f"/apply/#p-{quote(p['id'], safe='')}" if admin and p.get("kind") != "compare"
                   and not str(p["id"]).startswith("pred_") else None)     # V4 (round 10): admin-only apply preview
        out.append({"id": p["id"], "title": p["title"], "n": p["waiting"],
                    "url": entity_href("images", p["id"]) if known else None, "preview": preview})
    return sorted(out, key=lambda x: x["id"])


def replaced_by(records, label_id):
    """The newer label version(s) whose `supersedes` is this one (T4: the registry stores only the backward link)."""
    return sorted(i for i, r in records.get("label_version", {}).items() if _text(r.get("supersedes")) == label_id)


def detail_labels(records, record, viewer, page) -> dict:
    label_id = record["id"]
    package = _text(record.get("pkg"))
    package_record = records["review_package"].get(package) or {"id": package}
    package_allowed = bool(package) and set_is_visible(viewer, "review_package", package_record)
    note, tone = review_note(record)
    built = sorted(d["id"] for d in records["dataset"].values() if d.get("label_version") == label_id)
    trained = sorted({m["id"] for m in records["model"].values() if m.get("dataset") in built})
    waiting = waiting_on(records, label_id, viewer)
    ops = {k: v for k, v in _dict(record.get("counts_by_op")).items() if _number(v)}
    replaces = _text(record.get("supersedes"))
    shown_package = package if package_allowed else None
    fields = present(
        field("Status", status_word(record["status"]) if _text(record.get("status")) else None),
        field("Checking", note),
        field("Images changed", str(record["n_images_applied"]) if record.get("n_images_applied") is not None
              else None),
        field("Changes", ", ".join(f"{k} {v}" for k, v in ops.items())),
        field("Starts from", hide_private(records, viewer, _text(record.get("base")))),
        field("Review package", shown_package,
              entity_href("images", shown_package) if shown_package in records["review_package"] else None),
        field("Replaces", replaces, entity_href("labels", replaces) if replaces in records["label_version"] else None),
        *[field("Replaced by", newer, entity_href("labels", newer)) for newer in replaced_by(records, label_id)],
        field("Created", short_date(_when(record))), field("Note", _text(record.get("note"))))
    detail = make_detail("labels", label_id, label_title(records, label_id, viewer), [(note, tone)] + status_tags(record),
                         fields, datasets=[{"id": i, "url": entity_href("dataset", i)} for i in built],
                         models=[{"id": i, "url": entity_href("model", i)} for i in trained], waiting=waiting)
    files = package_images_for(records, package, viewer) if package else []
    detail.update(thumbnails("labels", label_id, files, page))
    return detail


def dataset_images(records, record, viewer) -> list:
    label = records["label_version"].get(record.get("label_version") or "")
    package = _text((label or {}).get("pkg"))
    return package_images_for(records, package, viewer) if package else []


def detail_dataset(records, record, viewer, page) -> dict:
    dataset_id = record["id"]
    files = dataset_images(records, record, viewer)
    source, label = _dict(record.get("source")), _text(record.get("label_version"))
    models = sorted(m["id"] for m in records["model"].values() if m.get("dataset") == dataset_id)
    split = _dict(record.get("split"))
    source_id = _text(source.get("id"))
    source_is_set = bool(source_id) and visible_set(records, source_id, viewer)[1] is not None
    source_hidden = bool(source_id) and ref_hidden(records, viewer, source_id, source.get("kind"))
    fields = present(
        field("Status", status_word(record["status"]) if _text(record.get("status")) else None),
        field("Created", short_date(_when(record))), field("Splits (images)", split_counts(record)),
        field("Classes", class_names(record)),
        field("Label version", label, entity_href("labels", label) if label in records["label_version"] else None),
        field("Source", ("Not visible to you" if source_hidden else source_id or _text(source.get("kind")))
              if source else None,
              entity_href("images", source_id) if source_is_set else None),
        field("Split rule", f"{split['method']}, seed {split.get('seed')}" if split.get("method") else None),
        field("Note", _text(record.get("note"))))
    detail = make_detail("dataset", dataset_id, dataset_id, status_tags(record), fields,
                         models=[{"id": m, "url": entity_href("model", m)} for m in models],
                         class_map=class_mapping(record))
    if not files:
        detail["note"] = "No example images are linked to this dataset."
    detail.update(thumbnails("dataset", dataset_id, files, page))
    return detail


def detail(records, kind, entity_id, viewer, page=None):
    """-> detail dict, or None when unknown or hidden from this viewer."""
    if kind == "session":
        record = records["session"].get(entity_id)
        return detail_session(records, record, page, viewer) if record else None
    if kind == "grid":
        record = records["grid"].get(entity_id)
        return detail_grid(records, record, page) if record else None
    if kind == "model":
        record = records["model"].get(entity_id)
        return detail_model(records, record) if record else None
    if kind == "dataset":
        record = records["dataset"].get(entity_id)
        return detail_dataset(records, record, viewer, page) if record else None
    if kind == "labels":
        record = records["label_version"].get(entity_id)
        return detail_labels(records, record, viewer, page) if record else None
    if kind == "images":
        entity, record = visible_set(records, entity_id, viewer)
        return detail_images(records, entity, record, viewer, page) if record else None
    return None


def thumbnail_sources(records, kind, entity_id, viewer) -> list:
    """The ordered image files behind /data/thumb/<kind>/<id>/<n>.jpg, same order as the detail pages use."""
    if kind == "session":
        record = records["session"].get(entity_id)
        return session_files(record, viewer) if record else []
    if kind == "grid":
        return grid_images(records, entity_id) if entity_id in records["grid"] else []
    if kind == "images":
        return set_images(records, entity_id, viewer)
    if kind == "labels":
        package = _text(_dict(records["label_version"].get(entity_id)).get("pkg"))
        return package_images_for(records, package, viewer) if package else []
    if kind == "dataset":
        record = records["dataset"].get(entity_id)
        return dataset_images(records, record, viewer) if record else []
    return []


# ---------------------------------------------------------------- thumbnails
def to_8bit(image):
    """A 16-bit / float image -> 8-bit grey, stretched from its own min to max so a dim image stays visible."""
    image = image.convert("I") if image.mode != "F" else image
    low, high = image.getextrema()
    scale = 255.0 / (high - low) if high > low else 1.0
    return image.point(lambda v: v * scale - low * scale).convert("L")


def make_thumb(path: Path, cache_dir: Path):
    """-> JPEG bytes (cached on disk by path + mtime + size), or None when the file can't be read or is too large."""
    from PIL import Image
    try:
        st = path.stat()
        stamp = f"{path}|{st.st_mtime_ns}|{st.st_size}|{THUMB_PX}"
        key = hashlib.sha256(stamp.encode("utf-8", "replace")).hexdigest()[:40]
        cache_file = cache_dir / key[:2] / f"{key}.jpg"
        try:
            return cache_file.read_bytes()
        except OSError:
            pass
        Image.MAX_IMAGE_PIXELS = THUMB_MAX_PIXELS
        with _decoders:                                  # at most THUMB_DECODERS decodes at once
            with Image.open(path) as image:
                if image.size[0] * image.size[1] > THUMB_MAX_PIXELS:
                    return None
                image.draft("RGB", (THUMB_PX * 2, THUMB_PX * 2))
                if image.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
                    image = to_8bit(image)
                elif image.mode not in ("L", "RGB"):
                    image = image.convert("RGB")
                image.thumbnail((THUMB_PX, THUMB_PX))
                buffer = io.BytesIO()
                image.save(buffer, "JPEG", quality=80, optimize=True)
        data = buffer.getvalue()
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache_file.with_name(cache_file.name + f".{os.getpid()}.{threading.get_ident()}.tmp")
            tmp.write_bytes(data)
            os.replace(tmp, cache_file)
        except OSError:
            pass                                         # a cache that can't be written still serves the image
        return data
    except Exception:                                    # unreadable / too large / not an image
        return None
