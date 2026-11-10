"""
Predict, the pages of sets (every text through esc): My analyses, New analysis, the session form (plain and kept
tiles), the upload form, a set's page and the preparing page.

Split out of portal/modules/predict.py (round 7, P7; no behaviour change); predict.py re-exports every name.
Names the tests patch are used from other parts only as <PART>.<name> (portal/predict_parts/__init__.py).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from jobs import predict_set as PS
from portal import predict_sources as PSRC
from portal import names as NM

from portal.predict_parts.common import (NM_FROM_MANIFEST, NM_HINT, NM_TOLERANCE, NO_MODEL, RAW_ID_RE, RAW_ROLES,
                                         Refusal, SCHEMA_SET, _read_json, chip, esc, msg_box, now_iso, problem_list,
                                         tech, when, wrap)
from portal.predict_parts.sets import (allowed_models, can_restart, edits_mtime, images_rows, list_sets,
                                       nm_range_text, results_file, samples_of, spot_list, status_view, task)
from portal.predict_parts.estimate import estimate_html
from portal.predict_parts.raw import processing_repo
from portal.predict_parts.runner import (conv_status_view, list_conversions)
from portal.predict_parts.create import (audit_problem, sources_safe)
from portal.predict_parts import common as COMMON
from portal.predict_parts import sets as SETS


# ---------------------------------------------------------------- html (every text through esc)
def session_label(req: dict, srcs, admin=False):
    """The configured label of a session source, found by its folder (never shown as a path)."""
    src = req.get("source") or {}
    return next((s["label"] for s in srcs if src.get("dir") and (admin or not s.get("problem")) and
                 os.path.normcase(str(s["dir"].resolve())) == os.path.normcase(str(src.get("dir")))), None)


def source_text(req: dict, srcs, admin=False) -> str:
    """A source with a problem (hidden, incompatible) lends its label only when admin=True (callers pass nothing
    yet, so for everyone it is "Session <id>" alone)."""
    return NM.source_label(req, session_label(req, srcs, admin))


def tab_title(text: str) -> str:
    """A browser-tab title safe for ctx.page's `{{...}}` substitution (it escapes & so entities would show): the
    braces of a free-text name become full-width ones."""
    return text.replace("{", "\uff5b").replace("}", "\uff5d")


def set_title(req: dict, srcs, admin=False) -> str:
    """What the analysis is called everywhere: its name, else "<source> · <date>" (names.display_name)."""
    return NM.display_name(req, session_label(req, srcs, admin))


def status_html(view: dict) -> str:
    bar = ""
    if view["progress"]:
        v, t = max(0, min(int(view["n_done"] or 0), int(view["n_total"]))), int(view["n_total"])
        bar = (f'<progress class="cj-predict-bar" max="{t}" value="{v}" '
               f'aria-label="{v} of {t} images done"></progress>')
    detail = f'<p class="cj-predict-detail">{esc(view["detail"])}</p>' if view["detail"] else ""
    return (f'<div class="cj-card cj-predict-status"><div class="cj-predict-top"><b>Status</b>'
            f'{chip(view["label"], view["tone"])}</div>{bar}{detail}{tech(view["tech"])}</div>')


def actions_html(cfg, d: Path, req: dict, st: dict, admin: bool, rows) -> str:
    ps = d.name
    parts = []
    if can_restart(st):
        parts.append(f'<form method="post" action="/predict/{esc(ps)}/start" class="cj-predict-inline">'
                     f'<button class="cj-predict-btn" type="submit">Start again</button></form>')
    if st.get("state") != "done":
        return "".join(parts)
    pkg = SETS.package_dir(cfg, ps)
    res = results_file(d)
    t_rev, t_res = task("review", ps), task("results", ps)
    # Review
    if pkg is not None:
        rev = (f'<a class="cj-predict-btn cj-predict-primary" href="/review/{esc(pkg.name)}/">Review</a>'
               f'<p class="cj-mut">Check the model’s outlines and fix mistakes. Start with the suggested '
               f'images below.</p>')
        if admin and t_rev and t_rev["state"] == "failed":     # e.g. built, but the registry line failed
            rev += tech(f"review task: {t_rev['error']}")
    elif t_rev and t_rev["state"] == "running":
        rev = '<p class="cj-predict-busy" role="status">Preparing the review… this page updates by itself.</p>'
    else:
        err = ""
        if t_rev and t_rev["state"] == "failed":
            err = msg_box("Preparing the review failed.", extra=tech(t_rev["error"]) if admin else "")
        rev = (f'{err}<form method="post" action="/predict/{esc(ps)}/review" class="cj-predict-inline">'
               f'<button class="cj-predict-btn cj-predict-primary" type="submit">Review</button></form>'
               f'<p class="cj-mut">Prepares the images for checking (about a minute), then opens the review.</p>')
    # Results
    stale = res is not None and edits_mtime(pkg) > res.stat().st_mtime
    if t_res and t_res["state"] == "running":
        out = '<p class="cj-predict-busy" role="status">Counting… this page updates by itself.</p>'
    else:
        err = ""
        if t_res and t_res["state"] == "failed":
            err = msg_box("Counting failed.", extra=tech(t_res["error"]) if admin else "")
        if res is not None:
            out = (f'{err}<a class="cj-predict-btn" href="/results/{esc(ps)}/">Results</a>'
                   f'<p class="cj-mut">EV counts, sizes and multilayer % per sample.</p>')
            if stale:
                out += (f'<form method="post" action="/predict/{esc(ps)}/results" class="cj-predict-inline">'
                        f'<p class="cj-mut">The review has changed since these results were counted.</p>'
                        f'<button class="cj-predict-btn" type="submit">Update results</button></form>')
        else:
            out = (f'{err}<form method="post" action="/predict/{esc(ps)}/results" class="cj-predict-inline">'
                   f'<button class="cj-predict-btn" type="submit">Results</button></form>'
                   f'<p class="cj-mut">Counts EVs per sample (model only until images are reviewed).</p>')
        out += (f'<p class="cj-mut"><a href="/results/{esc(ps)}/#cj-results-zips">Downloads</a>: '
                f'images, model outlines, reviewed labels.</p>')
    parts.append(f'<div class="cj-predict-acts"><div class="cj-card">{rev}</div><div class="cj-card">{out}</div>'
                 f'</div>')
    # spot check
    spot = spot_list(cfg, d, ps, rows)
    if spot:
        items = []
        for s in spot:
            name = f'Image {esc(s["image"])}' + (f' <span class="cj-mut cj-wrap">{esc(s["orig_name"])}</span>'
                                                  if s.get("orig_name") else "")
            link = (f'<a href="/review/{esc(pkg.name)}/?img={esc(s["file"])}">{name}</a>'
                    if pkg is not None and s.get("file") else name)
            items.append(f'<li><span class="cj-predict-li-main">{link}</span>'
                         f'<span class="cj-predict-tag">{esc(s.get("sample_label") or "")}</span></li>')
        parts.append(f'<h2 class="cj-sec">Check these first</h2><div class="cj-card"><p class="cj-mut">'
                     f'A few images spread across your samples. Checking them (strongly encouraged) shows how much '
                     f'to trust the counts, and your fixes count in the results.</p><ol class="cj-predict-list">'
                     f'{"".join(items)}</ol></div>')
    return "".join(parts)


def set_view(cfg, user, d: Path, req: dict, with_html: bool) -> dict:
    admin = user.get("role") == "admin"
    st = _read_json(d / PS.STATUS) or {}
    view = status_view(st, admin)
    ps = d.name
    busy = any((task(k, ps) or {}).get("state") == "running" for k in ("review", "results"))
    pkg, res = SETS.package_dir(cfg, ps), results_file(d)
    out = {"schema": SCHEMA_SET, "ps": ps, "state": view["state"], "label": view["label"], "detail": view["detail"],
           "n_done": view["n_done"], "n_total": view["n_total"], "live": (not view["final"]) or busy,
           "review_href": f"/review/{pkg.name}/" if pkg else None,
           "results_href": f"/results/{ps}/" if res else None}
    if with_html:
        rows = images_rows(d)
        out["html"] = {"status": status_html(view), "actions": actions_html(cfg, d, req, st, admin, rows)}
    return out


def item_of(cfg, user, d: Path, req: dict, st: dict, srcs) -> dict:
    ps = d.name
    view = status_view(st, user.get("role") == "admin")
    pkg = SETS.package_dir(cfg, ps)
    admin = user.get("role") == "admin"
    return {"ps": ps, "href": f"/predict/{ps}/", "title": set_title(req, srcs, admin),
            "named": bool(isinstance(req.get("name"), str) and req["name"].strip()),
            "created": req.get("created"), "owner": req.get("owner"), "model": req.get("model"),
            "source": source_text(req, srcs, admin=user.get("role") == "admin"), "state": view["state"], "label": view["label"],
            "n_done": view["n_done"], "n_total": view["n_total"],
            "review_href": f"/review/{pkg.name}/" if pkg else None,
            "results_href": f"/results/{ps}/" if results_file(d) else None}


def mine_html(items, admin) -> str:
    if not items:
        return '<p class="cj-mut">No analyses yet. Start one with “New analysis”.</p>'
    li = []
    for it in items:
        tone = {"done": "ok", "running": "run", "failed": "warn"}.get(it["state"], "wait")
        prog = f'{int(it["n_done"] or 0)} of {int(it["n_total"] or 0)} images' if it["state"] == "running" \
            else f'{int(it["n_total"] or 0)} images'
        who = f' · {esc(it["owner"])}' if admin else ""
        links = "".join(f'<a href="{esc(h)}">{t}</a>' for h, t in ((it["review_href"], "Review"),
                                                                    (it["results_href"], "Results")) if h)
        li.append(f'<li class="cj-card cj-predict-item"><div class="cj-predict-top">'
                  f'<a class="cj-predict-ttl" href="{esc(it["href"])}">{esc(it["title"])}</a>'
                  f'{chip(it["label"], tone)}</div>'
                  f'<p class="cj-mut">{esc(it["source"] + " · " + when(it["created"]) + " · ") if it.get("named") else ""}'
                  f'{esc(prog)}{who}</p>'
                  + (f'<p class="cj-predict-links">{links}</p>' if links else "") + '</li>')
    return f'<ul class="cj-predict-items">{"".join(li)}</ul>'


def page_mine(cfg, user, can_see) -> str:
    admin = user.get("role") == "admin"
    srcs, _ = sources_safe(cfg, user, can_see)
    items = [item_of(cfg, user, d, req, st, srcs) for d, req, st in list_sets(cfg, user, can_see)]
    convs = list_conversions(cfg, user, can_see)
    conv = ""
    if convs or user.get("role") in RAW_ROLES:
        conv = (f'<h2 class="cj-sec">{"Raw conversions" if admin else "My raw conversions"}</h2>'
                + conversions_html(convs, admin))
    return wrap(f'<div class="cj-predict"><div class="cj-predict-head"><div><h1>Predict</h1>'
                f'<p class="cj-mut">Count and size the EVs in your images with the lab’s model.</p></div>'
                f'<a class="cj-predict-btn cj-predict-primary" href="/predict/new/">New analysis</a></div>'
                f'<h2 class="cj-sec">{"All analyses" if admin else "My analyses"}</h2>{mine_html(items, admin)}'
                f'{conv}</div>')


def model_block(cfg, user, chosen=None) -> str:
    default, choices = allowed_models(cfg, user)
    if not choices:
        return msg_box(NO_MODEL)
    pick_id = chosen if any(c["id"] == chosen for c in choices) else default
    if len(choices) == 1:
        c = choices[0]
        q = f'<p class="cj-mut">{esc(c["quality"])}</p>' if c["quality"] else ""
        return (f'<fieldset class="cj-card cj-predict-fs"><legend>Model</legend>'
                f'<input type="hidden" name="model" value="{esc(c["id"])}">'
                f'<p><b>{esc(c["id"])}</b>{" (the lab’s production model)" if c["production"] else ""}</p>'
                f'{q}</fieldset>')
    opts = f'<option value=""{"" if pick_id else " selected"} disabled>Choose…</option>' + "".join(
        f'<option value="{esc(c["id"])}"{" selected" if c["id"] == pick_id else ""}>'
        f'{esc(c["id"])}{" (production)" if c["production"] else ""}</option>' for c in choices)
    qual = "".join(f'<li><b>{esc(c["id"])}</b>: {esc(c["quality"])}</li>' for c in choices if c["quality"])
    return (f'<fieldset class="cj-card cj-predict-fs"><legend>Model</legend>'
            f'<label class="cj-predict-field"><span>Model (keep the production model unless you know why)</span>'
            f'<select name="model" required>{opts}</select></label>'
            + (f'<ul class="cj-mut cj-predict-qual">{qual}</ul>' if qual else "") + '</fieldset>')


def page_new(cfg, user, can_see) -> str:
    admin = user.get("role") == "admin"
    srcs, err = sources_safe(cfg, user, can_see)
    cards = []
    for s in srcs:
        if s["problem"]:
            if admin:
                cards.append(f'<div class="cj-card cj-predict-src cj-predict-off"><b>{esc(s["label"])}</b>'
                             f'<p class="cj-mut">Not available: {esc(s["problem"])}</p>{tech(str(s["dir"]))}</div>')
            continue
        who = (f' · converted by {esc(s["owner"])}' if admin and s.get("owner") else
               ' · converted from raw' if s.get("owner") else "")
        cards.append(f'<a class="cj-card cj-predict-src" href="/predict/new/session/{esc(s["key"])}/">'
                     f'<b>{esc(s["label"])}</b><span class="cj-mut">Session {esc(s["session"])}{who}</span>'
                     + (tech(str(s["dir"])) if admin else "") + '</a>')
    if err and admin:
        cards.append(msg_box("The session list setting can't be read.", extra=tech(err)))
    no_model = "" if allowed_models(cfg, user)[1] else msg_box(NO_MODEL)
    sess = ("".join(cards) or '<p class="cj-mut">No processed sessions are set up for analysis yet'
            + (' (set CJ_PREDICT_SCAN_ROOTS or CJ_PREDICT_SOURCES).' if admin else
               '; ask the admin, or upload your own images.') + '</p>')
    raw = ""
    if user.get("role") in RAW_ROLES:
        repo, why = processing_repo(cfg)
        if repo is None:
            raw = ('<h2 class="cj-sec">Or convert raw images</h2>'
                   + msg_box("Raw conversion is not set up on this server.",
                             extra=tech(why) if admin else '<p>Ask the admin.</p>'))
        else:
            raw = ('<h2 class="cj-sec">Or convert raw images</h2><div class="cj-predict-srcs">'
                   '<a class="cj-card cj-predict-src" href="/predict/new/raw/"><b>Upload raw images</b>'
                   '<span class="cj-mut">.mrc with its .mrc.mdoc, or .idoc with its .tif tiles: made into tiles '
                   'the way the lab’s processing does it, then analysed here</span></a>'
                   + ('<a class="cj-card cj-predict-src" href="/predict/new/raw/sessions/"><b>Pick a raw session</b>'
                      '<span class="cj-mut">Imaging sessions on the server with no processed tiles yet</span></a>'
                      if admin else "") + '</div>')
    return wrap(
        '<div class="cj-predict"><p class="cj-mut"><a href="/predict/">← My analyses</a></p>'
        f'<h1>New analysis</h1>{no_model}'
        '<div class="cj-card cj-predict-what"><h2>What images work</h2><ul>'
        '<li>Processed cryo-EM tiles: 8-bit PNG or JPEG, contrast-adjusted the way the lab’s processing does '
        'it' + (' (raw MRC or TIFF montages: convert them first, below).</li>' if raw else
                ' (not raw MRC or TIFF).</li>') +
        '<li>The same kind of images and magnification as the model was trained on. If unsure, ask the admin.</li>'
        f'<li>Up to {COMMON.MAX_FILES} images per analysis. You must confirm the pixel size (nm per pixel) of each '
        'sample: sizes in nm depend on it.</li></ul></div>'
        '<h2 class="cj-sec">1. Pick a processed session</h2>'
        f'<div class="cj-predict-srcs">{sess}</div>'
        '<h2 class="cj-sec">Or upload your own tiles</h2>'
        '<a class="cj-card cj-predict-src" href="/predict/new/upload/"><b>Upload images</b>'
        f'<span class="cj-mut">Up to {COMMON.MAX_FILES} PNG or JPEG files, '
        f'{COMMON.MAX_FILE_BYTES // 2 ** 20} MB each</span></a>'
        f'{raw}</div>')


def name_field(value, prefill) -> str:
    """The Name box, prefilled with the default name; the prefill travels in a hidden field so an unchanged box is
    not stored as a name (chosen_name)."""
    return (f'<fieldset class="cj-card cj-predict-fs"><legend>Name</legend>'
            f'<label class="cj-predict-field"><span>Analysis name (optional; you can rename it later)</span>'
            f'<input name="name" maxlength="{PS.NAME_MAX}" value="{esc(value)}"></label>'
            f'<input type="hidden" name="name_default" value="{esc(prefill)}"></fieldset>')


def default_name_for(source: dict, label=None) -> str:
    """The prefill of the Name field: what the analysis would be called without a name (kept as typed if unchanged)."""
    return NM.default_name({"source": source, "created": now_iso()}, label)[:PS.NAME_MAX]


def page_session_form(cfg, user, src: dict, t: dict, form=None, error=None) -> str:
    admin = user.get("role") == "admin"
    form = form or {}
    picked = set(form["samples"] if "samples" in form else
                 [g.get("gid", f"g{i}") for i, g in enumerate(t["groups"]) if not g["mixed"]])
    rows = []
    for i, g in enumerate(t["groups"]):
        gid = g.get("gid", f"g{i}")
        nm_val = form.get(f"nm_{gid}", f"{g['nm']:g}" if g["nm"] else "")
        if g["mixed"]:
            note = ('<span class="cj-predict-warn">Pixel sizes in this sample differ by more than '
                    f'{NM_TOLERANCE * 100:g} % (' + esc(", ".join(f"{v:g}" for v in g["nm_values"])) +
                    ' nm/px): it can’t be analysed as one sample. Ask the admin.</span>')
            rows.append(f'<tr><td data-l="Sample"><label class="cj-predict-off"><input type="checkbox" disabled> '
                        f'{esc(g["label"])}</label></td><td data-l="Tiles" class="cj-num">{len(g["names"])}</td>'
                        f'<td data-l="nm/px">{note}</td></tr>')
            continue
        if g["nm_range"]:
            # several values within the tolerance: each tile keeps its own; a number typed here replaces them all
            nm_val = form.get(f"nm_{gid}", "")
            nm_cell = (f'<span class="cj-predict-range">{esc(nm_range_text(g["nm_min"], g["nm_max"]))} nm/px across '
                       f'tiles, within {NM_TOLERANCE * 100:g} %; each tile keeps its own value</span> '
                       f'<span class="cj-mut">({NM_FROM_MANIFEST})</span><br>'
                       f'<input class="cj-predict-nm" name="nm_{gid}" inputmode="decimal" '
                       f'aria-label="one nm per pixel for all tiles of {esc(g["label"])} (optional)" '
                       f'placeholder="each tile’s own" value="{esc(nm_val)}">'
                       f' <span class="cj-mut">leave empty, or type one value to use for all '
                       f'{len(g["names"])} tiles</span>')
        else:
            src_note = NM_FROM_MANIFEST if g["nm"] else "not recorded: enter it"
            if g["nm_missing"] and g["nm_values"]:
                src_note = (f"not recorded for {g['nm_missing']} of {len(g['names'])} tiles (the others: "
                            f"{nm_range_text(g['nm_min'], g['nm_max'])}): enter one value for all of them")
            nm_cell = (f'<input class="cj-predict-nm" name="nm_{gid}" inputmode="decimal" '
                       f'aria-label="nm per pixel for {esc(g["label"])}" value="{esc(nm_val)}" required>'
                       f' <span class="cj-mut">{esc(src_note)}</span>')
        rows.append(
            f'<tr><td data-l="Sample"><label><input type="checkbox" name="samples" value="{gid}" '
            f'data-n="{len(g["names"])}"{" checked" if gid in picked else ""}> {esc(g["label"])}</label></td>'
            f'<td data-l="Tiles" class="cj-num">{len(g["names"])}</td>'
            f'<td data-l="nm/px">{nm_cell}</td></tr>')
    label_field = ""
    if t["column"] is None:
        label_field = (f'<label class="cj-predict-field"><span>Sample name (shown in the results)</span>'
                       f'<input name="label" maxlength="80" value="{esc(form.get("label", src["label"]))}" '
                       f'required></label>')
    how = form.get("how", "all")
    count = form.get("count", "")
    over = t["n"] > COMMON.MAX_FILES
    err = msg_box("", extra=f'<div>{problem_list(error)}</div>') if error else ""
    n0 = sum(len(g["names"]) for i, g in enumerate(t["groups"]) if g.get("gid", f"g{i}") in picked and not g["mixed"])
    prefill = form["name_default"] if "name_default" in form else default_name_for(
        {"kind": "session", "session": src["session"]}, src["label"])
    name_val = form["name"] if "name" in form else prefill
    kept = t.get("kept")
    kept_note = kept_hidden = ""
    if kept:
        kept_note = (f'<p class="cj-predict-kept"><b>{t["n"]} kept tile{"" if t["n"] == 1 else "s"} from the '
                     f'gallery</b>{" (keep and maybe)" if kept["kept_maybe"] else ""} · '
                     f'<a href="/gallery/{esc(kept["from"])}/">back to the Gallery</a></p>')
        kept_hidden = "".join(f'<input type="hidden" name="{k}" value="{esc(v)}">' for k, v in kept.items())
    return wrap(
        f'<div class="cj-predict"><p class="cj-mut"><a href="/predict/new/">← Back</a></p>'
        f'<h1>{esc(src["label"])}</h1><p class="cj-mut">Session {esc(src["session"])} · {t["n"]} tiles'
        f'</p>{kept_note}{tech(str(src["dir"])) if admin else ""}{err}'
        f'<form method="post" class="cj-predict-form" id="cj-predict-session" data-max="{COMMON.MAX_FILES}">'
        f'<input type="hidden" name="sig" value="{esc(t["sig"])}">{kept_hidden}'
        f'<fieldset class="cj-card cj-predict-fs"><legend>Samples and pixel size</legend>{label_field}'
        f'<div class="cj-predict-tablewrap"><table class="cj-predict-table"><thead><tr><th>Sample</th><th>Tiles</th>'
        f'<th>nm per pixel</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
        f'<label class="cj-predict-confirm"><input type="checkbox" name="confirm" value="yes" required> '
        f'I confirm the pixel size of each sample above</label>'
        f'<p class="cj-mut">Sizes in nm are computed from it. Your name is recorded as the person who confirmed it.'
        f'</p></fieldset>'
        f'<fieldset class="cj-card cj-predict-fs"><legend>Tiles</legend>'
        + ('<p>Exactly the kept tiles of the ticked samples.</p>' if kept else
           f'<label class="cj-predict-radio"><input type="radio" name="how" value="all"'
           f'{" checked" if how != "count" else ""}> All tiles of the ticked samples</label>'
           f'<label class="cj-predict-radio"><input type="radio" name="how" value="count"'
           f'{" checked" if how == "count" else ""}> Only <input class="cj-predict-count" name="count" '
           f'inputmode="numeric" aria-label="number of tiles" value="{esc(count)}"> tiles, spread across the '
           f'ticked samples</label>') +
        f'<p class="cj-mut" id="cj-predict-sel" role="status"></p>'
        + estimate_html(cfg, min(n0, COMMON.MAX_FILES))
        + (f'<p class="cj-mut">At most {COMMON.MAX_FILES} tiles per analysis: {"untick a sample, or reject some tiles in the Gallery" if kept else "choose a number"}.</p>' if over else "")
        + f'</fieldset>{name_field(name_val, prefill)}{model_block(cfg, user, form.get("model"))}'
        f'<p><button class="cj-predict-btn cj-predict-primary" type="submit" data-busy="Preparing your images… '
        f'this can take a minute">Start analysis</button></p></form></div>')


def page_upload_form(cfg, user, form=None, error=None) -> str:
    form = form or {}
    up_prefill = form["name_default"] if "name_default" in form else default_name_for({"kind": "upload"})
    err = msg_box("", extra=f'<div>{problem_list(error)}</div>') if error else ""
    again = '<p class="cj-mut">Choose the files again (a browser forgets them after a refusal).</p>' if error else ""
    return wrap(
        f'<div class="cj-predict"><p class="cj-mut"><a href="/predict/new/">← Back</a></p>'
        f'<h1>Upload images</h1>{err}{again}'
        f'<form method="post" action="/predict/new/upload/" enctype="multipart/form-data" class="cj-predict-form" '
        f'id="cj-predict-upload" data-cj-upload="tiles" data-max-files="{COMMON.MAX_FILES}" '
        f'data-max-bytes="{COMMON.MAX_FILE_BYTES}">'
        f'<div class="cj-predict-msg cj-predict-t-warn" data-cj-upload-msg role="alert" hidden></div>'
        f'<noscript><p class="cj-predict-msg">JavaScript is off: this form then sends everything in one request, '
        f'at most {COMMON.NOJS_MAX_BYTES // 2 ** 20} MB in total. Turn JavaScript on for bigger sets (files are then '
        f'sent one at a time).</p></noscript>'
        # the text fields come BEFORE the files: the browser sends parts in this order, so the server checks the
        # tick, nm/px and model as soon as the first file part arrives (fail fast, before any big file lands)
        f'<fieldset class="cj-card cj-predict-fs"><legend>Sample and pixel size</legend>'
        f'<label class="cj-predict-field"><span>Sample name (shown in the results)</span>'
        f'<input name="sample_label" maxlength="80" required value="{esc(form.get("sample_label", ""))}"></label>'
        f'<label class="cj-predict-field"><span>Pixel size (nm per pixel)</span>'
        f'<input class="cj-predict-nm" name="nm_per_px" inputmode="decimal" required '
        f'value="{esc(form.get("nm_per_px", ""))}"></label>'
        f'<p class="cj-mut">{esc(NM_HINT)}</p>'
        f'<label class="cj-predict-confirm"><input type="checkbox" name="confirm" value="yes" required> '
        f'I confirm the pixel size</label>'
        f'<p class="cj-mut">Sizes in nm are computed from it. Your name is recorded as the person who confirmed it.'
        f'</p></fieldset>'
        f'{name_field(form["name"] if "name" in form else up_prefill, up_prefill)}'
        f'{model_block(cfg, user, form.get("model"))}'
        f'<fieldset class="cj-card cj-predict-fs"><legend>Images</legend>'
        f'<label class="cj-predict-field"><span>Processed 8-bit PNG or JPEG tiles of one sample '
        f'(up to {COMMON.MAX_FILES}, '
        f'{COMMON.MAX_FILE_BYTES // 2 ** 20} MB each)</span><input type="file" name="files" multiple required '
        f'accept=".png,.jpg,.jpeg,image/png,image/jpeg"></label>'
        f'<p class="cj-mut" id="cj-predict-files" data-cj-upload-info role="status"></p>'
        f'{estimate_html(cfg, 0)}</fieldset>'
        f'<p><button class="cj-predict-btn cj-predict-primary" type="submit" data-busy="Uploading… keep this '
        f'page open">Upload and start</button></p></form></div>')


def rename_form(ps: str, req: dict) -> str:
    """A small inline form on the analysis page (the owner and the admin are the only ones who see the page)."""
    return (f'<form method="post" action="/predict/{esc(ps)}/rename" class="cj-predict-inline cj-predict-rename">'
            f'<label>Name <input name="name" maxlength="{PS.NAME_MAX}" value="{esc(req.get("name") or "")}" '
            f'placeholder="{esc(NM.default_name(req))}"></label> '
            f'<button class="cj-predict-btn" type="submit">Rename</button></form>')


def page_set(cfg, user, d: Path, req: dict, can_see=None) -> str:
    admin = user.get("role") == "admin"
    v = set_view(cfg, user, d, req, True)
    rows = images_rows(d)
    srcs, _ = sources_safe(cfg, user, can_see)
    samples = "".join(f'<tr><td data-l="Sample">{esc(s["label"])}</td><td data-l="Images" class="cj-num">{s["n"]}'
                      f'</td><td data-l="nm/px">{esc(s["nm_per_px"])}</td>'
                      f'<td data-l="Confirmed by">{esc(s["confirmed_by"] or "?")}</td></tr>'
                      for s in samples_of(rows))
    owner = f'<div><dt>Owner</dt><dd>{esc(req.get("owner"))}</dd></div>' if admin else ""
    src_tech = (tech(f"{d} · source {json.dumps(req.get('source'))}") + tech(audit_problem(d.name))) if admin else ""
    return wrap(
        f'<div class="cj-predict" id="cj-predict-set" data-ps="{esc(d.name)}" data-live="{int(v["live"])}" '
        f'data-every="{COMMON.REFRESH_S}"><p class="cj-mut"><a href="/predict/">← My analyses</a></p>'
        f'<h1>{esc(set_title(req, srcs, admin))}</h1>'
        + rename_form(d.name, req)
        + (f'<p class="cj-mut cj-wrap">{esc(d.name)}</p>' if user.get("role") != "basic" else "") +
        f'<p id="cj-predict-err" class="cj-mut" role="status" hidden></p>'
        f'<section id="cj-predict-status">{v["html"]["status"]}</section>'
        f'<section id="cj-predict-actions">{v["html"]["actions"]}</section>'
        f'<h2 class="cj-sec">Details</h2><div class="cj-card"><dl class="cj-predict-kv">'
        f'<div><dt>Images from</dt><dd>{esc(source_text(req, srcs, admin=admin))}</dd></div>'
        f'<div><dt>Model</dt><dd>{esc(req.get("model"))}</dd></div>'
        f'<div><dt>Started</dt><dd>{esc(when(req.get("created")))}</dd></div>{owner}</dl>'
        f'<div class="cj-predict-tablewrap"><table class="cj-predict-table"><thead><tr><th>Sample</th><th>Images</th>'
        f'<th>nm/px</th><th>Confirmed by</th></tr></thead><tbody>{samples}</tbody></table></div>{src_tech}</div>'
        f'</div>')


# ---------------------------------------------------------------- the preparing page (a set made in the background)
def prep_text(t: dict) -> str:
    return f"Preparing your images: {int(t.get('done') or 0)} of {int(t.get('total') or 0)}"


def prep_fragment(t: dict, tid: str, admin: bool) -> str:
    """The progress page's status box (rendered and escaped here; predict.js swaps it in)."""
    if t["state"] == "failed":
        tech_ = f"\n(admin) {t['tech']}" if admin and t.get("tech") else ""
        retry = ""
        if t.get("upload"):
            retry = (f'<form method="post" action="/predict/preparing/{esc(tid)}/retry" class="cj-predict-inline">'
                     f'<button class="cj-predict-btn cj-predict-primary" type="submit">Try again</button></form> '
                     f'<span class="cj-mut">Your uploaded images are kept for an hour.</span>')
        return msg_box("", extra=f'<div>{problem_list(str(t["error"]) + tech_)}</div>') + \
            f'<p>{retry} <a href="{esc(t["back"])}">Back to the form</a></p>'
    if t["state"] == "done":
        return (f'<div class="cj-card cj-predict-status"><b>Ready</b><p><a class="cj-predict-btn cj-predict-primary" '
                f'href="/predict/{esc(t["ps"])}/">Open the analysis</a></p></div>')
    total, done = int(t.get("total") or 0), int(t.get("done") or 0)
    bar = (f'<progress class="cj-predict-bar" max="{total}" value="{max(0, min(done, total))}" '
           f'aria-label="{done} of {total} images prepared"></progress>' if total else
           '<progress class="cj-predict-bar" aria-label="starting"></progress>')
    phase = {"checking": "Checking each image.", "placing": "Adding them to your analysis."}.get(t.get("phase"), "")
    return (f'<div class="cj-card cj-predict-status"><div class="cj-predict-top"><b>{esc(prep_text(t))}</b></div>'
            f'{bar}<p class="cj-mut">{esc(phase)} You can close this page: the analysis appears under '
            f'“My analyses” when it is ready.</p></div>')


def prep_view(t: dict, tid: str, admin: bool, with_html: bool) -> dict:
    out = {"schema": "cj_predict_prep/1", "id": tid, "state": t["state"], "done": t["done"], "total": t["total"],
           "text": prep_text(t), "live": t["state"] == "running",
           "href": f"/predict/{t['ps']}/" if t["state"] == "done" else None}
    if with_html:
        out["html"] = {"status": prep_fragment(t, tid, admin)}
    return out


def page_prep(t: dict, tid: str, admin: bool) -> str:
    return wrap(f'<div class="cj-predict" id="cj-predict-prep" data-tid="{esc(tid)}" data-every="2" '
                f'data-live="{int(t["state"] == "running")}"><p class="cj-mut"><a href="/predict/">← My analyses</a></p>'
                f'<h1>Starting your analysis</h1><p id="cj-predict-err" class="cj-mut" role="status" hidden></p>'
                f'<section id="cj-predict-prepstatus">{prep_fragment(t, tid, admin)}</section>'
                f'<noscript><p class="cj-mut">JavaScript is off: reload this page to update it.</p></noscript></div>')


# ---------------------------------------------------------------- the raw conversions list (My analyses)
def conversions_html(convs, admin) -> str:
    if not convs:
        return '<p class="cj-mut">No raw conversions yet. Start one from “New analysis”.</p>'
    li = []
    for d, meta, st in convs:
        v = conv_status_view(meta, st, False)
        n = f' · {int(st.get("n_tiles") or 0)} tiles' if st.get("state") == "done" else ""
        who = f' · {esc(meta.get("owner"))}' if admin else ""
        li.append(f'<li class="cj-card cj-predict-item"><div class="cj-predict-top">'
                  f'<a class="cj-predict-ttl" href="/predict/convert/{esc(d.name)}/">'
                  f'{esc(PSRC.default_label(RAW_ID_RE.match(d.name).group(1)))} · {esc(meta.get("recipe"))} recipe'
                  f'</a>{chip(v["label"], v["tone"])}</div>'
                  f'<p class="cj-mut">{esc(when(meta.get("created")))}{n}{who}</p></li>')
    return f'<ul class="cj-predict-items">{"".join(li)}</ul>'


# ---------------------------------------------------------------- a source that can't be read
def unreadable_page(src: dict, e: Refusal, user) -> str:
    extra = tech(e.tech) if user.get("role") == "admin" else ""
    return wrap(f'<div class="cj-predict"><p class="cj-mut"><a href="/predict/new/">← Back</a></p>'
                f'<h1>{esc(src["label"])}</h1>{msg_box(e.msg, extra=extra)}</div>')
