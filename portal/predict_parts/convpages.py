"""
Predict, the raw pages (every text through esc): the raw upload form, a conversion's page and stitch card, the
raw session pages, and the conversion settings (readout, Adjust panel).

Split out of portal/modules/predict.py (round 7, P7; no behaviour change); predict.py re-exports every name.
Names the tests patch are used from other parts only as <PART>.<name> (portal/predict_parts/__init__.py).
"""
from __future__ import annotations

import csv
import json
import re
from datetime import datetime
from pathlib import Path

from jobs import convert_raw as CR
from portal import convert_settings as CS
from portal import predict_sources as PSRC
from portal import upload_sessions as US

from portal.predict_parts.common import (CONVERT_INPUT, CONVERT_TILES, RAW_MAX_FILES, RAW_MAX_FILE_BYTES, RAW_ROLES,
                                         _plain, chip, esc, msg_box, problem_list, tech, when, wrap)
from portal.predict_parts.raw import (processing_repo, raw_sessions, session_recipe)
from portal.predict_parts.runner import (_converter_cmd, can_register_again, can_retry, can_stitch, conv_paths,
                                         conv_status_view, inputs_kept_until, log_tail, registration_line,
                                         stitch_busy)
from portal.predict_parts.create import converted_sources
from portal.predict_parts import common as COMMON
from portal.predict_parts import raw as RAW


# ---------------------------------------------------------------- raw form widgets (recipe list, GPU box)
def recipe_select(repo, chosen=None, session=None) -> str:
    """The recipe <select> (default first) and the warning about the default."""
    names = RAW.recipes(repo) if repo else []
    chosen = chosen if chosen in names else "default"
    def summ(n):                      # the values, once known (cached; the page script fills the others in)
        t = settings_summary(CS.peek_recipe(repo, n, recipe_file_name(n))) if repo else ""
        return f" · {t}" if t else ""
    opts = (f'<option value="default"{" selected" if chosen == "default" else ""}>'
            f'default ({esc(CR.DEFAULT_RECIPE)}, the lab’s template){esc(summ("default"))}</option>' +
            "".join(f'<option value="{esc(n)}"{" selected" if n == chosen else ""}>'
                    f'{esc(n)}{esc(summ(n))}</option>' for n in names))
    own = session_recipe(session, names) if session else None
    hint = (f' This session has its own recipe: <b>{esc(own)}</b>.' if own else
            ' A session’s own recipe is named after it (20260827-2 → 20260827_2).')
    return (f'<label class="cj-predict-field"><span>Recipe (the lab’s processing settings for a session: denoise, '
            f'contrast)</span><select name="recipe" data-cj-recipe>{opts}</select></label>'
            f'<p class="cj-predict-warn" data-cj-recipe-warn>“default” is the lab’s template ({esc(CR.DEFAULT_RECIPE)}): '
            f'its contrast and denoise were tuned on that session, and a sampling far from it is refused. Pick the '
            f'recipe of the session these images come from when there is one.{hint}</p>'
            + (recipe_readout_box(repo, chosen) if repo else ""))


def gpu_box(checked=False) -> str:
    return (f'<label class="cj-predict-radio"><input type="checkbox" name="gpu" value="yes"'
            f'{" checked" if checked else ""}> Use the GPU (faster; only if it is free. Results differ from the CPU '
            f'by ±1 grey level on under 0.01 % of pixels)</label>')


# ---------------------------------------------------------------- raw pages (every text through esc)
def conv_fragments(cfg, user, can_see, d: Path, meta: dict, st: dict, details=None) -> dict:
    """details: {"error": text} or {"saved": True} for the Session details form (upload_sessions)."""
    admin = user.get("role") == "admin"
    paths = conv_paths(d, meta)
    v = conv_status_view(meta, st, admin, paths)
    status = (f'<div class="cj-card cj-predict-status"><div class="cj-predict-top"><b>Status</b>'
              f'{chip(v["label"], v["tone"])}</div>'
              + (f'<p class="cj-predict-detail">{esc(v["detail"])}</p>' if v["detail"] else "") + tech(v["tech"])
              + '</div>')
    parts = []
    warns = [_plain(w, paths) for w in st.get("warnings") or []]
    if st.get("state") == "done":
        src = next((s for s in converted_sources(cfg, user, can_see) if s.get("conversion") == d.name), None)
        if src is not None and not src["problem"]:
            parts.append(f'<div class="cj-card"><a class="cj-predict-btn cj-predict-primary" '
                         f'href="/predict/new/session/{esc(src["key"])}/">Analyse these tiles</a>'
                         f'<p class="cj-mut">Pick samples and confirm the pixel size, as for a processed session. '
                         f'Also listed under “New analysis”.</p></div>')
        elif src is not None:
            parts.append(msg_box(f"These tiles can’t be analysed: {src['problem']}"))
        res = st.get("result") or {}
        nm = ", ".join(f"{float(x):g}" for x in res.get("nm_per_px") or [])
        parts.append(f'<div class="cj-card"><dl class="cj-predict-kv">'
                     f'<div><dt>Tiles</dt><dd>{int(st.get("n_tiles") or 0)}</dd></div>'
                     f'<div><dt>nm per pixel</dt><dd>{esc(nm or "?")}</dd></div>'
                     f'<div><dt>Montages</dt><dd>{esc(", ".join(res.get("montages") or []))}</dd></div>'
                     f'<div><dt>Computed on</dt><dd>{esc(res.get("compute") or "?")}</dd></div></dl></div>')
        if (d / CONVERT_TILES / PSRC.MANIFEST).is_file():
            parts.append(f'<p><a class="cj-predict-btn" href="/predict/convert/{esc(d.name)}/tiles.zip" '
                         f'data-cj-tiles-zip>Download tiles (zip)</a> <span class="cj-mut">The PNG tiles and their '
                         f'MANIFEST.csv.</span></p>')
        parts.append(stitch_card(d, st, admin, paths))
        parts.append(US.conversion_details_html(cfg, user, d, meta, st, **(details or {})))
    if warns:
        parts.append('<h2 class="cj-sec">Warnings</h2><div class="cj-card"><ul class="cj-predict-warns">'
                     + "".join(f"<li>{esc(w)}</li>" for w in warns) + '</ul></div>')
    left = [x for x in meta.get("skipped") or [] if isinstance(x, dict)]
    if left:                                         # the upload's files left out (never converted)
        parts.append(f'<div class="cj-card" id="cj-predict-skipped"><b>Left out {len(left)} file(s)</b>'
                     f'<p class="cj-mut">These uploaded files were not converted (and were not kept):</p>'
                     '<ul class="cj-predict-warns">'
                     + "".join(f'<li>{esc(str(x.get("name")))}: {esc(_plain(str(x.get("why")), paths))}</li>'
                               for x in left[:200])
                     + (f"<li>… and {len(left) - 200} more</li>" if len(left) > 200 else "") + '</ul></div>')
    until = inputs_kept_until(cfg, d, st)
    if until is not None:
        parts.append(f'<p class="cj-mut">The uploaded raw files are kept until '
                     f'{esc(when(datetime.fromtimestamp(until).isoformat(timespec="seconds")))}, then removed.</p>')
    if admin:
        acts = []
        base = f'/predict/convert/{esc(d.name)}'
        if st.get("state") in ("queued", "running"):
            acts.append(f'<form method="post" action="{base}/stop" class="cj-predict-inline">'
                        f'<button class="cj-predict-btn" type="submit">Stop</button></form>')
        elif stitch_busy(st):
            acts.append(f'<form method="post" action="{base}/stop" class="cj-predict-inline">'
                        f'<button class="cj-predict-btn" type="submit">Stop making the montages</button></form>')
        if can_stitch(meta, d, st):
            acts.append(f'<form method="post" action="{base}/stitch" class="cj-predict-inline">'
                        f'<button class="cj-predict-btn" type="submit">Make the montages</button></form>')
        if can_register_again(cfg, meta, st):
            acts.append(f'<form method="post" action="{base}/register" class="cj-predict-inline">'
                        f'<button class="cj-predict-btn" type="submit">Register again</button></form>')
        line = registration_line(cfg, meta, st)
        if line:
            acts.append(f'<p class="cj-mut" data-cj-registration>{esc(line)}</p>')
        if can_retry(meta, d, st):
            acts.append(f'<form method="post" action="{base}/retry" class="cj-predict-inline">'
                        f'<button class="cj-predict-btn" type="submit">Retry</button></form>')
        if until is not None:
            acts.append(f'<form method="post" action="{base}/inputs" class="cj-predict-inline">'
                        f'<button class="cj-predict-btn" type="submit">Delete raw files now</button></form>')
        if acts:
            parts.append(f'<div class="cj-card cj-predict-adminacts"><b>Admin</b>{"".join(acts)}</div>')
    if admin and st.get("state") in ("failed", "running", "done"):
        tail = log_tail(d)
        if tail:
            parts.append(f'<details class="cj-card"><summary>Converter log (admin)</summary>'
                         f'<pre class="cj-predict-log">{esc(tail)}</pre></details>')
    return {"status": status, "result": "".join(parts), "live": not v["final"] or stitch_busy(st),
            "state": st.get("state")}


STITCH_LABEL = {"queued": ("Waiting", "wait"), "running": ("Being made", "run"), "done": ("Ready", "ok"),
                "failed": ("Not made", "warn"), "skipped": ("Not made", "warn")}


def stitch_card(d: Path, st: dict, admin: bool, paths=()) -> str:
    """The Montages card on a done conversion: where its stitch job stands, its warnings; empty when it never had
    one (a conversion made before round 7 shows that the montages can be made)."""
    s = st.get("stitch") if isinstance(st.get("stitch"), dict) else None
    if s is None:
        return ('<div class="cj-card" id="cj-predict-montages"><div class="cj-predict-top"><b>Montages</b>'
                f'{chip("Not made", "wait")}</div><p class="cj-mut">This conversion was made before montages were '
                'stitched automatically. The admin can make them while the raw files are kept.</p></div>')
    label, tone = STITCH_LABEL.get(s.get("state"), ("Unknown", "wait"))
    n = int(s.get("n") or 0)
    detail = {"queued": "The stitched montages are made next, after any conversion that is waiting.",
              "running": "Stitching the montages (about half a minute each). This page updates by itself.",
              "done": f"{n} stitched montage(s) made.",
              }.get(s.get("state"))
    if s.get("state") in ("failed", "skipped"):
        why = ("Stopped by the admin." if s.get("kind") == "stopped" else
               "It ran too long." if s.get("kind") == "timeout" else
               _plain(str(s.get("error") or "unknown"), paths))
        detail = (f"Some or all montages could not be stitched: {why} The tiles are not affected."
                  + (" The montages made before are kept." if s.get("kept_previous") else ""))
    warns = [_plain(str(w), paths) for w in s.get("warnings") or [] if "stitching failed" not in str(w)]
    tech_line = None
    if admin:
        tech_line = "; ".join(f"{k} {s.get(k)}" for k in ("job", "kind", "pid", "queued", "started", "finished")
                              if s.get(k))
    return ('<div class="cj-card" id="cj-predict-montages"><div class="cj-predict-top"><b>Montages</b>'
            f'{chip(label, tone)}</div>'
            + (f'<p class="cj-predict-detail">{esc(detail)}</p>' if detail else "")
            + ('<ul class="cj-predict-warns">' + "".join(f"<li>{esc(w)}</li>" for w in warns) + "</ul>"
               if warns else "") + tech(tech_line) + '</div>')


# ---------------------------------------------------------------- conversion settings (round 7 phase 2, piece CU)
# The readout (`convert_raw describe`, a child process, cached by portal/convert_settings.py) and the "Adjust settings"
# panel (`convert_raw preview`, one at a time). The processing code is never imported here.
CPREV = "cprev"                           # <raw base>/.portal/cprev/<cid>/<token>/{before,after}.png
PREV_TOKEN_RE = re.compile(r"^[0-9a-f]{12}$")
PREV_KEEP_S = 3600
PREV_FILES = ("before.png", "after.png")
MAX_TILE_NAMES = 2000


def recipe_file_name(name) -> str:
    return CR.DEFAULT_RECIPE if name == "default" else str(name)


def settings_summary(doc) -> str:
    """One line for a recipe <option>: 'bandpass 2 nm · clip 1–99 %'."""
    try:
        dn, c = doc["denoise"], doc["contrast"]
        if not dn.get("on"):
            den = "no denoise"
        elif dn.get("bandpass_nm") is not None:
            den = f"bandpass {float(dn['bandpass_nm']):g} nm"
        else:
            den = str(dn.get("method") or "denoise")
        return f"{den} · clip {float(c['clip_lo_pct']):g}–{float(c['clip_hi_pct']):g} %"
    except (KeyError, TypeError, ValueError):
        return ""


def _fnum(v) -> str:
    try:
        return f"{float(v):g}"
    except (TypeError, ValueError):
        return "?"


def recipe_readout(doc, err, name) -> str:
    """The readout under the recipe <select> (also the fragment the script fetches)."""
    if doc is None:
        return (f'<p class="cj-mut">The values of “{esc(name)}” could not be read'
                f'{": " + esc(err) if err else ""}.</p>')
    src = doc.get("source") or {}
    return (settings_rows(doc) + f'<p class="cj-mut cj-convset-src">From {esc(src.get("recipe_file") or "?")}'
            f' · version {esc(str(src.get("recipe_md5") or "")[:8])}</p>')


def recipe_readout_box(repo, chosen) -> str:
    """The server-side readout for the chosen recipe: the cached values when there are some, else a placeholder that the
    script fills in (no child process at page load)."""
    doc = CS.peek_recipe(repo, chosen, recipe_file_name(chosen)) if repo else None
    inner = (recipe_readout(doc, None, chosen) if doc else
             '<p class="cj-mut" data-cj-convset-wait>Loading the values of this recipe… (needs JavaScript)</p>')
    return (f'<div class="cj-card cj-convset-box" data-cj-recipe-readout aria-live="polite">'
            f'<b>What this recipe does</b><div data-cj-recipe-values>{inner}</div></div>')


def raw_kept_reason(d: Path, meta: dict, st: dict) -> str | None:
    """None when the conversion's raw input is still there; else why not (shown next to the disabled buttons)."""
    if st.get("inputs_removed"):
        return (f"The raw files were removed on {when(st.get('inputs_removed'))} (they are kept for a limited time). "
                f"Upload them again to convert with other settings.")
    if (meta.get("source") or {}).get("kind") == "upload" and not (d / CONVERT_INPUT).is_dir():
        return "The raw files are no longer kept for this conversion. Upload them again to convert with other settings."
    if not meta.get("inputs") or not all(Path(p).is_file() for p in meta["inputs"]):
        return "The raw files are no longer there. Upload them again to convert with other settings."
    return None


def tile_names(d: Path) -> list:
    """The MANIFEST's output_name column (the tiles that can be previewed)."""
    try:
        with open(d / CONVERT_TILES / PSRC.MANIFEST, newline="", encoding="utf-8-sig") as f:
            return [r["output_name"] for _, r in zip(range(MAX_TILE_NAMES), csv.DictReader(f)) if r.get("output_name")]
    except (OSError, KeyError, csv.Error):
        return []


def settings_rows(doc, eff=None, rec=None) -> str:
    """The values of a describe document as a <dl> (every value escaped). eff: the values the conversion really used
    (recipe + its overrides); rec: the bare recipe's values (to tell a custom bandpass from the recipe's)."""
    dn, c, px = doc.get("denoise") or {}, doc.get("contrast") or {}, doc.get("pixel") or {}
    eff, rec = eff or {}, rec or doc.get("params") or {}
    on = bool(dn.get("on")) if eff.get("denoise") is None else eff.get("denoise") is not False
    den = "Off: the tiles are not denoised."
    if on:
        den = dn.get("text") or "On"
        b, rb = eff.get("bandpass_nm"), rec.get("bandpass_nm")
        if isinstance(b, (int, float)) and isinstance(rb, (int, float)) and abs(b - rb) > 1e-9:
            den = f"Gaussian bandpass, keeps structures >{_fnum(b)} nm (the recipe: {den})"
    lo = eff.get("clip_lo_pct", c.get("clip_lo_pct"))
    hi = eff.get("clip_hi_pct", c.get("clip_hi_pct"))
    rows = [("Denoise", den),
            ("Clip low / high", f"{_fnum(lo)} % / {_fnum(hi)} % of the pixel values"),
            ("Level sampling", c.get("sampling_text") or "measured on the data as acquired"),
            ("Tile pixel size", f"{_fnum(px.get('target_nm_px'))} nm/px ({_fnum(px.get('target_A_px'))} Å/px)")]
    if doc.get("variant"):
        rows.append(("Tile variant", str(doc.get("variant"))))
    if on and dn.get("match_contrast"):
        rows.append(("Denoised tile contrast", f"levels from {c.get('levels_from') or 'the clean pass'}"))
    return ('<dl class="cj-predict-kv cj-convset-kv">'
            + "".join(f'<div><dt>{esc(k)}</dt><dd>{esc(v)}</dd></div>' for k, v in rows) + '</dl>')


def bare_recipe(cfg, repo, meta: dict, doc) -> dict:
    """The recipe's own four values (never the conversion's: `describe --conv` params = recipe + recorded overrides)."""
    name = str(meta.get("recipe") or "default")
    rdoc, _err = CS.describe_recipe(_converter_cmd(), repo, name, recipe_file_name(name))
    return dict((rdoc or {}).get("params") or (doc or {}).get("params") or {})


def conv_values(doc: dict, meta: dict) -> dict:
    """What the conversion used (or will use): the recorded effective values, then what a queued re-conversion asked for
    (its overrides are only recorded when it finishes)."""
    return {**(doc.get("params_effective") or {}), **(meta.get("params") or {})}


def _slider(name, label, lo, hi, step, val, unit, disabled=False, hint="") -> str:
    v = _fnum(val)
    return (f'<div class="cj-convset-row"><label for="cj-cs-{name}">{esc(label)}</label>'
            f'<input type="range" id="cj-cs-{name}" name="{name}" min="{lo:g}" max="{hi:g}" step="{step:g}" '
            f'value="{esc(v)}" data-cj-slider data-recipe="{esc(v)}"{" disabled" if disabled else ""}>'
            f'<output for="cj-cs-{name}" data-cj-out>{esc(v)}</output><span class="cj-mut">{esc(unit)}</span>'
            + (f'<small class="cj-mut">{esc(hint)}</small>' if hint else "") + '</div>')


def adjust_panel(d: Path, doc: dict, eff: dict, rec: dict, why_not: str | None) -> str:
    """The "Adjust settings" panel of a finished conversion. Disabled, with the reason, when the raw input is gone.
    Sliders start at eff (what the conversion used)."""
    base = f"/predict/convert/{esc(d.name)}"
    if why_not:
        return (f'<div class="cj-card cj-convset-adjust" id="cj-convset-adjust" data-disabled="1"><b>Adjust settings</b>'
                f'<p class="cj-convset-why">{esc(why_not)}</p>'
                f'<p><button type="button" class="cj-predict-btn" disabled>Preview</button> '
                f'<button type="button" class="cj-predict-btn" disabled>Re-convert with these settings</button></p></div>')
    dn = doc.get("denoise") or {}
    is_bp = dn.get("method") == "bandpass"
    on = eff.get("denoise", True) is not False
    bp = eff.get("bandpass_nm")
    if bp is None:
        bp = rec.get("bandpass_nm")
    mx = dn.get("max_nm")
    hi = 10.0 if not isinstance(mx, (int, float)) or isinstance(mx, bool) else min(10.0, round(float(mx) - 0.1, 4))
    opts = "".join(f'<option value="{esc(n)}">{esc(n)}</option>' for n in tile_names(d))
    return (
        f'<div class="cj-card cj-convset-adjust" id="cj-convset-adjust" data-preview="{base}/preview"><b>Adjust settings'
        f'</b><p class="cj-mut">Try other contrast and denoise settings on one tile. A preview takes a few seconds; '
        f'nothing changes until you re-convert, which makes a new conversion and keeps this one.</p>'
        f'<form method="post" action="{base}/reconvert" class="cj-convset-form" data-cj-convset-form>'
        f'<div class="cj-convset-row"><label for="cj-cs-tile">Tile</label><select id="cj-cs-tile" name="tile" '
        f'data-cj-tile>{opts}</select></div>'
        + _slider("clip_lo_pct", "Clip low %", 0, 5, 0.1, eff.get("clip_lo_pct", rec.get("clip_lo_pct", 1.0)), "%")
        + _slider("clip_hi_pct", "Clip high %", 95, 100, 0.1, eff.get("clip_hi_pct", rec.get("clip_hi_pct", 99.0)), "%")
        + (_slider("bandpass_nm", "Bandpass nm", 0.5, hi, 0.1, bp, "nm", disabled=not on,
                   hint="Structures smaller than this are smoothed away.") if is_bp and bp is not None else
           '<p class="cj-mut">This recipe does not denoise with a bandpass filter: there is no cutoff to adjust.</p>')
        + f'<div class="cj-convset-row"><label for="cj-cs-denoise">Denoise</label>'
          f'<label class="cj-convset-switch"><input type="checkbox" id="cj-cs-denoise" name="denoise" value="true" '
          f'data-cj-denoise data-recipe="{1 if on else 0}"{" checked" if on else ""}>'
          f'<span data-cj-denoise-text>{"On" if on else "Off"}</span></label></div>'
        f'<p class="cj-convset-acts"><button type="button" class="cj-predict-btn" data-cj-preview>Preview</button> '
        f'<button type="button" class="cj-predict-btn" data-cj-reset>Reset to this conversion’s values</button> '
        f'<button type="submit" class="cj-predict-btn cj-predict-primary" data-cj-reconvert>'
        f'Re-convert with these settings</button></p></form>'
        f'<p class="cj-convset-msg" data-cj-convset-msg role="status" hidden></p>'
        f'<div class="cj-convset-pv" data-cj-pv hidden></div></div>')


def convset_fragment(cfg, user, d: Path, meta: dict, st: dict, doc, err, rec=None) -> str:
    """The Settings section of a conversion page: the values used (and the diff when custom) + the adjust panel.
    rec: the bare recipe's values (the diff is against these, never against doc["params"])."""
    admin = user.get("role") == "admin"
    if doc is None:
        return (f'<h2 class="cj-sec">Settings</h2><div class="cj-card"><p class="cj-mut">The settings could not be '
                f'read.</p>{tech(err) if admin and err else ""}</div>')
    rec = rec or doc.get("params") or {}
    eff = conv_values(doc, meta)
    custom = bool(doc.get("params_overrides")) or bool(meta.get("params"))
    changed = CS.diff(rec, eff) if custom else []
    repo, _ = processing_repo(cfg)
    hide = [d, *([repo] if repo else []), CS.HOME]
    parts = ['<div class="cj-predict-top"><b>Settings used</b>'
             + (chip("custom settings", "warn") if custom else chip(f"{meta.get('recipe')} recipe", "ok")) + '</div>']
    if meta.get("derived_from"):
        by = f" by {esc(meta['made_by'])}" if meta.get("made_by") else ""
        parts.append(f'<p class="cj-mut">Re-converted from <a href="/predict/convert/{esc(meta["derived_from"])}/">'
                     f'{esc(meta["derived_from"])}</a>{by} with other settings.</p>')
    if not doc.get("recorded") and not custom:
        parts.append('<p class="cj-mut">This conversion did not record its settings; these are the recipe’s values '
                     'now.</p>')
    parts.append(settings_rows(doc, eff, rec))
    if changed:
        def show(v):
            return "on" if v is True else "off" if v is False else ("–" if v is None else _fnum(v))
        parts.append('<table class="cj-convset-diff"><caption>Differences from the recipe</caption><thead><tr><th>Setting'
                     '</th><th>Recipe</th><th>Used</th></tr></thead><tbody>'
                     + "".join(f'<tr><td>{esc(CS.LABELS.get(k, k))}</td><td>{esc(show(a))}</td>'
                               f'<td><b>{esc(show(b))}</b></td></tr>' for k, a, b in changed) + '</tbody></table>')
    elif custom:
        parts.append('<p class="cj-mut">These are the recipe’s own values.</p>')
    for w in (doc.get("warnings") or [])[:5]:
        parts.append(f'<p class="cj-mut">{esc(_plain(str(w), hide))}</p>')
    out = '<h2 class="cj-sec">Settings</h2><div class="cj-card cj-convset-used">' + "".join(parts) + '</div>'
    if st.get("state") == "done" and user.get("role") in RAW_ROLES:
        out += adjust_panel(d, doc, eff, rec, raw_kept_reason(d, meta, st))
    return out


def convset_section(d: Path, st: dict) -> str:
    """The section in the page: a placeholder the script fills in (the child process never runs at page load)."""
    return (f'<section id="cj-convset" data-url="/predict/convert/{esc(d.name)}/settings" '
            f'data-state="{esc(st.get("state"))}"><p class="cj-mut" data-cj-convset-wait>Loading the settings…</p>'
            f'</section>')


def page_convert(cfg, user, can_see, d: Path, meta: dict, st: dict, details=None) -> str:
    admin = user.get("role") == "admin"
    fr = conv_fragments(cfg, user, can_see, d, meta, st, details)
    src = meta.get("source") or {}
    what = ("Uploaded files" if src.get("kind") == "upload" else
            f"Raw session folder {src.get('folder') or '?'}")
    default = (msg_box(f"Converted with the default recipe ({CR.DEFAULT_RECIPE}, the lab’s template), not this "
                       f"session’s own: contrast and denoise may differ from the lab’s delivered tiles.")
               if meta.get("recipe") == "default" else "")
    owner = f'<div><dt>Owner</dt><dd>{esc(meta.get("owner"))}</dd></div>' if admin else ""
    return wrap(
        f'<div class="cj-predict" id="cj-predict-conv" data-cid="{esc(d.name)}" data-live="{int(fr["live"])}" '
        f'data-every="{COMMON.REFRESH_S}"><p class="cj-mut"><a href="/predict/">← My analyses</a></p>'
        f'<h1>Raw conversion: {esc(PSRC.default_label(meta.get("session") or "19700101"))}</h1>'
        + (f'<p class="cj-mut cj-wrap">{esc(d.name)}</p>' if user.get("role") != "basic" else "") +
        f'<p id="cj-predict-err" class="cj-mut" role="status" hidden></p>{default}'
        f'<section id="cj-predict-cstatus">{fr["status"]}</section>'
        f'<section id="cj-predict-cresult">{fr["result"]}</section>'
        f'{convset_section(d, st)}'
        f'<h2 class="cj-sec">Details</h2><div class="cj-card"><dl class="cj-predict-kv">'
        f'<div><dt>Session</dt><dd>{esc(meta.get("session"))}</dd></div>'
        f'<div><dt>Recipe</dt><dd>{esc(meta.get("recipe"))}</dd></div>'
        f'<div><dt>Compute</dt><dd>{"GPU (asked)" if meta.get("gpu") else "CPU"}</dd></div>'
        f'<div><dt>Images from</dt><dd>{esc(what)}</dd></div>'
        f'<div><dt>Started</dt><dd>{esc(when(meta.get("created")))}</dd></div>{owner}</dl>'
        + (tech(f"{d} · inputs {json.dumps(meta.get('inputs'))} · sample key {meta.get('sample_key')}")
           if admin else "") + '</div></div>')


def not_set_up(cfg, user) -> str:
    _, why = processing_repo(cfg)
    return msg_box("Raw conversion is not set up on this server.",
                   extra=tech(why) if user.get("role") == "admin" else "<p>Ask the admin.</p>")


def page_raw_upload(cfg, user, error=None, form=None) -> str:
    """form: the fields to show again after a refusal (the no-JavaScript post)."""
    repo, _ = processing_repo(cfg)
    form = form or {}
    head = '<div class="cj-predict"><p class="cj-mut"><a href="/predict/new/">← Back</a></p><h1>Convert raw images</h1>'
    if repo is None:
        return wrap(head + not_set_up(cfg, user) + '</div>')
    err = msg_box("", extra=f'<div>{problem_list(error)}</div>') if error else ""
    return wrap(
        head + err +
        '<div class="cj-card cj-predict-what"><h2>What this does</h2><ul>'
        '<li>Your raw montages are made into model-ready tiles by the lab’s own processing (the same code that made '
        'the delivered “3d” tile folders), then you analyse them as a session.</li>'
        '<li>Upload whole montages: each <b>.mrc</b> with its <b>.mrc.mdoc</b>, or an <b>.idoc</b> with all its '
        '<b>.tif</b> tiles. Name the sample on each grid below, or add the session’s <b>sample_key.csv</b> (else '
        'samples are “Grid n”).</li>'
        + ('<li>When it is done, the upload is listed under Data → Sessions and you can add the session details '
           '(grid ids, prep notes).</li>' if US.enabled(cfg) else '') +
        '<li>The pixel size comes from the .mdoc / .idoc; it is never guessed.</li></ul></div>'
        f'<form method="post" action="/predict/new/raw/" enctype="multipart/form-data" class="cj-predict-form" '
        f'id="cj-predict-raw" data-cj-upload="raw" data-max-files="{RAW_MAX_FILES}" '
        f'data-max-bytes="{RAW_MAX_FILE_BYTES}">'
        f'<div class="cj-predict-msg cj-predict-t-warn" data-cj-upload-msg role="alert" hidden></div>'
        f'<noscript><p class="cj-predict-msg">Raw files are sent one at a time, which needs JavaScript. Turn it on to '
        f'upload here.</p></noscript>'
        f'<fieldset class="cj-card cj-predict-fs"><legend>Session and recipe</legend>'
        f'<label class="cj-predict-field"><span>Session id (the imaging date, YYYYMMDD; YYYYMMDD-n for a later '
        f'session that day)</span><input name="session" required maxlength="12" '
        f'pattern="[0-9]{{8}}(-[0-9]+)?" inputmode="numeric" placeholder="20261027" '
        f'value="{esc(str(form.get("session") or "")[:12], quote=True)}"></label>'
        f'{recipe_select(repo, form.get("recipe"))}{gpu_box(form.get("gpu") == "yes")}</fieldset>'
        f'{US.samples_fieldset(form)}'
        f'<fieldset class="cj-card cj-predict-fs"><legend>Raw files</legend>'
        f'<label class="cj-predict-field"><span>.mrc + .mrc.mdoc, or .idoc + .tif tiles (+ sample_key.csv)</span>'
        f'<input type="file" name="files" multiple required accept=".mrc,.mdoc,.idoc,.tif,.tiff,.csv"></label>'
        f'<p class="cj-mut" data-cj-upload-info role="status"></p></fieldset>'
        f'<p><button class="cj-predict-btn cj-predict-primary" type="submit" data-busy="Uploading… keep this page '
        f'open">Upload and convert</button></p></form></div>')


def page_raw_sessions(cfg, user) -> str:
    repo, _ = processing_repo(cfg)
    head = ('<div class="cj-predict"><p class="cj-mut"><a href="/predict/new/">← Back</a></p>'
            '<h1>Pick a raw session</h1>')
    if repo is None:
        return wrap(head + not_set_up(cfg, user) + '</div>')
    cards = [f'<a class="cj-card cj-predict-src" href="/predict/new/raw/sessions/{esc(s["key"])}/">'
             f'<b>{esc(s["label"])}</b><span class="cj-mut">Session {esc(s["session"])} · folder '
             f'{esc(s["folder"])}</span>{tech(str(s["dir"]))}</a>' for s in raw_sessions(cfg)]
    body = "".join(cards) or ('<p class="cj-mut">Every session folder under the scan roots already has processed '
                              'tiles (or none was found).</p>')
    return wrap(head + '<p class="cj-mut">Session folders under the scan roots that have no processed tiles yet. The '
                       'folder is only read; the tiles are written under the Predict folder.</p>'
                f'<div class="cj-predict-srcs">{body}</div></div>')


def human_bytes(n) -> str:
    if not n:
        return ""
    return f"{n / 2 ** 30:.1f} GB" if n >= 2 ** 30 else f"{n / 2 ** 20:.0f} MB"


def page_raw_session(cfg, user, s: dict, listing: dict, form=None, error=None) -> str:
    repo, _ = processing_repo(cfg)
    head = (f'<div class="cj-predict"><p class="cj-mut"><a href="/predict/new/raw/sessions/">← Back</a></p>'
            f'<h1>{esc(s["label"])}</h1><p class="cj-mut">Session {esc(s["session"])} · folder {esc(s["folder"])}'
            f'</p>{tech(str(s["dir"]))}')
    if repo is None:
        return wrap(head + not_set_up(cfg, user) + '</div>')
    form = form or {}
    picked = set(form.get("montages") or [m["rel"] for m in listing["montages"] if not m["atlas"]])
    def row(m):
        size = f' · {esc(human_bytes(m["bytes"]))}' if m["bytes"] else ""
        atlas = ' <span class="cj-predict-warn">atlas: usually not wanted</span>' if m["atlas"] else ""
        return (f'<li><label><input type="checkbox" name="montages" value="{esc(m["rel"])}"'
                f'{" checked" if m["rel"] in picked else ""}> <span class="cj-wrap">{esc(m["rel"])}</span></label> '
                f'<span class="cj-predict-tag">{esc(m["kind"])}{size}</span>{atlas}</li>')
    rows = "".join(row(m) for m in listing["montages"])
    err = msg_box("", extra=f'<div>{problem_list(error)}</div>') if error else ""
    if not rows:
        return wrap(head + err + '<p class="cj-mut">No montages (.mrc with its .mrc.mdoc, or .idoc) found in this '
                                 'folder.</p></div>')
    key = (f'Sample key: {esc(listing["sample_key"].relative_to(s["dir"]).as_posix())} (sample names come from it).'
           if listing["sample_key"] else 'No sample_key.csv found: samples will be named “Grid n”.')
    cut = '<p class="cj-predict-warn">The folder is very big: only part of it was looked at.</p>' \
        if listing["truncated"] else ""
    return wrap(
        head + err +
        f'<form method="post" class="cj-predict-form" id="cj-predict-rawsession">'
        f'<fieldset class="cj-card cj-predict-fs"><legend>Montages ({len(listing["montages"])})</legend>{cut}'
        f'<ul class="cj-predict-mlist">{rows}</ul><p class="cj-mut">{key}</p></fieldset>'
        f'<fieldset class="cj-card cj-predict-fs"><legend>Recipe</legend>'
        f'{recipe_select(repo, form.get("recipe"), s["session"])}{gpu_box(form.get("gpu") == "yes")}</fieldset>'
        f'<p><button class="cj-predict-btn cj-predict-primary" type="submit" data-busy="Starting…">Convert</button>'
        f'</p></form></div>')
