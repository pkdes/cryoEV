"""
Start a review (CryoEV portal, round 13, B4): one page offering the three kinds of review, and the progress page of
a label-set build. The logic lives in portal/review_build.py; this module holds the routes and the HTML.

    GET  /reviews/start/          (a) your finished prediction sets without a review package, each with the
                                  EXISTING Predict button (POST /predict/<ps>/review; all roles, own sets only);
                                  (b) Revise a label set and (c) Blind audit forms (admin only): label set, compare
                                  with a model or another label set, the "Images found N of M" coverage line
                                  (the label set is picked first in its own GET form, "Check images": ?lv=<id>;
                                  no form token in any URL, no script),
                                  an audit's reviewers from the roster (the owner can't be one)
    POST /reviews/start/build     admin, form token (t) + the Gate's Origin check, never bearer: purpose, lv,
                                  b (model:<id> | lv:<id>), reviewer (repeatable) -> review_build.start() ->
                                  303 to the progress page; a refusal page (400/403/409) otherwise. One audit line
                                  "review build started" (purpose, label set, B, number of reviewers; never decoy
                                  numbers)
    POST /reviews/start/objects   (d) Object set (round 14, admin, form token + Origin, never bearer): multipart
                                  name, schema, items (CSV, at most 2 MB), ps, blind (repeatable; set by the
                                  page's small script from the CSV's header, all ticked; without it every non-key
                                  column is hidden), outline (target | target+contained), reviewer (repeatable)
                                  -> review_build.start_objects (checks, then a background thread runs
                                  annotation/object_set.build and register) -> 303 to its progress page
                                  /reviews/start/<obj job>. One audit line "object set built" (counts only)
    GET  /reviews/start/<name>    the progress page (admin; any other id or person: 404): the allowlisted phases
                                  from <review root>/.building/<name>/status.json, reloading itself every few
                                  seconds (meta refresh) until done or failed; Continue opens /reviews/go/<name>.
                                  An object set job (obj_<stamp>_<4hex>): its status under the portal state, at most
                                  3 problem lines of a refusal (paths scrubbed); done links the set

Non-admins get 403 on (b)/(c) (the POST) and 404 on progress pages; the page itself shows them (a) only.
The nested PREFIX /reviews/start (under the reviews module's /reviews) is portal/modules/__init__.py's rule: its
routes come before /reviews' and its own ROLES decide under it. Every text goes through html.escape.
"""
from __future__ import annotations

import hashlib
import hmac
import html
import json
import secrets
from urllib.parse import parse_qs, quote, urlencode

from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import RedirectResponse
from starlette.routing import Route

from portal import audit as PA
from portal import names
from portal import review_build as RB
from portal.modules import admin as AD

PREFIX = "/reviews/start"
ROLES = ("admin", "advanced", "basic")
MAX_FORM = 4096
OBJ_FORM_EXTRA = 64 * 1024              # the object set form's other fields and multipart framing, beyond the CSV
REFRESH_S = 4
esc = html.escape

CSS = """<style>
.cj-rs{max-width:960px;display:flex;flex-direction:column;gap:18px}
.cj-rs-back{font-size:.85rem;text-decoration:none}
.cj-rs-kinds{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,240px),1fr));gap:10px}
.cj-rs-kind,a.cj-rs-kind{display:flex;flex-direction:column;gap:3px;padding:12px 14px;border:1px solid var(--line);border-radius:var(--cj-r);
  background:var(--card);color:var(--fg);text-decoration:none}
.cj-rs-kind:hover{border-color:var(--acc)}
.cj-rs-kind b{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.cj-rs-chip{display:inline-flex;padding:0 7px;border-radius:4px;font-size:.72rem;font-weight:600;border:1px solid var(--line)}
.cj-rs-chip-pred{color:var(--acc);background:var(--accbg)}
.cj-rs-chip-lab{color:var(--prod);background:var(--prodbg)}
.cj-rs-chip-aud{color:var(--warn);background:var(--warnbg)}
.cj-rs-chip-obj{color:#7c3aed;background:color-mix(in srgb,#7c3aed 12%,transparent)}
.cj-rs-grid input[type=file]{font:inherit;font-size:.85rem}
.cj-rs-adm{font-family:var(--cj-mono);font-size:.7rem;color:var(--mut);border:1px solid var(--line);border-radius:4px;padding:0 5px}
.cj-rs-sec{background:var(--card);border:1px solid var(--line);border-radius:var(--cj-r);padding:16px;display:flex;
  flex-direction:column;gap:12px;scroll-margin-top:calc(var(--cj-bar-h) + 8px)}
.cj-rs-sec h2{font-size:1.1rem;margin:0}
.cj-rs-row{display:flex;flex-wrap:wrap;align-items:center;gap:10px 14px;padding:10px 0;border-top:1px solid var(--line)}
.cj-rs-row:first-of-type{border-top:0}
.cj-rs-name{flex:3 1 300px;min-width:0;display:flex;flex-direction:column;gap:2px}
.cj-rs-id{font-family:var(--cj-mono);font-size:.75rem;color:var(--mut);overflow-wrap:anywhere}
.cj-rs-via{padding:0 6px;border-radius:999px;font-size:.72rem;color:var(--prod);background:var(--prodbg)}
.cj-rs-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,260px),1fr));gap:12px}
.cj-rs-grid label,.cj-rs-field{display:flex;flex-direction:column;gap:5px;font-size:.85rem;font-weight:500}
.cj-rs select{padding:7px 8px;border:1px solid var(--line);border-radius:6px;background:var(--card);color:var(--fg);font:inherit;font-size:.85rem}
.cj-rs-fixed{padding:7px 10px;border:1px solid var(--line);border-radius:6px;font-family:var(--cj-mono);font-size:.85rem}
.cj-rs-cov{display:flex;gap:8px;align-items:flex-start;padding:9px 12px;border-radius:6px;font-size:.85rem;border:1px solid var(--line)}
.cj-rs-cov-ok{background:var(--prodbg);color:var(--prod)}
.cj-rs-cov-bad{background:var(--warnbg);color:var(--warn)}
.cj-rs-revs{display:flex;flex-wrap:wrap;gap:6px 14px;font-size:.85rem}
.cj-rs-revs label{display:inline-flex;gap:6px;align-items:center;font-weight:400}
.cj-rs-acts{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
.cj-rs-btn{padding:7px 14px;border-radius:6px;border:1px solid var(--acc);background:var(--card);color:var(--acc);font:inherit;
  font-size:.85rem;font-weight:600;cursor:pointer;text-decoration:none;display:inline-block}
.cj-rs-primary,a.cj-rs-btn.cj-rs-primary{background:var(--acc);color:#fff}  /* a.: beats the shell's .cj-page a */
.cj-rs-btn[disabled]{opacity:.5;cursor:not-allowed}
.cj-rs-note{padding:10px 12px;border-radius:6px;border:1px solid var(--line);background:var(--accbg);font-size:.85rem}
.cj-rs-err{background:var(--warnbg);color:var(--warn)}
.cj-rs-ph{list-style:none;margin:0;padding:0}
.cj-rs-ph li{display:flex;gap:10px;align-items:flex-start;padding:8px 0;border-bottom:1px solid var(--line)}
.cj-rs-dot{flex:0 0 22px;height:22px;border-radius:50%;display:inline-flex;align-items:center;justify-content:center;
  font-size:.75rem;font-weight:600;font-family:var(--cj-mono);border:1px solid var(--line);color:var(--mut)}
.cj-rs-ph [data-s="done"] .cj-rs-dot{background:var(--prod);border-color:var(--prod);color:#fff}
.cj-rs-ph [data-s="now"] .cj-rs-dot{background:var(--acc);border-color:var(--acc);color:#fff}
.cj-rs-ph [data-s="failed"] .cj-rs-dot{background:var(--warn);border-color:var(--warn);color:#fff}
.cj-rs-ph [data-s="skipped"] .cj-rs-dot{border-style:dashed}
.cj-rs-ph [data-s="now"] b{font-weight:700}
.cj-rs-ph [data-s="waiting"] b,.cj-rs-ph [data-s="skipped"] b{color:var(--mut);font-weight:500}
.cj-rs-state{margin-left:auto;font-size:.8rem;color:var(--mut)}
</style>"""


# ---------------------------------------------------------------- data for the page
def finished_without_package(cfg, user) -> list:
    """(a): the person's OWN prediction sets that are done and have no review package yet, newest first."""
    from portal.predict_parts import sets as SETS
    me = user.get("id")
    out = []
    for d, req, st in SETS.list_sets(cfg, user, lambda u, owner: bool(me) and owner == me):
        if st.get("state") != "done" or SETS.package_dir(cfg, d.name) is not None:
            continue
        t = SETS.task("review", d.name)
        n = st.get("n_total")
        out.append({"ps": d.name, "name": names.display_name(req), "via": req.get("owner_via") or None,
                    "images": n if isinstance(n, int) else None, "created": req.get("created"),
                    "preparing": bool(t and t.get("state") == "running")})
    return out


def roster_ids(roster) -> list:
    """Portal ids a reviewer may be, sorted: the hub's one rule (reviews.reviewer_ids: not revoked, a known role)."""
    from portal.modules import reviews as REV
    try:
        people = roster.people()
    except Exception:                                   # noqa: BLE001  unreadable roster: nobody
        return []
    return REV.reviewer_ids(people)


# ---------------------------------------------------------------- HTML
def chip(kind) -> str:
    text, cls = {"pred": ("Prediction", "pred"), "lab": ("Label set", "lab"), "aud": ("Audit", "aud"),
                 "obj": ("Object set", "obj")}[kind]
    return f'<span class="cj-rs-chip cj-rs-chip-{cls}">{text}</span>'


def render_kinds(admin) -> str:
    adm = '<span class="cj-rs-adm">admin</span>'
    cards = [('#rs-pred', f"A model's predictions {chip('pred')}",
              "Check what a model found in a finished prediction set. Everyone, for their own sets.")]
    if admin:
        cards += [("#rs-revise", f"Revise a label set {chip('lab')}{adm}",
                   "Show a label set next to a model or another label set, fix what disagrees, then Apply it as a "
                   "new label set."),
                  ("#rs-audit", f"Blind audit {chip('aud')}{adm}",
                   "Reviewers decide cards without knowing which side made them. Decoys check attention. Counts only "
                   "until it ends."),
                  ("#rs-objects", f"Object set {chip('obj')}{adm}",
                   "A list of objects (a CSV) with a label schema's questions on each card. Measured columns can stay "
                   "hidden from reviewers until it is closed.")]
    return ('<nav class="cj-rs-kinds" aria-label="What do you want to review?">'
            + "".join(f'<a class="cj-rs-kind" href="{h}"><b>{t}</b><span class="cj-mut">{esc(d)}</span></a>'
                      for h, t, d in cards) + "</nav>")


def render_pred(rows) -> str:
    head = ('<section class="cj-rs-sec" id="rs-pred" aria-labelledby="rs-pred-h"><div><h2 id="rs-pred-h">Finished '
            'prediction sets without a review</h2><span class="cj-mut">Create review builds the package (no GPU). '
            'You add reviewers afterwards from the Reviews page.</span></div>')
    if not rows:
        return head + ('<p class="cj-mut">None of your finished prediction sets is waiting for a review. Run a model '
                       'in <a href="/predict/">Predict</a> first.</p></section>')
    body = []
    for r in rows:
        via = f' <span class="cj-rs-via">{esc(r["via"])} for you</span>' if r["via"] else ""
        imgs = f'{r["images"]} img' if r["images"] is not None else ""
        when = names.when(r["created"]) if r["created"] else ""
        act = ('<span class="cj-mut" role="status">Preparing the review…</span>' if r["preparing"] else
               f'<form method="post" action="/predict/{esc(r["ps"])}/review">'
               f'<button class="cj-rs-btn" type="submit">Create review</button></form>')
        body.append(f'<div class="cj-rs-row"><div class="cj-rs-name"><b>{esc(r["name"])}</b>'
                    f'<span class="cj-rs-id">{esc(r["ps"])}</span></div><span>{via}</span>'
                    f'<span class="cj-mut">{esc(imgs)}</span><span class="cj-mut">{esc(when)}</span>{act}</div>')
    return head + "".join(body) + "</section>"


def lv_options(sets, chosen, skip=None) -> str:
    out = []
    for s in sets:
        if s["id"] == skip:
            continue
        bits = [s["id"]]
        if s["n_annotations"] is not None:
            bits.append(f'{s["n_annotations"]} annotations')
        if s["pkg"]:
            bits.append(f'from {s["pkg"]}')
        label = " · ".join(bits) + (" (current)" if s["current"] else "")
        sel = " selected" if s["id"] == chosen else ""
        out.append(f'<option value="{esc(s["id"])}"{sel}>{esc(label)}</option>')
    return "".join(out)


def b_options(models, sets, chosen_b, lv) -> str:
    ms = "".join(f'<option value="model:{esc(m["id"])}"{" selected" if chosen_b == "model:" + m["id"] else ""}>'
                 f'{esc(m["id"])}{" (production)" if m["production"] else ""}</option>' for m in models)
    ls = "".join(f'<option value="lv:{esc(s["id"])}"{" selected" if chosen_b == "lv:" + s["id"] else ""}>'
                 f'{esc(s["id"])}</option>' for s in sets if s["id"] != lv)
    return (f'<optgroup label="Models">{ms}</optgroup>' if ms else "") + \
        (f'<optgroup label="Label sets">{ls}</optgroup>' if ls else "")


def render_cov(cov) -> str:
    ok = bool(cov.get("ok"))
    mark = "&#10003;" if ok else "!"
    return (f'<div class="cj-rs-cov {"cj-rs-cov-ok" if ok else "cj-rs-cov-bad"}" role="status">'
            f'<span aria-hidden="true">{mark}</span><span>{esc(RB.coverage_line(cov))}</span></div>')


def render_picker(sets, lv, cov) -> str:
    """Step 1 of (b)/(c): the label set, in its own GET form (no form token: nothing is written), and the
    "Images found N of M" line for it. Both build forms below take this label set."""
    return (f'<section class="cj-rs-sec" id="rs-lv" aria-labelledby="rs-lv-h"><div><h2 id="rs-lv-h">Label set to '
            f'review</h2><span class="cj-mut">For a revision or a blind audit (admin). Every image is looked up in '
            f'the label set\'s Roboflow export before anything is built.</span></div>'
            f'<form method="get" action="{PREFIX}/#rs-lv" class="cj-rs-acts">'
            f'<label class="cj-rs-field" style="flex:1 1 360px">Label set<select name="lv">{lv_options(sets, lv)}'
            f'</select></label><button class="cj-rs-btn" type="submit" style="align-self:flex-end">Check images'
            f'</button></form>{render_cov(cov)}</section>')


def render_form(purpose, tok, sets, models, lv, b, cov, roster, me, busy) -> str:
    rid = f"rs-{purpose}"
    title = "Revise a label set" if purpose == "revise" else "Blind audit"
    grid = [f'<div class="cj-rs-field"><span>Label set</span><span class="cj-rs-fixed">{esc(lv or "none")}</span>'
            f'<a class="cj-mut" href="#rs-lv">change above</a></div>',
            f'<label>{"Compare with (model or label set)" if purpose == "revise" else "Versus (model or label set)"}'
            f'<select name="b">{b_options(models, sets, b, lv)}</select></label>']
    if purpose == "audit":
        grid.append('<div class="cj-rs-field"><span>Decoys</span><span class="cj-rs-fixed">10% of cards</span></div>')
    revs = ""
    if purpose == "audit":
        boxes = "".join(f'<label><input type="checkbox" name="reviewer" value="{esc(p)}"> {esc(p)}</label>'
                        for p in roster if p != me)
        if me in roster:
            boxes += f'<label class="cj-mut"><input type="checkbox" disabled> {esc(me)} (owner, can\'t review)</label>'
        revs = (f'<fieldset class="cj-rs-field" style="border:0;margin:0;padding:0"><legend>Reviewers '
                f'<b style="color:var(--warn)">required</b></legend><div class="cj-rs-revs">{boxes or "No one else is on the roster."}'
                f'</div><span class="cj-mut">You own this audit, so you can\'t review it. Who decided what, and the '
                f'decoy counts, stay hidden until it ends.</span></fieldset>')
    note = ("No decoys. Reviewers are added afterwards. When done, Apply makes a new label set that supersedes this "
            "one." if purpose == "revise" else "")
    dis = " disabled" if busy or not cov.get("ok") else ""
    why = "" if not dis else (' <span class="cj-mut">' + ("Another review is being built." if busy else
                                                         "The label set's images aren't ready (see above).")
                              + "</span>")
    return (f'<section class="cj-rs-sec" id="{rid}" aria-labelledby="{rid}-h"><h2 id="{rid}-h">{title}</h2>'
            f'<form method="post" action="{PREFIX}/build">{AD.hidden_token(tok)}'
            f'<input type="hidden" name="purpose" value="{purpose}">'
            f'<input type="hidden" name="lv" value="{esc(lv or "")}">'
            f'<div class="cj-rs-grid">{"".join(grid)}</div>{revs}'
            + (f'<span class="cj-mut">{esc(note)}</span>' if note else "")
            + f'<div class="cj-rs-acts"><button class="cj-rs-btn cj-rs-primary" type="submit"{dis}>'
            f'{"Create review" if purpose == "revise" else "Create audit"}</button>{why}</div></form></section>')


# ---------------------------------------------------------------- (d) Object set (round 14, admin; contract C4)
OBJ_SCRIPT = """<script>
(() => {  // the items CSV's columns as "Hide from reviewers" boxes (all on); without this script every non-key column is hidden
  const f = document.getElementById("rs-obj-items"), box = document.getElementById("rs-obj-blind"),
        ui = document.getElementById("rs-obj-blind-ui"), KEYS = %s;
  if (!f || !box || !ui) return;
  f.addEventListener("change", async () => {
    box.textContent = ""; ui.value = "";
    const file = f.files && f.files[0]; if (!file) return;
    const head = (await file.slice(0, 65536).text()).replace(/^\\ufeff/, "").split(/\\r?\\n/)[0] || "";
    const cols = head.split(",").map(c => c.trim().replace(/^"|"$/g, "")).filter(Boolean);
    const blind = cols.filter(c => !KEYS.includes(c.toLowerCase()));
    if (!blind.length) { box.textContent = "No columns to hide besides the object's own."; ui.value = "1"; return; }
    for (const c of blind) {
      const l = document.createElement("label"), cb = document.createElement("input");
      cb.type = "checkbox"; cb.name = "blind"; cb.value = c; cb.checked = true;
      l.append(cb, " " + c); box.append(l);
    }
    ui.value = "1";
  });
})();
</script>"""


def render_objects_form(tok, schemas, sets, roster, me, error="") -> str:
    """(d) Object set: name, label schema, items CSV, the prediction set the objects come from, the columns to hide,
    the outlines on the cards, reviewers. One multipart POST to /reviews/start/objects (admin)."""
    def s_label(s):
        return f'{s["name"]}' + (f' v{s["version"]}' if s.get("version") else "") + f' · {s["id"]}'

    def p_label(p):
        return f'{p["name"]} · {p["id"]}' + (f' · {p["images"]} img' if p.get("images") is not None else "")
    s_opts = "".join(f'<option value="{esc(s["id"])}">{esc(s_label(s))}</option>' for s in schemas)
    p_opts = "".join(f'<option value="{esc(p["id"])}">{esc(p_label(p))}</option>' for p in sets)
    boxes = "".join(f'<label><input type="checkbox" name="reviewer" value="{esc(p)}"> {esc(p)}</label>' for p in roster)
    err = f'<p class="cj-rs-note cj-rs-err" role="alert">{esc(error)}</p>' if error else ""
    dis = "" if schemas and sets else " disabled"
    why = ("" if schemas and sets else '<span class="cj-mut">' + (
        "No label schema is registered yet (cryoev schema register)." if not schemas else
        "No finished prediction set to take the objects from.") + "</span>")
    return (f'<section class="cj-rs-sec" id="rs-objects" aria-labelledby="rs-objects-h"><div><h2 id="rs-objects-h">'
            f'Object set</h2><span class="cj-mut">One card per listed object: reviewers answer the label schema\'s '
            f'questions about it. Admin only.</span></div>{err}'
            f'<form method="post" action="{PREFIX}/objects" enctype="multipart/form-data">{AD.hidden_token(tok)}'
            '<input type="hidden" name="blind_ui" id="rs-obj-blind-ui" value="">'
            '<div class="cj-rs-grid">'
            '<label>Name<input name="name" maxlength="60" required placeholder="Lamellarity check 1" '
            'style="padding:7px 8px;border:1px solid var(--line);border-radius:6px;background:var(--card);'
            'color:var(--fg);font:inherit;font-size:.85rem"></label>'
            f'<label>Label schema<select name="schema" required>{s_opts}</select></label>'
            '<label>Items CSV<input type="file" name="items" id="rs-obj-items" accept=".csv,text/csv" required></label>'
            f'<label>Objects come from<select name="ps" required>{p_opts}</select></label></div>'
            '<fieldset class="cj-rs-field" style="border:0;margin:0;padding:0"><legend>Hide from reviewers (blind)'
            '</legend><div class="cj-rs-revs" id="rs-obj-blind"><span class="cj-mut">Pick the CSV: its columns '
            'show here, all ticked.</span></div><noscript><span class="cj-mut">Without JavaScript every column '
            'except the object\'s own (image, line, id) is hidden.</span></noscript></fieldset>'
            '<fieldset class="cj-rs-field" style="border:0;margin:0;padding:0"><legend>Outlines on cards</legend>'
            '<div class="cj-rs-revs"><label><input type="radio" name="outline" value="target" checked> Target EV only'
            '</label><label><input type="radio" name="outline" value="target+contained"> Target + contained '
            '<span class="cj-mut">(shows what is inside, so it hints at the counts)</span></label></div></fieldset>'
            f'<fieldset class="cj-rs-field" style="border:0;margin:0;padding:0"><legend>Reviewers</legend>'
            f'<div class="cj-rs-revs">{boxes or "No one is on the roster."}</div></fieldset>'
            f'<div class="cj-rs-acts"><button class="cj-rs-btn cj-rs-primary" type="submit"{dis}>Create object set'
            f'</button>{why}<span class="cj-mut">Order shuffled. Exports are for you and admins until the set is '
            f'closed.</span></div></form>{OBJ_SCRIPT % json.dumps(list(RB.OBJ_KEY_COLS))}</section>')


def render_start(user, pred_rows, admin_parts) -> str:
    admin = user.get("role") == "admin"
    out = ['<div class="cj-rs">', CSS, '<a class="cj-rs-back" href="/reviews/">&larr; Reviews</a>',
           '<h1>Start a review</h1>', render_kinds(admin)]
    if admin_parts and admin_parts.get("active"):
        a = admin_parts["active"]
        link = (f' <a href="{PREFIX}/{quote(a["name"])}">See its progress</a>' if a.get("name") else "")
        out.append(f'<p class="cj-rs-note" role="status">A review is being built now; one at a time.{link}</p>')
    out.append(render_pred(pred_rows))
    if admin and admin_parts:
        out += admin_parts["forms"]
    out.append("</div>")
    return "".join(out)


PHASE_ROWS = ("checking", "waiting", "predicting", "writing", "done")
PHASE_LINE = {"checking": "Every image of the label set is looked up in its export and checked.",
              "waiting": "Starts when the predict lock and the GPU are free.",
              "predicting": "Running the model on the images.",
              "writing": "Cards and pictures written.",
              "done": "Ready to review."}


def render_progress(st) -> str:
    phase, failed = st["phase"], st["phase"] == "failed"
    at = st.get("failed_at") if failed else phase
    model = st.get("b_kind") == "model"
    rows = []
    idx = PHASE_ROWS.index(at) if at in PHASE_ROWS else 0
    for i, ph in enumerate(PHASE_ROWS):
        skip = ph in ("waiting", "predicting") and not model
        if skip:
            s, mark, line = "skipped", "–", "Skipped: compared with a label set (no GPU)."
        elif failed and i == idx:
            s, mark, line = "failed", "!", RB.REASONS.get(st.get("reason") or "", "Failed")
        elif phase == "done" or i < idx:
            s, mark, line = "done", "&#10003;", PHASE_LINE[ph]
        elif i == idx:
            s, mark, line = "now", str(i + 1), PHASE_LINE[ph]
            if st.get("reason") in ("lock", "gpu_busy"):
                line = RB.REASONS[st["reason"]]
        else:
            s, mark, line = "waiting", str(i + 1), ""
        if ph == "checking" and st.get("detail") and s in ("done", "now", "failed"):
            line = st["detail"]
        rows.append(f'<li data-s="{s}"><span class="cj-rs-dot" aria-hidden="true">{mark}</span>'
                    f'<span style="display:flex;flex-direction:column;gap:1px"><b>{esc(RB.PHASE_TEXT[ph])}</b>'
                    f'<span class="cj-mut">{esc(line)}</span></span><span class="cj-rs-state">{s}</span></li>')
    kind = "Label-set revision" if st.get("purpose") == "revise" else "Blind audit"
    vs = f'{st.get("lv") or ""} vs {st.get("b_id") or ""}'
    acts = []
    if phase == "done":
        acts.append(f'<a class="cj-rs-btn cj-rs-primary" href="/reviews/go/{quote(st["name"])}">Continue &rarr;</a>')
        if st.get("reason") == "registry":
            acts.append(f'<span class="cj-mut">{esc(RB.REASONS["registry"])}.</span>')
    elif failed:
        acts.append(f'<a class="cj-rs-btn" href="{PREFIX}/">Back to Start a review</a>')
    else:
        acts.append(f'<span class="cj-mut">This page reloads itself every {REFRESH_S} s. One build at a time; the '
                    'status is kept on disk, so you can leave and come back.</span>')
    refresh = "" if phase in RB.TERMINAL else f'<meta http-equiv="refresh" content="{REFRESH_S}">'
    return (f'{refresh}<div class="cj-rs">{CSS}<a class="cj-rs-back" href="{PREFIX}/">&larr; Start a review</a>'
            f'<section class="cj-rs-sec" aria-labelledby="rs-prog-h" aria-live="polite">'
            f'<div><h2 id="rs-prog-h">Building review</h2><span class="cj-rs-id">{esc(st["name"])}</span><br>'
            f'<span class="cj-mut">{esc(kind)} · {esc(vs)}</span></div>'
            f'<ol class="cj-rs-ph">{"".join(rows)}</ol><div class="cj-rs-acts">{"".join(acts)}</div></section></div>')


OBJ_PHASE_LINE = {"checking": "The items CSV, the schema and the prediction set are checked.",
                  "building": "The cards, pictures and the shuffled item order are written.",
                  "registering": "The set is recorded so the Reviews hub lists it.",
                  "done": "Ready to review."}


def render_obj_progress(st) -> str:
    """The progress page of an object set build (review_build.read_obj_status)."""
    phase, failed = st["phase"], st["phase"] == "failed"
    at = st.get("failed_at") if failed else phase
    order = ("checking", "building", "registering", "done")
    idx = order.index(at) if at in order else 0
    rows = []
    for i, ph in enumerate(order):
        if failed and i == idx:
            s_, mark, line = "failed", "!", "Failed"
        elif phase == "done" or i < idx:
            s_, mark, line = "done", "&#10003;", OBJ_PHASE_LINE[ph]
        elif i == idx:
            s_, mark, line = "now", str(i + 1), OBJ_PHASE_LINE[ph]
        else:
            s_, mark, line = "waiting", str(i + 1), ""
        rows.append(f'<li data-s="{s_}"><span class="cj-rs-dot" aria-hidden="true">{mark}</span>'
                    f'<span style="display:flex;flex-direction:column;gap:1px"><b>{esc(RB.OBJ_PHASE_TEXT[ph])}</b>'
                    f'<span class="cj-mut">{esc(line)}</span></span><span class="cj-rs-state">{s_}</span></li>')
    acts, probs = [], ""
    if phase == "done" and st.get("pkg"):
        acts.append(f'<a class="cj-rs-btn cj-rs-primary" href="/reviews/go/{quote(st["pkg"])}">Open the set &rarr;</a>')
        acts.append('<a class="cj-rs-btn" href="/reviews/?kind=objects">Object sets</a>')
    elif failed:
        if st.get("problems"):
            probs = ('<ul class="cj-rs-note cj-rs-err" role="alert">'
                     + "".join(f"<li>{esc(x)}</li>" for x in st["problems"]) + "</ul>")
        acts.append(f'<a class="cj-rs-btn" href="{PREFIX}/">Back to Start a review</a>')
    else:
        acts.append(f'<span class="cj-mut">This page reloads itself every {REFRESH_S} s. One object set at a time; '
                    'you can leave and come back.</span>')
    refresh = "" if phase in ("done", "failed") else f'<meta http-equiv="refresh" content="{REFRESH_S}">'
    n = st.get("n_items")
    sub = f'{esc(str(st.get("title") or ""))}' + (f" · {n} items" if isinstance(n, int) else "")
    return (f'{refresh}<div class="cj-rs">{CSS}<a class="cj-rs-back" href="{PREFIX}/">&larr; Start a review</a>'
            f'<section class="cj-rs-sec" aria-labelledby="rs-prog-h" aria-live="polite">'
            f'<div><h2 id="rs-prog-h">Building object set</h2><span class="cj-rs-id">{esc(st["name"])}</span><br>'
            f'<span class="cj-mut">{sub}</span></div>'
            f'<ol class="cj-rs-ph">{"".join(rows)}</ol>{probs}<div class="cj-rs-acts">{"".join(acts)}</div>'
            f'</section></div>')


def parse_b(text) -> dict | None:
    kind, _, ident = str(text or "").partition(":")
    if kind == "model" and RB.MODEL_RE.match(ident):
        return {"kind": "model", "id": ident}
    if kind == "lv" and RB.LV_RE.match(ident):
        return {"kind": "label_version", "id": ident}
    return None


async def read_objects_form(form):
    """(fields, CSV bytes, None), or (None, None, (refusal text, status)). Text fields only as text; one file."""
    def one(k):
        v = form.get(k)
        return v if isinstance(v, str) else ""
    up = form.get("items")
    if up is None or isinstance(up, str):
        return None, None, ("Pick the items CSV.", 400)
    data = await up.read(RB.ITEMS_MAX_BYTES + 1)
    if len(data) > RB.ITEMS_MAX_BYTES:
        return None, None, ("The items CSV may be up to 2 MB.", 413)
    fields = {k: one(k) for k in ("t", "name", "schema", "ps", "outline", "blind_ui")}
    fields["reviewer"] = [v for v in form.getlist("reviewer") if isinstance(v, str)]
    fields["blind"] = [v for v in form.getlist("blind") if isinstance(v, str)]
    return fields, data, None


# ---------------------------------------------------------------- routes
def routes(ctx) -> list:
    from portal import auth as A
    secret = secrets.token_bytes(32)                     # per process: a restart makes open forms stale
    roster = A.Roster(ctx.cfg.roster) if getattr(ctx.cfg, "roster", None) else None

    def token(request: Request) -> str:
        return hmac.new(secret, hashlib.sha256(AD.binding(request).encode()).digest(), hashlib.sha256).hexdigest()[:40]

    def show(request, title, body, status=200):
        r = ctx.page(request, title, body, active="reviews")
        r.status_code = status
        return r

    def refuse(request, msg, status, back=True):
        link = f'<p><a href="{PREFIX}/">Back to Start a review</a></p>' if back else ""
        return show(request, "Start a review", f'<div class="cj-rs">{CSS}<p class="cj-rs-note cj-rs-err" '
                    f'role="alert">{esc(msg)}</p>{link}</div>', status)

    def is_admin(request) -> bool:
        return ctx.user(request).get("role") == "admin" and request.scope.get("cj_auth") != "bearer"

    def admin_parts(request) -> dict:
        user = ctx.user(request)
        me = user.get("id")
        sets, models = RB.label_sets(ctx.cfg), RB.models(ctx.cfg)
        q = request.query_params
        current = next((s["id"] for s in sets if s["current"]), sets[0]["id"] if sets else None)
        lv = q.get("lv") if q.get("lv") in {s["id"] for s in sets} else current
        prod = next((f'model:{m["id"]}' for m in models if m["production"]), None)
        b = q.get("b") if parse_b(q.get("b")) else prod
        cov = RB.coverage(ctx.cfg, lv) if lv else {"ok": False, "problem": "no label sets found here"}
        rr = RB.review_root(ctx.cfg)
        active = RB.active_build(rr) if rr else None
        tok = token(request)
        ids = roster_ids(roster) if roster else []
        forms = [render_picker(sets, lv, cov)] + [render_form(p, tok, sets, models, lv, b, cov, ids, me, bool(active))
                                                  for p in RB.PURPOSES]
        forms.append(render_objects_form(tok, RB.schema_choices(ctx.cfg), RB.ps_choices(ctx.cfg, me), ids, me))
        return {"forms": forms, "active": active}

    def start_page(request: Request):
        user = ctx.user(request)
        pred_rows = finished_without_package(ctx.cfg, user)
        parts = admin_parts(request) if is_admin(request) else None
        return show(request, "Start a review", render_start(user, pred_rows, parts))

    async def build(request: Request):
        if not is_admin(request):
            return refuse(request, "Only admins can build label-set reviews and audits.", 403)
        ctype = (request.headers.get("content-type") or "").split(";")[0].strip().lower()
        buf = bytearray()
        async for chunk in request.stream():
            buf += chunk
            if len(buf) > MAX_FORM:
                return refuse(request, "That form is too large.", 413)
        try:
            f = parse_qs(bytes(buf).decode("utf-8"), max_num_fields=4 + RB.MAX_REVIEWERS)
        except (UnicodeDecodeError, ValueError):
            f = None
        if ctype != "application/x-www-form-urlencoded" or f is None:
            return refuse(request, "Send the form from the Start a review page.", 400)
        one = lambda k: (f.get(k) or [""])[0]  # noqa: E731
        if not hmac.compare_digest(one("t").encode(), token(request).encode()):
            return refuse(request, "This form has expired. Reload the page and try again.", 403)
        purpose, lv, b = one("purpose"), one("lv"), parse_b(one("b"))
        reviewers = f.get("reviewer") or []
        user = ctx.user(request)

        def start():                                    # the roster read and the build start: off the event loop
            ids = set(roster_ids(roster)) if roster else set()
            return RB.start(ctx.cfg, user.get("id"), purpose, lv, b, reviewers, roster_ids=ids)

        try:
            name = await run_in_threadpool(start)
        except RB.BuildError as e:
            busy = "one at a time" in str(e) or "starting" in str(e)
            return refuse(request, f"Not started: {e}.", 409 if busy else 400)
        PA.event(ctx.cfg.state_dir, request, "review build started", name,
                 {"purpose": purpose, "lv": lv, "b": f'{b["kind"]}:{b["id"]}', "reviewers": len(reviewers)})
        return RedirectResponse(f"{PREFIX}/{quote(name)}", status_code=303)

    async def objects(request: Request):
        """POST /reviews/start/objects (admin, never bearer): the multipart form of (d), at most RB.ITEMS_MAX_BYTES of
        CSV. RB.start_objects checks it all and starts the build in a background thread (object_set.build, then
        register), then one audit line "object set built" (schema, prediction set, item / blind column / reviewer
        COUNTS, never a value) and 303 to the build's progress page. A refusal page (400/403/409/413) otherwise."""
        if not is_admin(request):
            return refuse(request, "Only admins can build object sets.", 403)
        ctype = (request.headers.get("content-type") or "").split(";")[0].strip().lower()
        if ctype != "multipart/form-data":
            return refuse(request, "Send the form from the Start a review page.", 400)
        try:
            size = int(request.headers.get("content-length") or "0")
        except ValueError:
            size = 0
        if size <= 0 or size > RB.ITEMS_MAX_BYTES + OBJ_FORM_EXTRA:
            return refuse(request, "That upload is too large (the items CSV may be up to 2 MB)." if size > 0
                          else "Send the form from the Start a review page.", 413 if size > 0 else 400)
        try:
            form = await request.form(max_files=1, max_fields=40 + RB.MAX_REVIEWERS + RB.ITEMS_MAX_COLS)
        except Exception:                               # noqa: BLE001  malformed multipart / too many parts
            return refuse(request, "That form could not be read.", 400)
        try:
            got = await read_objects_form(form)
        finally:
            await form.close()                          # the spooled upload: closed on every path
        fields, data, refusal = got
        if refusal:
            return refuse(request, *refusal)
        if not hmac.compare_digest(fields["t"].encode(), token(request).encode()):
            return refuse(request, "This form has expired. Reload the page and try again.", 403)
        reviewers, blind_ui, blind = fields["reviewer"], fields["blind_ui"] == "1", fields["blind"]
        user = ctx.user(request)
        name, schema, ps, outline = fields["name"], fields["schema"], fields["ps"], fields["outline"] or "target"

        def build():                                    # the roster read, the CSV check and the start: off the loop
            info = RB.parse_items_csv(data)
            cols = info["blindable"] if not blind_ui else blind   # no script ran: hide every non-key column
            ids = set(roster_ids(roster)) if roster else set()
            job = RB.start_objects(ctx.cfg, user.get("id"), name=name, schema=schema, ps=ps, items_csv=data,
                                   blind_cols=cols, outline_mode=outline, reviewers=reviewers, roster_ids=ids)
            return job, len(cols), info["rows"]

        try:
            job, n_blind, n_items = await run_in_threadpool(build)
        except RB.BuildError as e:
            busy = "one at a time" in str(e)
            return refuse(request, f"Not built: {e}.", 409 if busy else 400)
        PA.event(ctx.cfg.state_dir, request, "object set built", job,
                 {"schema": schema, "ps": ps, "items": n_items, "blind_cols": n_blind, "reviewers": len(reviewers),
                  "outline_mode": outline})
        return RedirectResponse(f"{PREFIX}/{quote(job)}", status_code=303)

    def progress(request: Request):
        name = request.path_params["name"]
        if RB.OBJ_JOB_RE.match(name):                   # an object set build (round 14)
            ost = RB.read_obj_status(ctx.cfg, name) if is_admin(request) else None
            if ost is None:
                return show(request, "Start a review", f'<div class="cj-rs">{CSS}<p class="cj-card">No such build.'
                            f'</p><p><a href="{PREFIX}/">Back</a></p></div>', 404)
            return show(request, "Building object set", render_obj_progress(ost))
        st = RB.read_status(RB.review_root(ctx.cfg), name) if is_admin(request) else None
        if st is None:
            return show(request, "Start a review", f'<div class="cj-rs">{CSS}<p class="cj-card">No such build.</p>'
                        f'<p><a href="{PREFIX}/">Back</a></p></div>', 404)
        return show(request, "Building review", render_progress(st))

    return [Route(PREFIX + "/", start_page, methods=["GET"]),
            Route(PREFIX + "/build", build, methods=["POST"]),
            Route(PREFIX + "/objects", objects, methods=["POST"]),
            Route(PREFIX + "/{name}", progress, methods=["GET"])]


def start_url(**q) -> str:
    """/reviews/start/ with a query (e.g. lv=...), for links from other pages."""
    return f"{PREFIX}/" + (f"?{urlencode(q)}" if q else "")
