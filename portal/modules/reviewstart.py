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
    GET  /reviews/start/<name>    the progress page (admin; any other id or person: 404): the allowlisted phases
                                  from <review root>/.building/<name>/status.json, reloading itself every few
                                  seconds (meta refresh) until done or failed; Continue opens /reviews/go/<name>

Non-admins get 403 on (b)/(c) (the POST) and 404 on progress pages; the page itself shows them (a) only.
The nested PREFIX /reviews/start (under the reviews module's /reviews) is portal/modules/__init__.py's rule: its
routes come before /reviews' and its own ROLES decide under it. Every text goes through html.escape.
"""
from __future__ import annotations

import hashlib
import hmac
import html
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
REFRESH_S = 4
esc = html.escape

CSS = """<style>
.cj-rs{max-width:960px;display:flex;flex-direction:column;gap:18px}
.cj-rs-back{font-size:.85rem;text-decoration:none}
.cj-rs-kinds{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,240px),1fr));gap:10px}
.cj-rs-kind{display:flex;flex-direction:column;gap:3px;padding:12px 14px;border:1px solid var(--line);border-radius:var(--cj-r);
  background:var(--card);color:var(--fg);text-decoration:none}
.cj-rs-kind:hover{border-color:var(--acc)}
.cj-rs-kind b{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.cj-rs-chip{display:inline-flex;padding:0 7px;border-radius:4px;font-size:.72rem;font-weight:600;border:1px solid var(--line)}
.cj-rs-chip-pred{color:var(--acc);background:var(--accbg)}
.cj-rs-chip-lab{color:var(--prod);background:var(--prodbg)}
.cj-rs-chip-aud{color:var(--warn);background:var(--warnbg)}
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
.cj-rs-primary{background:var(--acc);color:#fff}
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
    """Portal ids of the people who can sign in (key or access mode), sorted."""
    from portal import auth as A
    try:
        people = roster.people()
    except Exception:
        return []
    return sorted(pid for pid, p in people.items() if A.Roster.usable(p) or A.Roster.usable_access(p))


# ---------------------------------------------------------------- HTML
def chip(kind) -> str:
    text, cls = {"pred": ("Prediction", "pred"), "lab": ("Label set", "lab"), "aud": ("Audit", "aud")}[kind]
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
                   "until it ends.")]
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


def parse_b(text) -> dict | None:
    kind, _, ident = str(text or "").partition(":")
    if kind == "model" and RB.MODEL_RE.match(ident):
        return {"kind": "model", "id": ident}
    if kind == "lv" and RB.LV_RE.match(ident):
        return {"kind": "label_version", "id": ident}
    return None


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
        ids = set(roster_ids(roster)) if roster else set()
        try:
            name = await run_in_threadpool(RB.start, ctx.cfg, user.get("id"), purpose, lv, b, reviewers,
                                           roster_ids=ids)
        except RB.BuildError as e:
            busy = "one at a time" in str(e) or "starting" in str(e)
            return refuse(request, f"Not started: {e}.", 409 if busy else 400)
        PA.event(ctx.cfg.state_dir, request, "review build started", name,
                 {"purpose": purpose, "lv": lv, "b": f'{b["kind"]}:{b["id"]}', "reviewers": len(reviewers)})
        return RedirectResponse(f"{PREFIX}/{quote(name)}", status_code=303)

    def progress(request: Request):
        name = request.path_params["name"]
        st = RB.read_status(RB.review_root(ctx.cfg), name) if is_admin(request) else None
        if st is None:
            return show(request, "Start a review", f'<div class="cj-rs">{CSS}<p class="cj-card">No such build.</p>'
                        f'<p><a href="{PREFIX}/">Back</a></p></div>', 404)
        return show(request, "Building review", render_progress(st))

    return [Route(PREFIX + "/", start_page, methods=["GET"]),
            Route(PREFIX + "/build", build, methods=["POST"]),
            Route(PREFIX + "/{name}", progress, methods=["GET"])]


def start_url(**q) -> str:
    """/reviews/start/ with a query (e.g. lv=...), for links from other pages."""
    return f"{PREFIX}/" + (f"?{urlencode(q)}" if q else "")
