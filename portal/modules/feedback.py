"""
Feedback module (round 7, F): bug reports, feature ideas and questions.

    PREFIX = "/feedback"    ROLES = every role

    GET  /feedback/?page=/gallery/x      the form (page prefilled, path only) + "My reports" with status and the reply
    POST /feedback/                      create (urlencoded: kind, text, steps, page) -> 303 back with a thank-you
    GET  /feedback/all?kind=&status=     admin: every report, filtered
    POST /feedback/<id>/status           admin: set the status  (form token)
    POST /feedback/<id>/reply            admin: set the reply the reporter sees  (form token)

Storage and the 20-a-day limit: portal/feedback_store.py (file = $CJ_FEEDBACK_FILE else <state>/feedback.jsonl).
A basic or advanced person only ever gets their own reports in the HTML; another person's text is never rendered for
them. The POSTs need the Gate's Origin check; the admin actions also carry the admin form token (admin.py's scheme:
HMAC of the sign-in under a per-process secret). Each new report sends one ntfy push (training/notify.py) with a title
like "CryoEV: new bug report #12" and NO report text and NO user id (ntfy.sh is public), from a background thread;
a failure there never fails the request. Never posted anywhere else (the repo is public). All text is escaped.
"""
from __future__ import annotations

import hashlib
import hmac
import html
import secrets
import threading
from datetime import datetime, timezone
from urllib.parse import parse_qs, quote

from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import RedirectResponse
from starlette.routing import Route

from portal import auth as A
from portal import feedback_store as FS
from portal.modules import admin as AD

PREFIX = "/feedback"
ROLES = tuple(A.ROLES)
# a worst-case legitimate form: text + steps + page, every character a 4-byte UTF-8 character (an emoji) that urlencodes
# to 12 bytes (%XX x4; 9 for CJK, 6 for accented Latin), plus the other fields
MAX_FORM = 12 * (FS.TEXT_MAX + FS.STEPS_MAX + FS.PAGE_MAX) + 2048
KIND_WORD = {"bug": "Bug", "feature": "Feature idea", "question": "Question"}
KIND_TITLE = {"bug": "bug report", "feature": "feature idea", "question": "question"}
esc = html.escape


def _ntfy(title: str, message: str) -> None:
    """Send one push. Silent when CRYOEV_NTFY_TOPIC is unset (training/notify.py)."""
    try:
        from training.notify import _send_ntfy
        _send_ntfy(title, message)
    except Exception:                                    # noqa: BLE001  never let a push break feedback
        pass


def notify_new(rid: int, kind: str) -> threading.Thread:
    """ntfy in a background thread (the call can block 5 s). Title only says the kind and the number."""
    title = f"CryoEV: new {KIND_TITLE.get(kind, 'report')} #{rid}"
    def run():
        try:
            _ntfy(title, "Open Feedback in the portal.")
        except Exception:                                # noqa: BLE001
            pass
    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t


def fmt_time(t) -> str:
    return esc(str(t or "")[:16].replace("T", " ") + " UTC") if t else ""


def local_time(t) -> str:
    """UTC ISO -> this computer's local time, readable ("11 Nov 2026 14:03"). Unparseable -> the raw text, escaped."""
    try:
        d = datetime.fromisoformat(str(t))
        if d.tzinfo is None:
            d = d.replace(tzinfo=timezone.utc)
        return esc(d.astimezone().strftime("%d %b %Y %H:%M"))
    except (TypeError, ValueError):
        return esc(str(t or ""))


SNIP = 140


def routes(ctx) -> list:
    secret = secrets.token_bytes(32)                      # per process, like admin's: a restart makes open forms stale

    def path() -> str:
        return str(FS.feedback_path(ctx.cfg.state_dir))

    def token(request: Request) -> str:
        sess = AD.binding(request)
        return hmac.new(secret, hashlib.sha256(sess.encode()).digest(), hashlib.sha256).hexdigest()[:40]

    def is_admin(request: Request) -> bool:
        return ctx.user(request).get("role") == "admin"

    def me(request: Request) -> str:
        return str(ctx.user(request).get("id"))

    def show(request, title, content, status=200):
        r = ctx.page(request, title, f'<div class="cj-feedback">{content}</div>', active="feedback")
        r.status_code = status
        return r

    def note(text, tone="ok"):
        return f'<p class="cj-feedback-note cj-feedback-{tone}" role="{"alert" if tone == "warn" else "status"}">{esc(text)}</p>'

    async def form(request: Request, redo=None):
        """-> (fields, None) or (None, error response). urlencoded only. The body is read from the stream with a
        running cap (a chunked body has no Content-Length). Over the cap: 413, and `redo(fields)` (when given) renders
        the form again with what could be read of the typed text."""
        ctype = (request.headers.get("content-type") or "").split(";")[0].strip().lower()
        if ctype != "application/x-www-form-urlencoded":
            return None, show(request, "Feedback", note("Send the form from the Feedback page.", "warn"), 415)
        try:
            n = int(request.headers.get("content-length") or 0)
        except ValueError:
            n = MAX_FORM + 1
        buf, over = bytearray(), n > MAX_FORM
        async for chunk in request.stream():          # a running cap: a chunked body has no Content-Length to trust
            buf += chunk
            if len(buf) > MAX_FORM:
                over = True
                del buf[MAX_FORM:]
                break
        try:
            q = parse_qs(bytes(buf).decode("utf-8", "ignore" if over else "strict"), keep_blank_values=True,
                         max_num_fields=12)
        except (UnicodeDecodeError, ValueError):
            return None, show(request, "Feedback", note("That form could not be read.", "warn"), 400)
        fields = {k: v[0] for k, v in q.items() if v}
        if over:
            msg = note("That is too long to send. Shorten it and try again.", "warn")
            if redo is not None:
                return None, await redo(fields, msg, 413)
            return None, show(request, "Feedback", msg, 413)
        return fields, None

    def hidden_token(request) -> str:
        return AD.hidden_token(token(request))

    def back(request, text, to=None):
        tok = AD.flash_put(AD.binding(request), text)
        return RedirectResponse(f"{to or PREFIX + '/'}{'&' if '?' in (to or '') else '?'}f={quote(tok)}", status_code=303)

    def flash(request) -> str:
        text = AD.flash_take(AD.binding(request), request.query_params.get("f"))
        return note(text) if text else ""

    # ------------------------------------------------------------ pages
    def report_card(r, request, admin_view=False):
        head = (f'<h3>#{r["id"]} {esc(KIND_WORD[r["kind"]])} <span class="cj-feedback-st" data-st="{esc(r["status"])}">'
                f'{esc(r["status"])}</span></h3>'
                f'<p class="cj-feedback-meta">{fmt_time(r["t"])}'
                + (f' &middot; {esc(r["who"])}' if admin_view else "")
                + (f' &middot; page <code>{esc(r["page"])}</code>' if r["page"] else "") + "</p>")
        body = f'<p class="cj-feedback-text">{esc(r["text"])}</p>'
        if r["steps"]:
            body += f'<p class="cj-feedback-meta">Steps to reproduce</p><p class="cj-feedback-text">{esc(r["steps"])}</p>'
        if r["reply"]:
            body += (f'<div class="cj-feedback-reply"><b>Reply</b> <span class="cj-feedback-meta">{fmt_time(r["reply_t"])}'
                     f'</span><p class="cj-feedback-text">{esc(r["reply"])}</p></div>')
        acts = ""
        if admin_view:
            opts = "".join(f'<option{" selected" if s == r["status"] else ""}>{esc(s)}</option>' for s in FS.STATUSES)
            acts = (f'<form class="cj-feedback-acts" method="post" action="{PREFIX}/{r["id"]}/status">'
                    f'{hidden_token(request)}<label>Status <select name="status">{opts}</select></label>'
                    f'<button type="submit">Set status</button></form>'
                    f'<form class="cj-feedback-acts" method="post" action="{PREFIX}/{r["id"]}/reply">'
                    f'{hidden_token(request)}<label>Reply the reporter sees<textarea name="reply" rows="2" '
                    f'maxlength="{FS.REPLY_MAX}">{esc(r["reply"])}</textarea></label>'
                    f'<button type="submit">Save reply</button></form>')
        return f'<article class="cj-feedback-card" id="r{r["id"]}">{head}{body}{acts}</article>'

    def report_table(rows, request) -> str:
        """Admin's All reports: one row per report. Everything escaped; the status / reply forms keep their token."""
        trs = []
        for r in rows:
            text = r["text"]
            snip = text if len(text) <= SNIP else text[:SNIP].rstrip() + "…"
            full = ""
            if len(text) > SNIP or r["steps"]:
                inner = f'<p class="cj-feedback-text">{esc(text)}</p>'
                if r["steps"]:
                    inner += f'<p class="cj-feedback-meta">Steps to reproduce</p><p class="cj-feedback-text">{esc(r["steps"])}</p>'
                full = f'<details><summary>{esc(snip)}</summary>{inner}</details>'
            what = full or esc(snip)
            opts = "".join(f'<option{" selected" if s == r["status"] else ""}>{esc(s)}</option>' for s in FS.STATUSES)
            st = (f'<span class="cj-feedback-st" data-st="{esc(r["status"])}">{esc(r["status"])}</span>'
                  f'<form class="cj-feedback-acts" method="post" action="{PREFIX}/{r["id"]}/status">'
                  f'{hidden_token(request)}<select name="status" aria-label="Status of #{r["id"]}">{opts}</select>'
                  f'<button type="submit">Set status</button></form>')
            rp = ((f'<p class="cj-feedback-text">{esc(r["reply"])}</p>') if r["reply"] else "") + (
                f'<form class="cj-feedback-acts" method="post" action="{PREFIX}/{r["id"]}/reply">'
                f'{hidden_token(request)}<textarea name="reply" rows="2" maxlength="{FS.REPLY_MAX}" '
                f'aria-label="Reply to #{r["id"]}">{esc(r["reply"])}</textarea>'
                f'<button type="submit">Save reply</button></form>')
            page_cell = f'<code>{esc(r["page"])}</code>' if r["page"] else ""
            trs.append(f'<tr id="r{r["id"]}"><td>{r["id"]}</td><td>{esc(r["who"])}</td><td>{local_time(r["t"])}</td>'
                       f'<td>{esc(KIND_WORD.get(r["kind"], r["kind"]))}</td><td>{page_cell}</td>'
                       f'<td class="cj-fbt-text">{what}</td><td>{st}</td><td>{rp}</td></tr>')
        head = "".join(f"<th scope=\"col\">{h}</th>" for h in
                       ("#", "From", "When", "Type", "Page", "What they wrote", "Status", "Reply"))
        return (f'<div class="cj-fbt-wrap" tabindex="0" role="region" aria-label="All reports">'
                f'<table class="cj-fbt"><thead><tr>{head}</tr></thead><tbody>{"".join(trs)}</tbody></table></div>')

    def sub(request, active):
        a = '<a href="/feedback/all"{}>All reports</a>'.format(' aria-current="page"' if active == "all" else "") \
            if is_admin(request) else ""
        return (f'<nav class="cj-feedback-sub"><a href="{PREFIX}/"{" aria-current=page" if active == "mine" else ""}>'
                f'Send feedback</a>{a}</nav>')

    async def index(request: Request):
        page = FS.clean_page(request.query_params.get("page", ""))
        kind = request.query_params.get("kind", "bug")
        kind = kind if kind in FS.KINDS else "bug"
        return await run_in_threadpool(render_index, request, page, kind, "", "", "")

    def render_index(request, page, kind, text, steps, msg, status=200):
        opts = "".join(f'<option value="{k}"{" selected" if k == kind else ""}>{esc(KIND_WORD[k])}</option>'
                       for k in FS.KINDS)
        allr = FS.read_all(path()).values()
        mine = sorted((r for r in allr if r["who"] == me(request)), key=lambda r: -r["id"])
        cards = "".join(report_card(r, request) for r in mine) or '<p class="cj-feedback-meta">Nothing sent yet.</p>'
        n_new = sum(1 for r in allr if r["status"] == "New") if is_admin(request) else 0
        admin_note = (f'<p class="cj-feedback-meta cj-feedback-admin-note">{n_new} new report{"" if n_new == 1 else "s"} '
                      f'from everyone. <a href="{PREFIX}/all">See All reports</a></p>') if n_new else ""
        f = (f'<form class="cj-feedback-form" method="post" action="{PREFIX}/">'
             f'<label>What is it?<select name="kind" id="cj-fb-kind">{opts}</select></label>'
             f'<label>What happened, or what you would like<textarea name="text" rows="6" maxlength="{FS.TEXT_MAX}" '
             f'required>{esc(text)}</textarea></label>'
             f'<label class="cj-feedback-steps">Steps to reproduce (for a bug)<textarea name="steps" rows="4" '
             f'maxlength="{FS.STEPS_MAX}">{esc(steps)}</textarea></label>'
             f'<label>Page<input type="text" name="page" value="{esc(page)}" maxlength="{FS.PAGE_MAX}"></label>'
             f'<button type="submit">Send</button></form>')
        content = (f'<h1>Feedback</h1>{sub(request, "mine")}{msg}{flash(request)}'
                   f'<p class="cj-feedback-meta">Bugs, ideas and questions go to the people who run the portal. '
                   f'Do not put passwords or keys in here.</p>{f}{admin_note}<h2>My reports</h2>{cards}')
        return show(request, "Feedback", content, status)

    async def create(request: Request):
        async def redo(f, msg, status):
            k = f.get("kind", "bug")
            return await run_in_threadpool(render_index, request, FS.clean_page(f.get("page", "")),
                                           k if k in FS.KINDS else "bug", f.get("text", ""), f.get("steps", ""), msg,
                                           status)
        f, err = await form(request, redo)
        if err:
            return err
        kind, text, steps, page = f.get("kind", ""), f.get("text", ""), f.get("steps", ""), f.get("page", "")
        try:
            rec = await run_in_threadpool(FS.create, path(), me(request), kind, text, steps if kind == "bug" else "",
                                          page)
        except FS.RateLimited as e:
            return await redo(f, note(str(e), "warn"), 429)
        except FS.Invalid as e:
            return await redo(f, note(str(e), "warn"), 400)
        except OSError:
            return await redo(f, note("That could not be saved just now. Try again in a minute.", "warn"), 503)
        notify_new(rec["id"], rec["kind"])
        return back(request, f"Thank you. Your report #{rec['id']} was sent.")

    async def all_reports(request: Request):
        if not is_admin(request):
            return show(request, "Feedback", note("Only admins see every report.", "warn"), 403)
        kind, st = request.query_params.get("kind", ""), request.query_params.get("status", "")
        allr = await run_in_threadpool(FS.read_all, path())
        rows = [r for r in allr.values()
                if (kind not in FS.KINDS or r["kind"] == kind) and (st not in FS.STATUSES or r["status"] == st)]
        rows.sort(key=lambda r: -r["id"])
        ko = '<option value="">All kinds</option>' + "".join(
            f'<option value="{k}"{" selected" if k == kind else ""}>{esc(KIND_WORD[k])}</option>' for k in FS.KINDS)
        so = '<option value="">Any status</option>' + "".join(
            f'<option{" selected" if s == st else ""}>{esc(s)}</option>' for s in FS.STATUSES)
        flt = (f'<form class="cj-feedback-filter" method="get" action="{PREFIX}/all"><select name="kind" '
               f'aria-label="Kind">{ko}</select><select name="status" aria-label="Status">{so}</select>'
               f'<button type="submit">Filter</button></form>')
        cards = report_table(rows, request) if rows else '<p class="cj-feedback-meta">No reports.</p>'
        n_new = sum(1 for r in allr.values() if r["status"] == "New")
        newnote = (f'<p class="cj-feedback-meta cj-feedback-admin-note">{n_new} new report{"" if n_new == 1 else "s"} '
                   f'from everyone.</p>') if n_new else ""
        return show(request, "All feedback",
                    f'<h1>All reports</h1>{sub(request, "all")}{flash(request)}{newnote}{flt}{cards}')

    async def admin_action(request: Request, what: str):
        if not is_admin(request):
            return show(request, "Feedback", note("Only admins can do that.", "warn"), 403)
        f, err = await form(request)
        if err:
            return err
        if not hmac.compare_digest(f.get("t", "").encode(), token(request).encode()):
            return show(request, "Feedback", note("This form has expired. Reload the page and try again.", "warn")
                        + f'<p><a href="{PREFIX}/all">Back to all reports</a></p>', 403)
        try:
            rid = int(request.path_params["rid"])
            if what == "status":
                await run_in_threadpool(FS.set_status, path(), rid, me(request), f.get("status", ""))
            else:
                await run_in_threadpool(FS.set_reply, path(), rid, me(request), f.get("reply", ""))
        except (FS.Invalid, ValueError) as e:
            return show(request, "Feedback", note(str(e), "warn") + f'<p><a href="{PREFIX}/all">Back</a></p>', 400)
        except OSError:
            return show(request, "Feedback", note("That could not be saved just now.", "warn"), 503)
        return back(request, f"Report #{rid} updated.", f"{PREFIX}/all")

    async def status(request: Request):
        return await admin_action(request, "status")

    async def reply(request: Request):
        return await admin_action(request, "reply")

    return [Route(PREFIX + "/", index, methods=["GET"]),
            Route(PREFIX + "/", create, methods=["POST"]),
            Route(PREFIX + "/all", all_reports, methods=["GET"]),
            Route(PREFIX + "/{rid:int}/status", status, methods=["POST"]),
            Route(PREFIX + "/{rid:int}/reply", reply, methods=["POST"])]
