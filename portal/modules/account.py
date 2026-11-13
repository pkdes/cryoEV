"""
My account (CryoEV portal, every role, both doors): who you are, and tokens for your own AI agent (round 10 #60;
write scope round 11 #65; docs/API.md).

    GET  /account/                  your id and role; your agent tokens (agent, access, created, expires, last used,
                                    state) with Revoke; the form for a new one (agent name, expiry, Read only or
                                    Read + write)
    POST /account/tokens/new        agent + days + scope -> a new token, shown ONCE on the answer page (no-store);
                                    never logged, stored as sha256 only (portal/auth.py create_token). Read: 1-90
                                    days; read + write: 1-30 days (auth.WRITE_MAX_DAYS)
    POST /account/tokens/revoke     id -> your own token revoked (it stops working on its next request)

Rules:
  - The store is the one shared file (auth.tokens_path(cfg): $CJ_AGENT_TOKENS or ~/.cryoev/agent_tokens.json); each
    change is one read-modify-write under auth.tokens_lock.
  - Every POST: the Gate checks Origin; here a form token bound to the sign-in (admin.binding) as Admin does. A bearer
    token never reaches these routes (the Gate's allowlist), so an agent can't make or revoke tokens.
  - Tokens work on the tailnet address only for now; the agent sees what you see, capped at Advanced (an admin's
    agent is not an admin). A read + write token can also upload images, start predictions, compute results and send
    feedback (portal/app.py BEARER_WRITE_ROUTES; 60 writes an hour per person).
  - Audit: "agent token made" / "agent token revoked" (target = the token id; detail = for, agent, scope, expires),
    through portal.audit.event. Never the token or its hash.
"""
from __future__ import annotations

import hashlib
import hmac
import html
import logging
import secrets
from datetime import datetime
from urllib.parse import parse_qs, quote

from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import RedirectResponse
from starlette.routing import Route

from portal import audit as PA
from portal import auth as A
from portal.modules import admin as ADM

PREFIX = "/account"
ROLES = tuple(A.ROLES)
MAX_FORM = 4 * 1024
DAY_CHOICES = (7, 30, 90)
SCOPE_WORD = {"read": "read only", "write": "read+write"}
WRITE_LIMIT_TEXT = f"{A.WRITE_LIMIT} writes an hour"
esc = html.escape
log = logging.getLogger("cj.account")
CSS = """<style>
/* My account (portal/modules/account.py). cj.css tokens only; every class is cj-acct-*. Phone first. */
.cj-acct{max-width:860px;min-width:0}
.cj-acct-form{display:flex;flex-direction:column;gap:6px}
.cj-acct-lbl{font-size:.8rem;font-weight:600;color:var(--mut)}
.cj-acct-in{font:inherit;color:var(--fg);background:var(--bg);border:1px solid var(--line);border-radius:8px;
  min-height:44px;padding:8px 10px;width:100%;min-width:0}
select.cj-acct-in{width:auto;min-width:9em}
.cj-acct-btn{font:inherit;font-weight:600;min-height:44px;padding:8px 14px;border-radius:8px;cursor:pointer;
  border:1px solid var(--line);background:var(--card);color:var(--fg);align-self:flex-start}
.cj-acct-btn:focus-visible,.cj-acct-in:focus-visible{outline:2px solid var(--acc);outline-offset:2px}
.cj-acct-primary{background:var(--acc);border-color:var(--acc);color:var(--card)}
.cj-acct-danger{color:var(--warn);border-color:var(--warn)}
.cj-acct-list{list-style:none;margin:0;padding:0;display:flex;flex-direction:column;gap:8px}
.cj-acct-tok{border:1px solid var(--line);border-radius:8px;padding:8px 10px;display:flex;flex-wrap:wrap;gap:6px 12px;
  align-items:center;justify-content:space-between;min-width:0}
.cj-acct-tok form{margin:0}
.cj-acct-secret{font-family:var(--cj-mono);font-size:.85rem;word-break:break-all;resize:vertical}
.cj-acct-note{border-left:3px solid var(--acc);padding:6px 10px;margin:10px 0}
.cj-acct-scope{border:0;margin:4px 0;padding:0;display:flex;flex-direction:column;gap:6px;min-width:0}
.cj-acct-scope label{display:flex;gap:8px;align-items:flex-start;min-height:44px;cursor:pointer}
.cj-acct-scope input{margin-top:3px;width:18px;height:18px;flex:none}
.cj-acct-warn{border-left-color:var(--warn)}
.cj-acct pre{white-space:pre-wrap;word-break:break-all;font-size:.82rem}
.cj-acct section{margin-bottom:16px}
</style>"""


def note(text, warn=False) -> str:
    return f'<p class="cj-acct-note{" cj-acct-warn" if warn else ""}" role="status">{esc(text)}</p>'


def _when(iso) -> str:
    t = A._ts(iso)
    return datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M") if t else "never"


def render_tokens(rows, tok) -> str:
    """Your tokens: live ones with Revoke, then the rest (expired / revoked) without."""
    if not rows:
        return '<p class="cj-mut">No agent tokens yet.</p>'
    items = []
    for r in rows:
        state = r["state"]
        meta = (f'made {esc(_when(r["created"]))} · expires {esc(_when(r["expires"]))} · last used '
                f'{esc(_when(r["last_used"]))}')
        if state == "active":
            act = (f'<form method="post" action="{PREFIX}/tokens/revoke"><input type="hidden" name="t" '
                   f'value="{esc(tok)}"><input type="hidden" name="id" value="{esc(r["id"])}">'
                   f'<button class="cj-acct-btn cj-acct-danger" type="submit">Revoke</button></form>')
        else:
            act = f'<span class="cj-mut">{esc(state)}</span>'
        scope = SCOPE_WORD.get(r["scope"], str(r["scope"]))
        items.append(f'<li class="cj-acct-tok" data-token-id="{esc(r["id"])}" data-state="{esc(state)}" '
                     f'data-scope="{esc(str(r["scope"]))}"><span>'
                     f'<b>{esc(r["agent"])}</b> <span class="cj-mut">{esc(r["id"])} · {esc(scope)} · '
                     f'{meta}</span></span>{act}</li>')
    return f'<ul class="cj-acct-list">{"".join(items)}</ul>'


def render_page(user, rows, tok, msg_html="", door_note="") -> str:
    live = sum(1 for r in rows if r["state"] == "active")
    days = "".join(f'<option value="{d}"{" selected" if d == A.TOKEN_DEFAULT_DAYS else ""}>{d} days</option>'
                   for d in DAY_CHOICES)
    full = live >= A.TOKENS_PER_PERSON
    form = (f'<form class="cj-acct-form" method="post" action="{PREFIX}/tokens/new">'
            f'<input type="hidden" name="t" value="{esc(tok)}">'
            f'<label class="cj-acct-lbl" for="cj-acct-agent">Agent name <span class="cj-mut">(lower-case, e.g. '
            f'claude; shown as "claude for {esc(user["id"])}")</span></label>'
            f'<input class="cj-acct-in" id="cj-acct-agent" name="agent" value="claude" required maxlength="32" '
            f'pattern="[a-z][a-z0-9_\\-]{{0,31}}" autocomplete="off" autocapitalize="none" spellcheck="false">'
            f'<label class="cj-acct-lbl" for="cj-acct-days">Expires after <span class="cj-mut">(read: at most '
            f'{A.TOKEN_MAX_DAYS} days · read + write: at most {A.WRITE_MAX_DAYS})</span></label>'
            f'<select class="cj-acct-in" id="cj-acct-days" name="days">{days}</select>'
            f'<fieldset class="cj-acct-scope"><legend class="cj-acct-lbl">Access</legend>'
            f'<label><input type="radio" name="scope" value="read" checked> <span><b>Read only</b></span></label>'
            f'<label><input type="radio" name="scope" value="write"> <span><b>Read + write</b> '
            f'<span class="cj-mut">upload images, start predictions, compute results, send feedback '
            f'({WRITE_LIMIT_TEXT})</span></span></label></fieldset>'
            f'<button class="cj-acct-btn cj-acct-primary" type="submit"{" disabled" if full else ""}>'
            f'Make a token</button></form>'
            + (note(f"You have {A.TOKENS_PER_PERSON} live tokens. Revoke one to make another.", True) if full else ""))
    role = ADM.ROLE_WORD.get(user.get("role"), str(user.get("role")))
    return (f'{CSS}<div class="cj-acct"><h1>My account</h1>'
            f'<p>Signed in as <b>{esc(user["id"])}</b> ({esc(role)}).</p>{msg_html}'
            f'<section class="cj-card"><h2>Agent tokens</h2>'
            f'<p>A token lets your own AI agent (e.g. Claude Code) <b>read</b> the portal for you: data, lineage, '
            f'your predictions and results, the jobs summary. A <b>read + write</b> token can also upload images, '
            f'start predictions, compute results and send feedback; nothing else (no admin, review, Apply or '
            f'training). It sees what you see, but never more than an Advanced user. What it does is shown as '
            f'"agent for you" in the audit log and on what it makes.</p>'
            f'<p class="cj-mut">Works on the Tailscale address only for now. Keep a token private, like a key '
            f'link. Client: <a href="/cj/cryoev_client.py">cryoev_client.py</a> (one file, Python standard '
            f'library); docs: docs/API.md in the repo.</p>{door_note}'
            f'{render_tokens(rows, tok)}</section>'
            f'<section class="cj-card"><h2>New token</h2>{form}</section></div>')


def render_new(user, rec, token, base) -> str:
    """The answer page: the token, once."""
    snippet = (f'# set these two environment variables for your agent, then:\n'
               f'CRYOEV_URL = {base or "https://<the portal Tailscale address>"}\n'
               f'CRYOEV_TOKEN = <the token above>\n'
               f'python cryoev_client.py me')
    what = "read + write" if rec.get("scope") == "write" else "read-only"
    made = f"Token {rec['id']} made: {what}, expires {_when(rec['expires'])}."
    return (f'{CSS}<div class="cj-acct"><h1>Token for {esc(rec["agent"])}</h1>{note(made)}'
            f'<section class="cj-card"><h2>Copy it now: it is shown only once</h2>'
            f'<label class="cj-acct-lbl" for="cj-acct-secret">Token</label>'
            f'<textarea class="cj-acct-in cj-acct-secret" id="cj-acct-secret" readonly rows="2" '
            f'spellcheck="false">{esc(token)}</textarea>'
            f'<ul><li><b>Shown only now.</b> It is stored only as a fingerprint; if you lose it, revoke it and make '
            f'another.</li><li>Anyone who has it can read the portal as {esc(rec["agent"])} for '
            f'{esc(user["id"])}'
            + (" and upload, start predictions, compute results and send feedback" if rec.get("scope") == "write"
               else "") +
            f' until it expires or is revoked.</li></ul>'
            f'<pre>{esc(snippet)}</pre></section>'
            f'<p><a href="{PREFIX}/">Back to My account</a></p></div>')


def routes(ctx) -> list:
    secret = secrets.token_bytes(32)                     # per process: a restart makes open forms stale

    def form_token(request: Request) -> str:
        sess = ADM.binding(request)
        return hmac.new(secret, hashlib.sha256(sess.encode()).digest(), hashlib.sha256).hexdigest()[:40]

    def store():
        return A.tokens_path(ctx.cfg)

    def me(request) -> dict:
        return ctx.user(request)

    def show(request, body, status=200):
        r = ctx.page(request, "My account", body, active="account")
        r.status_code = status
        return r

    def door_note() -> str:
        if getattr(ctx.cfg, "auth", "key") == "access":
            return note("You are on the Cloudflare address: tokens made here work on the Tailscale address only "
                        "(for now).", True)
        return ""

    def page(request, msg_html="", status=200):
        user = me(request)
        try:
            rows = A.list_tokens(store(), user["id"])
        except A.TokenError as e:
            rows, msg_html = [], msg_html + note(f"Your tokens can't be listed: {e}.", True)
        return show(request, render_page(user, rows, form_token(request), msg_html, door_note()), status)

    async def form(request: Request):
        """-> (fields, None) or (None, the page saying why)."""
        ctype = (request.headers.get("content-type") or "").split(";")[0].strip().lower()
        if ctype != "application/x-www-form-urlencoded":
            return None, await run_in_threadpool(page, request, note("Send the form from this page.", True), 415)
        body = await request.body()
        if len(body) > MAX_FORM:
            return None, await run_in_threadpool(page, request, note("That form is too large.", True), 413)
        try:
            q = parse_qs(body.decode("utf-8"), keep_blank_values=True, max_num_fields=10)
        except (UnicodeDecodeError, ValueError):
            return None, await run_in_threadpool(page, request, note("That form could not be read.", True), 400)
        f = {k: v[0].strip() for k, v in q.items() if v}
        if not hmac.compare_digest(f.get("t", "").encode(), form_token(request).encode()):
            return None, await run_in_threadpool(page, request, note("This form has expired. Try again.", True), 403)
        return f, None

    def audit_line(request, action, rec) -> str:
        """The self-logged line (portal.audit.event; SELF_LOGGED in app.py). -> "" or a warning note."""
        if PA.event(ctx.cfg.state_dir, request, action, rec.get("id"),
                    {"for": rec.get("person"), "agent": rec.get("agent"), "scope": rec.get("scope"),
                     "expires": rec.get("expires")}, log_to=log):
            return ""
        return note("Done, but the audit log could not be written.", True)

    def index(request: Request):                         # sync: a file read
        text = ADM.flash_take(ADM.binding(request), request.query_params.get("f"))
        return page(request, note(text) if text else "")

    async def new(request: Request):
        f, err = await form(request)
        if err:
            return err
        user = me(request)
        scope = f.get("scope") or "read"
        if scope not in A.TOKEN_SCOPES:
            return await run_in_threadpool(page, request, note("No token was made: pick Read only or Read + write.",
                                                               True), 400)
        try:
            token, rec = await run_in_threadpool(A.create_token, store(), user["id"], f.get("agent", ""),
                                                 f.get("days", ""), scope, ctx.cfg.roster)
        except A.TokenError as e:
            return await run_in_threadpool(page, request, note(f"No token was made: {e}", True), 400)
        except OSError as e:
            log.error("account: token store not written (%s)", type(e).__name__)
            return await run_in_threadpool(page, request, note("No token was made: the token store is busy or "
                                                               "can't be written. Try again.", True), 503)
        warn = await run_in_threadpool(audit_line, request, "agent token made", rec)
        base = ADM.link_base(ctx.cfg, request.headers.get("origin")) if getattr(ctx.cfg, "auth", "key") != "access" \
            else ""
        r = show(request, warn + render_new(user, rec, token, base))
        del token
        r.headers["Cache-Control"] = "no-store"
        r.headers["Pragma"] = "no-cache"
        return r

    async def revoke(request: Request):
        f, err = await form(request)
        if err:
            return err
        user = me(request)
        try:
            rec = await run_in_threadpool(A.revoke_token, store(), f.get("id", ""), user["id"], user["id"])
        except A.TokenError as e:
            return await run_in_threadpool(page, request, note(f"Nothing was revoked: {e}", True), 404)
        except OSError as e:
            log.error("account: token store not written (%s)", type(e).__name__)
            return await run_in_threadpool(page, request, note("Nothing was revoked: the token store is busy or "
                                                               "can't be written. Try again.", True), 503)
        warn = await run_in_threadpool(audit_line, request, "agent token revoked", rec)
        if warn:
            return await run_in_threadpool(page, request, warn)
        tok = ADM.flash_put(ADM.binding(request), f"Token {rec['id']} ({rec['agent']}) is revoked.")
        return ctx.no_store(RedirectResponse(f"{PREFIX}/?f={quote(tok)}", status_code=303))

    return [Route(PREFIX + "/", index, methods=["GET"]),
            Route(PREFIX + "/tokens/new", new, methods=["POST"]),
            Route(PREFIX + "/tokens/revoke", revoke, methods=["POST"])]
