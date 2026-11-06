"""
CJ portal, P1 shell (admin only, tailnet): one address, one sign-in, one look. Spec: P1_SHELL_SPEC.md (private).

    python -m portal.app --host 127.0.0.1 --port 8770 [--registry <dir>] [--review-upstream http://127.0.0.1:8766]
        [--review-package <dir with .review_key>] [--roster <path>] [--state-dir <path>] [--origin <url> ...]

Routes (everything except OPEN needs a session; see Gate):
  /                      home: platform status from <registry>/live/live.json + module cards from modules.json
  /login?k=<key>         key link -> session cookie -> relative redirect to /
  /healthz               {"ok": true}
  /cj/cj.css, brand.json, header.js, live.js, review.js, modules.json, me.json     the shell's own files
  /history/history.html, /history/live/live.json               served from the registry folder (allowlist only);
                         history.html gets the cj.css / header.js include lines before </head> (file untouched).
                         A hidden "classic" page: not in the nav, linked from home's footer.
  /review/               the review shell: the bar + a full-height same-origin iframe of /review/app/ (its query,
                         minus any k, goes into the iframe's src; review.js mirrors the frame's URL back)
  /review/app/*          proxy to the real review server (portal/proxy.py), upstream path = the part after /review/app
  /review/<other>        404: kept free for the multi-package route /review/<pkg_id>/ (wave 2; see review_mount)
The sandbox is NOT proxied (it shares the real package's localStorage key); modules.json links out to :8443.
Roles: plug-in modules declare ROLES (portal/modules); the core pages start from CORE_ROLES (review, history: admin)
and a valid modules.json entry may replace one (manifest_rules; fail closed). The rules are read once at start (an
unreadable modules.json stops the portal; a role change needs a restart). The Gate answers 403 to any other role.
A top-level document GET of /review/app/ (a copied tool link) answers 302 /review/?<query minus k>.

Defaults: --registry = data_utils.config.registry_root(); --review-package = the newest open revise job on port
8766 in that registry (its review_package path); roster and state dir under %USERPROFILE%\\.cryoev (portal/auth.py).
uvicorn runs with its access log OFF and proxy headers OFF; the app writes its own access log with `k` redacted.
"""
from __future__ import annotations

import argparse
import contextlib
import html
import importlib.util
import json
import logging
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote_plus, urlsplit

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from starlette.applications import Starlette  # noqa: E402
from starlette.requests import Request, cookie_parser  # noqa: E402
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse, Response  # noqa: E402
from starlette.routing import Route  # noqa: E402

from data_utils import config as repo_config  # noqa: E402
from portal import auth as A  # noqa: E402
from portal import modules as M  # noqa: E402
from portal import proxy as P  # noqa: E402

OPEN = frozenset({"/login", "/cj/cj.css", "/cj/brand.json", "/healthz"})   # the ONLY routes without a session
UNSAFE = frozenset({"POST", "PUT", "DELETE", "PATCH"})                      # need Origin == this portal
SECURITY_HEADERS = ((b"x-content-type-options", b"nosniff"), (b"x-frame-options", b"DENY"),
                    (b"referrer-policy", b"same-origin"))
INCLUDES = b'<link rel="stylesheet" href="/cj/cj.css">\n<script src="/cj/header.js" defer></script>\n'
STATIC = {"cj.css": "text/css; charset=utf-8", "header.js": "text/javascript; charset=utf-8",
          "live.js": "text/javascript; charset=utf-8", "review.js": "text/javascript; charset=utf-8",
          "brand.json": "application/json"}
REVIEW_BASE = "/review"           # the real review: shell at /review/, tool proxied at /review/app/
MANIFEST = HERE / "modules.json"
CORE_ROLES = {"/review": ("admin",), "/history": ("admin",)}   # built-in defaults; modules.json may only replace
                                                                # one with a valid entry (manifest_rules)
CORE_IDS = {"/review": "review", "/history": "history"}       # the only entry that may set each core prefix
TPL_RE = re.compile(r"\{\{(\w+)\}\}")
# module icons (24 x 24, stroked; design direction A)
ICONS = {
    "home": '<path d="M4 11l8-7 8 7"/><path d="M6 10v10h12V10"/>',
    "data": '<ellipse cx="12" cy="6" rx="7" ry="2.6"/><path d="M5 6v12c0 1.4 3.1 2.6 7 2.6s7-1.2 7-2.6V6"/>'
            '<path d="M5 12c0 1.4 3.1 2.6 7 2.6s7-1.2 7-2.6"/>',
    "jobs": '<path d="M9 6h11M9 12h11M9 18h11"/><circle cx="5" cy="6" r="1.3"/><circle cx="5" cy="12" r="1.3"/>'
            '<circle cx="5" cy="18" r="1.3"/>',
    "history": '<circle cx="6" cy="6" r="2"/><circle cx="6" cy="18" r="2"/><circle cx="18" cy="13" r="2"/>'
               '<path d="M6 8v8M8 6h3a5 5 0 015 5v0"/>',
    "review": '<circle cx="12" cy="12" r="8.5"/><circle cx="12" cy="12" r="5.5" stroke-dasharray="2 2"/>'
              '<path d="M9.5 12.2l1.8 1.8 3.4-3.6"/>',
    "sandbox": '<rect x="4" y="4" width="16" height="16" rx="2" stroke-dasharray="3 2.2"/><path d="M9 15l6-6M10 9h5v5"/>',
    "predict": '<path d="M12 15V4M8 8l4-4 4 4M4 14v6h16v-6"/>',
    "results": '<path d="M5 20v-8M10 20V5M15 20v-6M20 20v-10"/>',
    "train": '<path d="M19.5 13a7.5 7.5 0 01-13.3 4.2M4.5 11a7.5 7.5 0 0113.3-4.2"/><path d="M18 3.5v3.5h-3.5M6 20.5V17h3.5"/>',
    "admin": '<path d="M12 3l7 3v5.5c0 4.3-2.9 7.6-7 9.5-4.1-1.9-7-5.2-7-9.5V6z"/><path d="M9.5 12l1.8 1.8 3.4-3.5"/>',
}
EXT_ICON = '<svg class="cj-ext" viewBox="0 0 24 24" aria-hidden="true"><path d="M8 16L16 8M10 8h6v6"/></svg>'
REVIEW_PORT = 8766
PUBLIC_PORTS = {"", "443", "8443", "8445", "8446"}   # tailscale serve's https ports (links use https there)
HOST_RE = re.compile(r"[A-Za-z0-9.-]{1,253}|\[[0-9A-Fa-f:.]{2,45}\]")
log = logging.getLogger("cj")
access_log = logging.getLogger("cj.access")


@dataclass
class Config:
    registry: Path | None
    roster: Path
    state_dir: Path
    review_upstream: str = f"http://127.0.0.1:{REVIEW_PORT}"
    review_key: str | None = None
    review_package: Path | None = None
    origins: tuple = ()
    review_timeout: float = P.TIMEOUT_S
    max_body: int = P.MAX_BODY
    manifest: Path = MANIFEST


# ---------------------------------------------------------------- helpers
def redact_query(qs: str) -> str:
    """The query string for logs: any `k` value replaced by ***."""
    return "&".join(("k=***" if unquote_plus(p.split("=", 1)[0]) == "k" else p) for p in qs.split("&") if p)


def header(scope, name: bytes) -> str | None:
    for k, v in scope.get("headers", []):
        if k.lower() == name:
            return v.decode("latin-1")
    return None


def origin_ok(scope, extra=()) -> bool:
    """Origin must be this portal's own origin (scheme://Host); http only for a loopback Host. No Origin = no."""
    origin = header(scope, b"origin")
    if not origin or origin == "null":
        return False
    if origin in extra:
        return True
    host = (header(scope, b"host") or "").lower()
    u = urlsplit(origin)
    if u.scheme not in ("https", "http") or u.path or not host:
        return False
    if u.scheme == "http" and (u.hostname or "") not in ("127.0.0.1", "localhost", "::1"):
        return False
    return u.netloc.lower() == host


def read_json(path: Path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def page(status: int, title: str, text: str) -> HTMLResponse:
    return P.error_page(status, title, text)


def dur(s) -> str:
    """Seconds -> '40 s' / '12 min' / '3 h 5 min' / '2 d 4 h' (as the history page's Live tab)."""
    if s is None:
        return "?"
    try:
        s = max(0, round(float(s)))
    except (TypeError, ValueError):
        return "?"
    if s < 60:
        return f"{s} s"
    m = round(s / 60)
    if m < 60:
        return f"{m} min"
    h, m = divmod(m, 60)
    if h < 48:
        return f"{h} h" + (f" {m} min" if m else "")
    return f"{h // 24} d {h % 24} h"


def snapshot_age(live) -> float | None:
    try:
        t = datetime.fromisoformat(str(live["generated_utc"]).replace("Z", "+00:00"))
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - t).total_seconds()
    except (KeyError, TypeError, ValueError):
        return None


def is_stale(live) -> bool:
    age = snapshot_age(live)
    try:
        every = float(live.get("interval_s") or 30)
    except (TypeError, ValueError):
        every = 30.0
    return age is None or age > 3 * every


def external_href(mod: dict, host: str | None) -> str | None:
    """A module on its own port (the sandbox): https://<host>:<public_port> behind tailscale serve, else
    http://<host>:<local_port>. None for a Host header that is not a plain host name."""
    ext = mod.get("external") or {}
    if not host:
        return None
    name, _, port = host.rpartition(":") if not host.endswith("]") else (host, "", "")
    if not name:
        name, port = host, ""
    if not HOST_RE.fullmatch(name):
        return None
    path = ext.get("path") or "/"
    if port in PUBLIC_PORTS:
        return f"https://{name}:{int(ext['public_port'])}{path}"
    return f"http://{name}:{int(ext['local_port'])}{path}"


def registry_review_package(registry: Path | None, port: int = REVIEW_PORT) -> Path | None:
    """The real review package: the newest open revise job on `port` in the registry -> its review_package path.
    Read through registry/registry.py's current(); None when anything is missing."""
    if registry is None or not Path(registry).is_dir():
        return None
    try:
        spec = importlib.util.spec_from_file_location("cj_registry", REPO / "registry" / "registry.py")
        reg = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reg)
        jobs = reg.current("job", Path(registry))
        pkgs = reg.current("review_package", Path(registry))
    except Exception as e:          # an unreadable registry must not stop the portal
        log.warning("registry not readable for the review package: %s", e)
        return None
    open_jobs = sorted((j for j in jobs.values() if j.get("type") == "revise" and j.get("port") == port
                        and j.get("status") not in ("done", "cancelled") and j.get("package") in pkgs),
                       key=lambda j: j["id"])
    if not open_jobs:
        return None
    path = pkgs[open_jobs[-1]["package"]].get("path")
    return Path(path) if path else None


def read_review_key(pkg: Path | None) -> str | None:
    if pkg is None:
        return None
    try:
        return (Path(pkg) / ".review_key").read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


# ---------------------------------------------------------------- gate
class Gate:
    """Outermost ASGI layer: security headers on every response, deny by default (only OPEN without a session),
    Origin check on unsafe methods, and the access log with `k` redacted (never cookies or keys)."""

    def __init__(self, app: Starlette, auth: A.Auth, origins=(), rules=()):
        self.app = app
        self.auth = auth
        self.origins = tuple(origins)
        self.rules = list(rules)            # [(module PREFIX, roles)] from portal.modules.load, longest first
        self.routes = app.routes

    async def __call__(self, scope, receive, send):
        if scope["type"] == "lifespan":
            return await self.app(scope, receive, send)
        if scope["type"] != "http":                     # no websockets in P1
            await receive()
            return await send({"type": "websocket.close", "code": 1008})
        box = {"status": None}

        frameable = scope.get("path", "").startswith("/review/")   # the shell may frame the review tool (same origin)

        async def send_secure(msg):
            if msg["type"] == "http.response.start":
                names = {k for k, _ in SECURITY_HEADERS}
                sec = [(k, (b"SAMEORIGIN" if frameable and k == b"x-frame-options" else v)) for k, v in SECURITY_HEADERS]
                hdrs = [(k, v) for k, v in msg.get("headers", []) if k.lower() not in names] + sec
                msg = {**msg, "headers": hdrs}
                box["status"] = msg["status"]
            await send(msg)

        user = None
        try:
            path, method = scope["path"], scope["method"]
            deny = None
            if path not in OPEN:
                cookies = cookie_parser(header(scope, b"cookie") or "")
                state, user = self.auth.check(cookies.get(A.COOKIE))
                if state == "none":
                    deny = page(401, "Sign in", "Open your key link (the address ending in /login?k=...) on this "
                                                "device to sign in.")
                elif state != "ok":
                    deny = page(403, "Not signed in", "This sign-in is unknown, expired or revoked. Open a current "
                                                      "key link.")
            if deny is None and user is not None and not M.allowed(self.rules, path, user.get("role")):
                deny = page(403, "Not allowed", "Your role can't open this part of the portal.")
            if deny is None and method in UNSAFE and not origin_ok(scope, self.origins):
                deny = page(403, "Refused", "Cross-origin request refused.")
            if deny is not None:
                return await deny(scope, receive, send_secure)
            scope["cj_user"] = user
            await self.app(scope, receive, send_secure)
        finally:
            qs = redact_query(scope.get("query_string", b"").decode("latin-1"))
            client = (scope.get("client") or ("-",))[0]
            who = (user or {}).get("id") or scope.get("cj_login") or "-"
            access_log.info('%s "%s %s%s" %s %s', client, scope.get("method"), scope.get("path"),
                            f"?{qs}" if qs else "", box["status"], who)


# ---------------------------------------------------------------- app
def load_manifest(path: Path = MANIFEST) -> list:
    """modules.json, strictly: the Gate's rules come from it at start, so a file that can't be read or parsed stops
    the portal (ValueError) instead of starting with guessed roles."""
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise ValueError(f"modules.json not readable ({path}): {e}") from None
    mods = doc.get("modules") if isinstance(doc, dict) else None
    if not isinstance(mods, list) or not all(isinstance(m, dict) for m in mods):
        raise ValueError(f"modules.json has no list of module objects ({path})")
    return mods


def manifest(path: Path = MANIFEST) -> list:
    """modules.json for display (home cards, /cj/modules.json), read per request; [] when unreadable. Never used for
    access decisions: those are the Gate's rules, fixed at start."""
    try:
        return load_manifest(path)
    except ValueError:
        return []


def manifest_roles(mods, mod_id) -> tuple:
    for m in mods:
        if m.get("id") == mod_id:
            return tuple(m.get("roles") or ())
    return ()


def valid_roles(roles):
    """A non-empty list of known roles (portal/auth.py ROLES) as a tuple, else None."""
    if not isinstance(roles, (list, tuple)) or not roles or not all(isinstance(r, str) and r in A.ROLES for r in roles):
        return None
    return tuple(dict.fromkeys(roles))


def manifest_rules(mods, taken=()) -> list:
    """Gate rules for the core pages: [(prefix, roles)]. Starts from the built-in CORE_ROLES (fail closed); a
    modules.json entry with a local route below "/" and valid roles sets its prefix (replacing a default only then).
    A missing, route-less or invalid entry leaves the default. A plug-in module's own ROLES win: its prefix is in
    `taken`. A core prefix (/review, /history) may be set only by its own entry (id "review" / "history"); two
    entries on one prefix raise ValueError (refuse to start)."""
    rules, owner = dict(CORE_ROLES), {}
    for m in mods:
        if not isinstance(m, dict):
            continue
        r = m.get("route")
        if m.get("external") or not isinstance(r, str) or not r.startswith("/") or r.startswith("//") or r == "/":
            continue
        prefix = "/" + r.strip("/").split("/")[0]
        if prefix in owner:
            raise ValueError(f"modules.json: {owner[prefix]!r} and {m.get('id')!r} both map to {prefix}")
        owner[prefix] = m.get("id")
        roles = valid_roles(m.get("roles"))
        if roles is None or (prefix in CORE_IDS and m.get("id") != CORE_IDS[prefix]):
            continue
        rules[prefix] = roles
    return [(p, r) for p, r in rules.items() if p not in taken]


def fill(tpl: str, values: dict) -> str:
    """{{name}} placeholders in one pass (a value containing "{{x}}" is never expanded again)."""
    return TPL_RE.sub(lambda mt: values.get(mt.group(1), mt.group(0)), tpl)


def script_json(obj) -> str:
    """JSON for a <script type="application/json"> block: no "<", ">" or "&" can end the element."""
    return (json.dumps(obj, ensure_ascii=True).replace("&", "\\u0026").replace("<", "\\u003c")
            .replace(">", "\\u003e"))


def icon(name) -> str:
    return f'<svg viewBox="0 0 24 24" aria-hidden="true">{ICONS.get(name, ICONS["home"])}</svg>'


def render_home(request: Request, cfg: Config, user: dict, rules=None) -> str:
    """Home's body (goes into base.html): status cards for roles the Gate lets read the live snapshot (History's
    rules), module cards from modules.json for the user's role, and the Classic history link."""
    brand = read_json(HERE / "static" / "brand.json") or {"name": "CJ"}
    mods = manifest(cfg.manifest)
    role = user.get("role")
    esc = html.escape
    rules = list(rules) if rules is not None else manifest_rules([])
    sees_live = M.allowed(rules, "/history/live/live.json", role)       # the Gate's answer, not the file's
    live = read_json(cfg.registry / "live" / "live.json") if (cfg.registry and sees_live) else None
    stale = live is None or is_stale(live)
    age = snapshot_age(live) if live else None

    def dot(state):
        return f'<span class="cj-dot {state}" aria-hidden="true"></span>'

    def card(title, inner, extra=""):
        return f'<section class="cj-card"{extra}><h2>{title}</h2>{inner}</section>'

    status = ""
    if sees_live:
        if live is None:
            snap = "No live snapshot (is jobs/monitor.py running?)"
        elif stale:
            snap = f"Live snapshot is stale ({dur(age)} old): the dots show unknown."
        else:
            snap = f"From live.json, {dur(age)} old."
        servers, n_up = [], 0
        srv_list = (live or {}).get("servers") or []
        for s in srv_list:
            st = "unk" if stale else ("up" if s.get("up") else "down")
            n_up += st == "up"
            word = {"unk": "unknown", "up": "up", "down": "down"}[st]
            ms = f" · {s['ms']} ms" if st == "up" and s.get("ms") is not None else ""
            servers.append(f'{dot(st)}<b>{esc(str(s.get("name", "?")).capitalize())}</b> '
                           f'<span class="cj-mut">{word}{esc(ms)}</span>')
        if not servers:
            servers = [f'{dot("unk")}<span class="cj-mut">unknown (no snapshot)</span>']
        big = (f'<div class="cj-big">{n_up} <small>of {len(srv_list)} up</small></div>' if srv_list and not stale
               else '<div class="cj-big">? <small>unknown</small></div>')
        srv_card = card("Servers", big + '<ul class="cj-list">' + "".join(f"<li>{r}</li>" for r in servers) + "</ul>")

        watchers, running = [], []
        for q in (live or {}).get("queues") or []:
            where = esc(str((q.get("watcher") or {}).get("machine") or "desktop"))
            if q.get("reach") != "ok":
                watchers.append(f'<div class="cj-big warn">{dot("down")}{esc(str(q.get("reach") or "?"))}</div>'
                                f'<div class="cj-mut">{where}</div>')
                continue
            w = q.get("watcher") or {}
            disp = str(w.get("display") or "unknown")
            st = "up" if disp == "alive" and not stale else "unk" if stale else "down"
            hb = f"heartbeat {dur(w.get('age_s'))} ago" if w.get("age_s") is not None else "no heartbeat"
            watchers.append(f'<div class="cj-big">{dot(st)}{esc(disp.capitalize())}</div>'
                            f'<div class="cj-mut">{esc(disp)} · {esc(hb)} · {where}</div>')
            r = q.get("running")
            if r and not stale:
                ep = f"epoch {r.get('epoch') if r.get('epoch') is not None else '?'}/{r.get('epochs') or '?'}"
                eta = f"ETA {dur(r.get('eta_s'))}" if r.get("eta_s") is not None else "ETA ?"
                running.append(f'<div class="cj-wrap"><b>{esc(str(r.get("job") or "?"))}</b></div>'
                               f'<div class="cj-mut">{esc(str(r.get("phase") or ""))} · {esc(ep)} · {esc(eta)}</div>')
        if not watchers:
            watchers = ['<div class="cj-big">? <small>unknown</small></div>'
                        '<div class="cj-mut">no queue in the snapshot</div>']
        if not running:
            running = ['<div class="cj-mut">' + ("unknown" if live is None else
                                                 "unknown (stale snapshot)" if stale else "no job running") + "</div>"]
        # the training snapshot: server-rendered text first, then CJLive (live.js) draws the epoch bar, ETA and last
        # mAP from the embedded snapshot and refreshes it every 30 s
        slim = None if live is None else {
            "generated_utc": live.get("generated_utc"), "interval_s": live.get("interval_s"),
            "queues": [{"reach": q.get("reach"), "running": q.get("running")} for q in live.get("queues") or []]}
        train = (f'<script type="application/json" id="cj-live-data">{script_json(slim)}</script>'
                 f'<div id="cj-train" data-cj-live="compact" data-cj-live-json="cj-live-data" '
                 f'data-cj-live-src="/history/live/live.json" data-cj-live-every="30">{"".join(running)}</div>')
        status = (f'<h2 class="cj-sec">Status</h2><div class="cj-mut" id="cj-snap">{esc(snap)}</div>'
                  f'<div class="cj-cards" id="cj-status">'
                  + srv_card + card("Desktop watcher", "".join(watchers)) + card("Training", train, ' id="cj-training"')
                  + card("Last backup", '<div class="cj-big warn">Unknown</div>'
                                        '<div class="cj-mut">live.json has no backup field yet</div>')
                  + "</div>")

    srv = {s.get("name"): s for s in (live or {}).get("servers") or []}
    cards = []
    for m in mods:
        if m.get("id") == "home" or not m.get("nav") or role not in (m.get("roles") or []):
            continue
        mid = str(m.get("id", ""))
        name, desc = esc(m.get("name", "?")), esc(m.get("desc", ""))
        ico = f'<span class="cj-ico">{icon(m.get("icon"))}</span>'
        label = f'<span class="cj-evh">{esc(m["label"])}</span>' if m.get("label") and mid == "review" else ""
        if m.get("status") != "on":
            cards.append(f'<div class="cj-mod off" data-id="{esc(mid)}" aria-disabled="true"><div class="cj-mt">{ico}'
                         f'<b>{name}</b><span class="cj-tag">coming</span></div><p>{desc}</p></div>')
            continue
        href = m.get("route") if not m.get("external") else external_href(m, request.headers.get("host"))
        h = m.get("health")
        st = "" if not h or not sees_live else dot("unk" if stale or h not in srv else
                                                   ("up" if srv[h].get("up") else "down"))
        ext = f' <span class="cj-mut">(own address)</span>{EXT_ICON}' if m.get("external") else ""
        cls = "cj-mod cj-mod-review" if mid == "review" else "cj-mod"
        cards.append(f'<a class="{cls}" data-id="{esc(mid)}" href="{esc(href or "#")}"><div class="cj-mt">{ico}'
                     f'<b>{name}</b>{label}{ext}{st}</div><p>{desc}</p></a>')
    if not cards:
        cards = ['<div class="cj-mut">No modules for your role yet.</div>']
    footer = ""
    hist = next((m for m in mods if m.get("id") == "history"), None)
    if hist and sees_live and role in (hist.get("roles") or []) and hist.get("status") == "on":
        footer = (f'<footer class="cj-foot"><a href="{esc(hist.get("route") or "/history/")}" id="cj-classic">'
                  f'Classic history</a><span>registry history, queue and the Live tab</span></footer>')

    tpl = (HERE / "templates" / "home.html").read_text(encoding="utf-8")
    return fill(tpl, {"heading": esc(brand.get("name", "CJ")), "tagline": esc(brand.get("tagline", "")),
                      "user": esc(user.get("id", "")), "status": status, "module_cards": "".join(cards),
                      "footer": footer})


def review_mount(base: str, proxy: P.ReviewProxy, title: str = "Review") -> list:
    """The routes for one review under `base` (no trailing slash):
         base/          the shell page: the bar + an iframe of base/app/ (the shell's query, minus k, goes along)
         base/app       -> base/app/
         base/app/...   the tool, through `proxy` (whose prefix must be base + "/app"). A top-level document GET of
                        the tool's page (a copied link) is sent to the shell instead: 302 base/?<query minus k>
    The real review is review_mount("/review", ...). Wave 2's multi-package route can mount each package the same
    way (/review/<pkg_id>/, ids never "app"), in front of the /review/{rest} catch-all."""
    app_prefix = base + "/app"
    if proxy.prefix != app_prefix:
        raise ValueError(f"review proxy prefix {proxy.prefix!r} != {app_prefix!r}")

    async def shell(request: Request):
        q = P.strip_k(request.scope.get("query_string", b"").decode("latin-1"))
        src = app_prefix + "/" + (f"?{q}" if q else "")
        body = (f'<iframe id="cj-review-frame" class="cj-review-frame" title="{html.escape(title)} tool" '
                f'src="{html.escape(src)}" data-base="{html.escape(base)}/" data-app="{html.escape(app_prefix)}/" '
                f'allow="clipboard-write; fullscreen"></iframe>\n<script src="/cj/review.js" defer></script>')
        return M.Ctx.page(request, title, body, active="review")

    def to_shell(request: Request) -> bool:
        """A browser opening the tool's page as the whole window (not in the shell's frame)."""
        if request.method != "GET" or request.url.path not in (app_prefix + "/", app_prefix + "/index.html"):
            return False
        dest = request.headers.get("sec-fetch-dest")
        if dest is not None:
            return dest == "document"                       # the frame's loads say "iframe"
        if "text/html" not in request.headers.get("accept", ""):
            return False
        ref = request.headers.get("referer")                # older browsers: framed loads come from the shell page
        if not ref:
            return False                                    # unknown: serve the tool (a redirect could loop)
        u = urlsplit(ref)
        return not (u.netloc.lower() == (request.headers.get("host") or "").lower() and u.path.startswith(base + "/"))

    async def tool(request: Request):
        if to_shell(request):
            q = P.strip_k(request.scope.get("query_string", b"").decode("latin-1"))
            r = RedirectResponse(base + "/" + (f"?{q}" if q else ""), status_code=302)
            r.headers["Cache-Control"] = "no-store"
            return r
        return await proxy.handle(request, request.scope["cj_user"]["id"])

    async def to_app(request: Request):
        return RedirectResponse(app_prefix + "/", status_code=307)

    return [Route(base + "/", shell, methods=["GET"]),
            Route(app_prefix, to_app, methods=["GET"]),
            Route(app_prefix + "/{path:path}", tool, methods=["GET", "HEAD", "PUT", "POST", "DELETE"])]


def create_app(cfg: Config) -> Gate:
    """Raises ValueError when modules.json can't be read (the Gate's rules come from it): refuse to start."""
    auth = A.Auth(cfg.roster, cfg.state_dir)
    rules_box: list = []                            # the Gate's rules, filled below; home asks them, not the file
    proxy = P.ReviewProxy(cfg.review_upstream, cfg.review_key, cfg.review_timeout, cfg.max_body,
                          prefix=REVIEW_BASE + "/app")

    def no_store(r: Response) -> Response:
        r.headers["Cache-Control"] = "no-store"
        return r

    async def home(request: Request):
        brand = read_json(HERE / "static" / "brand.json") or {}
        body = render_home(request, cfg, request.scope["cj_user"], rules_box)
        return M.Ctx.page(request, str(brand.get("name") or "CJ"), body, active="home")

    async def login(request: Request):
        key = request.query_params.get("k")
        if not key:
            state, _ = auth.check(request.cookies.get(A.COOKIE))
            if state == "ok":
                return no_store(RedirectResponse("/", status_code=303))
            return page(401, "Sign in", "Open your key link (the address ending in /login?k=...) to sign in.")
        res = auth.login(key)
        if res is None:
            return page(403, "Key not valid", "That key link is not valid, or it was replaced or revoked.")
        token, person = res
        request.scope["cj_login"] = person["id"]
        r = no_store(RedirectResponse("/", status_code=303))
        r.raw_headers.append((b"set-cookie", A.cookie_header(token).encode("latin-1")))
        return r

    async def healthz(request: Request):
        return no_store(JSONResponse({"ok": True}))

    def static(name):
        async def serve(request: Request):
            r = Response((HERE / "static" / name).read_bytes(), media_type=STATIC[name])
            r.headers["Cache-Control"] = "no-cache"
            return r
        return serve

    async def modules(request: Request):
        role = request.scope["cj_user"].get("role")                 # only the modules this role can see
        mods = [m for m in manifest(cfg.manifest) if role in (m.get("roles") or [])]
        return no_store(JSONResponse({"schema": "cj_modules/1", "modules": mods}))

    async def me(request: Request):
        u = request.scope["cj_user"]
        return no_store(JSONResponse({"id": u["id"], "role": u["role"]}))

    def registry_file(rel) -> Path | None:
        return (cfg.registry / rel) if cfg.registry else None

    async def history_html(request: Request):
        p = registry_file("history.html")
        try:
            body = p.read_bytes()
        except (OSError, AttributeError):
            return page(404, "History not built", "No history.html in the registry folder yet "
                                                  "(python registry/history_page.py).")
        i = body.lower().find(b"</head>")
        if i >= 0:                                      # the served copy only; the file on disk is untouched
            body = body[:i] + INCLUDES + body[i:]
        return no_store(Response(body, media_type="text/html; charset=utf-8"))

    async def live_json(request: Request):
        p = registry_file("live/live.json")
        try:
            body = p.read_bytes()
        except (OSError, AttributeError):
            return no_store(page(404, "No live snapshot", "live/live.json not found: is jobs/monitor.py running?"))
        return no_store(Response(body, media_type="application/json"))

    async def history_other(request: Request):
        return page(404, "Not found", "Only history.html and live/live.json are served here.")

    def redirect(to):
        async def go(request: Request):
            return RedirectResponse(to, status_code=307)
        return go

    async def review_free(request: Request):
        return page(404, "Not found", "Nothing here yet. The review is at /review/.")

    routes = [
        Route("/", home, methods=["GET"]),
        Route("/login", login, methods=["GET"]),
        Route("/healthz", healthz, methods=["GET"]),
        *[Route(f"/cj/{n}", static(n), methods=["GET"]) for n in STATIC],
        Route("/cj/modules.json", modules, methods=["GET"]),
        Route("/cj/me.json", me, methods=["GET"]),
        Route("/history", redirect("/history/"), methods=["GET"]),
        Route("/history/", redirect("history.html"), methods=["GET"]),
        Route("/history/history.html", history_html, methods=["GET"]),
        Route("/history/live/live.json", live_json, methods=["GET"]),
        Route("/history/{rest:path}", history_other, methods=["GET"]),
        Route(REVIEW_BASE, redirect(REVIEW_BASE + "/"), methods=["GET"]),
        *review_mount(REVIEW_BASE, proxy),
        # kept free for /review/<pkg_id>/ (wave 2: mount those before this catch-all)
        Route(REVIEW_BASE + "/{rest:path}", review_free, methods=["GET", "HEAD", "PUT", "POST", "DELETE"]),
    ]
    mod_routes, rules = M.load(M.Ctx(cfg))          # portal/modules/<id>.py plug-ins (BUILD_CONTRACTS.md section 1)
    routes.extend(mod_routes)
    # core pages (review, history): built-in defaults, replaced only by valid modules.json entries; a plug-in's own
    # ROLES win for its PREFIX. Read ONCE here: a role change in modules.json needs a restart.
    core = manifest_rules(load_manifest(cfg.manifest), {pr for pr, _ in rules})      # ValueError: refuse to start
    rules = sorted(rules + core, key=lambda pr: -len(pr[0]))
    rules_box[:] = rules

    @contextlib.asynccontextmanager
    async def lifespan(app):
        await proxy.start()
        try:
            yield
        finally:
            await proxy.close()

    star = Starlette(routes=routes, lifespan=lifespan)
    star.router.redirect_slashes = False
    gate = Gate(star, auth, cfg.origins, rules)
    gate.proxy = proxy
    gate.auth = auth
    return gate


def build_config(args) -> Config:
    registry = args.registry if args.registry else repo_config.registry_root(required=False)
    pkg = args.review_package or registry_review_package(registry)
    key = read_review_key(pkg)
    return Config(registry=Path(registry) if registry else None, roster=args.roster or A.default_roster(),
                  state_dir=args.state_dir or A.default_state_dir(), review_upstream=args.review_upstream,
                  review_key=key, review_package=pkg, origins=tuple(args.origin or ()),
                  manifest=getattr(args, "modules", None) or MANIFEST)


def startup_lines(cfg: Config, host: str, port: int) -> list:
    """What runs, printed at start (no secrets)."""
    n = len(A.Roster(cfg.roster).people())
    return [
        f"[cj] CJ portal on http://{host}:{port} (put tailscale serve :8446 in front)",
        f"[cj] gate: deny by default; open without a session: {', '.join(sorted(OPEN))}; "
        f"Origin check on {', '.join(sorted(UNSAFE))}",
        f"[cj] roster: {cfg.roster} ({n} people){'' if cfg.roster.is_file() else ' MISSING: nobody can sign in'}",
        f"[cj] sessions: {cfg.state_dir / 'sessions.json'} (sha256 of session ids only)",
        f"[cj] history: {cfg.registry or 'registry NOT FOUND'} (history.html + live/live.json only)",
        f"[cj] review: /review/ = shell, /review/app/ -> {cfg.review_upstream}; key from "
        + (f"{cfg.review_package}" if cfg.review_key else
           f"NOWHERE ({cfg.review_package or 'no package'}: no .review_key), so review answers 403"),
        "[cj] roles (read once, from built-in defaults + modules.json): "
        + "; ".join(f"{p} {'/'.join(r)}" for p, r in manifest_rules(manifest(cfg.manifest)))
        + "; plug-in modules: their own ROLES. Restart the portal after changing roles in modules.json",
        "[cj] sandbox: not proxied (link out to :8443)",
        "[cj] logs: uvicorn access log off, proxy headers off; own access log with k= redacted",
    ]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1", help="bind address (default 127.0.0.1; tailscale serve fronts it)")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--registry", type=Path, default=None, help="registry folder (default: data_utils.config)")
    ap.add_argument("--review-upstream", default=f"http://127.0.0.1:{REVIEW_PORT}")
    ap.add_argument("--review-package", type=Path, default=None,
                    help="the real review package (its .review_key is sent upstream); default: from the registry")
    ap.add_argument("--roster", type=Path, default=None, help=f"default {A.default_roster()}")
    ap.add_argument("--state-dir", type=Path, default=None, help=f"default {A.default_state_dir()}")
    ap.add_argument("--modules", type=Path, default=None,
                    help="the module manifest (default portal/modules.json); read once for the roles; unreadable = no start")
    ap.add_argument("--origin", action="append", default=None,
                    help="extra allowed Origin for PUT/POST/DELETE, e.g. https://<name>:8446 (repeatable)")
    args = ap.parse_args(argv)
    import uvicorn
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", stream=sys.stdout)
    cfg = build_config(args)
    if args.host in ("0.0.0.0", "::"):
        print("[WARN] listening on every interface; P1 expects 127.0.0.1 behind tailscale serve", flush=True)
    try:
        app = create_app(cfg)
    except ValueError as e:                          # e.g. modules.json unreadable: no guessed roles, no start
        print(f"[ERROR] refusing to start: {e}", flush=True)
        return 2
    for line in startup_lines(cfg, args.host, args.port):
        print(line, flush=True)
    uvicorn.run(app, host=args.host, port=args.port, access_log=False, proxy_headers=False,
                server_header=False, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
