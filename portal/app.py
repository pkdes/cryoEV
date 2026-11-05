"""
CJ portal, P1 shell (admin only, tailnet): one address, one sign-in, one look. Spec: P1_SHELL_SPEC.md (private).

    python -m portal.app --host 127.0.0.1 --port 8770 [--registry <dir>] [--review-upstream http://127.0.0.1:8766]
        [--review-package <dir with .review_key>] [--roster <path>] [--state-dir <path>] [--origin <url> ...]

Routes (everything except OPEN needs a session; see Gate):
  /                      home: platform status from <registry>/live/live.json + module cards from modules.json
  /login?k=<key>         key link -> session cookie -> relative redirect to /
  /healthz               {"ok": true}
  /cj/cj.css, brand.json, header.js, modules.json, me.json     the shell's own files
  /history/history.html, /history/live/live.json               served from the registry folder (allowlist only);
                         history.html gets the cj.css / header.js include lines before </head> (file untouched)
  /review/*              proxy to the real review server (portal/proxy.py); /review -> /review/
The sandbox is NOT proxied (it shares the real package's localStorage key); modules.json links out to :8443.

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
          "brand.json": "application/json"}
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
def render_home(request: Request, cfg: Config, user: dict) -> str:
    brand = read_json(HERE / "static" / "brand.json") or {"name": "CJ"}
    mods = (read_json(HERE / "modules.json") or {}).get("modules", [])
    live = read_json(cfg.registry / "live" / "live.json") if cfg.registry else None
    esc = html.escape
    stale = live is None or is_stale(live)
    age = snapshot_age(live) if live else None

    if live is None:
        snap = "No live snapshot (is jobs/monitor.py running?)"
    elif stale:
        snap = f"Live snapshot is stale ({dur(age)} old): the dots show unknown."
    else:
        snap = f"Live snapshot {dur(age)} old."

    def card(title, rows, cls=""):
        items = "".join(f"<li>{r}</li>" for r in rows)
        return f'<section class="cj-card cj-stat{cls}"><h2>{esc(title)}</h2><ul class="cj-list">{items}</ul></section>'

    def dot(state):
        return f'<span class="cj-dot {state}" aria-hidden="true"></span>'

    servers = []
    for s in (live or {}).get("servers") or []:
        st = "unk" if stale else ("up" if s.get("up") else "down")
        word = {"unk": "unknown", "up": "up", "down": "down"}[st]
        ms = f" · {s['ms']} ms" if st == "up" and s.get("ms") is not None else ""
        servers.append(f'{dot(st)}<b>{esc(str(s.get("name", "?")).capitalize())}</b> '
                       f'<span class="cj-mut">{word}{esc(ms)}</span>')
    if not servers:
        servers = [f'{dot("unk")}<span class="cj-mut">unknown (no snapshot)</span>']

    watchers, running = [], []
    for q in (live or {}).get("queues") or []:
        where = esc(str((q.get("watcher") or {}).get("machine") or "desktop"))
        if q.get("reach") != "ok":
            watchers.append(f'{dot("down")}<b>{where}</b> <span class="cj-mut">{esc(str(q.get("reach") or "?"))}'
                            f'</span>')
            continue
        w = q.get("watcher") or {}
        disp = str(w.get("display") or "unknown")
        st = "up" if disp == "alive" and not stale else "unk" if stale else "down"
        hb = f" · heartbeat {dur(w.get('age_s'))} ago" if w.get("age_s") is not None else ""
        watchers.append(f'{dot(st)}<b>{where}</b> <span class="cj-mut">{esc(disp + hb)}</span>')
        r = q.get("running")
        if r:
            ep = f"epoch {r.get('epoch') if r.get('epoch') is not None else '?'}/{r.get('epochs') or '?'}"
            eta = f"ETA {dur(r.get('eta_s'))}" if r.get("eta_s") is not None else "ETA ?"
            running.append(f'<b class="cj-wrap">{esc(str(r.get("job") or "?"))}</b>'
                           f'<span class="cj-mut">{esc(str(r.get("phase") or ""))} · {esc(ep)} · {esc(eta)}</span>')
    if not watchers:
        watchers = ['<span class="cj-mut">unknown (no queue in the snapshot)</span>']
    if not running:
        running = ['<span class="cj-mut">' + ("unknown" if live is None else "no job running") + '</span>']
    status = (card("Servers", servers) + card("Desktop watcher", watchers) + card("Running job", running)
              + card("Last backup", ['<b>unknown</b> <span class="cj-mut">(live.json has no backup field yet)</span>']))

    srv = {s.get("name"): s for s in (live or {}).get("servers") or []}
    role = user.get("role")
    cards = []
    for m in mods:
        if m.get("id") == "home" or role not in (m.get("roles") or []):
            continue
        name, desc = esc(m.get("name", "?")), esc(m.get("desc", ""))
        ico = f'<span class="cj-ico" aria-hidden="true">{esc(m.get("name", "?")[:1])}</span>'
        if m.get("status") != "on":
            cards.append(f'<div class="cj-mod off" aria-disabled="true">{ico}<span class="cj-modt"><b>{name}</b>'
                         f'<span class="cj-mut">{desc}</span></span><span class="cj-tag">coming</span></div>')
            continue
        href = m.get("route") if not m.get("external") else external_href(m, request.headers.get("host"))
        h = m.get("health")
        st = "" if not h else dot("unk" if stale or h not in srv else ("up" if srv[h].get("up") else "down"))
        ext = ' <span class="cj-mut">(own address)</span>' if m.get("external") else ""
        cards.append(f'<a class="cj-mod" href="{esc(href or "#")}">{ico}<span class="cj-modt"><b>{name}</b>{ext}'
                     f'<span class="cj-mut">{desc}</span></span>{st}</a>')

    tpl = (HERE / "templates" / "home.html").read_text(encoding="utf-8")
    for k, v in {"title": esc(brand.get("name", "CJ")), "tagline": esc(brand.get("tagline", "")),
                 "user": esc(user.get("id", "")), "snapshot": esc(snap), "status_cards": status,
                 "module_cards": "".join(cards)}.items():
        tpl = tpl.replace("{{" + k + "}}", v)
    return tpl


def create_app(cfg: Config) -> Gate:
    auth = A.Auth(cfg.roster, cfg.state_dir)
    proxy = P.ReviewProxy(cfg.review_upstream, cfg.review_key, cfg.review_timeout, cfg.max_body)

    def no_store(r: Response) -> Response:
        r.headers["Cache-Control"] = "no-store"
        return r

    async def home(request: Request):
        return no_store(HTMLResponse(render_home(request, cfg, request.scope["cj_user"])))

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
        r = Response((HERE / "modules.json").read_bytes(), media_type="application/json")
        r.headers["Cache-Control"] = "no-cache"
        return r

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

    async def review(request: Request):
        return await proxy.handle(request, request.scope["cj_user"]["id"])

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
        Route("/review", redirect("/review/"), methods=["GET"]),
        Route("/review/{path:path}", review, methods=["GET", "HEAD", "PUT", "POST", "DELETE"]),
    ]
    mod_routes, rules = M.load(M.Ctx(cfg))          # portal/modules/<id>.py plug-ins (BUILD_CONTRACTS.md section 1)
    routes.extend(mod_routes)

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
                  review_key=key, review_package=pkg, origins=tuple(args.origin or ()))


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
        f"[cj] review: /review/ -> {cfg.review_upstream}; key from "
        + (f"{cfg.review_package}" if cfg.review_key else
           f"NOWHERE ({cfg.review_package or 'no package'}: no .review_key), so review answers 403"),
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
    ap.add_argument("--origin", action="append", default=None,
                    help="extra allowed Origin for PUT/POST/DELETE, e.g. https://<name>:8446 (repeatable)")
    args = ap.parse_args(argv)
    import uvicorn
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", stream=sys.stdout)
    cfg = build_config(args)
    if args.host in ("0.0.0.0", "::"):
        print("[WARN] listening on every interface; P1 expects 127.0.0.1 behind tailscale serve", flush=True)
    for line in startup_lines(cfg, args.host, args.port):
        print(line, flush=True)
    uvicorn.run(create_app(cfg), host=args.host, port=args.port, access_log=False, proxy_headers=False,
                server_header=False, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
