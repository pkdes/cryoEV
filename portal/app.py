"""
CryoEV portal, P1 shell (admin only, tailnet): one address, one sign-in, one look. Spec: P1_SHELL_SPEC.md (private).

    python -m portal.app --host 127.0.0.1 --port 8770 [--registry <dir>] [--review-upstream http://127.0.0.1:8766]
        [--review-package <dir with .review_key>] [--roster <path>] [--state-dir <path>] [--origin <url> ...]
    python -m portal.app --auth access --port 8448 [--access-config <path>] [same options]

Two sign-in modes, one app, never mixed on one port (--auth):
  key     (default) key links -> session cookie (portal/auth.py); tailnet. A Cf-Access-Jwt-Assertion header is ignored.
  access  Cloudflare Access (portal/access.py): ONLY the verified Cf-Access-Jwt-Assertion JWT counts; its email must
          be one active roster person's. Binds 127.0.0.1 and [::1] (cloudflared in front). /login answers 404, session
          cookies are ignored, admins are refused unless CJ_ACCESS_ADMIN=on, sandbox links are hidden, the bar gets a
          Sign out link (/cdn-cgi/access/logout), and writes need Origin = https://<configured host>.

Routes (everything except OPEN needs a session; see Gate):
  /                      home: platform status from <registry>/live/live.json + module cards from modules.json
  /about                 About CryoEV (every signed-in role): text from a private JSON file outside the repo
                         ($CJ_ABOUT_FILE, default ~/.cryoev/about.json; keys platform, credit, support, cite);
                         missing -> "[not set yet]" placeholders. Never under static/ (served without sign-in)
  /changes               What changed (every signed-in role): the repo's CHANGELOG.md ($CJ_CHANGELOG_FILE overrides)
                         as escaped HTML (render_changelog: headings, bullets, **bold**, `code`; no raw HTML)
  /login?k=<key>         key link -> session cookie -> relative redirect to /
  /healthz               {"ok": true}
  /cj/cj.css, brand.json, header.js, live.js, review.js, modules.json, me.json     the shell's own files
  /cj/version.json       {"version": the portal's stamp, "review_page": {"version", "updated"} of the real review
                         package's index.html, "review_tool": the same for the repo's review tool (prediction
                         packages)}: PAGE_VERSION parsed from the page ("unknown" without it) + the file's mtime
  /history/history.html, /history/live/live.json               served from the registry folder (allowlist only);
                         history.html gets the cj.css / header.js include lines before </head> (file untouched).
                         A hidden "classic" page: not in the nav, linked from home's footer.
  /review/               the review shell: the bar + a full-height same-origin iframe of /review/app/ (its query,
                         minus any k, goes into the iframe's src; review.js mirrors the frame's URL back)
  /review/app/*          proxy to the real review server (portal/proxy.py), upstream path = the part after /review/app
  /review/<pkg_id>/      a prediction review package (pred_ps_<8 digits>_<6 hex>) in the same shell; its tool at
                         /review/<pkg_id>/app/... -> <pred upstream>/<pkg_id>/... with the package's own key; owner or
                         admin only (pred_mount). Anything else under /review/ -> 404
The sandbox is NOT proxied (it shares the real package's localStorage key); modules.json links out to :8443.
Roles: plug-in modules declare ROLES (portal/modules); the core pages start from CORE_ROLES (review, history: admin)
and a valid modules.json entry may replace one (manifest_rules; fail closed). The rules are read once at start (an
unreadable modules.json stops the portal; a role change needs a restart). The Gate answers 403 to any other role.
A top-level document GET of /review/app/ (a copied tool link) answers 302 /review/?<query minus k>.

Defaults: --registry = data_utils.config.registry_root(); --review-package = the newest open revise job on port
8766 in that registry (its review_package path); roster and state dir under %USERPROFILE%\\.cryoev (portal/auth.py).
uvicorn runs with its access log OFF and proxy headers OFF; the app writes its own access log with `k` redacted.

Audit (Admin > Recent actions): after every POST/PUT/DELETE/PATCH, done or refused, the Gate queues one line in
<state dir>/audit/portal.jsonl (portal/audit.py Writer: one worker thread, never blocks or raises): actor, a readable
action from AUDIT_LABELS (keyed by the matched route template), a safe id target, {"status"}. Upload chunks, review
saves / claims and gallery marks: the first per person x action x target per 10 min, then a counter line. Plus the
first sign-in per person per day on this door. main() removes CRYOEV_VIA / CRYOEV_FOR (registry/actor.strip_env).
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import errno
import html
import importlib.util
import json
import logging
import os
import re
import socket
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
from starlette.routing import Match, Route  # noqa: E402

from data_utils import config as repo_config  # noqa: E402
from portal import audit as PA  # noqa: E402
from portal import auth as A  # noqa: E402
from portal import feedback_store as FEEDBACK  # noqa: E402
from portal import modules as M  # noqa: E402
from portal import proxy as P  # noqa: E402
from annotation import multi_review_server as MRS  # noqa: E402

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
PRED_PORT = 8767                                              # annotation/multi_review_server.py
PRED_ID_RE = re.compile(r"^pred_ps_\d{8}_[0-9a-f]{6}\Z")         # prediction review package ids (never "app")
OWNER_PATH_RE = re.compile(r"^/review/(?!app(?:/|$))[^/]+(?:/|$)")   # /review/<pkg id>...: owner-checked routes
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
    "gallery": '<rect x="4" y="4" width="7" height="7" rx="1.2"/><rect x="13" y="4" width="7" height="7" rx="1.2"/>'
               '<rect x="4" y="13" width="7" height="7" rx="1.2"/><path d="M13 20l3.2-4.2 2.2 2.4 1.6-2.2"/>',
    "results": '<path d="M5 20v-8M10 20V5M15 20v-6M20 20v-10"/>',
    "reviews": '<rect x="4" y="4" width="16" height="16" rx="2"/><path d="M8 9.5l1.5 1.5L12 8.5M14 10h3M8 15.5l1.5 1.5 2.5-2.5M14 16h3"/>',
    "train": '<path d="M19.5 13a7.5 7.5 0 01-13.3 4.2M4.5 11a7.5 7.5 0 0113.3-4.2"/><path d="M18 3.5v3.5h-3.5M6 20.5V17h3.5"/>',
    "admin": '<path d="M12 3l7 3v5.5c0 4.3-2.9 7.6-7 9.5-4.1-1.9-7-5.2-7-9.5V6z"/><path d="M9.5 12l1.8 1.8 3.4-3.5"/>',
}
EXT_ICON = '<svg class="cj-ext" viewBox="0 0 24 24" aria-hidden="true"><path d="M8 16L16 8M10 8h6v6"/></svg>'
REVIEW_PORT = 8766
PUBLIC_PORTS = {"", "443", "8443", "8445", "8446"}   # tailscale serve's https ports (links use https there)
HOST_RE = re.compile(r"[A-Za-z0-9.-]{1,253}|\[[0-9A-Fa-f:.]{2,45}\]")
AUTH_MODES = ("key", "access")       # key links (tailnet) | Cloudflare Access JWT (portal/access.py); never mixed
KEY_PORT = 8770
ACCESS_PORT = 8448
ACCESS_SIGN_IN = ("This address is reached through the lab's Cloudflare sign-in. If you signed in and still see this, "
                  "ask the admin to add your email to the portal.")
# ---- the Gate's audit catch-all (round 9 #17): one line per POST/PUT/DELETE/PATCH, successful or refused. The action
# is a readable name keyed by the matched route TEMPLATE (never the request path); the target is a safe id path
# parameter only (AUDIT_IDS, SAFE_ID_RE); detail = {"status"}. Never a body, key, token, cookie or session id.
_CONVERT = (("stop", "stopped"), ("retry", "retried"), ("inputs", "inputs changed"), ("stitch", "stitch run"),
            ("register", "registration run"), ("details", "details edited"), ("preview", "preview made"),
            ("reconvert", "reconverted"))
AUDIT_LABELS = {      # route template -> (action, frequent: first per 10 min written, then counted)
    "/predict/api/upload": ("upload started", False),
    "/predict/api/upload/{uid}/file": ("upload chunk", True),
    "/predict/api/upload/{uid}/create": ("prediction set created", False),
    "/predict/api/upload/{uid}/cancel": ("upload cancelled", False),
    "/predict/new/upload/": ("prediction set created", False),
    "/predict/new/session/{key}/": ("prediction set created", False),
    "/predict/new/raw/": ("conversion started", False),
    "/predict/new/raw/sessions/{key}/": ("conversion started", False),
    "/predict/preparing/{tid}/retry": ("prediction set retried", False),
    "/predict/{ps}/start": ("prediction set started", False),
    "/predict/{ps}/rename": ("prediction set renamed", False),
    "/predict/{ps}/review": ("review package made", False),
    "/predict/{ps}/results": ("results requested", False),
    **{f"/predict/convert/{{cid}}/{a}": (f"conversion {w}", False) for a, w in _CONVERT},
    "/results/{ps}/compute": ("results computed", False),
    "/gallery/{cid}/api/event": ("gallery mark", True),
    "/feedback/": ("feedback filed", False),
    "/feedback/{rid:int}/status": ("feedback status changed", False),
    "/feedback/{rid:int}/reply": ("feedback reply", False),
    "/data/session/{eid}/details": ("session details edited", False),
    "/train/check": ("training checked", False),
    "/train/queue": ("training queued", False),
    "/jobs/api/cancel": ("job cancel", False),
    "/jobs/api/release": ("job release", False),
    "/admin/people/add": ("person added", False),
    "/admin/people/role": ("role changed", False),
    "/admin/people/email": ("sign-in email changed", False),
    "/admin/people/key": ("key link made", False),
    "/admin/people/revoke": ("person revoked", False),
    "/admin/predict/label": ("Predict session renamed", False),
    "/admin/predict/show": ("Predict session shown / hidden", False),
    "/review/{rest:path}": ("review write (no such page)", False),
}
REVIEW_TOOL_TEMPLATES = frozenset({"/review/app/{path:path}", "/review/{pkg_id}/app/{path:path}"})
REVIEW_TOOL_ACTIONS = {"api/edits": ("review save", True), "api/claim": ("review claim", True)}   # review_server.py
# routes whose handler writes its own, richer line on success: the catch-all logs them only when refused / failed
SELF_LOGGED = frozenset({"/admin/people/add", "/admin/people/role", "/admin/people/email", "/admin/people/key",
                         "/admin/people/revoke", "/admin/predict/label", "/admin/predict/show", "/jobs/api/cancel",
                         "/jobs/api/release", "/train/queue"})
AUDIT_IDS = ("ps", "pkg_id", "cid", "rid", "eid", "uid", "tid")      # the only path parameters a line may carry
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}\Z")
SWEEP_S = 60.0                                                       # close due 10-min windows this often
ABOUT_ENV = "CJ_ABOUT_FILE"                                          # /about's private text (never under static/)
ABOUT_KEYS = (("platform", "The platform"), ("credit", "Who builds it"), ("support", "Support"),
              ("cite", "How to cite"))
ABOUT_UNSET = "[not set yet]"
CHANGELOG_ENV = "CJ_CHANGELOG_FILE"                                  # /changes' source (default: REPO/CHANGELOG.md)
PAGE_VERSION_RE = re.compile(r'\bPAGE_VERSION\s*=\s*"([0-9A-Za-z][0-9A-Za-z._-]{0,39})"')
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
    pred_upstream: str = f"http://127.0.0.1:{PRED_PORT}"
    review_root: Path | None = None              # <CryoAI-laptop>/review: the prediction packages
    auth: str = "key"                            # AUTH_MODES
    access: object = None                        # portal.access.AccessConfig (access mode only)
    access_admin: bool | None = None             # admins via Cloudflare; None = env CJ_ACCESS_ADMIN ("on" only)


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


def audit_label(template, params, method) -> tuple:
    """-> (readable action, frequent) for a write on `template` (None = no route matched). The review tool's
    proxied writes are told apart by their API path (a fixed set), never logged as a path."""
    if template in REVIEW_TOOL_TEMPLATES:
        return REVIEW_TOOL_ACTIONS.get(str((params or {}).get("path") or "").strip("/"), ("review tool write", False))
    if template is None:
        return "write to an unknown address", False
    return AUDIT_LABELS.get(template, (f"{method} {template}", False))


def audit_target(params) -> str | None:
    """The first safe id among the path parameters (AUDIT_IDS, SAFE_ID_RE), else None. "#<n>" for a feedback id."""
    for k in AUDIT_IDS:
        v = (params or {}).get(k)
        if v is not None and SAFE_ID_RE.match(str(v)) and ".." not in str(v):
            return f"#{v}" if k == "rid" else str(v)
    return None


def about_path() -> Path:
    return Path(os.environ.get(ABOUT_ENV) or (Path.home() / ".cryoev" / "about.json"))


def render_about(doc, version: str) -> str:
    """/about's body: each ABOUT_KEYS text (escaped; blank lines = paragraphs), ABOUT_UNSET when missing."""
    doc = doc if isinstance(doc, dict) else {}
    secs = []
    for key, head in ABOUT_KEYS:
        v = doc.get(key)
        text = v.strip() if isinstance(v, str) else ""
        paras = [p.strip() for p in text.split("\n\n") if p.strip()] or [ABOUT_UNSET]
        body = "".join(f'<p{" class=cj-mut" if p == ABOUT_UNSET else ""}>{html.escape(p)}</p>' for p in paras)
        secs.append(f'<section class="cj-about-sec" data-key="{key}"><h2>{html.escape(head)}</h2>{body}</section>')
    return (f'<div class="cj-about"><h1>About CryoEV</h1>{"".join(secs)}'
            f'<p class="cj-mut cj-about-ver">Portal version: {html.escape(version)} · '
            '<a href="/changes">What changed (changelog)</a></p>'
            '<p class="cj-mut cj-about-ver">Only signed-in people see this page.</p></div>')


def changelog_path() -> Path:
    return Path(os.environ.get(CHANGELOG_ENV) or (REPO / "CHANGELOG.md"))


def _md_inline(text: str) -> str:
    """Escaped text with `code` and **bold** (the markers survive html.escape; nothing else becomes markup)."""
    if text.count("`") % 2:                         # an unpaired backtick: no code spans at all
        text = text.replace("`", "'")
    out = []
    for i, part in enumerate(text.split("`")):
        if i % 2:
            out.append(f"<code>{html.escape(part)}</code>")
        else:
            out.append(re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html.escape(part)))
    return "".join(out)


def render_changelog(md: str) -> str:
    """CHANGELOG.md -> HTML, in file order (newest first by convention). A tiny subset: #/##/### headings, "- "/"* "
    bullets, paragraphs (blank-line separated), **bold**, `code`. Every character of text is escaped: raw HTML in the
    file shows as text."""
    out, para, items = [], [], []

    def flush():
        if para:
            out.append(f"<p>{_md_inline(' '.join(para))}</p>")
            para.clear()
        if items:
            out.append("<ul>" + "".join(f"<li>{_md_inline(t)}</li>" for t in items) + "</ul>")
            items.clear()
    for raw in md.splitlines():
        line = raw.strip()
        m = re.match(r"^(#{1,3})\s+(.*)$", line)
        if not line:
            flush()
        elif m:
            flush()
            n = len(m.group(1)) + 1                     # # -> h2 (the page's own h1 is the title)
            out.append(f'<h{n} class="cj-chg-h{n}">{_md_inline(m.group(2))}</h{n}>')
        elif re.match(r"^[-*]\s+", line):
            if para:
                flush()
            items.append(re.sub(r"^[-*]\s+", "", line))
        elif items and raw[:1] in (" ", "\t"):      # a bullet's wrapped continuation line
            items[-1] += " " + line
        else:
            if items:
                flush()
            para.append(line)
    flush()
    return "".join(out)


_PAGE_VER_CACHE: dict = {}


def page_version(path: Path | None) -> dict:
    """{"version": PAGE_VERSION in the page (else "unknown"), "updated": the file's mtime "YYYY-MM-DD HH:MM" (None when
    there is no file)}. Re-read only when the file's mtime/size change (one stat per call). Never shows a path."""
    try:
        st = Path(path).stat()
    except (OSError, TypeError):
        return {"version": "unknown", "updated": None}
    key = str(path)
    hit = _PAGE_VER_CACHE.get(key)
    if hit and hit[0] == (st.st_mtime_ns, st.st_size):
        return dict(hit[1])
    try:
        m = PAGE_VERSION_RE.search(Path(path).read_text(encoding="utf-8", errors="replace"))
    except OSError:
        m = None
    res = {"version": m.group(1) if m else "unknown",
           "updated": datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M")}
    _PAGE_VER_CACHE[key] = ((st.st_mtime_ns, st.st_size), res)
    return dict(res)


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

    def __init__(self, app: Starlette, auth: A.Auth, origins=(), rules=(), mode: str = "key", verifier=None,
                 access_admin: bool = False, writer=None, review_target=None):
        if mode not in AUTH_MODES or (mode == "access") != (verifier is not None):
            raise ValueError(f"auth mode {mode!r} needs {'a' if mode == 'access' else 'no'} JWT verifier")
        self.app = app
        self.auth = auth
        self.origins = tuple(origins)
        self.rules = list(rules)            # [(module PREFIX, roles)] from portal.modules.load, longest first
        self.routes = app.routes
        self.mode = mode                    # "key": key-link session cookie only; "access": Cloudflare JWT only
        self.verifier = verifier            # portal.access.Verifier in access mode
        self.access_admin = bool(access_admin)
        self.writer = writer                # portal.audit.Writer: the catch-all (None = no audit lines)
        self.review_target = review_target  # the real review package's id, the target of /review/app/ writes

    def route_of(self, scope) -> tuple:
        """-> (route template or None, path params) of this request. The endpoint the router chose when it ran (one
        route with that endpoint), else the router's own matching (a refused request never reached it): the first
        FULL match, else the first PARTIAL (wrong method)."""
        ep = scope.get("endpoint")
        if ep is not None:
            hits = [r for r in self.routes if getattr(r, "endpoint", None) is ep]
            if len(hits) == 1:
                return getattr(hits[0], "path", None), dict(scope.get("path_params") or {})
        partial = (None, {})
        probe = {**scope, "type": "http", "path_params": {}}
        for r in self.routes:
            try:
                m, child = r.matches(probe)
            except Exception:                           # noqa: BLE001  a route that can't match this scope
                continue
            if m == Match.FULL:
                return getattr(r, "path", None), dict(child.get("path_params") or {})
            if m == Match.PARTIAL and partial[0] is None:
                partial = (getattr(r, "path", None), dict(child.get("path_params") or {}))
        return partial

    def record(self, scope, user, status) -> None:
        """The audit catch-all, after the response: the day's first sign-in per person on this door, and one line
        per unsafe-method request (successful or refused). Only queues work (portal.audit.Writer); never raises."""
        w = self.writer
        if w is None:
            return
        try:
            door = PA.DOOR_NAMES.get(self.mode, self.mode)
            if scope.get("cj_login") and status == 303:
                w.signin(scope["cj_login"], door, "key link")
            elif user and user.get("id"):
                w.signin(user["id"], door, "session" if self.mode == "key" else "cloudflare")
            method = scope.get("method")
            if method not in UNSAFE:
                return
            st = int(status or 500)                     # no response started = the app raised
            tpl, params = self.route_of(scope)
            if tpl in SELF_LOGGED and st < 400:
                return
            action, frequent = audit_label(tpl, params, method)
            target = audit_target(params)
            if target is None and tpl == REVIEW_BASE + "/app/{path:path}":
                target = self.review_target
            actor = (user or {}).get("id") or "-"
            w.event(actor, action, target, {"status": st}, frequent=frequent or actor == "-")
        except Exception as e:                          # noqa: BLE001  the audit never breaks a request
            log.error("audit catch-all failed (%s)", type(e).__name__)

    async def access_user(self, scope):
        """Access mode: -> (roster person, None) or (None, the 403 page). Only the verified JWT counts; cookies and
        key links are never looked at. Refusals are logged with a reason and Cf-Connecting-IP, never the token."""
        from portal import access as AX
        vals = [v for k, v in scope.get("headers", []) if k.lower() == AX.HEADER]
        ip = AX.client_ip(header(scope, b"cf-connecting-ip"))
        if len(vals) != 1:
            claims, why = None, ("no Cf-Access-Jwt-Assertion header" if not vals else "repeated JWT header")
        else:
            claims, why = await self.verifier.verify(vals[0].decode("latin-1"))
        if claims is None:
            log.warning("access refused: %s (cf ip %s)", why, ip)
            return None, page(403, "Not signed in", ACCESS_SIGN_IN)
        email = A.access_email(claims["email"])          # ASCII + EMAIL_RE checked before lower-casing
        if email is None:
            log.warning("access refused: the email claim is not a plain ASCII address (cf ip %s)", ip)
            return None, page(403, "Not on the list", ACCESS_SIGN_IN)
        person = self.auth.roster.match_email(email)
        if person is None:
            log.warning("access refused: %r is not the email of one active person (cf ip %s)", email[:254], ip)
            return None, page(403, "Not on the list", f"You signed in as {email}, but that email is not on the "
                                                      "portal's list. Ask the admin to add your email.")
        if person.get("role") == "admin" and not self.access_admin:
            log.warning("access refused: %s is an admin and CJ_ACCESS_ADMIN is off (cf ip %s)", person.get("id"), ip)
            return None, page(403, "Admin is available on Tailscale only", "Admin is available on Tailscale only. "
                                                                            "Open the portal over Tailscale.")
        return person, None

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
            if path not in OPEN and self.mode == "access":
                user, deny = await self.access_user(scope)
            elif path not in OPEN:
                cookies = cookie_parser(header(scope, b"cookie") or "")
                state, user = self.auth.check(cookies.get(A.COOKIE))
                if state == "none":
                    deny = page(401, "Sign in", "Open your key link (the address ending in /login?k=...) on this "
                                                "device to sign in.")
                elif state != "ok":
                    deny = page(403, "Not signed in", "This sign-in is unknown, expired or revoked. Open a current "
                                                      "key link.")
            if (deny is None and user is not None and not owner_checked(path)     # pred packages: owner check
                    and not M.allowed(self.rules, path, user.get("role"))):
                deny = page(403, "Not allowed", "Your role can't open this part of the portal.")
            if deny is None and method in UNSAFE and not self.origin_allowed(scope):
                deny = page(403, "Refused", "Cross-origin request refused.")
            if deny is not None:
                return await deny(scope, receive, send_secure)
            scope["cj_user"] = user
            scope["cj_auth"] = self.mode
            await self.app(scope, receive, send_secure)
        finally:
            qs = redact_query(scope.get("query_string", b"").decode("latin-1"))
            client = (scope.get("client") or ("-",))[0]
            if self.mode == "access":
                from portal import access as AX
                client = f"{client} cf={AX.client_ip(header(scope, b'cf-connecting-ip'))}"
            who = (user or {}).get("id") or scope.get("cj_login") or "-"
            access_log.info('%s "%s %s%s" %s %s', client, scope.get("method"), scope.get("path"),
                            f"?{qs}" if qs else "", box["status"], who)
            self.record(scope, user, box["status"])

    def origin_allowed(self, scope) -> bool:
        """Key mode: origin_ok (this portal's own scheme://Host, or an --origin). Access mode: ONLY the configured
        public origin https://<host> (and any --origin), whatever the Host header says."""
        if self.mode == "access":
            o = header(scope, b"origin")
            return bool(o) and o in self.origins
        return origin_ok(scope, self.origins)


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
    brand = read_json(HERE / "static" / "brand.json") or {"name": "CryoEV"}
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
        if m.get("external") and getattr(cfg, "auth", "key") == "access":    # host:port links don't exist through the tunnel
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
    return fill(tpl, {"heading": esc(brand.get("name", "CryoEV")), "tagline": esc(brand.get("tagline", "")),
                      "user": esc(user.get("id", "")), "status": status, "module_cards": "".join(cards),
                      "footer": footer})


def shell_response(request: Request, base: str, title: str = "Review") -> HTMLResponse:
    """The review shell page for the tool at base/app/: the bar + an iframe (the shell's query, minus k, goes along)."""
    app_prefix = base + "/app"
    q = P.strip_k(request.scope.get("query_string", b"").decode("latin-1"))
    src = app_prefix + "/" + (f"?{q}" if q else "")
    body = (f'<iframe id="cj-review-frame" class="cj-review-frame" title="{html.escape(title)} tool" '
            f'src="{html.escape(src)}" data-base="{html.escape(base)}/" data-app="{html.escape(app_prefix)}/" '
            f'allow="clipboard-write; fullscreen"></iframe>\n<script src="/cj/review.js" defer></script>')
    return M.Ctx.page(request, title, body, active="review")


def wants_shell(request: Request, base: str) -> bool:
    """A browser opening the tool's page (base/app/) as the whole window, not in the shell's frame."""
    app_prefix = base + "/app"
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


def shell_redirect(request: Request, base: str) -> Response:
    q = P.strip_k(request.scope.get("query_string", b"").decode("latin-1"))
    r = RedirectResponse(base + "/" + (f"?{q}" if q else ""), status_code=302)
    r.headers["Cache-Control"] = "no-store"
    return r


def review_mount(base: str, proxy: P.ReviewProxy, title: str = "Review") -> list:
    """The routes for one review under `base` (no trailing slash):
         base/          the shell page: the bar + an iframe of base/app/ (the shell's query, minus k, goes along)
         base/app       -> base/app/
         base/app/...   the tool, through `proxy` (whose prefix must be base + "/app"). A top-level document GET of
                        the tool's page (a copied link) is sent to the shell instead: 302 base/?<query minus k>
    The real review is review_mount("/review", ...); prediction packages are pred_mount (/review/<pkg_id>/)."""
    app_prefix = base + "/app"
    if proxy.prefix != app_prefix:
        raise ValueError(f"review proxy prefix {proxy.prefix!r} != {app_prefix!r}")

    async def shell(request: Request):
        return shell_response(request, base, title)

    async def tool(request: Request):
        if wants_shell(request, base):
            return shell_redirect(request, base)
        return await proxy.handle(request, request.scope["cj_user"]["id"])

    async def to_app(request: Request):
        return RedirectResponse(app_prefix + "/", status_code=307)

    return [Route(base + "/", shell, methods=["GET"]),
            Route(app_prefix, to_app, methods=["GET"]),
            Route(app_prefix + "/{path:path}", tool, methods=["GET", "HEAD", "PUT", "POST", "DELETE"])]


def pred_package(root: Path | None, pkg_id: str):
    """(package folder, owner id) for a prediction review package id, else None. The id must match PRED_ID_RE; the
    folder is found with the multi server's own check (well-formed id, complete package, reached under exactly that
    name: annotation/multi_review_server.package_dir); the owner comes from its review_package.json."""
    if root is None or not isinstance(pkg_id, str) or not PRED_ID_RE.match(pkg_id):
        return None
    try:
        pkg = MRS.package_dir(Path(root), pkg_id)
    except OSError:
        return None
    if pkg is None:
        return None
    rec = read_json(pkg / MRS.COMPLETE)
    owner = rec.get("owner") if isinstance(rec, dict) else None
    return (pkg, owner) if isinstance(owner, str) and owner else None


def owner_checked(path: str) -> bool:
    """/review/<pkg id>/... (any segment but app): the Gate's role rule for /review (the real package) doesn't
    apply; pred_mount checks the id and the owner, and anything else under /review/ is the 404 catch-all, so every
    role gets the same 404 for an id that isn't theirs or isn't an id. Never for a path with dot segments."""
    return bool(OWNER_PATH_RE.match(path)) and not any(s in (".", "..") for s in path.split("/"))


def pred_mount(proxy: P.ReviewProxy, root: Path | None, title: str = "Review") -> list:
    """Prediction review packages in the shell, one URL space per package:
         /review/<pkg_id>/          the shell page (as /review/)
         /review/<pkg_id>/app       -> /review/<pkg_id>/app/
         /review/<pkg_id>/app/...   the multi-package review server at <upstream>/<pkg_id>/..., with that package's
                                    own .review_key as X-Review-Key; every proxy rule applies (portal/proxy.py)
    Who: ctx.can_see(user, owner) (admins all, others their own). Unknown, malformed or not yours: one 404 page."""

    def lookup(request: Request):
        pid = request.path_params.get("pkg_id", "")
        hit = pred_package(root, pid)
        if hit is None or not M.Ctx.can_see(request.scope.get("cj_user"), hit[1]) or not MRS.package_key(hit[0]):
            return None, None                           # unknown, not yours, or no key (not served): one answer
        return pid, hit[0]

    def missing():
        return page(404, "Not found", "No such review package, or it isn't yours.")

    async def shell(request: Request):
        pid, _ = lookup(request)
        return missing() if pid is None else shell_response(request, f"{REVIEW_BASE}/{pid}", title)

    async def to_app(request: Request):
        pid, _ = lookup(request)
        return missing() if pid is None else RedirectResponse(f"{REVIEW_BASE}/{pid}/app/", status_code=307)

    async def tool(request: Request):
        pid, pkg = lookup(request)
        if pid is None:
            return missing()
        base = f"{REVIEW_BASE}/{pid}"
        if wants_shell(request, base):
            return shell_redirect(request, base)
        key = MRS.package_key(pkg)
        if not key:                                     # the key went away since lookup
            return missing()
        return await proxy.handle(request, request.scope["cj_user"]["id"], key=key, prefix=base + "/app",
                                  up_path="/" + pid)

    return [Route(REVIEW_BASE + "/{pkg_id}/", shell, methods=["GET"]),
            Route(REVIEW_BASE + "/{pkg_id}/app", to_app, methods=["GET"]),
            Route(REVIEW_BASE + "/{pkg_id}/app/{path:path}", tool, methods=["GET", "HEAD", "PUT", "POST", "DELETE"])]


VERSION_UNKNOWN = "version unknown"


def version_stamp(repo: Path = REPO) -> str:
    """"Updated <date time> . <short commit>" of the repo the portal runs from; computed once at startup. Never shows
    a path. git missing / slow / not a repo -> "version unknown"."""
    import subprocess
    try:
        r = subprocess.run(["git", "-C", str(repo), "log", "-1", "--format=%h|%cI"], capture_output=True, text=True,
                           timeout=3, check=True)
        sha, _, iso = r.stdout.strip().partition("|")
        when = datetime.fromisoformat(iso.strip()).strftime("%Y-%m-%d %H:%M")
        if not re.fullmatch(r"[0-9a-f]{4,40}", sha):
            return VERSION_UNKNOWN
        return f"Updated {when} · {sha}"
    except Exception:                                   # noqa: BLE001  any failure: a quiet fallback
        return VERSION_UNKNOWN


def create_app(cfg: Config) -> Gate:
    """Raises ValueError when modules.json can't be read (the Gate's rules come from it), or for an access mode
    without its config: refuse to start."""
    if cfg.auth not in AUTH_MODES:
        raise ValueError(f"unknown auth mode {cfg.auth!r}")
    access = cfg.auth == "access"
    verifier, origins, access_admin = None, tuple(cfg.origins), False
    if access:
        if cfg.access is None:
            raise ValueError("access mode needs the Cloudflare access config (team, aud, host)")
        from portal import access as AX                 # PyJWT: imported in access mode only
        verifier = AX.Verifier(cfg.access)
        origins += (cfg.access.origin,)
        access_admin = cfg.access_admin if cfg.access_admin is not None else AX.admin_allowed()
    auth = A.Auth(cfg.roster, cfg.state_dir)
    version = version_stamp()
    rules_box: list = []                            # the Gate's rules, filled below; home asks them, not the file
    proxy = P.ReviewProxy(cfg.review_upstream, cfg.review_key, cfg.review_timeout, cfg.max_body,
                          prefix=REVIEW_BASE + "/app")
    pred_proxy = P.ReviewProxy(cfg.pred_upstream, None, cfg.review_timeout, cfg.max_body)   # key + prefix per call

    def no_store(r: Response) -> Response:
        r.headers["Cache-Control"] = "no-store"
        return r

    async def home(request: Request):
        brand = read_json(HERE / "static" / "brand.json") or {}
        body = render_home(request, cfg, request.scope["cj_user"], rules_box)
        return M.Ctx.page(request, str(brand.get("name") or "CryoEV"), body, active="home")

    async def login(request: Request):
        if access:                                      # key links are off on the Cloudflare port
            return page(404, "Not found", "Key links don't work on this address. Sign in through Cloudflare.")
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
        mods = [m for m in manifest(cfg.manifest) if role in (m.get("roles") or [])
                and not (access and m.get("external"))]            # no host:port links through the tunnel
        return no_store(JSONResponse({"schema": "cj_modules/1", "modules": mods}))

    def version_json(request: Request):                 # sync: a stat (and a read when a page changed)
        pkg_page = (Path(cfg.review_package) / "index.html") if cfg.review_package else None
        return no_store(JSONResponse({"version": version, "review_page": page_version(pkg_page),
                                      "review_tool": page_version(MRS.TOOL_PAGE)}))

    async def me(request: Request):
        u = request.scope["cj_user"]
        out = {"id": u["id"], "role": u["role"]}
        if access:
            out["signout"] = "/cdn-cgi/access/logout"                 # Cloudflare's own sign-out (header.js)
        return no_store(JSONResponse(out))

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

    def about(request: Request):                    # sync: Starlette runs it in the threadpool (a file read)
        doc = read_json(about_path())
        return M.Ctx.page(request, "About CryoEV", render_about(doc, version), active="about")

    def changes(request: Request):                  # sync: a file read
        try:
            md = changelog_path().read_text(encoding="utf-8")
        except OSError:
            md = ""
        body = render_changelog(md) or '<p class="cj-mut">No changelog yet.</p>'
        return M.Ctx.page(request, "What changed", '<div class="cj-about cj-changes"><h1>What changed</h1>'
                          f'<p class="cj-mut">Portal: {html.escape(version)}</p>{body}</div>', active="about")

    routes = [
        Route("/", home, methods=["GET"]),
        Route("/about", about, methods=["GET"]),
        Route("/changes", changes, methods=["GET"]),
        Route("/login", login, methods=["GET"]),
        Route("/healthz", healthz, methods=["GET"]),
        *[Route(f"/cj/{n}", static(n), methods=["GET"]) for n in STATIC],
        Route("/cj/modules.json", modules, methods=["GET"]),
        Route("/cj/me.json", me, methods=["GET"]),
        Route("/cj/version.json", version_json, methods=["GET"]),
        Route("/history", redirect("/history/"), methods=["GET"]),
        Route("/history/", redirect("history.html"), methods=["GET"]),
        Route("/history/history.html", history_html, methods=["GET"]),
        Route("/history/live/live.json", live_json, methods=["GET"]),
        Route("/history/{rest:path}", history_other, methods=["GET"]),
        Route(REVIEW_BASE, redirect(REVIEW_BASE + "/"), methods=["GET"]),
        *review_mount(REVIEW_BASE, proxy),
        *pred_mount(pred_proxy, cfg.review_root),
        # anything else under /review/: 404 (the same answer as an unknown or not-yours package)
        Route(REVIEW_BASE + "/{rest:path}", review_free, methods=["GET", "HEAD", "PUT", "POST", "DELETE"]),
    ]
    mod_routes, rules = M.load(M.Ctx(cfg))          # portal/modules/<id>.py plug-ins (BUILD_CONTRACTS.md section 1)
    routes.extend(mod_routes)
    # core pages (review, history): built-in defaults, replaced only by valid modules.json entries; a plug-in's own
    # ROLES win for its PREFIX. Read ONCE here: a role change in modules.json needs a restart.
    core = manifest_rules(load_manifest(cfg.manifest), {pr for pr, _ in rules})      # ValueError: refuse to start
    rules = sorted(rules + core, key=lambda pr: -len(pr[0]))
    rules_box[:] = rules

    writer = PA.Writer(cfg.state_dir)               # the Gate's audit catch-all (one worker thread)

    async def sweeper():
        while True:
            await asyncio.sleep(SWEEP_S)
            try:
                writer.sweep()                       # memory only: due windows -> queued counter lines
            except Exception as e:                  # noqa: BLE001
                log.error("audit sweep failed (%s)", type(e).__name__)

    @contextlib.asynccontextmanager
    async def lifespan(app):
        await proxy.start()
        await pred_proxy.start()
        sweep = asyncio.create_task(sweeper())
        try:
            yield
        finally:
            sweep.cancel()
            await proxy.close()
            await pred_proxy.close()
            await asyncio.get_running_loop().run_in_executor(None, writer.close)   # open windows -> counter lines

    star = Starlette(routes=routes, lifespan=lifespan)
    star.router.redirect_slashes = False
    pkg_name = cfg.review_package.name if cfg.review_package else None
    gate = Gate(star, auth, origins, rules, mode=cfg.auth, verifier=verifier, access_admin=access_admin,
                writer=writer, review_target=pkg_name if pkg_name and SAFE_ID_RE.match(pkg_name) else None)
    gate.proxy = proxy
    gate.pred_proxy = pred_proxy
    gate.auth = auth
    return gate


def default_review_root() -> Path | None:
    root = repo_config.laptop_root(required=False)
    return Path(root) / "review" if root else None


def build_config(args) -> Config:
    """Raises ValueError (access.ConfigError, or PyJWT missing) when access mode can't get its config."""
    registry = args.registry if args.registry else repo_config.registry_root(required=False)
    pkg = args.review_package or registry_review_package(registry)
    key = read_review_key(pkg)
    mode = getattr(args, "auth", None) or "key"
    access = None
    if mode == "access":
        try:
            from portal import access as AX
        except ImportError as e:
            raise ValueError(f"access mode needs PyJWT + cryptography (pip install -r requirements-portal.txt): "
                             f"{e}") from None
        access = AX.load_config(getattr(args, "access_config", None))
    return Config(registry=Path(registry) if registry else None, roster=args.roster or A.default_roster(),
                  state_dir=args.state_dir or A.default_state_dir(), review_upstream=args.review_upstream,
                  review_key=key, review_package=pkg, origins=tuple(args.origin or ()),
                  manifest=getattr(args, "modules", None) or MANIFEST,
                  pred_upstream=getattr(args, "pred_upstream", None) or f"http://127.0.0.1:{PRED_PORT}",
                  review_root=getattr(args, "review_root", None) or default_review_root(),
                  auth=mode, access=access)


def startup_lines(cfg: Config, host: str, port: int) -> list:
    """What runs, printed at start (no secrets)."""
    n = len(A.Roster(cfg.roster).people())
    access = cfg.auth == "access"
    if access:
        from portal import access as AX
        admin = cfg.access_admin if cfg.access_admin is not None else AX.admin_allowed()
        head = [
            f"[cj] CryoEV portal, Cloudflare ACCESS mode, on http://127.0.0.1:{port} and http://[::1]:{port} "
            f"(cloudflared in front; public origin {cfg.access.origin})",
            "[cj] sign-in: ONLY a verified Cf-Access-Jwt-Assertion JWT (RS256, aud, iss "
            f"{cfg.access.issuer}, exp/nbf leeway {AX.LEEWAY_S} s); key links (/login) 404, session cookies ignored",
            f"[cj] keys: {cfg.access.certs_url} (worker thread, {AX.FETCH_TIMEOUT_S:g} s timeout, refetch at most "
            f"once per {AX.REFETCH_S:g} s); no keys = everyone refused",
            "[cj] who: the JWT email must be the email of exactly one active roster person; admins "
            + ("ALLOWED (CJ_ACCESS_ADMIN=on)" if admin else "refused (CJ_ACCESS_ADMIN off: Admin is Tailscale only)"),
            f"[cj] gate: deny by default; open without sign-in: {', '.join(sorted(OPEN))}; Origin on "
            f"{', '.join(sorted(UNSAFE))} must be {cfg.access.origin}"
            + (f" or {', '.join(cfg.origins)}" if cfg.origins else ""),
        ]
    else:
        head = [
            f"[cj] CryoEV portal on http://{host}:{port} (put tailscale serve :8446 in front)",
            f"[cj] gate: deny by default; open without a session: {', '.join(sorted(OPEN))}; "
            f"Origin check on {', '.join(sorted(UNSAFE))}",
        ]
    return head + [
        f"[cj] roster: {cfg.roster} ({n} people){'' if cfg.roster.is_file() else ' MISSING: nobody can sign in'}",
        (f"[cj] sessions: {cfg.state_dir / 'sessions.json'} (sha256 of session ids only)" if not access else
         "[cj] sessions: none (every request carries its own JWT)"),
        f"[cj] history: {cfg.registry or 'registry NOT FOUND'} (history.html + live/live.json only)",
        f"[cj] feedback: {FEEDBACK.feedback_path(cfg.state_dir)}",
        f"[cj] review: /review/ = shell, /review/app/ -> {cfg.review_upstream}; key from "
        + (f"{cfg.review_package}" if cfg.review_key else
           f"NOWHERE ({cfg.review_package or 'no package'}: no .review_key), so review answers 403"),
        "[cj] roles (read once, from built-in defaults + modules.json): "
        + "; ".join(f"{p} {'/'.join(r)}" for p, r in manifest_rules(manifest(cfg.manifest)))
        + "; plug-in modules: their own ROLES. Restart the portal after changing roles in modules.json",
        f"[cj] prediction packages: /review/<pred_ps_...>/ = shell, /review/<id>/app/ -> {cfg.pred_upstream}/<id>/; "
        f"keys from {cfg.review_root or 'NOWHERE (no review root): every package answers 404'}; owner or admin only",
        "[cj] sandbox: not proxied (link out to :8443)" if not access else
        "[cj] sandbox: hidden (its host:port links don't exist through the tunnel)",
        "[cj] logs: uvicorn access log off, proxy headers off; own access log with k= redacted"
        + ("; refusals logged with reason + Cf-Connecting-IP, never the token" if access else ""),
        f"[cj] audit: {PA.audit_path(cfg.state_dir)}: every {'/'.join(sorted(UNSAFE))} (done or refused; route "
        f"name + safe ids only), frequent saves counted per {PA.WINDOW_S // 60:g} min, the first sign-in per person "
        "per day; written off the event loop",
        f"[cj] about: /about (signed in) from {about_path()}"
        + ("" if about_path().is_file() else " (missing: placeholders shown)"),
        f"[cj] changes: /changes (signed in) from {changelog_path()}"
        + ("" if changelog_path().is_file() else " (missing: empty page)"),
    ]


def loopback_sockets(port: int) -> list:
    """Listening sockets on 127.0.0.1 AND [::1] (never a wildcard address), for access mode: cloudflared may resolve
    localhost either way. No IPv6 loopback on this machine = a warning and 127.0.0.1 only; any other bind error
    (e.g. the port is taken) raises OSError after closing what was opened."""
    out = []
    no_v6 = {errno.EADDRNOTAVAIL, errno.EAFNOSUPPORT, 10047, 10049}       # + the Windows WSAE* numbers
    for fam, addr in ((socket.AF_INET, "127.0.0.1"), (socket.AF_INET6, "::1")):
        s = None
        try:
            if fam == socket.AF_INET6 and not socket.has_ipv6:
                raise OSError(errno.EAFNOSUPPORT, "no IPv6")
            s = socket.socket(fam, socket.SOCK_STREAM)
            if fam == socket.AF_INET6:
                s.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
            if os.name == "nt":                     # Windows SO_REUSEADDR would let another process share the port
                s.setsockopt(socket.SOL_SOCKET, getattr(socket, "SO_EXCLUSIVEADDRUSE", socket.SO_REUSEADDR), 1)
            else:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((addr, port))
            s.setblocking(False)
        except OSError as e:
            if s is not None:
                s.close()
            if fam == socket.AF_INET6 and e.errno in no_v6:
                print(f"[WARN] no IPv6 loopback here ({e}); listening on 127.0.0.1:{port} only", flush=True)
                continue
            for o in out:
                o.close()
            raise
        out.append(s)
    return out


def serve_sockets(server, sockets) -> None:
    """Run uvicorn on sockets we bound ourselves (a seam for the tests)."""
    server.run(sockets=sockets)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--auth", choices=AUTH_MODES, default="key",
                    help="key = key links (tailnet, default); access = Cloudflare Access JWT only (portal/access.py)")
    ap.add_argument("--access-config", type=Path, default=None,
                    help="access mode config (default ~/.cryoev/cloudflare_access.json; env CJ_ACCESS_* override it)")
    ap.add_argument("--host", default="127.0.0.1",
                    help="bind address in key mode (default 127.0.0.1; tailscale serve fronts it). Access mode always "
                         "binds 127.0.0.1 and [::1]")
    ap.add_argument("--port", type=int, default=None, help=f"default {KEY_PORT} (key) / {ACCESS_PORT} (access)")
    ap.add_argument("--registry", type=Path, default=None, help="registry folder (default: data_utils.config)")
    ap.add_argument("--review-upstream", default=f"http://127.0.0.1:{REVIEW_PORT}")
    ap.add_argument("--review-package", type=Path, default=None,
                    help="the real review package (its .review_key is sent upstream); default: from the registry")
    ap.add_argument("--roster", type=Path, default=None, help=f"default {A.default_roster()}")
    ap.add_argument("--state-dir", type=Path, default=None, help=f"default {A.default_state_dir()}")
    ap.add_argument("--pred-upstream", default=f"http://127.0.0.1:{PRED_PORT}",
                    help="the multi-package review server (prediction packages)")
    ap.add_argument("--review-root", type=Path, default=None,
                    help="folder of the prediction review packages (default <CryoAI-laptop>/review)")
    ap.add_argument("--modules", type=Path, default=None,
                    help="the module manifest (default portal/modules.json); read once for the roles; unreadable = no start")
    ap.add_argument("--origin", action="append", default=None,
                    help="extra allowed Origin for PUT/POST/DELETE, e.g. https://<name>:8446 (repeatable)")
    args = ap.parse_args(argv)
    access = args.auth == "access"
    port = args.port or (ACCESS_PORT if access else KEY_PORT)
    import uvicorn
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", stream=sys.stdout)
    gone = PA.actor_module().strip_env()             # here, never at import: the CLI imports portal modules.
    if gone:                                         # A server never credits people's actions to an agent, and
                                                     # the children it spawns inherit the clean env
        print(f"[cj] removed {', '.join(gone)} from this server's environment (agent attribution is for CLI runs "
              "only)", flush=True)
    try:
        cfg = build_config(args)
    except ValueError as e:                          # access mode without its config: no start
        print(f"[ERROR] refusing to start: {e}", flush=True)
        return 2
    if not access and args.host in ("0.0.0.0", "::"):
        print("[WARN] listening on every interface; P1 expects 127.0.0.1 behind tailscale serve", flush=True)
    if access and args.host != "127.0.0.1":
        print(f"[WARN] --host {args.host} ignored: access mode binds 127.0.0.1 and [::1] only", flush=True)
    try:
        app = create_app(cfg)
    except ValueError as e:                          # e.g. modules.json unreadable: no guessed roles, no start
        print(f"[ERROR] refusing to start: {e}", flush=True)
        return 2
    for line in startup_lines(cfg, args.host, port):
        print(line, flush=True)
    if not access:
        uvicorn.run(app, host=args.host, port=port, access_log=False, proxy_headers=False,
                    server_header=False, log_level="info")
        return 0
    try:
        socks = loopback_sockets(port)
    except OSError as e:
        print(f"[ERROR] refusing to start: can't listen on loopback port {port}: {e}", flush=True)
        return 2
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, access_log=False, proxy_headers=False,
                                           server_header=False, log_level="info"))
    serve_sockets(server, socks)
    return 0


if __name__ == "__main__":
    sys.exit(main())
