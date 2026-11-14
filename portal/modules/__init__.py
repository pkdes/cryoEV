"""
Portal module plug-ins (contract: BUILD_CONTRACTS.md section 1, private).

Each portal/modules/<id>.py defines:
    PREFIX = "/jobs"            # the URL space it owns: pages at PREFIX/, JSON at PREFIX/api/...
    ROLES = ("admin",)          # roles allowed anywhere under PREFIX (Gate enforces; 403 otherwise)
    def routes(ctx) -> list[Route]

`load(ctx)` imports every module file, checks the declarations, adds the PREFIX -> PREFIX/ redirect, and returns
(routes, rules) where rules = [(prefix, roles), ...] longest prefix first, for the Gate.

A nested PREFIX (round 13: "/reviews/start", portal/modules/reviewstart.py) is one more segment under another
module's PREFIX; that module must be loaded too, the nested ROLES must be a subset of the parent's, and no parent
route may equal or start with the nested PREFIX. Its routes come first, so the parent's patterns (e.g. /reviews/{id})
never shadow it; its own ROLES decide under it (the Gate's longest prefix wins).
"""
from __future__ import annotations

import html
import importlib
import pkgutil
import re
from dataclasses import dataclass
from pathlib import Path

from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from starlette.routing import Route

HERE = Path(__file__).resolve().parent
TEMPLATES = HERE.parent / "templates"
PREFIX_RE = re.compile(r"^/[a-z][a-z0-9-]*(?:/[a-z][a-z0-9-]*)?$")   # one segment, or one nested under a module's
# /api stays free: the agent API (round 10) opens existing routes (portal/app.py BEARER_ROUTES), no module owns /api
RESERVED = {"/", "/login", "/healthz", "/cj", "/history", "/review", "/api"}


def no_store(r: Response) -> Response:
    r.headers["Cache-Control"] = "no-store"
    return r


@dataclass
class Ctx:
    cfg: object

    @staticmethod
    def user(request: Request) -> dict:
        return request.scope["cj_user"]

    @staticmethod
    def can_see(user: dict, owner_id) -> bool:
        """Admins see everything; everyone else only what they own."""
        return bool(user) and (user.get("role") == "admin" or (owner_id is not None and user.get("id") == owner_id))

    @staticmethod
    def no_store(r: Response) -> Response:
        return no_store(r)

    @staticmethod
    def json(data, status: int = 200) -> JSONResponse:
        return no_store(JSONResponse(data, status_code=status))

    @staticmethod
    def page(request: Request, title: str, body_html: str, active: str = "") -> HTMLResponse:
        """Shell page: base.html with the cj bar (header.js) and cj.css. `body_html` must already be escaped."""
        tpl = (TEMPLATES / "base.html").read_text(encoding="utf-8")
        out = (tpl.replace("{{title}}", html.escape(title)).replace("{{active}}", html.escape(active))
               .replace("{{body}}", body_html))
        return no_store(HTMLResponse(out))


def _redirect(to: str):
    async def go(request: Request):
        return RedirectResponse(to, status_code=307)
    return go


def discover() -> list:
    return sorted(m.name for m in pkgutil.iter_modules([str(HERE)]) if not m.name.startswith("_"))


def load(ctx: Ctx, names=None):
    """-> (routes, rules). Raises ValueError on a bad declaration (fail loudly at start, not at request time)."""
    from portal import auth as A
    routes, rules, seen, nested = [], [], set(), []
    for name in (discover() if names is None else names):
        mod = importlib.import_module(f"portal.modules.{name}")
        prefix, roles = getattr(mod, "PREFIX", None), tuple(getattr(mod, "ROLES", ()) or ())
        if not isinstance(prefix, str) or not PREFIX_RE.match(prefix) or prefix in RESERVED or prefix in seen \
                or "/" + prefix.split("/")[1] in RESERVED:
            raise ValueError(f"portal module {name}: bad or duplicate PREFIX {prefix!r}")
        if not roles or any(r not in A.ROLES for r in roles):
            raise ValueError(f"portal module {name}: ROLES must be a non-empty subset of {A.ROLES}")
        mod_routes = list(mod.routes(ctx))
        for r in mod_routes:
            if not (r.path == prefix + "/" or r.path.startswith(prefix + "/")):
                raise ValueError(f"portal module {name}: route {r.path} is outside {prefix}/")
        seen.add(prefix)
        own = [Route(prefix, _redirect(prefix + "/"), methods=["GET"])] + mod_routes
        if prefix.count("/") > 1:
            nested.append((name, prefix, roles, own))
        else:
            routes.extend(own)
        rules.append((prefix, roles))
    by_prefix = dict(rules)
    for name, prefix, roles, own in reversed(nested):
        parent = "/" + prefix.split("/")[1]
        if parent not in seen:
            raise ValueError(f"portal module {name}: nested PREFIX {prefix!r} needs the module that owns its parent")
        if not set(roles) <= set(by_prefix[parent]):     # never opens a door the parent module keeps shut
            raise ValueError(f"portal module {name}: nested PREFIX {prefix!r} ROLES must be within {parent}'s")
        clash = [r.path for r in routes if getattr(r, "path", "") == prefix or
                 getattr(r, "path", "").startswith(prefix + "/")]
        if clash:                                         # the parent may not already route into the nested space
            raise ValueError(f"portal module {name}: {parent}'s routes {clash} fall inside nested PREFIX {prefix!r}")
        routes[:0] = own                                 # before every one-segment module's routes
    rules.sort(key=lambda pr: -len(pr[0]))
    return routes, rules


def allowed(rules, path: str, role) -> bool:
    """Gate check: the longest matching PREFIX decides; paths outside every module PREFIX pass (core routes)."""
    for prefix, roles in rules:
        if path == prefix or path.startswith(prefix + "/"):
            return role in roles
    return True
