"""
The Data routes (routes(ctx)); the package re-exports routes for the module loader.

Part of the Data module (portal/modules/data/__init__.py has the rules); moved from data.py unchanged.
"""
from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import RedirectResponse, Response
from starlette.routing import Route

from portal import upload_sessions as US
from portal.modules.data.records import (
    DETAIL_KINDS, KIND_TITLES, LIST_KINDS, PREFIX, SCHEMA, Viewer, esc, inside, known_roots, _lock, overview,
    read_registry, _registry_cache, rows_for, search, thumb_version, upload_roots, with_disk_packages)
from portal.modules.data.details import detail, make_thumb, thumbnail_sources
from portal.modules.data.lineage import lineage_tree
from portal.modules.data.graphs import (
    SCRIPT, graph_focus_key, history_registry, item_graph, lab_graph, source_focus, source_graph)
from portal.modules.data.render import (
    breadcrumb, page_body, render_detail, render_lab_graph, render_lineage, render_list, render_overview,
    render_search)


# ---------------------------------------------------------------- routes
def routes(ctx) -> list:
    def state(request: Request):
        records, error = read_registry(getattr(ctx.cfg, "registry", None))
        return with_disk_packages(records), error, Viewer(ctx.can_see, ctx.user(request), ctx.cfg)

    def requested_page(request: Request, default=None):
        value = request.query_params.get("page")
        if value is None:
            return default
        return int(value) if value.isdigit() and len(value) < 6 else 1

    def html_page(request, title, content, status=200):
        response = ctx.page(request, title, page_body(content), active="data")
        response.status_code = status
        return response

    def missing(request, what="That item"):
        return html_page(request, "Not found", breadcrumb(("Data", PREFIX + "/")) + "<h1>Not found</h1>"
                         f'<p class="cj-mut">{esc(what)} is not in the catalog, or you do not have access to it.</p>',
                         404)

    def index(request: Request):
        records, error, viewer = state(request)
        return html_page(request, "Data", render_overview(overview(records, viewer), error))

    def find(request: Request):
        records, error, viewer = state(request)
        return html_page(request, "Search",
                         render_search(search(records, request.query_params.get("q", ""), viewer), error))

    def listing(request: Request):
        kind = request.path_params["kind"]
        if kind not in LIST_KINDS:
            return missing(request, "That list")
        records, error, viewer = state(request)
        return html_page(request, KIND_TITLES[kind], render_list(KIND_TITLES[kind], rows_for(records, kind, viewer),
                                                                 error))

    def details_html(records, d, viewer, problem=None, saved=False) -> str:
        """The Session details form for its uploader / the admin (when switched on); else the read-only view."""
        up = d.get("upload")
        if not up or not up["can_edit"]:
            return ""
        record = records["session"].get(d["id"]) or {}
        return US.details_form(record, f'{PREFIX}/session/{quote(d["id"], safe="")}/details', problem, saved,
                               viewer.user)

    def item(kind):
        def handler(request: Request):
            records, error, viewer = state(request)
            page = requested_page(request, None if kind == "session" else 1)
            d = detail(records, kind, request.path_params["eid"], viewer, page)
            if d is None:
                return missing(request)
            extra = details_html(records, d, viewer, saved=request.query_params.get("saved") == "1") \
                if kind == "session" else ""
            return html_page(request, d["title"], render_detail(d, error, extra))
        return handler

    async def session_details(request: Request):
        """POST the Session details form: the session's uploader or the admin, an upload session, switched on."""
        user, sid = ctx.user(request), request.path_params["eid"]
        record = await run_in_threadpool(US.current_record, ctx.cfg, sid)
        if not record or not US.may_edit(user, record):
            return await run_in_threadpool(missing, request)
        fd = await request.form(max_files=0, max_fields=4 * US.MAX_GRIDS + 10)
        form = {k: v for k, v in fd.items() if isinstance(v, str)}
        res = await run_in_threadpool(US.save_details, sid, form, user.get("id"), ctx.cfg,
                                      user.get("role") == "admin")
        root = getattr(ctx.cfg, "registry", None)
        with _lock:
            _registry_cache.pop(str(root), None)              # the page after the redirect shows the new line
        if res.get("ok"):
            return RedirectResponse(f"{PREFIX}/session/{quote(sid, safe='')}?saved=1#cj-us-details", status_code=303)
        problem = res["error"] + (f"\n(admin) {res['tech']}" if res.get("tech") and user.get("role") == "admin"
                                  else "")

        def again():
            records, error, viewer = state(request)
            d = detail(records, "session", sid, viewer)
            if d is None:
                return missing(request)
            return html_page(request, d["title"], render_detail(d, error, details_html(records, d, viewer, problem)),
                             res.get("status", 400))
        return await run_in_threadpool(again)

    def not_found_text():
        return Response("not found", status_code=404, media_type="text/plain")

    def thumb(request: Request):
        kind, eid, n = request.path_params["kind"], request.path_params["eid"], request.path_params["n"]
        if kind not in DETAIL_KINDS or not n.isdigit() or len(n) > 7 or not eid or len(eid) > 200 or "\x00" in eid:
            return not_found_text()
        records, _, viewer = state(request)
        files = thumbnail_sources(records, kind, eid, viewer)
        index_ = int(n)
        roots = known_roots() + (upload_roots(ctx.cfg) if kind == "session" else [])
        if index_ >= len(files) or not inside(files[index_], roots):
            return not_found_text()
        data = make_thumb(files[index_], Path(ctx.cfg.state_dir) / "data_thumbs")
        if data is None:
            return not_found_text()
        current = request.query_params.get("v") == thumb_version(files[index_])
        response = Response(data, media_type="image/jpeg")
        response.headers["Cache-Control"] = "private, max-age=86400" if current else "private, max-age=60"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    def api_overview(request: Request):
        records, error, viewer = state(request)
        return ctx.json(overview(records, viewer) | {"error": error})

    def api_search(request: Request):
        records, _, viewer = state(request)
        return ctx.json(search(records, request.query_params.get("q", ""), viewer))

    def api_list(request: Request):
        kind = request.path_params["kind"]
        if kind not in LIST_KINDS:
            return ctx.json({"error": "not found"}, 404)
        records, error, viewer = state(request)
        return ctx.json({"schema": SCHEMA, "kind": kind, "rows": rows_for(records, kind, viewer), "error": error})

    def api_item(request: Request):
        records, _, viewer = state(request)
        kind = request.path_params["kind"]
        d = detail(records, kind, request.path_params["eid"], viewer,
                   requested_page(request, None if kind == "session" else 1))
        return ctx.json(d) if d is not None else ctx.json({"error": "not found"}, 404)

    def lineage_page(request: Request):
        records, error, viewer = state(request)
        tree = lineage_tree(records, viewer, request.path_params["kind"], request.path_params["eid"],
                            getattr(ctx.cfg, "registry", None))
        if tree is None:
            return missing(request)
        focus = graph_focus_key(records, tree["kind"], tree["id"])
        return html_page(request, "Lineage of " + tree["title"], render_lineage(tree, error, focus))

    def api_lineage(request: Request):
        records, _, viewer = state(request)
        tree = lineage_tree(records, viewer, request.path_params["kind"], request.path_params["eid"],
                            getattr(ctx.cfg, "registry", None))
        return ctx.json(tree) if tree is not None else ctx.json({"error": "not found"}, 404)

    def graph_page(request: Request):
        _, error = history_registry(getattr(ctx.cfg, "registry", None))
        return html_page(request, "Lab graph", render_lab_graph(error))

    def api_graph(request: Request):
        records, _, viewer = state(request)
        focus = request.query_params.get("focus")
        focus = focus if focus and len(focus) <= 300 else None
        root = getattr(ctx.cfg, "registry", None)
        if request.query_params.get("scope") == "item" and source_focus(focus):
            return ctx.json(source_graph(records, viewer, root, focus))     # a session / grid
        g = lab_graph(records, viewer, root, focus)
        return ctx.json(item_graph(g) if request.query_params.get("scope") == "item" else g)

    def script(request: Request):
        r = Response(SCRIPT.read_bytes(), media_type="text/javascript; charset=utf-8")
        r.headers["Cache-Control"] = "no-cache"
        r.headers["X-Content-Type-Options"] = "nosniff"
        return r

    found = [Route(PREFIX + "/lineage/graph/", graph_page, methods=["GET"]),     # before /lineage/{kind}/{eid}
             Route(PREFIX + "/lineage/graph", graph_page, methods=["GET"]),
             Route(PREFIX + "/api/lineage/graph", api_graph, methods=["GET"]),   # before /api/{kind}/{eid}
             Route(PREFIX + "/data.js", script, methods=["GET"])]
    found += [Route(PREFIX + "/", index, methods=["GET"]),
             Route(PREFIX + "/search", find, methods=["GET"]),
             Route(PREFIX + "/list/{kind}", listing, methods=["GET"])]
    found += [Route(PREFIX + f"/{k}/{{eid}}", item(k), methods=["GET"]) for k in DETAIL_KINDS]
    found += [Route(PREFIX + "/session/{eid}/details", session_details, methods=["POST"])]
    found += [Route(PREFIX + "/lineage/{kind}/{eid}/", lineage_page, methods=["GET"]),
              Route(PREFIX + "/lineage/{kind}/{eid}", lineage_page, methods=["GET"]),
              Route(PREFIX + "/api/lineage/{kind}/{eid}", api_lineage, methods=["GET"])]
    found += [Route(PREFIX + "/thumb/{kind}/{eid}/{n}.jpg", thumb, methods=["GET"]),
              Route(PREFIX + "/api/overview", api_overview, methods=["GET"]),
              Route(PREFIX + "/api/search", api_search, methods=["GET"]),
              Route(PREFIX + "/api/list/{kind}", api_list, methods=["GET"]),
              Route(PREFIX + "/api/{kind}/{eid}", api_item, methods=["GET"])]
    return found
