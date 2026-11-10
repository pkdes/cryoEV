"""
My reviews module (CryoEV portal, all roles): the prediction review packages a person may open, with how far along
each is.

    GET  /reviews/        a row per package <review root>/pred_*: the analysis name, "checked N of M images", the
                          last edit time and an Open link to /review/<pkg>/ (the eVHarmony tool, owner-checked there).
                          Admins also get a Mine / All toggle (?show=all; default Mine); All adds an owner column.

Rules:
  - Owner = review_package.json `owner`; ctx.can_see(user, owner) before a row is built or rendered, so a package
    that is not yours leaves no trace in the page (id, name or owner).
  - Name = portal.names.display_name(<predict root>/<ps>/request.json), else the package id.
  - Checked = images whose edits/<file>.json holds an edit that is Reviewed, edited (the rule reviewed_polygons
    uses: annotation.prediction_package.is_edited) or rejected (is_rejected). Of M = review_package.json n_images.
  - The scan is cached per package by the edits folder's mtime (a saved edit replaces a file, which changes it).
  - This prefix (/reviews) is not the core /review page and does not match the Gate's /review/<pkg> owner rule
    (that needs the slash after /review): the module's own ROLES decide, all three roles.
  - Every text goes through html.escape.
"""
from __future__ import annotations

import html
import json
import re
import threading
from datetime import datetime
from pathlib import Path

from starlette.requests import Request
from starlette.routing import Route

from portal import names
from portal.modules import results as RES

PREFIX = "/reviews"
ROLES = ("admin", "advanced", "basic")
PKG_RE = re.compile(r"pred_[A-Za-z0-9][A-Za-z0-9_.\-]{0,120}")
esc = html.escape

_cache: dict = {}
_lock = threading.Lock()


def _read_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _is_checked(rec) -> bool:
    edit = rec.get("edit") if isinstance(rec, dict) else None
    if not edit:
        return False
    if isinstance(edit, dict) and edit.get("reviewed"):
        return True
    try:
        from annotation.prediction_package import is_edited, is_rejected
        return bool(is_edited(edit) or is_rejected(edit))  # a rejected image is checked: nothing left to review
    except Exception:                                   # the tool's own rule is unavailable: any decision counts
        return isinstance(edit, dict) and bool(edit.get("dec"))


def edit_stats(pkg: Path) -> tuple:
    """(images checked, newest edit mtime or None), cached by the edits folder's mtime."""
    ed = pkg / "edits"
    try:
        sig = ed.stat().st_mtime_ns
    except OSError:
        return 0, None
    key = str(pkg)
    with _lock:
        hit = _cache.get(key)
        if hit and hit[0] == sig:
            return hit[1], hit[2]
    n, newest = 0, None
    try:
        for f in ed.glob("*.json"):
            try:
                t = f.stat().st_mtime
            except OSError:
                continue
            newest = t if newest is None or t > newest else newest
            if _is_checked(_read_json(f)):
                n += 1
    except OSError:
        pass
    with _lock:
        _cache[key] = (sig, n, newest)
    return n, newest


def list_packages(cfg, user: dict, can_see, only_mine: bool) -> list:
    root = RES.review_root(cfg)
    if root is None:
        return []
    proot = RES.predict_root(cfg)
    try:
        dirs = sorted(d for d in Path(root).iterdir() if PKG_RE.fullmatch(d.name) and d.is_dir())
    except OSError:
        return []
    from annotation import multi_review_server as MRS
    from portal.app import PRED_ID_RE
    rows = []
    for d in dirs:
        # only what the Open link can open: the same checks as app.pred_package and the review server
        if not PRED_ID_RE.match(d.name) or MRS.package_dir(Path(root), d.name) is None or not MRS.package_key(d):
            continue
        meta = _read_json(d / "review_package.json")
        if not isinstance(meta, dict):
            continue
        owner = meta.get("owner")
        if not isinstance(owner, str) or not owner or not can_see(user, owner):
            continue
        if only_mine and owner != user.get("id"):
            continue
        ps = d.name[len("pred_"):]
        req = _read_json(proot / ps / "request.json") if proot else None
        name = names.display_name(req) if isinstance(req, dict) else d.name
        total = meta.get("n_images")
        if not isinstance(total, int):
            total = None
        checked, last = edit_stats(d)
        rows.append({"pkg": d.name, "owner": owner, "name": name, "checked": checked, "total": total, "last": last})
    rows.sort(key=lambda r: (r["last"] or 0, r["pkg"]), reverse=True)
    return rows


def render(rows, is_admin: bool, show_all: bool) -> str:
    out = ['<h1>My reviews</h1>',
           '<p class="cj-mut">Prediction review packages you can open, and how many images have been checked.</p>']
    if is_admin:
        mine, every = ("cj-reviews-on" if not show_all else ""), ("cj-reviews-on" if show_all else "")
        out.append(f'<p class="cj-reviews-toggle"><a class="{mine}" href="/reviews/">Mine</a> '
                   f'<a class="{every}" href="/reviews/?show=all">All</a></p>')
    if not rows:
        out.append('<p class="cj-card cj-reviews-empty">No reviews yet. Run a model in Predict first.</p>')
        return "".join(out)
    head = ['<th>Analysis</th>'] + (['<th>Owner</th>'] if show_all else []) + ['<th>Checked</th>', '<th>Last edit</th>',
                                                                              '<th></th>']
    body = []
    for r in rows:
        of = f"checked {r['checked']} of {r['total']} images" if r["total"] is not None else f"checked {r['checked']}"
        last = names.when(datetime.fromtimestamp(r["last"]).isoformat()) if r["last"] else "No edits yet"
        body.append(
            f'<tr><td class="cj-wrap"><b>{esc(r["name"])}</b></td>'
            + (f'<td class="cj-wrap">{esc(r["owner"])}</td>' if show_all else "")
            + f'<td>{esc(of)}</td><td class="cj-mut">{esc(last)}</td>'
            f'<td><a class="cj-reviews-open" href="/review/{esc(r["pkg"])}/">Open</a></td></tr>')
    out.append(f'<div class="cj-reviews-wrap"><table class="cj-reviews-tab"><thead><tr>{"".join(head)}</tr></thead>'
               f'<tbody>{"".join(body)}</tbody></table></div>')
    return "".join(out)


def routes(ctx) -> list:
    def index(request: Request):
        user = ctx.user(request)
        is_admin = user.get("role") == "admin"
        show_all = is_admin and request.query_params.get("show") == "all"
        rows = list_packages(ctx.cfg, user, ctx.can_see, only_mine=not show_all)
        return ctx.page(request, "My reviews", render(rows, is_admin, show_all), active="reviews")

    return [Route(PREFIX + "/", index, methods=["GET"])]
