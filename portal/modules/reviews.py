"""
My reviews module (CryoEV portal, all roles): the review package picker (R2). The packages a person may open, what
each is made from, how far along it is in cards, and an Open link.

    GET  /reviews/        a row per package <review root>/pred_* (prediction) and lab_* (lab / compare): its name and
                          id, what it is made from (model / label set, images, the person's part in it), "D of T cards
                          decided" with a bar (plus "checked N of M images"), the last edit time and an Open link to
                          /review/<pkg>/ (the review tool, access-checked there too).
                          Mine (default) = packages the person owns or is listed in `reviewers` for. Admins also get
                          All (?show=all): every package, with an owner column.

Rules:
  - Access = may_open(): review_package.json `owner` via ctx.can_see (admins all, others their own), or the person's
    id in its `reviewers` list. Checked before a row is built or rendered, so a package that is not yours leaves no
    trace in the page (id, name or owner).
  - Only what the Open link can open: the portal route's id rule (portal/app.py PRED_ID_RE), a key, and what the
    multi server serves: serving() = multi_review_server.serving() (its prefixes: pred_, plus lab_ only with
    $CJ_REVIEW_PREFIXES, never when the services config can't be read; its excluded names: the services config's
    single-package review_server.py packages, $CJ_REVIEW_EXCLUDE) plus this portal's own real-review package
    (cfg.review_package). The same serving() gates /review/<id>/ in portal/app.py.
  - Name = portal.names.display_name(<predict root>/<ps>/request.json) for a prediction package, else the
    review_package.json `title`, else the package id.
  - Cards = card_progress(): the Quick check queue as the tool builds it in its default mode (flagged objects):
    predictions flagged u at or above data.js cand_conf (0.35 without it), labels flagged u. Decided = the card's own
    decision is set (not a September seed), or its image is rejected. Cached per package by data.js and the edits
    folder's mtimes. edit_stats() still counts images (Reviewed, edited or rejected).
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
PKG_RE = re.compile(r"(?:pred|lab)_[A-Za-z0-9][A-Za-z0-9_.\-]{0,120}")
DEFAULT_CARD_CONF = 0.35   # the tool's card confidence filter without data.js cand_conf
SPLITS = ("train", "valid", "val", "test")
esc = html.escape

_cache: dict = {}
_cards: dict = {}
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


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def package_cards(data: dict) -> dict:
    """{image file: [decision key, ...]}: the tool's Quick check cards (buildQueue in review_tool/index.html, flagged
    objects mode, its default confidence filter): predictions with u at or above cand_conf, labels with u and an id."""
    th = data.get("cand_conf") if _num(data.get("cand_conf")) else DEFAULT_CARD_CONF
    out = {}
    for im in data.get("images") or []:
        if not isinstance(im, dict) or not isinstance(im.get("file"), str):
            continue
        keys = [f"p{k}" for k, p in enumerate(im.get("pr") or [])
                if isinstance(p, dict) and p.get("u") and _num(p.get("s")) and p["s"] >= th]
        keys += [f"g{g['id']}" for g in im.get("gt") or [] if isinstance(g, dict) and g.get("u") and g.get("id") is not None]
        if keys:
            out[im["file"]] = keys
    return out


def _decided(edit, keys) -> int:
    """How many of an image's cards this record decides: all of them when the image is rejected, else each card whose
    decision has a value and is not a September seed (the tool's resolved())."""
    if not isinstance(edit, dict):
        return 0
    dec = edit.get("dec") if isinstance(edit.get("dec"), dict) else {}
    rej = dec.get("i:reject")
    if isinstance(rej, dict) and isinstance(rej.get("v"), str) and rej["v"].startswith("reject:"):
        return len(keys)
    n = 0
    for k in keys:
        d = dec.get(k)
        if isinstance(d, str):
            n += bool(d)
        elif isinstance(d, dict) and d.get("v") and not d.get("seed"):
            n += 1
    return n


def card_progress(pkg: Path) -> tuple:
    """(cards decided, cards in all) for a package, or (None, None) when its data.js can't be read. Cached by the
    data.js and edits folder mtimes."""
    pkg = Path(pkg)
    try:
        st = (pkg / "data.js").stat()
    except OSError:
        return None, None
    try:
        esig = (pkg / "edits").stat().st_mtime_ns
    except OSError:
        esig = None
    dsig = (st.st_mtime_ns, st.st_size)
    key = str(pkg)
    with _lock:
        hit = _cards.get(key)
        if hit and hit[0] == dsig and hit[1] == esig:
            return hit[3], hit[4]
        cards = hit[2] if hit and hit[0] == dsig else None
    if cards is None:
        try:
            data = json.loads((pkg / "data.js").read_text(encoding="utf-8").split("=", 1)[1].strip().rstrip(";"))
            cards = package_cards(data) if isinstance(data, dict) and isinstance(data.get("images"), list) else None
        except (OSError, ValueError, IndexError):
            cards = None
        if cards is None:
            return None, None
    total = sum(len(v) for v in cards.values())
    done = 0
    for file, keys in cards.items():
        rec = _read_json(pkg / "edits" / f"{file}.json") if esig is not None else None
        done += _decided(rec.get("edit") if isinstance(rec, dict) else None, keys)
    with _lock:
        _cards[key] = (dsig, esig, cards, done, total)
    return done, total


def may_open(user: dict, meta: dict, can_see) -> bool:
    """The package's owner (ctx.can_see: admins all, others their own) or a person in its `reviewers` list."""
    if not user or not isinstance(meta, dict):
        return False
    owner = meta.get("owner")
    if isinstance(owner, str) and owner and can_see(user, owner):
        return True
    revs = meta.get("reviewers")
    return isinstance(revs, list) and isinstance(user.get("id"), str) and user["id"] in [r for r in revs if isinstance(r, str)]


def _folder_name(path) -> str:
    """The folder a label set / model file lives in (never a full local path): its file dropped, a split folder too."""
    parts = [p for p in re.split(r"[\\/]+", str(path or "")) if p]
    if len(parts) > 1 and re.search(r"\.[A-Za-z0-9]+$", parts[-1]):
        parts.pop()
    while len(parts) > 1 and parts[-1].lower() in SPLITS:
        parts.pop()
    return parts[-1] if parts else ""


def made_from(meta: dict) -> str:
    """What a package reviews: prediction: "<model> predictions"; compare: "<label set> vs <model>"; + images."""
    kind, out = meta.get("kind"), []
    if kind == "prediction" and isinstance(meta.get("model"), str):
        out.append(f"{meta['model']} predictions")
    elif kind == "compare":
        cmp = meta.get("compare") if isinstance(meta.get("compare"), dict) else {}
        ca, cb = cmp.get("a"), cmp.get("b")
        a = (_folder_name(ca.get("coco")) if isinstance(ca, dict) else "") or _folder_name(meta.get("base"))
        b = _folder_name(cb.get("weights")) if isinstance(cb, dict) else ""
        out.append(f"{a or 'labels'} vs {b or 'a model'}")
    if isinstance(meta.get("n_images"), int):
        out.append(f"{meta['n_images']} images")
    return " · ".join(out)


def serving(cfg=None, review_package=None) -> tuple:
    """(prefixes, excluded lower-case names): what the multi server serves (multi_review_server.serving(), the one
    source), minus the portal's own real-review package (review_package, else cfg.review_package)."""
    from annotation import multi_review_server as MRS
    own = review_package or getattr(cfg, "review_package", None)
    return MRS.serving(exclude=[str(own)] if own else [])


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
    prefixes, excluded = serving(cfg)
    me = user.get("id") if isinstance(user, dict) else None
    rows = []
    for d in dirs:
        # only what the Open link can open: the same checks as app.pred_package and the review server
        if not PRED_ID_RE.match(d.name) or d.name.lower() in excluded or MRS.package_dir(Path(root), d.name, prefixes) is None \
                or not MRS.package_key(d):
            continue
        meta = _read_json(d / "review_package.json")
        if not isinstance(meta, dict) or not may_open(user, meta, can_see):
            continue
        owner = meta.get("owner") if isinstance(meta.get("owner"), str) else ""
        revs = [r for r in meta["reviewers"] if isinstance(r, str)] if isinstance(meta.get("reviewers"), list) else []
        mine = bool(me) and (owner == me or me in revs)
        if only_mine and not mine:
            continue
        name = None
        if d.name.startswith("pred_") and proot:
            req = _read_json(proot / d.name[len("pred_"):] / "request.json")
            name = names.display_name(req) if isinstance(req, dict) else None
        if not name:
            name = meta["title"] if isinstance(meta.get("title"), str) and meta["title"].strip() else d.name
        total = meta.get("n_images")
        if not isinstance(total, int):
            total = None
        checked, last = edit_stats(d)
        decided, cards = card_progress(d)
        role = "yours" if owner and owner == me else "reviewer: you" if me in revs else ""
        rows.append({"pkg": d.name, "owner": owner, "name": name, "checked": checked, "total": total, "last": last,
                     "decided": decided, "cards": cards, "made_from": made_from(meta), "role": role})
    rows.sort(key=lambda r: (r["last"] or 0, r["pkg"]), reverse=True)
    return rows


def _progress(r) -> str:
    imgs = f"checked {r['checked']} of {r['total']} images" if r["total"] is not None else f"checked {r['checked']}"
    if r["cards"] is None:
        return f'<span class="cj-reviews-prog">{esc(imgs)}</span>'
    if not r["cards"]:
        return f'<span class="cj-reviews-prog">no cards to check<br><small class="cj-mut">{esc(imgs)}</small></span>'
    pct = round(100 * r["decided"] / r["cards"])
    return (f'<span class="cj-reviews-prog"><b>{r["decided"]} of {r["cards"]} cards</b> decided'
            f'<span class="cj-reviews-bar" role="progressbar" aria-valuemin="0" aria-valuemax="{r["cards"]}" '
            f'aria-valuenow="{r["decided"]}" style="display:block;height:6px;margin:4px 0 2px;border-radius:3px;'
            f'background:color-mix(in srgb,currentColor 14%,transparent)">'
            f'<span style="display:block;height:6px;width:{pct}%;border-radius:3px;background:currentColor;opacity:.75">'
            f'</span></span><small class="cj-mut">{esc(imgs)}</small></span>')


def render(rows, is_admin: bool, show_all: bool) -> str:
    out = ['<h1>Choose a review package</h1>',
           '<p class="cj-mut">Each package shows what it reviews, your part in it and how far along it is. '
           'Your decisions save as you go.</p>']
    if is_admin:
        mine, every = ("cj-reviews-on" if not show_all else ""), ("cj-reviews-on" if show_all else "")
        out.append(f'<p class="cj-reviews-toggle"><a class="{mine}" href="/reviews/">Mine</a> '
                   f'<a class="{every}" href="/reviews/?show=all">All</a></p>')
    if not rows:
        out.append('<p class="cj-card cj-reviews-empty">No reviews yet. Run a model in Predict first.</p>')
        return "".join(out)
    head = ['<th>Package</th>', '<th>Made from</th>'] + (['<th>Owner</th>'] if show_all else []) + \
        ['<th>Progress</th>', '<th>Last edit</th>', '<th></th>']
    body = []
    for r in rows:
        last = names.when(datetime.fromtimestamp(r["last"]).isoformat()) if r["last"] else "No edits yet"
        sub = f'<br><small class="cj-mut"><code>{esc(r["pkg"])}</code></small>' if r["name"] != r["pkg"] else ""
        made = " · ".join(x for x in (r["made_from"], r["role"]) if x)
        body.append(
            f'<tr><td class="cj-wrap"><b>{esc(r["name"])}</b>{sub}</td>'
            f'<td class="cj-wrap cj-mut">{esc(made)}</td>'
            + (f'<td class="cj-wrap">{esc(r["owner"])}</td>' if show_all else "")
            + f'<td>{_progress(r)}</td><td class="cj-mut">{esc(last)}</td>'
            f'<td><a class="cj-reviews-open" href="/review/{esc(r["pkg"])}/">Open</a></td></tr>')
    out.append(f'<div class="cj-reviews-wrap"><table class="cj-reviews-tab"><thead><tr>{"".join(head)}</tr></thead>'
               f'<tbody>{"".join(body)}</tbody></table></div>')
    out.append('<p class="cj-mut">Only one review server saves to a package at a time, so no save is lost.</p>')
    return "".join(out)


def routes(ctx) -> list:
    def index(request: Request):
        user = ctx.user(request)
        is_admin = user.get("role") == "admin"
        show_all = is_admin and request.query_params.get("show") == "all"
        rows = list_packages(ctx.cfg, user, ctx.can_see, only_mine=not show_all)
        return ctx.page(request, "My reviews", render(rows, is_admin, show_all), active="reviews")

    return [Route(PREFIX + "/", index, methods=["GET"])]
