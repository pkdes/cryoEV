"""
My data (CryoEV portal, round 17r): a basic account's Data page. The lab's Data catalog (/data/, admin and advanced)
lists every session, grid, label set and model; a basic account sees only the images it can already use.

    GET /mydata/                        the image sets this person may analyse (Predict's sources: the shared
                                        sessions and demo sets the admin lists, plus their own converted uploads),
                                        each with a strip of thumbnails; their own analyses; "add your own images"
    GET /mydata/set/<key>/?page=N       one image set: every image (24 per page) with its sample and nm/px from the
                                        set's MANIFEST.csv, and "Analyse these"
    GET /mydata/thumb/<key>/<n>.jpg     one thumbnail (at most THUMB_PX on the long side), the n-th image of that set

Rules:
  - Nothing new is exposed: a set is shown only when Predict's New analysis already offers it to this person
    (predict_parts.create.sources_safe with their own can_see, and no problem); the rest is 404. Analyses come from
    predict_parts.sets.list_sets (owner or admin), as on /predict/.
  - Read only. The client never sends a path: <key> picks a source from that server-side list and <n> an index into
    its sorted image names (the listing predict_sources.check_folder counts). Symlinks are skipped, and a file whose
    real path is not inside the set's real folder is refused.
  - Thumbnails are kept in memory (THUMB_CACHE entries, keyed on the file's mtime and size), never on disk.
  - Roles: admin, advanced and basic may open it (an admin can check what a basic account sees); only basic has it in
    the nav (modules.json), as "Data". Guests have no modules, so never reach it.
  - Everything shown is escaped here.
"""
from __future__ import annotations

import csv
import html
import io
import os
import threading
from collections import OrderedDict
from pathlib import Path
from urllib.parse import quote

from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route

from jobs import predict_set as PS
from portal import predict_sources as PSRC
from portal.predict_parts.create import sources_safe
from portal.predict_parts.sets import list_sets

PREFIX = "/mydata"
ROLES = ("admin", "advanced", "basic")
PER_PAGE = 24
STRIP = 6
THUMB_PX = 360
THUMB_CACHE = 400
esc = html.escape

_cache: OrderedDict = OrderedDict()
_cache_lock = threading.Lock()

CSS = """<style>
/* My data (portal/modules/mydata.py). cj.css tokens only; every class is cj-md-*. Phone first. */
.cj-md{max-width:1100px;min-width:0}
.cj-md-sets{display:grid;gap:12px;grid-template-columns:repeat(auto-fill,minmax(min(100%,320px),1fr))}
.cj-md-set{display:flex;flex-direction:column;gap:8px;min-width:0}
.cj-md-set b{font-size:1.02rem}
.cj-md-strip{display:grid;grid-template-columns:repeat(3,1fr);gap:4px}
.cj-md-strip img,.cj-md-grid img{width:100%;aspect-ratio:1;object-fit:cover;border-radius:6px;background:var(--line);
  display:block}
.cj-md-grid{display:grid;gap:10px;grid-template-columns:repeat(auto-fill,minmax(min(46%,180px),1fr))}
.cj-md-grid figure{margin:0;min-width:0}
.cj-md-grid figcaption{font-size:.78rem;color:var(--mut);overflow-wrap:anywhere;margin-top:3px}
.cj-md-acts{display:flex;flex-wrap:wrap;gap:8px;align-items:center}
.cj-md-btn{display:inline-flex;align-items:center;min-height:44px;padding:8px 14px;border-radius:8px;font-weight:600;
  border:1px solid var(--line);background:var(--card);color:var(--fg);text-decoration:none}
.cj-md-btn.cj-md-go{background:var(--acc);border-color:var(--acc);color:var(--bg)}
.cj-md-list{list-style:none;padding:0;margin:0;display:flex;flex-direction:column;gap:6px}
.cj-md-list a{font-weight:600}
.cj-md-pages{display:flex;gap:8px;flex-wrap:wrap;margin:12px 0}
</style>"""


# ---------------------------------------------------------------- data
def visible_sets(cfg, user, can_see) -> list:
    """Predict's sources this person may start an analysis on (no problem)."""
    srcs, _err = sources_safe(cfg, user, can_see)
    return [s for s in srcs if not s.get("problem")]


def find_set(cfg, user, can_see, key):
    return next((s for s in visible_sets(cfg, user, can_see) if s.get("key") == key), None)


def image_names(d: Path) -> list:
    """The set's images, sorted, as check_folder counts them (real files only, no symlinks)."""
    try:
        return sorted(e.name for e in os.scandir(d) if e.is_file(follow_symlinks=False) and not e.is_symlink()
                      and not e.name.startswith(".") and os.path.splitext(e.name)[1].lower() in PSRC.IMAGE_EXT)
    except OSError:
        return []


def manifest_rows(src) -> dict:
    """{image name: (sample, nm/px text)} from the set's MANIFEST.csv; {} when it can't be read."""
    man = src.get("manifest") or (Path(src["dir"]) / PSRC.MANIFEST)
    try:
        with open(man, newline="", encoding="utf-8-sig") as f:
            rd = csv.DictReader(f)
            head = list(rd.fieldnames or [])
            rows = list(rd)
    except (OSError, UnicodeDecodeError, csv.Error, TypeError):
        return {}
    key = next((c for c in PS.NAME_COLUMNS if c in head), None)
    if key is None:
        return {}
    sc = src.get("sample_column")
    col = sc if sc in head else next((c for c in PSRC.SAMPLE_COLUMNS if c in head), None)
    return {r.get(key): (str(r.get(col) or "").strip() if col else "", str(r.get(PS.NM_COLUMN) or "").strip())
            for r in rows if r.get(key)}


def thumb_bytes(src, n: int):
    """JPEG bytes of the n-th image's thumbnail, or None (no such image, outside the folder, unreadable)."""
    d = Path(src["dir"])
    names = image_names(d)
    if not 0 <= n < len(names):
        return None
    path = d / names[n]
    try:
        real_dir = os.path.realpath(d)
        real = os.path.realpath(path)
        if os.path.commonpath([real_dir, real]) != real_dir or real == real_dir:
            return None
        stt = os.stat(real)
    except (OSError, ValueError):
        return None
    ck = (real, stt.st_mtime_ns, stt.st_size)
    with _cache_lock:
        if ck in _cache:
            _cache.move_to_end(ck)
            return _cache[ck]
    try:
        from PIL import Image
        with Image.open(real) as im:
            im.thumbnail((THUMB_PX, THUMB_PX))
            buf = io.BytesIO()
            im.convert("L" if im.mode in ("L", "I;16", "I", "F", "1") else "RGB").save(buf, "JPEG", quality=82)
    except Exception:  # noqa: BLE001  a broken image is "no thumbnail", never a 500
        return None
    data = buf.getvalue()
    with _cache_lock:
        _cache[ck] = data
        while len(_cache) > THUMB_CACHE:
            _cache.popitem(last=False)
    return data


# ---------------------------------------------------------------- pages
def _thumb(key, i, alt):
    return f'<img loading="lazy" src="{PREFIX}/thumb/{quote(key)}/{i}.jpg" alt="{esc(alt)}">'


def _summary(src, n_img, rows) -> str:
    samples = sorted({s for s, _ in rows.values() if s})
    nms = sorted({v for _, v in rows.values() if v})
    bits = [f"{n_img} image{'s' if n_img != 1 else ''}"]
    if samples:
        bits.append(f"{len(samples)} sample{'s' if len(samples) != 1 else ''}: " + ", ".join(samples[:6])
                    + ("…" if len(samples) > 6 else ""))
    if nms:
        bits.append("nm/px " + " / ".join(nms[:3]) + ("…" if len(nms) > 3 else ""))
    return " · ".join(bits)


def page_index(cfg, user, can_see) -> str:
    cards = []
    for s in visible_sets(cfg, user, can_see):
        names = image_names(Path(s["dir"]))
        rows = manifest_rows(s)
        k = s["key"]
        strip = "".join(_thumb(k, i, names[i]) for i in range(min(STRIP, len(names))))
        cards.append(
            f'<div class="cj-card cj-md-set"><b>{esc(s["label"])}</b>'
            f'<span class="cj-mut">{esc(_summary(s, len(names), rows))}</span>'
            f'<a class="cj-md-strip" href="{PREFIX}/set/{quote(k)}/" aria-label="See all images">{strip}</a>'
            f'<div class="cj-md-acts"><a class="cj-md-btn" href="{PREFIX}/set/{quote(k)}/">See all images</a>'
            f'<a class="cj-md-btn cj-md-go" href="/predict/new/session/{quote(k)}/">Analyse these →</a></div></div>')
    sets_html = ('<div class="cj-md-sets">' + "".join(cards) + '</div>') if cards else \
        '<p class="cj-mut">No image sets are shared with you yet. Upload your own below.</p>'
    mine = []
    for d, req, st in list_sets(cfg, user, can_see)[:20]:
        name = req.get("name") or d.name
        n = st.get("n_total") or ""
        state = st.get("state") or ""
        mine.append(f'<li><a href="/predict/{esc(d.name)}/">{esc(str(name))}</a> '
                    f'<span class="cj-mut">· {esc(str(n))} images · {esc(str(state))}</span></li>')
    mine_html = (('<ul class="cj-md-list">' + "".join(mine) + '</ul>') if mine else
                 '<p class="cj-mut">No analyses yet. Pick an image set above and press “Analyse these”.</p>')
    return (CSS + '<div class="cj-md"><h1>Data</h1>'
            '<p class="cj-mut">Your images: the sets you can analyse, and what you have run on them.</p>'
            f'<h2 class="cj-sec">Image sets</h2>{sets_html}'
            f'<h2 class="cj-sec">Your analyses</h2>{mine_html}'
            '<p><a href="/results/">Results →</a></p>'
            '<h2 class="cj-sec">Add your own images</h2>'
            '<p>Processed 8-bit PNG or JPEG tiles: <a class="cj-md-btn" href="/predict/new/upload/">Upload images</a></p>'
            '</div>')


def page_set(src, page: int) -> str:
    names = image_names(Path(src["dir"]))
    rows = manifest_rows(src)
    k = src["key"]
    pages = max(1, -(-len(names) // PER_PAGE))
    page = min(max(1, page), pages)
    lo = (page - 1) * PER_PAGE
    figs = []
    for i in range(lo, min(lo + PER_PAGE, len(names))):
        sample, nm = rows.get(names[i], ("", ""))
        cap = " · ".join(x for x in (names[i], sample, f"{nm} nm/px" if nm else "") if x)
        figs.append(f'<figure>{_thumb(k, i, names[i])}<figcaption>{esc(cap)}</figcaption></figure>')
    nav = ""
    if pages > 1:
        nav = '<div class="cj-md-pages">' + "".join(
            f'<a class="cj-md-btn" href="?page={p}"{" aria-current=page" if p == page else ""}>{p}</a>'
            for p in range(1, pages + 1)) + '</div>'
    return (CSS + f'<div class="cj-md"><p class="cj-mut"><a href="{PREFIX}/">← Data</a></p>'
            f'<h1>{esc(src["label"])}</h1><p class="cj-mut">{esc(_summary(src, len(names), rows))}</p>'
            f'<div class="cj-md-acts"><a class="cj-md-btn cj-md-go" href="/predict/new/session/{quote(k)}/">'
            'Analyse these →</a></div>'
            f'{nav}<div class="cj-md-grid" style="margin-top:12px">{"".join(figs)}</div>{nav}</div>')


# ---------------------------------------------------------------- routes
def routes(ctx) -> list:
    def not_found(request):
        r = ctx.page(request, "Not found", '<h1>Not found</h1><p>That image set is not available to you.</p>'
                     f'<p><a href="{PREFIX}/">← Data</a></p>', active="mydata")
        r.status_code = 404
        return r

    def index(request: Request):                   # sync handlers: Starlette runs them in a worker thread
        return ctx.page(request, "Data", page_index(ctx.cfg, ctx.user(request), ctx.can_see), active="mydata")

    def one_set(request: Request):
        src = find_set(ctx.cfg, ctx.user(request), ctx.can_see, request.path_params["key"])
        if src is None:
            return not_found(request)
        try:
            page = int(request.query_params.get("page") or 1)
        except ValueError:
            page = 1
        return ctx.page(request, src["label"], page_set(src, page), active="mydata")

    def thumb(request: Request):
        src = find_set(ctx.cfg, ctx.user(request), ctx.can_see, request.path_params["key"])
        try:
            n = int(request.path_params["n"])
        except ValueError:
            n = -1
        data = thumb_bytes(src, n) if src is not None else None
        if data is None:
            return Response(b"", status_code=404, headers={"Cache-Control": "no-store"})
        return Response(data, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=600"})

    return [Route(PREFIX + "/", index, methods=["GET"]),
            Route(PREFIX + "/set/{key}/", one_set, methods=["GET"]),
            Route(PREFIX + "/thumb/{key}/{n}.jpg", thumb, methods=["GET"])]
