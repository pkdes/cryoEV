"""
The HTML of every Data page (every value escaped).

Part of the Data module (portal/modules/data/__init__.py has the rules); moved from data.py unchanged.
"""
from __future__ import annotations

from urllib.parse import quote

from portal import upload_sessions as US
from portal.modules.data.records import (
    DETAIL_LIST, KIND_TITLES, PAGE_N, PREFIX, TEMPLATE, THUMB_PX, entity_href, esc, join_bits, lineage_href,
    _number, plural)
from portal.modules.data.graphs import SOURCE_ENTITIES, lab_graph_href, source_focus


# ---------------------------------------------------------------- HTML
def chip(text, tone="") -> str:
    cls = {"prod": " cj-data-t-prod", "warn": " cj-data-t-warn", "acc": " cj-data-t-acc"}.get(tone, "")
    return f'<span class="cj-data-chip{cls}">{esc(str(text))}</span>'


def render_rows(rows) -> str:
    if not rows:
        return '<p class="cj-mut cj-data-empty">Nothing here yet.</p>'
    items = []
    for r in rows:
        tags = "".join(chip(t, tone) for t, tone in r["tags"])
        sub = f'<span class="cj-mut">{esc(r["sub"])}</span>' if r["sub"] else ""
        if r["title"] != r["id"]:
            sub = f'<span class="cj-mut cj-vis-id">{esc(r["id"])}</span>' + sub
        items.append(f'<li class="cj-data-row"><span class="cj-data-top"><a href="{esc(r["url"])}">'
                     f'{esc(r["title"])}</a>{tags}<a class="cj-data-lnk" href="{esc(r["lineage"])}">Lineage</a>'
                     f'</span>{sub}</li>')
    return '<ul class="cj-data-rows">' + "".join(items) + "</ul>"


def search_form(query="") -> str:
    return ('<form class="cj-data-search" role="search" method="get" action="/data/search">'
            '<input type="search" name="q" maxlength="100" placeholder="Search by id or sample label" '
            f'aria-label="Search the catalog" value="{esc(query, quote=True)}"><button type="submit">Search</button>'
            '</form>')


def banner(error) -> str:
    if not error:
        return ""
    return ('<p class="cj-data-banner" role="status">The catalog could not be read just now, so the lists below may '
            f'be empty. ({esc(error)})</p>')


def breadcrumb(*parts) -> str:
    return '<p class="cj-data-crumb cj-mut">' + " / ".join(
        f'<a href="{esc(href)}">{esc(text)}</a>' for text, href in parts) + "</p>"


def card_section(title, count, rows_html, extra_id="") -> str:
    return (f'<section class="cj-card cj-data-card"{extra_id}><h2><span>{esc(title)}</span>'
            f'<span class="cj-data-n">{count}</span></h2>{rows_html}</section>')


def render_overview(ov, error) -> str:
    cards = []
    for c in ov["cards"]:
        more = (f'<p class="cj-data-more"><a href="{esc(c["all"])}">See all {c["count"]}</a></p>'
                if c["count"] > len(c["rows"]) else "")
        cards.append(card_section(c["title"], c["count"], render_rows(c["rows"]) + more,
                                  f' id="cj-data-c-{esc(c["kind"])}"'))
    grids = ov["counts"].get("grids", 0)
    return ('<h1>Data</h1><p class="cj-mut cj-data-sub">What data exists: imaging sessions, image sets, labels, '
            f'datasets and models. <a href="{PREFIX}/list/grids">{plural(grids, "grid")}</a> are listed under '
            f'their sessions. <a href="{lab_graph_href()}">Lab graph</a>: how label sets, datasets, models and '
            'reviews connect.</p>' + banner(error) + search_form()
            + '<div class="cj-data-cards">' + "".join(cards) + "</div>"
            + '<p class="cj-mut cj-data-foot">Read only: nothing here changes any data (except the session details '
              'of an upload, which its uploader and the admin can edit).</p>')


def render_search(result, error) -> str:
    if not result["q"]:
        body = '<p class="cj-mut">Type an id or a sample label.</p>'
    elif not result["groups"]:
        body = f'<p class="cj-mut">Nothing found for &ldquo;{esc(result["q"])}&rdquo;.</p>'
    else:
        body = "".join(card_section(g["title"], g["count"], render_rows(g["rows"])) for g in result["groups"])
    return (breadcrumb(("Data", PREFIX + "/")) + "<h1>Search</h1>" + banner(error) + search_form(result["q"])
            + '<div class="cj-data-cards">' + body + "</div>")


def render_list(title, rows, error) -> str:
    return (breadcrumb(("Data", PREFIX + "/")) + f"<h1>{esc(title)}</h1>"
            f'<p class="cj-mut cj-data-sub">{len(rows)} in all.</p>' + banner(error)
            + '<div class="cj-card">' + render_rows(rows) + "</div>")


def list_item(v) -> str:
    """One bullet: a plain value escaped, or [text, href] (a link when href is set)."""
    if isinstance(v, (list, tuple)) and len(v) == 2:
        text, href = v
        return f'<a href="{esc(href)}">{esc(str(text))}</a>' if href else esc(str(text))
    return esc(str(v))


def render_fields(fields) -> str:
    """A list value is a bulleted list (one item per value; an item [text, href] is a link when href is set);
    anything else is one escaped line."""
    if not fields:
        return ""
    items = []
    for label, value, href in fields:
        if isinstance(value, (list, tuple)):
            shown = '<ul class="cj-data-vlist">' + "".join(f"<li>{list_item(v)}</li>" for v in value) + "</ul>"
        else:
            shown = f'<a href="{esc(href)}">{esc(value)}</a>' if href else esc(value)
        items.append(f"<div><dt>{esc(label)}</dt><dd>{shown}</dd></div>")
    return '<dl class="cj-data-kv">' + "".join(items) + "</dl>"


def render_thumbs(d, strip=False) -> str:
    if not d["thumbs"]:
        return '<p class="cj-mut cj-data-empty">' + esc(d.get("note") or "No images found for this yet.") + "</p>"
    n = d["n_images"]
    items = "".join(f'<li><img class="cj-data-img" src="{esc(t["url"])}" alt="Image {t["n"] + 1} of {n}" '
                    f'loading="lazy" decoding="async" width="{THUMB_PX}" height="{THUMB_PX}"></li>'
                    for t in d["thumbs"])
    out = f'<ul class="{"cj-data-strip" if strip else "cj-data-thumbs"}">{items}</ul>'
    if strip:
        if n > len(d["thumbs"]):
            out += (f'<p class="cj-data-more"><a href="{esc(entity_href(d["kind"], d["id"]))}?page=1">'
                    f'See all {plural(n, "image")}</a></p>')
    elif d["pages"] > 1:
        base = entity_href(d["kind"], d["id"])
        first, last = (d["page"] - 1) * PAGE_N + 1, min(n, d["page"] * PAGE_N)
        previous = (f'<a href="{esc(base)}?page={d["page"] - 1}" rel="prev">Previous</a>' if d["page"] > 1
                    else '<span class="cj-data-off">Previous</span>')
        following = (f'<a href="{esc(base)}?page={d["page"] + 1}" rel="next">Next</a>' if d["page"] < d["pages"]
                     else '<span class="cj-data-off">Next</span>')
        out += (f'<nav class="cj-data-pager" aria-label="Pages">{previous}'
                f'<span class="cj-mut">Showing {first}-{last} of {n}</span>{following}</nav>')
    return out


def render_metric_table(rows) -> str:
    if not rows:
        return ""
    return '<table class="cj-data-metrics"><tbody>' + "".join(
        f'<tr><th scope="row">{esc(name)}</th><td>{esc(value)}</td></tr>' for name, value in rows) + "</tbody></table>"


def render_session_extras(d, extra="") -> str:
    """extra: the Session details form (the page handler renders it from the record; never in the JSON)."""
    out = ['<h2 class="cj-sec">Grids</h2>']
    if d["grids"]:
        items = []
        for g in d["grids"]:
            slot_name = f'Slot {g["slot"]}'
            head = (f'<a href="{esc(entity_href("grid", g["grid"]))}">Grid {esc(g["grid"])}</a>' if g["grid"]
                    else esc(slot_name))
            if g["grid"]:
                where = slot_name
            elif g["grid_id_recorded"]:
                where = f'grid {g["grid_id_recorded"]} not in the list'
            elif g.get("grid_id_typed"):
                where = f'grid id {g["grid_id_typed"]} (session details)'
            else:
                where = "grid number not recorded"
            count = f'{plural(g["n_images"], "image")} ({g["count_source"]})' if g["n_images"] else None
            items.append(f'<li><span class="cj-data-top"><b>{head}</b></span>'
                         f'<span class="cj-mut">{esc(join_bits(g["label"], g["grid_type"], count, where))}</span></li>')
        out.append('<ul class="cj-data-grids">' + "".join(items) + "</ul>")
    else:
        out.append('<p class="cj-mut">No grids recorded for this session.</p>')
    up = d.get("upload")
    if up is not None:
        out.append('<h2 class="cj-sec">Uploaded raw files</h2>')
        if up["conversions"]:
            items = "".join(
                f'<li class="cj-data-row"><span class="cj-data-top"><a href="{esc(c["url"])}">Conversion '
                f'{esc(c["id"])}</a><a class="cj-data-lnk" href="{esc(c["gallery"])}">Gallery</a></span>'
                + (f'<span class="cj-mut">{esc(plural(c["n_tiles"], "tile"))}</span>'
                   if _number(c.get("n_tiles")) is not None else "")
                + (f'<span class="cj-mut">re-converted from {esc(c["derived_from"])}</span>'
                   if c.get("derived_from") else "") + "</li>"
                for c in up["conversions"])
            out.append(f'<div class="cj-card"><ul class="cj-data-rows">{items}</ul></div>')
        else:
            out.append('<p class="cj-mut">Converted from a portal upload. Its conversion and gallery are open to the '
                       'person who uploaded it and the admin.</p>')
        out.append(extra or US.details_view({"details": up["details"]}))
    if d.get("analyses"):
        items = "".join(f'<li class="cj-data-row"><span class="cj-data-top"><a href="{esc(a["url"])}">'
                        f'{esc(a["title"])}</a>' + (chip(a["status"]) if a.get("status") else "") + "</span></li>"
                        for a in d["analyses"])
        out.append(f'<h2 class="cj-sec">Analyses of this session</h2><div class="cj-card"><ul class="cj-data-rows">'
                   f'{items}</ul></div>')
    out.append(f'<h2 class="cj-sec">Images ({d["n_images"]} {esc(d.get("images_word") or "processed")})</h2>'
               + render_thumbs(d, strip=d["view"] == "strip"))
    return "".join(out)


def render_model_extras(d) -> str:
    out = ['<h2 class="cj-sec">Scores</h2>']
    table = render_metric_table(d["metrics"])
    if table:
        out.append(f'<div class="cj-card">{table}</div>')
    if d["checks"]:
        items = []
        for c in d["checks"]:
            sub = join_bits(f"session {c['session']}" if c.get("session") else None,
                            plural(c["n"], "image") if c.get("n") is not None else None, c.get("status"))
            items.append(f'<li class="cj-data-row"><span>{esc(c["id"])}</span><span class="cj-mut">{esc(sub)}</span>'
                         + render_metric_table(c["metrics"]) + "</li>")
        out.append('<div class="cj-card"><h2>Checks on new images</h2><ul class="cj-data-rows">'
                   + "".join(items) + "</ul></div>")
    if not table and not d["checks"]:
        out.append('<p class="cj-mut">No scores are recorded in the registry for this model yet.</p>')
    return "".join(out)


def link_list(heading, items) -> str:
    return (f'<h2 class="cj-sec">{esc(heading)}</h2><p>'
            + ", ".join(f'<a href="{esc(m["url"])}">{esc(m["id"])}</a>' for m in items) + "</p>")


def render_used_by(d) -> str:
    """Label version page: what was built from it, and which review packages wait on it (all text escaped)."""
    out = []
    if d["datasets"]:
        out.append(link_list("Datasets built from it", d["datasets"]))
    if d.get("models"):
        out.append(link_list("Models trained from those datasets", d["models"]))
    if d.get("waiting"):
        items = "".join(
            '<li class="cj-data-row">' + (f'<a href="{esc(w["url"])}">{esc(w["title"])}</a>' if w.get("url")
                                          else f'<span>{esc(w["title"])}</span>')
            + f'<span class="cj-mut cj-vis-id">{esc(w["id"])}</span>'
            f'<span class="cj-mut">{esc(plural(w["n"], "decision") + " not applied yet")}</span>'
            + (f'<a href="{esc(w["preview"])}">Preview apply</a>' if w.get("preview") else "") + '</li>'
            for w in d["waiting"])
        out.append(f'<h2 class="cj-sec">Waiting on it</h2><div class="cj-card"><ul class="cj-data-rows cj-vis-wait">'
                   f'{items}</ul></div>')
    return "".join(out)


def render_detail(d, error, extra="") -> str:
    kind = d["kind"]
    list_kind = DETAIL_LIST[kind]
    card_cls = "cj-card" + (" cj-data-hl" if d.get("production") else "")
    out = [breadcrumb(("Data", PREFIX + "/"), (KIND_TITLES[list_kind], f"{PREFIX}/list/{list_kind}")),
           f'<div class="cj-data-top"><h1 class="cj-wrap">{esc(d["title"])}</h1>'
           + "".join(chip(t, tone) for t, tone in d["chips"]) + "</div>",
           (f'<p class="cj-mut cj-vis-id">{esc(d["id"])}</p>' if d["title"] != d["id"] else ""),
           banner(error), f'<div class="{card_cls}">{render_fields(d["fields"])}</div>']
    if d.get("review_url"):
        out.append(f'<p class="cj-data-more"><a href="{esc(d["review_url"])}">Open in review</a></p>')
    out.append(f'<p class="cj-data-more"><a class="cj-data-lnk" href="{esc(lineage_href(kind, d["id"]))}">'
               + ("Lineage: what was made from it" if kind in SOURCE_ENTITIES else "Lineage: where this came from")
               + "</a></p>")
    if d.get("gap"):
        out.append(f'<p class="cj-mut">Acquisition details are not available: {esc(d["gap"])}</p>')
    if kind == "session":
        out.append(render_session_extras(d, extra))
    elif kind == "model":
        out.append(render_model_extras(d))
    else:
        if kind == "grid" and d["sessions"]:
            out.append(link_list("Imaged in", [{"id": f"Session {s['id']}", "url": s["url"]} for s in d["sessions"]]))
        if kind == "dataset" and d.get("class_map"):
            out.append(render_class_map(d["class_map"]))
        if kind == "dataset" and d["models"]:
            out.append(link_list("Models trained on it", d["models"]))
        if kind == "labels":
            out.append(render_used_by(d))
        heading = "Example images" if kind == "dataset" else "Images"
        out.append(f'<h2 class="cj-sec">{heading} ({d["n_images"]})</h2>' + render_thumbs(d))
    return "".join(out)


def render_class_map(cm) -> str:
    """#14: the remap from the source labels, collapsed: one line per training class, dropped classes last."""
    items = "".join(f"<li><b>{esc(t)}</b> ← {esc(', '.join(srcs))}</li>" for t, srcs in cm["lines"])
    if cm["dropped"]:
        items += f'<li class="cj-mut">dropped: {esc(", ".join(cm["dropped"]))}</li>'
    return ('<details class="cj-card cj-data-cmap"><summary>Class mapping (from the source labels)</summary>'
            f"<ul>{items}</ul></details>")


def render_lineage_node(node) -> str:
    status = node["status"]
    if status == "more":
        return f'<li class="cj-data-ln"><p class="cj-mut cj-data-lmore">{esc(node["title"])}</p></li>'
    title = esc(node["title"])
    if node.get("url") and status in ("ok", "cut", "loop"):
        title = f'<a href="{esc(node["url"])}">{title}</a>'
    links = ""
    if node.get("lineage") and status in ("ok", "cut"):
        links = f'<a class="cj-data-lnk" href="{esc(node["lineage"])}">Lineage from here</a>'
    chips = "".join(chip(t, tone) for t, tone in node["chips"])
    step = esc(node["step"] if status != "hidden" else "Step")
    if node.get("set_kind"):
        step = esc(node["set_kind"])
    parts = [f'<span class="cj-data-lstep">{step}</span>',
             f'<div class="cj-data-top"><b class="cj-wrap">{title}</b>{chips}{links}</div>']
    if status == "unknown" and node.get("recorded"):
        parts.append(f'<p class="cj-mut cj-data-lnote">Recorded as {esc(node["recorded"])}.</p>')
    if node.get("note"):
        parts.append(f'<p class="cj-mut cj-data-lnote">{esc(node["note"])}</p>')
    parts.append(render_fields(node["fields"]))
    if node.get("thumbs"):
        parts.append(render_thumbs(node["thumbs"], strip=True))
    card = f'<div class="cj-card cj-data-lc cj-data-ls-{esc(status)}">{"".join(parts)}</div>'
    kids = ""
    if node["parents"]:
        kids = ('<p class="cj-data-lfrom cj-mut">Made from</p><ol class="cj-data-lkids">'
                + "".join(render_lineage_node(p) for p in node["parents"]) + "</ol>")
    return f'<li class="cj-data-ln">{card}{kids}</li>'


def view_toggle(focus) -> str:
    """Tree | Graph, for an item with a graph (a lab graph node, or a session / grid: source_graph). Hidden until
    /data/data.js runs: without script the Tree is the page. A session's or grid's tree is only itself (and a grid's
    sessions), so its page opens on the Graph unless the person last chose Tree (data-default)."""
    if not focus:
        return ""
    default = ' data-default="graph"' if source_focus(focus) else ""
    return ('<div class="cj-data-graph-toggle" id="cj-data-view-toggle" role="group" aria-label="View"'
            f'{default} hidden>'
            '<button type="button" data-view="tree" aria-pressed="true">Tree</button>'
            '<button type="button" data-view="graph" aria-pressed="false">Graph</button></div>')


SOURCE_NOTES = {"session": "This session, its grids and uploads, and everything made from their images. "
                           "Arrows: parent → child (inputs).",
                "grid": "This grid, the sessions that imaged it, and everything made from its images. "
                        "Arrows: parent → child (inputs). Analyses appear on the session's graph (here too when "
                        "the session imaged only this grid)."}
SOURCE_EMPTY = "Not found or not visible to you."


def graph_panel(focus) -> str:
    if not focus:
        return ""
    ep = f"{PREFIX}/api/lineage/graph?focus={quote(focus, safe='')}&scope=item"
    src = source_focus(focus)
    return ('<div id="cj-data-graph-panel" hidden>'
            + graph_frame(ep, True, lab_graph_href(focus), SOURCE_NOTES[src[0]] if src else None,
                          SOURCE_EMPTY if src else None) + "</div>"
            f'<script src="{PREFIX}/data.js" defer></script>')


def render_lineage(tree, error, focus=None) -> str:
    kind, root = tree["kind"], tree["root"]
    s = tree["summary"]
    cut = f", {s['cut']} not followed" if s["cut"] else ""
    loop = f", {plural(s['loop'], 'loop')} in the registry" if s["loop"] else ""
    untraced = (f" {plural(s['untraced_images'], 'listed image')} not traced to an imaging session."
                if s.get("untraced_images") else "")
    summary = (f'{plural(s["steps"], "step")}: {s["known"]} found, {s["unknown"]} unknown, '
               f'{s["hidden"]} not visible to you{cut}{loop}.{untraced}')
    lab = (f'Graph: what was made from its images. <a href="{esc(lab_graph_href())}">Lab graph</a> shows how '
           'everything connects.' if source_focus(focus) else
           f'<a href="{esc(lab_graph_href(focus))}">Lab graph</a> shows it among everything else.')
    return (breadcrumb(("Data", PREFIX + "/"), (KIND_TITLES[DETAIL_LIST[kind]], f"{PREFIX}/list/{DETAIL_LIST[kind]}"),
                       (tree["title"], entity_href(kind, tree["id"])))
            + f'<h1 class="cj-wrap">Lineage of {esc(tree["title"])}</h1>'
            + '<p class="cj-mut cj-data-sub">The chain back to the raw imaging sessions, read from the registry. '
              f'A step the registry does not record is shown as unknown, never guessed. {lab}</p>'
            + banner(error) + view_toggle(focus)
            + '<div id="cj-data-tree-panel">' + f'<p class="cj-data-lsum">{esc(summary)}</p>'
            + '<ol class="cj-data-lroot">' + render_lineage_node(root) + "</ol></div>"
            + graph_panel(focus))


def graph_frame(endpoint: str, item: bool = False, lab_href: str = "", note=None, empty=None) -> str:
    """The graph's controls and empty canvas. /data/data.js finds the frame by id cj-data-graph-root and takes its
    endpoint (and, on the lab page, the page's own ?focus) from data-* attributes; no record text is in this HTML.
    note: the line beside the controls (a session's / grid's own; default: the lab's or an item's). empty: the text
    data.js shows when the focus comes back null (default: "This item is not in the lab graph.")."""
    attrs = (f'data-endpoint="{esc(endpoint)}"' + (' data-scope="item"' if item else ' data-focus-from-url="1"')
             + (f' data-empty="{esc(empty)}"' if empty else ""))
    note = note or ("Arrows: parent \u2192 child (inputs); the number on a session's arrow is its images in that "
                    "export. Dashed boxes: reviews between lanes. Dashed arrows: model predictions \u2192 review. "
                    "\u2605 = production." if not item else
                    "This item, what it was built from, and what was built from it. Arrows: parent \u2192 child "
                    "(inputs); the number on a session's arrow is its images in that export.")
    return ('<div class="cj-data-graph-bar">'
            '<label><input type="checkbox" id="cj-data-graph-hist"> <span id="cj-data-graph-histl">Show history'
            '</span></label>'
            '<label><input type="checkbox" id="cj-data-graph-an"> <span id="cj-data-graph-anl">Show analyses'
            '</span></label>'
            f'<span class="cj-mut cj-data-graph-note">{note}</span></div>'
            f'<div id="cj-data-graph-root" {attrs}>'
            '<p class="cj-mut cj-data-graph-status" id="cj-data-graph-status" role="status">Loading the graph\u2026</p>'
            '<div class="cj-data-graph-wrap" id="cj-data-graph-wrap"><div class="cj-data-graph-canvas" '
            'id="cj-data-graph-canvas"><svg id="cj-data-graph-svg" aria-hidden="true"></svg></div></div>'
            + ('<p class="cj-data-graph-more" id="cj-data-graph-more" hidden>'
               f'<a href="{esc(lab_href)}">More downstream: open lab graph</a></p>' if item else "")
            + '</div>')


def render_lab_graph(error) -> str:
    """The Lab graph page: a frame only. /data/data.js reads /data/api/lineage/graph (with this page's ?focus) and
    draws it, so no record text is in this HTML."""
    return (breadcrumb(("Data", PREFIX + "/")) + '<h1>Lab graph</h1>'
            '<p class="cj-mut cj-data-sub">How imaging sessions, label sets (made by a model, by people or by a '
            'review), datasets and models connect, read from the registry. Open a box for its lineage.'
            '</p>' + banner(error)
            + graph_frame(PREFIX + "/api/lineage/graph")
            + '<noscript><p class="cj-mut">The Lab graph needs JavaScript.</p></noscript>'
            + f'<script src="{PREFIX}/data.js" defer></script>')


def page_body(content: str) -> str:
    """data.html with the content put in ONE replace (text inside the content is never re-scanned)."""
    return TEMPLATE.read_text(encoding="utf-8").replace("{{content}}", content, 1)
