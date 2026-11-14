"""
Read-only pages for class maps and label schemas (round 14, contracts C1 / C2; the approved mockup's screen 2).

    GET /data/classmap/                 every class map (built-ins and user maps)
    GET /data/classmap/<cm>             "Class map · <name>": version and built-in pills, description, the table
                                        Label class -> Training class grouped by target (dropped = "dropped"),
                                        "Used by" (datasets built with it -> their training runs), "Versions"
    GET /data/schema/                   every label schema
    GET /data/schema/<schema>           "Label schema · <name>": version pill, maker and date, description, the table
                                        Attribute | Choices · keys, "Used by" (object sets the viewer may see),
                                        "Versions", and the note that it is read-only this round
Data's own roles (admin, advanced). Nothing here writes. Every value is escaped; a colour is shown only when it is
#rrggbb. Object sets are review packages: listed under "Used by" only when the viewer may see them (set_is_visible,
the rule of every other Data page), never counted otherwise.
"""
from __future__ import annotations

from urllib.parse import quote

from portal.modules.data import labelmeta as LM
from portal.modules.data.records import (
    PREFIX, _dict, _list, _text, entity_href, esc, plural, set_is_visible, short_date)


def classmap_href(cm_id) -> str:
    return f"{PREFIX}/classmap/{quote(str(cm_id), safe='')}"


def schema_href(schema_id) -> str:
    return f"{PREFIX}/schema/{quote(str(schema_id), safe='')}"


def _crumb(*parts) -> str:
    from portal.modules.data.render import breadcrumb
    return breadcrumb(*parts)


def _pill(text, tone="") -> str:
    from portal.modules.data.render import chip
    return chip(text, tone)


def _versions(items: dict, item_id: str, href) -> str:
    chain = LM.version_chain(items, item_id)
    if len(chain) < 2:
        v = _dict(items.get(item_id)).get("version")
        return (f'<h2 class="cj-sec">Versions</h2><p class="cj-mut">Only this version'
                f'{f" (v{esc(str(v))})" if isinstance(v, int) else ""}.</p>')
    bits = []
    for k in chain:
        v = _dict(items.get(k)).get("version")
        label = f"v{v}" if isinstance(v, int) and not isinstance(v, bool) else k
        bits.append(f'<b aria-current="page">{esc(label)} (this one)</b>' if k == item_id
                    else f'<a href="{esc(href(k))}">{esc(label)}</a>')
    return f'<h2 class="cj-sec">Versions</h2><p class="cj-data-chain">{" → ".join(bits)}</p>'


STYLE = """<style>
.cj-lm-table{border-collapse:collapse;width:100%;font-size:.9rem}
.cj-lm-table th,.cj-lm-table td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line);vertical-align:top}
.cj-lm-table th{font-size:.75rem;text-transform:uppercase;letter-spacing:.04em;color:var(--mut);font-weight:600}
.cj-lm-table td.cj-lm-t{font-weight:600}
.cj-lm-drop{color:var(--mut);font-style:italic}
.cj-lm-sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:-1px}
.cj-lm-ch{display:inline-flex;gap:5px;align-items:center;margin:2px 8px 2px 0;white-space:nowrap}
.cj-lm-ch kbd{font:11px var(--cj-mono,ui-monospace,monospace);border:1px solid var(--line);border-radius:4px;padding:0 4px;color:var(--mut)}
.cj-lm-note{padding:9px 12px;border:1px solid var(--line);border-radius:6px;background:var(--accbg);font-size:.85rem}
.cj-lm-wrap{overflow-x:auto}
</style>"""


# ---------------------------------------------------------------- class maps
def classmap_used_by(records, cm_id) -> list:
    """[{id, url, runs: [ids], models: [ids]}] for every dataset recorded with class_map_id == cm_id."""
    out = []
    for ds_id, ds in sorted(records["dataset"].items()):
        if _dict(ds).get("class_map_id") != cm_id:
            continue
        runs = sorted(r for r, rec in records.get("run", {}).items() if _dict(rec).get("dataset") == ds_id)
        models = sorted(m for m, rec in records.get("model", {}).items() if _dict(rec).get("dataset") == ds_id)
        out.append({"id": ds_id, "url": entity_href("dataset", ds_id), "runs": runs, "models": models})
    return out


def classmap_detail(records, cm_id, maps=None) -> dict | None:
    maps = LM.list_classmaps() if maps is None else maps
    cm = maps.get(cm_id) or LM.load_classmap(cm_id)
    if not cm:
        return None
    maps = {**maps, cm_id: cm}
    return {"id": cm_id, "cm": cm, "view": LM.classmap_view(cm), "used_by": classmap_used_by(records, cm_id),
            "maps": maps}


def render_classmap(d) -> str:
    cm, view = d["cm"], d["view"]
    name = _text(cm.get("name")) or d["id"]
    v = cm.get("version")
    pills = (_pill(f"v{v}", "acc") if isinstance(v, int) and not isinstance(v, bool) else "") + \
        (_pill("built-in") if cm.get("maker") == "builtin" else "")
    maker = "" if cm.get("maker") == "builtin" else _text(cm.get("maker"))
    out = [STYLE, _crumb(("Data", PREFIX + "/"), ("Class maps", PREFIX + "/classmap/")),
           f'<div class="cj-data-top"><h1 class="cj-wrap">Class map · {esc(name)}</h1>{pills}</div>',
           f'<p class="cj-mut cj-vis-id">{esc(d["id"])}</p>']
    if _text(cm.get("description")):
        out.append(f'<p>{esc(cm["description"])}</p>')
    facts = []
    if maker:
        facts.append(f"Made by {esc(maker)}")
    src = _text(cm.get("source_schema"))
    if src:
        facts.append(f'From label schema <a href="{esc(schema_href(src))}">{esc(src)}</a>')
    ev = [str(c) for c in _list(cm.get("ev_classes")) if isinstance(c, str)]
    if ev:
        facts.append(f'Counted as EV: {esc(", ".join(ev))}')
    if facts:
        out.append(f'<p class="cj-mut">{" · ".join(facts)}</p>')
    rows = []
    for target, srcs in view["targets"]:
        if not srcs:
            rows.append(f'<tr><td class="cj-mut">(no label class)</td><td class="cj-lm-t">{esc(target)}</td></tr>')
        for k, s in enumerate(srcs):
            cell = (f'<td class="cj-lm-t" rowspan="{len(srcs)}">{esc(target)}</td>' if k == 0 else "")
            rows.append(f"<tr><td>{esc(s)}</td>{cell}</tr>")
    for s in view["dropped"]:
        rows.append(f'<tr><td>{esc(s)}</td><td class="cj-lm-drop">dropped</td></tr>')
    out.append('<div class="cj-card cj-lm-wrap"><table class="cj-lm-table"><thead><tr><th scope="col">Label class</th>'
               f'<th scope="col">→ Training class</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>')
    out.append('<h2 class="cj-sec">Used by</h2>')
    if d["used_by"]:
        items = []
        for u in d["used_by"]:
            tail = []
            if u["runs"]:
                tail.append("runs " + ", ".join(esc(r) for r in u["runs"]))
            if u["models"]:
                tail.append("models " + ", ".join(f'<a href="{esc(entity_href("model", m))}">{esc(m)}</a>'
                                                   for m in u["models"]))
            items.append(f'<li class="cj-data-row"><span class="cj-data-top"><a href="{esc(u["url"])}">'
                         f'{esc(u["id"])}</a></span>'
                         + (f'<span class="cj-mut">→ {" · ".join(tail)}</span>' if tail else
                            '<span class="cj-mut">no training run yet</span>') + "</li>")
        out.append(f'<div class="cj-card"><ul class="cj-data-rows">{"".join(items)}</ul></div>')
    else:
        out.append('<p class="cj-mut">No dataset records this class map yet.</p>')
    out.append(_versions(d["maps"], d["id"], classmap_href))
    return "".join(out)


def render_classmap_list(maps: dict) -> str:
    items = []
    for cm_id, cm in sorted(maps.items()):
        sub = " · ".join(x for x in (
            f'v{cm["version"]}' if isinstance(cm.get("version"), int) else "",
            "built-in" if cm.get("maker") == "builtin" else _text(cm.get("maker")) or "",
            plural(len(_list(cm.get("classes"))), "training class")) if x)
        items.append(f'<li class="cj-data-row"><span class="cj-data-top"><a href="{esc(classmap_href(cm_id))}">'
                     f'{esc(_text(cm.get("name")) or cm_id)}</a></span><span class="cj-mut cj-vis-id">{esc(cm_id)}</span>'
                     f'<span class="cj-mut">{esc(sub)}</span></li>')
    body = (f'<ul class="cj-data-rows">{"".join(items)}</ul>' if items
            else '<p class="cj-mut cj-data-empty">No class maps yet.</p>')
    return (_crumb(("Data", PREFIX + "/")) + '<h1>Class maps</h1><p class="cj-mut cj-data-sub">How label classes '
            'become training classes (merged, renamed or dropped) when a dataset is built.</p>'
            f'<div class="cj-card">{body}</div>')


# ---------------------------------------------------------------- label schemas
def object_set_schema(rec) -> str | None:
    """The label schema an object set's registry line names (label_schema, else schema when it is an ls_ id)."""
    for k in ("label_schema", "schema"):
        v = _text(_dict(rec).get(k))
        if v and v.startswith("ls_"):
            return v
    return None


def schema_used_by(records, viewer, schema_id) -> list:
    """[{id, url, title}] for object sets on this schema that the viewer may see."""
    out = []
    for pid, rec in sorted(records["review_package"].items()):
        rec = _dict(rec)
        if rec.get("kind") != "objects" or object_set_schema(rec) != schema_id:
            continue
        if not set_is_visible(viewer, "review_package", {**rec, "id": pid}):
            continue
        n = rec.get("n_items")
        out.append({"id": pid, "url": entity_href("images", pid),
                    "sub": plural(n, "item") if isinstance(n, int) and not isinstance(n, bool) else ""})
    return out


def schema_detail(records, viewer, registry_root, schema_id, schemas=None) -> dict | None:
    schemas = LM.list_schemas(registry_root) if schemas is None else schemas
    s = schemas.get(schema_id) or LM.load_schema(schema_id, registry_root)
    if not s:
        return None
    line = LM.schema_lines(registry_root).get(schema_id) or {}
    return {"id": schema_id, "schema": s, "line": line, "rows": LM.schema_rows(s),
            "used_by": schema_used_by(records, viewer, schema_id), "schemas": {**schemas, schema_id: s}}


def render_schema(d) -> str:
    s, line = d["schema"], d["line"]
    name = _text(s.get("name")) or d["id"]
    v = s.get("version")
    pills = _pill(f"v{v}", "acc") if isinstance(v, int) and not isinstance(v, bool) else ""
    out = [STYLE, _crumb(("Data", PREFIX + "/"), ("Label schemas", PREFIX + "/schema/")),
           f'<div class="cj-data-top"><h1 class="cj-wrap">Label schema · {esc(name)}</h1>{pills}</div>',
           f'<p class="cj-mut cj-vis-id">{esc(d["id"])}</p>']
    maker = _text(s.get("maker")) or _text(line.get("maker")) or _text(line.get("who"))
    when = short_date(line.get("created") or line.get("logged") or s.get("created"))
    facts = [x for x in ((f"Made by {maker}" if maker else ""), (f"registered {when}" if when else "")) if x]
    if facts:
        out.append(f'<p class="cj-mut">{esc(" · ".join(facts))}</p>')
    if _text(s.get("description")):
        out.append(f'<p>{esc(s["description"])}</p>')
    classes = []
    for c in _list(s.get("classes")):
        c = _dict(c)
        if not _text(c.get("name")):
            continue
        col = c.get("color") if isinstance(c.get("color"), str) and LM.COLOR_RE.match(c["color"]) else None
        sw = f'<span class="cj-lm-sw" style="background:{col}"></span>' if col else ""
        key = f'<kbd>{esc(c["key"])}</kbd>' if _text(c.get("key")) else ""
        classes.append(f'<span class="cj-lm-ch">{sw}{esc(c["name"])}{key}</span>')
    if classes:
        out.append(f'<p><b>Classes</b> {"".join(classes)}</p>')
    rows = []
    for r in d["rows"]:
        ch = "".join(f'<span class="cj-lm-ch">{esc(lbl)}{f"<kbd>{esc(k)}</kbd>" if k else ""}</span>'
                     for lbl, k in r["choices"])
        req = ' <span class="cj-mut">(required)</span>' if r["required"] else ""
        rows.append(f'<tr><td class="cj-lm-t">{esc(r["label"])}{req}<br><span class="cj-mut cj-vis-id">'
                    f'{esc(r["id"])} · {esc(r["kind"])}</span></td><td>{ch or "—"}</td></tr>')
    out.append('<div class="cj-card cj-lm-wrap"><table class="cj-lm-table"><thead><tr><th scope="col">Attribute</th>'
               '<th scope="col">Choices · keys</th></tr></thead><tbody>'
               + ("".join(rows) or '<tr><td colspan="2" class="cj-mut">No object attributes.</td></tr>')
               + "</tbody></table></div>")
    out.append('<h2 class="cj-sec">Used by</h2>')
    if d["used_by"]:
        out.append('<div class="cj-card"><ul class="cj-data-rows">' + "".join(
            f'<li class="cj-data-row"><span class="cj-data-top"><a href="{esc(u["url"])}">{esc(u["id"])}</a></span>'
            f'<span class="cj-mut">Object set{(" · " + esc(u["sub"])) if u["sub"] else ""}</span></li>'
            for u in d["used_by"]) + "</ul></div>")
    else:
        out.append('<p class="cj-mut">No object set you can see uses this schema.</p>')
    out.append(_versions(d["schemas"], d["id"], schema_href))
    out.append('<p class="cj-lm-note">Read-only this round. A new version is registered from a file with '
               '<code>cryoev schema register</code>.</p>')
    return "".join(out)


def render_schema_list(schemas: dict) -> str:
    items = []
    for sid, s in sorted(schemas.items()):
        n = len([a for a in _list(s.get("object_attrs")) if _text(_dict(a).get("id"))])
        sub = " · ".join(x for x in (f'v{s["version"]}' if isinstance(s.get("version"), int) else "",
                                     _text(s.get("maker")) or "", plural(n, "attribute")) if x)
        items.append(f'<li class="cj-data-row"><span class="cj-data-top"><a href="{esc(schema_href(sid))}">'
                     f'{esc(_text(s.get("name")) or sid)}</a></span><span class="cj-mut cj-vis-id">{esc(sid)}</span>'
                     f'<span class="cj-mut">{esc(sub)}</span></li>')
    body = (f'<ul class="cj-data-rows">{"".join(items)}</ul>' if items
            else '<p class="cj-mut cj-data-empty">No label schemas yet.</p>')
    return (_crumb(("Data", PREFIX + "/")) + '<h1>Label schemas</h1><p class="cj-mut cj-data-sub">What reviewers '
            'answer about each object (attributes, choices and their keys). Read only this round.</p>'
            f'<div class="cj-card">{body}</div>')
