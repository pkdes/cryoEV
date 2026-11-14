"""
Shared list helper for the portal's list pages (round 13, feedback #22): filter, sort and group rows from a plain GET
form, with the state kept in the URL. Server-side only (no JS). Used by Results, Predict "My analyses" and the Reviews
hub (portal/modules/reviews.py).

INTERFACE CONTRACT (fixed by the PM before the build; B2 owns this file and fills the bodies; B1 and B2 call it):

    facets(ps_dir, request, *, viewer=None, hidden=None) -> dict
        Group/filter keys for one prediction set, from its request.json (`request`, already parsed) and folder:
          person   owner id (agent-made sets stay under their owner)
          via      owner_via or None (the "include agent-made" toggle)
          created  epoch seconds (float) or None
          session  request source.session, else "upload · YYYY-MM-DD"
          source   label of the root of the derived_from chain (prediction_set / dataset links only), else the set's
                   own display name
          model    request model id
          samples  list of sample labels from images.csv (may be empty)
        Every key that names another item passes through portal.modules.data.records.ref_hidden (via `hidden`,
        a callable(kind, id) -> bool for this viewer), so a private set id never shows as a group heading.

    parse_query(query_params) -> ListQuery
        Reads ?person=&agents=&from=&to=&model=&status=&q=&sort=&group= (unknown values fall back to defaults).
        Defaults: person any, agents on, no dates, model any, status "open" on the Reviews hub and "any" elsewhere
        (the caller passes default_status), q "", sort "newest", group "none".

    apply(rows, q, *, key=lambda r: r["facets"]) -> list
        Filter + sort rows (dicts). Each row's facets come from key(row); rows also carry "status" and "title".

    group(rows, by, *, key=lambda r: r["facets"]) -> list[tuple[str, list]]
        by in {"none","person","session","source","model","sample"}; "none" -> [("", rows)].
        sample: a row with <= 3 samples sits in each sample's group; more -> "Mixed (more than 3 samples)".

    controls_html(q, *, action, people, models, statuses, show_group=True) -> str
        The GET form (escaped), state prefilled from q. Each control has a stable id.

ListQuery is a frozen dataclass with the fields above plus .to_query() -> str for links.

How the contract is filled (B2):
  - facets: `source` walks provenance.derived_from (request.json `provenance`, round 11b): the FIRST entry of kind
    prediction_set or dataset at each step, at most MAX_DEPTH steps, reading <predict root>/<ps>/request.json for a
    prediction-set parent (the set's own folder's parent is the predict root). The walk stops at the last item the
    viewer may see: a link that `hidden` says is private is never followed or named. A dataset is a root, labelled
    "<id> (dataset)". A cycle picks the member with the smallest id, so every set of the cycle lands in one group.
    With hidden=None the walk is not taken at all (fail closed: the set's own name). session / model keys also pass
    through hidden ("Not visible to you" if it ever says so). samples are cached by images.csv path + mtime + size.
  - Rows other than prediction sets (the hub's label reviews, audits) may have facets {}: apply / group then read the
    row's own owner / via / created (or last) / model / n_images fields, and samples [].
  - status filter: "any" keeps all; "open" keeps every status not in CLOSED ("finished", "applied"); any other value
    must equal the row's "status".
  - agents: the form sends a hidden agents=0 before the checkbox's agents=1, so an unticked box is told apart from no
    form at all (no agents value = the default, on).
  - Rendering shared by Results and Predict: rows_html(groups, by, ...) renders standard row dicts (id, title, href,
    owner, via, model, n_images, samples, created, status_html, open_href, actions_html, row_class) as one list,
    with <details open> groups (count + image total). STYLE is the inline <style> of the list (cj.css tokens only).
"""
from __future__ import annotations

import csv
import html
import json
import re
import threading
from dataclasses import dataclass, fields as dc_fields
from datetime import date, datetime
from pathlib import Path
from urllib.parse import urlencode

from portal import names as NM

GROUPS = ("none", "person", "session", "source", "model", "sample")
SORTS = ("newest", "name", "size")
MIXED = "Mixed (more than 3 samples)"

GROUP_LABELS = {"none": "None", "person": "Person", "session": "Imaging session", "source": "Source images",
                "model": "Model", "sample": "Sample"}
GROUP_HINTS = {"none": "Newest first, no groups.", "person": "Agent-made sets stay under their owner.",
               "session": 'Sets with no session group as "upload · date".',
               "source": "Sets derived from the same images, via derived_from.",
               "model": "Grouped by the model that made the predictions.",
               "sample": "A set can sit in several sample groups."}
SORT_LABELS = {"newest": "Newest", "name": "Name", "size": "Size"}
CLOSED = ("finished", "applied")
MAX_DEPTH = 12                         # derived_from steps followed at most
MAX_SAMPLES_IN_GROUPS = 3
HIDDEN_LABEL = "Not visible to you"
WALK_KINDS = ("prediction_set", "dataset")
PS_RE = re.compile(r"^ps_\d{8}_[0-9a-f]{6}$")
PERSON_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
MODEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,99}$")
STATUS_RE = re.compile(r"^[a-z][a-z_]{0,31}$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
Q_MAX = 100

esc = html.escape


@dataclass(frozen=True)
class ListQuery:
    person: str = "any"
    agents: bool = True
    date_from: str = ""
    date_to: str = ""
    model: str = "any"
    status: str = "any"
    q: str = ""
    sort: str = "newest"
    group: str = "none"

    def to_query(self) -> str:
        """'person=..&..' with the fields that differ from the defaults, plus status always (its default depends on
        the page). No leading '?'."""
        names = {"date_from": "from", "date_to": "to"}
        out = []
        for f in dc_fields(self):
            v = getattr(self, f.name)
            if f.name != "status" and v == f.default:
                continue
            out.append((names.get(f.name, f.name), ("1" if v else "0") if isinstance(v, bool) else v))
        return urlencode(out)


# ---------------------------------------------------------------- facets
_lock = threading.Lock()
_samples_cache: dict = {}              # str(images.csv) -> ((mtime_ns, size), [labels])
_req_cache: dict = {}                  # str(request.json) -> ((mtime_ns, size), dict | None)
CACHE_MAX = 2048


def _stamp(p: Path):
    try:
        s = p.stat()
    except OSError:
        return None
    return (s.st_mtime_ns, s.st_size)


def _cached(store: dict, p: Path, compute):
    stamp = _stamp(p)
    if stamp is None:
        return None
    key = str(p)
    with _lock:
        hit = store.get(key)
        if hit is not None and hit[0] == stamp:
            return hit[1]
    value = compute(p)
    with _lock:
        store[key] = (stamp, value)
        if len(store) > CACHE_MAX:
            store.pop(next(iter(store)))
    return value


def _read_samples(p: Path) -> list:
    out, seen = [], set()
    try:
        with open(p, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                label = (row.get("sample_label") or "").strip()
                if label and label not in seen:
                    seen.add(label)
                    out.append(label)
    except (OSError, ValueError, csv.Error, UnicodeDecodeError):
        return []
    return out


def sample_labels(ps_dir) -> list:
    """images.csv sample_label values, unique, in file order (cached by mtime + size)."""
    got = _cached(_samples_cache, Path(ps_dir) / "images.csv", _read_samples)
    return list(got or [])


def _read_request(p: Path):
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return None
    return doc if isinstance(doc, dict) else None


def _parent_request(predict_root: Path, ps_id: str):
    if not PS_RE.match(ps_id or ""):
        return None
    doc = _cached(_req_cache, predict_root / ps_id / "request.json", _read_request)
    return doc if isinstance(doc, dict) and doc.get("ps") == ps_id else None


def _epoch(iso):
    try:
        return datetime.fromisoformat(str(iso)).timestamp()
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def _text(x):
    return x.strip() if isinstance(x, str) and x.strip() else None


def _name(request) -> str:
    """names.display_name, safe for a request.json whose source is not an object."""
    req = request if isinstance(request.get("source"), dict) or "source" not in request else \
        {**request, "source": {}}
    return NM.display_name(req)


def _walk_link(request):
    """The first derived_from entry of a walked kind with a usable id: (kind, id) or None."""
    prov = request.get("provenance") if isinstance(request, dict) else None
    entries = prov.get("derived_from") if isinstance(prov, dict) else None
    for e in entries if isinstance(entries, list) else ():
        if isinstance(e, dict) and e.get("kind") in WALK_KINDS and _text(e.get("id")):
            return e["kind"], e["id"].strip()
    return None


def source_root(ps_dir, request, hidden) -> str:
    """The label of the root of the set's derived_from chain (see the module doc)."""
    ps_dir = Path(ps_dir)
    own_id = request.get("ps") if isinstance(request.get("ps"), str) else ps_dir.name
    own = _name(request)
    if hidden is None:
        return own
    path = [(own_id, own)]                 # (id, label) from the set to the current root
    cur = request
    for _ in range(MAX_DEPTH):
        link = _walk_link(cur)
        if link is None:
            break
        kind, rid = link
        try:
            if hidden(kind, rid):
                break                          # never followed, never named
        except Exception:                      # noqa: BLE001 - a broken check fails closed
            break
        if kind == "dataset":
            return f"{rid} (dataset)"
        ids = [i for i, _ in path]
        if rid in ids:                         # a cycle: its smallest id names the group
            cycle = path[ids.index(rid):]
            return min(cycle)[1]
        parent = _parent_request(ps_dir.parent, rid)
        if parent is None:
            return rid                         # a visible id whose folder is gone: still one group
        path.append((rid, _name(parent)))
        cur = parent
    return path[-1][1]


def hidden_for(cfg, user, can_see):
    """The `hidden` callable for this viewer: portal.modules.data.records.ref_hidden over the registry (read every
    10 s, shared). A registry that can't be read gives no records, so a private id falls back to its request.json
    owner, else admins only (ref_hidden fails closed)."""
    from portal.modules.data import records as REC
    recs, _err = REC.read_registry(getattr(cfg, "registry", None))
    viewer = REC.Viewer(can_see, user, cfg)
    return lambda kind, rid: REC.ref_hidden(recs, viewer, rid, kind)


def _checked(hidden, kind, value):
    if value is None or hidden is None:
        return value
    try:
        return HIDDEN_LABEL if hidden(kind, value) else value
    except Exception:                          # noqa: BLE001
        return HIDDEN_LABEL


def facets(ps_dir, request, *, viewer=None, hidden=None) -> dict:
    request = request if isinstance(request, dict) else {}
    created = _epoch(request.get("created"))
    src = request.get("source") if isinstance(request.get("source"), dict) else {}
    session = _text(src.get("session"))
    if session:
        session = _checked(hidden, "session", session)
    else:
        session = "upload · " + (datetime.fromtimestamp(created).date().isoformat() if created is not None else "?")
    model = _text(request.get("model"))
    return {"person": _text(request.get("owner")), "via": _text(request.get("owner_via")), "created": created,
            "session": session, "source": source_root(ps_dir, request, hidden),
            "model": _checked(hidden, "model", model), "samples": sample_labels(ps_dir)}


# ---------------------------------------------------------------- the query
def _values(qp, name) -> list:
    getlist = getattr(qp, "getlist", None)
    if callable(getlist):
        vals = getlist(name)
    else:
        v = qp.get(name) if hasattr(qp, "get") else None
        vals = v if isinstance(v, (list, tuple)) else ([] if v is None else [v])
    return [str(v) for v in vals]


def _one(qp, name):
    vals = _values(qp, name)
    return vals[-1].strip() if vals else None


def _date(v) -> str:
    if not v or not DATE_RE.match(v):
        return ""
    try:
        date.fromisoformat(v)
    except ValueError:
        return ""
    return v


def parse_query(query_params, *, default_status: str = "any") -> ListQuery:
    qp = query_params if query_params is not None else {}
    d = ListQuery()
    person = _one(qp, "person")
    agents_vals = [v.strip().lower() for v in _values(qp, "agents")]
    agents = d.agents
    if any(v in ("1", "on", "true") for v in agents_vals):
        agents = True
    elif any(v in ("0", "off", "false") for v in agents_vals):
        agents = False
    model, status = _one(qp, "model"), _one(qp, "status")
    q = _one(qp, "q") or ""
    q = "" if any(ord(c) < 32 or ord(c) == 127 for c in q) else q[:Q_MAX]
    sort, grp = _one(qp, "sort"), _one(qp, "group")
    default_status = default_status if STATUS_RE.match(default_status or "") else "any"
    return ListQuery(person=person if person and (person == "any" or PERSON_RE.match(person)) else d.person,
                     agents=agents, date_from=_date(_one(qp, "from")), date_to=_date(_one(qp, "to")),
                     model=model if model and (model == "any" or MODEL_RE.match(model)) else d.model,
                     status=status if status and STATUS_RE.match(status) else default_status,
                     q=q, sort=sort if sort in SORTS else d.sort, group=grp if grp in GROUPS else d.group)


# ---------------------------------------------------------------- filter, sort, group
def _f(row, key) -> dict:
    try:
        f = key(row)
    except (KeyError, TypeError):
        f = None
    return f if isinstance(f, dict) else {}


def _num(v):
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def person_of(row, f):
    return f.get("person") if "person" in f else _text(row.get("owner"))


def via_of(row, f):
    return f.get("via") if "via" in f else _text(row.get("via"))


def created_of(row, f):
    for v in (f.get("created"), row.get("created"), row.get("last")):
        if _num(v) is not None:
            return _num(v)
    return None


def model_of(row, f):
    return f.get("model") if "model" in f else _text(row.get("model"))


def size_of(row):
    for k in ("n_images", "images"):
        if _num(row.get(k)) is not None:
            return _num(row.get(k))
    return None


def samples_of(row, f) -> list:
    s = f.get("samples")
    return [x for x in s if isinstance(x, str)] if isinstance(s, list) else []


def _id(row) -> str:
    return str(row.get("id") or row.get("ps") or "")


def _day(ts):
    try:
        return datetime.fromtimestamp(ts).date().isoformat()
    except (OverflowError, OSError, ValueError):
        return None


def _keep(row, f, q: ListQuery) -> bool:
    if q.person != "any" and person_of(row, f) != q.person:
        return False
    if not q.agents and via_of(row, f):
        return False
    if q.date_from or q.date_to:
        ts = created_of(row, f)
        day = _day(ts) if ts is not None else None
        if day is None or (q.date_from and day < q.date_from) or (q.date_to and day > q.date_to):
            return False
    if q.model != "any" and model_of(row, f) != q.model:
        return False
    status = row.get("status")
    if q.status == "open":
        if status in CLOSED:
            return False
    elif q.status != "any" and status != q.status:
        return False
    if q.q:
        needle = q.q.casefold()
        hay = " ".join(str(x) for x in (row.get("title") or "", _id(row)))
        if needle not in hay.casefold():
            return False
    return True


def apply(rows, q: ListQuery, *, key=lambda r: r["facets"]) -> list:
    kept = [(r, _f(r, key)) for r in rows]
    kept = [(r, f) for r, f in kept if _keep(r, f, q)]
    if q.sort == "name":
        kept.sort(key=lambda t: (str(t[0].get("title") or "").casefold(), _id(t[0])))
    elif q.sort == "size":
        kept.sort(key=lambda t: _id(t[0]), reverse=True)
        kept.sort(key=lambda t: (size_of(t[0]) is None, -(size_of(t[0]) or 0)))
    else:
        kept.sort(key=lambda t: _id(t[0]), reverse=True)
        kept.sort(key=lambda t: (created_of(t[0], t[1]) is None, -(created_of(t[0], t[1]) or 0)))
    return [r for r, _ in kept]


def _group_keys(row, f, by) -> list:
    if by == "person":
        return [person_of(row, f) or "Unknown"]
    if by == "session":
        return [f.get("session") or "No session"]
    if by == "source":
        return [f.get("source") or str(row.get("title") or _id(row) or "?")]
    if by == "model":
        return [model_of(row, f) or "No model"]
    if by == "sample":
        s = samples_of(row, f)
        if not s:
            return ["No sample label"]
        return [MIXED] if len(s) > MAX_SAMPLES_IN_GROUPS else list(dict.fromkeys(s))
    return [""]


def group(rows, by: str, *, key=lambda r: r["facets"]) -> list:
    rows = list(rows)
    if by not in GROUPS or by == "none":
        return [("", rows)]
    out: dict = {}
    for r in rows:
        for k in _group_keys(r, _f(r, key), by):
            out.setdefault(k, []).append(r)
    return list(out.items())


# ---------------------------------------------------------------- the form
def _pairs(items) -> list:
    out = []
    for it in items or ():
        if isinstance(it, (tuple, list)) and len(it) == 2:
            out.append((str(it[0]), str(it[1])))
        elif it is not None:
            out.append((str(it), str(it)))
    return out


def _select(cid, name, label, current, options, any_label) -> str:
    opts = [("any", any_label)] + [(v, t) for v, t in options if v != "any"]
    if current not in [v for v, _ in opts]:
        opts.append((current, current))          # a value from the URL that has no row now: still shown as chosen
    body = "".join(f'<option value="{esc(v)}"{" selected" if v == current else ""}>{esc(t)}</option>'
                   for v, t in opts)
    return (f'<label class="cj-list-f" for="{cid}"><span>{esc(label)}</span>'
            f'<select id="{cid}" name="{name}">{body}</select></label>')


def controls_html(q: ListQuery, *, action: str, people, models, statuses, show_group: bool = True) -> str:
    people, models = _pairs(people), _pairs(models)
    stat = [(v, t) for v, t in _pairs(statuses) if v != "any"]
    row1 = []
    if people:
        row1.append(_select("cj-list-person", "person", "Person", q.person, people, "Anyone"))
    row1.append('<label class="cj-list-f cj-list-chk" for="cj-list-agents">'
                '<input type="hidden" name="agents" value="0">'
                f'<input type="checkbox" id="cj-list-agents" name="agents" value="1"{" checked" if q.agents else ""}>'
                '<span>include agent-made</span></label>')
    row1.append(f'<span class="cj-list-f cj-list-dates"><label for="cj-list-from">From</label>'
                f'<input type="date" id="cj-list-from" name="from" value="{esc(q.date_from)}">'
                f'<label for="cj-list-to">to</label>'
                f'<input type="date" id="cj-list-to" name="to" value="{esc(q.date_to)}"></span>')
    if models:
        row1.append(_select("cj-list-model", "model", "Model", q.model, models, "Any"))
    if stat:
        row1.append(_select("cj-list-status", "status", "Status", q.status, stat, "Any"))
    row1.append(f'<label class="cj-list-f cj-list-search" for="cj-list-q"><span>Search</span>'
                f'<input type="search" id="cj-list-q" name="q" value="{esc(q.q)}" maxlength="{Q_MAX}" '
                f'placeholder="Search names and ids"></label>')
    seg = "".join(f'<label class="cj-list-segb" for="cj-list-sort-{k}"><input type="radio" id="cj-list-sort-{k}" '
                  f'name="sort" value="{k}"{" checked" if q.sort == k else ""}><span>{esc(SORT_LABELS[k])}</span>'
                  f'</label>' for k in SORTS)
    row2 = [f'<fieldset class="cj-list-seg" id="cj-list-sort"><legend>Sort</legend>{seg}</fieldset>']
    if show_group:
        gopts = "".join(f'<option value="{k}"{" selected" if q.group == k else ""}>{esc(GROUP_LABELS[k])}</option>'
                        for k in GROUPS)
        row2.append(f'<label class="cj-list-f" for="cj-list-group"><span>Group by</span>'
                    f'<select id="cj-list-group" name="group">{gopts}</select></label>'
                    f'<span class="cj-mut cj-list-hint" id="cj-list-hint">{esc(GROUP_HINTS.get(q.group, ""))}</span>')
    row2.append(f'<span class="cj-list-btns"><button type="submit" class="cj-list-go" id="cj-list-go">Apply</button>'
                f'<a class="cj-list-reset" id="cj-list-reset" href="{esc(action)}">Reset</a></span>')
    return (f'<form class="cj-list-form" id="cj-list-form" method="get" action="{esc(action)}" role="search" '
            f'aria-label="Filter the list"><div class="cj-list-r1">{"".join(row1)}</div>'
            f'<div class="cj-list-r2">{"".join(row2)}</div></form>')


# ---------------------------------------------------------------- the list (Results + Predict)
def when_text(ts) -> str:
    if _num(ts) is None:
        return ""
    try:
        return NM.when(datetime.fromtimestamp(ts).isoformat())
    except (OverflowError, OSError, ValueError):
        return ""


def owner_html(owner, via) -> str:
    out = f'<span class="cj-list-who">{esc(owner or "?")}</span>'
    if via and owner:
        from data_utils import actor
        out += f'<span class="cj-list-via">{esc(actor.label(owner, via))}</span>'
    return out


def row_html(r: dict) -> str:
    samples = [s for s in r.get("samples") or [] if isinstance(s, str)]
    n_img = r.get("n_images")
    title = esc(str(r.get("title") or r.get("id") or "?"))
    name = (f'<a class="cj-list-t {esc(r.get("title_class") or "")}" href="{esc(r["href"])}">'
            f'<b class="cj-wrap">{title}</b></a>' if r.get("href") else f'<b class="cj-wrap">{title}</b>')
    open_ = (f'<a class="cj-list-open" href="{esc(r["open_href"])}">Open</a>' if r.get("open_href") else "")
    return (f'<li class="cj-list-row {esc(r.get("row_class") or "")}" data-id="{esc(str(r.get("id") or ""))}">'
            f'<div class="cj-list-c cj-list-name">{name}'
            f'<span class="cj-mut cj-list-id cj-wrap">{esc(str(r.get("id") or ""))}</span></div>'
            f'<div class="cj-list-c cj-list-owner">{owner_html(r.get("owner"), r.get("via"))}</div>'
            f'<span class="cj-list-c cj-list-model cj-wrap" data-k="model">{esc(str(r.get("model") or ""))}</span>'
            f'<span class="cj-list-c cj-list-num" data-k="images">'
            f'{esc(str(int(n_img))) if _num(n_img) is not None else ""}<small> images</small></span>'
            f'<span class="cj-list-c cj-list-num" data-k="samples" title="{esc("; ".join(samples))}">'
            f'{len(samples)}<small> {"sample" if len(samples) == 1 else "samples"}</small></span>'
            f'<span class="cj-list-c cj-list-when">{esc(when_text(r.get("created")))}</span>'
            f'<span class="cj-list-c cj-list-status">{r.get("status_html") or ""}</span>'
            f'<span class="cj-list-c cj-list-act">{r.get("actions_html") or ""}{open_}</span></li>')


HEAD = ('<div class="cj-list-head" aria-hidden="true"><span class="cj-list-name">Name</span>'
        '<span class="cj-list-owner">Owner</span><span class="cj-list-model">Model</span>'
        '<span class="cj-list-num">Images</span><span class="cj-list-num">Samples</span>'
        '<span class="cj-list-when">Created</span><span class="cj-list-status">Status</span>'
        '<span class="cj-list-act"></span></div>')


def plural(n, word) -> str:
    return f"{n} {word}" + ("" if n == 1 else "s")


def group_label(by: str, label: str) -> str:
    return ("Source: " + label) if by == "source" else label


def rows_html(groups, by: str, *, shown: int, total: int, empty_text="No sets match these filters.",
              ul_class="") -> str:
    """The list: a count line, a column header, then one <ul> (by none) or one <details open> per group."""
    count = f'<p class="cj-mut cj-list-count" id="cj-list-count">{shown} of {plural(total, "set")}</p>'
    if shown == 0:
        return count + f'<div class="cj-list"><p class="cj-list-empty">{esc(empty_text)}</p></div>'
    parts = []
    for n, (label, rows) in enumerate(groups):
        body = f'<ul class="cj-list-rows {esc(ul_class)}">{"".join(row_html(r) for r in rows)}</ul>'
        if by == "none" or not label:
            parts.append(body)
            continue
        imgs = sum(int(size_of(r) or 0) for r in rows)
        parts.append(f'<details class="cj-list-group" open id="cj-list-g{n}"><summary>'
                     f'<span class="cj-list-gname cj-wrap">{esc(group_label(by, label))}</span>'
                     f'<span class="cj-list-gcount">· {plural(len(rows), "set")}</span>'
                     f'<span class="cj-list-gimg">{plural(imgs, "image")}</span></summary>{body}</details>')
    return count + f'<div class="cj-list">{HEAD}{"".join(parts)}</div>'


def choices(rows, *, key=lambda r: r["facets"]) -> tuple:
    """(people, models, n_agent) from the rows the viewer may see (never from anyone else's)."""
    people, models, n_agent = set(), set(), 0
    for r in rows:
        f = _f(r, key)
        p, m = person_of(r, f), model_of(r, f)
        if p:
            people.add(p)
        if m and m != HIDDEN_LABEL:
            models.add(m)
        n_agent += 1 if via_of(r, f) else 0
    return sorted(people), sorted(models), n_agent


STYLE = """<style>
/* portal/listing.py: the shared list (Results, Predict). cj.css tokens only; every class is cj-list-*. */
.cj-list-form{display:flex;flex-direction:column;gap:10px;margin:12px 0;padding:12px 14px;background:var(--card);
  border:1px solid var(--line);border-radius:var(--cj-r)}
.cj-list-r1,.cj-list-r2{display:flex;flex-wrap:wrap;gap:10px 14px;align-items:center}
.cj-list-r2{padding-top:10px;border-top:1px solid var(--line)}
.cj-list-f{display:flex;align-items:center;gap:6px;font-size:.85rem;color:var(--mut);min-width:0}
.cj-list-f select,.cj-list-f input[type=date],.cj-list-f input[type=search]{font:inherit;font-size:.85rem;
  color:var(--fg);background:var(--card);border:1px solid var(--line);border-radius:6px;padding:4px 6px;min-width:0;
  max-width:100%}
.cj-list-chk{color:var(--fg)}
.cj-list-search{flex:1 1 200px}
.cj-list-search input{flex:1 1 auto;width:100%}
.cj-list-seg{display:flex;align-items:center;gap:0;border:0;margin:0;padding:0;font-size:.85rem;color:var(--mut)}
.cj-list-seg legend{float:left;margin-right:6px;padding:0}
.cj-list-segb{position:relative;display:inline-flex}
.cj-list-segb input{position:absolute;opacity:0;width:1px;height:1px}
.cj-list-segb span{padding:4px 12px;border:1px solid var(--line);margin-left:-1px;background:var(--card);color:var(--fg);
  cursor:pointer}
.cj-list-segb:first-of-type span{border-radius:6px 0 0 6px;margin-left:0}
.cj-list-segb:last-of-type span{border-radius:0 6px 6px 0}
.cj-list-segb input:checked+span{background:var(--acc);border-color:var(--acc);color:var(--card);font-weight:600}
.cj-list-segb input:focus-visible+span{outline:2px solid var(--acc);outline-offset:2px}
.cj-list-btns{margin-left:auto;display:flex;gap:10px;align-items:center}
.cj-list-go{font:600 .85rem/1.2 var(--cj-font);padding:6px 14px;border-radius:6px;border:1px solid var(--acc);
  background:var(--acc);color:var(--card);cursor:pointer}
.cj-list-reset{font-size:.85rem}
.cj-list-count{margin:4px 0}
.cj-list{background:var(--card);border:1px solid var(--line);border-radius:var(--cj-r);overflow:hidden}
.cj-list-rows{list-style:none;margin:0;padding:0}
.cj-list-head,.cj-list-row{display:flex;flex-direction:row;flex-wrap:wrap;gap:6px 12px;align-items:center;padding:10px 14px}
.cj-list-head{background:var(--bg);border-bottom:1px solid var(--line);font-size:.72rem;font-weight:600;
  letter-spacing:.04em;text-transform:uppercase;color:var(--mut)}
.cj-list-row{border-bottom:1px solid var(--line)}
.cj-list-rows:last-child .cj-list-row:last-child{border-bottom:0}
.cj-list-name{flex:3 1 260px;min-width:0;display:flex;flex-direction:column;gap:2px}
.cj-list-t{text-decoration:none}
.cj-list-id{font-family:var(--cj-mono);font-size:.75rem}
.cj-list-owner{flex:1.2 1 140px;display:flex;flex-wrap:wrap;gap:4px 6px;align-items:center;font-size:.85rem}
.cj-list-via{padding:0 6px;border-radius:999px;font-size:.72rem;color:var(--prod);background:var(--prodbg);
  border:1px solid var(--line)}
.cj-list-model{flex:1 1 120px;font-family:var(--cj-mono);font-size:.78rem;min-width:0}
.cj-list-num{flex:0 0 70px;text-align:right;font-family:var(--cj-mono);font-size:.8rem}
.cj-list-num small{display:none}
.cj-list-when{flex:.8 1 110px;font-size:.8rem;color:var(--mut)}
.cj-list-status{flex:0 0 120px}
.cj-list-act{flex:0 0 auto;margin-left:auto;display:flex;flex-wrap:wrap;gap:6px 10px;align-items:center;
  justify-content:flex-end}
.cj-list-act form{margin:0}
.cj-list-open{display:inline-flex;padding:4px 10px;border-radius:6px;border:1px solid var(--acc);font-weight:600;
  font-size:.8rem;text-decoration:none}
.cj-list-group{border-bottom:1px solid var(--line)}
.cj-list-group:last-child{border-bottom:0}
.cj-list-group>summary{display:flex;flex-wrap:wrap;align-items:center;gap:8px;padding:9px 14px;background:var(--accbg);
  cursor:pointer;font-size:.88rem}
.cj-list-group[open]>summary{border-bottom:1px solid var(--line)}
.cj-list-gname{font-weight:600;min-width:0}
.cj-list-gcount,.cj-list-gimg{font-family:var(--cj-mono);font-size:.78rem;color:var(--mut)}
.cj-list-gimg{margin-left:auto}
.cj-list-empty{margin:0;padding:24px 14px;text-align:center;color:var(--mut)}
@media (max-width:760px){
  .cj-list-head{display:none}
  .cj-list-num,.cj-list-when,.cj-list-status,.cj-list-model{flex:0 1 auto;text-align:left}
  .cj-list-num small{display:inline}
  .cj-list-btns{margin-left:0}
}
</style>"""
