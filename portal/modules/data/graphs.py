"""
The Lab graph (History's lineage graph, filtered per viewer) and the Session / grid graph.

Part of the Data module (portal/modules/data/__init__.py has the rules); moved from data.py unchanged.
"""
from __future__ import annotations

import importlib.util
import os
import re
from pathlib import Path
from urllib.parse import quote

from portal import upload_sessions as US
from portal.modules.data.records import (
    HERE, PREDICTION_SET_ID, PREFIX, REPO, SCHEMA, SET_ENTITIES, Viewer, cached, _dict, _files_cache, find_set,
    grid_sort_key, lineage_href, _lock, _number, plural, predict_folder, read_small_json, set_is_visible,
    short_date, _text)
from portal.modules.data.lineage import KIND_TO_NODE, Lineage


# ---------------------------------------------------------------- Lab graph (History's lineage graph, filtered)
GRAPH_PAGE_KIND = {"roboflow_export": "images", "prediction_set": "images", "review_package": "images",
                   "label_version": "labels", "dataset": "dataset", "model": "model",
                   "session": "session"}                        # pulls and jobs: no page
GRAPH_SHORT = {"roboflow_export": "export", "review_package": "package", "label_version": "label version",
               "prediction_set": "predictions"}                 # the box's tag, as History's drawGraph SHORT
GRAPH_NODE_KEYS = ("title", "kind", "col", "hist", "part", "sub", "status", "url", "maker")   # all a node sends
GRAPH_UNKNOWN_SESSION = "session|?"     # History's "Session unknown" node (listed images with no session): no record
SCRIPT = HERE.parent / "static" / "data.js"                    # served as /data/data.js (app.py serves only /cj/*)
GRAPH_CACHE_N = 32                  # registry reads and built graphs kept (one graph per registry state per group)
GRAPH_SHORT_ID = 6                  # a hidden id shorter than this is matched as a whole word, never as a substring
_graph_cache: dict = {}             # key -> value, oldest first
_history_module_cache = None


def _history_module():
    """registry/history_page.py, loaded once by path (read only here: its reader and builders are reused)."""
    global _history_module_cache
    if _history_module_cache is None:
        spec = importlib.util.spec_from_file_location("cj_data_history_page", REPO / "registry" / "history_page.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _history_module_cache = mod
    return _history_module_cache


def _recall(key):
    """A kept value, made the newest on a hit (so a shared entry such as the registry read is evicted last)."""
    with _lock:
        hit = _graph_cache.pop(key, None)
        if hit is not None:
            _graph_cache[key] = hit
        return hit


def _remember(key, value):
    with _lock:
        _graph_cache.pop(key, None)
        _graph_cache[key] = value
        while len(_graph_cache) > GRAPH_CACHE_N:
            _graph_cache.pop(next(iter(_graph_cache)))


def files_signature(paths) -> tuple:
    """(path, mtime_ns, size) per file; a file that cannot be read counts as (path, None, None)."""
    out = []
    for p in paths:
        try:
            st = os.stat(p)
            out.append((str(p), st.st_mtime_ns, st.st_size))
        except OSError:
            out.append((str(p), None, None))
    return tuple(out)


def history_registry(root) -> tuple:
    """History's own read of <root>/*.jsonl (plus the label versions' owner files it reads) -> ((signature, current,
    history, stats) or None, error or None). Read again only when one of those files changes (mtime_ns, size); a
    failed read is not kept. Records are shared between requests and never changed."""
    if root is None:
        return None, "no registry folder configured"
    if not Path(root).is_dir():
        return None, "registry folder not found"
    hp = _history_module()
    try:
        files = hp.input_files([Path(root)])
    except OSError as exc:
        return None, f"registry not readable ({type(exc).__name__})"
    sig = files_signature(files)                       # taken BEFORE the read: a write during it shows next time
    hit = _recall(("read", str(root)))
    if hit is not None and hit[0] == sig and files_signature(hit[1]) == hit[2]:
        return hit[3], None
    try:
        current, history, stats = hp.read_registry(files)
    except Exception as exc:                           # an unreadable registry must not take the page down
        return None, f"registry not readable ({type(exc).__name__})"
    owners = sorted({str(r["owner"]).partition("#")[0] for k, r in current.items()
                     if k.split("|", 1)[0] in ("label_version", "inference_check") and _text(r.get("owner"))}
                    | export_list_files(current, root))
    owner_sig = files_signature(owners)
    base = (sig + owner_sig, current, history, stats)
    _remember(("read", str(root)), (sig, owners, owner_sig, base))
    return base, None


def export_list_files(current, root) -> set:
    """The export image lists the lab graph reads (#15: session nodes and counts), as History resolves them (a
    relative file under the registry folder, or an absolute one): part of the cached graph's signature, so editing a
    list builds the graph again."""
    out = set()
    for k, r in current.items():
        f = _text(_dict(r.get("image_list")).get("file")) if k.startswith("roboflow_export|") else None
        if f:
            out.add(str(Path(f) if Path(f).is_absolute() else Path(root) / f))
    return out


def _strings_in(value):
    """Every string inside a record value (nested dicts and lists included, dict keys too)."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for k, v in value.items():
            yield from _strings_in(k)
            yield from _strings_in(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _strings_in(v)


def hidden_matcher(hidden_ids):
    """-> text -> True when the text quotes a hidden id: a substring test for ids of GRAPH_SHORT_ID characters or
    more, a whole-word test for shorter ones (so a hidden 'rf_x' does not catch 'rf_x_v2'); empty ids never match."""
    long_ids = [h for h in hidden_ids if len(h) >= GRAPH_SHORT_ID]
    short = [h for h in hidden_ids if 0 < len(h) < GRAPH_SHORT_ID]
    word = re.compile("|".join(rf"(?<![A-Za-z0-9_]){re.escape(h)}(?![A-Za-z0-9_])" for h in short)) if short else None

    def quotes(text) -> bool:
        text = str(text)
        return any(h in text for h in long_ids) or bool(word and word.search(text))
    return quotes


def derived_ids(hidden_ids):
    """Matches a record id that IS a hidden id or a known derived form: pred_<id>, <id>_part<N>, pred_<id>_part<N>."""
    alts = "|".join(re.escape(h) for h in sorted(hidden_ids) if h)
    return re.compile(rf"(?:pred_)?(?:{alts})(?:_part\d+)?") if alts else None


def graph_hidden(records, viewer: Viewer, current: dict) -> tuple:
    """-> (History keys to leave out of the graph's INPUT, hidden set ids). Visibility is decided on Data's own record
    (records[entity][id], the last line, as the Lineage tree reads it): a prediction set, review package or Roboflow
    export the viewer may not see, or one Data does not have (unknown owner: admins only). Then every job that quotes
    a hidden id anywhere (any string or key, its own id too: hidden_matcher), and any other record whose id is a
    hidden id or a derived form of one (derived_ids). A dataset or label version built from a hidden set stays."""
    drop, hidden_ids = set(), set()
    for key in current:
        entity, rid = key.split("|", 1)
        if entity not in SET_ENTITIES:
            continue
        mine = records.get(entity, {}).get(rid)
        if not (set_is_visible(viewer, entity, mine) if mine else viewer.may_see(None)):
            drop.add(key)
            hidden_ids.add(rid)
    quotes, derived = hidden_matcher(hidden_ids), derived_ids(hidden_ids)
    if derived is None:
        return drop, hidden_ids
    for key, rec in current.items():
        if key in drop:
            continue
        entity, rid = key.split("|", 1)
        if entity == "job":
            if quotes(rid) or any(quotes(s) for s in _strings_in(rec)):
                drop.add(key)
        elif derived.fullmatch(rid):
            drop.add(key)
    return drop, hidden_ids


def graph_url(walker: "Lineage", entity: str, rid: str):
    """The item's Data Lineage page when it has one and this viewer may open it, else None."""
    kind = GRAPH_PAGE_KIND.get(entity)
    if kind is None:
        return None
    found = walker.lookup(KIND_TO_NODE[kind], rid)
    return lineage_href(kind, rid) if found and found[0] != "hidden" else None


def lab_graph(records, viewer: Viewer, root, focus=None, extra=None) -> dict:
    """History's lineage graph for this viewer: {schema, cols, nodes: [{GRAPH_NODE_KEYS}], edges: [[from, to, role]],
    focus: node index or None, error}. The registry is filtered BEFORE graph_model runs (graph_hidden), with History's
    own read_registry / build_model, so an admin gets exactly History's graph.
    Cost: build_model also builds History's image lookup and model pages (about 0.4 s on the lab registry, 11-2026);
    the graph is kept per registry state (the files' mtime_ns + size) and per set of hidden records (a visibility
    group), so it is built again only when the registry changes or a new group asks. A subtitle, status or edge role
    that quotes a hidden id is blanked.
    extra: a dict that, when given, also receives {keys: History key per node, base, drop, hidden_ids} for
    source_graph (never sent to a client)."""
    base, error = history_registry(root)
    if base is None:
        return {"schema": SCHEMA, "cols": [], "nodes": [], "edges": [], "focus": None, "error": error}
    sig, current, history, stats = base
    drop, hidden_ids = graph_hidden(records, viewer, current)
    key = ("graph", str(root), sig, frozenset(drop))
    graph = _recall(key)
    if graph is None:
        kept = {k: r for k, r in current.items() if k not in drop}
        # sources and changes only feed History's image lookup and model pages, never the graph
        graph = _history_module().build_model(kept, {k: history[k] for k in kept}, stats, [], [],
                                              graph_sources=[Path(root)])["graph"]
        _remember(key, graph)
    quotes = hidden_matcher(hidden_ids)
    walker = Lineage(records, viewer, root)
    nodes, focus_index = [], None
    for i, (k, col, hist, part, sub, maker) in enumerate(graph["nodes"]):
        entity, rid = k.split("|", 1)
        status = current.get(k, {}).get("status")
        status = None if status is None else str(status)
        unknown = k == GRAPH_UNKNOWN_SESSION
        nodes.append({"title": "Session unknown" if unknown else rid, "kind": GRAPH_SHORT.get(entity, entity),
                      "col": col, "hist": hist, "part": part,
                      "sub": "" if quotes(sub) else sub,        # an export's note may quote a hidden id
                      "status": "" if status is not None and quotes(status) else status,
                      "url": None if unknown else graph_url(walker, entity, rid), "maker": maker})
        if k == focus:
            focus_index = i
    edges = [[a, b, "input" if quotes(role) else role] for a, b, role in graph["edges"]]
    if extra is not None:
        extra.update(keys=[n[0] for n in graph["nodes"]], base=base, drop=drop, hidden_ids=hidden_ids)
    return {"schema": SCHEMA, "cols": list(graph["cols"]), "nodes": nodes, "edges": edges, "focus": focus_index,
            "error": error}


ITEM_DOWNSTREAM_MAX = 150           # descendants in one item graph; past it a link to the lab graph (no count)


def item_graph(g: dict, skip=frozenset()) -> dict:
    """The slice of a lab_graph() result (already visibility-filtered and whitelisted) that one item needs: its
    ancestors over REVERSE edges, its descendants over FORWARD edges (never an undirected walk: that would reach a
    sibling dataset through a shared label set), following `part` links too (a part's parent package is an ancestor,
    a package's parts are descendants). Descendants stop at ITEM_DOWNSTREAM_MAX, then "more" is true (no count).
    Node payloads are copied as they are; edges, `part` and `focus` are remapped to the slice's indices (a part whose
    parent is outside the slice -> -1). An unknown or hidden focus (focus None) gives an empty graph.
    skip: (from, to) pairs the walk does not follow (source_graph: another session's analyses); such an edge is still
    drawn when both its ends are in the slice."""
    nodes, edges, f = g["nodes"], g["edges"], g.get("focus")
    out = {"schema": g["schema"], "scope": "item", "cols": [], "nodes": [], "edges": [], "focus": None,
           "more": False, "error": g.get("error")}
    if f is None:
        return out
    up, down, kids = {}, {}, {}
    for a, b, _ in edges:
        if (a, b) in skip:
            continue
        up.setdefault(b, []).append(a)
        down.setdefault(a, []).append(b)
    def pof(i):                                             # the package a part belongs to, else -1 (never itself)
        p = nodes[i]["part"]
        return p if p is not None and p >= 0 and p != i else -1

    for i in range(len(nodes)):
        if pof(i) >= 0:
            kids.setdefault(pof(i), []).append(i)

    def parents(i):
        return up.get(i, []) + ([pof(i)] if pof(i) >= 0 else [])

    def children(i):
        return down.get(i, []) + kids.get(i, [])

    keep, queue = {f}, [f]
    for i in queue:                                         # ancestors: uncapped
        for j in parents(i):
            if j not in keep:
                keep.add(j)
                queue.append(j)
    seen, queue, n_down = {f}, [f], 0
    for i in queue:                                         # descendants: breadth first, capped; an ancestor that
        for j in children(i):                               # is also downstream (a cycle) is still expanded
            if j in seen:
                continue
            if j not in keep:
                if n_down >= ITEM_DOWNSTREAM_MAX:
                    out["more"] = True
                    continue
                n_down += 1
                keep.add(j)
            seen.add(j)
            queue.append(j)
    order = sorted(keep)
    new = {old: k for k, old in enumerate(order)}
    out["cols"] = list(g["cols"])
    out["nodes"] = [dict(nodes[i], part=new.get(pof(i), -1)) for i in order]
    out["edges"] = [[new[a], new[b], role] for a, b, role in edges if a in new and b in new]
    out["focus"] = new[f]
    return out


def graph_focus_key(records, kind, entity_id):
    """The graph key (History's '<entity>|<id>') of a catalog item, for its Lineage page's Graph and "Lab graph" link.
    A session or grid gets 'session|<id>' / 'grid|<id>': not a lab graph node, its Graph is source_graph()."""
    if kind == "images":
        entity, _ = find_set(records, entity_id)
    else:
        entity = {"labels": "label_version", "dataset": "dataset", "model": "model", "session": "session",
                  "grid": "grid"}.get(kind)
    return f"{entity}|{entity_id}" if entity else None


def lab_graph_href(focus=None) -> str:
    focus = None if source_focus(focus) else focus          # a session / grid is not in the lab graph
    return f"{PREFIX}/lineage/graph/" + (f"?focus={quote(focus, safe='')}" if focus else "")


# ---------------------------------------------------------------- Session / grid graph (lab graph + its sources)
SOURCE_COLS = ("Grids and uploads",)       # inserted after the lab graph's own first lane, "Imaging sessions"
SOURCE_AT = 1                               # the lane index the source columns are inserted at
SOURCE_ENTITIES = ("session", "grid")
SOURCE_ROLES = {"grid": "grid", "images": "used its images", "upload": "upload", "analysed": "analysed"}


def source_focus(focus):
    """('session' | 'grid', id) for a session / grid graph key, else None."""
    if not isinstance(focus, str):
        return None
    entity, sep, rid = focus.partition("|")
    return (entity, rid) if sep and entity in SOURCE_ENTITIES and rid else None


def analysis_set_id(key, rec):
    """The prediction set an analysis belongs to: a prediction set's own id, a prediction package's set (its
    prediction_set, a base of kind prediction_set, or the id after pred_), else None (not an analysis)."""
    entity, rid = key.split("|", 1)
    if entity == "prediction_set":
        return rid
    if entity != "review_package":
        return None
    return (_text(rec.get("prediction_set"))
            or (_text(rec.get("base")) if rec.get("base_kind") == "prediction_set" else None)
            or (rid[len("pred_"):] if rid.startswith("pred_") else None))


def build_source_links(current, root) -> dict:
    """What used the images of each session and grid, over the WHOLE registry (no visibility: source_links takes a
    group's hidden keys out of this):
      links[(session, grid or None)] = {History key}: an image record names the session (its grid: the image's own,
        else the session's session_grids slot; a grid the registry does not have counts as no grid), and an item
        lists that image -- History's own image lookup (registry/history_page.py lookup_index: a dataset's image
        list, a label version's change log and reviewed images, a review package's data.js, a revise / annotate
        job's list, a Roboflow image -> the pull it was first seen in), plus a Roboflow export's image list;
      grids[session] = its grids; sessions[grid] = its sessions; ps[session] = prediction sets whose source is that
      session; pkgs[prediction set id] = its prediction packages; analysis[key] = the prediction set of each
      prediction set / prediction package key (analysis_set_id)."""
    hp = _history_module()
    where = {}                                              # image id -> (session, grid or None)
    for k, r in current.items():
        if not k.startswith("image|") or not _text(r.get("session")):
            continue
        sid, slot = r["session"], r.get("session_grid")
        gid = _text(r.get("grid")) or _dict(_dict(current.get("session|" + sid)).get("session_grids")).get(
            str(slot) if slot is not None else "")
        where[r["id"]] = (sid, str(gid) if gid and f"grid|{gid}" in current else None)
    links: dict = {}

    def add(image, k):
        w = where.get(image)
        if w and k:
            links.setdefault(w, set()).add(k)
    lk = hp.lookup_index(current, [Path(root)], [])
    for name, refs in lk["i"]:
        if name not in where:
            continue
        for ref in refs:
            k = lk["e"][ref if isinstance(ref, int) else ref[0]]
            if k.startswith("roboflow_image|"):
                pull = _text(current[k].get("first_seen_pull"))
                k = f"pull|{pull}" if pull else None
            add(name, k)
    for k, r in current.items():                            # History's lookup reads datasets' lists, not exports'
        if k.startswith("roboflow_export|") and isinstance(r.get("image_list"), dict):
            for stem, _ in hp.read_list_file(r["image_list"], [Path(root)])["rows"] or []:
                add(stem, k)
    grids, sessions = {}, {}
    for k, r in current.items():
        if k.startswith("session|"):
            for gid in _dict(r.get("session_grids")).values():
                if f"grid|{gid}" in current:
                    grids.setdefault(r["id"], set()).add(str(gid))
    for sid, gid in where.values():
        if gid:
            grids.setdefault(sid, set()).add(gid)
    for sid, gs in grids.items():
        for gid in gs:
            sessions.setdefault(gid, set()).add(sid)
    ps, pkgs, analysis = {}, {}, {}
    for k, r in current.items():
        set_id = analysis_set_id(k, r)
        if set_id is None:
            continue
        analysis[k] = set_id
        if k.startswith("prediction_set|"):
            if _text(_dict(r.get("source")).get("session")):
                ps.setdefault(r["source"]["session"], []).append(k)
        elif _text(r.get("prediction_set")) or r.get("base_kind") == "prediction_set":
            pkgs.setdefault(set_id, []).append(k)
    return {"links": {w: frozenset(ks) for w, ks in links.items()},
            "grids": {s: sorted(g, key=grid_sort_key) for s, g in grids.items()},
            "sessions": {g: sorted(s) for g, s in sessions.items()},
            "ps": {s: sorted(ks) for s, ks in ps.items()}, "pkgs": {p: sorted(ks) for p, ks in pkgs.items()},
            "analysis": analysis}


def source_links(root, base, drop) -> dict:
    """build_source_links for one visibility group: built ONCE per registry state on the whole registry, then the
    group's hidden keys (graph_hidden's drop) are taken out of every value. Exact, because every value is a History
    key and a job that names a hidden set is itself in drop (test_data_session_graph checks this against a build on
    the filtered registry for each viewer). Both are kept like the lab graph."""
    sig, current, _, _ = base
    full = _recall(("sources", str(root), sig))
    if full is None:
        full = build_source_links(current, root)
        _remember(("sources", str(root), sig), full)
    if not drop:
        return full
    key = ("sources", str(root), sig, frozenset(drop))
    hit = _recall(key)
    if hit is not None:
        return hit
    links = {w: ks - drop for w, ks in full["links"].items()}
    ps = {s: [k for k in ks if k not in drop] for s, ks in full["ps"].items()}
    pkgs = {p: [k for k in ks if k not in drop] for p, ks in full["pkgs"].items()}
    out = {"links": {w: ks for w, ks in links.items() if ks}, "grids": full["grids"], "sessions": full["sessions"],
           "ps": {s: ks for s, ks in ps.items() if ks}, "pkgs": {p: ks for p, ks in pkgs.items() if ks},
           "analysis": {k: v for k, v in full["analysis"].items() if k not in drop}}
    _remember(key, out)
    return out


def source_node(entity, rid, col, **kw) -> dict:
    """A session / grid / upload box: exactly GRAPH_NODE_KEYS, never a path or a person."""
    node = {"title": rid, "kind": entity, "col": 0 if col == "Imaging sessions" else SOURCE_AT + SOURCE_COLS.index(col),
            "hist": 0, "part": -1, "sub": "", "status": None, "url": None, "maker": entity}
    node.update(kw)
    return node


def session_box(records, sid, quotes) -> dict:
    """quotes: hidden_matcher of the viewer's hidden ids; a subtitle or status quoting one is blanked, as on lab nodes."""
    rec = records["session"].get(sid)
    if not rec:                                             # named by images only: no page to open
        return source_node("session", sid, "Imaging sessions")
    sub = short_date(rec.get("date"))
    status = str(rec["status"]) if rec.get("status") is not None else None
    return source_node("session", sid, "Imaging sessions", sub="" if quotes(sub) else sub,
                       status="" if status is not None and quotes(status) else status,
                       url=lineage_href("session", sid))


def grid_box(records, gid, quotes) -> dict:
    rec = records["grid"].get(gid)
    sub = _text(_dict(rec).get("sample_label")) or ""
    return source_node("grid", gid, "Grids and uploads", sub="" if quotes(sub) else sub,
                       url=lineage_href("grid", gid) if rec else None)


def analysis_conversion(ps_id):
    """The conversion a prediction set was run on (its request.json source folder is <raw_tiles>/<cid>/tiles), or
    None. Only the folder NAME is read, compared with the session's conversions; the path is never sent."""
    def compute():
        root = predict_folder()
        if root is None or not PREDICTION_SET_ID.match(str(ps_id)):
            return None
        folder = _text(_dict(read_small_json(root / ps_id / "request.json").get("source")).get("dir"))
        if not folder:
            return None
        p = Path(folder)
        return p.parent.name if p.name == "tiles" else None
    return cached(_files_cache, ("ps-conversion", str(ps_id)), compute)


def source_graph(records, viewer: Viewer, root, focus) -> dict:
    """The Graph of a session's or grid's Lineage page, in item_graph's shape (scope item, cap and `more` included):
    the session -> its grids -> what used their images (source_links), its upload conversions (only those the viewer
    may open: the owner and the admin) -> the analyses run on them, and analyses of the session; then everything
    downstream of those over the lab graph's own edges (item_graph's directed walk, never a second walker). A grid's
    graph has its sessions as ancestors and only what used the grid's own images; a session with only this grid also
    gives the grid its analyses.
    Another session's analyses stay out: a model's edge to a prediction set / prediction package is not followed
    unless that analysis is this session's own (its source session, or run on one of its conversions; for a grid,
    the analyses of its one-grid sessions). They (and their parts and what follows) appear only when reached
    another way, e.g. a package listing this session's images.
    Built from the viewer's FILTERED lab graph: an item the viewer may not see is not a node there, so it is never
    linked, counted or reached. Grid / upload boxes go in a lane after the lab's "Imaging sessions" (SOURCE_COLS); a
    session the lab graph already has (an export lists its images) keeps its lab box, else one is added. Their
    subtitles and statuses go through the same hidden-id blanking as lab nodes.
    An unknown session / grid (or any other key) gives the empty graph with focus None and no echo."""
    extra: dict = {}
    g = lab_graph(records, viewer, root, None, extra)
    want = source_focus(focus)
    if want is None or "keys" not in extra or want[1] not in records[want[0]]:
        return item_graph(g)                                # focus None: the empty graph
    entity, rid = want
    sl = source_links(root, extra["base"], extra["drop"])
    keys, quotes = extra["keys"], hidden_matcher(extra["hidden_ids"])
    idx = {k: i for i, k in enumerate(keys)}
    nodes = g["nodes"]                                      # fresh dicts per call: shifting the columns is safe
    for n in nodes:
        if n["col"] >= SOURCE_AT:
            n["col"] += len(SOURCE_COLS)
    added: dict = {}
    lab_edges = {(a, b) for a, b, _ in g["edges"]}

    def box(node) -> int:
        nodes.append(node)
        return len(nodes) - 1

    def session_at(sid) -> int:
        """The lab graph's own box for a session an export lists, else a new one in the sessions lane."""
        i = idx.get("session|" + sid)
        if i is None:
            return box(session_box(records, sid, quotes))
        nodes[i]["hist"] = 0                    # a source of this graph is never faded, whatever the lab says
        return i

    def link(a, b, role):
        if a is not None and b is not None and a != b and (a, b) not in lab_edges:
            added.setdefault((a, b), SOURCE_ROLES[role])

    def analysed(src, set_ids):
        for set_id in sorted(set_ids):
            link(src, idx.get("prediction_set|" + set_id), "analysed")
            for pk in sl["pkgs"].get(set_id, []):
                link(src, idx.get(pk), "analysed")

    if entity == "session":
        focus_i = session_at(rid)
        own = {k.split("|", 1)[1] for k in sl["ps"].get(rid, [])}
        conv_ids = {c["id"] for c in US.conversions_of(records["session"][rid])}
        for gid in sl["grids"].get(rid, []):
            gi = box(grid_box(records, gid, quotes))
            link(focus_i, gi, "grid")
            for k in sorted(sl["links"].get((rid, gid), ())):
                link(gi, idx.get(k), "images")
        for k in sorted(sl["links"].get((rid, None), ())):
            link(focus_i, idx.get(k), "images")
        convs = {}
        if viewer.cfg is not None and conv_ids:
            for c in US.visible_conversions(viewer.cfg, records["session"][rid], viewer.user or {}, viewer.can_see):
                n_tiles = _number(c.get("n_tiles"))
                convs[c["id"]] = box(source_node("upload", c["id"], "Grids and uploads",
                                                 sub=plural(int(n_tiles), "tile") if n_tiles is not None else "",
                                                 url=c["href"]))
                link(focus_i, convs[c["id"]], "upload")

        def is_own(set_id):                                 # run on one of this session's conversions counts too
            if set_id not in own and conv_ids and analysis_conversion(set_id) in conv_ids:
                own.add(set_id)
            return set_id in own
    else:
        focus_i = box(grid_box(records, rid, quotes))
        own = set()
        for sid in sl["sessions"].get(rid, []):
            link(session_at(sid), focus_i, "grid")
            if sl["grids"].get(sid) == [rid]:               # the session imaged only this grid: its analyses too
                own |= {k.split("|", 1)[1] for k in sl["ps"].get(sid, [])}
        for (sid, gid), ks in sl["links"].items():
            if gid == rid:
                for k in sorted(ks):
                    link(focus_i, idx.get(k), "images")

        def is_own(set_id):
            return set_id in own
    skip = {(a, b) for a, b, _ in g["edges"]
            if keys[a].startswith("model|") and keys[b] in sl["analysis"] and not is_own(sl["analysis"][keys[b]])}
    if entity == "session":                                 # from the conversion it ran on when the viewer sees it
        for set_id in sorted(own):
            analysed(convs.get(analysis_conversion(set_id), focus_i), [set_id])
    else:
        analysed(focus_i, own)
    full = {"schema": g["schema"], "cols": g["cols"][:SOURCE_AT] + list(SOURCE_COLS) + g["cols"][SOURCE_AT:],
            "nodes": nodes,
            "edges": g["edges"] + [[a, b, role] for (a, b), role in added.items()], "focus": focus_i,
            "error": g["error"]}
    return item_graph(full, frozenset(skip))
