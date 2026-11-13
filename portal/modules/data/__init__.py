"""
Data module (CryoEV portal, read only): a plain catalog of WHAT DATA EXISTS, from the registry (sessions, grids, image
sets, label versions, datasets, models) and the processed images on disk.

    GET /data/                          overview: counts and short lists, search box
    GET /data/search?q=                 search by id or sample label across everything
    GET /data/list/<kind>               the full list (sessions, grids, images, labels, datasets, models)
    GET /data/session/<id>              acquisition summary, its grids, a strip of up to 12 images;
                                        ?page=N shows all its images, 24 per page
    GET /data/grid/<id>                 one grid: sample, type, sessions, thumbnails (paged, 24 per page)
    GET /data/model/<id>                status (production badge), dataset, scores from the registry
    GET /data/dataset/<id>              splits, classes, source, models trained on it, thumbnails
    GET /data/labels/<id>               a label version: what it changed, how much was verified, thumbnails
    GET /data/images/<id>               an image set (review package, prediction set, Roboflow export), thumbnails
    GET /data/thumb/<kind>/<id>/<n>.jpg?v=<version>
                                        one thumbnail (about 240 px), disk-cached under <state dir>/data_thumbs
    GET /data/lineage/<kind>/<id>/      the chain back to the raw imaging sessions (model, training run, dataset,
                                        label version / Roboflow export, image sets, grids, session), with each
                                        step's key facts and a few thumbnails; <kind> is one of the detail kinds
    GET /data/api/overview | search?q= | list/<kind> | <kind>/<id>[?page=N]    the same information as JSON
                                        (schema cj_data/1; other modules reuse it)
    GET /data/api/lineage/<kind>/<id>   the lineage as a JSON tree (schema cj_data/1)
    GET /data/lineage/graph/[?focus=<entity>|<id>]
                                        the Lab graph: History's lineage graph (imaging sessions -> label sets
                                        by maker, reviews between them -> datasets -> models) for the whole lab,
                                        drawn by /data/data.js
    GET /data/api/lineage/graph[?focus=<entity>|<id>]   its data as JSON (schema cj_data/1)
    GET /data/data.js                   the Lab graph's script (portal/static/data.js; no inline script)
    POST /data/session/<id>/details     the Session details form of an uploaded session (its uploader or the admin;
                                        portal/upload_sessions.save_details, only when CJ_UPLOAD_SESSIONS=on)

Rules:
  - Nothing here writes to a data folder. The only registry write is the Session details form above (a merge-append
    through portal/upload_sessions.py); the other write is the thumbnail cache in the portal's own state folder.
  - Upload sessions (registry `conversions`, round 7): the list shows an "Uploaded" chip; the session page shows the
    acquisition summary (the record's own, else the newest upload's), sample per grid (display names from the
    details first), grid ids and prep notes (every admin / advanced user), and links to the conversion page and its
    Gallery (/gallery/<cid>/) ONLY for viewers who may see that conversion (ctx.can_see on its convert.json owner),
    plus the analyses (prediction sets whose source session is this one) the viewer may see. Without lab images the
    thumbnails come from those visible conversions' montages / tiles, through the same cache and containment checks
    (the conversion folders' roots are added to the known roots for those files only).
  - A thumbnail file is resolved on the server from registry / package data only (the kind + id + index in the URL
    pick it); the client never sends a path. Each kind reads only its own folder: a session's
    `<processing>/<id>/3. Processed images`, `<laptop>/review` for packages, `<laptop>/predict` for prediction sets.
    A path is checked as TEXT against that folder first (so nothing outside it is ever touched, which matters for
    network shares), then by real path with links followed.
  - Privacy fails closed. A prediction set, and a review package that is a prediction package (kind "prediction",
    or has a prediction_set, or an id starting pred_), is listed only for its owner and admins (ctx.can_see). The
    owner is read from the registry record, else the package's review_package.json, else the prediction set's
    request.json; when none is known only admins see it. Prediction packages that are on disk but not in the
    registry are found by their folder name and follow the same rule. A label version or dataset shows (and loads
    images from) its package only to people who may see that package.
  - Lineage: every object in the chain goes through the same visibility test as its own page (ctx.can_see for
    prediction sets and prediction packages). A hidden object is shown as "not visible to you" with no id, and
    nothing below it is followed. A parent the registry does not record is "unknown", never guessed. The walk is
    capped (depth and number of steps) and stops at a step it is already inside, so bad registry data cannot loop.
    Folder paths are shown to admins only. A Roboflow export with an image list (registry `lists/`, written by the
    importer) leads to its images' imaging sessions, each with its image count and its grids where known, plus one
    "Session unknown" step for listed images no image record places in a session (#15); an export with no list says
    "Image list not recorded". A re-split shows its sessions under the export it was made from only when the viewer
    may see that one and its list gives exactly the same sessions, counts, grids and unknowns. A dataset that took its
    images from such an export shows its sessions there and compares the two lists in its note (counts only). The
    summary counts listed images (datasets and exports shown) that trace to no session, as a number only.
  - Lanes (#16, the graphs): Imaging sessions | Labels: automatic | Labels: human (Roboflow exports and pulls "via
    Roboflow") | Labels: revised (with the review packages and jobs, dashed, between lanes) | Datasets | Models.
  - Attribution: a "by" an agent wrote reads "claude for pujan" (registry/actor.py label; who_label here).
  - Lab graph: History's own reader and builders (registry/history_page.py read_registry, build_model ->
    graph_model, imported, never changed), run on a FILTERED registry: every prediction set, review package and
    Roboflow export the viewer may not see (set_is_visible on Data's own record, the rule above; one Data does not
    have is for admins only) is removed BEFORE the graph is built, and so is every job that quotes one (any value or
    key, or its own id; ids under 6 characters as whole words only) and any record whose id is a hidden id or its
    pred_ / _part<N> form. A part whose parent is a prediction package is private like its parent. A subtitle,
    status or edge role quoting a hidden id is blanked. So the fade-out and the node order never reflect a hidden record; a label version or
    dataset built from a hidden set stays and only that edge goes. Each node sends only its id (History's box
    text), kind, column, faded flag, part-of, subtitle, status and its Data Lineage address; edges are
    [from, to, role]. ?focus names a node to highlight; a key the viewer may not see highlights nothing.
  - Session / grid Graph (?focus=session|<id> or grid|<id> with scope=item; source_graph): the session, its grids,
    its upload conversions (only those the viewer may open), and what used their images, read from the same filtered
    registry (image records' session / grid; History's image lookup for datasets, label versions, review packages,
    jobs and pulls; exports' image lists; prediction sets' source session, their packages, and the conversion a set
    was run on), then the lab graph's own edges downstream through item_graph. A hidden item is not in the filtered
    graph, so it is never linked or counted. A model's edge to another session's analysis is not followed.
  - Plain words; every value is escaped; no script on the pages except the Lab graph's file /data/data.js.

Package layout (round 11: data.py split, the code moved unchanged; the pages are byte-identical, checked with
portal/tests/data_md5.py):
    records.py      constants and small helpers, Viewer, caches, read_registry, data folders and path checks,
                    images on disk, image-set visibility, list rows, search
    details.py      detail_* (the detail dicts behind each page and its JSON), thumbnails
    lineage.py      the Lineage walker and lineage_tree
    graphs.py       the Lab graph and the Session / grid graph
    render.py       the HTML
    routes.py       routes(ctx)
This file keeps PREFIX, ROLES and routes for the module loader (portal/modules/__init__.py) and re-exports the names
other modules and the tests use (`from portal.modules import data as D; D.read_registry(...)`).

To CHANGE or PATCH a name (a test's `D.X = ...`, `mock.patch.object(D, "X")`), do it on the part that defines it:
the re-export here is a separate binding, so setting it reaches only callers that read `D.X` on the package, never
the parts (portal/tests/test_data_patch_guard.py checks every test). `portal.modules.data.routes` here is the
routes FUNCTION; the routes module is importlib.import_module("portal.modules.data.routes").
"""
from __future__ import annotations

from portal.modules.data.records import (  # noqa: F401
    DETAIL_LIST, ENTITIES, LIST_KINDS, OVERVIEW_N, PREFIX, PROCESSED, STRIP_N, THUMB_MAX_PIXELS, THUMB_PX, Viewer,
    class_mapping, dataset_rows, _decoders, esc, _files_cache, image_set_rows, inside, is_contained, known_roots,
    label_rows, metric_name, _owner_cache, _package_cache, package_meta, package_visible, plural, pred_display_name,
    read_registry, _registry_cache, scan_images, search, session_images, status_word, who_label, with_disk_packages)
from portal.modules.data.details import (  # noqa: F401
    detail, make_thumb, replaced_by, session_analyses, session_grid_cards, to_8bit)
from portal.modules.data.lineage import (  # noqa: F401
    LINEAGE_MAX_DEPTH, LINEAGE_MAX_NODES, LINEAGE_THUMBS, lineage_tree, locate)
from portal.modules.data.graphs import (  # noqa: F401
    GRAPH_CACHE_N, GRAPH_NODE_KEYS, GRAPH_SHORT, GRAPH_UNKNOWN_SESSION, ITEM_DOWNSTREAM_MAX, SOURCE_AT, SOURCE_COLS,
    build_source_links, _graph_cache, graph_focus_key, hidden_matcher, _history_module, item_graph, lab_graph,
    lab_graph_href, _recall, _remember, source_focus, source_graph, source_links)
from portal.modules.data.render import (  # noqa: F401
    render_class_map, render_detail, render_lineage, render_rows, render_session_extras, render_used_by)
from portal.modules.data.routes import routes  # noqa: F401

ROLES = ("admin", "advanced")
