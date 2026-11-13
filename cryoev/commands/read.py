"""find / show / lineage: read the registry (registry/registry.py current / history / main) and the Data page's
lineage walk (portal/modules/data.py lineage_tree, imported read-only). Nothing is written."""
from __future__ import annotations

import json

from cryoev import common as CM

# the Data page's node types -> its catalog kinds (portal/modules/data.py KIND_TO_NODE, inverted; an export is an
# image set there)
NODE_TO_KIND = {"session": "session", "grid": "grid", "images": "images", "export": "images", "label": "labels",
                "dataset": "dataset", "model": "model"}


def add(sub):
    p = sub.add_parser("find", help="registry ids containing TEXT (read)")
    p.add_argument("text")
    p.add_argument("--entity", help="only this entity (model, dataset, label_version, prediction_set, job, ...)")
    p.add_argument("--status")
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=find)
    p = sub.add_parser("show", help="one registry record, as registry/registry.py show (read)")
    p.add_argument("id")
    p.add_argument("--json", action="store_true", help="the current record only, as JSON")
    p.set_defaults(func=show)
    p = sub.add_parser("lineage", help="the chain back to the imaging sessions (read)")
    p.add_argument("id")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=lineage)


def find(a) -> int:
    root = CM.registry_root()
    text = a.text.lower()
    rows = []
    for i, r in sorted(CM.registry().current(a.entity, root).items(), key=lambda kv: (kv[1]["schema"], kv[0])):
        if text in i.lower() and (not a.status or r.get("status") == a.status):
            rows.append({"id": i, "schema": r["schema"], "status": r.get("status")})
    more = len(rows) > a.limit
    rows = rows[:a.limit]
    if a.json:
        CM.emit_json({"matches": rows, "truncated": more})
    else:
        for r in rows:
            print(f"{r['schema']:20} {r['status'] or '-':12} {r['id']}")
        print(f"({len(rows)} match(es){'; more: raise --limit' if more else ''})")
    return 0 if rows else 1


def show(a) -> int:
    root = CM.registry_root()
    if not a.json:                                     # the registry's own command, unchanged
        rc, _ = CM.call_main(CM.registry().main, ["--root", str(root), "show", a.id])
        return rc
    h = CM.registry().history(a.id, root)
    if not h:
        print(f"unknown id {a.id}")
        return 1
    print(json.dumps(h[-1], indent=1, ensure_ascii=False, default=str))   # = registry.py show's record
    return 0


def _strip(node):
    """The tree without thumbnail and page addresses (portal-only)."""
    out = {k: v for k, v in node.items() if k not in ("thumbs", "url", "lineage")}
    out["parents"] = [_strip(p) for p in node.get("parents") or []]
    return out


def _print(node, depth=0):
    chips = ", ".join(str(c[0]) for c in node.get("chips") or [] if isinstance(c, (list, tuple)) and c)
    ident = node.get("id") or node.get("recorded") or ""
    print(f"{'  ' * depth}{'<- ' if depth else ''}{node.get('step')}: {node.get('title')}"
          + (f" [{ident}]" if ident and str(ident) not in str(node.get("title")) else "")
          + (f" ({chips})" if chips else "") + ("" if node.get("status") == "ok" else f" <{node.get('status')}>"))
    for p in node.get("parents") or []:
        _print(p, depth + 1)


def lineage(a) -> int:
    from portal.modules import data as D             # read-only: the Data page's own walk
    root = CM.registry_root()
    records, error = D.read_registry(root)
    if error:
        raise CM.CliError(f"registry: {error}")
    records = D.with_disk_packages(records)
    ntype = D.locate(records, a.id)
    if ntype is None:
        print(f"unknown id {a.id}")
        return 1
    kind = NODE_TO_KIND.get(ntype)
    if kind is None:
        raise CM.CliError(f"{a.id} is a {ntype}: lineage starts from a session, grid, image set, label version, "
                          f"dataset or model (try the model or dataset it belongs to)")
    viewer = D.Viewer(lambda user, owner: True, {"id": "cryoev-cli", "role": "admin"})   # the laptop sees all
    tree = D.lineage_tree(records, viewer, kind, a.id, root)
    if tree is None:
        print(f"no lineage for {a.id}")
        return 1
    tree = {**tree, "root": _strip(tree["root"])}
    if a.json:
        CM.emit_json(tree)
    else:
        _print(tree["root"])
        s = tree["summary"]
        print(f"({s['steps']} steps: {s['known']} known, {s['unknown']} unknown, {s['cut']} cut, {s['loop']} loop; "
              f"{s['untraced_images']} listed image(s) with no session)")
    return 0
