"""
One list of every review package a person may open (round 13, the Reviews hub). Data only, no routes: it lives
outside portal/modules/ because modules.discover() imports every module there.

INTERFACE CONTRACT (fixed by the PM before the build; B1 owns this file and fills the bodies; B3's switcher strip
and B1's hub call it):

    list_reviews(cfg, user, can_see, *, show_all=False, access_mode=False, host=None, with_facets=True) -> list[dict]
        Every row is filtered through reviews.may_open BEFORE it is built, and built from this field allowlist only
        (never review_package.json "meta", "cards", "path", decoy counts, or who/via on audits):
          id          package folder name ("sandbox" for the practice sandbox)
          kind        "prediction" | "label_review" | "audit" | "sandbox"
          title       readable name
          made_from   short text ("v5 F5 · 677 images · Session 20261106-2")
          owner       portal id, or None on audits and the sandbox
          reviewers   list of portal ids ([] on audits)
          role        "yours" | "reviewer" | ""
          status      "not_started" | "in_progress" | "finished" | "applied"
          cards, decided   ints from reviews.card_progress, or None
          n_images    int or None
          last        epoch seconds of the last edit (else built time), or None
          open_url    "/review/<id>/", "/review/" (core package, admin only), the sandbox link, or None
          go_url      "/reviews/go/<id>"
          mode        "cards" if cards are open, else "editor"
          needs_you   bool: owner or reviewer, status not_started/in_progress, open_url set, not the owner of an audit
          can_assign  bool: owner or admin, a served pred_/lab_ package, never the core package
          blind       bool (audits)
          via         owner_via or None (None on audits)
          facets      portal.listing.facets(...) for prediction rows, else {} (B2's helper)
        Excluded/core duplicates are dropped (the core row opens /review/); dot-folders and folders the multi server
        can't serve are skipped; the sandbox is admin only and absent when access_mode is True.
        show_all=False keeps "Mine": rows with a role, plus (admins) the core package and the sandbox.
        host (optional, B1 addition): the request's Host header, for the sandbox's external link; without it the
        sandbox row's open_url is None and its go_url (/reviews/go/sandbox) resolves the link.
        with_facets (optional, B1 addition): False skips listing.facets (facets {}) where they are not shown (the
        go route, the reviewers panel, the switcher).

    needs_you(rows) -> list[dict]
        The "Needs your review" rows, most recent activity first.

    switcher_groups(rows, current_id, *, per_group=8) -> list[tuple[str, list[dict]]]
        Groups for the switcher strip's dropdown: "Needs you" first, then by kind, at most per_group rows each.

Where rows come from: reviews.served_packages (pred_/lab_ the multi server serves, with a key, may_open first), the
portal's own real-review package (cfg.review_package, admins only) and the sandbox. Status, first match wins: an
explicit registry status (a review_package line with status_by), applied (titles.applied_map: a label set was built
from it), finished (every card decided), in progress (any edit), not started. No cache of its own: card_progress,
edit_stats and the registry read are already cached per package / per root.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

KINDS = ("prediction", "label_review", "audit", "sandbox")
STATUSES = ("not_started", "in_progress", "finished", "applied")
KIND_NAMES = {"prediction": "Prediction", "label_review": "Label set", "audit": "Audit", "sandbox": "Practice"}
KIND_GROUPS = {"prediction": "Prediction reviews", "label_review": "Label set reviews", "audit": "Blind audits",
               "sandbox": "Practice"}
STATUS_NAMES = {"not_started": "Not started", "in_progress": "In progress", "finished": "Finished",
                "applied": "Applied"}
SANDBOX_ID = "sandbox"
# registry review_package status (explicit, with status_by) -> hub status
REGISTRY_STATUS = {"built": "not_started", "in_review": "in_progress", "reviewed": "finished", "applied": "applied",
                   "superseded": "finished", "abandoned": "finished"}
FIELDS = ("id", "kind", "title", "made_from", "owner", "reviewers", "role", "status", "cards", "decided", "n_images",
          "last", "open_url", "go_url", "mode", "needs_you", "can_assign", "blind", "via", "facets")
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")


def _read_json(p: Path):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _str(v):
    return v.strip() if isinstance(v, str) and v.strip() else None


def _epoch(iso):
    """Epoch seconds of an ISO time (naive = local), else None."""
    try:
        return datetime.fromisoformat(str(iso)).timestamp()
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def _records(cfg) -> dict:
    """The registry records (portal/modules/data/records.read_registry, cached per root); {} when unreadable."""
    reg = getattr(cfg, "registry", None)
    if not reg:
        return {}
    try:
        from portal.modules.data import records as DREC
        return DREC.read_registry(reg)[0] or {}
    except Exception:                                   # an unreadable registry must not take the hub down
        return {}


def core_dir(cfg) -> Path | None:
    """The portal's own real-review package folder (cfg.review_package, served by :8766 at /review/), or None."""
    p = getattr(cfg, "review_package", None)
    if not p:
        return None
    p = Path(p)
    return p if p.is_dir() else None


def _same(a: Path, b: Path | None) -> bool:
    if b is None:
        return False
    try:
        return Path(a).resolve() == Path(b).resolve()
    except OSError:
        return str(a).lower() == str(b).lower()


def kind_of(pkg_id: str, meta: dict) -> str:
    """prediction (pred_*); audit (a compare package that is not purpose "revise": blind); else label_review."""
    if pkg_id.startswith("pred_") or meta.get("kind") == "prediction":
        return "prediction"
    if meta.get("kind") == "compare" and meta.get("purpose") != "revise":
        return "audit"
    return "label_review"


def _compare_sides(meta: dict) -> tuple:
    from portal import titles as TT
    from portal.modules import reviews as REV
    cmp = meta.get("compare") if isinstance(meta.get("compare"), dict) else {}
    a = cmp.get("a") if isinstance(cmp.get("a"), dict) else {}
    b = cmp.get("b") if isinstance(cmp.get("b"), dict) else {}
    a_text = TT.lv_from_path(a.get("coco") or a.get("id")) or TT.lv_from_path(meta.get("base")) or "labels"
    b_text = (TT.lv_from_path(b.get("coco") or b.get("id")) or REV._folder_name(b.get("weights"))
              or _str(cmp.get("label")) or "a model")
    return a_text, b_text


def _title(pkg_id, meta, kind, records, req) -> str:
    from portal import names
    from portal import titles as TT
    if kind == "prediction" and req:
        name = names.display_name(req)
        if name:
            return name
    if _str(meta.get("title")):
        return meta["title"].strip()
    if kind == "label_review" and meta.get("kind") == "compare":
        a, b = _compare_sides(meta)
        return f"Revise {a} against {b}"
    t = TT.package_title(records, pkg_id, meta)
    return t[len("Audit: "):] if kind == "audit" and t.startswith("Audit: ") else t


def _made_from(meta, kind, req) -> str:
    from portal.modules import reviews as REV
    n = meta.get("n_images") if isinstance(meta.get("n_images"), int) and not isinstance(meta.get("n_images"), bool) \
        else None
    if kind == "audit":                                 # counts only: never which sets are compared
        return " · ".join(x for x in ("Blind comparison", f"{n} images" if n is not None else "") if x)
    if kind == "label_review" and meta.get("kind") == "compare":
        a, b = _compare_sides(meta)
        return " · ".join(x for x in (f"{b} vs {a}", f"{n} images" if n is not None else "") if x)
    out = REV.made_from(meta)
    src = (req or {}).get("source") if isinstance((req or {}).get("source"), dict) else {}
    sess = _str(src.get("session")) if src.get("kind") != "upload" else None
    if kind == "prediction" and sess:
        out = " · ".join(x for x in (out, f"Session {sess}") if x)
    return out


def _status(pkg_id, records, applied, decided, cards, last_edit) -> str:
    reg = (records.get("review_package") or {}).get(pkg_id)
    if isinstance(reg, dict) and _str(reg.get("status_by")) and reg.get("status") in REGISTRY_STATUS:
        return REGISTRY_STATUS[reg["status"]]
    if pkg_id in applied:
        return "applied"
    if cards is not None and decided is not None and decided >= cards:
        return "finished"
    if last_edit is not None or (decided or 0) > 0:
        return "in_progress"
    return "not_started"


def _facets(ps_dir, req, user, records) -> dict:
    """listing.facets for a prediction package's set (B2's helper); {} when it can't be built."""
    if ps_dir is None:
        return {}
    try:
        from portal import listing
        from portal.modules.data import records as DREC

        def hidden(kind, rid):
            return DREC.ref_hidden(records, user, rid, kind)

        out = listing.facets(ps_dir, req if isinstance(req, dict) else {}, viewer=user, hidden=hidden)
        return out if isinstance(out, dict) else {}
    except Exception:                                   # noqa: BLE001  a facet failure never hides the row
        return {}


def _row(d: Path, meta: dict, *, user, records, applied, proot, core: bool, with_facets: bool = True) -> dict:
    """One allowlisted row. The caller has already checked access (may_open, or admin for the core package)."""
    from portal.modules import reviews as REV
    pid = d.name
    me = user.get("id") if isinstance(user, dict) else None
    is_admin = isinstance(user, dict) and user.get("role") == "admin"
    kind = kind_of(pid, meta)
    blind = kind == "audit"
    owner = _str(meta.get("owner"))
    revs = [r for r in meta["reviewers"] if isinstance(r, str)] if isinstance(meta.get("reviewers"), list) else []
    role = "yours" if owner and owner == me else "reviewer" if me and me in revs else ""
    decided, cards = REV.card_progress(d)
    _, last_edit = REV.edit_stats(d)
    reg = (records.get("review_package") or {}).get(pid)
    built = _epoch(meta.get("built")) or (_epoch(reg.get("built")) if isinstance(reg, dict) else None)
    status = _status(pid, records, applied, decided, cards, last_edit)
    ps_dir = Path(proot) / pid[len("pred_"):] if kind == "prediction" and proot else None
    req = _read_json(ps_dir / "request.json") if ps_dir is not None else None
    req = req if isinstance(req, dict) else None
    facets = _facets(ps_dir, req, user, records) if with_facets else {}
    open_url = "/review/" if core else f"/review/{pid}/"
    via = _str(meta.get("owner_via")) or _str((req or {}).get("owner_via"))
    n = meta.get("n_images")
    return {
        "id": pid, "kind": kind, "title": _title(pid, meta, kind, records, req),
        "made_from": _made_from(meta, kind, req),
        "owner": None if blind else owner, "reviewers": [] if blind else revs, "role": role, "status": status,
        "cards": cards, "decided": decided, "n_images": n if isinstance(n, int) and not isinstance(n, bool) else None,
        "last": last_edit or built, "open_url": open_url, "go_url": f"/reviews/go/{pid}",
        "mode": "cards" if cards and decided is not None and decided < cards else "editor",
        "needs_you": bool(role) and status in ("not_started", "in_progress") and not (blind and role == "yours"),
        "can_assign": not core and (is_admin or (bool(owner) and owner == me)),
        "blind": blind, "via": None if blind else via, "facets": facets,
    }


def sandbox_url(host, manifest=None) -> str | None:
    """The sandbox's external link (modules.json "sandbox" entry, portal.app.external_href), or None."""
    if not host:
        return None
    try:
        from portal import app as APP
        mod = next((m for m in APP.manifest(manifest or APP.MANIFEST) if m.get("id") == "sandbox" and m.get("external")), None)
        return APP.external_href(mod, host) if mod else None
    except Exception:                                   # noqa: BLE001  no manifest: no link
        return None


def _sandbox_row(host, manifest=None) -> dict:
    return {"id": SANDBOX_ID, "kind": "sandbox", "title": "Practice sandbox",
            "made_from": "Practice copy · edits never ship", "owner": None, "reviewers": [], "role": "",
            "status": "not_started", "cards": None, "decided": None, "n_images": None, "last": None,
            "open_url": sandbox_url(host, manifest), "go_url": f"/reviews/go/{SANDBOX_ID}", "mode": "editor",
            "needs_you": False, "can_assign": False, "blind": False, "via": None, "facets": {}}


def list_reviews(cfg, user, can_see, *, show_all=False, access_mode=False, host=None, with_facets=True) -> list:
    from portal import titles as TT
    from portal.modules import reviews as REV
    from portal.modules import results as RES
    if not isinstance(user, dict) or not user.get("id"):
        return []
    is_admin = user.get("role") == "admin"
    records = _records(cfg)
    applied = TT.applied_map(records)
    proot = RES.predict_root(cfg)
    core = core_dir(cfg)
    rows = []
    for d, meta in REV.served_packages(cfg, user, can_see):        # may_open is checked inside, before any read
        if d.name.startswith(".") or _same(d, core):
            continue
        rows.append(_row(d, meta, user=user, records=records, applied=applied, proot=proot, core=False,
                         with_facets=with_facets))
    if is_admin and core is not None and ID_RE.match(core.name) and not core.name.startswith("."):
        meta = _read_json(core / "review_package.json")
        if isinstance(meta, dict) and (core / "data.js").is_file():
            rows.append(_row(core, meta, user=user, records=records, applied=applied, proot=proot, core=True,
                             with_facets=with_facets))
    if is_admin and not access_mode:
        rows.append(_sandbox_row(host, getattr(cfg, "manifest", None)))
    if not show_all:
        rows = [r for r in rows if r["role"] or (is_admin and r["open_url"] == "/review/") or r["kind"] == "sandbox"]
    rows.sort(key=lambda r: (r["last"] or 0, r["id"]), reverse=True)
    return [{k: r[k] for k in FIELDS} for r in rows]


def needs_you(rows) -> list:
    """The rows that need this person, most recent activity first (last, else built time)."""
    return sorted((r for r in rows if r.get("needs_you") and r.get("open_url")),
                  key=lambda r: (r.get("last") or 0, r.get("id") or ""), reverse=True)


def switcher_groups(rows, current_id, *, per_group: int = 8) -> list:
    """[(heading, rows)]: "Needs you" first, then one group per kind (KIND_GROUPS order), each capped at per_group.
    The current package is kept in its group even past the cap; empty groups are left out."""
    out = []
    need = needs_you(rows)
    if need:
        out.append(("Needs you", _cap(need, current_id, per_group)))
    taken = {r["id"] for r in need}
    for kind in KINDS:
        rest = [r for r in rows if r.get("kind") == kind and r["id"] not in taken]
        if rest:
            out.append((KIND_GROUPS[kind], _cap(rest, current_id, per_group)))
    return out


def _cap(rows, current_id, n) -> list:
    head = list(rows[:max(0, n)])
    if current_id and all(r["id"] != current_id for r in head):
        cur = next((r for r in rows if r["id"] == current_id), None)
        if cur is not None:
            head.append(cur)
    return head
