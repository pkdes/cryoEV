"""
Reviews module (CryoEV portal, all roles; round 13, the Reviews hub): every review a person may open, in one place.

    GET  /reviews/                       the hub: "Needs your review" cards (first 4, ?more=1 for all), then "All reviews"
                                         with kind tabs (?kind=), Mine | All for admins (?show=all), the shared list
                                         filters (portal/listing.py; status "open" by default) and grouping. Rows come
                                         from portal/review_index.list_reviews (allowlisted fields only).
    GET  /reviews/go/<id>                -> the review's open_url + ?mode=cards (cards still open) or ?mode=editor;
                                         the core package -> /review/ (admins); "sandbox" -> the sandbox link (admins,
                                         not on the Cloudflare door). One identical 404 for unknown, not yours, not
                                         served, or the core package for a non-admin.
    GET  /reviews/<id>/reviewers         the Assign reviewers panel (owner or admin)
    POST /reviews/<id>/reviewers         add= / remove= (form token t; the Gate checks Origin). Roster ids only (unrevoked),
                                         at most MAX_REVIEWERS, never a blind audit's owner. Changes only `reviewers` in
                                         review_package.json (OS lock <review root>/.meta_locks/<id>.lock, re-read, temp
                                         file + os.replace), one audit line "reviewers changed" {added, removed}, 303 to
                                         the hub with a flash. Self-logged: portal/app.py SELF_LOGGED.
    GET  /reviews/<id>/export.csv        an object set's answers (round 14, contract C4: annotation/object_set.export,
    GET  /reviews/<id>/export.json       csv / COCO with attributes): admins and the set's maker; anyone who may open it
                                         only once the set is closed (closed: true, or a registry status past review).
                                         The same 404 as the hub otherwise; never with an agent token. One audit line
                                         "object set exported" {format}.

Rules:
  - Access = may_open(): review_package.json `owner` via ctx.can_see (admins all, others their own), or the person's
    id in its `reviewers` list. Checked before a row is built or rendered, so a package that is not yours leaves no
    trace in the page (id, name or owner). Reviewers: owner or admin only (ctx.can_see on the owner, not may_open),
    and the package is resolved exactly like portal.app pred_mount.lookup, never the portal's real-review package.
  - Only what the Open link can open: served_packages() = the portal route's id rule (portal/app.py PRED_ID_RE), a
    key, and what the multi server serves: serving() = multi_review_server.serving() (its prefixes: pred_, plus lab_
    only with $CJ_REVIEW_PREFIXES, never when the services config can't be read; its excluded names: the services
    config's single-package review_server.py packages, $CJ_REVIEW_EXCLUDE) plus this portal's own real-review package
    (cfg.review_package). The same serving() gates /review/<id>/ in portal/app.py.
  - Audits (blind compare packages) show counts only: no owner, reviewers, who decided or decoy counts.
  - Object sets (round 14, pred_obj_*): their own kind tab ("Object sets", shown when there is one); progress =
    items on which every assigned reviewer has set every required attribute (object_progress).
  - Cards = card_progress(): the Quick check queue as the tool builds it in its default mode (flagged objects):
    predictions flagged u at or above data.js cand_conf (0.35 without it), labels flagged u. Decided = the card's own
    decision is set (not a September seed), or its image is rejected. Cached per package by data.js and the edits
    folder's mtimes. edit_stats() gives the newest edit time.
  - This prefix (/reviews) is not the core /review page and does not match the Gate's /review/<pkg> owner rule
    (that needs the slash after /review): the module's own ROLES decide, all three roles.
  - Every text goes through html.escape. Styles are inline (<style>): /cj/ serves a fixed set of files.
"""
from __future__ import annotations

import hashlib
import hmac
import html
import json
import os
import re
import secrets
import threading
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qs, quote

from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import RedirectResponse, Response
from starlette.routing import Route

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


# ---------------------------------------------------------------- object sets (round 14, contracts C3 / C4)
_objcards: dict = {}
OBJECTS_PREFIX = "pred_obj_"


def object_items(data: dict):
    """(items, required attribute ids) of an object set's data.js: items = [(image file, package object key)], in
    data.js order; None when it is not an object set. An item whose image is not in data.js images is left out."""
    if not isinstance(data, dict) or data.get("kind") != "objects" or not isinstance(data.get("items"), list):
        return None
    files = {}
    for im in data.get("images") or []:
        if isinstance(im, dict) and isinstance(im.get("file"), str):
            files[im["file"]] = im["file"]
            files.setdefault(re.sub(r"\.(png|jpe?g|tiff?)$", "", im["file"], flags=re.I), im["file"])
    items = []
    for it in data["items"]:
        if isinstance(it, dict) and isinstance(it.get("obj"), str) and files.get(str(it.get("image"))):
            items.append((files[str(it["image"])], it["obj"]))
    schema = data.get("schema") if isinstance(data.get("schema"), dict) else {}
    req = [a["id"] for a in schema.get("object_attrs") or []
           if isinstance(a, dict) and isinstance(a.get("id"), str) and a.get("required") is True]
    return items, req


def _attr_answers(dec) -> dict:
    """{(object key, attribute id): {reviewer id}} for every a:<obj>:<attr>@<rid> decision with a value."""
    out: dict = {}
    for k, d in (dec or {}).items():
        if not isinstance(k, str) or not k.startswith("a:") or "@" not in k:
            continue
        v = d.get("v") if isinstance(d, dict) else d
        if v is None or v == "" or (isinstance(d, dict) and d.get("seed")):
            continue
        body, rid = k[2:].rsplit("@", 1)
        obj, _, attr = body.rpartition(":")
        if obj and attr and rid:
            out.setdefault((obj, attr), set()).add(rid)
    return out


def object_progress(pkg: Path, items, required, reviewers) -> tuple:
    """(items done, items) of an object set: done = every reviewer has set every required attribute (with no
    required attribute: any attribute). Reviewers = review_package.json `reviewers`; with none assigned, everyone who
    has answered anything in the set (so a set nobody has touched is 0 done)."""
    recs: dict = {}
    for file in {f for f, _ in items}:
        rec = _read_json(Path(pkg) / "edits" / f"{file}.json")
        edit = rec.get("edit") if isinstance(rec, dict) else None
        recs[file] = _attr_answers(edit.get("dec") if isinstance(edit, dict) else None)
    who = set(r for r in reviewers if isinstance(r, str) and r)
    if not who:
        who = {rid for ans in recs.values() for rids in ans.values() for rid in rids}
    if not who:
        return 0, len(items)
    done = 0
    for file, obj in items:
        ans = recs.get(file) or {}
        if required:
            ok = all(who <= ans.get((obj, a), set()) for a in required)
        else:
            ok = all(any(r in rids for (o, _), rids in ans.items() if o == obj) for r in who)
        done += ok
    return done, len(items)


def _object_card_progress(pkg: Path, dsig, esig) -> tuple | None:
    """card_progress for an object set (None when the package is not one). Cached by data.js, edits and
    review_package.json (its reviewers)."""
    try:
        m = (pkg / "review_package.json").stat()
        msig = (m.st_mtime_ns, m.st_size)
    except OSError:
        msig = None
    key = str(pkg)
    with _lock:
        hit = _objcards.get(key)
    if hit and hit[0] == (dsig, esig, msig):
        return hit[1]
    if hit and hit[0][0] == dsig and hit[2] is None:
        return None                                      # known not to be an object set (this data.js)
    parsed = hit[2] if hit and hit[0][0] == dsig else None
    if parsed is None:
        try:
            data = json.loads((pkg / "data.js").read_text(encoding="utf-8").split("=", 1)[1].strip().rstrip(";"))
        except (OSError, ValueError, IndexError):
            data = None
        parsed = object_items(data)
    if parsed is None:
        with _lock:
            _objcards[key] = ((dsig, esig, msig), None, None)
        return None
    meta = _read_json(pkg / "review_package.json")
    revs = meta.get("reviewers") if isinstance(meta, dict) and isinstance(meta.get("reviewers"), list) else []
    got = object_progress(pkg, parsed[0], parsed[1], revs)
    with _lock:
        _objcards[key] = ((dsig, esig, msig), got, parsed)
    return got


def card_progress(pkg: Path) -> tuple:
    """(cards decided, cards in all) for a package, or (None, None) when its data.js can't be read. Cached by the
    data.js and edits folder mtimes. An object set (round 14): (items done by every reviewer, items)."""
    pkg = Path(pkg)
    try:
        st = (pkg / "data.js").stat()
    except OSError:
        return None, None
    if pkg.name.startswith(OBJECTS_PREFIX):            # contract C4: every object set is pred_obj_<slug>_<4hex>
        try:
            esig0 = (pkg / "edits").stat().st_mtime_ns
        except OSError:
            esig0 = None
        got = _object_card_progress(pkg, (st.st_mtime_ns, st.st_size), esig0)
        if got is not None:
            return got
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
        out.append(f"{meta['n_images']} image{'' if meta['n_images'] == 1 else 's'}")
    return " · ".join(out)


def serving(cfg=None, review_package=None) -> tuple:
    """(prefixes, excluded lower-case names): what the multi server serves (multi_review_server.serving(), the one
    source), minus the portal's own real-review package (review_package, else cfg.review_package)."""
    from annotation import multi_review_server as MRS
    own = review_package or getattr(cfg, "review_package", None)
    return MRS.serving(exclude=[str(own)] if own else [])


def served_packages(cfg, user: dict, can_see) -> list:
    """[(package folder, review_package.json)] for every package this person may open (may_open, checked before
    anything else is read from it) that the Open link can open: the portal route's id rule (PRED_ID_RE), what the
    multi server serves (serving(): prefixes, excluded names, this portal's own real-review package) and a key.
    Sorted by folder name. The one source for portal/review_index.py (the hub, go, the switcher)."""
    root = RES.review_root(cfg)
    if root is None:
        return []
    try:
        dirs = sorted(d for d in Path(root).iterdir() if PKG_RE.fullmatch(d.name) and d.is_dir())
    except OSError:
        return []
    from annotation import multi_review_server as MRS
    from portal.app import PRED_ID_RE
    prefixes, excluded = serving(cfg)
    out = []
    for d in dirs:
        # only what the Open link can open: the same checks as app.pred_package and the review server
        if not PRED_ID_RE.match(d.name) or d.name.lower() in excluded \
                or MRS.package_dir(Path(root), d.name, prefixes) is None or not MRS.package_key(d):
            continue
        meta = _read_json(d / "review_package.json")
        if isinstance(meta, dict) and may_open(user, meta, can_see):
            out.append((d, meta))
    return out


# ---------------------------------------------------------------- reviewers (owner or admin; review_package.json)
MAX_REVIEWERS = 20
MAX_FORM = 4096
LOCK_DIR = ".meta_locks"
LOCK_WAIT_S = 2.0
REPLACE_RETRY_S = 1.0


class ReviewersError(Exception):
    """A refused reviewers change; str(e) is a plain sentence for the page (never a path). busy=True: try again."""

    def __init__(self, text: str, busy: bool = False):
        super().__init__(text)
        self.busy = busy


def reviewer_ids(people) -> list:
    """The one reviewer rule (this hub's Assign panel and Start a review): roster ids, not revoked, with a known
    role. people = auth.Roster.people()."""
    from portal import auth as A
    return sorted(pid for pid, p in (people or {}).items()
                  if isinstance(pid, str) and isinstance(p, dict) and not p.get("revoked") and p.get("role") in A.ROLES)


def roster_ids(roster_path) -> list:
    """Portal ids a reviewer may be (reviewer_ids) from the roster file; [] when it can't be read."""
    from portal import auth as A
    try:
        people = A.Roster(roster_path).people() if roster_path else {}
    except Exception:                                   # noqa: BLE001  unreadable roster: nobody
        people = {}
    return reviewer_ids(people)


def _replace(tmp: Path, dest: Path, retry_s: float = REPLACE_RETRY_S) -> None:
    """os.replace, retried for about retry_s while a reader holds the file open (Windows: PermissionError)."""
    end = time.monotonic() + retry_s
    while True:
        try:
            os.replace(tmp, dest)
            return
        except PermissionError:
            if time.monotonic() >= end:
                raise
            time.sleep(0.05)


def write_reviewers(root, pkg: Path, add=(), remove=(), *, allowed=None, lock_wait: float = LOCK_WAIT_S,
                    retry_s: float = REPLACE_RETRY_S) -> tuple:
    """Change only `reviewers` in <pkg>/review_package.json: OS lock <root>/.meta_locks/<pkg>.lock, re-read the file,
    apply remove then add, write a temp file in the same folder and os.replace it (retried on PermissionError).
    Every other key is kept as read. allowed = the ids `add` may hold (the roster). -> (added, removed) as applied.
    Raises ReviewersError for a refused change (unknown id, the audit's owner, over MAX_REVIEWERS, busy, unreadable)."""
    from data_utils.oslock import acquire_lock
    from portal import titles as TT
    pkg = Path(pkg)
    add = [a for a in dict.fromkeys(add) if isinstance(a, str) and a]
    remove = [r for r in dict.fromkeys(remove) if isinstance(r, str) and r]
    if allowed is not None and any(a not in allowed for a in add):
        raise ReviewersError("Reviewers must be people on the portal roster.")
    try:
        locks = Path(root) / LOCK_DIR
        locks.mkdir(exist_ok=True)
        lk = acquire_lock(locks / f"{pkg.name}.lock", wait=lock_wait)
    except (OSError, TypeError):
        lk = None
    if lk is None:
        raise ReviewersError("Someone is changing this review's reviewers right now. Try again in a moment.", busy=True)
    try:
        meta = _read_json(pkg / "review_package.json")
        if not isinstance(meta, dict):
            raise ReviewersError("This review's settings could not be read, so nothing was changed.")
        old = [r for r in meta["reviewers"] if isinstance(r, str)] if isinstance(meta.get("reviewers"), list) else []
        owner = meta.get("owner") if isinstance(meta.get("owner"), str) else None
        if TT.is_blind(meta, pkg) and owner and owner in add:
            raise ReviewersError("The audit's owner can't be a reviewer on it.")
        new = [r for r in old if r not in remove]
        new += [a for a in add if a not in new]
        added = [a for a in new if a not in old]
        removed = [r for r in old if r not in new]
        if added and len(new) > MAX_REVIEWERS:      # the cap stops adding only: an over-cap list can still be trimmed
            raise ReviewersError(f"A review can have at most {MAX_REVIEWERS} reviewers.")
        if not added and not removed:
            return [], []
        meta["reviewers"] = new
        tmp = pkg / f".review_package.json.{secrets.token_hex(4)}.tmp"
        try:
            tmp.write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
            _replace(tmp, pkg / "review_package.json", retry_s)
        except OSError:
            try:
                os.remove(tmp)
            except OSError:
                pass
            raise ReviewersError("The review's settings are open elsewhere and could not be saved. Try again.",
                                 busy=True) from None
        return added, removed
    finally:
        lk.close()


# ---------------------------------------------------------------- the hub page (styles inline: /cj/ serves a fixed set)
STYLE = """<style>
.cj-rv{display:flex;flex-direction:column;gap:26px}
.cj-rv h1{margin:0;font-size:1.9rem;line-height:1.1}
.cj-rv h2{margin:0;font-size:1.3rem}
.cj-rv-head{display:flex;align-items:flex-end;justify-content:space-between;gap:14px;flex-wrap:wrap}
.cj-rv-head p{margin:6px 0 0}
.cj-rv a.cj-rv-btn,.cj-rv-btn{display:inline-flex;align-items:center;gap:6px;padding:8px 14px;border-radius:6px;background:var(--acc);
  border:1px solid var(--acc);color:var(--card);font-weight:600;text-decoration:none;white-space:nowrap}
.cj-rv-btn2{display:inline-flex;align-items:center;gap:4px;padding:5px 10px;border-radius:6px;border:1px solid var(--line);
  background:var(--card);color:var(--acc);font-weight:600;font-size:.85rem;text-decoration:none;white-space:nowrap}
.cj-rv-sec{display:flex;flex-direction:column;gap:12px}
.cj-rv-sechead{display:flex;align-items:baseline;gap:10px;flex-wrap:wrap}
.cj-rv-n{font-family:var(--cj-mono);font-size:.78rem;color:var(--mut)}
.cj-rv-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:14px}
.cj-rv-card{background:var(--card);border:1px solid var(--line);border-radius:var(--cj-r);padding:13px;display:flex;
  flex-direction:column;gap:8px;min-width:0}
.cj-rv-card h3{margin:0;font-size:1rem;line-height:1.3;overflow-wrap:anywhere}
.cj-rv-line{display:flex;align-items:center;gap:6px;flex-wrap:wrap}
.cj-rv-push{margin-left:auto}
.cj-rv-chip{display:inline-flex;align-items:center;padding:1px 7px;border-radius:4px;font-size:.74rem;font-weight:600;
  white-space:nowrap;border:1px solid var(--line);background:var(--bg);color:var(--fg)}
.cj-rv-k-prediction{color:var(--acc);background:var(--accbg)}
.cj-rv-k-label_review{color:var(--prod);background:var(--prodbg)}
.cj-rv-k-audit{color:var(--warn);background:var(--warnbg)}
.cj-rv-k-objects{color:#7c3aed;background:color-mix(in srgb,#7c3aed 12%,transparent)}
.cj-rv-tag{padding:0 6px;border-radius:4px;font-size:.72rem;font-weight:600;border:1px dashed var(--mut);color:var(--mut)}
.cj-rv-lab{padding:0 6px;border-radius:4px;font-size:.72rem;font-weight:600;color:var(--warn);background:var(--warnbg)}
.cj-rv-via{align-self:flex-start;padding:0 6px;border-radius:999px;font-size:.72rem;color:var(--prod);background:var(--prodbg)}
.cj-rv-bar-wrap{display:flex;flex-direction:column;gap:4px}
.cj-rv-bar{display:block;height:6px;border-radius:3px;overflow:hidden;background:color-mix(in srgb,var(--mut) 22%,transparent)}
.cj-rv-bar>span{display:block;height:100%;border-radius:3px;background:var(--acc)}
.cj-rv-prog{font-family:var(--cj-mono);font-size:.78rem}
.cj-rv-bar2{display:flex;flex-wrap:wrap;gap:10px;align-items:center;justify-content:space-between}
.cj-rv-tabs{display:flex;flex-wrap:wrap;gap:4px;padding:3px;border-radius:8px;background:color-mix(in srgb,var(--line) 70%,transparent)}
.cj-rv-tabs a{padding:5px 11px;border-radius:6px;text-decoration:none;color:var(--mut);font-size:.88rem}
.cj-rv-tabs a[aria-current]{background:var(--card);color:var(--fg);font-weight:600;box-shadow:0 1px 2px var(--cj-shadow)}
.cj-rv-scope{display:flex;border:1px solid var(--line);border-radius:6px;overflow:hidden}
.cj-rv-scope a{padding:5px 12px;text-decoration:none;color:var(--fg);background:var(--card);font-size:.88rem}
.cj-rv-scope a+a{border-left:1px solid var(--line)}
.cj-rv-scope a[aria-current]{background:var(--acc);color:var(--card);font-weight:600}
.cj-rv-list{background:var(--card);border:1px solid var(--line);border-radius:var(--cj-r);overflow:hidden}
.cj-rv-row{display:flex;flex-wrap:wrap;gap:6px 12px;align-items:center;padding:10px 14px;border-top:1px solid var(--line)}
.cj-rv-row:first-child{border-top:0}
.cj-rv-th{font-size:.72rem;font-weight:600;letter-spacing:.04em;text-transform:uppercase;color:var(--mut);
  background:color-mix(in srgb,var(--line) 35%,transparent)}
.cj-rv-grp{display:flex;gap:8px;align-items:baseline;padding:7px 14px;border-top:1px solid var(--line);font-weight:600;
  font-size:.88rem;background:color-mix(in srgb,var(--acc) 8%,transparent)}
.cj-rv-c1{flex:3 1 280px;min-width:0;display:flex;flex-direction:column;gap:3px;overflow-wrap:anywhere}
.cj-rv-c2{flex:1 1 110px;display:flex;flex-direction:column;gap:3px;font-size:.88rem;min-width:0;overflow-wrap:anywhere}
.cj-rv-c3{flex:1.2 1 140px;display:flex;gap:8px;align-items:center;flex-wrap:wrap;font-size:.88rem;min-width:0}
.cj-rv-c4{flex:1.2 1 130px}
.cj-rv-c5{flex:.8 1 90px;font-size:.85rem;color:var(--mut)}
.cj-rv-c6{flex:.8 1 90px}
.cj-rv-c7{flex:0 0 auto;margin-left:auto}
.cj-rv-pill{display:inline-flex;padding:2px 9px;border-radius:999px;font-size:.78rem;font-weight:500;white-space:nowrap;
  background:color-mix(in srgb,var(--mut) 16%,transparent);color:var(--fg)}
.cj-rv-s-in_progress{background:var(--accbg);color:var(--acc)}
.cj-rv-s-finished,.cj-rv-s-applied{background:var(--prodbg);color:var(--prod)}
.cj-rv-note{margin:0;padding:9px 12px;border-radius:6px;border:1px solid var(--line);background:var(--card)}
.cj-rv-warn{border-color:var(--warn);background:var(--warnbg);color:var(--warn)}
.cj-rv-empty{padding:22px 14px;text-align:center;color:var(--mut)}
.cj-rv-panel{max-width:560px;background:var(--card);border:1px solid var(--line);border-radius:12px;display:flex;
  flex-direction:column;overflow:hidden;box-shadow:0 8px 24px var(--cj-shadow)}
.cj-rv-panel>div{padding:14px 18px}
.cj-rv-panel>div+div{border-top:1px solid var(--line)}
.cj-rv-panel h1{font-size:1.3rem}
.cj-rv-who{display:flex;flex-wrap:wrap;gap:8px;list-style:none;margin:8px 0 0;padding:0}
.cj-rv-who li{display:inline-flex;align-items:center;gap:4px;padding:3px 4px 3px 12px;border-radius:999px;
  background:var(--accbg);border:1px solid var(--line)}
.cj-rv-who li.cj-rv-none{background:none;border:0;padding:0}
.cj-rv-who form{display:inline}
.cj-rv-x{width:24px;height:24px;border:0;border-radius:50%;background:transparent;color:var(--acc);cursor:pointer;font:inherit;
  font-weight:700}
.cj-rv-add{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
.cj-rv-add label{flex-basis:100%;font-weight:600}
.cj-rv-add select{flex:1 1 200px;padding:6px 8px;border:1px solid var(--line);border-radius:6px;background:var(--card);
  color:var(--fg);font:inherit}
.cj-rv-add button{padding:6px 14px;border-radius:6px;border:1px solid var(--acc);background:var(--card);color:var(--acc);
  font:inherit;font-weight:600;cursor:pointer}
@media (max-width:640px){.cj-rv-th{display:none}.cj-rv-c1{flex-basis:100%}.cj-rv h1{font-size:1.6rem}}
</style>"""

HUB_EMPTY = "No reviews yet. Start one from a finished prediction set, or ask an admin."
STATUS_FILTER = (("open", "Open (hides finished and applied)"), ("any", "Any, including finished"),
                 ("not_started", "Not started"), ("in_progress", "In progress"), ("finished", "Finished"),
                 ("applied", "Applied"))
TABS = (("all", "All"), ("prediction", "Prediction"), ("objects", "Object sets"), ("label_review", "Label sets"),
        ("audit", "Audits"), ("sandbox", "Practice"))
NEEDS_FIRST = 4
NEVER_GROUPED = {"audit": "Blind audit (never grouped)", "sandbox": "Practice sandbox (never grouped)"}


def _when(epoch, built=False) -> str:
    if not epoch:
        return "—"
    try:
        dt = datetime.fromtimestamp(epoch)
    except (OverflowError, OSError, ValueError, TypeError):
        return "—"
    day = "today" if dt.date() == datetime.now().date() else f"{dt.day} {dt.strftime('%b')}"
    return f"{'built ' if built else ''}{day} {dt.strftime('%H:%M')}"


def cards_text(n, kind=None) -> str:
    """"1 card", "0 cards", "12 cards"; an object set counts items ("1 item", "155 items")."""
    word = "item" if kind == "objects" else "card"
    return f"{n} {word}{'' if n == 1 else 's'}"


def _bar(r) -> str:
    if r.get("cards") is None:
        return '<span class="cj-rv-prog">—</span>'
    d, t = r.get("decided") or 0, r["cards"]
    pct = 0 if not t else max(1.5 if d else 0, round(1000 * d / t) / 10)
    return (f'<span class="cj-rv-bar-wrap"><span class="cj-rv-bar" role="progressbar" aria-valuemin="0" '
            f'aria-valuemax="{t}" aria-valuenow="{d}" aria-label="{"items done by every reviewer" if r.get("kind") == "objects" else "cards decided"}">'
            f'<span style="width:{pct}%"></span></span><span class="cj-rv-prog">{d} of {cards_text(t, r.get("kind"))}'
            f'</span></span>')


def _chip(kind) -> str:
    from portal import review_index as RI
    return f'<span class="cj-rv-chip cj-rv-k-{esc(kind)}">{esc(RI.KIND_NAMES.get(kind, kind))}</span>'


def _status_pill(r) -> str:
    from portal import review_index as RI
    text = "Always open" if r["kind"] == "sandbox" else RI.STATUS_NAMES.get(r["status"], r["status"])
    return f'<span class="cj-rv-pill cj-rv-s-{esc(r["status"])}">{esc(text)}</span>'


def _open_href(r) -> str:
    return r["open_url"] if r["kind"] == "sandbox" and r.get("open_url") else r["go_url"]


def _card(r, me) -> str:
    blind = r["blind"]
    if blind:
        who = "Reviewers hidden while blind"
    elif r["role"] == "yours":
        who = ("Yours" + (f" (made by {r['via']} for {me})" if r.get("via") else "") + " · "
               + (f"reviewers: {', '.join(r['reviewers'])}" if r["reviewers"] else "no reviewers yet"))
    else:
        who = f"{r['owner']}'s · you were added" if r.get("owner") else "You were added"
    open_cards = (r["cards"] - (r["decided"] or 0)) if r.get("cards") else 0
    where = "/review/ in " if r["open_url"] == "/review/" else ""
    if r["kind"] == "objects":
        opens = f"opens the object cards · {cards_text(open_cards, 'objects')} not done by everyone"
    else:
        opens = (f"opens {where}Quick check · {cards_text(open_cards)} open" if r["mode"] == "cards"
                 else f"opens {where}the Editor · no open cards")
    tag = '<span class="cj-rv-tag">blind</span>' if blind else ""
    last = "—" if blind else _when(r["last"], built=r["status"] == "not_started")
    return (f'<article class="cj-rv-card"><div class="cj-rv-line">{_chip(r["kind"])}{tag}'
            f'<span class="cj-rv-push cj-mut">{esc(last)}</span></div>'
            f'<h3>{esc(r["title"])}</h3><div class="cj-mut">{esc(r["made_from"])}</div><div>{esc(who)}</div>'
            f'{_bar(r)}<div class="cj-rv-line"><span class="cj-mut">{esc(opens)}</span>'
            f'<a class="cj-rv-btn cj-rv-push" href="{esc(r["go_url"])}">Continue →</a></div></article>')


def _row_html(r) -> str:
    kind, blind = r["kind"], r["blind"]
    tags = ('<span class="cj-rv-tag">blind</span>' if blind else "") + \
        ('<span class="cj-rv-lab">lab network only</span>' if kind == "sandbox" else "")
    if blind:
        owner, revs = "hidden (blind)", "hidden (blind)"
    elif kind == "sandbox":
        owner, revs = "shared", "anyone in the lab"
    else:
        owner = r["owner"] or "—"
        revs = ", ".join(r["reviewers"]) or "—"
    via = (f'<span class="cj-rv-via">{esc(r["via"])} for {esc(r["owner"])}</span>'
           if r.get("via") and r.get("owner") and not blind else "")
    assign = f'<a href="{PREFIX}/{esc(r["id"])}/reviewers">Assign</a>' if r.get("can_assign") else ""
    if kind == "objects" and r.get("can_assign"):       # owner or admin: the maker's export (blind columns joined)
        assign += f' <a href="{PREFIX}/{esc(r["id"])}/export.csv">Export CSV</a>'
    last = "—" if blind or kind == "sandbox" else _when(r["last"], built=r["status"] == "not_started")
    label = "Open /review/" if r["open_url"] == "/review/" else ("Open ↗" if kind == "sandbox" else "Open")
    return (f'<div class="cj-rv-row"><div class="cj-rv-c1"><div class="cj-rv-line">{_chip(kind)}'
            f'<b>{esc(r["title"])}</b>{tags}</div><span class="cj-mut">{esc(r["made_from"])}</span></div>'
            f'<div class="cj-rv-c2"><span>{esc(owner)}</span>{via}</div>'
            f'<div class="cj-rv-c3"><span>{esc(revs)}</span>{assign}</div>'
            f'<div class="cj-rv-c4">{_bar(r) if kind != "sandbox" else ""}</div>'
            f'<div class="cj-rv-c5">{esc(last)}</div><div class="cj-rv-c6">{_status_pill(r)}</div>'
            f'<div class="cj-rv-c7"><a class="cj-rv-btn2" href="{esc(_open_href(r))}">{esc(label)}</a></div></div>')


def facet_key(r) -> dict:
    """The facets listing.apply / group read: a prediction row's own, else filled from the row (an audit carries no
    owner, so it never has a person)."""
    f = dict(r.get("facets") or {})
    for k, v in (("person", r.get("owner")), ("via", r.get("via")), ("created", r.get("last")), ("session", None),
                 ("source", None), ("model", None), ("samples", [])):
        f.setdefault(k, v)
    return f


def _link(base_q: str, pairs) -> str:
    """/reviews/?<listing state>&<our own pairs>, our pairs (show / kind / more) only when set."""
    from urllib.parse import urlencode
    q = "&".join(x for x in ((base_q or "").lstrip("?"), urlencode([(k, v) for k, v in pairs if v])) if x)
    return f"{PREFIX}/" + (f"?{q}" if q else "")


def render_hub(rows, *, user, params, flash_html="") -> str:
    """The hub's HTML (escaped). rows = review_index.list_reviews(...) for this viewer (already Mine/All scoped)."""
    from portal import listing
    from portal import review_index as RI
    me = user.get("id")
    is_admin = user.get("role") == "admin"
    show = "all" if is_admin and params.get("show") == "all" else ""
    kind = params.get("kind") if params.get("kind") in RI.KINDS else ""
    more = params.get("more") == "1"
    q = listing.parse_query(params, default_status="open")
    base_q = q.to_query()
    keep = (("show", show), ("kind", kind))

    lead = ("Every review you can open: predictions, label sets, audits and practice." if is_admin
            else "Every review you own or were added to.")
    out = [STYLE, getattr(listing, "STYLE", ""),          # the shared filter form's own styles (B2)
           '<div class="cj-rv"><div class="cj-rv-head"><div><h1>Reviews</h1>'
           f'<p class="cj-mut">{esc(lead)}</p></div><a class="cj-rv-btn" href="/reviews/start/">+ Start a review</a></div>',
           flash_html]

    need = RI.needs_you(rows)
    if need:
        out.append('<section class="cj-rv-sec" aria-labelledby="cj-rv-need-h"><div class="cj-rv-sechead">'
                   f'<h2 id="cj-rv-need-h">Needs your review</h2><span class="cj-rv-n">{len(need)} open</span>'
                   '<span class="cj-mut">Yours with open cards, plus reviews you were added to. Most recent activity '
                   'first.</span></div>')
        out.append('<div class="cj-rv-grid">' + "".join(_card(r, me) for r in (need if more else need[:NEEDS_FIRST]))
                   + "</div>")
        if len(need) > NEEDS_FIRST:
            out.append(f'<p style="margin:0"><a href="{esc(_link(base_q, keep))}">Show fewer</a></p>' if more else
                       f'<p style="margin:0"><a href="{esc(_link(base_q, keep + (("more", "1"),)))}">'
                       f'Show {len(need) - NEEDS_FIRST} more</a></p>')
        out.append("</section>")

    out.append('<section class="cj-rv-sec" aria-labelledby="cj-rv-all-h"><h2 id="cj-rv-all-h">All reviews</h2>')
    if not rows:
        out.append(f'<div class="cj-rv-list"><div class="cj-rv-empty">{esc(HUB_EMPTY)} '
                   '<a href="/reviews/start/">Start a review</a></div></div></section></div>')
        return "".join(out)
    counts = {k: (len(rows) if k == "all" else sum(1 for r in rows if r["kind"] == k)) for k, _ in TABS}
    tabs = []
    for k, label in TABS:
        if k in ("audit", "sandbox", "objects") and not counts[k]:
            continue                                     # no trace of a kind this viewer has none of
        cur = ' aria-current="page"' if k == (kind or "all") else ""
        tabs.append(f'<a href="{esc(_link(base_q, (("show", show), ("kind", "" if k == "all" else k))))}"{cur}>'
                    f'{esc(label)} <span class="cj-rv-n">{counts[k]}</span></a>')
    bar = f'<nav class="cj-rv-tabs" aria-label="Kind">{"".join(tabs)}</nav>'
    if is_admin:
        mine_cur, all_cur = ("", ' aria-current="page"') if show else (' aria-current="page"', "")
        bar += ('<div class="cj-rv-line"><span class="cj-mut">admin</span><nav class="cj-rv-scope" aria-label="Whose reviews">'
                f'<a href="{esc(_link(base_q, (("kind", kind),)))}"{mine_cur}>Mine</a>'
                f'<a href="{esc(_link(base_q, (("show", "all"), ("kind", kind))))}"{all_cur}>All</a></nav></div>')
    out.append(f'<div class="cj-rv-bar2">{bar}</div>')

    people = sorted({p for r in rows if not r["blind"] for p in ([r["owner"]] if r.get("owner") else []) + r["reviewers"]})
    models = sorted({str(r["facets"]["model"]) for r in rows if (r.get("facets") or {}).get("model")})
    form = listing.controls_html(q, action=f"{PREFIX}/", people=people, models=models, statuses=STATUS_FILTER)
    hidden = "".join(f'<input type="hidden" name="{k}" value="{esc(v)}">' for k, v in keep if v)
    if hidden and "</form>" in form:                  # keep the kind tab and Mine/All across a filter submit
        form = form.replace("</form>", hidden + "</form>", 1)
    out.append(form)

    scoped = [r for r in rows if not kind or r["kind"] == kind]
    shown = listing.apply(scoped, q, key=facet_key)
    if q.group == "none":
        groups = [("", shown)]
    else:
        groups = listing.group([r for r in shown if r["kind"] not in NEVER_GROUPED], q.group, key=facet_key)
        groups += [(label, g) for k, label in NEVER_GROUPED.items() for g in [[r for r in shown if r["kind"] == k]] if g]
    body = ['<div class="cj-rv-row cj-rv-th" aria-hidden="true"><span class="cj-rv-c1">Name / made from</span>'
            '<span class="cj-rv-c2">Owner</span><span class="cj-rv-c3">Reviewers</span><span class="cj-rv-c4">Progress</span>'
            '<span class="cj-rv-c5">Last</span><span class="cj-rv-c6">Status</span><span class="cj-rv-c7"></span></div>']
    for label, g in groups:
        if label:
            body.append(f'<div class="cj-rv-grp">{esc(label)} <span class="cj-rv-n">· {len(g)} '
                        f'review{"" if len(g) == 1 else "s"}</span></div>')
        body.extend(_row_html(r) for r in g)
    if not shown:
        body.append('<div class="cj-rv-empty">No reviews match these filters.</div>')
    out.append(f'<div class="cj-rv-list">{"".join(body)}</div>')
    notes = [f"{len(shown)} of {len(rows)} shown."]
    if counts["audit"]:
        notes.append("Audit rows show counts only: no owner, reviewers or decision details.")
    if counts["sandbox"]:
        notes.append("Practice opens the sandbox, which edits never leave.")
    out.append(f'<p class="cj-mut" style="margin:0">{esc(" ".join(notes))}</p></section></div>')
    return "".join(out)


def render_panel(pkg_id, row, meta, roster, token_html, error="", panel_dir=None) -> str:
    """The Assign reviewers panel (owner or admin only). row = the viewer's hub row, for the title and counts."""
    from portal import review_index as RI
    from portal import titles as TT
    blind = TT.is_blind(meta, panel_dir)
    owner = meta.get("owner") if isinstance(meta.get("owner"), str) else None
    revs = [r for r in meta["reviewers"] if isinstance(r, str)] if isinstance(meta.get("reviewers"), list) else []
    kind = RI.kind_of(pkg_id, meta, panel_dir)
    title = (row or {}).get("title") or pkg_id
    prog = (f'<span class="cj-rv-prog">{row.get("decided") or 0} of {cards_text(row["cards"], kind)}</span>'
            if row and row.get("cards") is not None else "")
    act = f"{PREFIX}/{esc(pkg_id)}/reviewers"
    chips = "".join(
        f'<li>{esc(r)}<form method="post" action="{act}">{token_html}<input type="hidden" name="remove" value="{esc(r)}">'
        f'<button class="cj-rv-x" type="submit" aria-label="Remove {esc(r)}" title="Remove {esc(r)}">×</button></form></li>'
        for r in revs) or '<li class="cj-rv-none cj-mut">No reviewers yet.</li>'
    opts, can_add = [], False
    for pid in roster:
        if pid in revs:
            continue
        if blind and pid == owner:
            opts.append(f'<option disabled>{esc(pid)} (audit owner)</option>')
        else:
            can_add = True
            opts.append(f'<option value="{esc(pid)}">{esc(pid)}{" (owner)" if pid == owner else ""}</option>')
    add = (f'<form class="cj-rv-add" method="post" action="{act}">{token_html}'
           '<label for="cj-rv-addsel">Add from the roster</label>'
           f'<select id="cj-rv-addsel" name="add">{"".join(opts)}</select><button type="submit">Add</button></form>'
           if can_add else '<p class="cj-mut" style="margin:0">Everyone on the roster is already a reviewer.</p>')
    owner_txt = f"<span>owner {esc(owner)}</span>" if owner and not blind else ""
    note = ('<p class="cj-rv-note cj-rv-warn">A blind audit: its owner can\'t be a reviewer on it.</p>' if blind else "")
    err = f'<p class="cj-rv-note cj-rv-warn" role="alert">{esc(error)}</p>' if error else ""
    return (STYLE + '<div class="cj-rv"><section class="cj-rv-panel" aria-labelledby="cj-rv-as-h">'
            f'<div><h1 id="cj-rv-as-h">Reviewers for {esc(title)}</h1>'
            f'<div class="cj-rv-line cj-mut" style="margin-top:6px">{_chip(kind)}{owner_txt}{prog}</div></div>'
            f'<div style="display:flex;flex-direction:column;gap:14px">{err}'
            f'<div><b id="cj-rv-cur">Current reviewers</b><ul class="cj-rv-who" aria-labelledby="cj-rv-cur">{chips}</ul>'
            f'</div>{add}<p class="cj-mut" style="margin:0">Only the owner or an admin can change this. Reviewers see '
            f'it under Needs your review. At most {MAX_REVIEWERS}.</p>{note}</div>'
            f'<div style="display:flex;justify-content:flex-end"><a class="cj-rv-btn2" href="{PREFIX}/">Back to reviews</a>'
            '</div></section></div>')


# ---------------------------------------------------------------- object set export (round 14, C4 / C7)
EXPORT_FORMATS = {"csv": ("csv", "text/csv; charset=utf-8"), "json": ("coco", "application/json")}
CLOSED_STATUSES = ("reviewed", "applied", "superseded", "abandoned")


def object_set_closed(meta: dict, reg: dict | None) -> bool:
    """A set is closed when its own review_package.json says closed: true, or its registry line has an explicit
    status (status_by) past review. Open otherwise (the blind columns stay with admins and the maker)."""
    if isinstance(meta, dict) and meta.get("closed") is True:
        return True
    return isinstance(reg, dict) and isinstance(reg.get("status_by"), str) and bool(reg["status_by"])         and reg.get("status") in CLOSED_STATUSES


def may_export(user: dict, meta: dict, closed: bool, can_see) -> bool:
    """Admins and the set's maker (owner) always; anyone who may open the set only once it is closed."""
    if not isinstance(user, dict) or not user.get("id") or not isinstance(meta, dict):
        return False
    if user.get("role") == "admin" or (isinstance(meta.get("owner"), str) and meta["owner"] == user["id"]):
        return True
    return closed and may_open(user, meta, can_see)


def export_object_set(pkg: Path, fmt: str) -> bytes:
    """B3's exporter (annotation/object_set.py, contract C4) -> the file's bytes. fmt "csv": to_csv(table(pkg));
    "coco": json.dumps(coco(pkg))."""
    from annotation import object_set as OS
    if fmt == "csv":
        return OS.to_csv(OS.table(Path(pkg))).encode("utf-8")
    if fmt == "coco":
        return json.dumps(OS.coco(Path(pkg)), ensure_ascii=False).encode("utf-8")
    raise ValueError(f"unknown export format {fmt!r}")


def routes(ctx) -> list:
    from portal import audit as PA
    from portal import review_index as RI
    from portal.modules import admin as AD
    secret = secrets.token_bytes(32)                  # per process, like admin's: a restart makes open forms stale

    def token(request: Request) -> str:
        return hmac.new(secret, hashlib.sha256(AD.binding(request).encode()).digest(), hashlib.sha256).hexdigest()[:40]

    def access_mode() -> bool:
        return getattr(ctx.cfg, "auth", "key") == "access"

    def show(request, title, body, status=200):
        r = ctx.page(request, title, body, active="reviews")
        r.status_code = status
        return r

    def missing(request):
        """The one 404: an unknown id, someone else's, one not served, or the core package for a non-admin."""
        return show(request, "Not found", '<p class="cj-card">No such review, or it isn\'t yours.</p>'
                    f'<p><a href="{PREFIX}/">All reviews</a></p>', 404)

    def rows_for(request, show_all=True, with_facets=True):
        return RI.list_reviews(ctx.cfg, ctx.user(request), ctx.can_see, show_all=show_all, access_mode=access_mode(),
                               host=request.headers.get("host"), with_facets=with_facets)

    def index(request: Request):
        user = ctx.user(request)
        params = request.query_params
        rows = rows_for(request, show_all=user.get("role") == "admin" and params.get("show") == "all")
        text = AD.flash_take(AD.binding(request), params.get("f"))
        flash = f'<p class="cj-rv-note" role="status">{esc(text)}</p>' if text else ""
        return show(request, "Reviews", render_hub(rows, user=user, params=params, flash_html=flash))

    def go(request: Request):
        pid = request.path_params.get("pkg_id", "")
        user = ctx.user(request)
        if pid == RI.SANDBOX_ID:
            url = (RI.sandbox_url(request.headers.get("host"), getattr(ctx.cfg, "manifest", None))
                   if user.get("role") == "admin" and not access_mode() else None)
            return RedirectResponse(url, status_code=303) if url else missing(request)
        row = next((r for r in rows_for(request, with_facets=False) if r["id"] == pid and r["kind"] != "sandbox"), None)
        if row is None or not row.get("open_url"):
            return missing(request)
        return RedirectResponse(f'{row["open_url"]}?mode={row["mode"]}', status_code=303)

    def resolve(request):
        """(package folder, review_package.json) whose reviewers this viewer may change, else None. Resolved exactly
        like portal.app pred_mount.lookup (serving() prefixes, excluded names, a key), never the portal's own
        real-review package; then owner or admin (ctx.can_see on its owner; not may_open: a reviewer can't)."""
        from annotation import multi_review_server as MRS
        from portal import app as APP
        prefixes, excluded = serving(ctx.cfg)
        hit = APP.pred_package(RES.review_root(ctx.cfg), request.path_params.get("pkg_id", ""), prefixes)
        if hit is None or not MRS.package_key(hit[0]) or hit[0].name.lower() in excluded \
                or RI._same(hit[0], RI.core_dir(ctx.cfg)):
            return None
        owner = hit[1].get("owner") if isinstance(hit[1].get("owner"), str) and hit[1]["owner"] else None
        return hit if ctx.can_see(ctx.user(request), owner) else None

    def panel(request, hit, error="", status=200):
        pid = hit[0].name
        row = next((r for r in rows_for(request, with_facets=False) if r["id"] == pid), None)
        body = render_panel(pid, row, hit[1], roster_ids(getattr(ctx.cfg, "roster", None)),
                            AD.hidden_token(token(request)), error, panel_dir=hit[0])
        return show(request, "Assign reviewers", body, status)

    def reviewers_get(request: Request):
        hit = resolve(request)
        return missing(request) if hit is None else panel(request, hit)

    def reviewers_change(request, body, too_big, ctype):
        """The reviewers POST after its body is read: everything that touches files (resolve, the meta lock wait and
        os.replace retries in write_reviewers, the panel, the hub rows for the flash) runs here, in a worker thread
        (reviewers_post), never on the event loop."""
        hit = resolve(request)
        if hit is None:
            return missing(request)
        if request.scope.get("cj_auth") == "bearer":    # never an agent token (the Gate opens none of these to one)
            return panel(request, hit, "Not with an agent token.", 403)
        if too_big:
            return panel(request, hit, "That form is too large.", 413)
        if ctype != "application/x-www-form-urlencoded":
            return panel(request, hit, "Send the form from this page.", 415)
        try:
            form = parse_qs(body.decode("utf-8"), max_num_fields=50)
        except (UnicodeDecodeError, ValueError):
            return panel(request, hit, "That form could not be read.", 400)
        if not hmac.compare_digest((form.get("t") or [""])[0].encode(), token(request).encode()):
            return panel(request, hit, "This form is out of date. Open it again and retry.", 403)
        add, remove = form.get("add") or [], form.get("remove") or []
        if not add and not remove:
            return panel(request, hit, "Pick someone to add or remove.", 400)
        try:
            added, removed = write_reviewers(RES.review_root(ctx.cfg), hit[0], add, remove,
                                             allowed=set(roster_ids(getattr(ctx.cfg, "roster", None))))
        except ReviewersError as e:
            fresh = _read_json(hit[0] / "review_package.json")
            return panel(request, (hit[0], fresh if isinstance(fresh, dict) else hit[1]), str(e), 409 if e.busy else 400)
        pid = hit[0].name
        if added or removed:
            PA.event(ctx.cfg.state_dir, request, "reviewers changed", pid, {"added": added, "removed": removed})
        row = next((r for r in rows_for(request, with_facets=False) if r["id"] == pid), None)
        what = "; ".join(x for x in ((f"added {', '.join(added)}" if added else ""),
                                    (f"removed {', '.join(removed)}" if removed else "")) if x) or "no change"
        tok = AD.flash_put(AD.binding(request), f"Reviewers for {(row or {}).get('title') or pid}: {what}.")
        return RedirectResponse(f"{PREFIX}/?f={quote(tok)}", status_code=303)

    async def reviewers_post(request: Request):
        """Only the body read (at most MAX_FORM bytes) is async; the rest is one sync call in the threadpool."""
        ctype = (request.headers.get("content-type") or "").split(";")[0].strip().lower()
        buf, too_big = bytearray(), False
        async for chunk in request.stream():
            buf += chunk
            if len(buf) > MAX_FORM:
                too_big = True
                break
        return await run_in_threadpool(reviewers_change, request, bytes(buf), too_big, ctype)

    def export_hit(request):
        """(package folder, review_package.json) of an object set this viewer may export, else None. Resolved like
        resolve() (served prefixes, excluded names, a key, never the core package), then kind objects, then
        may_export (admin or maker; others only once the set is closed)."""
        from annotation import multi_review_server as MRS
        from portal import app as APP
        prefixes, excluded = serving(ctx.cfg)
        hit = APP.pred_package(RES.review_root(ctx.cfg), request.path_params.get("pkg_id", ""), prefixes)
        if hit is None or not MRS.package_key(hit[0]) or hit[0].name.lower() in excluded                 or RI._same(hit[0], RI.core_dir(ctx.cfg)) or RI.kind_of(hit[0].name, hit[1], hit[0]) != "objects":
            return None
        reg = (RI._records(ctx.cfg).get("review_package") or {}).get(hit[0].name)
        closed = object_set_closed(hit[1], reg)
        return hit if may_export(ctx.user(request), hit[1], closed, ctx.can_see) else None

    def export_run(request, ext):
        hit = export_hit(request)
        if hit is None:
            return missing(request)
        if request.scope.get("cj_auth") == "bearer":    # never an agent token: the blind columns are in the file
            return show(request, "Export", '<p class="cj-card">Not with an agent token.</p>', 403)
        fmt, ctype = EXPORT_FORMATS[ext]
        try:
            body = export_object_set(hit[0], fmt)
        except Exception as e:                          # noqa: BLE001  B3's exporter refused or is missing
            return show(request, "Export", f'<p class="cj-card">The export could not be made ({esc(type(e).__name__)}).'
                        f'</p><p><a href="{PREFIX}/">All reviews</a></p>', 500)
        PA.event(ctx.cfg.state_dir, request, "object set exported", hit[0].name, {"format": fmt})
        return Response(body, media_type=ctype, headers={
            "Content-Disposition": f'attachment; filename="{hit[0].name}.{ext}"', "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff"})

    async def export_csv(request: Request):
        return await run_in_threadpool(export_run, request, "csv")

    async def export_json(request: Request):
        return await run_in_threadpool(export_run, request, "json")

    return [Route(PREFIX + "/", index, methods=["GET"]),
            Route(PREFIX + "/go/{pkg_id}", go, methods=["GET"]),
            Route(PREFIX + "/{pkg_id}/export.csv", export_csv, methods=["GET"]),
            Route(PREFIX + "/{pkg_id}/export.json", export_json, methods=["GET"]),
            Route(PREFIX + "/{pkg_id}/reviewers", reviewers_get, methods=["GET"]),
            Route(PREFIX + "/{pkg_id}/reviewers", reviewers_post, methods=["POST"])]
