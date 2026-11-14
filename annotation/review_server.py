"""
Serve a label review package (from build_review_package.py) over HTTP and save
edits to disk, so the browser tool can be used remotely (e.g. behind a
Microsoft Dev Tunnel) by several reviewers at once, and still work offline with later sync.

Standard library only. Every request needs the access key: open the printed
URL once (…/?k=<key>) and a cookie carries it from then on. The cookie is named review_key_<port> (this
server's listen port): browsers share cookies across the ports of one host, so two servers on one laptop
(each package with its own key) would overwrite one review_key cookie. The older review_key cookie is still
accepted when it holds this server's key; only the new name is set.

--sandbox (also on when the package folder's name starts with SANDBOX_) puts an amber "SANDBOX" bar on every
page: data.js is served with window.REVIEW_SERVER = {"sandbox": true} appended (the file on disk is unchanged).

Routes:
  GET  /, /data.js, /images/<name>   the package itself
  GET  /api/ping                     {"ok": true, "caps": ["i", "t", "m", "g", "a"]}: caps = what this server accepts
                                     beyond g/p/n decisions (i: image reject / note, t: object low-conf tag / note;
                                     round 12: m: GET /api/meta, g: group actions and the grp field on i:reject;
                                     round 14: a: attribute decisions a:<obj>:<attr>@<rid> / i:a:<attr>@<rid>, taken
                                     only by a package with a schema.json, see "Attribute decisions" below). The
                                     page hides those controls when they are missing (an older server answers
                                     {"ok": true} or fewer caps, and refuses the keys with a 400 / the routes with 404)
  GET  /api/meta                     image metadata (sample, montage, tile, defocus, recipe...), derived read-only
                                     from the prediction set and conversion folders: annotation/review_meta.py
  GET  /api/groups                   group notes and group rejects (edits/_groups.json): annotation/review_groups.py
  POST /api/group?who=<n>            {action: reject|undo|comment, kind: sample|montage, sample, montage?, reason?,
                                     note?}: one group action, applied to every image of the group under the write
                                     lock (annotation/review_groups.py); answers the changed records
  GET  /api/people                   the reviewer roster from annotation/people.json, for the name picker
  GET  /api/edits                    {file: edit record} for every saved image
  PUT  /api/edits?f=<file>           merge one image's record into the saved one (per decision, the latest
                                     received wins: the page's new decisions come marked pending and are
                                     stamped server_t here; see merge_edits) and return the merged record
  POST /api/claim?c=<case>&who=<n>   reserve the card (or "<file>|editor") on a reviewer's screen for CLAIM_SECONDS
  GET  /api/claims                   active claims {case: reviewer}, so other reviewers skip them

One writer per package (round 9): the server holds an OS lock on <review root>/.review_locks/<package>.lock
(data_utils/oslock; next to the package, never inside it, so a rebuild can still rename the folder) for as long as
it runs, and refuses to start when another process holds it. annotation/multi_review_server.py takes the same lock
per package, so the two never write one package's edits at once (their write locks are per process: two processes
on one package would lose saves).

Edits are written to <package>/edits/<file>.json (one file per image, atomic
replace). The record format (decision keys and values) is in docs/REVIEW_TOOL.md;
effective_labels() applies one record to its image without the browser, for the
planned annotation/apply_review_edits.py that turns them into a new COCO export.

Attribute decisions (round 14, contract C3; an object set, annotation/object_set.py): keys a:<obj>:<attr>@<rid>
(<obj> = g<id> | p<k> | n<uid>) and i:a:<attr>@<rid>, at most ATTR_KEY_MAX characters, value {v, t, who, server_t,
pending}. Only a package with <pkg>/schema.json (a label schema, labels/schemas.py) takes them; each new (pending)
one is checked against it (the attribute exists; v is one of its options, true / false, or text through _utf8 within
300 characters; null = undone), its who comes from the request's identity (_request_identity) and its key's @<rid>
is rewritten to that who (a client's rid never counts). Saved copies a request echoes (not pending) are dropped
before the merge, so the saved entry stays as it is. Each accepted decision is also appended to
edits/_attr_log.jsonl ({t, who, key, v, file}) under the package write lock: the history. That file is no image
record (every reader takes edits/*.json). Attributes never change an object: effective_labels ignores the keys, and
prediction_package.is_edited / apply_review_edits leave them out.
Round 14 QC: a new attribute decision needs a TRUSTED identity (trusted_identity: multi_review_server's portal
X-CJ-User; a standalone review_server has none and answers 403) that is one of the set's reviewers
(review_package.json, read per request; no list: anyone on the roster); at most ATTR_PUT_MAX per PUT; in an object set
only on the objects of that image's items (and their `also`). GET /api/edits of a package with a schema.json holds
only the requester's own attribute decisions (none without an identity): reviewers never see each other's answers;
the admin export reads the files. The history logs a key only when its saved entry changed (a resend logs nothing).

Usage:
  python annotation/review_server.py --package "../CryoAI/annotation review/cryoai_v4_review"
  python annotation/review_server.py --package <pkg> --host 100.x.y.z   # this PC's Tailscale IP: phones on the tailnet
  python annotation/review_server.py --package <SANDBOX_pkg> --port 8765   # a practice copy: sandbox bar on
"""
import argparse
import json
import math
import mimetypes
import os
import re
import secrets
import subprocess
import sys
import threading
import time
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

COOKIE = "review_key"  # legacy name, accepted (when it matches) but never set; the cookie is COOKIE_<port>
SANDBOX_PREFIX = "SANDBOX_"
DEC_PREFIXES = ("g", "p", "n")  # g<label id>, p<prediction index>, n<id of an outline drawn in the tool>
# Image and object state (docs/REVIEW_TOOL.md "Reject, notes, low conf"): decisions on keys of their own, so they
# merge per key, carry who and server_t, and undo like any decision. Never extra fields on a class decision.
CAPS = ("i", "t", "m", "g", "a")  # /api/ping caps: i:reject / i:note, t:<objkey>; m: /api/meta; g: group actions
                                  # (round 12); a: attribute decisions (round 14, a package with a schema.json)
IMAGE_KEYS = ("i:reject", "i:note")
TAG_KEY_RE = re.compile(r"t:(?:g[0-9]+|p[0-9]+|n[0-9a-z]+)")   # t:<objkey>, objkey = the object's decision key
REJECT_PREFIX = "reject:"
REJECT_REASONS = ("thick ice", "empty", "contaminated", "crystalline ice", "broken carbon", "aggregates",
                  "out of focus / drift", "other")  # the gallery's tags plus focus (the same list in index.html)
IMAGE_NOTE_MAX = 1000          # characters
TAG_NOTE_MAX = 300
TAG_VALUES = ("lowconf", "plain")  # lowconf = the Low conf tag (+ note); plain = a note without the tag
LOW_CONF_EV = "Low conf EV"    # the class stays (old data, the lab package) and is read as EV + low conf
STATE_FIELDS = frozenset({"v", "t", "who", "server_t", "pending", "note"})  # the only fields of an i: / t: decision
REJECT_FIELDS = STATE_FIELDS | {"grp"}  # i:reject also: grp, the group action that set it (annotation/review_groups.py)
GRP_MAX = 500                  # = review_groups.GRP_MAX
GROUP_BODY_MAX = 16 * 1024
STATE_KEY_MAX = 64             # an i: / t: key longer than this is refused
# Attribute decisions (round 14, contract C3; module doc "Attribute decisions")
ATTR_KEY_MAX = 96
ATTR_KEY_RE = re.compile(r"a:(g[0-9]+|p[0-9]+|n[0-9a-z]+):([a-z][a-z0-9_]{0,15})@(.+)")
IMAGE_ATTR_KEY_RE = re.compile(r"i:a:([a-z][a-z0-9_]{0,15})@(.+)")
ATTR_FIELDS = frozenset({"v", "t", "who", "server_t", "pending"})
ATTR_TEXT_MAX = 300
ATTR_PUT_MAX = 64              # new (pending) attribute decisions one PUT may bring
RECORD = "review_package.json"  # <pkg>/review_package.json: an object set's reviewers (changed by the portal hub)
ATTR_LOG = "_attr_log.jsonl"   # edits/_attr_log.jsonl: the decision history (not an image record: not *.json)
SCHEMA_FILE = "schema.json"    # <pkg>/schema.json: the package's label schema (object sets); never served
MAX_BODY = 20 * 2**20
CLAIM_SECONDS = 90
MAX_CLAIMS = 200      # active claims kept at most (one per reviewer is the norm); the soonest to expire go first
SOCKET_TIMEOUT = 20   # seconds a connection may sit idle before the server drops it
COORD_EPS = 1e-6      # normalized coordinates must lie in [0, 1], give or take float rounding
RECORD_FMT = 1        # edits/<file>.json wrapper version: {file, pkg, fmt, edit, rx} (docs/LABEL_CHANGES.md)
PEOPLE = Path(__file__).with_name("people.json")
WHO_MAX = 40
WHO_RE = re.compile(rf"[\w .\-]{{0,{WHO_MAX}}}")


def cookie_name(port: int) -> str:
    """review_key_<port>: one cookie per server, since a browser sends a host's cookies to all of its ports."""
    return f"{COOKIE}_{port}"


def is_sandbox(pkg: Path, flag: bool = False) -> bool:
    return bool(flag) or Path(pkg).name.startswith(SANDBOX_PREFIX)


def sandbox_js(on: bool) -> bytes:
    """Appended to data.js when served: the page shows the sandbox bar from it. Nothing for a real review."""
    return b"\nwindow.REVIEW_SERVER = {\"sandbox\": true};\n" if on else b""


# ---- one writer per package: an OS lock between processes, one write lock per package within this process ----
LOCK_DIR = ".review_locks"
_HELD: dict = {}      # lock file -> [open handle, holders in this process]
_WRITE: dict = {}     # edits folder -> threading.Lock, shared by every handler this process makes for it
_REG = threading.Lock()


def _norm(p: Path) -> str:
    s = str(Path(p).resolve())
    return s.lower() if os.name == "nt" else s  # one package, whatever case it was reached under


def lock_path(pkg: Path) -> Path:
    """<review root>/.review_locks/<package>.lock: beside the package folder, so a rebuild can rename the folder."""
    pkg = Path(pkg).resolve()
    return pkg.parent / LOCK_DIR / f"{pkg.name}.lock"


def hold_package(pkg: Path) -> bool:
    """Make this process the package's writer: take its OS lock (data_utils/oslock, dies with the process), or count
    one more holder when this process has it already. False when another process holds it."""
    f = lock_path(pkg)
    k = _norm(f)
    with _REG:
        if k in _HELD:
            _HELD[k][1] += 1
            return True
        repo = str(Path(__file__).resolve().parent.parent)
        if repo not in sys.path:
            sys.path.insert(0, repo)
        from data_utils.oslock import acquire_lock
        try:
            f.parent.mkdir(exist_ok=True)
            h = acquire_lock(f)
        except OSError:
            h = None
        if h is None:
            return False
        _HELD[k] = [h, 1]
        return True


def release_package(pkg: Path):
    """One holder fewer; the last one closes the lock file (the OS lock goes with it)."""
    k = _norm(lock_path(pkg))
    with _REG:
        hit = _HELD.get(k)
        if not hit:
            return
        hit[1] -= 1
        if hit[1] <= 0:
            del _HELD[k]
            try:
                hit[0].close()
            except OSError:
                pass


def _write_lock(edits_dir: Path) -> threading.Lock:
    """The package's write lock in this process: shared by every handler made for it (the multi server remakes a
    handler when data.js changes while a kept-alive connection may still use the old one)."""
    with _REG:
        return _WRITE.setdefault(_norm(edits_dir), threading.Lock())


def _annotation(name: str):
    """annotation.<name> (review_meta, review_groups), with the repo on sys.path: this file also runs as a script."""
    repo = str(Path(__file__).resolve().parent.parent)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    import importlib
    return importlib.import_module(f"annotation.{name}")


def package_meta(pkg: Path, valid_files=None) -> dict:
    """review_meta.package_meta(pkg), never raising (a broken folder costs the metadata, not the page); images limited
    to valid_files (this handler's data.js) when given."""
    try:
        meta = _annotation("review_meta").package_meta(pkg)
    except Exception as e:  # noqa: BLE001
        meta = {"schema": "review_meta/1", "package": {"error": f"metadata unavailable ({type(e).__name__})"},
                "images": {}}
    if valid_files is not None:
        meta["images"] = {f: m for f, m in meta.get("images", {}).items() if f in valid_files}
    return meta


def package_schema(pkg: Path):
    """<pkg>/schema.json (a label schema, labels/schemas.py), validated; None when the package has none (every
    package before round 14: attribute decisions are refused) or it doesn't validate."""
    f = Path(pkg) / SCHEMA_FILE
    if not f.is_file():
        return None
    try:
        doc = json.loads(f.read_text(encoding="utf-8"))
        return doc if not _labels_schemas().validate(doc) else None
    except Exception:  # noqa: BLE001  an unreadable schema: no attributes, never a broken page
        return None


def _labels_schemas():
    """labels.schemas, with the repo on sys.path (imported only for a package that has a schema.json)."""
    repo = str(Path(__file__).resolve().parent.parent)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    import importlib
    return importlib.import_module("labels.schemas")


def is_attr_key(k) -> bool:
    """An attribute decision key (a:<obj>:<attr>@<rid>, i:a:<attr>@<rid>): never an object change."""
    return isinstance(k, str) and (k.startswith("a:") or k.startswith("i:a:"))


def parse_attr_key(k):
    """(obj | None for an image attribute, attr, rid) of an attribute key, or None when it isn't one."""
    if not isinstance(k, str) or len(k) > ATTR_KEY_MAX:
        return None
    m = ATTR_KEY_RE.fullmatch(k)
    if m:
        return m.group(1), m.group(2), m.group(3)
    m = IMAGE_ATTR_KEY_RE.fullmatch(k)
    return (None, m.group(1), m.group(2)) if m else None


def attr_key(obj, attr: str, rid: str) -> str:
    return f"a:{obj}:{attr}@{rid}" if obj is not None else f"i:a:{attr}@{rid}"


def check_attrs(edit: dict, schema):
    """Prepare a PUT body's attribute decisions (valid_edit has checked their shape; who is already the request's
    identity): drop the saved copies it echoes (not pending: the saved entry stays), check each new one against the
    package schema, and re-key it to @<its who>. Changes edit in place. -> an error text, or None."""
    dec = edit.get("dec") or {}
    keys = [k for k in dec if is_attr_key(k)]
    if not keys:
        return None
    if schema is None:
        return "this package takes no attribute decisions"
    S = _labels_schemas()
    new = {}
    for k in keys:
        d = dec.pop(k)
        if not (isinstance(d, dict) and d.get("pending") is True):
            continue
        obj, attr, _ = parse_attr_key(k)
        a = S.attr(schema, attr, image=obj is None)
        if a is None:
            return f"unknown attribute {attr!r}"
        v = d.get("v")
        if not S.value_ok(a, v) or (isinstance(v, str) and (len(v) > ATTR_TEXT_MAX or not _utf8(v))):
            return f"bad value for {attr!r}"
        who = d.get("who")
        if not who or not valid_who(who) or not who.strip():
            return "an attribute decision needs a reviewer"
        nk = attr_key(obj, attr, who)
        if parse_attr_key(nk) is None:
            return "attribute key too long"
        new[nk] = d
    dec.update(new)
    return None


def append_attr_log(edits_dir: Path, file: str, keys, merged: dict, cur=None):
    """edits/_attr_log.jsonl += {t, who, key, v, file} per attribute decision this save stamped (call under the
    package write lock). cur: the saved record before this save; a key whose merged entry equals its saved one (a
    resend of a decision the server already has) is not logged again."""
    old = (_migrate(cur) if cur else {"dec": {}})["dec"]
    rows = []
    for k in keys:
        d = merged["dec"].get(k)
        if old.get(k) == d:
            continue
        if isinstance(d, dict) and _num(d.get("server_t")):
            rows.append(json.dumps({"t": d["server_t"], "who": d.get("who"), "key": k, "v": d.get("v"),
                                    "file": file}, ensure_ascii=False))
    if rows:
        with open(Path(edits_dir) / ATTR_LOG, "a", encoding="utf-8", newline="\n") as fh:
            fh.write("\n".join(rows) + "\n")


def trusted_identity(handler):
    """The request's TRUSTED reviewer id: multi_review_server's portal identity (X-CJ-User from loopback with the
    package key in X-Review-Key; Dispatch.reviewer), else None. A standalone review_server has none: its picker
    name is never trusted for attribute decisions."""
    who = getattr(handler, "reviewer", None)
    return who if isinstance(who, str) and who else None


def package_reviewers(pkg: Path):
    """<pkg>/review_package.json reviewers (read per request: the portal hub changes them), or None when the file
    has no reviewers list."""
    try:
        rec = json.loads((Path(pkg) / RECORD).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    revs = rec.get("reviewers") if isinstance(rec, dict) else None
    return [r for r in revs if isinstance(r, str)] if isinstance(revs, list) and revs else None


def object_targets(pkg: Path):
    """{image file: {object keys of its items (and their also)}} for an object set (data.js kind "objects"), else
    None (no restriction)."""
    try:
        data = json.loads((Path(pkg) / "data.js").read_text(encoding="utf-8").split("=", 1)[1].strip().rstrip(";"))
    except (OSError, ValueError, IndexError):
        return None
    if not isinstance(data, dict) or data.get("kind") != "objects":
        return None
    by_stem = {}
    for im in data.get("images") or []:
        f = str(im.get("file") or "")
        by_stem[f] = f
        by_stem[Path(f).stem] = f
    out = {}
    for it in data.get("items") or []:
        f = by_stem.get(str(it.get("image")))
        if f is None:
            continue
        s = out.setdefault(f, set())
        s.add(str(it.get("obj")))
        s.update(str(x) for x in it.get("also") or [])
    return out


def own_attrs_only(edit: dict, who):
    """A record with only `who`'s own attribute decisions (a: / i:a: keys @who); other reviewers' are left out (rater
    independence: a reviewer never sees another's answers). who None: no attribute decisions at all."""
    dec = edit.get("dec") if isinstance(edit, dict) else None
    if not isinstance(dec, dict):
        return edit
    keep = {}
    for k, d in dec.items():
        if is_attr_key(k):
            p = parse_attr_key(k)
            if who is None or p is None or p[2] != who:
                continue
        keep[k] = d
    return {**edit, "dec": keep}


def make_handler(pkg: Path, key: str, valid_files: set, pkg_id: str = "", people: list = (), sandbox: bool = False):
    people = list(people)
    edits_dir = pkg / "edits"
    edits_dir.mkdir(exist_ok=True)
    lock = _write_lock(edits_dir)
    schema = package_schema(pkg)  # round 14: attribute decisions only for a package with a schema.json
    targets = object_targets(pkg) if schema is not None else None  # an object set: a: only on its items' objects
    roster = {p.get("id") for p in people if isinstance(p, dict) and isinstance(p.get("id"), str)}

    def active_claims(now):
        for k in [k for k, (_, exp) in claims.items() if exp < now]:
            del claims[k]
        return {k: w for k, (w, _) in claims.items()}

    class Handler(BaseHTTPRequestHandler):
        timeout = SOCKET_TIMEOUT

        def log_message(self, fmt, *args):
            if not getattr(self, "path", "").startswith("/images/"):  # no path yet when a request times out
                msg = re.sub(r"([?&]k=)[^&\s\"]*", r"\1***", fmt % args)  # never log the access key
                print(f"{self.address_string()} {msg}", flush=True)

        def _send(self, code, body=b"", ctype="application/json", extra=None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store" if ctype.startswith(("application/json", "text/html")) else "max-age=86400")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _json(self, code, obj):
            self._send(code, json.dumps(obj).encode())

        def _cookie(self):
            return cookie_name(self.server.server_address[1])

        def _authed(self, query):
            try:
                cookie = SimpleCookie(self.headers.get("Cookie", ""))
            except Exception:  # http.cookies.CookieError on a malformed header: no cookie
                cookie = {}
            given = query.get("k", [None])[0] or self.headers.get("X-Review-Key")
            if given is not None:
                return secrets.compare_digest(given, key)
            # this port's cookie, else the legacy one (set before cookies were per port) if it holds this key
            return any(name in cookie and secrets.compare_digest(cookie[name].value, key)
                       for name in (self._cookie(), COOKIE))

        def do_GET(self):
            url = urlparse(self.path)
            q = parse_qs(url.query)
            if not self._authed(q):
                return self._send(403, b"Access key required: open the full link that includes ?k=...", "text/plain")
            extra = ({"Set-Cookie": f"{self._cookie()}={key}; Path=/; HttpOnly; SameSite=Lax; Max-Age=2592000"}
                     if "k" in q else None)
            if url.path == "/api/ping":
                return self._json(200, {"ok": True, "caps": list(CAPS)})
            if url.path == "/api/people":
                return self._json(200, {"people": people})
            if url.path == "/api/claims":
                with lock:
                    return self._json(200, {"claims": active_claims(time.time())})
            if url.path == "/api/meta":
                return self._json(200, package_meta(pkg, valid_files))
            if url.path == "/api/groups":
                rg = _annotation("review_groups")
                with lock:
                    try:
                        doc = rg.public(rg.load(edits_dir, pkg_id))
                    except rg.GroupsUnreadable as e:
                        doc = {**rg.public(rg.empty(pkg_id)), "error": str(e)}
                return self._json(200, doc)
            if url.path == "/api/edits":
                out = {}
                with lock:
                    for f in _annotation("review_groups").record_files(edits_dir):
                        try:
                            rec = json.loads(f.read_text(encoding="utf-8"))
                            out[rec["file"]] = rec["edit"]
                        except (ValueError, KeyError):
                            pass
                if schema is not None:  # round 14: each reviewer gets only their own attribute decisions
                    who = trusted_identity(self)
                    out = {f: own_attrs_only(e, who) for f, e in out.items()}
                return self._json(200, out)
            rel = "index.html" if url.path in ("/", "/index.html") else url.path.lstrip("/")
            if not (rel in ("index.html", "data.js") or (rel.startswith("images/") and "/" not in rel[7:] and ".." not in rel)):
                return self._send(404, b"not found", "text/plain")
            path = pkg / rel
            if not path.is_file():
                return self._send(404, b"not found", "text/plain")
            ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            if ctype.startswith("text/") or ctype.endswith("javascript"):
                ctype += "; charset=utf-8"
            body = path.read_bytes() + (sandbox_js(sandbox) if rel == "data.js" else b"")
            self._send(200, body, ctype, extra)

        do_HEAD = do_GET

        def do_PUT(self):
            url = urlparse(self.path)
            q = parse_qs(url.query)
            if not self._authed(q):
                return self._json(403, {"error": "access key required"})
            if url.path != "/api/edits":
                return self._json(404, {"error": "not found"})
            file = q.get("f", [""])[0]
            if file not in valid_files:
                return self._json(400, {"error": "unknown image"})
            try:
                n = int(self.headers.get("Content-Length", ""))
            except ValueError:
                n = -1
            if n <= 0:
                return self._json(400, {"error": "bad Content-Length"})
            if n > MAX_BODY:
                return self._json(413, {"error": "body too large"})
            try:
                edit = json.loads(self.rfile.read(n), parse_constant=_no_constant)  # NaN / Infinity are errors
                assert valid_edit(edit)
            except (ValueError, AssertionError):
                return self._json(400, {"error": "bad edit"})
            for d in (edit.get("dec") or {}).values():  # this request's own decisions: who comes from one place
                if isinstance(d, dict) and d.get("pending") is True:
                    who = _request_identity(self, d.get("who"))
                    if who != d.get("who"):
                        d["who"] = who
            attr_new = None
            if any(is_attr_key(k) for k in edit.get("dec") or {}):  # round 14: only bodies with attributes
                pend = [k for k, d in edit["dec"].items() if is_attr_key(k) and isinstance(d, dict)
                        and d.get("pending") is True]
                if pend and schema is not None:
                    who = trusted_identity(self)
                    if who is None:
                        return self._json(403, {"error": "attribute decisions need a signed-in reviewer (open the "
                                                         "set from the portal)"})
                    revs = package_reviewers(pkg)
                    if not (who in revs if revs else who in roster):
                        return self._json(403, {"error": "you are not a reviewer of this set"})
                    if len(pend) > ATTR_PUT_MAX:
                        return self._json(400, {"error": f"at most {ATTR_PUT_MAX} attribute decisions per save"})
                why = check_attrs(edit, schema)
                if why:
                    return self._json(400, {"error": why})
                attr_new = [k for k in edit["dec"] if is_attr_key(k)]
                if targets is not None:
                    allowed = targets.get(file, set())
                    if any((parse_attr_key(k) or (None,))[0] not in allowed | {None} for k in attr_new):
                        return self._json(400, {"error": "attribute decisions only on this set's objects"})
            target = edits_dir / f"{file}.json"
            with lock:
                saved = json.loads(target.read_text(encoding="utf-8")) if target.exists() else {}
                cur = saved.get("edit")
                now = int(time.time() * 1000)
                merged = merge_edits(cur, edit, now)  # stamps server_t on the decisions this request brings
                rx = stamp_received(saved.get("rx") or {}, cur, merged, now)
                tmp = target.with_suffix(".tmp")
                tmp.write_text(json.dumps({"file": file, "pkg": pkg_id, "fmt": RECORD_FMT, "edit": merged, "rx": rx}),
                               encoding="utf-8")
                os.replace(tmp, target)
                if attr_new:
                    append_attr_log(edits_dir, file, attr_new, merged, cur)
            if schema is not None:  # rater independence (QC 8): the reply holds only the requester's own answers
                merged = own_attrs_only(merged, trusted_identity(self))
            return self._json(200, {"edit": merged})

        def do_POST(self):
            url = urlparse(self.path)
            q = parse_qs(url.query)
            if not self._authed(q):
                return self._json(403, {"error": "access key required"})
            if url.path == "/api/group":
                return self._group(q)
            if url.path != "/api/claim":
                return self._json(404, {"error": "not found"})
            case, who = q.get("c", [""])[0][:600], q.get("who", ["anonymous"])[0][:WHO_MAX] or "anonymous"
            if not valid_who(who):
                return self._json(400, {"error": "bad reviewer name"})
            who = _request_identity(self, who)
            now = time.time()
            with lock:
                active_claims(now)
                for k in [k for k, (w, _) in claims.items() if w == who]:  # one card per reviewer at a time
                    del claims[k]
                if case and (case not in claims or claims[case][0] == who):
                    claims[case] = (who, now + CLAIM_SECONDS)
                for k in sorted(claims, key=lambda k: claims[k][1])[:max(0, len(claims) - MAX_CLAIMS)]:
                    del claims[k]
                active = active_claims(now)
            return self._json(200, {"claims": active})

        def _group(self, q):
            """POST /api/group: one group action (annotation/review_groups.py), under the package write lock."""
            who = q.get("who", [""])[0][:WHO_MAX]
            if not who.strip() or not valid_who(who):
                return self._json(400, {"error": "a reviewer name is required"})
            who = _request_identity(self, who)
            try:
                n = int(self.headers.get("Content-Length", ""))
            except ValueError:
                n = -1
            if n <= 0:
                return self._json(400, {"error": "bad Content-Length"})
            if n > GROUP_BODY_MAX:
                return self._json(413, {"error": "body too large"})
            rg = _annotation("review_groups")
            try:
                body = json.loads(self.rfile.read(n), parse_constant=_no_constant)
            except ValueError:
                return self._json(400, {"error": "bad body"})
            why = rg.check_body(body, REJECT_REASONS)
            if why:
                return self._json(400, {"error": why})
            meta = package_meta(pkg, valid_files)
            if meta.get("package", {}).get("kind") != "prediction":  # samples are a prediction package's (the page agrees)
                return self._json(404, {"error": "group actions are for prediction packages"})
            images = meta["images"]
            with lock:
                code, ans = rg.act(sys.modules[__name__], edits_dir, pkg_id, images, body, who,
                                   claims=active_claims(time.time()))
            return self._json(code, ans)

    claims = {}  # "<file>|<case key>" -> (reviewer, expiry): the card on each reviewer's screen

    return Handler


def _request_identity(handler, claimed):
    """THE ONE PLACE a request's reviewer comes from: the who saved on this request's new decisions (pending ones)
    and on its claim.

    Here: always `claimed`, the name the page's picker sent (a decision's who, a claim's ?who=), unverified; this
    server (the real review on :8766, the sandbox) reads no identity header and keeps the name picker.
    Portal identity lives in annotation/multi_review_server.py (prediction packages, BUILD_CONTRACTS.md section 8):
    it trusts the portal's X-CJ-User only from loopback with the package key in X-Review-Key, and stamps the request's
    decisions and claim with that id before this handler sees them, so `claimed` already is that id there."""
    return claimed


def _no_constant(name):
    raise ValueError(f"{name} is not allowed")


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _valid_rings(p):
    """A list of flat rings [x1, y1, x2, y2, ...] of >= 3 points, each coordinate in [0, 1]."""
    return isinstance(p, list) and len(p) > 0 and all(
        isinstance(r, list) and len(r) >= 6 and len(r) % 2 == 0
        and all(_num(v) and -COORD_EPS <= v <= 1 + COORD_EPS for v in r) for r in p)


def _valid_item(g):
    """One polygon-list item, shaped like a data.js label: {id, c, p, src?, t?} (+ flags such as u, x, lc)."""
    return (isinstance(g, dict) and isinstance(g.get("c"), str) and _valid_rings(g.get("p"))
            and (g.get("id") is None or (isinstance(g["id"], int) and not isinstance(g["id"], bool)))
            and (g.get("src") is None or (isinstance(g["src"], str) and g["src"][:1] in ("p", "n")))
            and (g.get("t") is None or _num(g["t"])))


def valid_who(who) -> bool:
    """A reviewer name or roster id: letters, digits, space, '.', '_', '-' (no markup), at most WHO_MAX long."""
    return isinstance(who, str) and WHO_RE.fullmatch(who) is not None


def valid_edit(edit) -> bool:
    """Shape check for a PUT body: a dict with a numeric t, decisions under known keys, optional polygon list."""
    if not isinstance(edit, dict) or not _num(edit.get("t", 0)):
        return False
    for f in ("gtT", "reviewedT"):
        if edit.get(f) is not None and not _num(edit[f]):
            return False
    if not isinstance(edit.get("dec") or {}, dict) or not isinstance(edit.get("gt") or [], list):
        return False
    if not all(_valid_item(g) for g in edit.get("gt") or []):
        return False
    for k, d in (edit.get("dec") or {}).items():
        if is_attr_key(k):  # round 14: a:<obj>:<attr>@<rid> / i:a:<attr>@<rid>; the schema check is check_attrs
            if not _valid_attr_dec(k, d):
                return False
            continue
        if k in IMAGE_KEYS or (len(k) <= STATE_KEY_MAX and TAG_KEY_RE.fullmatch(k)):  # a longer t: key: refused below
            if not _valid_state_dec(k, d):
                return False
            continue
        if not k or k[0] not in DEC_PREFIXES:
            return False
        if isinstance(d, str):  # the oldest records stored bare strings
            continue
        if not isinstance(d, dict) or not isinstance(d.get("v"), (str, type(None))) or not _num(d.get("t", 0)):
            return False
        if d.get("p") is not None and not _valid_rings(d["p"]):
            return False
        if d.get("who") is not None and not valid_who(d["who"]):
            return False
        if ("server_t" in d and not _num(d["server_t"])) or ("pending" in d and not isinstance(d["pending"], bool)):
            return False
        if (d.get("v") or "").startswith("new:") and d.get("p") is None:  # a drawn outline carries its polygon
            return False
    return True


def _utf8(s) -> bool:
    """A string that encodes as UTF-8: JSON can carry lone surrogates ("\\ud800"), which would crash every later
    writer of the package (apply_review_edits, the results CSVs)."""
    try:
        s.encode("utf-8")
        return True
    except UnicodeEncodeError:
        return False


def _valid_state_dec(k, d) -> bool:
    """An i:reject / i:note / t:<objkey> decision: a dict (never a bare string) with only STATE_FIELDS (i:reject also
    grp: the group action that set it, a UTF-8 string within GRP_MAX), v from its own set, notes UTF-8 strings within
    their caps."""
    if not isinstance(d, dict) or not set(d) <= (REJECT_FIELDS if k == "i:reject" else STATE_FIELDS) \
            or not _num(d.get("t", 0)):
        return False
    if "grp" in d and not (isinstance(d["grp"], str) and 0 < len(d["grp"]) <= GRP_MAX and _utf8(d["grp"])):
        return False
    if any(isinstance(d.get(f), str) and not _utf8(d[f]) for f in ("v", "note")):
        return False
    if d.get("who") is not None and not valid_who(d["who"]):
        return False
    if ("server_t" in d and not _num(d["server_t"])) or ("pending" in d and not isinstance(d["pending"], bool)):
        return False
    v = d.get("v")
    if k == "i:reject":
        return "note" not in d and (v is None or (isinstance(v, str) and v.startswith(REJECT_PREFIX)
                                                   and v[len(REJECT_PREFIX):] in REJECT_REASONS))
    if k == "i:note":
        return "note" not in d and (v is None or (isinstance(v, str) and len(v) <= IMAGE_NOTE_MAX))
    note = d.get("note")
    return (v is None or v in TAG_VALUES) and (note is None or (isinstance(note, str) and len(note) <= TAG_NOTE_MAX))


def _valid_attr_dec(k, d) -> bool:
    """An attribute decision's shape (round 14): a known key form within ATTR_KEY_MAX, a dict with only ATTR_FIELDS,
    v null / true / false / a UTF-8 string within ATTR_TEXT_MAX."""
    p = parse_attr_key(k)
    if p is None or not valid_who(p[2]) or not p[2].strip():
        return False
    if not isinstance(d, dict) or not set(d) <= ATTR_FIELDS or not _num(d.get("t", 0)):
        return False
    v = d.get("v")
    if not (v is None or isinstance(v, bool) or (isinstance(v, str) and len(v) <= ATTR_TEXT_MAX and _utf8(v))):
        return False
    if d.get("who") is not None and not valid_who(d["who"]):
        return False
    return not (("server_t" in d and not _num(d["server_t"])) or ("pending" in d and not isinstance(d["pending"], bool)))


def _migrate(e):
    """Older records stored decisions as bare strings and polygon lists without a timestamp."""
    e = dict(e or {})
    e["dec"] = {k: (v if isinstance(v, dict) else {"v": v, "t": e.get("t", 0), "who": ""})
                for k, v in (e.get("dec") or {}).items()}
    if e.get("gt") is not None and not e.get("gtT"):
        if e.get("modified"):
            e["gtT"] = e.get("t", 0) or 1
        else:
            e.pop("gt", None)
    return e


SERVER_SET = ("server_t", "pending", "who")  # fields the server sets or may rewrite on receipt (see _same)


def _rank(d) -> float:
    """A decision's place in the merge order: server_t, the server's receive time (ms). An entry saved before
    server_t existed has only the client's t, which stands in for it (a missing server_t counts as t)."""
    st = d.get("server_t")
    return st if _num(st) else d.get("t", 0)


def _stamped(d) -> bool:
    """Went through the server-time merge (server_t), or is this browser's newest, not sent yet (pending)."""
    return bool(d) and (_num(d.get("server_t")) or d.get("pending") is True)


def _same(x, y) -> bool:
    """One decision, whether or not the server has stamped it yet: equal apart from server_t, pending and who
    (who: a later trusted identity may replace the picker's name, _request_identity)."""
    return ({k: v for k, v in x.items() if k not in SERVER_SET} == {k: v for k, v in y.items() if k not in SERVER_SET})


def _pick(x, y):
    """The entry one decision key keeps (x: a's, the saved copy; y: b's). Mirrors pickDec() in the page.
      same decision           -> the stamped copy (an acknowledged upload replaces the pending one), else x
      one side pending        -> the pending one: the page's newest change, received when it is sent
      both pending            -> the newer client t (two copies of one browser's work, e.g. an import)
      otherwise               -> the larger rank (server_t, else t); a tie keeps x.
    With no server_t and no pending anywhere (records saved before server_t) this is the old rule exactly:
    the newer t wins, a tie keeps the saved copy."""
    if x is None or y is None:
        return x if y is None else y
    if _same(x, y):
        return y if _num(y.get("server_t")) and not _num(x.get("server_t")) else x
    px, py = x.get("pending") is True, y.get("pending") is True
    if px != py:
        return y if py else x
    if px:
        return y if y.get("t", 0) > x.get("t", 0) else x
    return y if _rank(y) > _rank(x) else x


def merge_edits(a, b, now_ms=None):
    """Merge two reviewers' records for one image: each decision key keeps its newest entry by server receive
    time (_pick), the polygon list keeps the newest Editor save (gtT), the reviewed flag keeps its newest toggle
    (reviewedT); gtT and reviewedT are still client times. Mirrors mergeEdit() in review_tool/index.html.

    now_ms: the server's receive time, for a PUT (b = the request's record). b's pending entries that win are
    stamped server_t = max(now_ms, rank of the entry they replace + 1) and lose the pending flag, so a new decision
    always sorts above what it replaced, even an old entry whose t ran ahead. Entries without pending are never
    stamped, so records from before server_t (and an old cached page's PUTs) merge byte-identical to before."""
    if not a:
        out = _migrate(b)
    else:
        a, b = _migrate(a), _migrate(b)
        out = {"dec": {}, "t": max(a.get("t", 0), b.get("t", 0)), "modified": bool(a.get("modified") or b.get("modified"))}
        for k in dict.fromkeys([*a["dec"], *b["dec"]]):  # a's keys, then b's new ones: the same order as mergeEdit()
            out["dec"][k] = _pick(a["dec"].get(k), b["dec"].get(k))
        if a.get("gtT") or b.get("gtT"):
            s = b if (b.get("gtT") or 0) > (a.get("gtT") or 0) else a
            out["gt"], out["gtT"] = s.get("gt"), s.get("gtT")
        ra, rb = a.get("reviewedT", 0) or 0, b.get("reviewedT", 0) or 0
        out["reviewed"] = bool(b.get("reviewed")) if rb > ra else bool(a.get("reviewed"))
        out["reviewedT"] = max(ra, rb)
    if now_ms is not None:
        old = (a or {}).get("dec") or {}
        for k, d in out["dec"].items():
            if d.get("pending") is True:
                prev = old.get(k)
                st = max(now_ms, _rank(prev) + 1) if prev else now_ms
                out["dec"][k] = {**{f: v for f, v in d.items() if f != "pending"}, "server_t": st}
    return out


def stamp_received(rx: dict, cur, merged: dict, now_ms: int) -> dict:
    """Server receive time (ms) per decision key, plus "gt" for the polygon list: set when a save changes that
    entry. The record's own t values are the tool's logical clock (never earlier than the entry they replace),
    so this is the wall time a change reached the server."""
    rx = dict(rx)
    old = _migrate(cur) if cur else {"dec": {}}
    for k, d in merged["dec"].items():
        if old["dec"].get(k) != d:
            rx[k] = now_ms
    if merged.get("gtT") and merged.get("gtT") != old.get("gtT"):
        rx["gt"] = now_ms
    return rx


def load_people(path: Path = PEOPLE) -> list:
    """The reviewer roster: [{id, name, aliases}], for the tool's name picker and the change log's who."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))["people"]
    except (OSError, ValueError, KeyError):
        return []


def _dec_v(e, key):
    d = e["dec"].get(key)
    return d.get("v") if d else None


def effective_labels(image: dict, edit) -> list:
    """One image's labels after its edit record, as the review tool shows them (effGt() in index.html).

    image: a data.js images[] entry (gt: [{id, c, p}], pr: [{s, p}]); edit: a saved record, any version.
    Returns [{id, c, p, src, ...}]: id is the COCO annotation id (None for added objects), c the class,
    p the outline (list of normalized flat rings), src "p<k>" (accepted prediction) or "n<id>" (drawn); low_conf
    (true) and note only when the object's t:<key> decision sets them (tag_of; the class decision is never touched).
    Base = the record's polygon list if it has one (gtT), else the package labels; decisions apply on top.
    A list item with its own t (saved by the current tool) takes its decision whenever the decision is at
    least as new as the item, so per-object edits survive anyone's newer list; an item without t (older
    lists) takes a decision's polygon only if the decision is newer than the whole list (gtT). A decision with
    server_t (or pending) always counts as newer: it won the server-time merge, and every Editor change writes its
    object's decision, so a list item with a larger client t only means that device's clock ran ahead."""
    e = _migrate(edit)
    gt_t = e.get("gtT") or 0
    base = e["gt"] if e.get("gtT") and e.get("gt") is not None else image["gt"]

    def dec_p(key):
        d = e["dec"].get(key)
        return d["p"] if d and d.get("p") and (d.get("t", 0) > gt_t or _stamped(d)) else None

    out = []
    for g in base:
        gkey = f"g{g['id']}" if g.get("id") is not None else None
        if _num(g.get("t")):
            d = e["dec"].get(g.get("src") or gkey) if (g.get("src") or gkey) else None
            if not d or (d.get("t", 0) < g["t"] and not _stamped(d)):
                out.append(g)
                continue
            v = d.get("v")
            if v in ("del", "no") or (not v and g.get("src")):
                continue
            if not v:  # undone: the package label
                o = next((x for x in image["gt"] if x["id"] == g.get("id")), None)
                out.append({**g, "c": o["c"], "p": o["p"]} if o else g)
                continue
            m = re.match(r"^(?:keep|add|new):(.*)$", v)
            out.append({**g, "c": m.group(1), "p": d.get("p") or g["p"]} if m else g)
            continue
        v = _dec_v(e, gkey) if gkey else None
        sv = _dec_v(e, g["src"]) if g.get("src") else None
        if v == "del" or sv in ("no", "del"):
            continue
        cls = v[5:] if v and v.startswith("keep:") else sv[4:] if sv and sv.startswith(("add:", "new:")) else g["c"]
        poly = (gkey and v and v.startswith("keep:") and dec_p(gkey)) or (g.get("src") and dec_p(g["src"])) or None
        out.append({**g, "c": cls, "p": poly or g["p"]})
    for key, d in e["dec"].items():
        v = d.get("v")
        if not v or any(o.get("src") == key for o in out):
            continue
        if key[0] == "p" and v.startswith("add:"):
            try:
                k = int(key[1:])
                if k < 0:  # "p-1" is out of range, as in effGt(); never Python's last-item index
                    continue
                pr = image["pr"][k]
            except (ValueError, IndexError):
                continue
            out.append({"id": None, "c": v[4:], "p": json.loads(json.dumps(d.get("p") or pr["p"])), "src": key})
        elif key[0] == "n" and v.startswith("new:") and d.get("p"):
            out.append({"id": None, "c": v[4:], "p": json.loads(json.dumps(d["p"])), "src": key})
    return _with_tags(out, e["dec"])


def object_key(g):
    """An object's decision key, as keyOf() in the page: its src (p<k> / n<uid>), else g<label id>, else None."""
    return g.get("src") or (f"g{g['id']}" if g.get("id") is not None else None)


def tag_of(dec: dict, key) -> tuple:
    """(v, note) of the t:<key> decision in a migrated record's dec: v "lowconf" | "plain" | None; note only while v
    is set (an undone tag carries nothing)."""
    d = dec.get(f"t:{key}") if key else None
    v = d.get("v") if isinstance(d, dict) else None
    note = d.get("note") if v and isinstance(d.get("note"), str) and d["note"].strip() else None
    return v, note


def is_low_conf(cls, tag_v) -> bool:
    """Low conf: the tag is set, or the class is Low conf EV (whatever its tag says: turning the tag off on such a
    label makes it EV, keep:EV, so a Low conf EV label with a note-only "plain" tag is still low conf)."""
    return tag_v == "lowconf" or cls == LOW_CONF_EV


def _with_tags(out: list, dec: dict) -> list:
    """effective_labels' objects with their t:<objkey> tag passed through: low_conf: true (the tag is set) and note,
    each only when set. The same list, untouched, when the record has no tags (outputs as before)."""
    if not any(k.startswith("t:") for k in dec) and not any("low_conf" in o or "note" in o for o in out):
        return out
    res = []
    for o in out:
        v, note = tag_of(dec, object_key(o))
        o = {k: x for k, x in o.items() if k not in ("low_conf", "note")}  # a list item never carries a stale copy
        if v == "lowconf":
            o["low_conf"] = True
        if note:
            o["note"] = note
        res.append(o)
    return res


def image_flags(edit) -> dict:
    """A record's image-level state, each key only when set: rejected (True) + reject_reason from i:reject, note
    from i:note; reject_scope ("sample" | "montage") + reject_group (its id) when a group action set the reject
    (round 12, annotation/review_groups.py). {} for a record without them (every record saved before round 7)."""
    dec = _migrate(edit)["dec"] if edit else {}
    out = {}
    v = (dec.get("i:reject") or {}).get("v")
    if isinstance(v, str) and v.startswith(REJECT_PREFIX):
        out["rejected"], out["reject_reason"] = True, v[len(REJECT_PREFIX):]
        grp = dec["i:reject"].get("grp")
        kind = None
        if isinstance(grp, str) and grp.startswith('["'):
            try:
                kind = json.loads(grp)[0]
            except (ValueError, IndexError, KeyError, TypeError):
                kind = None
        if kind in ("sample", "montage"):
            out["reject_scope"], out["reject_group"] = kind, grp
    n = (dec.get("i:note") or {}).get("v")
    if isinstance(n, str) and n.strip():
        out["note"] = n
    return out


def tailscale_hosts():
    """This PC's Tailscale 100.x addresses and MagicDNS name, if signed in (as in web_qc/server.py)."""
    for exe in ("tailscale", r"C:\Program Files\Tailscale\tailscale.exe"):
        try:
            r = subprocess.run([exe, "status", "--json"], capture_output=True, text=True, timeout=3)
            me = json.loads(r.stdout)["Self"]
        except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError):
            continue
        hosts = [ip for ip in me.get("TailscaleIPs") or [] if ip.startswith("100.")]
        return hosts + ([me["DNSName"].rstrip(".")] if me.get("DNSName") else [])
    return []


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--package", required=True, type=Path, help="Package folder holding index.html, data.js, images/")
    ap.add_argument("--host", default="127.0.0.1",
                    help="Bind address (default: 127.0.0.1, for this PC, a tunnel or `tailscale serve`). For phones "
                         "on the tailnet pass this PC's Tailscale IP (`tailscale ip -4`), not 0.0.0.0")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--key", default=None, help="Access key (default: reuse <package>/.review_key, else generate one)")
    ap.add_argument("--sandbox", action="store_true",
                    help=f"Show the SANDBOX bar on every page (also on when the package folder starts with {SANDBOX_PREFIX})")
    args = ap.parse_args()

    pkg = args.package.resolve()
    if not hold_package(pkg):  # held until this process ends
        print(f"[ERROR] {pkg.name} is open in another review server (lock: {lock_path(pkg)}). Two servers on one "
              "package lose saves: stop the other one first.", flush=True)
        raise SystemExit(2)
    data = json.loads((pkg / "data.js").read_text(encoding="utf-8").split("=", 1)[1].strip().rstrip(";"))
    valid = {im["file"] for im in data["images"]}
    key_file = pkg / ".review_key"
    key = args.key or (key_file.read_text().strip() if key_file.exists() else secrets.token_urlsafe(18))
    key_file.write_text(key)

    sandbox = is_sandbox(pkg, args.sandbox)
    srv = ThreadingHTTPServer((args.host, args.port),
                              make_handler(pkg, key, valid, data.get("id", ""), load_people(), sandbox))
    print(f"[OK] Serving {pkg.name}: {len(valid)} images, edits -> {pkg / 'edits'}", flush=True)
    if sandbox:
        print("[OK] Sandbox: every page shows the SANDBOX bar (edits here don't count toward the real review)", flush=True)
    # The key is printed once, in this link; request logs redact it. It is also in <package>/.review_key.
    shown = "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host
    print(f"[OK] Link: http://{shown}:{args.port}/?k={key}", flush=True)
    if args.host in ("0.0.0.0", "::"):
        ts = ", ".join(h for h in tailscale_hosts() if h[:1].isdigit()) or "see `tailscale ip -4`"
        print(f"[WARN] Listening on every interface, LAN included. Prefer --host <Tailscale IP> ({ts}).", flush=True)
    elif args.host in ("127.0.0.1", "localhost"):
        print(f"[OK] Phone over Tailscale: `tailscale serve --bg {args.port}`, or restart with --host <`tailscale ip -4`>",
              flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
