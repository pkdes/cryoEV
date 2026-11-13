"""
One process serving many review packages (BUILD_CONTRACTS.md section 3): GET/PUT/POST /<pkg id>/<path> is handled
by review_server.make_handler for that package, exactly as a single review_server.py would handle /<path>.
review_server.py is wrapped, not changed; the real review (review_server.py on :8766) is a separate process, kept
off by the exclude (below, fail-closed) and by the package lock.

  python -m annotation.multi_review_server [--root <CryoAI-laptop>/review] [--port 8767] [--host 127.0.0.1]
                                           [--prefix pred_ [--prefix lab_]] [--exclude <pkg id or folder> ...]

- Packages: complete folders <root>/<prefix>*/ (data.js + review_package.json), reached under exactly their own
  name (no case or short-name aliases). Default prefix pred_ (prediction packages). lab_ (lab / compare packages,
  compare_package.py) only when asked: --prefix (repeatable), else $CJ_REVIEW_PREFIXES (comma-separated); the
  portal reads the same env var (serving()), so its picker and /review/<id>/ follow what this server serves. A
  package added after start is found on its first request; a rebuilt data.js or a new .review_key is picked up on
  the next one.
- One writer per package (round 9): before anything is made for a package (after its key is checked), this process
  takes the package's OS lock (review_server.hold_package: <root>/.review_locks/<pkg>.lock) and keeps it until it
  exits: a package once opened here stays locked until this server is restarted. A package whose lock another
  process holds (a single review_server.py, another multi server) is refused with 409, on every request, until that
  process lets go. review_server.py takes the same lock at start. (Operations: a preview multi server pointed at the
  real review root would hold real packages' locks; give it a sandbox root.)
- Never served (serving(); one source for this server, the portal's picker and its /review/<id>/ lookup):
    * the --package of every annotation/review_server.py service in the services config (tools/services.py's file:
      $CRYOEV_SERVICES_CONFIG, else $CRYOEV_SERVICES, else ~/.cryoev/services.json; read only), enabled or not;
    * --exclude (repeatable) and $CJ_REVIEW_EXCLUDE (comma-separated), by id or folder path.
  Fail-closed: a services config that exists but can't be read (bad JSON, a review_server.py service without
  --package) means no lab_ package is served at all. A server started before the lock existed holds no lock, so
  the exclude is what keeps this server off its package. An excluded id gets the unknown-id answer (403); main()
  warns about an excluded name that is no folder under --root.
- Auth: each package keeps its own key (<pkg>/.review_key; no key, not served): ?k=<key> once, then a cookie
  review_key_<port>_<pkg id> (one per package, since every package shares this port), or X-Review-Key. The key is
  checked here before anything is made for the package (handler, cache entry, edits/ folder).
- Errors: an unknown id and a wrong key both get 403, so ids can't be enumerated; a malformed id or a path with
  .., \\ or : -> 400; / -> 404 (no listing). /<id> (with its key) redirects to /<id>/ so the page's relative URLs
  (data.js, images/, api/...) stay inside the package.
- Portal identity (BUILD_CONTRACTS.md section 8): the CryoEV portal proxies /review/<id>/app/... here with the package's key
  as X-Review-Key and the signed-in user's id as X-CJ-User (portal/proxy.py drops any client-sent X-CJ-* first). That
  id is trusted ONLY when the same request came from loopback (the portal runs on this PC) AND brought this package's
  key in X-Review-Key (no ?k=, no cookie). Then:
    GET /<id>/            the repo's CURRENT annotation/review_tool/index.html (the package's own copy only when that
                          is missing) with window.REVIEW_IDENTITY = {"who": <id>, "locked": true} put before its first
                          <script>: the page skips the "Who is reviewing?" picker. Not the package copy: a package
                          built before LOCKED_WHO would ask for a name, and with its claims rewritten (below) to an
                          id the page doesn't know, card mode would skip the user's own cards. Files on disk unchanged.
    PUT /<id>/api/edits   every decision in the body gets who = <id> (a client-sent who never counts) before
                          review_server merges it, so reviewed_by (prediction_package) names the portal user
    POST /<id>/api/claim  ?who= = <id>
  Without a trusted identity, every request is handled exactly as before: direct :8767/<id>/?k= links get the
  package's own index.html and the page asks for a name as it always did.
  So a package's data.js is frozen at build while the tool page is the repo's: later changes to review_tool/index.html
  must keep reading old data.js files (add fields only; never rename, remove or re-mean one).
- Capabilities: the page is read from disk on every request, but review_server (valid_edit) is the code this process
  started with. GET /<id>/api/ping answers review_server's caps (["i", "t"]: image reject / note, object low-conf tag);
  a page served by a process started on older code gets {"ok": true} without caps and hides those controls, so it
  never sends keys that process would refuse with a 400. Restart this server when review_server.py changes.
"""
import argparse
import io
import json
import os
import re
import secrets
import sys
import threading
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, parse_qsl, urlencode, urlsplit

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from annotation import review_server as rs  # noqa: E402

PORT = 8767
PREFIX = ("pred_",)               # prediction packages; lab_ (compare_package.py) only by --prefix / PREFIX_ENV
LAB_PREFIX = "lab_"
PREFIX_ENV = "CJ_REVIEW_PREFIXES"   # comma-separated prefixes to serve (the portal reads it too)
EXCLUDE_ENV = "CJ_REVIEW_EXCLUDE"   # comma-separated package ids or folders this server never serves
SERVICES_ENVS = ("CRYOEV_SERVICES_CONFIG", "CRYOEV_SERVICES")       # tools/services.py's config, read only
SERVICES_DEFAULT = Path.home() / ".cryoev" / "services.json"
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
BAD_PATH = ("..", "\\", ":")
COMPLETE = "review_package.json"  # prediction_package.build writes it last (and builds in a temp folder anyway)
IDENTITY_HEADER = "X-CJ-User"     # set by portal/proxy.py (client copies stripped); trusted only with the key header
LOCAL_CLIENTS = frozenset({"127.0.0.1", "::1", "::ffff:127.0.0.1"})  # the portal's address: the only identity source
TOOL_PAGE = REPO / "annotation" / "review_tool" / "index.html"     # served (with the identity) to portal requests


def cookie_name(port: int, pkg_id: str) -> str:
    """review_key_<port>_<pkg id>: every package shares this port, so each needs its own cookie."""
    return f"{rs.cookie_name(port)}_{pkg_id}"


def _prefixes(prefix) -> tuple:
    return (prefix,) if isinstance(prefix, str) else tuple(prefix)


def package_dir(root: Path, pkg_id: str, prefix=PREFIX) -> Path | None:
    """The resolved <root>/<pkg_id> when pkg_id is a well-formed id with a prefix (one, or a tuple of them), the
    folder is complete (data.js and review_package.json, which the builder writes last) and it is reached under
    exactly that name: on a case-insensitive disk /PRED_x/ or an 8.3 short name would otherwise be a second package
    with its own lock."""
    if not (ID_RE.match(pkg_id or "") and ".." not in pkg_id and pkg_id.startswith(_prefixes(prefix))):
        return None
    root = Path(root).resolve()
    p = (root / pkg_id).resolve()
    if p.parent != root or p.name != pkg_id:
        return None
    return p if (p / "data.js").is_file() and (p / COMPLETE).is_file() else None


def _name(v: str) -> str:
    """A package id or folder path -> its lower-case folder name (either slash)."""
    return re.split(r"[\\/]+", v.strip().rstrip("\\/"))[-1].lower()


def served_prefixes(environ=None) -> tuple:
    """The prefixes served without --prefix: $CJ_REVIEW_PREFIXES, else PREFIX (pred_ only)."""
    env = os.environ if environ is None else environ
    got = tuple(dict.fromkeys(v.strip() for v in str(env.get(PREFIX_ENV, "")).split(",") if v.strip()))
    return got or PREFIX


def services_config(environ=None) -> Path:
    env = os.environ if environ is None else environ
    for k in SERVICES_ENVS:
        if env.get(k):
            return Path(os.path.expanduser(env[k]))
    return SERVICES_DEFAULT


_svc_cache: dict = {}


def single_server_packages(environ=None) -> tuple:
    """(lower-case folder names, ok): the --package of every annotation/review_server.py service in the services
    config, enabled or not (read only). ok is False when the file exists but can't be read, or such a service has no
    --package: then nothing is known, and serving() serves no lab_ package (fail-closed). No file: (empty, True)."""
    path = services_config(environ)
    try:
        st = path.stat()
    except FileNotFoundError:
        return frozenset(), True
    except OSError:
        return frozenset(), False
    key = (str(path), st.st_mtime_ns, st.st_size)
    if _svc_cache.get("key") == key:
        return _svc_cache["val"]
    val = _parse_services(path)
    _svc_cache.update(key=key, val=val)
    return val


def _parse_services(path: Path) -> tuple:
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode("utf-8-sig")
        data = json.loads(text)
        services = data.get("services") or []
        if not isinstance(services, list):
            return frozenset(), False
    except (OSError, ValueError, UnicodeError, AttributeError):
        return frozenset(), False
    names = set()
    for svc in services:
        argv = svc.get("argv") if isinstance(svc, dict) else None
        if not isinstance(argv, list) or not all(isinstance(a, str) for a in argv):
            return frozenset(), False
        is_single = any(re.split(r"[\\/]+", a)[-1].lower() == "review_server.py" for a in argv) or any(
            a == "-m" and b == "annotation.review_server" for a, b in zip(argv, argv[1:]))
        if not is_single:
            continue
        pkg = next((b for a, b in zip(argv, argv[1:]) if a == "--package"), None) or next(
            (a.split("=", 1)[1] for a in argv if a.startswith("--package=")), None)
        if not pkg or not pkg.strip():
            return frozenset(), False
        names.add(_name(os.path.expanduser(os.path.expandvars(pkg.replace("{repo}", str(REPO))))))
    return frozenset(names), True


def excluded_names(values=(), environ=None) -> frozenset:
    """Lower-case folder names never to serve: `values` (--exclude), $CJ_REVIEW_EXCLUDE and the single-package
    review servers' packages from the services config, each an id or a path."""
    env = os.environ if environ is None else environ
    items = list(values or ()) + str(env.get(EXCLUDE_ENV, "")).split(",")
    return frozenset(_name(v) for v in items if v and v.strip()) | single_server_packages(environ)[0]


def serving(prefix=None, exclude=(), environ=None) -> tuple:
    """(prefixes, excluded names) this server serves, as the portal must see them too: prefix (else
    served_prefixes()), minus lab_ when the services config can't be read (fail-closed); excluded_names(exclude)."""
    pre = _prefixes(prefix) if prefix else served_prefixes(environ)
    if not single_server_packages(environ)[1]:
        pre = tuple(p for p in pre if not p.startswith(LAB_PREFIX))
    return pre, excluded_names(exclude, environ)


def package_key(pkg: Path) -> str:
    """The package's access key from <pkg>/.review_key, "" when there is none (then the package is not served)."""
    f = Path(pkg) / ".review_key"
    try:
        return f.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def identity_script(who: str) -> bytes:
    """The inline script that tells the review tool who the (portal) reviewer is; <, > and & escaped for HTML."""
    js = json.dumps({"who": who, "locked": True}).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    return f"<script>window.REVIEW_IDENTITY = {js};</script>\n".encode()


def inject_identity(page: bytes, who: str) -> bytes:
    """index.html with identity_script(who) before its first <script> (so before data.js and the tool), else before
    </head>, else at the start."""
    low = page.lower()
    at = low.find(b"<script")
    if at < 0:
        at = max(low.find(b"</head>"), 0)
    return page[:at] + identity_script(who) + page[at:]


def stamp_identity(edit, who: str):
    """A PUT body with every decision's who set to `who` (a bare-string decision becomes {v, t, who}, as
    review_server._migrate would read it). Echoed copies of other reviewers' saved decisions keep their saved who:
    review_server._same ignores who, so the saved entry wins the merge. Anything not shaped like an edit is returned
    unchanged (review_server rejects it)."""
    if not isinstance(edit, dict) or not isinstance(edit.get("dec"), dict):
        return edit
    for k, d in list(edit["dec"].items()):
        if isinstance(d, dict):
            d["who"] = who
        elif isinstance(d, str):
            edit["dec"][k] = {"v": d, "t": edit.get("t", 0), "who": who}
    return edit


LOCKED = object()  # Packages.handler: the key is right, but another process holds the package's lock


class Packages:
    """pkg id -> handler class for that package, made on the first request with its key and remade when data.js or
    .review_key change. Cached by resolved folder; nothing is cached or created for an unknown id, an excluded one, a
    wrong key, or a package whose lock another process holds."""

    def __init__(self, root: Path, prefix=None, people=None, exclude=()):
        self.root = Path(root)
        self.prefix, self.exclude = serving(prefix, exclude)  # the services config's packages always count
        self.people = rs.load_people() if people is None else list(people)
        self._cache, self._lock, self._held = {}, threading.Lock(), set()

    def handler(self, pkg_id: str, given):
        """The package's handler class when `given` is its key; LOCKED when it is, but another process writes the
        package; else None (unknown or excluded id and wrong key alike)."""
        pkg = package_dir(self.root, pkg_id, self.prefix)
        if pkg is not None and pkg.name.lower() in self.exclude:
            pkg = None
        key = package_key(pkg) if pkg else ""
        # bytes: compare_digest refuses non-ASCII str, and a ?k= can hold anything
        ok = secrets.compare_digest(str(given or "").encode(), (key or "\0no package\0").encode()) and bool(key)
        if not ok:
            return None
        with self._lock:
            if pkg not in self._held:  # the package's one writer: this process, or nobody here
                if not rs.hold_package(pkg):
                    return LOCKED
                self._held.add(pkg)
            try:
                sig = ((pkg / "data.js").stat().st_mtime_ns, (pkg / ".review_key").stat().st_mtime_ns)
            except OSError:
                return None
            hit = self._cache.get(pkg)
            if hit and hit[0] == sig:
                return hit[1]
            data = json.loads((pkg / "data.js").read_text(encoding="utf-8").split("=", 1)[1].strip().rstrip(";"))
            inner = rs.make_handler(pkg, key, {im["file"] for im in data["images"]}, data.get("id", pkg_id),
                                    self.people, rs.is_sandbox(pkg))

            def _cookie(h, _id=pkg_id):  # one cookie per package: they all share this port
                return cookie_name(h.server.server_address[1], _id)

            cls = type(f"Pkg_{pkg_id}", (Dispatch, inner), {"_cookie": _cookie, "pkg_id": pkg_id, "packages": self,
                                                            "pkg_dir": pkg, "log_message": inner.log_message})
            self._cache[pkg] = (sig, cls)
            return cls

    def ids(self):
        return sorted({p.name for pre in self.prefix for p in self.root.glob(f"{pre}*")
                       if package_dir(self.root, p.name, self.prefix) and p.name.lower() not in self.exclude})

    def close(self):
        """Let go of every package lock this instance holds (the server is closing)."""
        with self._lock:
            for pkg in self._held:
                rs.release_package(pkg)
            self._held.clear()
            self._cache.clear()


class Dispatch(BaseHTTPRequestHandler):
    """Reads the request line, picks the package from the first path segment, then hands the request to that
    package's review_server handler: self.path loses the /<id> prefix and the instance takes the package's class
    (it stays first in the MRO, so a kept-alive connection is dispatched again on its next request)."""
    timeout = rs.SOCKET_TIMEOUT
    packages: Packages = None
    pkg_dir: Path = None
    reviewer = None  # the portal user this request is for (_portal_identity), else None: the page's own claims, as before

    def _given_key(self, query, pkg_id):
        """The key this request brings for pkg_id: ?k=, else X-Review-Key, else this package's cookie."""
        given = parse_qs(query).get("k", [None])[0] or self.headers.get("X-Review-Key")
        if given is not None:
            return given
        try:
            cookie = SimpleCookie(self.headers.get("Cookie", ""))
        except Exception:  # a malformed header: no cookie
            return None
        name = cookie_name(self.server.server_address[1], pkg_id)
        return cookie[name].value if name in cookie else None

    def _plain(self, code, text, extra=None, ctype="text/plain; charset=utf-8"):
        body = text.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)
        self.close_connection = True

    def parse_request(self):
        if not super().parse_request():
            return False
        url = urlsplit(self.path)
        parts = url.path.split("/", 2)  # "", <id>, <rest>
        pkg_id = parts[1] if len(parts) > 1 else ""
        if not pkg_id:
            self._plain(404, "not found")
            return False
        if not ID_RE.match(pkg_id) or ".." in pkg_id or any(b in url.path for b in BAD_PATH):
            self._plain(400, "bad path")
            return False
        cls = self.packages.handler(pkg_id, self._given_key(url.query, pkg_id))
        if cls is LOCKED:  # the key was right: say why, so nobody retries into a second writer
            msg = "This package is open in another review server: only one may save to it at a time."
            if self.command in ("GET", "HEAD"):
                self._plain(409, msg)
            else:
                self._plain(409, json.dumps({"error": msg}), ctype="application/json")
            return False
        if cls is None:  # unknown id or wrong key: one answer, as review_server gives it, so ids can't be listed
            if self.command in ("GET", "HEAD"):
                self._plain(403, "Access key required: open the full link that includes ?k=...")
            else:
                self._plain(403, json.dumps({"error": "access key required"}), ctype="application/json")
            return False
        try:
            self.reviewer = self._portal_identity(url.query)
        except ValueError:
            self._plain(400, json.dumps({"error": "bad reviewer identity"}), ctype="application/json")
            return False
        if len(parts) == 2:  # /<id> -> /<id>/, keeping ?k=
            self._plain(301, "moved", {"Location": f"/{pkg_id}/" + (f"?{url.query}" if url.query else "")})
            return False
        self.path = "/" + parts[2] + (f"?{url.query}" if url.query else "")
        self.__class__ = cls
        return True

    def _portal_identity(self, query):
        """The portal user this request is for: the X-CJ-User header, trusted only because this request brought the
        package's key in X-Review-Key (checked by Packages.handler just before; no ?k=, which would outrank it, and not
        a cookie, which any browser that once opened a ?k= link sends). None when there is no such header, or the key
        came another way, or the request is not from loopback (LOCAL_CLIENTS: the portal runs on this PC; a header
        from any other address is ignored). ValueError for a trusted header that isn't one valid id."""
        vals = self.headers.get_all(IDENTITY_HEADER) or []
        if not vals or self.client_address[0] not in LOCAL_CLIENTS:
            return None
        if parse_qs(query).get("k", [None])[0] or self.headers.get("X-Review-Key") is None:
            return None
        who = vals[0].strip()
        if len(vals) != 1 or not who or not rs.valid_who(who):
            raise ValueError("bad identity")
        return who

    # ---- after dispatch (self is the package's class): with a portal identity, the page and saves carry it ----
    def do_GET(self):
        url = urlsplit(self.path)
        if self.reviewer and url.path in ("/", "/index.html"):
            path = TOOL_PAGE if TOOL_PAGE.is_file() else self.pkg_dir / "index.html"  # the current tool (docstring)
            if path.is_file():
                return self._send(200, inject_identity(path.read_bytes(), self.reviewer), "text/html; charset=utf-8")
        return super().do_GET()

    def do_HEAD(self):
        return self.do_GET()

    def do_PUT(self):
        if not self.reviewer:
            return super().do_PUT()
        try:
            n = int(self.headers.get("Content-Length", ""))
        except ValueError:
            n = -1
        if not 0 < n <= rs.MAX_BODY:  # review_server answers these (400 / 413) without reading the body
            return super().do_PUT()
        raw = self.rfile.read(n)
        try:
            body = json.dumps(stamp_identity(json.loads(raw, parse_constant=rs._no_constant), self.reviewer)).encode()
        except ValueError:  # not JSON: review_server rejects the original bytes
            body = raw
        del self.headers["Content-Length"]
        self.headers["Content-Length"] = str(len(body))
        rfile, self.rfile = self.rfile, io.BytesIO(body)
        try:
            return super().do_PUT()
        finally:
            self.rfile = rfile

    def do_POST(self):
        if self.reviewer:
            url = urlsplit(self.path)
            q = [(k, v) for k, v in parse_qsl(url.query, keep_blank_values=True) if k != "who"]
            self.path = url.path + "?" + urlencode(q + [("who", self.reviewer)])
        return super().do_POST()

    def log_message(self, fmt, *args):  # before dispatch (400/403/404/301): the same redaction as review_server
        msg = re.sub(r"([?&]k=)[^&\s\"]*", r"\1***", fmt % args)
        print(f"{self.address_string()} {msg}", flush=True)


class MultiServer(ThreadingHTTPServer):
    def server_close(self):
        super().server_close()
        self.RequestHandlerClass.packages.close()  # the package locks go with the server


def make_server(root: Path, host: str = "127.0.0.1", port: int = PORT, prefix=None, people=None, exclude=()):
    """A ThreadingHTTPServer for every <root>/<prefix>* package (port 0 = a free one; see server_address).
    prefix: default served_prefixes(); exclude: more names never served, on top of excluded_names() (serving())."""
    handler = type("MultiDispatch", (Dispatch,), {"packages": Packages(root, prefix, people, exclude)})
    return MultiServer((host, port), handler)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=None, help="Review root (default: <CryoAI-laptop>/review)")
    ap.add_argument("--host", default="127.0.0.1",
                    help="Bind address (default 127.0.0.1). Keep it local: phones reach packages through the CryoEV portal, "
                         "and a portal identity is trusted only from loopback, so don't expose this port (no "
                         "Tailscale IP, no `tailscale serve`, whose requests also arrive from loopback)")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--prefix", action="append", default=None,
                    help=f"Serve package folders starting with this; repeatable (default ${PREFIX_ENV}, else "
                         f"{' and '.join(PREFIX)}). lab_ is never served when the services config can't be read")
    ap.add_argument("--exclude", action="append", default=[],
                    help=f"A package id or folder never to serve; repeatable, added to ${EXCLUDE_ENV} (comma-separated). "
                         "Set it to the package the live single-package review server has open")
    a = ap.parse_args(argv)
    from registry import actor
    gone = actor.strip_env()  # main only: a server restarted from an agent session never credits its saves to the agent
    if gone:
        print(f"[OK] Removed {', '.join(gone)} from this server's environment (reviewers' saves are their own)", flush=True)
    prefix, exclude = serving(a.prefix, a.exclude)
    if not single_server_packages()[1]:
        print(f"[WARN] The services config {services_config()} can't be read: no lab_ package is served "
              "(the single-package review server's package is unknown)", flush=True)
    if a.root is None:
        from data_utils import config
        a.root = config.laptop_root() / "review"
    root = a.root.resolve()
    srv = make_server(root, a.host, a.port, prefix, exclude=a.exclude)
    for n in sorted(exclude):
        if not (root / n).is_dir() and not any(d.name.lower() == n for d in root.iterdir() if d.is_dir()):
            print(f"[WARNING] Excluded package {n!r} is no folder under {root}: check --exclude, ${EXCLUDE_ENV} and "
                  "the services config", flush=True)
    ids = srv.RequestHandlerClass.packages.ids()
    print(f"[OK] Serving {len(ids)} package(s) {'* / '.join(prefix)}* under {root} on {a.host}:{srv.server_address[1]}; "
          "new packages are found on their first request", flush=True)
    print(f"[OK] Never served (services config, --exclude, ${EXCLUDE_ENV}): {', '.join(sorted(exclude)) or 'none'}. "
          "One writer per package: a package another server has open is refused (409)", flush=True)
    for i in ids:
        print(f"[OK]   http://{a.host}:{srv.server_address[1]}/{i}/?k=<key in {root / i / '.review_key'}>", flush=True)
    if a.host in ("0.0.0.0", "::"):
        print("[WARN] Listening on every interface, LAN included. Keep --host 127.0.0.1: reach packages through "
              "the CryoEV portal.", flush=True)
    elif a.host not in LOCAL_CLIENTS and a.host != "localhost":
        print("[WARN] Not bound to loopback. Keep --host 127.0.0.1: reach packages through the CryoEV portal.", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
