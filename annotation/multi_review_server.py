"""
One process serving many review packages (BUILD_CONTRACTS.md section 3): GET/PUT/POST /<pkg id>/<path> is handled
by review_server.make_handler for that package, exactly as a single review_server.py would handle /<path>.
review_server.py is wrapped, not changed; the real review (review_server.py on :8766) is a separate process.

  python -m annotation.multi_review_server [--root <CryoAI-laptop>/review] [--port 8767] [--host 127.0.0.1]

- Packages: complete folders <root>/<prefix>*/ (data.js + review_package.json), reached under exactly their own
  name (no case or short-name aliases). Default prefix pred_: prediction packages only, so this server never writes
  the edits of a package another server has open. A package added after start is found on its first request; a
  rebuilt data.js or a new .review_key is picked up on the next one.
- Auth: each package keeps its own key (<pkg>/.review_key; no key, not served): ?k=<key> once, then a cookie
  review_key_<port>_<pkg id> (one per package, since every package shares this port), or X-Review-Key. The key is
  checked here before anything is made for the package (handler, cache entry, edits/ folder).
- Errors: an unknown id and a wrong key both get 403, so ids can't be enumerated; a malformed id or a path with
  .., \\ or : -> 400; / -> 404 (no listing). /<id> (with its key) redirects to /<id>/ so the page's relative URLs
  (data.js, images/, api/...) stay inside the package.
- Portal identity (BUILD_CONTRACTS.md section 8): the CJ portal proxies /review/<id>/app/... here with the package's key
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
"""
import argparse
import io
import json
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
PREFIX = "pred_"
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
BAD_PATH = ("..", "\\", ":")
COMPLETE = "review_package.json"  # prediction_package.build writes it last (and builds in a temp folder anyway)
IDENTITY_HEADER = "X-CJ-User"     # set by portal/proxy.py (client copies stripped); trusted only with the key header
LOCAL_CLIENTS = frozenset({"127.0.0.1", "::1", "::ffff:127.0.0.1"})  # the portal's address: the only identity source
TOOL_PAGE = REPO / "annotation" / "review_tool" / "index.html"     # served (with the identity) to portal requests


def cookie_name(port: int, pkg_id: str) -> str:
    """review_key_<port>_<pkg id>: every package shares this port, so each needs its own cookie."""
    return f"{rs.cookie_name(port)}_{pkg_id}"


def package_dir(root: Path, pkg_id: str, prefix: str = PREFIX) -> Path | None:
    """The resolved <root>/<pkg_id> when pkg_id is a well-formed id with the prefix, the folder is complete
    (data.js and review_package.json, which the builder writes last) and it is reached under exactly that name: on a
    case-insensitive disk /PRED_x/ or an 8.3 short name would otherwise be a second package with its own lock."""
    if not (ID_RE.match(pkg_id or "") and ".." not in pkg_id and pkg_id.startswith(prefix)):
        return None
    root = Path(root).resolve()
    p = (root / pkg_id).resolve()
    if p.parent != root or p.name != pkg_id:
        return None
    return p if (p / "data.js").is_file() and (p / COMPLETE).is_file() else None


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


class Packages:
    """pkg id -> handler class for that package, made on the first request with its key and remade when data.js or
    .review_key change. Cached by resolved folder; nothing is cached or created for an unknown id or a wrong key."""

    def __init__(self, root: Path, prefix: str = PREFIX, people=None):
        self.root, self.prefix = Path(root), prefix
        self.people = rs.load_people() if people is None else list(people)
        self._cache, self._lock = {}, threading.Lock()

    def handler(self, pkg_id: str, given):
        """The package's handler class when `given` is its key, else None (unknown id and wrong key alike)."""
        pkg = package_dir(self.root, pkg_id, self.prefix)
        key = package_key(pkg) if pkg else ""
        # bytes: compare_digest refuses non-ASCII str, and a ?k= can hold anything
        ok = secrets.compare_digest(str(given or "").encode(), (key or "\0no package\0").encode()) and bool(key)
        if not ok:
            return None
        with self._lock:
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
        return sorted(p.name for p in self.root.glob(f"{self.prefix}*") if package_dir(self.root, p.name, self.prefix))


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


def make_server(root: Path, host: str = "127.0.0.1", port: int = PORT, prefix: str = PREFIX, people=None):
    """A ThreadingHTTPServer for every <root>/<prefix>* package (port 0 = a free one; see server_address)."""
    handler = type("MultiDispatch", (Dispatch,), {"packages": Packages(root, prefix, people)})
    return ThreadingHTTPServer((host, port), handler)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=None, help="Review root (default: <CryoAI-laptop>/review)")
    ap.add_argument("--host", default="127.0.0.1",
                    help="Bind address (default 127.0.0.1). Keep it local: phones reach packages through the CJ portal, "
                         "and a portal identity is trusted only from loopback, so don't expose this port (no "
                         "Tailscale IP, no `tailscale serve`, whose requests also arrive from loopback)")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--prefix", default=PREFIX, help=f"Serve only package folders starting with this (default {PREFIX})")
    a = ap.parse_args(argv)
    if a.root is None:
        from data_utils import config
        a.root = config.laptop_root() / "review"
    root = a.root.resolve()
    srv = make_server(root, a.host, a.port, a.prefix)
    ids = srv.RequestHandlerClass.packages.ids()
    print(f"[OK] Serving {len(ids)} package(s) {a.prefix}* under {root} on {a.host}:{srv.server_address[1]}; "
          "new packages are found on their first request", flush=True)
    for i in ids:
        print(f"[OK]   http://{a.host}:{srv.server_address[1]}/{i}/?k=<key in {root / i / '.review_key'}>", flush=True)
    if a.host in ("0.0.0.0", "::"):
        print("[WARN] Listening on every interface, LAN included. Keep --host 127.0.0.1: reach packages through "
              "the CJ portal.", flush=True)
    elif a.host not in LOCAL_CLIENTS and a.host != "localhost":
        print("[WARN] Not bound to loopback. Keep --host 127.0.0.1: reach packages through the CJ portal.", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
