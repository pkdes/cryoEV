"""
One process serving many review packages (BUILD_CONTRACTS.md section 3): GET/PUT/POST /<pkg id>/<path> is handled
by review_server.make_handler for that package, exactly as a single review_server.py would handle /<path>.
review_server.py is wrapped, not changed; the real review (review_server.py on :8766) is a separate process.

  python -m annotation.multi_review_server [--root <CryoAI-laptop>/review] [--port 8767] [--host 127.0.0.1]

- Packages: folders <root>/<prefix>*/ with data.js (default prefix pred_: prediction packages only, so this server
  never writes the edits of a package another server has open). A package added after start is found on its first
  request; a rebuilt data.js or a new .review_key is picked up on the next one.
- Auth: each package keeps its own key (<pkg>/.review_key, made when missing): ?k=<key> once, then a cookie
  review_key_<port>_<pkg id> (one per package, since every package shares this port), or X-Review-Key.
- Errors: /<id>/ for an unknown id -> 404; a malformed id or a path with .., \\ or : -> 400; / -> 404 (no listing).
  /<id> redirects to /<id>/ so the page's relative URLs (data.js, images/, api/...) stay inside the package.
"""
import argparse
import json
import re
import secrets
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from annotation import review_server as rs  # noqa: E402

PORT = 8767
PREFIX = "pred_"
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
BAD_PATH = ("..", "\\", ":")


def package_dir(root: Path, pkg_id: str, prefix: str = PREFIX) -> Path | None:
    """<root>/<pkg_id> when pkg_id is a well-formed id with the prefix and that folder holds data.js, else None."""
    if not (ID_RE.match(pkg_id or "") and ".." not in pkg_id and pkg_id.startswith(prefix)):
        return None
    p = Path(root) / pkg_id
    return p if (p / "data.js").is_file() and p.resolve().parent == Path(root).resolve() else None


def package_key(pkg: Path) -> str:
    """The package's access key from <pkg>/.review_key; one is made (and saved) when there is none."""
    f = Path(pkg) / ".review_key"
    key = f.read_text(encoding="utf-8").strip() if f.is_file() else ""
    if not key:
        key = secrets.token_urlsafe(18)
        f.write_text(key, encoding="utf-8")
    return key


class Packages:
    """pkg id -> handler class for that package, made on first use and remade when data.js or .review_key change."""

    def __init__(self, root: Path, prefix: str = PREFIX, people=None):
        self.root, self.prefix = Path(root), prefix
        self.people = rs.load_people() if people is None else list(people)
        self._cache, self._lock = {}, threading.Lock()

    def handler(self, pkg_id: str):
        pkg = package_dir(self.root, pkg_id, self.prefix)
        with self._lock:
            if pkg is None:
                self._cache.pop(pkg_id, None)
                return None
            key = package_key(pkg)
            sig = ((pkg / "data.js").stat().st_mtime_ns, (pkg / ".review_key").stat().st_mtime_ns)
            hit = self._cache.get(pkg_id)
            if hit and hit[0] == sig:
                return hit[1]
            data = json.loads((pkg / "data.js").read_text(encoding="utf-8").split("=", 1)[1].strip().rstrip(";"))
            inner = rs.make_handler(pkg, key, {im["file"] for im in data["images"]}, data.get("id", pkg_id),
                                    self.people, rs.is_sandbox(pkg))

            def _cookie(h, _id=pkg_id):  # one cookie per package: they all share this port
                return f"{rs.cookie_name(h.server.server_address[1])}_{_id}"

            cls = type(f"Pkg_{pkg_id}", (Dispatch, inner), {"_cookie": _cookie, "pkg_id": pkg_id, "packages": self,
                                                            "log_message": inner.log_message})
            self._cache[pkg_id] = (sig, cls)
            return cls

    def ids(self):
        return sorted(p.name for p in self.root.glob(f"{self.prefix}*") if package_dir(self.root, p.name, self.prefix))


class Dispatch(BaseHTTPRequestHandler):
    """Reads the request line, picks the package from the first path segment, then hands the request to that
    package's review_server handler: self.path loses the /<id> prefix and the instance takes the package's class
    (it stays first in the MRO, so a kept-alive connection is dispatched again on its next request)."""
    timeout = rs.SOCKET_TIMEOUT
    packages: Packages = None

    def _plain(self, code, text, extra=None):
        body = text.encode()
        self.send_response(code)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
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
        cls = self.packages.handler(pkg_id)
        if cls is None:
            self._plain(404, "unknown package")
            return False
        if len(parts) == 2:  # /<id> -> /<id>/, keeping ?k=
            self._plain(301, "moved", {"Location": f"/{pkg_id}/" + (f"?{url.query}" if url.query else "")})
            return False
        self.path = "/" + parts[2] + (f"?{url.query}" if url.query else "")
        self.__class__ = cls
        return True

    def log_message(self, fmt, *args):  # before dispatch (400/404/301): the same redaction as review_server
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
                    help="Bind address (default 127.0.0.1; for phones use `tailscale serve` or the Tailscale IP)")
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
        print("[WARN] Listening on every interface, LAN included. Prefer --host <Tailscale IP>.", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
