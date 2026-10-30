"""
Serve a label review package (from build_review_package.py) over HTTP and save
edits to disk, so the browser tool can be used remotely (e.g. behind a
Microsoft Dev Tunnel) by several reviewers at once, and still work offline with later sync.

Standard library only. Every request needs the access key: open the printed
URL once (…/?k=<key>) and a cookie carries it from then on.

Routes:
  GET  /, /data.js, /images/<name>   the package itself
  GET  /api/ping                     {"ok": true}
  GET  /api/edits                    {file: edit record} for every saved image
  PUT  /api/edits?f=<file>           merge one image's record into the saved one (per-decision, newest
                                     wins; see merge_edits) and return the merged record
  POST /api/claim?c=<case>&who=<n>   reserve the card (or "<file>|editor") on a reviewer's screen for CLAIM_SECONDS
  GET  /api/claims                   active claims {case: reviewer}, so other reviewers skip them

Edits are written to <package>/edits/<file>.json (one file per image, atomic
replace). The record format (decision keys and values) is in docs/REVIEW_TOOL.md;
effective_labels() applies one record to its image without the browser, for the
planned annotation/apply_review_edits.py that turns them into a new COCO export.

Usage:
  python annotation/review_server.py --package "../CryoAI/annotation review/cryoai_v4_review"
  python annotation/review_server.py --package <pkg> --host 0.0.0.0   # reachable from a phone over Tailscale
"""
import argparse
import json
import mimetypes
import os
import secrets
import subprocess
import threading
import time
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

COOKIE = "review_key"
DEC_PREFIXES = ("g", "p", "n")  # g<label id>, p<prediction index>, n<id of an outline drawn in the tool>
MAX_BODY = 20 * 2**20
CLAIM_SECONDS = 90


def make_handler(pkg: Path, key: str, valid_files: set):
    edits_dir = pkg / "edits"
    edits_dir.mkdir(exist_ok=True)
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            if not self.path.startswith("/images/"):
                print(f"{self.address_string()} {fmt % args}", flush=True)

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

        def _authed(self, query):
            cookie = SimpleCookie(self.headers.get("Cookie", ""))
            given = (query.get("k", [None])[0] or self.headers.get("X-Review-Key")
                     or (cookie[COOKIE].value if COOKIE in cookie else None))
            return given is not None and secrets.compare_digest(given, key)

        def do_GET(self):
            url = urlparse(self.path)
            q = parse_qs(url.query)
            if not self._authed(q):
                return self._send(403, b"Access key required: open the full link that includes ?k=...", "text/plain")
            extra = {"Set-Cookie": f"{COOKIE}={key}; Path=/; HttpOnly; SameSite=Lax; Max-Age=2592000"} if "k" in q else None
            if url.path == "/api/ping":
                return self._json(200, {"ok": True})
            if url.path == "/api/claims":
                now = time.time()
                with lock:
                    return self._json(200, {"claims": {k: w for k, (w, exp) in claims.items() if exp >= now}})
            if url.path == "/api/edits":
                out = {}
                with lock:
                    for f in edits_dir.glob("*.json"):
                        try:
                            rec = json.loads(f.read_text(encoding="utf-8"))
                            out[rec["file"]] = rec["edit"]
                        except (ValueError, KeyError):
                            pass
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
            self._send(200, path.read_bytes(), ctype, extra)

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
            n = int(self.headers.get("Content-Length", 0))
            if n <= 0 or n > MAX_BODY:
                return self._json(413, {"error": "bad body size"})
            try:
                edit = json.loads(self.rfile.read(n))
                assert valid_edit(edit)
            except (ValueError, AssertionError):
                return self._json(400, {"error": "bad edit"})
            target = edits_dir / f"{file}.json"
            with lock:
                cur = json.loads(target.read_text(encoding="utf-8"))["edit"] if target.exists() else None
                merged = merge_edits(cur, edit)
                tmp = target.with_suffix(".tmp")
                tmp.write_text(json.dumps({"file": file, "edit": merged}), encoding="utf-8")
                os.replace(tmp, target)
            return self._json(200, {"edit": merged})

        def do_POST(self):
            url = urlparse(self.path)
            q = parse_qs(url.query)
            if not self._authed(q):
                return self._json(403, {"error": "access key required"})
            if url.path != "/api/claim":
                return self._json(404, {"error": "not found"})
            case, who = q.get("c", [""])[0][:600], q.get("who", ["anonymous"])[0][:40] or "anonymous"
            now = time.time()
            with lock:
                for k in [k for k, (_, exp) in claims.items() if exp < now]:
                    del claims[k]
                for k in [k for k, (w, _) in claims.items() if w == who]:  # one card per reviewer at a time
                    del claims[k]
                if case and (case not in claims or claims[case][0] == who):
                    claims[case] = (who, now + CLAIM_SECONDS)
                active = {k: w for k, (w, _) in claims.items()}
            return self._json(200, {"claims": active})

    claims = {}  # "<file>|<case key>" -> (reviewer, expiry): the card on each reviewer's screen

    return Handler


def _valid_rings(p):
    return isinstance(p, list) and len(p) > 0 and all(
        isinstance(r, list) and len(r) >= 6 and len(r) % 2 == 0
        and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in r) for r in p)


def valid_edit(edit) -> bool:
    """Shape check for a PUT body: a dict with a numeric t, decisions under known keys, optional polygon list."""
    if not isinstance(edit, dict) or not isinstance(edit.get("t", 0), (int, float)):
        return False
    if not isinstance(edit.get("dec") or {}, dict) or not isinstance(edit.get("gt") or [], list):
        return False
    for k, d in (edit.get("dec") or {}).items():
        if not k or k[0] not in DEC_PREFIXES:
            return False
        if isinstance(d, str):  # the oldest records stored bare strings
            continue
        if not isinstance(d, dict) or not isinstance(d.get("v"), (str, type(None))) \
                or not isinstance(d.get("t", 0), (int, float)):
            return False
        if d.get("p") is not None and not _valid_rings(d["p"]):
            return False
    return True


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


def merge_edits(a, b):
    """Merge two reviewers' records for one image: each decision key keeps its newest entry, the polygon
    list keeps the newest Editor save, the reviewed flag keeps its newest toggle. Mirrors mergeEdit()
    in review_tool/index.html."""
    if not a:
        return _migrate(b)
    a, b = _migrate(a), _migrate(b)
    out = {"dec": {}, "t": max(a.get("t", 0), b.get("t", 0)), "modified": bool(a.get("modified") or b.get("modified"))}
    for k in dict.fromkeys([*a["dec"], *b["dec"]]):  # a's keys, then b's new ones: the same order as mergeEdit()
        x, y = a["dec"].get(k), b["dec"].get(k)
        out["dec"][k] = y if x is None else x if y is None else (y if y.get("t", 0) > x.get("t", 0) else x)
    if a.get("gtT") or b.get("gtT"):
        s = b if (b.get("gtT") or 0) > (a.get("gtT") or 0) else a
        out["gt"], out["gtT"] = s.get("gt"), s.get("gtT")
    ra, rb = a.get("reviewedT", 0) or 0, b.get("reviewedT", 0) or 0
    out["reviewed"] = bool(b.get("reviewed")) if rb > ra else bool(a.get("reviewed"))
    out["reviewedT"] = max(ra, rb)
    return out


def _dec_v(e, key):
    d = e["dec"].get(key)
    return d.get("v") if d else None


def effective_labels(image: dict, edit) -> list:
    """One image's labels after its edit record, as the review tool shows them (effGt() in index.html).

    image: a data.js images[] entry (gt: [{id, c, p}], pr: [{s, p}]); edit: a saved record, any version.
    Returns [{id, c, p, src, ...}]: id is the COCO annotation id (None for added objects), c the class,
    p the outline (list of normalized flat rings), src "p<k>" (accepted prediction) or "n<id>" (drawn).
    Base = the record's polygon list if it has one (gtT), else the package labels; every decision then
    applies on top, and its own polygon only if it is newer than that list."""
    e = _migrate(edit)
    gt_t = e.get("gtT") or 0
    base = e["gt"] if e.get("gtT") and e.get("gt") is not None else image["gt"]

    def dec_p(key):
        d = e["dec"].get(key)
        return d["p"] if d and d.get("p") and d.get("t", 0) > gt_t else None

    out = []
    for g in base:
        gkey = f"g{g['id']}" if g.get("id") is not None else None
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
                pr = image["pr"][int(key[1:])]
            except (ValueError, IndexError):
                continue
            out.append({"id": None, "c": v[4:], "p": json.loads(json.dumps(d.get("p") or pr["p"])), "src": key})
        elif key[0] == "n" and v.startswith("new:") and d.get("p"):
            out.append({"id": None, "c": v[4:], "p": json.loads(json.dumps(d["p"])), "src": key})
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
                    help="Bind address: 127.0.0.1 behind a tunnel or `tailscale serve`; 0.0.0.0 for phones over Tailscale")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--key", default=None, help="Access key (default: reuse <package>/.review_key, else generate one)")
    args = ap.parse_args()

    pkg = args.package.resolve()
    data = json.loads((pkg / "data.js").read_text(encoding="utf-8").split("=", 1)[1].strip().rstrip(";"))
    valid = {im["file"] for im in data["images"]}
    key_file = pkg / ".review_key"
    key = args.key or (key_file.read_text().strip() if key_file.exists() else secrets.token_urlsafe(18))
    key_file.write_text(key)

    srv = ThreadingHTTPServer((args.host, args.port), make_handler(pkg, key, valid))
    print(f"[OK] Serving {pkg.name}: {len(valid)} images, edits -> {pkg / 'edits'}", flush=True)
    local = "127.0.0.1" if args.host == "0.0.0.0" else args.host
    print(f"[OK] Local link: http://{local}:{args.port}/?k={key}", flush=True)
    if args.host not in ("127.0.0.1", "localhost"):
        for h in tailscale_hosts():
            print(f"[OK] Tailscale link (phone): http://{h}:{args.port}/?k={key}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
