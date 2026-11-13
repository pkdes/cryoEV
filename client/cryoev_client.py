"""
CryoEV agent API client: one file, Python standard library only (docs/API.md).

A person makes a token for their own agent on the portal's My account page (/account/); the agent sends it as
`Authorization: Bearer <token>`. The portal opens a fixed list of JSON read routes to it, with that person's rights
capped at Advanced. A "Read + write" token also opens a few writes: upload images, start a prediction from a session,
compute results, send feedback (60 writes an hour per person; an upload's pieces count once). Tailnet address only
for now.

    set CRYOEV_URL=https://<the portal's tailnet address>      (PowerShell: $env:CRYOEV_URL = "...")
    set CRYOEV_TOKEN=<the token from My account>
    python cryoev_client.py me
    python cryoev_client.py overview | search <text> | list <kind> | item <kind> <id> | lineage <kind> <id> | graph
    python cryoev_client.py mine | set <ps id> | results <ps id> | jobs
    python cryoev_client.py session-form <key> | wait <tid> | convert <cid> | compute <ps id> | feedback <text>
                                                                                              (write token)

    from cryoev_client import Client
    c = Client()                     # CRYOEV_URL / CRYOEV_TOKEN, or Client(url, token)
    c.me()                           # {"id": ..., "role": ..., "via": <agent>, "auth": "bearer", "scope": "read"}
    # a write token:
    up = c.upload(["a.png", "b.png"], sample_label="EV prep", nm_per_px=1.47, model="<model id>", confirm_nm=True)
    c.wait(up["tid"])                # {"state": "done", "ps": "ps_...", ...}
    c.predict_session("<key>", confirm_nm=True)   # the key: ask the person (/predict/new/session/<key>/), or
                                                  # convert(cid)["key"] for a raw upload's finished conversion
    c.compute("ps_...")              # then poll c.results_status("ps_...")
    c.feedback("The upload page ...", kind="bug")

Every call returns the parsed JSON. A refusal raises ApiError(status, message): 401 = the token is unknown, expired,
revoked, the API is off, or the route is not on the API's list; 403 = the person's role can't open it, a read-only
token on a write, writes switched off (or the Cloudflare address, which doesn't take tokens yet); 404 = not found or
not yours; 429 = a limit (ApiError.retry_after: seconds). The token is never printed.

Redirects are never followed (the token is never sent anywhere else): a write that answers 303 is read from its
Location (the progress id, the set, the feedback id); the HTML pages it points to are not on the API's list.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

URL_ENV, TOKEN_ENV = "CRYOEV_URL", "CRYOEV_TOKEN"
TIMEOUT_S = 30.0
LIST_KINDS = ("sessions", "grids", "images", "labels", "datasets", "models")
DETAIL_KINDS = ("session", "grid", "images", "labels", "dataset", "model")
TAG_RE = re.compile(r"<[^>]+>")
ALERT_RE = re.compile(r'role="alert"[^>]*>(.*?)</div>', re.S)


class ApiError(Exception):
    def __init__(self, status: int, message: str, retry_after: float | None = None):
        super().__init__(f"{status}: {message}")
        self.status = status
        self.message = message
        self.retry_after = retry_after


def _seg(x) -> str:
    """One path segment, percent-encoded (an id can never add a path segment)."""
    return urllib.parse.quote(str(x), safe="")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Never follow a redirect: the caller reads Location (and the token never goes to another address)."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_OPENER = urllib.request.build_opener(_NoRedirect)


def _message(raw: bytes) -> str:
    """A refusal's text: the JSON "error", else an HTML page's alert box (or its text), cut to 300 characters."""
    try:
        doc = json.loads(raw)
        if isinstance(doc, dict) and doc.get("error"):
            return str(doc["error"])[:300]
    except ValueError:
        pass
    text = raw.decode("utf-8", "replace")
    m = ALERT_RE.search(text)
    text = TAG_RE.sub(" ", m.group(1) if m else text)
    return " ".join(text.split())[:300]


def location_part(location: str, pattern: str) -> str | None:
    """A Location's part matching `pattern`'s group 1 (e.g. the progress id of /predict/preparing/<tid>/)."""
    m = re.search(pattern, urllib.parse.urlsplit(location or "").path + "?" +
                  urllib.parse.urlsplit(location or "").query)
    return m.group(1) if m else None


class Client:
    def __init__(self, url: str | None = None, token: str | None = None, timeout: float = TIMEOUT_S):
        self.url = (url or os.environ.get(URL_ENV) or "").rstrip("/")
        self._token = token or os.environ.get(TOKEN_ENV) or ""
        self.timeout = timeout
        if not self.url:
            raise ValueError(f"no portal address: pass url or set {URL_ENV}")
        if not self._token:
            raise ValueError(f"no token: pass token or set {TOKEN_ENV} (make one on the portal's My account page)")

    def __repr__(self):                                  # never the token
        return f"Client({self.url!r})"

    # ---- HTTP
    def request(self, method: str, path: str, params: dict | None = None, body: bytes | None = None,
                ctype: str | None = None) -> tuple:
        """One request with the bearer token -> (status, headers, body bytes). A 2xx or 3xx is returned (3xx: never
        followed); anything else raises ApiError."""
        q = {k: v for k, v in (params or {}).items() if v is not None}
        full = self.url + path + (("?" + urllib.parse.urlencode(q)) if q else "")
        hdrs = {"Authorization": f"Bearer {self._token}", "Accept": "application/json", "User-Agent": "cryoev-client/2"}
        if ctype:
            hdrs["Content-Type"] = ctype
        req = urllib.request.Request(full, data=body, method=method, headers=hdrs)
        try:
            with _OPENER.open(req, timeout=self.timeout) as r:
                return r.status, r.headers, r.read()
        except urllib.error.HTTPError as e:
            raw = e.read()
            if 300 <= e.code < 400:
                return e.code, e.headers, raw
            ra = e.headers.get("Retry-After") if e.headers else None
            try:
                ra = float(ra) if ra is not None else None
            except ValueError:
                ra = None
            raise ApiError(e.code, _message(raw), ra) from None

    def get(self, path: str, params: dict | None = None):
        """GET <url><path>?<params> -> parsed JSON. Raises ApiError for a non-2xx answer."""
        st, _, body = self.request("GET", path, params)
        if st >= 300:
            raise ApiError(st, "unexpected redirect (not followed)")
        return json.loads(body)

    def post_json(self, path: str, doc: dict):
        st, _, body = self.request("POST", path, body=json.dumps(doc).encode("utf-8"), ctype="application/json")
        return json.loads(body) if body else {}

    def post_form(self, path: str, fields) -> str:
        """POST an urlencoded form; the portal answers 303 -> its Location (not followed)."""
        st, h, _ = self.request("POST", path, body=urllib.parse.urlencode(fields, doseq=True).encode("utf-8"),
                                ctype="application/x-www-form-urlencoded")
        loc = h.get("Location") if h else None
        if not (300 <= st < 400) or not loc:
            raise ApiError(st, "the portal did not answer with a redirect")
        return loc

    # ---- the API's read routes (portal/app.py BEARER_ROUTES)
    def me(self):
        return self.get("/cj/me.json")

    def overview(self):
        return self.get("/data/api/overview")

    def search(self, text: str):
        return self.get("/data/api/search", {"q": text})

    def list(self, kind: str):
        return self.get(f"/data/api/list/{_seg(kind)}")

    def item(self, kind: str, eid: str, page: int | None = None):
        return self.get(f"/data/api/{_seg(kind)}/{_seg(eid)}", {"page": page})

    def lineage(self, kind: str, eid: str):
        return self.get(f"/data/api/lineage/{_seg(kind)}/{_seg(eid)}")

    def graph(self, focus: str | None = None, scope: str | None = None):
        return self.get("/data/api/lineage/graph", {"focus": focus, "scope": scope})

    def mine(self):
        return self.get("/predict/api/mine")

    def prediction_set(self, ps: str):
        return self.get(f"/predict/api/set/{_seg(ps)}")

    def results_status(self, ps: str):
        return self.get(f"/results/{_seg(ps)}/api/status")

    def jobs_summary(self):
        return self.get("/jobs/api/summary")

    # ---- writes (a read + write token; portal/app.py BEARER_WRITE_ROUTES)
    def upload(self, files, sample_label: str | None = None, nm_per_px=None, model: str | None = None,
               name: str | None = None, kind: str = "tiles", fields: dict | None = None, create: bool = True,
               confirm_nm: bool = False):
        """Upload image files (paths) as one upload session, piece by piece, then (create=True) start the analysis.
        Tiles: 8-bit PNG/JPEG, one sample label and pixel size (nm per pixel) for all, a model id (see the portal's
        New analysis page; None = the production model is NOT guessed: the portal refuses). confirm_nm=True is
        required for tiles: it confirms the pixel size, as the page's tick box does (the person is recorded as having
        confirmed it, through this agent: only pass the value the person gave). kind="raw" (advanced): `fields` as
        the raw form's. -> {"upload": id, "tid": progress id or None, "conversion": id or None, "href": where the
        page would go}."""
        if kind == "tiles" and confirm_nm is not True:
            raise ValueError("pass confirm_nm=True to confirm the pixel size (only the value the person gave)")
        paths = [os.fspath(p) for p in files]
        decl = [{"name": os.path.basename(p), "size": os.path.getsize(p)} for p in paths]
        f = dict(fields or {})
        if kind == "tiles":
            f.update({"sample_label": sample_label, "nm_per_px": None if nm_per_px is None else str(nm_per_px),
                      "confirm": "yes", "model": model})
            if name:
                f["name"] = name
        f = {k: v for k, v in f.items() if v is not None}
        st = self.post_json("/predict/api/upload", {"kind": kind, "fields": f, "files": decl})
        uid = st["upload"]
        for p, d in zip(paths, decl):
            self._send_file(uid, p, d["name"], d["size"], st)
        out = {"upload": uid, "tid": None, "conversion": None, "href": None}
        if not create:
            return out
        href = self.post_json(f"/predict/api/upload/{_seg(uid)}/create", {}).get("href") or ""
        out["href"] = href
        out["tid"] = location_part(href, r"^/predict/preparing/([^/]+)/")
        out["conversion"] = location_part(href, r"^/predict/convert/([^/]+)/")
        return out

    def _send_file(self, uid: str, path: str, name: str, size: int, state: dict):
        chunk = int(state.get("chunk") or 8 * 2 ** 20)
        offset = int((state.get("received") or {}).get(name) or 0)
        with open(path, "rb") as fh:
            while offset < size:
                fh.seek(offset)
                data = fh.read(min(chunk, size - offset))
                try:
                    self.request("POST", f"/predict/api/upload/{_seg(uid)}/file", {"name": name, "offset": offset},
                                 data, "application/octet-stream")
                    offset += len(data)
                except ApiError as e:
                    if e.status != 409:                  # 409: the portal has another offset; carry on from there
                        raise
                    cur = self.upload_state(uid)
                    offset = int((cur.get("received") or {}).get(name) or 0)

    def upload_state(self, uid: str):
        return self.get(f"/predict/api/upload/{_seg(uid)}")

    def cancel_upload(self, uid: str):
        return self.post_json(f"/predict/api/upload/{_seg(uid)}/cancel", {})

    def preparing(self, tid: str):
        return self.get(f"/predict/api/preparing/{_seg(tid)}")

    def wait(self, tid: str, timeout: float = 1800.0, every: float = 2.0):
        """Poll the create's progress until it is done or failed -> its JSON plus "ps" (the new set's id, when
        done). Raises TimeoutError."""
        end = time.monotonic() + timeout
        while True:
            doc = self.preparing(tid)
            if doc.get("state") in ("done", "failed"):
                doc["ps"] = location_part(doc.get("href") or "", r"^/predict/(ps_[^/]+)/")
                return doc
            if time.monotonic() > end:
                raise TimeoutError(f"{tid} still {doc.get('state')} after {timeout:g} s")
            time.sleep(every)

    def session_form(self, key: str):
        """A session's form data: sig, its samples (gid, label, n, nm ...), the models you may pick."""
        return self.get(f"/predict/new/session/{_seg(key)}/", {"json": "1"})

    def predict_session(self, key: str, nm: dict | None = None, samples=None, model: str | None = None,
                        label: str | None = None, count: int | None = None, name: str | None = None,
                        confirm_nm: bool = False):
        """Start an analysis of a session's tiles (its key: ask the person, it is in the session's address
        /predict/new/session/<key>/; or convert(cid)["key"] for a raw upload's finished conversion). samples: gids
        or labels (default every sample that can be analysed); nm: {gid or label: nm per pixel} (default the
        manifest's value; a sample with no value needs one). confirm_nm=True is required: it confirms the pixel
        sizes, as the page's tick box does. -> {"tid": progress id, "href"}; then wait(tid)."""
        if confirm_nm is not True:
            raise ValueError("pass confirm_nm=True to confirm the pixel sizes (check them in session_form first)")
        form = self.session_form(key)
        nm = dict(nm or {})
        groups = [g for g in form["groups"] if not g.get("mixed")]
        want = None if samples is None else {str(s) for s in samples}
        fields = [("sig", form["sig"]), ("confirm", "yes"), ("model", model or form.get("default_model") or ""),
                  ("how", "count" if count else "all")]
        for g in groups:
            if want is not None and g["gid"] not in want and g["label"] not in want:
                continue
            fields.append(("samples", g["gid"]))
            v = nm.get(g["gid"], nm.get(g["label"]))
            if v is None and not g.get("nm_range") and g.get("nm"):
                v = g["nm"]
            fields.append((f"nm_{g['gid']}", "" if v is None else str(v)))
        if count:
            fields.append(("count", str(int(count))))
        if form.get("needs_label"):
            fields.append(("label", label or form.get("label") or ""))
        if name:
            fields.append(("name", name))
        for k, v in (form.get("kept") or {}).items():
            fields.append((k, v))
        loc = self.post_form(f"/predict/new/session/{_seg(key)}/", fields)
        return {"tid": location_part(loc, r"^/predict/preparing/([^/]+)/"), "href": loc}

    def convert(self, cid: str):
        """A raw upload's conversion (one this agent made): state, tiles; "key" when done (for predict_session)."""
        return self.get(f"/predict/api/convert/{_seg(cid)}")

    def compute(self, ps: str):
        """Compute (or refresh) a finished set's results; poll results_status(ps). -> {"ps", "href"}. A set whose
        predictions are not finished: ApiError 409."""
        loc = self.post_form(f"/results/{_seg(ps)}/compute", {})
        return {"ps": location_part(loc, r"^/results/([^/]+)/"), "href": loc}

    def feedback(self, text: str, kind: str = "bug", steps: str = "", page: str = ""):
        """Send a bug report, feature idea or question (kind bug | feature | question). -> {"id": its number}."""
        loc = self.post_form("/feedback/", {"kind": kind, "text": text, "steps": steps, "page": page})
        rid = location_part(loc, r"[?&]id=(\d+)")
        return {"id": int(rid) if rid else None, "href": loc}


COMMANDS = {                                  # name -> (Client method, number of arguments)
    "me": ("me", 0), "overview": ("overview", 0), "search": ("search", 1), "list": ("list", 1),
    "item": ("item", 2), "lineage": ("lineage", 2), "graph": ("graph", 0), "mine": ("mine", 0),
    "set": ("prediction_set", 1), "results": ("results_status", 1), "jobs": ("jobs_summary", 0),
    "session-form": ("session_form", 1), "wait": ("wait", 1), "convert": ("convert", 1), "compute": ("compute", 1),
    "feedback": ("feedback", 1),
}


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in COMMANDS:
        print(__doc__)
        return 2
    name, n = COMMANDS[argv[0]]
    if len(argv) - 1 != n:
        print(f"{argv[0]} takes {n} argument(s)", file=sys.stderr)
        return 2
    try:
        out = getattr(Client(), name)(*argv[1:])
    except ValueError as e:
        print(f"[--] {e}", file=sys.stderr)
        return 2
    except ApiError as e:
        print(f"[--] {e}", file=sys.stderr)
        return 1
    except OSError as e:                      # unreachable, TLS, timeout
        print(f"[--] can't reach the portal: {e}", file=sys.stderr)
        return 1
    print(json.dumps(out, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
