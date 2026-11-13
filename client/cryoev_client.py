"""
CryoEV agent API client: one file, Python standard library only (docs/API.md). Read-only.

A person makes a token for their own agent on the portal's My account page (/account/); the agent sends it as
`Authorization: Bearer <token>`. The portal opens a fixed list of JSON read routes to it, with that person's rights
capped at Advanced. Tailnet address only for now.

    set CRYOEV_URL=https://<the portal's tailnet address>      (PowerShell: $env:CRYOEV_URL = "...")
    set CRYOEV_TOKEN=<the token from My account>
    python cryoev_client.py me
    python cryoev_client.py overview | search <text> | list <kind> | item <kind> <id> | lineage <kind> <id> | graph
    python cryoev_client.py mine | set <ps id> | results <ps id> | jobs

    from cryoev_client import Client
    c = Client()                     # CRYOEV_URL / CRYOEV_TOKEN, or Client(url, token)
    c.me()                           # {"id": ..., "role": ..., "via": <agent>, "auth": "bearer"}

Every call returns the parsed JSON. A refusal raises ApiError(status, message): 401 = the token is unknown, expired,
revoked, the API is off, or the route is not on the API's list; 403 = the person's role can't open it (or the
Cloudflare address, which doesn't take tokens yet); 404 = not found or not yours. The token is never printed.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

URL_ENV, TOKEN_ENV = "CRYOEV_URL", "CRYOEV_TOKEN"
TIMEOUT_S = 30.0
LIST_KINDS = ("sessions", "grids", "images", "labels", "datasets", "models")
DETAIL_KINDS = ("session", "grid", "images", "labels", "dataset", "model")


class ApiError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(f"{status}: {message}")
        self.status = status
        self.message = message


def _seg(x) -> str:
    """One path segment, percent-encoded (an id can never add a path segment)."""
    return urllib.parse.quote(str(x), safe="")


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

    def get(self, path: str, params: dict | None = None):
        """GET <url><path>?<params> with the bearer token -> parsed JSON. Raises ApiError for a non-2xx answer."""
        q = {k: v for k, v in (params or {}).items() if v is not None}
        full = self.url + path + (("?" + urllib.parse.urlencode(q)) if q else "")
        req = urllib.request.Request(full, method="GET", headers={
            "Authorization": f"Bearer {self._token}", "Accept": "application/json",
            "User-Agent": "cryoev-client/1"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                body = r.read()
        except urllib.error.HTTPError as e:
            raw = e.read()
            try:
                msg = json.loads(raw).get("error") or raw.decode("utf-8", "replace")[:300]
            except (ValueError, AttributeError):
                msg = raw.decode("utf-8", "replace")[:300]
            raise ApiError(e.code, str(msg)) from None
        return json.loads(body)

    # ---- the API's routes (portal/app.py BEARER_ROUTES)
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


COMMANDS = {                                  # name -> (Client method, number of arguments)
    "me": ("me", 0), "overview": ("overview", 0), "search": ("search", 1), "list": ("list", 1),
    "item": ("item", 2), "lineage": ("lineage", 2), "graph": ("graph", 0), "mine": ("mine", 0),
    "set": ("prediction_set", 1), "results": ("results_status", 1), "jobs": ("jobs_summary", 0),
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
