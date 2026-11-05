"""
The /review/ proxy (P1): the CJ portal's only proxied module. httpx streaming both ways.

    /review/<path>?<query>  ->  <upstream>/<path>?<query without k>

- Paths: the raw path is percent-decoded (repeatedly, so %252e is caught too) and every segment checked;
  `..`, `.`, empty segments (`//`), backslashes, NUL and other control characters are rejected (400). A clean path
  is forwarded byte for byte as the browser sent it.
- Request headers: an allowlist only (Accept*, Content-Type, Content-Length, Range, If-*, User-Agent). Inbound
  Cookie, X-CJ-*, X-Forwarded-*, Forwarded, Tailscale-*, Cf-* and X-Review-Key never pass. Added: X-Review-Key (the
  real package's key, read once at start) and X-CJ-User (for P1b). Any `k` in the query is dropped, since on the
  review server a `k` in the query outranks the header.
- Response headers: hop-by-hop headers (and any named in Connection), Set-Cookie, Server and Date are dropped; a
  Location pointing at the upstream is made relative to /review/.
- Limits: request body <= MAX_BODY (413 above), upstream timeout TIMEOUT_S; 502 page when the upstream is down or
  times out.
"""
from __future__ import annotations

import html
from http.cookiejar import CookieJar, DefaultCookiePolicy
from urllib.parse import unquote, unquote_plus

import httpx
from starlette.background import BackgroundTask
from starlette.requests import Request
from starlette.responses import HTMLResponse, Response, StreamingResponse

PREFIX = "/review"
TIMEOUT_S = 30.0
MAX_BODY = 25 * 2**20
FORWARD = {"accept", "accept-language", "accept-encoding", "content-type", "content-length", "range", "user-agent"}
FORWARD_PREFIX = ("if-",)
STRIP = {"cookie", "forwarded", "x-review-key", "authorization"}
STRIP_PREFIX = ("x-cj-", "x-forwarded-", "tailscale-", "cf-")
HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "proxy-connection", "te", "trailer",
       "trailers", "transfer-encoding", "upgrade"}
DROP_RESPONSE = HOP | {"set-cookie", "set-cookie2", "server", "date"}
BODY_METHODS = {"PUT", "POST", "DELETE", "PATCH"}


class BadPath(ValueError):
    pass


def clean_subpath(raw: str) -> str:
    """The path after /review, as sent (e.g. "/api/edits"). Raises BadPath for traversal or odd characters."""
    if not raw.startswith("/"):
        raise BadPath("no leading slash")
    dec = raw
    for _ in range(4):
        nxt = unquote(dec, errors="strict")
        if nxt == dec:
            break
        dec = nxt
    else:
        raise BadPath("encoded too many times")
    for s in (raw, dec):
        if "\\" in s or any(ord(c) < 0x20 or ord(c) == 0x7F for c in s):
            raise BadPath("backslash or control character")
    segs = dec.split("/")[1:]
    for i, seg in enumerate(segs):
        if seg in (".", ".."):
            raise BadPath("dot segment")
        if seg == "" and i != len(segs) - 1:
            raise BadPath("empty segment")
    return raw


def strip_k(query: str) -> str:
    """The query string without any `k` parameter; everything else kept byte for byte."""
    keep = [p for p in query.split("&") if p and unquote_plus(p.split("=", 1)[0]) != "k"]
    return "&".join(keep)


def request_headers(raw_headers, key: str | None, user: str) -> list:
    out = []
    for name, value in raw_headers:
        n = name.decode("latin-1").lower()
        if n in STRIP or n.startswith(STRIP_PREFIX):
            continue
        if n in FORWARD or n.startswith(FORWARD_PREFIX):
            out.append((n, value.decode("latin-1")))
    if not any(n == "accept-encoding" for n, _ in out):
        out.append(("accept-encoding", "identity"))     # else httpx asks for gzip on the browser's behalf
    if key:
        out.append(("x-review-key", key))
    out.append(("x-cj-user", user))
    return out


def rewrite_location(value: str, upstream: str) -> str:
    if value.startswith(upstream):
        value = value[len(upstream):] or "/"
    if value.startswith("/") and not value.startswith("//"):
        return PREFIX + value
    return value


def response_headers(headers: httpx.Headers, upstream: str) -> list:
    named = {t.strip().lower() for v in headers.get_list("connection") for t in v.split(",") if t.strip()}
    out = []
    for k, v in headers.multi_items():
        lk = k.lower()
        if lk in DROP_RESPONSE or lk in named:
            continue
        if lk == "location":
            v = rewrite_location(v, upstream)
        out.append((lk.encode("latin-1"), v.encode("latin-1")))
    return out


def error_page(status: int, title: str, text: str) -> HTMLResponse:
    body = (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title>'
            f'<link rel="stylesheet" href="/cj/cj.css"></head><body class="cj-page"><main class="cj-main">'
            f'<h1>{html.escape(title)}</h1><p>{html.escape(text)}</p><p><a href="/">Home</a></p></main></body></html>')
    return HTMLResponse(body, status_code=status, headers={"Cache-Control": "no-store"})


class ReviewProxy:
    def __init__(self, upstream: str, key: str | None, timeout: float = TIMEOUT_S, max_body: int = MAX_BODY):
        self.upstream = upstream.rstrip("/")
        self.key = key
        self.timeout = timeout
        self.max_body = max_body
        self.client: httpx.AsyncClient | None = None

    async def start(self):
        # cookies=a jar that accepts none: httpx's default jar would keep an upstream Set-Cookie and replay it on
        # every later request, for every user (caught by test_upstream_sees_key_and_user_but_no_cookie_or_k)
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(self.timeout), follow_redirects=False,
                                        trust_env=False, cookies=CookieJar(policy=DefaultCookiePolicy(allowed_domains=[])))

    async def close(self):
        if self.client is not None:
            await self.client.aclose()
            self.client = None

    async def _body(self, request: Request):
        """-> (content for httpx, error response or None). Streamed when Content-Length is given; a chunked body
        is read up to the cap first (the review server needs a Content-Length)."""
        method = request.method
        cl = request.headers.get("content-length")
        if cl is not None:
            try:
                n = int(cl)
            except ValueError:
                return None, error_page(400, "Bad request", "Bad Content-Length.")
            if n < 0:
                return None, error_page(400, "Bad request", "Bad Content-Length.")
            if n > self.max_body:
                return None, error_page(413, "Too large", f"Request bodies are capped at {self.max_body >> 20} MB.")
            return (request.stream() if n else b""), None
        if method in BODY_METHODS and "chunked" in request.headers.get("transfer-encoding", "").lower():
            buf = bytearray()
            async for chunk in request.stream():
                buf += chunk
                if len(buf) > self.max_body:
                    return None, error_page(413, "Too large",
                                            f"Request bodies are capped at {self.max_body >> 20} MB.")
            return bytes(buf), None
        return None, None

    async def handle(self, request: Request, user: str) -> Response:
        scope = request.scope
        raw = (scope.get("raw_path") or scope["path"].encode("utf-8")).decode("latin-1")
        if not raw.startswith(PREFIX + "/"):
            return error_page(400, "Bad request", "Bad path.")
        try:
            sub = clean_subpath(raw[len(PREFIX):])
        except (BadPath, UnicodeDecodeError):
            return error_page(400, "Bad request", "That path is not allowed.")
        query = strip_k(scope.get("query_string", b"").decode("latin-1"))
        url = self.upstream + sub + (f"?{query}" if query else "")
        content, err = await self._body(request)
        if err is not None:
            return err
        headers = request_headers(scope["headers"], self.key, user)
        if isinstance(content, bytes) and content and not any(n == "content-length" for n, _ in headers):
            headers.append(("content-length", str(len(content))))
        if self.client is None:
            await self.start()
        try:
            req = self.client.build_request(request.method, httpx.URL(url), headers=headers, content=content)
            resp = await self.client.send(req, stream=True)
        except httpx.TimeoutException:
            return error_page(502, "Review is not answering",
                              f"The review server did not answer within {self.timeout:g} s. Try again shortly.")
        except httpx.HTTPError:
            return error_page(502, "Review is down",
                              "The review server is not running. Start it on the laptop, then reload this page.")
        out_headers = response_headers(resp.headers, self.upstream)
        if request.method == "HEAD":
            await resp.aclose()
            r = Response(b"", status_code=resp.status_code)
        else:
            async def body():
                try:
                    async for chunk in resp.aiter_raw():
                        yield chunk
                finally:                                 # also when the browser goes away mid-stream
                    await resp.aclose()
            r = StreamingResponse(body(), status_code=resp.status_code, background=BackgroundTask(resp.aclose))
        r.raw_headers = out_headers
        return r
