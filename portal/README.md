# CJ portal (P1 shell)

One address, one sign-in, one look, for the laptop's own tools: a home page, plus History and Review inside the
shell. P1 is admin only and tailnet only. Nothing about today's links (443 / 8443 / 8445) changes.

```
phone/laptop ── tailnet ── tailscale serve :8446 ──> 127.0.0.1:8770  CJ app (Starlette + uvicorn)
                                                      ├─ /              home: status (live.json) + module cards
                                                      ├─ /cj/*          cj.css, header.js, brand.json, modules.json, me.json
                                                      ├─ /history/*     history.html + live/live.json, from the registry folder
                                                      ├─ /review/*      proxy -> 127.0.0.1:8766 (the real review)
                                                      └─ /login?k=…     key link -> session cookie
Sandbox: NOT under the shell. It stays on its own origin (:8443); the nav links out to it.
```

| File | Holds |
|---|---|
| `app.py` | The app: the Gate (deny by default, security headers, Origin check, access log), routes, History serving |
| `auth.py` | Key links, the private roster, sessions; the `add-key` / `revoke` / `list` tool |
| `proxy.py` | The `/review/` proxy (httpx streaming) |
| `modules.json` | Module manifest: `id, name, route, icon, roles, health` (+ `status`, `nav`, `external`, `desc`) |
| `static/cj.css` | Design tokens (the history page's own names and values, light/dark) and the `cj-` classes |
| `static/header.js` | The top bar (wordmark, nav from `modules.json` with server dots, user chip); hides History's own `#srv` bar |
| `static/brand.json` | `{"name": "CJ", ...}`: renaming the portal = editing this one file |
| `templates/home.html` | Home page template (plain string placeholders, filled by `app.render_home`) |
| `tests/` | `python -m unittest discover -s portal/tests` |

## Install
```
python -m pip install -r requirements-portal.txt     # exact pins: starlette, uvicorn, httpx
```

## Roster (who may sign in)
The roster lives **outside the repo**: `%USERPROFILE%\.cryoev\portal_roster.json` (keys stored only as sha256).
```
python -m portal.auth add-key --id <id> --role admin     # prints the key link ONCE; keep it private
python -m portal.auth list                               # ids, roles, active/revoked (never keys)
python -m portal.auth revoke --id <id>                   # ends that person's sessions on their next request
```
- Open the printed link (`https://<laptop>:8446/login?k=...`) once per device: it sets the session cookie
  (`__Host-cj_session`, 30 days) and redirects home. `--base` changes the address printed in the link.
- Running `add-key` again for the same id replaces the key; the old key and all its sessions stop working.
- Sessions: `%USERPROFILE%\.cryoev\portal\sessions.json`, storing only the sha256 of each session id.

## Run
From the repo root:
```
python -m portal.app --host 127.0.0.1 --port 8770
```
Options: `--registry <dir>` (default: `data_utils/config.py`, i.e. `<CryoAI-laptop>/registry`),
`--review-upstream http://127.0.0.1:8766`, `--review-package <dir>` (its `.review_key` is sent upstream; default: the
newest open revise job on port 8766 in the registry), `--roster <path>`, `--state-dir <dir>`, `--origin <url>`
(an extra allowed Origin for PUT/POST/DELETE: pass `--origin https://<laptop>:8446` if review saves through :8446
answer 403 "Cross-origin request refused", which means `tailscale serve` rewrote the Host header). The console lists what is active at start: the open routes, roster, sessions, History folder, where
the review key came from, and the logging mode. No secrets are printed.

- **Laptop:** a scheduled task `cj-portal` at logon, run as the user, working folder = the repo:
  `python -m portal.app --port 8770`.
- **Tailnet:** `tailscale serve --bg --https=8446 http://127.0.0.1:8770`. Existing 443 / 8443 / 8445 are unchanged.

## Rollback
1. `tailscale serve --https=8446 off`
2. Stop the process (or the `cj-portal` task).

Nothing else is touched: the review servers, the history server, the monitor and their links keep working as before.

## Security (P1)
- **Deny by default.** Only `/login`, `/cj/cj.css`, `/cj/brand.json` and `/healthz` answer without a session
  (`app.OPEN`; a test enumerates the router). No session: 401 ("open your key link"); unknown, expired or revoked: 403.
- **Keys:** compared with `hmac.compare_digest` against every roster entry; the roster is re-checked on every request.
- **Logs:** uvicorn's access log is off; the app's own access log replaces any `k` value with `***`. Keys and session
  ids are never logged.
- **CSRF:** PUT/POST/DELETE/PATCH need `Origin` equal to the portal's own origin (other services on the same host are
  same-site, so SameSite=Lax alone is not enough).
- **Every response:** `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: same-origin`.
- **Redirects:** Starlette's `redirect_slashes` off; every `Location` is relative (uvicorn runs without proxy
  headers, so absolute URLs would come out as `http://`).
- **History:** only `history.html` and `live/live.json` are served, no listing. The served `history.html` gets two
  lines before `</head>` (`cj.css`, deferred `header.js`); the file on disk is never changed.
- **Review proxy:** paths are percent-decoded and checked (`..`, `.`, `//`, backslash, NUL and control characters
  give 400) and forwarded as sent; request headers by allowlist only (inbound `X-CJ-*`, `X-Forwarded-*`,
  `Forwarded`, `Tailscale-*`, `Cf-*`, `Cookie`, `X-Review-Key` never pass); any `k` is removed from the query (on
  the review server a query `k` outranks the header); the real package's key goes as `X-Review-Key`, the user as
  `X-CJ-User`; upstream `Set-Cookie` and hop-by-hop headers are dropped, and the proxy's own HTTP client keeps no
  cookies. Body cap 25 MB (413), upstream timeout 30 s, 502 page when the review server is down.
- **Review keeps its own look** in P1 (no injection into its pages). Sandbox is not proxied: it shares the real
  package's localStorage key (`cryoev-review:<id>`), so on one origin the real review would upload sandbox edits.
  Fix that (a distinct package id or storage key) before ever moving it under the shell.

## Pipelines groundwork (decided; no code in P1)
Future job outputs carry `step`, `step_version`, `params` and `kind`. There is no pipeline schema until a second real
pipeline exists.

## Later
- P1b: other users and roles; the review server checks a gateway-secret header and trusts `X-CJ-User`.
- C3: Cloudflare `access` mode on its own port and middleware (never mixed with key links on one port).
- `live.json` has no backup field yet, so home shows "Last backup: unknown".
