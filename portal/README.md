# CJ portal (shell)

One address, one sign-in, one look, for the laptop's own tools: a home page, the review inside the shell, plug-in
modules (Jobs, and later Predict, Results, ...) and History as a hidden "classic" page. Tailnet only. Nothing about
today's links (443 / 8443 / 8445) changes.

```
phone/laptop ── tailnet ── tailscale serve :8446 ──> 127.0.0.1:8770  CJ app (Starlette + uvicorn)
                                                      ├─ /              home: status (live.json, admin) + module cards
                                                      ├─ /cj/*          cj.css, header.js, live.js, review.js, brand.json,
                                                      │                 modules.json (filtered by role), me.json
                                                      ├─ /review/       the review shell: the bar + a same-origin iframe
                                                      ├─ /review/app/*  proxy -> 127.0.0.1:8766 (the real review tool)
                                                      ├─ /history/*     history.html + live/live.json (classic page)
                                                      ├─ /<module>/*    plug-in modules (portal/modules/<id>.py)
                                                      └─ /login?k=…     key link -> session cookie
Sandbox: NOT under the shell. It stays on its own origin (:8443); the nav links out to it.
```

| File | Holds |
|---|---|
| `app.py` | The app: the Gate (deny by default, roles, security headers, Origin check, access log), routes, home, the review shell, History serving |
| `auth.py` | Key links, the private roster, sessions, roles; the `add-key` / `revoke` / `list` tool |
| `proxy.py` | The review proxy (httpx streaming), mounted at `/review/app/` |
| `modules/` | The plug-in loader (`__init__.py`) and one file per module (`jobs.py`, ...) |
| `modules.json` | Module manifest: `id, name, route, icon, roles, health, status, nav` (+ `external`, `desc`, `label`) |
| `static/cj.css` | Design tokens (direction A, light/dark) and the `cj-` classes |
| `static/header.js` | The top bar: grid-atlas wordmark, nav from `modules.json` filtered by role, server dots, user chip |
| `static/live.js` | `CJLive`: the History Live-tab charts as a small library (see below) |
| `static/review.js` | The review shell: mirrors the frame's URL into the address bar; links out of the tool open in the window |
| `static/brand.json` | `{"name": "CJ", ...}`: renaming the portal = editing this one file |
| `templates/base.html` | The page every shell page uses (`{{title}} {{active}} {{body}}`; the bar placeholder) |
| `templates/home.html` | Home's body (plain `{{name}}` placeholders, filled by `app.render_home`) |
| `tests/` | `python -m unittest discover -s portal/tests` |

## Install
```
python -m pip install -r requirements-portal.txt     # exact pins: starlette, uvicorn, httpx
```

## Roster (who may sign in)
The roster lives **outside the repo**: `%USERPROFILE%\.cryoev\portal_roster.json` (keys stored only as sha256).
```
python -m portal.auth add-key --id <id> --role admin     # admin | advanced | basic; prints the key link ONCE
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
newest open revise job on port 8766 in the registry), `--roster <path>`, `--state-dir <dir>`, `--modules <path>` (the
manifest; default `portal/modules.json`), `--origin <url>` (an extra allowed Origin for PUT/POST/DELETE: pass
`--origin https://<laptop>:8446` if review saves through :8446 answer 403 "Cross-origin request refused", which means
`tailscale serve` rewrote the Host header). The console lists what is active at start: the open routes, roster,
sessions, History folder, where the review key came from, the role rules and the logging mode. No secrets are printed.

- **Laptop:** a scheduled task `cj-portal` at logon, run as the user, working folder = the repo:
  `python -m portal.app --port 8770`.
- **Tailnet:** `tailscale serve --bg --https=8446 http://127.0.0.1:8770`. Existing 443 / 8443 / 8445 are unchanged.

## Rollback
1. `tailscale serve --https=8446 off`
2. Stop the process (or the `cj-portal` task).

Nothing else is touched: the review servers, the history server, the monitor and their links keep working as before.

## Roles
`admin` > `advanced` > `basic` (`auth.py ROLES`). The Gate answers 403 when a role may not enter a path:
- **Plug-in modules** declare `ROLES` in their own file; the Gate enforces them for their whole `PREFIX`.
- **Core pages** (`/review`, `/history`) have built-in defaults in `app.py` (`CORE_ROLES`: admin only). A
  `modules.json` entry replaces a default only when it is valid (a local route and a non-empty list of known roles);
  a missing entry, `route: null` or bad roles leave the default (fail closed). A `modules.json` that can't be read or
  parsed **stops the portal** at start.
- The rules are read **once at start**: after changing roles in `modules.json`, restart the portal. Display (home
  cards, `/cj/modules.json`) is read per request, so it can be ahead of the rules until then; access never is.
- Home's status cards and the nav's server dots come from History's `live.json`, so only roles that may open
  `/history` see them (admin today).

## Navigation and modules
- Nav (from `modules.json`, filtered by the user's role): Home · Data · Jobs · Review · Predict · Results · Train ·
  Admin · Sandbox. `status: "coming"` = greyed, not a link (nav and home cards). Sandbox is `external` (its own port).
- History has `nav: false`: not in the nav, still at `/history/`, linked from home's footer as "Classic history".
- **Plug-in modules** (the private contract doc, section 1): `portal/modules/<id>.py` defines `PREFIX`, `ROLES` and
  `routes(ctx)`; pages render through `ctx.page(...)` (base.html, the bar, cj.css). Add the module to `modules.json`
  for the nav. Module CSS uses only the cj.css tokens and `cj-<module>-` classes.

## Design (direction A, "Vitreous")
- `cj.css` sets A's light and dark values under the **frozen token names** (`--bg --fg --mut --card --line --acc
  --accbg --prod --prodbg --warn --warnbg --dim`; new tokens may be added, none renamed or removed), plus `--s1..--s4`
  (chart series), `--hot/--hot2` (the gradient "eVHarmony" label on the Review card only) and `--cj-*`.
- The tokens are scoped to `.cj-page` (pages the shell renders) and `.cj-bar`, never `:root`: History keeps its own
  look; only the bar there wears the shell's tokens.
- Fonts: a system stack (IBM Plex first, if the device has it). No web fonts; nothing is fetched from elsewhere.
- Wordmark: the grid-atlas mark (SVG, header.js) + `brand.json`'s `wordmark`.
- The bar is a `div role="banner"` of fixed height (52 px; 88 px on a phone, two rows; 44 px on the review shell's
  phone layout). `base.html` holds an empty placeholder of that height and cj.css holds the space on History until
  header.js inserts the bar, so pages never jump.

## The review in the shell
- `GET /review/` = the shell page: the bar + a full-height iframe of `/review/app/`. The shell's query (`?img=`,
  `?mode=`, `?who=`; any `k` removed) goes into the iframe's `src`. `review.js` mirrors the frame's URL changes into the
  address bar (`history.replaceState`), so reload, bookmarks and Back work; a same-origin link out of the tool opens in
  the window (only ever to an absolute URL of this origin).
- `/review/app/<path>` is proxied to `<upstream>/<path>`. A top-level document GET of the tool's page (a copied
  `/review/app/?...` link) answers 302 `/review/?<query minus k>`; the frame's own loads and every other request reach
  the tool.
- `/review/<anything else>` answers 404: kept free for the multi-package route `/review/<pkg_id>/` (wave 2:
  `app.review_mount(base, proxy)` mounts a shell + app pair; package ids are never `app`).
- `X-Frame-Options`: `SAMEORIGIN` for `/review/*`, `DENY` everywhere else.

## Live charts (`/cj/live.js`)
`CJLive.render(el, liveJson, {compact})` draws a `live.json` snapshot (jobs/monitor.py): `compact: false` = every queue
as History's Live tab (watcher, running job with loss / mAP charts, queue, finished); `compact: true` = one small block
per running job (epoch bar, ETA, last val mAP). Also `lineChart`, `runBox`, `queueBox`, `snapshot`, `dur`, `redraw`.
Markup hook without inline script: `<div data-cj-live="compact" data-cj-live-json="<id of a JSON script>"
data-cj-live-src="/history/live/live.json" data-cj-live-every="30">`. History keeps its own copy of the code.

## Security
- **Deny by default.** Only `/login`, `/cj/cj.css`, `/cj/brand.json` and `/healthz` answer without a session
  (`app.OPEN`; a test enumerates the router). No session: 401 ("open your key link"); unknown, expired or revoked: 403.
  Then the role rules above (403).
- **Keys:** compared with `hmac.compare_digest` against every roster entry; the roster is re-checked on every request.
- **Logs:** uvicorn's access log is off; the app's own access log replaces any `k` value with `***`. Keys and session
  ids are never logged.
- **CSRF:** PUT/POST/DELETE/PATCH need `Origin` equal to the portal's own origin (other services on the same host are
  same-site, so SameSite=Lax alone is not enough).
- **Every response:** `X-Content-Type-Options: nosniff`, `X-Frame-Options` (above), `Referrer-Policy: same-origin`.
- **Redirects:** Starlette's `redirect_slashes` off; every `Location` is relative (uvicorn runs without proxy
  headers, so absolute URLs would come out as `http://`).
- **History:** only `history.html` and `live/live.json` are served, no listing. The served `history.html` gets two
  lines before `</head>` (`cj.css`, deferred `header.js`); the file on disk is never changed.
- **Review proxy:** paths are percent-decoded and checked (`..`, `.`, `//`, backslash, NUL and control characters
  give 400) and forwarded as sent; request headers by allowlist only (inbound `X-CJ-*`, `X-Forwarded-*`,
  `Forwarded`, `Tailscale-*`, `Cf-*`, `Cookie`, `X-Review-Key` never pass); any `k` is removed from the query (on
  the review server a query `k` outranks the header); the real package's key goes as `X-Review-Key`, the user as
  `X-CJ-User`; upstream `Set-Cookie` and hop-by-hop headers are dropped, and the proxy's own HTTP client keeps no
  cookies. An upstream `Location` is kept under `/review/app/`; one starting with `//` or with dot segments becomes
  `/review/app/`. Body cap 25 MB (413), upstream timeout 30 s, 502 page when the review server is down.
- **Sandbox is not proxied:** it shares the real package's localStorage key (`cryoev-review:<id>`), so on one origin
  the real review would upload sandbox edits. Fix that (a distinct package id or storage key) before ever moving it
  under the shell.

## Pipelines groundwork (decided; no code yet)
Future job outputs carry `step`, `step_version`, `params` and `kind`. There is no pipeline schema until a second real
pipeline exists.

## Later
- The review server checks a gateway-secret header and trusts `X-CJ-User`.
- Cloudflare `access` mode on its own port and middleware (never mixed with key links on one port).
- `live.json` has no backup field yet, so home shows "Last backup: Unknown".
