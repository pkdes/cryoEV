# CryoEV portal (shell)

The lab's multi-user web platform for cryo-EM EV analysis: a home page, the review inside the shell, plug-in modules
(Data, Jobs, Predict, Results, Train, Admin) and History as a hidden "classic" page. Roles: admin > advanced > basic.

Two doors, one roster (`~/.cryoev/portal_roster.json`), never mixed on one port:
- **Tailnet** (`--auth key`, default): key links -> session cookie, behind `tailscale serve :8446`.
- **Internet** (`--auth access`, port 8448): Cloudflare Tunnel + Cloudflare Access (email one-time PIN); the portal trusts
  only the verified Access JWT, and the email must be an active roster person's. See "Cloudflare access mode" below.

Nothing about today's links (443 / 8443 / 8445) changes.

```
phone/laptop ── tailnet ── tailscale serve :8446 ──> 127.0.0.1:8770  CryoEV app (Starlette + uvicorn)
                                                      ├─ /              home: status (live.json, admin) + module cards
                                                      ├─ /cj/*          cj.css, header.js, live.js, review.js, brand.json,
                                                      │                 modules.json (filtered by role), me.json
                                                      ├─ /review/       the review shell: the bar + a same-origin iframe
                                                      ├─ /review/app/*  proxy -> 127.0.0.1:8766 (the real review tool)
                                                      ├─ /review/<pred pkg>/(app/*)  shell + proxy -> :8767/<pkg>/ (owner)
                                                      ├─ /history/*     history.html + live/live.json (classic page)
                                                      ├─ /<module>/*    plug-in modules (portal/modules/<id>.py)
                                                      └─ /login?k=…     key link -> session cookie
Sandbox: NOT under the shell. It stays on its own origin (:8443); the nav links out to it.
```

| File | Holds |
|---|---|
| `app.py` | The app: the Gate (deny by default, roles, security headers, Origin check, access log), routes, home, the review shell, History serving |
| `auth.py` | Key links, the private roster (+ sign-in emails), sessions, roles; the `add-key` / `set-email` / `revoke` / `list` tool |
| `access.py` | Cloudflare access mode: config, the team's signing keys, the JWT check (`--auth access`) |
| `proxy.py` | The review proxy (httpx streaming), mounted at `/review/app/` |
| `modules/` | The plug-in loader (`__init__.py`) and one file per module (`jobs.py`, ...) |
| `modules.json` | Module manifest: `id, name, route, icon, roles, health, status, nav` (+ `external`, `desc`, `label`) |
| `static/cj.css` | Design tokens (direction A, light/dark) and the `cj-` classes |
| `static/header.js` | The top bar: grid-atlas wordmark, nav from `modules.json` filtered by role, server dots, user chip |
| `static/live.js` | `CJLive`: the History Live-tab charts as a small library (see below) |
| `static/review.js` | The review shell: mirrors the frame's URL into the address bar; links out of the tool open in the window |
| `static/predict.js` | Predict's script (served by the module as `/predict/predict.js`): status refresh, form counts, one-file-per-request uploads |
| `static/gallery.js` | Gallery's script (served as `/gallery/gallery.js`): marks, filters, full-size viewer with keys and scalebar |
| `zipstream.py` | `stream_zip(entries)`: streamed zip files for the portal (Gallery downloads) |
| `static/brand.json` | `{"name": "CryoEV", ...}`: renaming the portal = editing this one file |
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
  a missing entry, `route: null` or bad roles leave the default (fail closed); a core prefix can be set only by its
  own entry (`id` `review` / `history`). A `modules.json` that can't be read or parsed, or has two entries on one
  prefix, **stops the portal** at start.
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
  `/review/app/?...` link: `Sec-Fetch-Dest: document`, or HTML asked with a Referer that isn't the shell) answers 302
  `/review/?<query minus k>`; the frame's own loads, requests without a Referer and every other request reach the
  tool. If the frame ever shows `/review/` itself, the window is left alone (no loop).
- **Prediction review packages** (`pred_ps_<8 digits>_<6 hex>`): `/review/<pkg_id>/` is the same shell, and
  `/review/<pkg_id>/app/<path>` is proxied to the multi-package review server (`--pred-upstream`, default
  `http://127.0.0.1:8767`) at `/<pkg_id>/<path>`, with that package's own `.review_key` (from `--review-root`,
  default `<CryoAI-laptop>/review`; the folder is found with `annotation/multi_review_server.package_dir`: exact
  name, complete package). Every proxy rule applies; a Location inside the package stays under its prefix, anything
  else goes to its root. **Who:** the package's owner (`review_package.json`) or an admin (`ctx.can_see`), so basic
  and advanced users open their own; the admin-only `/review` rule covers only the real package. An unknown id, a
  package without a key, or one that isn't yours all get the same 404; so does anything else under `/review/`.
- `X-Frame-Options`: `SAMEORIGIN` for `/review/*`, `DENY` everywhere else.

## Predict (`/predict/`, every role)
The full contract is the module docstring in `modules/predict.py`. In short:
- **Where the images come from.** (1) A processed session: the session folders under the scan roots
  (`CJ_PREDICT_SCAN_ROOTS`, default the processing root) that have `3. Processed images/3d. Tiles no scalebar*`, plus
  the admin's hand list (`CJ_PREDICT_SOURCES`), with **Admin > Predict sessions** (hide, rename). Sessions are named by
  their full registry id (`20261027`, `20260827-2`). (2) Uploaded tiles (8-bit PNG/JPEG, up to 1000 files of 100 MB).
  (3) Raw montages converted on the laptop (below).
- **Every analysis:** pick samples, see each sample's nm/px and tick "I confirm the pixel size" (never defaulted), at
  most 1000 tiles. Basic users get the production model only and at most 3 analyses waiting or running.
- **Uploads go one file per request** (a big file in pieces of at most 32 MB): a Cloudflare free tunnel caps a request
  at about 100 MB. The page's script opens an upload session (`POST /predict/api/upload` with the file list: fields,
  caps and free disk checked before a byte is sent, the declared size held on the disk), sends each file, then asks
  for "create". A dropped connection is retried; pressing the button again resumes where it stopped. A refusal (cap,
  disk, content, fields) keeps nothing; a session never finished is removed after an hour, with its disk hold. Without
  JavaScript the tile form posts as one request, for at most 90 MB in total (else refused, saying why).
- **Raw conversion** (`CJ_PROCESSING_REPO` = the lab's processing repo, read only; unset = off):
  - *Upload raw images* (admin, advanced): each `.mrc` with its `.mrc.mdoc`, or an `.idoc` with its `.tif` tiles, plus
    an optional `sample_key.csv`; a session id (`YYYYMMDD` or `YYYYMMDD-n`) and a recipe.
  - *Pick a raw session* (admin): a session folder under the scan roots with no 3d tiles yet; tick montages. The
    folder is only read.
  - Recipe: `default` (convert_raw's default, 20261012, with a warning on the page) or one of the repo's session
    recipes that `jobs/convert_raw.py`'s static check accepts. GPU only when ticked and free; the CPU otherwise.
  - The portal runs `python -m jobs.convert_raw` detached, one conversion at a time (the others wait, queued), into
    `<predict root>/raw_tiles/<session id>_<6 hex>/` (`convert.json`, `status.json` queued/running/done/failed with the
    converter's JSON result and warnings, `convert.log`, `tiles/`). When done, `tiles/` is a Predict source seen only
    by its owner and the admin.
  - Uploaded raw files are kept `CJ_RAW_KEEP_DAYS` (default 7; 0 = removed at once) after the conversion ends, done
    or failed; the admin can delete them sooner. At most 2 conversions per person waiting or running; a conversion
    needs about twice its raw size free on the disk.
  - Admin: **Stop** (kills the converter), **Retry** (a failed one whose inputs remain). A converter running over 2 h,
    or with a log silent for 20 min, is stopped by itself. Its process is known by pid and start time, so a reused pid
    is never mistaken for it.
  - Every `[Image = …]` in an uploaded `.idoc` / `.mrc.mdoc` must be a bare file name that was uploaded with it (no
    drive, folder, UNC share or `..`); `jobs/convert_raw.py` checks again before opening any tile.
  - **`CJ_RAW_ROOT`** (env; `cfg.raw_root` wins) moves the whole raw pipeline to another folder, e.g. a NAS or an
    external drive: the raw-upload staging (`<raw root>/.portal`) and `<raw root>/raw_tiles/`. The 2x-free check runs on
    that drive. Unset = the predict root, as before. Conversions already under `<predict root>/raw_tiles` stay listed
    and swept (retention, stale scratch). A prediction set made from a conversion copies its tiles (no hard link across
    drives). Changing or unsetting `CJ_RAW_ROOT` stops listing and sweeping the old raw root's conversions and
    `.portal`: move them first, or set it back. On Windows keep `CJ_RAW_ROOT` short (long paths can fail) or enable
    LongPathsEnabled. **Admin > Health** shows free GB for both drives and warns under 20 GB.
- Basic users get 403 on everything raw. Everything shown is escaped; paths and raw errors are on admin-only lines.

## Gallery (`/gallery/`, admin + advanced)
The full contract is the module docstring in `modules/gallery.py`. In short:
- Opens the person's own raw conversions (admin: Mine / All; owner or admin only, a foreign cid is the same 404 as a
  missing one): montages by grid and sample (from `montages/layout.json`, else grouped from `tiles/MANIFEST.csv`) with
  clickable tile boxes, a tile strip, a full-size viewer (keys: arrows, K/M/R, 0-5, T, N, Esc), filters, and
  `/gallery/compare?c=<cid>&c=<cid>` side by side. Page images are thumbnails; full size loads on click.
- Marks (keep / maybe / reject, reasons, stars, note) append to `raw_tiles/<cid>/gallery/events.jsonl` under an OS lock;
  the latest value per (key, field) wins. The POST needs the Gate's Origin check and `X-Requested-With: gallery`.
- `/gallery/<cid>/report` summarizes the run (acquisition, samples per grid, counts and reject reasons, starred images,
  notes). `report.zip` is a standalone offline HTML page with its images; `images.zip` is tiles + montages + MANIFEST
  + acquisition.json. Both stream through `portal/zipstream.py` (no temp files, ZIP_STORED images, zip64); at most 2
  per person at once (429 otherwise).

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
  cookies. An upstream `Location` is kept under `/review/app/`; any other absolute URL (a scheme that isn't the
  upstream), one starting with `//`, or one with dot segments becomes `/review/app/`. Body cap 25 MB (413), upstream timeout 30 s, 502 page when the review server is down.
- **Sandbox is not proxied:** it shares the real package's localStorage key (`cryoev-review:<id>`), so on one origin
  the real review would upload sandbox edits. Fix that (a distinct package id or storage key) before ever moving it
  under the shell.

## Cloudflare access mode (`--auth access`, port 8448)
For lab members on the internet: a Cloudflare Tunnel (`cloudflared`) in front, guarded by Cloudflare Access
(one-time PIN email login). The same app; the default `--auth key` (above) is unchanged, and the two never share a port.
```
python -m portal.app --auth access --port 8448 [--access-config <path>] [--state-dir ...] [--roster ...]
```
- **Config:** `%USERPROFILE%\.cryoev\cloudflare_access.json` = `{"team": "example-team", "aud": "<Access app AUD
  tag>", "host": "portal.example.org"}` (outside the repo; `--access-config` picks another file). Env
  `CJ_ACCESS_TEAM`, `CJ_ACCESS_AUD`, `CJ_ACCESS_HOST` override it. Missing or malformed = the portal refuses to start.
- **Needs** `PyJWT` + `cryptography` (pinned in `requirements-portal.txt`); key mode never imports them.
- **Binds** `127.0.0.1` and `[::1]` on the port (never a wildcard); `--host` is ignored. Point `cloudflared` at
  `http://localhost:8448`.
- **People:** the roster's `email` field. Set it on **Admin > People** (the add form, or Set / Change email per person)
  or `python -m portal.auth set-email --id <id> --email <address>` (`add-key --email` too; re-keying keeps it).
  Emails are lower-cased and unique among active people.

**Security model**
- Only the `Cf-Access-Jwt-Assertion` **header** counts (no cookie fallback; `CF_Authorization` and the key-link
  session cookie are ignored). PyJWT, **RS256 only** (`alg=none`, HS256 and anything else refused before any key is
  used); `aud` = the configured AUD, `iss` = `https://<team>.cloudflareaccess.com`, `exp` required, `exp`/`nbf` with
  60 s leeway, an `email` claim required.
- **Keys** from `https://<team>.cloudflareaccess.com/cdn-cgi/access/certs`, fetched in a worker thread with a 5 s
  timeout, cached, refetched on an unknown `kid` (or after an hour) at most once per 60 s. No usable keys (never
  fetched, fetch failing for 6 h, or a certs answer without RS256 keys) = **everyone refused**.
- **Who:** the JWT email (lower-cased) must be the email of exactly one active roster person (`Roster.usable_access`:
  not revoked, a known role, an email; no key needed). Revoking someone shuts them out at their next request even
  with a valid JWT. Cloudflare's own policy is a first fence; the roster is the second.
- **Admins** get 403 "Admin is available on Tailscale only" unless `CJ_ACCESS_ADMIN=on` (default off; only the word
  `on`). Advanced and basic users get their roles as on the tailnet.
- `/login?k=` answers 404; sandbox links are hidden (their host:port addresses don't exist through the tunnel); the
  bar gets **Sign out** (`/cdn-cgi/access/logout`).
- **CSRF:** writes need `Origin` = `https://<host>` exactly (or an `--origin`), whatever the Host header says. Admin's
  form token and one-time messages are bound to the verified user id (key mode: the session cookie).
- The review proxies still strip every client `X-CJ-*` and `Cf-*` header (so the JWT never goes upstream) and set
  `X-CJ-User` to the verified user.
- **Logs:** refusals with a reason and `Cf-Connecting-IP` (IP-shaped values only); the token is never logged. The
  access log adds `cf=<ip>`.

## Uptime
- **Portal process down:** the laptop's job monitor can watch it. Add `--portal http://127.0.0.1:8770/healthz` (key mode) and `--portal http://127.0.0.1:8448/healthz` (Cloudflare access mode) to the `monitor` service's argv (the example config has both) and set env `CRYOEV_NTFY_TOPIC` for it. After 2 failed checks it sends one ntfy, and one more on recovery (docs/JOBS.md, Portal uptime probes).
- **Laptop asleep, off or offline:** the monitor runs on that laptop, so it cannot report this. Turn on the **Cloudflare Tunnel Health Alert** (Cloudflare dashboard -> Zero Trust -> Notifications, email destination: `<your email>`) for the tunnel that serves `<portal hostname>`.
- **Power:** Windows Settings -> System -> Power -> set "When plugged in, put my device to sleep after" to **Never**, so the tunnel stays up.
- **Service entry** (private `services.json`): give the portal service `"health": "/healthz"` so `services.py status` shows `ok` only for a 200 with `{"ok": true}`.

## Pipelines groundwork (decided; no code yet)
Future job outputs carry `step`, `step_version`, `params` and `kind`. There is no pipeline schema until a second real
pipeline exists.

## Later
- The review server checks a gateway-secret header and trusts `X-CJ-User`.
- `live.json` has no backup field yet, so home shows "Last backup: Unknown".
