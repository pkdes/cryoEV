# Agent API

The portal lets a person's own AI agent (for example Claude Code) **read** what that person can read: data and
lineage, their prediction sets, results status, and the jobs summary. With a **read + write** token it can also do a
few things for that person: upload images, start predictions, compute results and send feedback (see
[Writes](#writes-read--write-tokens)). It can't touch admin, review, Apply, Box, Roboflow or training.

- **Tailnet only for now.** Use the portal's Tailscale address. The Cloudflare address refuses tokens with
  `403 "API is tailnet-only for now"`.
- **Two scopes.** A read token opens only the read routes below (`GET` only). A read + write token also opens the
  write routes.
- **Attributed.** Every request is the person's, done by their agent. The audit log (Admin > Recent actions) shows
  "claude for alice" with the AI tag, on the door "API (verified)". The portal checked the token, so the line is
  not marked "unverified".

## Getting a token

1. Sign in to the portal and open **My account** (`/account/`; also linked from About).
2. Under **New token**, enter the agent's name (lower-case, such as `claude`), pick an expiry (7, 30 or 90 days) and
   the access: **Read only** (the default) or **Read + write**.
3. Copy the token from the next page. **It is shown once.** The portal stores only its sha256.

Limits:
- at most 10 live tokens per person;
- read tokens: 90 days at most; read + write tokens: **30 days at most**, also checked on every request (a store
  edited by hand to give one longer is refused);
- tokens made before the write scope existed stay read-only.

To revoke a token:
- you revoke your own on My account;
- an admin revokes anyone's on Admin > People (each person's card lists their tokens);
- revoking a person revokes all their tokens;
- making a new key link for a person (re-keying them) revokes all their tokens too.

Revoked and expired tokens are dropped from the store 30 days later. The ids `cli` and `-` are reserved: no person can have them.

**CLI fallback** (on the portal computer):

```
python -m portal.auth mint-token --id alice --agent claude [--days 30] [--scope read|write]
python -m portal.auth list-tokens [--id alice]
python -m portal.auth revoke-token --token-id tk_0123456789ab
```

The CLI lines go to the same audit log, with actor `cli`.

## Calling it

Send the token as a header: `Authorization: Bearer cev_...`.

The one-file client uses only the Python standard library. Download it from the portal at `/cj/cryoev_client.py`
(signed in), or take `client/cryoev_client.py` from this repo.

```
CRYOEV_URL=https://<portal tailnet address>
CRYOEV_TOKEN=cev_...                      (from My account; never commit it)
python cryoev_client.py me
python cryoev_client.py overview
python cryoev_client.py list sessions
python cryoev_client.py mine
```

```python
from cryoev_client import Client, ApiError
c = Client()                 # reads CRYOEV_URL / CRYOEV_TOKEN
c.me()                       # {"id": "alice", "role": "advanced", "via": "claude", "auth": "bearer", "scope": "read"}
c.jobs_summary()
```

## Read routes (the whole read allowlist)

These are existing portal routes, so they return the same JSON the pages use. The list lives in `portal/app.py`
as `BEARER_ROUTES`; it matches exact route templates, never path prefixes.

| Route | Client method | Who can read it |
|---|---|---|
| `GET /cj/me.json` | `me()` | everyone |
| `GET /data/api/overview` | `overview()` | advanced, admin |
| `GET /data/api/search?q=` | `search(text)` | advanced, admin |
| `GET /data/api/list/{kind}` | `list(kind)` | advanced, admin |
| `GET /data/api/{kind}/{eid}` | `item(kind, eid)` | advanced, admin |
| `GET /data/api/lineage/{kind}/{eid}` | `lineage(kind, eid)` | advanced, admin |
| `GET /data/api/lineage/graph` | `graph(focus, scope)` | advanced, admin |
| `GET /predict/api/mine` | `mine()` | everyone (own sets) |
| `GET /predict/api/set/{ps}` | `prediction_set(ps)` | the set's owner |
| `GET /results/{ps}/api/status` | `results_status(ps)` | the set's owner |
| `GET /jobs/api/summary` | `jobs_summary()` | advanced, admin |

How access is decided:
- **Role cap.** The token acts with the person's role, capped at **advanced**: an admin's agent is not an admin.
  It never reaches Admin, and it sees only its person's own prediction sets and results.
- **Normal rules.** The portal's module role rules and owner checks then apply exactly as in the browser.
- **Bearer only.** A request with a bearer header is judged on the token alone. Cookies are ignored (no fallback),
  and the handler never sees the cookie or the token.

## Writes (read + write tokens)

The list lives in `portal/app.py` as `BEARER_WRITE_ROUTES` (exact route templates again). A read token on any of them
gets `403 "This token is read-only..."`.

| Route | Client method | What it does |
|---|---|---|
| `POST /predict/api/upload` | `upload(files, ...)` | opens an upload session (tiles: 8-bit PNG/JPEG; raw: advanced) |
| `POST /predict/api/upload/{uid}/file?name=&offset=` | (inside `upload`) | one piece of one file |
| `GET /predict/api/upload/{uid}` | `upload_state(uid)` | what has arrived |
| `POST /predict/api/upload/{uid}/create` | `upload(..., create=True)` | starts the analysis (tiles) or the conversion (raw) |
| `POST /predict/api/upload/{uid}/cancel` | `cancel_upload(uid)` | gives the upload up |
| `GET /predict/api/preparing/{tid}` | `preparing(tid)`, `wait(tid)` | the new set's progress; `href` names the set when done |
| `GET /predict/new/session/{key}/?json=1` | `session_form(key)` | a session's form data: `sig`, its samples, the models you may pick (an agent gets 404 without `json=1`) |
| `POST /predict/new/session/{key}/` | `predict_session(key, ...)` | starts an analysis of a session's tiles |
| `GET /predict/api/convert/{cid}` | `convert(cid)` | a raw upload's conversion: state, tiles, and `key` when done |
| `POST /results/{ps}/compute` | `compute(ps)` | computes a finished set's results; poll `results_status(ps)`; `409` while the predictions aren't finished |
| `POST /feedback/` | `feedback(text, kind)` | a bug report, feature idea or question |

```python
c = Client()                                  # a read + write token
up = c.upload(["a.png", "b.png", "c.png"], sample_label="EV prep", nm_per_px=1.47, model="<model id>",
             confirm_nm=True)
done = c.wait(up["tid"])                      # {"state": "done", "ps": "ps_20261113_0a1b2c", ...}
c.predict_session("<key>", nm={"S1": 1.47}, confirm_nm=True)
c.compute(done["ps"]); c.results_status(done["ps"])
c.feedback("The upload page ...", kind="bug") # {"id": 12, ...}
```

**The pixel size.** `upload` and `predict_session` refuse without `confirm_nm=True`: it confirms the pixel size, as
the page's tick box does. The person is recorded as the one who confirmed it (`nm_confirmed_by`) and the agent as the
way it came (`nm_confirmed_via` in `request.json` and the registry record), so use only the value the person gave.

**Session keys.** There is no list of them on the API. Ask the person: a session's key is in its address on the New
analysis page, `/predict/new/session/<key>/`. For a raw upload your agent made, `convert(cid)` gives the `key` of its
tiles once the conversion is done.

**Only your own.** On the upload routes and `/predict/api/convert/{cid}` an agent reaches only what it made itself
(the same person and agent): the person's browser uploads, and another agent's, answer `404`.

**Answers.** The routes answer as they do for the browser. The forms (`/predict/new/session/...`,
`/results/.../compute`, `/feedback/`) answer `303`; the client reads `Location` and **never follows it** (the HTML
pages it names are not on the API's list, and the token is never sent anywhere else): `/predict/preparing/<tid>/`
(then poll `/predict/api/preparing/<tid>`), `/results/<ps>/`, `/feedback/?id=<n>`. A refused form answers its HTML
page with the status (400, 404, 429 ...); the client turns its message into `ApiError`.

**Limits, per person across all their tokens** (in memory, per portal process):
- **60 writes an hour** (rolling): every `POST` above counts once, except an upload's pieces, which count once per
  upload. Over it: `429` with `Retry-After` (seconds). The `GET`s are not counted.
- **One open agent upload** at a time (all of the person's agents together, either kind). Another one: `429`.
  Finish it (create) or cancel it first. A new upload by the same agent gives up that agent's earlier one; an
  agent's upload never touches the person's browser upload, nor the reverse.
- **The active-analysis cap** that basic users have (3 queued or running sets) applies to every agent, whatever the
  person's role: `429`.
- The role cap, the module rules and the owner checks apply as for reads (an admin's agent acts as advanced, on
  that admin's own sets only).

**What is stamped** "agent for person", e.g. "claude for alice" (`via` = the token's agent):
- the prediction set: `request.json` and its registry record carry `owner_via`; it stays on every later status line
  of the record (the Jobs board shows it);
- the upload session's `upload.json` (`owner_via`), a raw conversion's `convert.json` (`made_via`) and its entry in
  the session record (`uploaded_via`);
- the feedback item (`via`);
- the Recent actions lines (Admin), on the door "API (verified)", with the AI tag.

Work started by an agent runs where the person's would: prediction on the laptop. Nothing is sent to the desktop
(that stays the manual `push-predict`).

## Answers

Errors come back as JSON: `{"error": "..."}` (a refused write form: its HTML page; see Writes).

| Status | Meaning |
|---|---|
| 200 | the route's JSON |
| 303 | a write form was accepted: read `Location` (never follow it) |
| 401 | the token is unknown, expired or revoked; the person is revoked; the API is switched off; or the route is in neither table |
| 403 | the person's role can't open that module (e.g. a basic user's agent on Data); a read-only token on a write route; writes are switched off; or the Cloudflare address |
| 404 | not found, or not yours (one answer for both) |
| 429 | a write limit (see Writes); `Retry-After` says when to try again |

## For the admin

- **Store.** One shared file: `~/.cryoev/agent_tokens.json`, or the path in `CJ_AGENT_TOKENS`. It is kept out of
  the roster.
  - Each change runs under an OS lock and re-reads the file first.
  - The portal re-reads the file when it changes.
  - `last_used` is merged in at most every 5 minutes, and that merge never undoes a revoke.
- **Kill switches.** `CJ_API=off` in the portal's environment refuses every bearer request; `CJ_API_WRITE=off`
  refuses only the write routes (reads go on). `off`, `0`, `false` and `no` all mean off (any case, spaces ignored).
  Browsers are not affected. Restart the portal to apply either.
- **Logs.** The token is never logged. The access log shows `alice via claude (bearer)`. A refusal logs only its
  reason.
- **Pre-push scan.** `tools/prepush_scan.py` blocks anything shaped like a token (`cev_` + 43 characters) from
  being pushed.

## Roadmap

- **Off the tailnet:** a Cloudflare service token in front of the access door, so agents can call the API from
  outside the tailnet.
