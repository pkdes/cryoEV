# Agent API (read-only)

The portal lets a person's own AI agent (for example Claude Code) **read** what that person can read: data and
lineage, their prediction sets, results status, and the jobs summary. It cannot change anything yet.

- **Tailnet only for now.** Use the portal's Tailscale address. The Cloudflare address refuses tokens with
  `403 "API is tailnet-only for now"`.
- **Read-only.** Only the routes in the table below, `GET` only.
- **Attributed.** Every request is the person's, done by their agent. The audit log (Admin > Recent actions) shows
  "claude for alice" with the AI tag, on the door "API (verified)". The portal checked the token, so the line is
  not marked "unverified".

## Getting a token

1. Sign in to the portal and open **My account** (`/account/`; also linked from About).
2. Under **New token**, enter the agent's name (lower-case, such as `claude`) and pick an expiry (7, 30 or 90 days).
3. Copy the token from the next page. **It is shown once.** The portal stores only its sha256.

Limits:
- at most 10 live tokens per person;
- 90 days at most;
- scope `read` only.

To revoke a token:
- you revoke your own on My account;
- an admin revokes anyone's on Admin > People (each person's card lists their tokens);
- revoking a person revokes all their tokens;
- making a new key link for a person (re-keying them) revokes all their tokens too.

Revoked and expired tokens are dropped from the store 30 days later. The ids `cli` and `-` are reserved: no person can have them.

**CLI fallback** (on the portal computer):

```
python -m portal.auth mint-token --id alice --agent claude [--days 30]
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
c.me()                       # {"id": "alice", "role": "advanced", "via": "claude", "auth": "bearer"}
c.jobs_summary()
```

## Routes (the whole allowlist)

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

## Answers

Errors come back as JSON: `{"error": "..."}`.

| Status | Meaning |
|---|---|
| 200 | the route's JSON |
| 401 | the token is unknown, expired or revoked; the person is revoked; the API is switched off; or the route is not in the table above (including every write) |
| 403 | the person's role can't open that module (e.g. a basic user's agent on Data), or the Cloudflare address |
| 404 | not found, or not yours (one answer for both) |

## For the admin

- **Store.** One shared file: `~/.cryoev/agent_tokens.json`, or the path in `CJ_AGENT_TOKENS`. It is kept out of
  the roster.
  - Each change runs under an OS lock and re-reads the file first.
  - The portal re-reads the file when it changes.
  - `last_used` is merged in at most every 5 minutes, and that merge never undoes a revoke.
- **Kill switch.** `CJ_API=off` in the portal's environment refuses every bearer request; browsers are not affected.
  Restart the portal to apply it.
- **Logs.** The token is never logged. The access log shows `alice via claude (bearer)`. A refusal logs only its
  reason.
- **Pre-push scan.** `tools/prepush_scan.py` blocks anything shaped like a token (`cev_` + 43 characters) from
  being pushed.

## Roadmap

- **Write scope** (upload, predict, compute, feedback): a `write` flag in the token store, with a rate limit, audit
  lines carrying `via`, and smoke tests on a scratch portal. The store already refuses `write` today.
- **Off the tailnet:** a Cloudflare service token in front of the access door, so agents can call the API from
  outside the tailnet.
