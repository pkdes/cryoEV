"""
Apply preview module (CryoEV portal, admin only; round 10 V4, PREVIEW ONLY): what applying a lab review package's
decisions to its label set would change, from a scratch apply, before anyone makes the new label set.

    GET  /apply/                        the review packages built on a label set: eligible ones with "Preview apply",
                                        their recent previews and the exact CLI command; the others with the reason
    POST /apply/preview                 form pkg + t (form token): in-process apply into scratch, then the change report
                                        -> 303 to the preview's page
    GET  /apply/run/<pkg>/<ts>          one preview: counts, the apply's log, the report link, the CLI command
    GET  /apply/view/<pkg>/<ts>/<path>  a file of that preview (report/index.html, its pictures, the scratch lv files)

Eligible (all of):
  - the package's base resolves to a label version: portal.titles.lv_from_path(review_package.json base, else the
    data.js source) (NOT base_kind), and data.js's source is that label version's own file (<...>/lv_<id>/<file>) on
    this computer (apply_review_edits refuses any other base);
  - at least one image marked reviewed (the apply is reviewed-only);
  - not named by any annotation/review_server.py `--package` in the services config (read-only: env
    CRYOEV_SERVICES_CONFIG, else ~/.cryoev/services.json; an unreadable config makes nothing eligible);
  - not a prediction package (pred_*: never scanned).
Excluded ENTIRELY (not even listed): compare packages and any package with a <review root>/_work_<pkg> folder (a blind
audit's answer key; only its existence is checked, it is never listed or read).

Preview: the package's edits/*.json are copied into <state>/apply_preview/<pkg>/<UTC ts>/edits (frozen: the report's
"decisions copy sha256" is theirs), then apply_review_edits.make_version (reviewed_only, out_root
<ts>/labels, who = the admin's people.json id) and change_report.build (<ts>/report). The who is the portal id mapped
through annotation/people.json (id, name or alias; the rule of apply_review_edits.roster_lookup); an unmapped id is
refused before anything is written. Each preview gets its own folder; folders older than KEEP_DAYS are pruned on every
visit. Nothing is written outside <state>/apply_preview (and the portal audit line): never the label sets
(CryoAI-laptop/labels), the package or the registry. There is NO apply button: the page shows the CLI command
(`--out-root` explicit, because the CLI's default <base>/../../labels is wrong for a label-set base).
Served files: only under that preview's folder, by a strict name pattern per path part, real path checked, a fixed
list of types; the report's HTML gets a CSP with no scripts. Every text is escaped.
"""
from __future__ import annotations

import hashlib
import hmac
import html
import json
import os
import re
import secrets
import shutil
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import parse_qs, quote

from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import FileResponse, RedirectResponse, Response
from starlette.routing import Route

from portal import audit as PA
from portal import titles as TT
from portal.modules import admin as AD
from portal.modules import results as RES

PREFIX = "/apply"
ROLES = ("admin",)
REPO = Path(__file__).resolve().parents[2]
PEOPLE = REPO / "annotation" / "people.json"
PREVIEW_DIR = "apply_preview"
KEEP_DAYS = 7
MAX_FORM = 4096
PKG_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.\-]{0,120}")
TS_RE = re.compile(r"(\d{8}T\d{6}Z)(?:-(\d{1,3}))?")
PART_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.\-]{0,200}")
SERVE = {".html": "text/html; charset=utf-8", ".jpg": "image/jpeg", ".json": "application/json",
         ".jsonl": "text/plain; charset=utf-8", ".csv": "text/plain; charset=utf-8"}
REPORT_CSP = ("default-src 'none'; img-src 'self'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'; "
              "frame-ancestors 'none'")
esc = html.escape
_busy = threading.Lock()   # one preview at a time (an apply + pictures can take a while)


# ---------------------------------------------------------------- inputs (read-only)
def services_config() -> Path:
    return Path(os.environ.get("CRYOEV_SERVICES_CONFIG") or Path.home() / ".cryoev" / "services.json")


def _package_arg(argv) -> list:
    out = []
    for i, x in enumerate(argv):
        x = str(x)
        if x == "--package" and i + 1 < len(argv):
            out.append(str(argv[i + 1]))
        elif x.startswith("--package="):
            out.append(x.split("=", 1)[1])
    return out


def served_packages(path=None):
    """-> (names, problem). Package folder names given to annotation/review_server.py --package by any services
    config entry (enabled or not). No config file: (set(), None). Unreadable or malformed: (None, why): fail closed."""
    p = Path(path) if path else services_config()
    if not p.exists():
        return set(), None
    try:
        doc = json.loads(p.read_text(encoding="utf-8-sig"))
        svcs = doc.get("services")
        if not isinstance(svcs, list):
            raise ValueError("no services list")
    except (OSError, ValueError, AttributeError) as e:
        return None, f"the services config can't be read ({type(e).__name__})"
    names = set()
    for s in svcs:
        argv = s.get("argv") if isinstance(s, dict) else None
        if isinstance(argv, str):
            argv = argv.split()
        if not isinstance(argv, list):
            continue
        if not any(re.split(r"[\\/]", str(x))[-1] == "review_server.py" for x in argv):
            continue
        for v in _package_arg(argv):
            name = re.split(r"[\\/]+", v.strip().rstrip("\\/ "))[-1]
            if name:
                names.add(name)
    return names, None


def load_people(path=None) -> list:
    try:
        people = json.loads(Path(path or PEOPLE).read_text(encoding="utf-8"))["people"]
        return [p for p in people if isinstance(p, dict) and isinstance(p.get("id"), str)]
    except (OSError, ValueError, KeyError, TypeError):
        return []


def people_id(portal_id, people=None):
    """The people.json id for a portal user id (its id, name or an alias; case- and space-insensitive: the rule of
    apply_review_edits.roster_lookup), else None."""
    if not isinstance(portal_id, str) or not portal_id.strip():
        return None
    for p in load_people() if people is None else people:
        for name in [p.get("id"), p.get("name"), *(p.get("aliases") or [])]:
            if isinstance(name, str) and name.strip().casefold() == portal_id.strip().casefold():
                return p["id"]
    return None


def base_file(row) -> Path | None:
    """data.js's source when it is the base label version's own file on this computer (<...>/lv_<id>/<file>)."""
    src, lv = row.get("source"), row.get("base_lv")
    if not (src and lv):
        return None
    p = Path(src)
    return p if p.parent.name == lv and p.is_file() else None


def packages(cfg, records, served, served_problem=None) -> list:
    """[{id, title, base_lv, base, reviewed, images, waiting, applied_by, last, eligible, why}] for the lab review
    packages under the review root whose base is a label version, compare / _work_ ones left out entirely."""
    root = RES.review_root(cfg)
    if root is None:
        return []
    out = []
    for p in TT.scan_packages(root, records):
        pkg = p["id"]
        if p["kind"] == "compare" or pkg.startswith("pred_") or not PKG_RE.fullmatch(pkg):
            continue
        if (Path(root) / f"_work_{pkg}").exists():       # a blind audit's answer key: exists() only, never read
            continue
        if not p["base_lv"]:
            continue                                     # a Roboflow export / pull base: not a label set
        prog = TT.package_progress(Path(root) / pkg) or {}
        row = {"id": pkg, "title": p["title"], "base_lv": p["base_lv"], "source": prog.get("source"),
               "reviewed": p["reviewed"], "images": p["images"], "waiting": p["waiting"], "applied_by": p["applied_by"],
               "last": p["last"], "dir": Path(root) / pkg}
        row["base"] = base_file(row)
        why = []
        if row["base"] is None:
            why.append("its data.js does not name the label set's own file on this computer")
        if not p["reviewed"]:
            why.append("no image is marked reviewed yet (the apply takes reviewed images only)")
        if served is None:
            why.append(served_problem or "the services config can't be read")
        elif pkg in served:
            why.append("it is being reviewed live (a review server in the services config serves it)")
        row["why"], row["eligible"] = why, not why
        out.append(row)
    return sorted(out, key=lambda r: r["id"])


def cli_command(row, who) -> str:
    base = Path(row["base"]) if row.get("base") else None
    q = lambda s: '"' + str(s).replace('"', '') + '"'  # noqa: E731
    return ("python annotation/apply_review_edits.py --base " + (q(base) if base else "<label set file>")
            + " --package " + q(row["dir"]) + " --out-root " + (q(base.parent.parent) if base else "<labels folder>")
            + " --who " + (who or "<your people.json id>"))


# ---------------------------------------------------------------- the scratch folder
def scratch_root(cfg) -> Path:
    return Path(cfg.state_dir) / PREVIEW_DIR


def ts_time(ts):
    m = TS_RE.fullmatch(ts or "")
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)


def prune(root: Path, now=None, keep_days=KEEP_DAYS) -> list:
    """Remove <root>/<pkg>/<ts>/ folders older than keep_days (by the ts in the name); -> removed. Only names this
    module makes are touched."""
    now = now or datetime.now(timezone.utc)
    gone = []
    try:
        pkgs = [d for d in Path(root).iterdir() if d.is_dir() and PKG_RE.fullmatch(d.name)]
    except OSError:
        return gone
    for d in pkgs:
        try:
            kids = list(d.iterdir())
        except OSError:
            continue
        for t in kids:
            when = ts_time(t.name) if t.is_dir() else None
            if when is not None and now - when > timedelta(days=keep_days):
                shutil.rmtree(t, ignore_errors=True)
                gone.append(t)
        try:
            next(d.iterdir())
        except StopIteration:
            d.rmdir()
        except OSError:
            pass
    return gone


def previews(cfg, pkg) -> list:
    """[{ts, info}] newest first, for one package."""
    d = scratch_root(cfg) / pkg
    out = []
    try:
        for t in d.iterdir():
            if t.is_dir() and TS_RE.fullmatch(t.name):
                try:
                    info = json.loads((t / "preview.json").read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    info = {}
                out.append({"ts": t.name, "info": info if isinstance(info, dict) else {}})
    except OSError:
        pass
    return sorted(out, key=lambda x: x["ts"], reverse=True)


def edits_sha256(edits_dir: Path) -> str:
    """The rule of annotation/compare_package.edits_sha256: sha256 over the sorted names and their bytes."""
    h = hashlib.sha256()
    for f in sorted(Path(edits_dir).glob("*.json")):
        h.update(f.name.encode("utf-8") + b"\x00" + f.read_bytes() + b"\x00")
    return h.hexdigest()


def new_folder(root: Path, pkg: str, now) -> Path:
    base = now.strftime("%Y%m%dT%H%M%SZ")
    for n in range(0, 1000):
        d = root / pkg / (base if n == 0 else f"{base}-{n}")
        try:
            d.mkdir(parents=True, exist_ok=False)
            return d
        except FileExistsError:
            continue
    raise OSError("no free preview folder name")


def run_preview(cfg, row, who, now=None) -> tuple:
    """-> (ts, info). Writes only under <state>/apply_preview/<pkg>/<ts>/."""
    from annotation import apply_review_edits as AR   # lazy: cv2 / numpy only when a preview runs
    from annotation import change_report as CR
    now = now or datetime.now(timezone.utc)
    root = scratch_root(cfg)
    prune(root, now)
    folder = new_folder(root, row["id"], now)
    info = {"pkg": row["id"], "who": who, "base_lv": row["base_lv"], "base": str(row["base"]),
            "started": now.isoformat(timespec="seconds"), "scope": "reviewed_only", "log": []}
    try:
        edits = folder / "edits"                       # frozen copy: the package may get new edits meanwhile
        edits.mkdir()
        for f in sorted((row["dir"] / "edits").glob("*.json")):
            (edits / f.name).write_bytes(f.read_bytes())
        info["edits_sha256"] = edits_sha256(edits)
        ver = AR.make_version(row["base"], row["dir"], edits, out_root=folder / "labels", who=who,
                              stamp=now.strftime("%Y%m%d_%H%M"), scope="reviewed_only", via=None,
                              log=lambda s: info["log"].append(str(s)))
        info["lv"] = ver["id"]
        info["counts"] = {k: ver.get(k) for k in ("n_package_images", "n_reviewed", "n_held_back", "n_images_applied",
                                                  "n_annotations_base", "n_annotations_out", "n_conflicts", "n_dropped",
                                                  "n_warnings", "n_rejected")}
        info["counts_by_op"] = ver.get("counts_by_op")
        res = CR.build(folder / "labels" / ver["id"], row["dir"] / "images", folder / "report", None, 900,
                       info["edits_sha256"])
        info["report"] = {"images": res["images"], "changed_images": res["changed_images"], "missing": res["missing"]}
    except AR.ApplyError as e:
        info["error"] = str(e)
    except Exception as e:                               # a preview must never take the page down
        info["error"] = f"the preview failed ({type(e).__name__}: {e})"
    info["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (folder / "preview.json").write_text(json.dumps(info, indent=1) + "\n", encoding="utf-8")
    return folder.name, info


def served_file(cfg, pkg, ts, rel) -> Path | None:
    """The file <state>/apply_preview/<pkg>/<ts>/<rel>, or None: strict names, no '..', a known type, real path
    inside the preview folder."""
    if not (PKG_RE.fullmatch(pkg or "") and TS_RE.fullmatch(ts or "")):
        return None
    parts = (rel or "").split("/")
    if not parts or any(not PART_RE.fullmatch(x) or ".." in x for x in parts):
        return None
    if Path(parts[-1]).suffix.lower() not in SERVE:
        return None
    folder = scratch_root(cfg) / pkg / ts
    p = folder.joinpath(*parts)
    try:
        real, top = os.path.realpath(p), os.path.realpath(folder)
        if os.path.commonpath([os.path.normcase(real), os.path.normcase(top)]) != os.path.normcase(top):
            return None
    except (OSError, ValueError):
        return None
    return p if p.is_file() else None


# ---------------------------------------------------------------- pages
STYLE = ("<style>.cj-apply table{border-collapse:collapse;width:100%;font-size:.9rem}"
         ".cj-apply td,.cj-apply th{padding:4px 6px;border-bottom:1px solid var(--line);text-align:left;"
         "vertical-align:top}.cj-apply code,.cj-apply pre{font-size:.8rem;overflow-wrap:anywhere;white-space:pre-wrap}"
         ".cj-apply .cj-card{margin:0 0 12px}.cj-apply form{display:inline}</style>")


def _when(iso) -> str:
    return esc(str(iso or "")[:19].replace("T", " ")) + (" UTC" if iso else "")


def render_index(rows, who, portal_id, tok, cfg, problem=None) -> str:
    out = [STYLE, '<div class="cj-apply"><h1>Apply preview</h1>',
           '<p class="cj-mut">What applying a review package\'s decisions (reviewed images only) would change in its '
           'label set. A preview is a scratch apply: nothing is written to the label sets or the registry. To make the '
           'new label set, run the command shown (there is no apply button yet).</p>']
    if problem:
        out.append(f'<p class="cj-card" role="alert">{esc(problem)}</p>')
    if who is None:
        out.append(f'<p class="cj-card" role="alert">Your portal id <b>{esc(str(portal_id))}</b> is not in '
                   f'annotation/people.json (as an id, name or alias), so a preview can\'t record who made it. Add it '
                   f'there first.</p>')
    ok = [r for r in rows if r["eligible"]]
    no = [r for r in rows if not r["eligible"]]
    if not ok:
        out.append('<p class="cj-card">No review package is ready for a preview.</p>')
    for r in ok:
        prev = previews(cfg, r["id"])[:5]
        plist = "".join(
            f'<li><a href="{PREFIX}/run/{quote(r["id"])}/{quote(p["ts"])}">{_when(p["info"].get("started"))}</a>'
            f'{" (failed)" if p["info"].get("error") else ""}</li>' for p in prev)
        applied = (f'<p class="cj-mut">Already applied as {esc(r["applied_by"])}: a new label set from it would '
                   f'supersede that one.</p>' if r["applied_by"] else "")
        waiting = "" if r["applied_by"] else f'; {r["waiting"]} decisions not applied'
        button = ("" if who is None else
                  f'<form method="post" action="{PREFIX}/preview">{AD.hidden_token(tok)}'
                  f'<input type="hidden" name="pkg" value="{esc(r["id"])}"><button type="submit">Preview apply</button>'
                  f'</form>')
        out.append(
            f'<section class="cj-card" id="p-{esc(r["id"])}"><h2>{esc(r["title"])}</h2>'
            f'<p>Package <code>{esc(r["id"])}</code> on label set <code>{esc(r["base_lv"])}</code>: '
            f'{r["reviewed"]} of {r["images"]} images reviewed{waiting}.</p>{applied}'
            f'{button}'
            + (f'<p class="cj-mut">Recent previews (kept {KEEP_DAYS} days):</p><ul>{plist}</ul>' if plist else "")
            + f'<p class="cj-mut">To make the label set (run in the cryoEV folder):</p>'
              f'<pre>{esc(cli_command(r, who))}</pre></section>')
    if no:
        body = "".join(f'<tr><td><code>{esc(r["id"])}</code></td><td>{esc(r["base_lv"])}</td>'
                       f'<td>{esc("; ".join(r["why"]))}</td></tr>' for r in no)
        out.append(f'<section class="cj-card"><h2>Not ready</h2><table><thead><tr><th>Package</th><th>Label set</th>'
                   f'<th>Why</th></tr></thead><tbody>{body}</tbody></table></section>')
    out.append("</div>")
    return "".join(out)


def render_run(pkg, ts, info, row, who) -> str:
    out = [STYLE, '<div class="cj-apply">', f'<p><a href="{PREFIX}/">Back to apply previews</a></p>',
           f'<h1>Preview: {esc(pkg)}</h1>',
           f'<p class="cj-mut">Started {_when(info.get("started"))} by {esc(str(info.get("who")))}; scratch only, '
           f'reviewed images only. Label set <code>{esc(str(info.get("base_lv")))}</code>.</p>']
    if info.get("error"):
        out.append(f'<section class="cj-card" role="alert"><h2>Refused</h2><pre>{esc(info["error"])}</pre></section>')
    else:
        c = info.get("counts") or {}
        ops = info.get("counts_by_op") or {}
        rows = "".join(f"<tr><td>{esc(k)}</td><td>{esc(str(v))}</td></tr>" for k, v in c.items())
        orow = "".join(f"<tr><td>{esc(k)}</td><td>{esc(str(v))}</td></tr>" for k, v in ops.items() if v)
        rep = info.get("report") or {}
        base = f'{PREFIX}/view/{quote(pkg)}/{quote(ts)}'
        out.append(
            f'<section class="cj-card"><h2>Change report</h2><p><a href="{base}/report/index.html">Open the change '
            f'report</a> ({esc(str(rep.get("images")))} of {esc(str(rep.get("changed_images")))} changed images '
            f'drawn).</p><p class="cj-mut">Scratch files: <a href="{base}/labels/{quote(str(info.get("lv")))}/version.json">'
            f'version.json</a> · <a href="{base}/labels/{quote(str(info.get("lv")))}/changes.jsonl">changes.jsonl</a> · '
            f'<a href="{base}/labels/{quote(str(info.get("lv")))}/conflicts.csv">conflicts.csv</a></p></section>'
            f'<section class="cj-card"><h2>Counts</h2><table>{rows}</table></section>'
            f'<section class="cj-card"><h2>Applied by kind</h2><table>{orow or "<tr><td>none</td></tr>"}</table>'
            f'</section>')
    out.append(f'<section class="cj-card"><h2>Apply log</h2><pre>{esc(chr(10).join(info.get("log") or []))}</pre>'
               f'<p class="cj-mut">Decisions copy sha256 <code>{esc(str(info.get("edits_sha256") or ""))}</code></p>'
               f'</section>')
    if row is not None:
        out.append(f'<section class="cj-card"><h2>Make the label set</h2><p class="cj-mut">After checking the report, '
                   f'run in the cryoEV folder:</p><pre>{esc(cli_command(row, who))}</pre></section>')
    out.append("</div>")
    return "".join(out)


# ---------------------------------------------------------------- routes
def routes(ctx) -> list:
    secret = secrets.token_bytes(32)                     # per process, like admin's: a restart makes open forms stale

    def token(request: Request) -> str:
        return hmac.new(secret, hashlib.sha256(AD.binding(request).encode()).digest(), hashlib.sha256).hexdigest()[:40]

    def show(request, title, body, status=200):
        r = ctx.page(request, title, body, active="data")
        r.status_code = status
        return r

    def admin_only(request):
        if ctx.user(request).get("role") != "admin":      # the Gate already refuses; defence in depth
            return show(request, "Apply preview", '<p class="cj-card">Admins only.</p>', 403)
        return None

    def state(request):
        from portal.modules import data as DATA
        records, _err = DATA.read_registry(getattr(ctx.cfg, "registry", None))
        served, problem = served_packages()
        return packages(ctx.cfg, records, served, problem), problem

    def index(request: Request):
        if (deny := admin_only(request)) is not None:
            return deny
        prune(scratch_root(ctx.cfg))
        rows, problem = state(request)
        pid = ctx.user(request).get("id")
        return show(request, "Apply preview", render_index(rows, people_id(pid), pid, token(request), ctx.cfg, problem))

    async def preview(request: Request):
        if (deny := admin_only(request)) is not None:
            return deny
        ctype = (request.headers.get("content-type") or "").split(";")[0].strip().lower()
        buf = bytearray()
        async for chunk in request.stream():
            buf += chunk
            if len(buf) > MAX_FORM:
                return show(request, "Apply preview", '<p class="cj-card">That form is too large.</p>', 413)
        try:
            f = {k: v[0] for k, v in parse_qs(bytes(buf).decode("utf-8"), max_num_fields=4).items() if v}
        except (UnicodeDecodeError, ValueError):
            f = None
        if ctype != "application/x-www-form-urlencoded" or f is None:
            return show(request, "Apply preview", '<p class="cj-card">Send the form from the Apply preview page.</p>',
                        400)
        if not hmac.compare_digest(f.get("t", "").encode(), token(request).encode()):
            return show(request, "Apply preview", '<p class="cj-card" role="alert">This form has expired. Reload the '
                        f'page and try again.</p><p><a href="{PREFIX}/">Back</a></p>', 403)
        pid = ctx.user(request).get("id")
        who = people_id(pid)
        if who is None:
            return show(request, "Apply preview", f'<p class="cj-card" role="alert">Refused: your portal id '
                        f'<b>{esc(str(pid))}</b> is not in annotation/people.json (as an id, name or alias). Add it there '
                        f'first; nothing was run.</p><p><a href="{PREFIX}/">Back</a></p>', 403)
        rows, _ = await run_in_threadpool(state, request)
        row = next((r for r in rows if r["id"] == f.get("pkg")), None)
        if row is None or not row["eligible"]:
            return show(request, "Apply preview", '<p class="cj-card" role="alert">That package is not ready for a '
                        f'preview (or does not exist).</p><p><a href="{PREFIX}/">Back</a></p>', 404)
        if not _busy.acquire(blocking=False):
            return show(request, "Apply preview", '<p class="cj-card">Another preview is running. Try again in a '
                        f'minute.</p><p><a href="{PREFIX}/">Back</a></p>', 409)
        try:
            ts, info = await run_in_threadpool(run_preview, ctx.cfg, row, who)
        finally:
            _busy.release()
        try:
            PA.audit(ctx.cfg.state_dir, pid, "apply preview", row["id"],
                     {"ts": ts, "lv": info.get("lv"), "ok": not info.get("error")})
        except OSError:
            pass
        return RedirectResponse(f"{PREFIX}/run/{quote(row['id'])}/{quote(ts)}", status_code=303)

    def run_page(request: Request):
        if (deny := admin_only(request)) is not None:
            return deny
        pkg, ts = request.path_params["pkg"], request.path_params["ts"]
        p = served_file(ctx.cfg, pkg, ts, "preview.json")
        if p is None:
            return show(request, "Apply preview", '<p class="cj-card">No such preview (they are kept '
                        f'{KEEP_DAYS} days).</p><p><a href="{PREFIX}/">Back</a></p>', 404)
        try:
            info = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            info = {"error": "this preview's record can't be read"}
        rows, _ = state(request)
        row = next((r for r in rows if r["id"] == pkg and r["eligible"]), None)
        pid = ctx.user(request).get("id")
        return show(request, f"Preview {pkg}", render_run(pkg, ts, info, row, people_id(pid)))

    def view(request: Request):
        if (deny := admin_only(request)) is not None:
            return deny
        p = served_file(ctx.cfg, request.path_params["pkg"], request.path_params["ts"], request.path_params["rel"])
        if p is None:
            return ctx.no_store(Response("Not found", status_code=404, media_type="text/plain"))
        r = FileResponse(p, media_type=SERVE[p.suffix.lower()])
        r.headers["Cache-Control"] = "no-store"
        if p.suffix.lower() == ".html":
            r.headers["Content-Security-Policy"] = REPORT_CSP
        return r

    return [Route(PREFIX + "/", index, methods=["GET"]),
            Route(PREFIX + "/preview", preview, methods=["POST"]),
            Route(PREFIX + "/run/{pkg}/{ts}", run_page, methods=["GET"]),
            Route(PREFIX + "/view/{pkg}/{ts}/{rel:path}", view, methods=["GET"])]
