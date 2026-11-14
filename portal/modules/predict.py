"""
Predict module (CryoEV portal, journey J1): a person runs the production model on their tiles and gets a prediction set
(BUILD_CONTRACTS.md section 2) to review and count. Raw cryo-EM montages can be converted to tiles first.

    GET  /predict/                        My analyses (admin: everyone's), raw conversions, "New analysis"
    GET  /predict/predict.js              the module's script (portal/static/predict.js; no inline script anywhere)
    GET  /predict/new/                    step 1: what images work; pick a processed session, upload, or convert raw
    GET  /predict/new/session/<key>/      step 2: tiles (all, a count, by sample), nm/px per sample + the
    POST /predict/new/session/<key>/        confirmation tick, model -> create() + start_detached() -> 303 set page
                                          (?json=1: the form's data, sig + samples + models, for an agent)
    GET  /predict/new/upload/             upload processed 8-bit PNG/JPEG tiles (caps MAX_FILES x MAX_FILE_BYTES),
    POST /predict/new/upload/               sample label + nm/px + tick. The page's script sends one file per request
                                            (the upload API below); this POST is the no-JavaScript form, for small
                                            sets only (at most NOJS_MAX_BYTES in all; else refused, saying why)
    GET  /predict/new/raw/                admin, advanced: upload raw montages (.mrc + .mrc.mdoc, or .idoc + .tif
                                            tiles; an optional sample key .csv), a session id and a recipe -> convert
    GET  /predict/new/raw/sessions/       admin: raw session folders under the scan roots with no 3d tiles yet
    GET  /predict/new/raw/sessions/<key>/ admin: pick montages of one; POST: convert (the folder is only read)
    GET  /predict/convert/<id>/           one conversion: status (auto-refresh), warnings, then "Analyse these tiles"
    POST /predict/convert/<id>/stop       admin: kill a running converter (or drop a queued one) -> failed "stopped"
    POST /predict/convert/<id>/retry      admin: a failed conversion whose inputs are still there is queued again
    POST /predict/convert/<id>/inputs     admin: delete its uploaded raw files now
    POST /predict/convert/<id>/stitch     admin: (re)make the montages of a done conversion whose raw files are
                                            still there (a queued "stitch" job: convert_raw --stitch-only)
    POST /predict/convert/<id>/register   admin: add the upload to Data -> Sessions again (the watcher does it)
    POST /predict/convert/<id>/details    owner / admin: the upload's Session details (portal/upload_sessions.py);
                                            the raw form's "Sample on each grid" becomes input/sample_key.csv
    GET  /predict/<ps>/                   the set: status (auto-refresh), Review / Results, spot-check list
    POST /predict/<ps>/review             build the prediction review package in a thread (if missing); 303 to
                                          /reviews/go/pred_<ps> when it exists (round 13)
    POST /predict/<ps>/results            compute results/results.json in a thread (if missing or older than edits)
    POST /predict/<ps>/start              start the laptop run again (failed, or no heartbeat for STALE_S)
    GET  /predict/api/mine                {"schema": "cj_predict_mine/1", "items": [...]}: Home's "My results"
    GET  /predict/api/set/<ps>            one set's status; ?html=1 adds the page fragments (the refresh swaps them)
    GET  /predict/api/convert/<id>        one conversion's status; ?html=1 adds the page fragments
    GET  /predict/api/recipe/<name>       advanced/admin: the recipe's real values (convert_raw describe, child process)
    GET  /predict/convert/<id>/settings   the Settings section: values used, "custom settings" diff, Adjust panel
    POST /predict/convert/<id>/preview    JSON {tile, params}: one tile before/after (convert_raw preview; ONE at a time
                                          per portal, 429 otherwise); GET .../preview/<token>/{before,after}.png
    POST /predict/convert/<id>/reconvert  owner/admin: a NEW conversion of the same raw input with --params
    GET  /predict/convert/<id>/tiles.zip  owner/admin: the PNG tiles + MANIFEST.csv (zipstream, 2 slots per person)
  Upload API: one file, or one piece of at most CHUNK_BYTES of a bigger file, per request (a Cloudflare free tunnel
  caps a request at about 100 MB; a raw .mrc is often more):
    POST /predict/api/upload              {"kind": "tiles"|"raw", "fields": {...}, "files": [{"name", "size"}],
                                           "resume": id|null} -> the session {"upload", "chunk", "received", "done"}.
                                           Fields, names, caps and free disk (the declared total is held) are checked
                                           here, before a byte is sent. "resume": the same person's session with the
                                           same files carries on where it stopped.
    GET  /predict/api/upload/<id>         the session (what has arrived)
    POST /predict/api/upload/<id>/file?name=<file>&offset=<n>   body = the bytes. offset must be what has arrived
                                           (else 409 + the session, and the page resumes from there). A file is
                                           content-checked once complete (tiles: 8-bit PNG/JPEG; raw: MRC, SerialEM
                                           sidecar, TIFF, sample key)
    POST /predict/api/upload/<id>/create  every file there -> the prediction set (tiles) or the conversion (raw)
    POST /predict/api/upload/<id>/cancel
  A session lives in <root>/.portal/us_<12 hex>/ (upload.json + files/); only its owner reaches it. Names: tiles
  .png/.jpg/.jpeg, raw the raw set; never a name ending in .part. A size or content violation, or a full disk, removes
  it ("nothing was kept"); a bad request (unknown name, bad offset: 400; wrong offset: 409), a dropped connection or
  a missing piece does not. A session never finished (no "create") is removed SCRATCH_MAX_AGE_S after its last
  request, and its disk hold released.
  Sidecars: every [Image = <name>] of an uploaded .idoc / .mrc.mdoc (read as cryoem_core.metadata.parse_metadata
  reads it) must be a bare file name (no \\ / :, not . or ..) and one of the uploaded files; a picked session's
  sidecars must name bare files too. Else refused: cryoem_core would open whatever path it names (another folder, a
  UNC share). jobs/convert_raw.py checks the same again before any tile is opened.

Sessions people may pick (portal/predict_sources.py): the session folders found under the scan roots
(CJ_PREDICT_SCAN_ROOTS, default the processing root) that have `3. Processed images/3d. Tiles no scalebar*`, plus the
admin's hand list (CJ_PREDICT_SOURCES), with the Admin panel's overrides (Admin > Predict sessions: hide, rename). A
session is named by its full registry id, e.g. 20261027 or 20260827-2. Plus the finished raw conversions, each seen
only by its owner and the admin (ctx.can_see).

Raw conversion (BUILD_CONTRACTS.md section 10): `python -m jobs.convert_raw` as a detached one-shot subprocess with
the processing repo from CJ_PROCESSING_REPO, one at a time (MAX_CONVERT_RUNNING; the others wait queued; a watcher
thread starts the next and records the end), at most MAX_CONV_PER_USER queued or running per person, and only with
DISK_FACTOR x the raw input free on the disk. Output in <root>/raw_tiles/<session id>_<6 hex>/ (the name starts with
the session id: Predict reads the id from it):
    convert.json   {"schema": "cj_convert/1", id, owner, session, recipe, gpu, source, inputs, sample_key, created}
    status.json    {state: queued|running|done|failed, updated, pid, started, finished, error, kind, warnings,
                    result (convert_raw's JSON), n_tiles}
    convert.out    the converter's stdout (one JSON object); convert.log its stderr (the pipeline's progress)
    input/         the uploaded raw files: kept CJ_RAW_KEEP_DAYS (default 7; 0 = removed at once) after the
                   conversion ends, done or failed, then swept; the admin may delete them sooner
    tiles/         the converter's --out: PNG tiles + MANIFEST.csv, the Predict source once done
    montages/      (round 7) <montage>_stitched.jpg, layout.json, stitch.json from a stitch job (convert_raw
                   --stitch-only, stdout stitch.out) queued when the conversion is done (job "convert") or by the
                   admin (job "backfill"); a stitch failure, timeout or stop never fails the conversion and never
                   replaces a finished set. acquisition.json beside it (written by the conversion itself).
                   status.json stitch {state queued|running|done|failed|skipped, job, error, n, kept_previous, pid
                   ...}; registered false|true|"error: ..."|"excluded: ..." (upload conversions, registered by the
                   watcher when portal.upload_sessions is on) with register_tries / register_since / register_next
                   (backoff) / register_wait while the registry is busy
  The recipe is "default" (convert_raw's DEFAULT_RECIPE; the page warns) or one of the processing repo's session
  recipes that convert_raw's static check accepts (the portal never runs one). The GPU is used only when ticked and the
  GPU check passes (else refused, saying why); the CPU otherwise. A session folder the admin picks is never written:
  the converter reads it and writes only under raw_tiles/.
  Stuck: a converter running over CONVERT_MAX_S, or whose log is silent for CONVERT_STALE_S, is killed and marked
  failed ("timeout"). Its process is known by pid AND start time (pid_created), so a reused pid is never taken for
  it (nor killed).

Configuration (environment, read per request; Config attributes of the same lower-case name win when present):
  CJ_PREDICT_SCAN_ROOTS, CJ_PREDICT_SOURCES   the session list (portal/predict_sources.py)
  CJ_PREDICT_ROOT      prediction sets (default <CryoAI-laptop>/predict, jobs.predict_set.default_root())
  CJ_REVIEW_ROOT       prediction review packages, when cfg.review_root (the shell's /review/<pkg_id>/ root) is unset
                       (default <CryoAI-laptop>/review)
  CJ_PROCESSING_REPO   the lab's processing repo (read only) for raw conversion; unset = no raw conversion
  CJ_RAW_ROOT          where the whole raw pipeline lives (raw-upload .portal staging + raw_tiles/), e.g. a NAS or an
                       external drive (cfg.raw_root wins); default = the predict root. Old <predict root>/raw_tiles
                       conversions stay listed and swept. Prediction sets made from it copy tiles (never link).
  CJ_RAW_KEEP_DAYS     days uploaded raw files are kept after their conversion ends (default 7; 0 = not kept)
  CJ_RAW_MIN_FREE_GB   below this much free disk (default 25; 0 = off), the oldest-ended kept raw inputs go early
  CJ_RAW_LEGACY_LIST   off: with CJ_RAW_ROOT set, the old <predict root>/raw_tiles is not listed (preview portals)
  CJ_UPLOAD_SESSIONS   on: finished upload conversions are registered as sessions (portal.upload_sessions; live only)
  The registry is the portal's (cfg.registry).

Rules:
  - Every set view, API and action checks ctx.can_see(user, request.json owner); not found and not allowed give the
    same 404. The same for conversions (convert.json owner).
  - nm/px is never defaulted: each sample's value is shown (from the manifest when it has one) and the person must
    tick "I confirm the pixel size"; their id goes into images.csv nm_confirmed_by. No tick: refused, nothing written.
  - Pixel-size tolerance (BUILD_CONTRACTS.md section 8): a sample whose manifest nm/px values spread (max - min) by
    at most NM_TOLERANCE (1 %) of their median is one scale. The form shows the range; left empty, each tile keeps
    its own value in images.csv; a number typed there applies to all of that sample's tiles and is recorded in the
    portal audit log (predict.nm_override). Over the tolerance: refused, as before.
  - Model: the registry's production model (also in models/models.yaml) by default; basic users get only that one.
  - Raw upload: admin and advanced. Raw session pick: admin. Basic users get 403 on both.
  - Raw upload files that can't be converted are left out, not fatal: atlas files (named *atlas*, also left out by
    the page) at the start; a file failing its content check, or a sidecar naming a path or a file not uploaded,
    is deleted on arrival / before conversion (never read by the converter) with its .mrc / .mrc.mdoc partner.
    The upload meta and convert.json keep skipped [{name, why}]; the conversion page lists them. Nothing
    convertible left: refused with the reasons. Sizes, disk and the piece protocol stay hard refusals.
  - Long work (create(), uploads to disk, the review package, results) runs in worker threads, never on the event
    loop. The predictor and the converter are detached processes; status.json says the rest.
  - Plain words for scientists: paths, ids of folders and raw errors only on admin-only lines.
  - Everything shown is escaped here.
  - Uploads: caps checked on the declared list and again while streaming; free disk checked at the start (the whole
    declared size held), at each piece and every 64 MB, counting every upload in flight (FREE_MARGIN_BYTES kept free).
  - Round 6 (P1): up to 1000 tiles (jobs/predict_set.MAX_IMAGES). Making a set takes longer than an HTTP request may
    last, so a form only validates, starts a background task (keyed by upload id, or by owner + session + pick: the
    same key joins the running task) and answers 303 to /predict/preparing/<id>/, which polls
    /predict/api/preparing/<id> and opens the analysis. Uploads are hard-linked into the set (link=True; a copy on
    another drive) and the staging is removed only after success: a failed create keeps the upload ("Try again").
    TILES_MAX_TOTAL_BYTES and a free-disk check (copies: total + 10 %) refuse early. The forms show an estimate
    (a + b * images fitted to finished laptop runs, cached 10 min) and an optional Name (request.json `name`,
    not stored when it is still the prefill); POST /predict/<ps>/rename changes it (portal/names.py has
    display_name).
  - A basic user has at most MAX_ACTIVE_BASIC sets queued or running. A review package built here is appended to
    the registry (review_package) once.

Layout (round 7, P7): this module keeps PREFIX, ROLES, the page script and routes() with its handlers; the rest
lives in portal/predict_parts/ (common, sets, estimate, raw, disk, runner, create, uploads, pages, convpages) and
is re-exported here. Patch a name on the part that defines it (portal/predict_parts/__init__.py).
"""
from __future__ import annotations

import asyncio
import contextlib
import csv
import errno
import hashlib
import html
import json
import math
import os
import re
import secrets
import shutil
import struct
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from starlette.concurrency import run_in_threadpool
from starlette.requests import ClientDisconnect, Request
from starlette.responses import FileResponse, RedirectResponse, Response
from starlette.routing import Route

from jobs import convert_raw as CR
from jobs import predict_local as PL
from jobs import predict_set as PS
from portal import audit as PA
from portal import convert_settings as CS
from portal import predict_sources as PSRC
from portal import names as NM
from portal import upload_sessions as US

# ---------------------------------------------------------------- the parts (portal/predict_parts; P7 split)
# Every name this module had before the split is re-exported, so reads of P.<name> keep working (tests, admin.py,
# gallery.py, data.py, upload_sessions.py). To PATCH a name, patch the part that defines it: a name the tests patch
# is used by the other parts and the handlers below only as <PART>.<name> (test_predict_patch_guard.py checks).
from portal.predict_parts.common import (CACHE_MAX, CHUNK_BYTES, CONVERT_INPUT, CONVERT_JSON, CONVERT_LOCK,
                                         CONVERT_LOG, CONVERT_MAX_S, CONVERT_OUT, CONVERT_POLL_S, CONVERT_STALE_S,
                                         CONVERT_TILES, CREATE_ID_RE, DISK_CHECK_BYTES, DISK_FACTOR,
                                         DISK_NEED_FACTOR, EST_FALLBACK, EST_MAX_SETS, EST_MAX_S_PER_IMAGE,
                                         EST_TTL_S, FREE_MARGIN_BYTES, GPU_CHECK_S, HERE, IMAGE_EXT,
                                         KEEP_DEFAULT_DAYS, KEEP_SETTING, MAX_ACTIVE_BASIC, MAX_CONVERT_RUNNING,
                                         MAX_CONV_PER_USER, MAX_FIELDS, MAX_FIELD_BYTES, MAX_FILES, MAX_FILE_BYTES,
                                         MODELS_YAML, NAME_BAD, NM_EPS, NM_FROM_MANIFEST, NM_HINT, NM_MAX, NM_MIN,
                                         NM_TOLERANCE, NOJS_MAX_BYTES, NOTES_SUB, NO_MODEL, PART, PID_START_SLACK_S,
                                         PORTAL_TMP, RAW_EXT, RAW_ID_RE, RAW_MAX_FILES, RAW_MAX_FILE_BYTES,
                                         RAW_MAX_TOTAL_BYTES, RAW_ROLES, RAW_SCAN_DEPTH, RAW_SCAN_MAX,
                                         RAW_SIDE_MAX_BYTES, RAW_SKIP_DIR, RAW_TILES, RECIPE_TTL_S, REFRESH_S, REPO,
                                         RESUBMIT_S, Refusal, SAMPLE_COLUMNS, SAMPLE_KEY, SCHEMA_CONVERT,
                                         SCHEMA_CONVERT_VIEW, SCHEMA_MINE, SCHEMA_SET, SCHEMA_UPLOAD,
                                         SCRATCH_MAX_AGE_S, SIZE_COLUMN, SPOT_N, STALE_S, START_MAX_BYTES, TEMPLATE,
                                         TEST_HOOKS, TILES_MAX_TOTAL_BYTES, UPLOAD_FILES, UPLOAD_META, UPLOAD_RE,
                                         WIN_RESERVED, WRITE_BATCH, _PH, _parse_local, _plain, _read_json, _setting,
                                         ago, chip, esc, msg_box, now_iso, os_error_text, predict_root, problem_list,
                                         registry_root, review_root, tech, when, wrap)
from portal.predict_parts.sets import (RESTART_REQUESTED, REVIEW_PACKAGE_LOCK, SIZE_RE, _SPOT_CACHE, _SPOT_LOCK,
                                       _TASKS, _TASK_LOCK, _models_yaml, _num, age_s, allowed_models, build_results,
                                       build_review, can_restart, edits_mtime, images_rows, list_sets, load_set,
                                       models, nm_range_text, nm_spread, one_scale, package_dir, pick, quality_line,
                                       register_package, restart, results_file, samples_of, set_dir, size_suffix,
                                       spot_list, start_task, status_view, task, tile_size, tiles_of, waiting_text)
from portal.predict_parts.estimate import (_EST_CACHE, _EST_LOCK, _RUN_START, estimate_html, estimate_line,
                                           estimate_text, fit_line, queue_ahead, run_seconds)
from portal.predict_parts.raw import (MRC_MAGIC, SIDECAR_EXT, TIFF_MAGIC, _RECIPES, _RECIPE_LOCK, check_gpu,
                                      file_problem, gpu_check, image_ref_problem, processing_repo, raw_fields,
                                      raw_montages, raw_problem, raw_sessions, recipes, session_recipe,
                                      sidecar_images, sidecar_problems, test_hooks, valid_session_id)
from portal.predict_parts.disk import (Reservation, _RESERVED, _RES_DEV, _RES_LOCK, _dev, _existing, _held, disk_free)
from portal.predict_parts.runner import (CONVERT_ACQ, CONVERT_MONTAGES, CONV_TONE, LEGACY_LIST_SETTING,
                                         MIN_FREE_DEFAULT_GB, MIN_FREE_SETTING, REGISTER_BACKOFF_MAX_S, STITCH_OUT,
                                         UPLOAD_SESSIONS_SETTING, _WATCH, _WATCH_LOCK, _converter_cmd, _off, _on,
                                         _same_path, _stitch_end, all_conversions, can_register_again, can_retry,
                                         can_stitch, check_conv_cap, check_disk_for, conv_dir, conv_paths,
                                         conv_status_view, convert_argv, converter_result, delete_inputs_now,
                                         ensure_watcher, finalize, finalize_stitch, input_bytes, inputs_kept_until,
                                         inputs_linked, inputs_present, is_upload, keep_days, kick, kill_proc,
                                         list_conversions, load_conversion, log_tail, min_free_bytes, montages_done,
                                         move_tree, needs_registration, new_conversion, overdue, proc_alive,
                                         proc_created, queue_stitch, queued_stitch, raw_base, raw_root, raw_roots,
                                         register_again, register_conversions, registration_due, registration_line,
                                         remove_inputs, restore_montages, retry_conversion, runner_lock,
                                         session_convert, spawn_converter, start_conversion, start_stitch,
                                         stitch_argv, stitch_busy, stitch_state, stitch_summary, stop_conversion,
                                         sweep_inputs, tmp_dirs, upload_sessions_module, write_conv_status)
from portal.predict_parts.create import (KEPT_CHANGED, _AUDIT_FAILED, _AUDIT_LOCK, _CREATES, _CREATE_LOCK, _LINK_OK,
                                         _RENAME_LOCKS, _RENAME_LOCKS_LOCK, _SRC_CACHE, _SRC_LOCK, active_sets,
                                         audit_event, audit_problem, can_link, check_active, check_create_disk,
                                         check_model, check_nm, chosen_name, converted_sources, create_and_start,
                                         create_running, create_task, kept_form, kept_scope, read_tiles,
                                         recently_done, record_nm_override, rename_lock, rename_set, running_creates,
                                         same_volume, scratch, selection_key, session_plan, session_run,
                                         session_submit, sources, sources_safe, start_create, tile_bytes)
from portal.predict_parts.uploads import (ATLAS_WHY, NOJS_REFILL_BYTES, Resync, SAMPLE_KEY_NAME, UploadSink,
                                          _UP_ALOCKS, _UP_RES, _UP_RES_LOCK, _stream, check_sample_source,
                                          check_upload_fields, clean_fields, declared_files, drop_upload, is_sidecar,
                                          load_upload, nojs_too_big, raw_leave_out, raw_pairing, receive_piece,
                                          receive_upload, safe_name, skip_raw_file, skipped_names, sweep_scratch,
                                          upload_create, upload_dir, upload_fields, upload_received,
                                          upload_reservation, upload_start, upload_state, upload_submit,
                                          write_sample_key)
from portal.predict_parts.pages import (actions_html, conversions_html, default_name_for, item_of, mine_html,
                                        model_block, name_field, page_mine, page_new, page_prep, page_session_form,
                                        page_set, page_upload_form, prep_fragment, prep_text, prep_view, rename_form,
                                        session_label, set_title, set_view, source_text, status_html, tab_title,
                                        unreadable_page)
from portal.predict_parts.convpages import (CPREV, MAX_TILE_NAMES, PREV_FILES, PREV_KEEP_S, PREV_TOKEN_RE,
                                            STITCH_LABEL, _fnum, _slider, adjust_panel, bare_recipe, conv_fragments,
                                            conv_values, convset_fragment, convset_section, gpu_box, human_bytes,
                                            not_set_up, page_convert, page_raw_session, page_raw_sessions,
                                            page_raw_upload, raw_kept_reason, recipe_file_name, recipe_readout,
                                            recipe_readout_box, recipe_select, settings_rows, settings_summary,
                                            stitch_card, tile_names)
from portal.predict_parts import sets as SETS
from portal.predict_parts import raw as RAW
from portal.predict_parts import runner as RUNNER
from portal.predict_parts import create as CREATE
from portal.predict_parts import pages as PAGES

PREFIX = "/predict"
ROLES = ("admin", "advanced", "basic")


# ---------------------------------------------------------------- the page's script (served as /predict/predict.js)
SCRIPT = HERE.parent / "static" / "predict.js"      # served as /predict/predict.js (app.py serves only /cj/*)
JS = SCRIPT.read_text(encoding="utf-8")


# ---------------------------------------------------------------- routes
def routes(ctx) -> list:
    cfg = ctx.cfg
    sweep_scratch(cfg)                               # at start (routes() runs once, before the portal serves)
    try:                                             # a conversion left queued / running by a portal that stopped;
        if kick(cfg):                                # old raw inputs swept
            ensure_watcher(cfg)
    except Exception:                                # no root configured or not there yet
        pass

    def not_found(request):
        r = ctx.page(request, "Not found", wrap('<div class="cj-predict"><h1>Not found</h1><p>There is no analysis '
                                                'here that you can open.</p><p><a href="/predict/">My analyses</a>'
                                                '</p></div>'), active="predict")
        r.status_code = 404
        return r

    def refused_page(request, title, body, status):
        r = ctx.page(request, title, body, active="predict")
        r.status_code = status
        return r

    def forbidden(request, text):
        return refused_page(request, "Not allowed", wrap(f'<div class="cj-predict"><h1>Not allowed</h1>'
                                                         f'<p>{esc(text)}</p><p><a href="/predict/new/">Back</a></p>'
                                                         f'</div>'), 403)

    def find_source(key, user):
        srcs, _ = sources_safe(cfg, user, ctx.can_see)
        return next((s for s in srcs if s["key"] == key and not s["problem"]), None)

    def admin_text(e: Refusal, user) -> str:
        return e.msg + (f"\n(admin) {e.tech}" if e.tech and user.get("role") == "admin" else "")

    def agent_mismatch(request, made_via) -> bool:
        """A bearer request on an upload / conversion another agent (or the person's browser) made: True (the
        handler answers 404, as for no such thing). An agent acts only on what it made itself (round 11 #65)."""
        via = PA.request_via(request)
        return via is not None and (made_via or None) != via

    def refusal_json(e: Refusal, user, dropped: bool, extra=None):
        doc = {"error": admin_text(e, user), "dropped": dropped}
        doc.update(extra or {})
        r = ctx.json(doc, e.status)
        if e.status == 429:                              # a limit (active sets, one agent upload): try again later
            r.headers["Retry-After"] = "60"
        return r

    # ---- pages (sync: Starlette runs them in a worker thread)
    def index(request: Request):
        try:
            if kick(cfg):                            # records ends, sweeps old raw inputs (cheap: a few small files)
                ensure_watcher(cfg)
        except Exception as e:                       # noqa: BLE001 - the list still shows
            print(f"[predict] kick: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        return ctx.page(request, "Predict", page_mine(cfg, ctx.user(request), ctx.can_see, request.query_params),
                        active="predict")

    def new(request: Request):
        return ctx.page(request, "New analysis", page_new(cfg, ctx.user(request), ctx.can_see), active="predict")

    def session_form(request: Request):
        user = ctx.user(request)
        as_json = request.query_params.get("json") == "1"         # the form's data for an agent (round 11 #65)
        if PA.request_via(request) is not None and not as_json:    # an agent gets the JSON only
            return ctx.json({"error": "not found (agents: add ?json=1)"}, 404)
        src = find_source(request.path_params["key"], user)
        if src is None:
            return ctx.json({"error": "not found"}, 404) if as_json else not_found(request)
        try:
            q = request.query_params
            t = kept_scope(cfg, src, q, read_tiles(src))
        except Refusal as e:
            if as_json:
                return ctx.json({"error": e.msg}, e.status)
            return refused_page(request, src["label"], unreadable_page(src, e, user), e.status)
        if as_json:
            return ctx.json(PAGES.session_form_view(cfg, user, src, t))
        return ctx.page(request, src["label"], page_session_form(cfg, user, src, t, kept_form(q)), active="predict")

    async def session_post(request: Request):
        user = ctx.user(request)
        src = await run_in_threadpool(find_source, request.path_params["key"], user)
        if src is None:
            return await run_in_threadpool(not_found, request)
        fd = await request.form(max_files=0, max_fields=1000)
        form = {k: v for k, v in fd.items() if isinstance(v, str)}
        form["samples"] = [v for v in fd.getlist("samples") if isinstance(v, str)]
        try:
            plan = await run_in_threadpool(session_plan, cfg, user, src, form, PA.request_via(request))
        except Refusal as e:
            msg, status = admin_text(e, user), e.status

            def again():
                try:
                    t = kept_scope(cfg, src, form, read_tiles(src))
                except Refusal as e2:
                    return refused_page(request, src["label"], unreadable_page(src, e2, user), e2.status)
                if e.msg == KEPT_CHANGED:               # kept set changed: nothing posted is carried over
                    form.pop("samples", None)
                    for k in [k for k in form if k.startswith("nm_")]:
                        del form[k]
                r = refused_page(request, src["label"], page_session_form(cfg, user, src, t, form, msg), status)
                if status == 429:                          # a limit (the active-analysis cap): try again later
                    r.headers["Retry-After"] = "60"
                return r
            return await run_in_threadpool(again)
        # the work runs in a task, keyed by the pick: pressing the button twice (or a retried request) joins it
        if not recently_done(plan["tid"]):
            start_create(plan["tid"], user["id"], lambda progress: session_run(cfg, user, src, plan, progress),
                         back=f"/predict/new/session/{src['key']}/")
        return RedirectResponse(f"/predict/preparing/{plan['tid']}/", status_code=303)

    def upload_form(request: Request):
        return ctx.page(request, "Upload images", page_upload_form(cfg, ctx.user(request)), active="predict")

    async def upload_post(request: Request):
        """The no-JavaScript form: one multipart request, at most NOJS_MAX_BYTES."""
        user = ctx.user(request)
        dest = await run_in_threadpool(scratch, cfg, "up")
        sink = None
        try:
            try:
                await run_in_threadpool(CREATE.check_active, cfg, user)          # before a byte is stored
                sink = await receive_upload(request, dest, lambda fields: upload_fields(cfg, user, fields, True))
                ps_dir = await run_in_threadpool(upload_submit, cfg, user, dest, sink)
            except OSError as e:                     # disk full, a locked file, a path too long: nothing is kept
                raise Refusal(os_error_text(e), 500, tech=f"{type(e).__name__}: {e}")
        except Refusal as e:
            msg, status = admin_text(e, user), e.status
            form = dict(sink.fields) if sink else {}

            def again():
                return refused_page(request, "Upload images", page_upload_form(cfg, user, form, msg), status)
            return await run_in_threadpool(again)
        finally:
            await run_in_threadpool(shutil.rmtree, dest, True)
        return RedirectResponse(f"/predict/{ps_dir.name}/", status_code=303)

    def raw_form(request: Request):
        user = ctx.user(request)
        if user.get("role") not in RAW_ROLES:
            return forbidden(request, "Converting raw images is for advanced users and the admin.")
        return ctx.page(request, "Convert raw images", page_raw_upload(cfg, user), active="predict")

    async def raw_post(request: Request):
        """The raw form without JavaScript: refused (raw files go one per request, which needs the page's script).
        A small post (no files to speak of) gets its typed fields back; a big one is not read."""
        user = ctx.user(request)
        if user.get("role") not in RAW_ROLES:
            return await run_in_threadpool(forbidden, request,
                                           "Converting raw images is for advanced users and the admin.")
        form = {}
        try:
            if int(request.headers.get("content-length") or 0) <= NOJS_REFILL_BYTES:
                fd = await request.form(max_files=RAW_MAX_FILES, max_fields=MAX_FIELDS)
                form = {k: v for k, v in fd.items() if isinstance(v, str) and len(v) <= MAX_FIELD_BYTES}
                await fd.close()                         # the posted files are not kept
        except Exception:                                # noqa: BLE001 - a bad body just gets an empty form
            form = {}
        return await run_in_threadpool(lambda: refused_page(request, "Convert raw images", page_raw_upload(
            cfg, user, "Raw files are sent one at a time, which needs JavaScript. Turn JavaScript on and try again "
                       "(nothing was kept).", form), 400))

    def raw_sessions_page(request: Request):
        user = ctx.user(request)
        if user.get("role") != "admin":
            return forbidden(request, "Picking a raw session is for the admin.")
        return ctx.page(request, "Pick a raw session", page_raw_sessions(cfg, user), active="predict")

    def find_raw(key):
        return next((s for s in raw_sessions(cfg) if s["key"] == key), None)

    def raw_session_page(request: Request):
        user = ctx.user(request)
        if user.get("role") != "admin":
            return forbidden(request, "Picking a raw session is for the admin.")
        s = find_raw(request.path_params["key"])
        if s is None:
            return not_found(request)
        return ctx.page(request, s["label"], page_raw_session(cfg, user, s, raw_montages(s)), active="predict")

    async def raw_session_post(request: Request):
        user = ctx.user(request)
        if user.get("role") != "admin":
            return await run_in_threadpool(forbidden, request, "Picking a raw session is for the admin.")
        s = await run_in_threadpool(find_raw, request.path_params["key"])
        if s is None:
            return await run_in_threadpool(not_found, request)
        fd = await request.form(max_files=0, max_fields=RAW_MAX_FILES + 20)
        form = {k: v for k, v in fd.items() if isinstance(v, str)}
        form["montages"] = [v for v in fd.getlist("montages") if isinstance(v, str)]
        try:
            d = await run_in_threadpool(session_convert, cfg, user, s, form)
        except Refusal as e:
            msg, status = admin_text(e, user), e.status
            return await run_in_threadpool(lambda: refused_page(
                request, s["label"], page_raw_session(cfg, user, s, raw_montages(s), form, msg), status))
        return RedirectResponse(f"/predict/convert/{d.name}/", status_code=303)

    def convert_page(request: Request):
        user = ctx.user(request)
        d, meta, st = load_conversion(cfg, user, ctx.can_see, request.path_params["cid"])
        if d is None:
            return not_found(request)
        if st.get("state") in ("queued", "running"):
            kick(cfg)
            ensure_watcher(cfg)
            st = _read_json(d / PS.STATUS) or st
        saved = {"saved": True} if request.query_params.get("saved") == "1" else None
        return ctx.page(request, "Raw conversion", page_convert(cfg, user, ctx.can_see, d, meta, st, saved),
                        active="predict")

    async def convert_details(request: Request):
        """The Session details form on the conversion page (its owner or the admin: load_conversion's can_see)."""
        user = ctx.user(request)
        d, meta, st = await run_in_threadpool(load_conversion, cfg, user, ctx.can_see, request.path_params["cid"])
        if d is None:
            return await run_in_threadpool(not_found, request)
        fd = await request.form(max_files=0, max_fields=4 * US.MAX_GRIDS + 10)
        form = {k: v for k, v in fd.items() if isinstance(v, str)}
        res = await run_in_threadpool(US.save_conversion_details, cfg, user, d, meta, form)
        if res.get("ok"):
            return RedirectResponse(f"/predict/convert/{d.name}/?saved=1#cj-us-details", status_code=303)
        err = res["error"] + (f"\n(admin) {res['tech']}" if res.get("tech") and user.get("role") == "admin" else "")
        return await run_in_threadpool(lambda: refused_page(
            request, "Raw conversion", page_convert(cfg, user, ctx.can_see, d, meta, st, {"error": err}),
            res.get("status", 400)))

    def convert_action(kind):
        def act(request: Request):            # sync: a kill or a few file changes under the runner lock
            user = ctx.user(request)
            if user.get("role") != "admin":
                return forbidden(request, "Only the admin can stop, retry or clean up a conversion.")
            d, meta, _st = load_conversion(cfg, user, ctx.can_see, request.path_params["cid"])
            if d is None:
                return not_found(request)
            try:
                if kind == "stop":
                    stop_conversion(cfg, d)
                elif kind == "retry":
                    retry_conversion(cfg, d, meta)
                elif kind == "stitch":
                    queue_stitch(cfg, d, meta)
                elif kind == "register":
                    register_again(cfg, d, meta)
                else:
                    delete_inputs_now(cfg, d)
            except Refusal as e:
                return refused_page(request, "Raw conversion", wrap(
                    f'<div class="cj-predict"><p class="cj-mut"><a href="/predict/convert/{esc(d.name)}/">← Back</a>'
                    f'</p>{msg_box(e.msg)}</div>'), e.status)
            return RedirectResponse(f"/predict/convert/{d.name}/", status_code=303)
        return act

    def set_page(request: Request):
        user = ctx.user(request)
        d, req = load_set(cfg, user, ctx.can_see, request.path_params["ps"])
        if d is None:
            return not_found(request)
        srcs, _ = sources_safe(cfg, user, ctx.can_see)
        return ctx.page(request, tab_title(set_title(req, srcs, user.get("role") == "admin")),
                        page_set(cfg, user, d, req, ctx.can_see), active="predict")

    def prep_load(request: Request):
        tid = request.path_params["tid"]
        t = create_task(tid) if CREATE_ID_RE.match(tid) else None
        user = ctx.user(request)
        return (tid, t) if t is not None and t["owner"] == user.get("id") else (tid, None)

    def prep_page(request: Request):
        tid, t = prep_load(request)
        if t is None:
            return not_found(request)
        if t["state"] == "done":
            return RedirectResponse(f"/predict/{t['ps']}/", status_code=303)
        return ctx.page(request, "Starting your analysis", page_prep(t, tid, ctx.user(request).get("role") == "admin"),
                        active="predict")

    def api_prep(request: Request):
        tid, t = prep_load(request)
        if t is None:
            return ctx.json({"error": "not found"}, 404)
        return ctx.json(prep_view(t, tid, ctx.user(request).get("role") == "admin",
                                  request.query_params.get("html") == "1"))

    def prep_retry(request: Request):
        """After a failed create from an upload: run it again from the kept upload."""
        user = ctx.user(request)
        tid, t = prep_load(request)
        if t is None or not t.get("upload"):
            return not_found(request)
        if t["state"] == "failed":
            d, meta = load_upload(cfg, user, tid)
            if d is None:
                return refused_page(request, "Starting your analysis", wrap(
                    '<div class="cj-predict"><h1>Upload gone</h1><p>These images are no longer kept (an hour passed). '
                    'Please upload them again.</p><p><a href="/predict/new/upload/">Upload images</a></p></div>'), 410)
            try:
                upload_create(cfg, user, d, meta)
            except Refusal as e:
                return refused_page(request, "Starting your analysis", wrap(
                    f'<div class="cj-predict">{msg_box(admin_text(e, user))}'
                    f'<p><a href="/predict/new/upload/">Upload images</a></p></div>'), e.status)
        return RedirectResponse(f"/predict/preparing/{tid}/", status_code=303)

    async def rename(request: Request):
        user = ctx.user(request)
        d, req = await run_in_threadpool(load_set, cfg, user, ctx.can_see, request.path_params["ps"])
        if d is None:
            return await run_in_threadpool(not_found, request)
        fd = await request.form(max_files=0, max_fields=20)
        raw = fd.get("name")
        try:
            await run_in_threadpool(CREATE.rename_set, cfg, user, d, raw if isinstance(raw, str) else "")
        except (Refusal, OSError) as e:
            msg, status = (e.msg, e.status) if isinstance(e, Refusal) else (
                "This analysis can’t be renamed right now. Try again later.", 503)
            return await run_in_threadpool(lambda: refused_page(
                request, "Rename", wrap(f'<div class="cj-predict">{msg_box(msg)}'
                                        f'<p><a href="/predict/{esc(d.name)}/">Back</a></p></div>'), status))
        return RedirectResponse(f"/predict/{d.name}/", status_code=303)

    def action(kind):
        def act(request: Request):            # sync: file checks and a thread start, no long work here
            user = ctx.user(request)
            d, req = load_set(cfg, user, ctx.can_see, request.path_params["ps"])
            if d is None:
                return not_found(request)
            ps = d.name
            st = _read_json(d / PS.STATUS) or {}
            back = RedirectResponse(f"/predict/{ps}/", status_code=303)
            if kind == "start":
                restart(cfg, d)                    # only when can_restart(); marks the set before spawning
                return back
            if st.get("state") != "done":
                return back
            if kind == "review":
                pkg = SETS.package_dir(cfg, ps)
                if pkg is not None:
                    return RedirectResponse(PAGES.review_go(pkg.name), status_code=303)
                start_task("review", ps, lambda: build_review(cfg, d))
            else:
                res, pkg = results_file(d), SETS.package_dir(cfg, ps)
                if res is None or edits_mtime(pkg) > res.stat().st_mtime:
                    start_task("results", ps, lambda: build_results(cfg, d, ps))
            return back
        return act

    # ---- JSON
    def api_mine(request: Request):
        user = ctx.user(request)
        srcs, _ = sources_safe(cfg, user, ctx.can_see)
        items = [item_of(cfg, user, d, req, st, srcs) for d, req, st in list_sets(cfg, user, ctx.can_see)]
        return ctx.json({"schema": SCHEMA_MINE, "items": items})

    def api_set(request: Request):
        user = ctx.user(request)
        d, req = load_set(cfg, user, ctx.can_see, request.path_params["ps"])
        if d is None:
            return ctx.json({"error": "not found"}, 404)
        return ctx.json(set_view(cfg, user, d, req, request.query_params.get("html") == "1"))

    def api_convert(request: Request):
        user = ctx.user(request)
        d, meta, st = load_conversion(cfg, user, ctx.can_see, request.path_params["cid"])
        if d is None or agent_mismatch(request, meta.get("made_via")):
            return ctx.json({"error": "not found"}, 404)
        if st.get("state") in ("queued", "running"):
            kick(cfg)
            ensure_watcher(cfg)
            st = _read_json(d / PS.STATUS) or st
        fr = conv_fragments(cfg, user, ctx.can_see, d, meta, st)
        out = {"schema": SCHEMA_CONVERT_VIEW, "id": d.name, "session": meta.get("session"), "state": st.get("state"),
               "n_tiles": st.get("n_tiles"), "warnings": st.get("warnings") or [], "live": fr["live"]}
        if request.query_params.get("html") == "1":
            out["html"] = {"status": fr["status"], "result": fr["result"]}
        if st.get("state") == "done" and (d / CONVERT_TILES).is_dir():     # the key its tiles are picked by (an
            out["key"] = PSRC.source_key(d / CONVERT_TILES)                # agent: session_form / predict_session)
        return ctx.json(out)

    # ---- the upload API (one file, or one piece of a file, per request)
    async def read_body(request: Request, cap: int):
        try:
            if int(request.headers.get("content-length") or 0) > cap:
                raise Refusal("Request too big.", 413)
        except ValueError:
            raise Refusal("Bad request.", 400)
        buf = bytearray()
        async for chunk in request.stream():
            buf += chunk
            if len(buf) > cap:
                raise Refusal("Request too big.", 413)
        try:
            return json.loads(bytes(buf) or b"{}")
        except ValueError:
            raise Refusal("Bad request (not JSON).", 400)

    async def api_upload_start(request: Request):
        user = ctx.user(request)
        try:
            body = await read_body(request, START_MAX_BYTES)
            return ctx.json(await run_in_threadpool(upload_start, cfg, user, body, PA.request_via(request)))
        except Refusal as e:
            return refusal_json(e, user, True)
        except OSError as e:
            return refusal_json(Refusal(os_error_text(e), 500, tech=f"{type(e).__name__}: {e}"), user, True)

    def upload_lock(uid):
        lock = _UP_ALOCKS.get(uid)
        if lock is None:
            lock = _UP_ALOCKS[uid] = asyncio.Lock()
        return lock

    GONE = {"error": "This upload is gone (finished, refused, or unused for an hour). Start again.", "dropped": True}

    def own_upload(request: Request, uid):
        """load_upload (the person's own), and for an agent only an upload that agent started."""
        d, meta = load_upload(cfg, ctx.user(request), uid)
        if d is None or agent_mismatch(request, meta.get("owner_via")):
            return None, None
        return d, meta

    def api_upload_state(request: Request):
        d, meta = own_upload(request, request.path_params["uid"])
        if d is None:
            return ctx.json(GONE, 404)
        return ctx.json(upload_state(d, meta))

    async def api_upload_file(request: Request):
        user, uid = ctx.user(request), request.path_params["uid"]
        d, _ = await run_in_threadpool(own_upload, request, uid)
        if d is None:
            return ctx.json(GONE, 404)
        async with upload_lock(uid):
            try:
                return ctx.json(await receive_piece(request, d, uid))
            except Resync:
                meta = await run_in_threadpool(_read_json, d / UPLOAD_META)
                if not isinstance(meta, dict):
                    return ctx.json(GONE, 404)
                return ctx.json(upload_state(d, meta), 409)
            except ClientDisconnect:                     # the page resumes from what arrived
                return Response(status_code=400)
            except Refusal as e:
                # a size or content violation (or a full disk) ends the session; a bad request leaves it as it is
                drop = e.drop or e.status == 507
                if drop:
                    await run_in_threadpool(drop_upload, d, uid)
                return refusal_json(e, user, drop)
            except OSError as e:
                await run_in_threadpool(drop_upload, d, uid)
                return refusal_json(Refusal(os_error_text(e), 500, tech=f"{type(e).__name__}: {e}"), user, True)

    async def api_upload_create(request: Request):
        user, uid = ctx.user(request), request.path_params["uid"]
        d, meta = await run_in_threadpool(own_upload, request, uid)
        if d is None:
            return ctx.json(GONE, 404)
        async with upload_lock(uid):
            try:
                href = await run_in_threadpool(upload_create, cfg, user, d, meta)
            except Refusal as e:
                keep = e.status in (409, 429, 503)       # files still to send, a limit, a busy GPU: try again later
                if not keep:
                    await run_in_threadpool(drop_upload, d, uid)
                return refusal_json(e, user, not keep)
            except OSError as e:                         # the files stay staged: pressing the button again retries
                return refusal_json(Refusal(os_error_text(e), 500, tech=f"{type(e).__name__}: {e}"), user, False)
            if meta["kind"] != "tiles":                  # a tiles upload is dropped by its task, after it succeeds
                await run_in_threadpool(drop_upload, d, uid)
        return ctx.json({"href": href})

    async def api_upload_cancel(request: Request):
        user, uid = ctx.user(request), request.path_params["uid"]
        d, _ = await run_in_threadpool(own_upload, request, uid)
        if d is None:
            return ctx.json(GONE, 404)
        if create_running(uid):                          # its images are being added to an analysis
            return ctx.json({"error": "These images are being prepared; they can’t be cancelled now.",
                             "dropped": False}, 409)
        async with upload_lock(uid):
            await run_in_threadpool(drop_upload, d, uid)
        return ctx.json({"ok": True})

    def script(request: Request):
        r = Response(JS, media_type="text/javascript; charset=utf-8")
        r.headers["Cache-Control"] = "no-cache"
        return r

    # ---- conversion settings: the readout, the preview, the re-conversion, the tiles zip (piece CU)
    def cs_conv(request):
        """-> (user, d, meta, st) when this person may see the conversion; else None (the same 404 for a missing one
        and one that is not theirs)."""
        user = ctx.user(request)
        d, meta, st = load_conversion(cfg, user, ctx.can_see, request.path_params["cid"])
        return (user, d, meta, st) if d is not None else None

    def prev_base() -> Path:
        return raw_base(cfg) / PORTAL_TMP / CPREV

    def prune_previews(max_age_s=PREV_KEEP_S):
        """Remove preview folders older than max_age_s (and empty conversion folders)."""
        base = prev_base()
        try:
            cdirs = [p for p in base.iterdir() if p.is_dir()]
        except OSError:
            return
        for c in cdirs:
            for t in [p for p in c.iterdir() if p.is_dir()]:
                try:
                    if time.time() - t.stat().st_mtime > max_age_s:
                        shutil.rmtree(t, ignore_errors=True)
                except OSError:
                    pass
            try:
                c.rmdir()
            except OSError:
                pass

    def api_recipe(request: Request):
        """The readout of one recipe (the raw form's script fetches it): JSON {html, summary}."""
        user = ctx.user(request)
        if user.get("role") not in RAW_ROLES:
            return ctx.json({"error": "not allowed"}, 403)
        repo, _why = processing_repo(cfg)
        name = request.path_params["name"]
        if repo is None or (name != "default" and name not in RAW.recipes(repo)):
            return ctx.json({"error": "not found"}, 404)
        doc, err = CS.describe_recipe(_converter_cmd(), repo, name, recipe_file_name(name))
        err = _plain(err, [repo, CS.HOME]) if err and user.get("role") != "admin" else err
        return ctx.json({"html": recipe_readout(doc, err, name), "summary": settings_summary(doc) if doc else ""})

    def convert_settings(request: Request):
        got = cs_conv(request)
        if got is None:
            return ctx.json({"error": "not found"}, 404)
        user, d, meta, st = got
        repo, why = processing_repo(cfg)
        if repo is None:
            return ctx.json({"html": '<h2 class="cj-sec">Settings</h2><div class="cj-card"><p class="cj-mut">'
                                     'The settings can’t be read: raw conversion is not set up on this server.</p></div>',
                             "state": st.get("state")})
        doc, err = CS.describe_conv(_converter_cmd(), repo, d)
        rec = bare_recipe(cfg, repo, meta, doc) if doc is not None else None
        return ctx.json({"html": convset_fragment(cfg, user, d, meta, st, doc, err, rec), "state": st.get("state")})

    async def convert_preview(request: Request):
        got = await run_in_threadpool(cs_conv, request)
        if got is None:
            return ctx.json({"error": "not found"}, 404)
        user, d, meta, st = got
        if user.get("role") not in RAW_ROLES:
            return ctx.json({"error": "Previews are for advanced users and the admin."}, 403)
        repo, why = processing_repo(cfg)
        if repo is None:
            return ctx.json({"error": why}, 503)
        try:
            body = json.loads((await request.body())[:20000] or b"{}")
            if not isinstance(body, dict):
                raise ValueError("not an object")
            params = CS.clean_params(body.get("params") or {})
            if "bandpass_nm" in params:                          # the recipe's own rules (method, upper cutoff)
                CS.check_against(params, (await run_in_threadpool(CS.describe_conv, _converter_cmd(), repo, d))[0])
        except (ValueError, CS.ParamError) as e:
            return ctx.json({"error": str(e) if isinstance(e, CS.ParamError) else "Bad request."}, 400)
        if st.get("state") != "done":
            return ctx.json({"error": "The conversion is not finished."}, 409)
        gone = await run_in_threadpool(raw_kept_reason, d, meta, st)
        if gone:
            return ctx.json({"error": gone}, 409)
        tile = body.get("tile")
        if not isinstance(tile, str) or tile not in await run_in_threadpool(tile_names, d):
            return ctx.json({"error": "That tile is not in this conversion."}, 400)
        if CS.preview_busy():
            r = ctx.json({"error": "A preview is already being made. Try again in a few seconds."}, 429)
            r.headers["Retry-After"] = "10"
            return r
        token = secrets.token_hex(6)
        out = prev_base() / d.name / token

        def work():
            return CS.run_preview(_converter_cmd(), repo, d, tile, params, out, before=prune_previews)
        try:
            doc, err = await run_in_threadpool(work)
        except CS.PreviewBusy:
            r = ctx.json({"error": "A preview is already being made. Try again in a few seconds."}, 429)
            r.headers["Retry-After"] = "10"
            return r
        if doc is None or not all((out / n).is_file() for n in PREV_FILES):
            await run_in_threadpool(shutil.rmtree, out, True)
            shown = _plain(str(err or "no image"), [d, repo, CS.HOME, prev_base()])
            return ctx.json({"error": "The preview could not be made: " + shown}, 502)
        u = f"/predict/convert/{d.name}/preview/{token}"
        return ctx.json({"ok": True, "tile": tile, "before": f"{u}/before.png", "after": f"{u}/after.png",
                         "effective": doc.get("params_effective"), "recipe_params": doc.get("params_recipe"), "conversion_params": doc.get("params_conversion"),
                         "overrides": doc.get("params_overrides"), "compute": doc.get("compute"),
                         "warnings": [_plain(str(w)[:300], [d, repo, CS.HOME, prev_base()])
                                      for w in (doc.get("warnings") or [])[:5]],
                         "seconds": doc.get("seconds")})

    def preview_file(request: Request):
        got = cs_conv(request)
        pp = request.path_params
        if got is None or not PREV_TOKEN_RE.match(pp["token"]) or pp["name"] not in PREV_FILES:
            return not_found(request)
        p = prev_base() / got[1].name / pp["token"] / pp["name"]
        if not p.is_file():
            return not_found(request)
        r = FileResponse(p, media_type="image/png")
        r.headers["Cache-Control"] = "no-store"
        return r

    def link_or_copy(src, dst):
        """Hard link (the new conversion shares the bytes: no extra disk, and the old one's sweep can't take them away),
        else a copy."""
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)

    def do_reconvert(user, d: Path, meta: dict, st: dict, form: dict) -> Path:
        if st.get("state") != "done":
            raise Refusal("Only a finished conversion can be converted again with other settings.", 409)
        why = raw_kept_reason(d, meta, st)
        if why:
            raise Refusal(why, 409)
        repo, whynot = processing_repo(cfg)
        if repo is None:
            raise Refusal(f"Raw conversion is not set up: {whynot}", 503)
        raw = {k: form[k] for k in CS.RANGES if k in form}
        raw["denoise"] = "true" if "denoise" in form else "false"
        doc, _err = CS.describe_conv(_converter_cmd(), repo, d)
        try:
            params = CS.clean_params(raw)                       # every value as previewed, not only the changed ones
            CS.check_against(params, doc)
        except CS.ParamError as e:
            raise Refusal(str(e), 400)
        if doc is not None:
            used = conv_values(doc, meta)
            if not CS.diff(used, {**used, **params}):
                raise Refusal("These are the settings this conversion already used: change a slider first.", 400)
        owner = str(meta.get("owner") or user["id"])             # an admin's re-conversion stays the owner's
        check_conv_cap(cfg, {"id": owner})
        fields = {"session": meta.get("session"), "recipe": meta.get("recipe"), "gpu": False}
        src = meta.get("source") or {}
        kw = dict(params=params, derived_from=d.name, owner=owner, made_by=user["id"],
                  nbytes=meta.get("input_bytes"))
        stage = None
        try:
            if src.get("kind") == "upload":
                base = raw_base(cfg) / PORTAL_TMP
                base.mkdir(parents=True, exist_ok=True)
                probe = base / f".lp{secrets.token_hex(4)}"
                first = next((q for q in (d / CONVERT_INPUT).iterdir() if q.is_file()), None)
                try:                                              # can the new input be hard links of the old one?
                    if first is None:
                        raise OSError("no input file")
                    os.link(first, probe)
                except OSError:                                   # no (another drive): it is a real copy; room for it?
                    RUNNER.check_disk_for(raw_root(cfg), input_bytes(meta), factor=1.0, what="to copy the raw files")
                finally:
                    try:
                        probe.unlink()
                    except OSError:
                        pass
                stage = base / f"sel_rc{secrets.token_hex(4)}"
                shutil.copytree(d / CONVERT_INPUT, stage, copy_function=link_or_copy)
                return new_conversion(cfg, user, fields, source=src, upload_files=stage,
                                      skipped=meta.get("skipped"), **kw)
            return new_conversion(cfg, user, fields, source=src, inputs=list(meta.get("inputs") or []),
                                  sample_key=meta.get("sample_key"), **kw)
        finally:
            if stage is not None and stage.exists():
                shutil.rmtree(stage, ignore_errors=True)

    async def convert_reconvert(request: Request):
        got = await run_in_threadpool(cs_conv, request)
        if got is None:
            return await run_in_threadpool(not_found, request)
        user, d, meta, st = got
        if user.get("role") not in RAW_ROLES:
            return await run_in_threadpool(forbidden, request, "Converting raw images is for advanced users and the admin.")
        fd = await request.form(max_files=0, max_fields=20)
        form = {k: v for k, v in fd.items() if isinstance(v, str) and len(v) <= 40}
        try:
            nd = await run_in_threadpool(do_reconvert, user, d, meta, st, form)
        except Refusal as e:
            return await run_in_threadpool(lambda: refused_page(request, "Raw conversion", wrap(
                f'<div class="cj-predict"><p class="cj-mut"><a href="/predict/convert/{esc(d.name)}/">← Back</a></p>'
                f'{msg_box(admin_text(e, user))}</div>'), e.status))
        return RedirectResponse(f"/predict/convert/{nd.name}/", status_code=303)

    def convert_tiles_zip(request: Request):
        got = cs_conv(request)
        if got is None:
            return not_found(request)
        user, d, _meta, st = got
        man = d / CONVERT_TILES / PSRC.MANIFEST
        if st.get("state") != "done" or not man.is_file():
            return not_found(request)
        from portal import zipstream
        from portal.modules import gallery as GAL            # here, not at the top: gallery imports this module
        entries = [(f"{d.name}/MANIFEST.csv", man)] + [(f"{d.name}/{p.name}", p)
                                                       for p in sorted((d / CONVERT_TILES).glob("*.png"))]
        slot = GAL.take_slot(str(user.get("id") or ""))
        if slot is None:
            r = Response("Two downloads are already running for you. Wait for one to finish.", status_code=429,
                         media_type="text/plain")
            r.headers["Retry-After"] = "30"
            return r
        try:
            stream = GAL.Progress(zipstream.stream_zip(entries), slot)
        except BaseException:
            slot.release()
            raise
        r = GAL.SlotResponse(stream, slot, media_type="application/zip")
        r.headers["Content-Disposition"] = f'attachment; filename="{d.name}_tiles.zip"'
        r.headers["Cache-Control"] = "no-store"
        return r

    try:                                   # previews left by a portal that stopped
        shutil.rmtree(prev_base(), ignore_errors=True)
    except Exception:                      # noqa: BLE001 - no root configured yet
        pass

    return [Route(PREFIX + "/", index, methods=["GET"]),
            Route(PREFIX + "/predict.js", script, methods=["GET"]),
            Route(PREFIX + "/api/recipe/{name}", api_recipe, methods=["GET"]),
            Route(PREFIX + "/convert/{cid}/settings", convert_settings, methods=["GET"]),
            Route(PREFIX + "/convert/{cid}/preview", convert_preview, methods=["POST"]),
            Route(PREFIX + "/convert/{cid}/preview/{token}/{name}", preview_file, methods=["GET"]),
            Route(PREFIX + "/convert/{cid}/reconvert", convert_reconvert, methods=["POST"]),
            Route(PREFIX + "/convert/{cid}/tiles.zip", convert_tiles_zip, methods=["GET"]),
            Route(PREFIX + "/new/", new, methods=["GET"]),
            Route(PREFIX + "/new/session/{key}/", session_form, methods=["GET"]),
            Route(PREFIX + "/new/session/{key}/", session_post, methods=["POST"]),
            Route(PREFIX + "/new/upload/", upload_form, methods=["GET"]),
            Route(PREFIX + "/new/upload/", upload_post, methods=["POST"]),
            Route(PREFIX + "/new/raw/", raw_form, methods=["GET"]),
            Route(PREFIX + "/new/raw/", raw_post, methods=["POST"]),
            Route(PREFIX + "/new/raw/sessions/", raw_sessions_page, methods=["GET"]),
            Route(PREFIX + "/new/raw/sessions/{key}/", raw_session_page, methods=["GET"]),
            Route(PREFIX + "/new/raw/sessions/{key}/", raw_session_post, methods=["POST"]),
            Route(PREFIX + "/convert/{cid}/", convert_page, methods=["GET"]),
            Route(PREFIX + "/convert/{cid}/stop", convert_action("stop"), methods=["POST"]),
            Route(PREFIX + "/convert/{cid}/retry", convert_action("retry"), methods=["POST"]),
            Route(PREFIX + "/convert/{cid}/inputs", convert_action("inputs"), methods=["POST"]),
            Route(PREFIX + "/convert/{cid}/stitch", convert_action("stitch"), methods=["POST"]),
            Route(PREFIX + "/convert/{cid}/register", convert_action("register"), methods=["POST"]),
            Route(PREFIX + "/convert/{cid}/details", convert_details, methods=["POST"]),
            Route(PREFIX + "/preparing/{tid}/", prep_page, methods=["GET"]),
            Route(PREFIX + "/preparing/{tid}/retry", prep_retry, methods=["POST"]),
            Route(PREFIX + "/api/preparing/{tid}", api_prep, methods=["GET"]),
            Route(PREFIX + "/api/mine", api_mine, methods=["GET"]),
            Route(PREFIX + "/api/set/{ps}", api_set, methods=["GET"]),
            Route(PREFIX + "/api/convert/{cid}", api_convert, methods=["GET"]),
            Route(PREFIX + "/api/upload", api_upload_start, methods=["POST"]),
            Route(PREFIX + "/api/upload/{uid}", api_upload_state, methods=["GET"]),
            Route(PREFIX + "/api/upload/{uid}/file", api_upload_file, methods=["POST"]),
            Route(PREFIX + "/api/upload/{uid}/create", api_upload_create, methods=["POST"]),
            Route(PREFIX + "/api/upload/{uid}/cancel", api_upload_cancel, methods=["POST"]),
            Route(PREFIX + "/{ps}/", set_page, methods=["GET"]),
            Route(PREFIX + "/{ps}/review", action("review"), methods=["POST"]),
            Route(PREFIX + "/{ps}/results", action("results"), methods=["POST"]),
            Route(PREFIX + "/{ps}/rename", rename, methods=["POST"]),
            Route(PREFIX + "/{ps}/start", action("start"), methods=["POST"])]
