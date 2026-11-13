# Agents on the CryoEV platform

How an AI agent (a Claude Code session, or any script acting for a person) puts data in, runs analyses, pulls data
down and reviews, without leaving a mess. The short version:

- **Use `python -m cryoev`** (run from the repo root) for what it covers, and the tools' own commands for the rest.
  Never write a record, a JSONL line or a data file by hand.
- **Every write is a dry run first.** `cryoev` writes nothing without `--do`; the wrapped scripts have `--dry-run`.
- **Everything an agent does is credited as "claude for pujan"**: the person it acts for, plus the agent (see
  Identity).
- **Some things are never automated** (below). The CLI refuses them and points here.

Machines are named by codename only (this repo is public): **home-laptop** (home base: registry, review servers,
portal, job queue), **gpu-desktop** (the shared GPU PC that trains; may be off), **queue-share** (the desktop's queue
folder, `\\<gpu-desktop>\cj`), **lab-login** (the lab's login node). Data paths are written `<CryoAI-laptop>`,
`<CryoAI>`; `data_utils/config.py` finds them.

> **Remote agents** (not on the laptop) use the portal's read-only HTTP API with a personal token: see [API.md](API.md).

## Identity: who did it, and through which agent

> **Live since 2026-11-12:** `CRYOEV_VIA` / `CRYOEV_FOR` are set for agent sessions on home-laptop. Only run
> registry code from an up-to-date checkout of `main` (or via `python -m cryoev`): older registry code doesn't know the
> `via` field and would re-append stamped lines. The registry's `FORMAT.json` (`min_reader`) makes newer code refuse
> to run on a registry it can't read; code from before this guard has no such check, so never use an old worktree.

| Variable | Meaning | Set by |
|---|---|---|
| `CRYOEV_FOR` | the person the agent acts for: a `people.json` id (`annotation/people.json`) | the agent harness (Claude Code user settings `env`) |
| `CRYOEV_VIA` | the agent: a short lower-case id, e.g. `claude` (`^[a-z][a-z0-9_-]{0,31}$`; anything else is an error) | the same |

- An explicit `--who` always wins over `CRYOEV_FOR`. `registry/actor.py` is the one reader (`who()`, `via()`,
  `label()`).
- Because the harness sets them for the whole session, **`!` commands typed by the user inside an agent session are
  stamped too.** A command run from the user's own shell is not.
- Servers never carry them: the portal, the review servers and every `tools/services.py` child drop both variables
  at start, so a service restarted from an agent session never credits lab members' clicks to the agent.

**What gets stamped where:**

| Write | Field | Where it shows |
|---|---|---|
| registry lines (`registry.py import`, `set-status`, `job new`, `jobs/sync.py pull`; prediction sets from `jobs/predict_set.py` / `predict_local.py` via the registry's env default) | `via` on the line; `status_by` + `status_via` for an explicit status | history page and portal Data / Jobs pages: "claude for pujan" with an **AI** tag |
| label versions (`apply_review_edits.py`) | `version.json` `created_by` / `created_via` | Data page, history page |
| feedback status / reply (`cryoev feedback`) | `who` + `via` on the event in the raw feedback file | the new status / reply on `/feedback/all` (that page does not show `via`); the agent shows in Admin, Recent actions (the action log) |
| every `cryoev ... --do` | two lines in `<CryoAI-laptop>/logs/agent_actions.jsonl` | Admin, **Recent actions**: "claude for pujan" **AI**, door **"laptop CLI (unverified)"** |

The action log is the portal audit's shape, `{t, actor, action, target, detail: {via, phase, args, result}}`: a
`requested` line before the action and a `result` line after (rc and a short result; never secrets, never file
contents, never a reply's text). The identity in it is **declared by the caller** (env / `--who`), not checked by
the portal; that is why the Admin page labels this door "unverified". The portal reads the file through
`CJ_AUDIT_EXTRA` (services config).

## Golden rules

1. **Never hand-edit `*.jsonl`** (registry, feedback, audit, action log) or `version.json` / `status.json`. The
   registry is append-only; the importers are the safety net (`cryoev register`).
2. **Dry run first, read the plan, then `--do`.** A dry run prints exactly what would be written; for a wrapped
   script without its own dry run it prints the command and says the outputs are not checked.
3. **Nothing under Box**, read or write. Pull destinations are refused under Box / `Cryo-image-processing`
   (`data_utils/path_guard.py`) and inside the data folders or the repo (compared by real path); pull sources under
   them are refused too.
4. **Blind audits stay blind.** Never open a `_work_*` folder under `<CryoAI-laptop>/review` (a compare package's
   answer key, `decoys.json`) and never read individual review decisions while a package is not applied. `cryoev
   review progress` gives counts only and refuses the rest.
5. **After any code change**, run `python tools/test_gate.py --affected` (docs/TOOLS.md, "Test gate").
6. **Ask the user** before anything on the never-automated list, before restarting a service, and before anything
   that sends work to gpu-desktop.

## Never automated

The CLI has no command for these and refuses look-alikes (`cryoev push`, `cryoev predict push`, `cryoev
set-production`, `cryoev roboflow ...`: exit 3, with a pointer here). The user runs them himself, each time.

| Action | Why | Who does it, how |
|---|---|---|
| Roboflow uploads, relabels, deletions | shared annotation project; edits can exist only there | the user, after `annotation/roboflow_snapshot.py` |
| Setting a model to production / promoting | the lab's default model | the user: `python registry/registry.py set-status <model> production --who pujan`; the registry refuses it with `CRYOEV_VIA` set |
| `git push` | deploys code to gpu-desktop; public repo | the user, after the pre-push scan (`.githooks/pre-push`) |
| Sending work to gpu-desktop (`jobs/sync.py push`, `push-predict`, `local`) | uses the shared GPU | the user's OK each time; the agent prepares the bundle and stops |
| Anything under Box | not ours to change | nobody |

## The CLI: `python -m cryoev`

| Command | Kind | Does (wraps) |
|---|---|---|
| `status [--share \\<gpu-desktop>\cj] [--json]` | read | services (`tools/services.py` status), desktop jobs (the monitor's `live.json` snapshot, or the queue itself via `jobs/sync.py` with `--share`), review packages with cards decided / total (`portal/titles.py`, counts only), open feedback |
| `find <text> [--entity E] [--status S] [--json]` | read | registry ids containing the text (`registry.current`) |
| `show <id> [--json]` | read | `registry/registry.py show`; `--json` = the current record only, the same JSON |
| `lineage <id> [--json]` | read | the Data page's lineage walk back to the imaging sessions (`portal/modules/data.py`) |
| `review progress <package> [--json]` | read | counts only; refuses `_work_*` and `--decisions` while the package is not applied |
| `feedback list [--json]` | read | `python -m portal.feedback_store list` |
| `feedback status <n> "<status>" [--do]` | write | New / Planned / "In progress" / Done / "Won't do"; backs the file up to `~/.cryoev/backups/feedback/<UTC>.jsonl` first |
| `feedback reply <n> "<text>" [--do]` | write | a reply its author sees; backup first |
| `pull labels\|predictions\|results\|dataset <id> --out DIR [--with-images] [--do]` | write (outside the data) | copies what the record points to into `DIR/<id>/` + `MANIFEST.sha256` + `pull.json`; never overwrites |
| `register [--do]` | write | `registry.py import`, then `registry/history_page.py` **regenerates** `<registry>/history.html`. The import may **restate production** (the user's recorded decision for the production model, when its current line lost it); the dry run says so |
| `predict new <predict_set create args> [--do]` | write | `jobs/predict_set.py create` (dry run = its `--dry-run`: every check, nothing written). Only exact create options (no abbreviations); `--dry-run` is the CLI's own; values under Box refused |

Common flags: `--do` (really do it), `--who <id>` (else `CRYOEV_FOR`; required with `--do`), `--json` (reads).
Exit codes: 0 ok; 1 not found (unknown id / package, no match) or done but its result line could not be logged;
2 refused or failed (bad arguments, a guard, an error); 3 refused by rule (never automated, blind audit).

## Recipes

Each recipe says what is recorded and where it shows. Details live in the linked docs; they are not repeated here.

### Put data in: a new imaging session to a prediction set
1. Tiles: the lab's processing workspace makes `3d. Tiles no scalebar ...` per session, or raw montages go through
   `python -m jobs.convert_raw --in <files> --out <new dir>` ([JOBS.md](JOBS.md), "Raw montages to tiles"). Writes a
   new folder only.
2. The set, dry run then real:
   ```
   python -m cryoev predict new --from-dir "<tiles dir>" --manifest "<tiles dir>\MANIFEST.csv" --session YYYYMMDD --model v5_evnonev_y8m_20261029 --owner pujan --sample-label-from manifest:sample_label --nm-from-manifest --nm-confirmed-by pujan
   python -m cryoev predict new ...same args... --do
   ```
   Recorded: `<CryoAI-laptop>/predict/<ps id>/`, a `prediction_set` registry line (status queued), two action-log
   lines. Shows: the owner's Predict / Results pages, Data page (image sets), Admin Recent actions.

   **Images made in an unusual way** (tiling, resampling, a subset): also give `--note "<text, at most 500
   characters>"` and `--provenance provenance.json`. Both go into `request.json` and the registry record (and stay on
   every later status line); the Data page shows Note, How the images were made, and Derived from. Only the keys
   below are allowed; anything else is refused before anything is written (limits: `jobs/predict_set.py` module
   doc). With `--manifest`, the set keeps a byte-identical `source_manifest.csv`, so per-tile columns (parent image,
   x/y offset) stay with the set.
   ```json
   {"derived_from": [{"kind": "dataset", "id": "<dataset id>"}, {"kind": "model", "id": "v5_evnonev_y8m_20261029"}],
    "source_images": "44 Jul30 full-resolution training images",
    "steps": ["cut 1440 px tiles at native resolution, 20% overlap", "8-bit PNG, no contrast change"],
    "tiling": {"tile_px": 1440, "overlap_px": 288, "overlap_frac": 0.2, "nm_per_px": 0.368, "downsample": 1},
    "script": "<script that cut the tiles>", "git": "<commit>"}
   ```
   `derived_from` kinds: dataset, label_version, session, prediction_set, model, other.
3. Predict on home-laptop: `python -m jobs.predict_local --ps <ps id>` ([JOBS.md](JOBS.md), "Predict jobs").
   Updates `status.json` and the registry line (stamped through the registry's env default). gpu-desktop instead
   (`sync.py push-predict`): the user's call.
4. `python -m cryoev register --do` if anything was made outside the tools.

### Analyse
- Counts per sample: `python -m results --ps <CryoAI-laptop>/predict/<ps id>` writes `<set>/results/`
  (`results.json`, `per_image.csv`, `per_sample.csv`); with a reviewed prediction package add `--reviewed
  <pkg>/reviewed.json` ([REVIEW_TOOL.md](REVIEW_TOOL.md), "Prediction packages"). Shows: the Results page.
- Look things up (read only): `python -m cryoev status`, `find`, `show <id>`, `lineage <id>`.
- Model scores: the registry's `run` / `model` / `inference_check` records (`show`); small score gaps are noise
  (CLAUDE.md, "Agents").

### Pull data down
```
python -m cryoev pull results ps_YYYYMMDD_xxxxxx --out C:\work\pulls          # dry run: file list + size
python -m cryoev pull results ps_YYYYMMDD_xxxxxx --out C:\work\pulls --do
```
Also `pull labels <lv id>`, `pull predictions <ps id> [--with-images]`, `pull dataset <dataset id>`. Recorded:
`<out>/<id>/MANIFEST.sha256` (each copy checked against its source) and `pull.json` (what, from where, who); the
action log. Shows: Admin Recent actions.

### The review loop (lab labeling / compare packages)
1. Build: `python annotation/compare_package.py ... --owner <id> --out <CryoAI-laptop>/review/<package>`
   (usage in the script's docstring). The decoy answer key goes to `<review>/_work_<package>/`: **never read it.**
2. Reviewers decide on cards (phone or desktop): [REVIEW_TOOL.md](REVIEW_TOOL.md). The agent watches counts only:
   `python -m cryoev review progress <package>`.
3. When the user closes the audit: score and drop decoys with `annotation/compare_package.py`'s `strip_decoys()`
   (it reads `_work_`; only with the user's OK, after the review is finished).
4. Apply, dry run first: `python annotation/apply_review_edits.py --base <base coco> --package <package> --dry-run`,
   then without `--dry-run`. Only images marked reviewed are applied unless the user asks for `--all-edits`
   ([LABEL_CHANGES.md](LABEL_CHANGES.md)). Recorded: a new `<CryoAI-laptop>/labels/lv_*/` with `created_by` /
   `created_via`.
5. Dataset: `python data_utils/dataset_from_lv.py --lv <lv id> --split-from "<export>" --output-name <name>
   --class-scheme ev_nonev --export-only drop --register` (script docstring). Then `python -m cryoev register --do`.
   Shows: Data page (label version, dataset, lineage), history page.

### The training loop
1. A train job: `python registry/registry.py job new --type train --where desktop --name <name> --images <list>`
   (stamped from the env).
2. Bundle: `python training/make_job_bundle.py --dataset-dir <dataset> -- --single-run ...` ([JOBS.md](JOBS.md)).
   It writes a thin job zip plus a content-keyed dataset pack (`ds_<sha12>.zip`, sent to the desktop once);
   `--single-zip` is the old one-zip format. Count EVs with the `classes:` map, never raw `n_detections`. Code the
   job needs must be on GitHub main first: **pushing is the user's.**
3. **Stop: the user sends it** (`jobs/sync.py push <job> --bundle <zip> --share \\<gpu-desktop>\cj`).
4. Watch: `python -m cryoev status`, or `python jobs/sync.py status --share \\<gpu-desktop>\cj` (agents may run
   it). The history page's **Live** tab (and the Jobs page) need `jobs/monitor.py` running (a laptop service).
   `sync.py cancel <job>` and `sync.py stop` / `start` (pause / resume the watcher) only with the user's OK.
5. Bring results home: `python jobs/sync.py pull --share \\<gpu-desktop>\cj` (runs the importer and sets the job
   done / blocked, stamped), then `python -m cryoev register --do` to regenerate the history page. Promotion to
   production is the user's.

## For people writing tools agents use
- Script entry points read the identity with `registry/actor.py` (`actor.current(flag)`); servers call
  `actor.strip_env()` in `main()`, never at import.
- New write commands belong in `cryoev/commands/`: a thin wrapper (call the function, or forward argv to `main()`
  and catch `SystemExit`), a dry run that writes nothing, and `common.run_write()` for the action log. Tests clear
  `CRYOEV_*` in `setUp` and hash the temp data before and after a dry run (`cryoev/tests/`).
