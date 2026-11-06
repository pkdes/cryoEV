# HITL platform: iteration 2 to-dos and spec draft

Iteration 1 closed 2026-10-30. It shipped:
- the review tool with phone cards, a name picker and September tags;
- `apply_review_edits.py` → `lv_20261030_0909`;
- the tracking contract, registry and history page;
- `inference_check.py`.

See README "Label review in the browser".

## To-dos
| Owner | Item |
|---|---|
| User | Real review of `prelipo_v5F5_review_v2` (laptop server on port 8766, so it can run next to the sandbox on the default 8765) |
| User OK | Roboflow push of the reviewed label version: re-pull and drift diff against the base pull, then a one-image round trip |
| Build | Job queue + model page (spec below, phase 1), then the lineage graph + image lookup (phase 2) |
| Build | Review tool UI: U4 (big Accept/Reject on prediction-only objects; editing vertices accepts) + the user's tweaks |
| Build | The 6 "suggested changes to files owned by others" in [TRACKING_CONTRACT.md](TRACKING_CONTRACT.md): `inference_check/1` lines, `supersedes` in version.json, data.js `id`/`base`/`model`/`parent`, a `job_bundle/1` line + `bundle_id`, `run.json`, `models.yaml` keys |
| Build | Move the hardcoded `C:\Users\pujan\…` defaults to config/env (the repo is a public fork) |
| Data gaps | JOB.json date format not parsed; grid `replaced_by` for 19 renumbered grids; 28 pre-manifest sessions not in the master CSV |
| Later | SAM, the paused 20261027 sampling report. (Done 2026-11-06: remote access via Cloudflare Tunnel + Access, see portal/README.md; process-raw jobs = portal raw → tiles.) |

Status 2026-11-02:
- `lv_20261030_0909` = 122 September carry-over changes on 28 of the 170 images (81 accepted predictions, 14 drawn, 18 reclass, 5 delete, 4 confirmed). It is **not pushed**. The push waits for the real review.
- No Box copy for now (user).
- The legacy model ids are confirmed as above.
- The v1 review packages, the v1 sandbox and the control scratch were moved to `CryoAI-laptop/archive/superseded_20261102/` and marked `superseded` in the registry. Their edits were automatic September seed copies only.

Iteration 2 shipped 2026-11-02 (local commits, not pushed):
- **Job queue + model page:** `job/1`; revise jobs from packages; dataset image lists. v5 F5 trained on 114/26/39.
- **Review tool U4:** Accept/Reject, vertex edit accepts, live drag, right-click deletes a point, click on the outline adds one. Accept uses the predicted class (user OK). Live in both v2 packages; backup `index.before_20261102.html`.
- `cryoai_v4_review` stays an in-progress revise job (user).
- **Next:** the user's UI mockups (marked-up screenshots or sketches); phase 2 lineage graph + image lookup; the producers logging their own lines; paths to config.

Decided:
- datasets may be built straight from a local `lv_*` (no Roboflow push first);
- only the user promotes a model (v5 F5 = production).

Round 3 shipped 2026-11-02/03 (pushed through 46e6c28; later commits local until the next push OK):
| Part | What | Where |
|---|---|---|
| Paths | Data roots from env or the sibling folders; no `C:\Users\pujan` in code; ntfy topic from `CRYOEV_NTFY_TOPIC` | `data_utils/config.py` |
| Graph + lookup | Columns label sets → datasets → models → review; revise back-edges; "Show history" (also unused exports); image lookup | history page |
| Runs log themselves | `run.json` (finished / error / oom) and a true exit code; the importer reads it | `run_experiments.py` |
| Job runner (our own; MLflow/ClearML trial 11-02 said build) | Desktop queue over a Tailscale share; laptop `sync.py local`; small-GPU guards | [JOBS.md](JOBS.md) |
| Registry | OS write lock per line | `registry.append` |

Round 3b shipped 2026-11-04 (all pushed, through 66ea92d):
| Part | What | Where |
|---|---|---|
| Desktop runner live | `gpu-desktop` set up; watcher as a scheduled task; trains with latest main; **self-updates and restarts** when its code changes | `jobs/watch.py`, [JOBS.md](JOBS.md) |
| Cancel | `sync.py cancel <job>`: a running job stops within ~10 s (partial run kept), a queued one never starts | `jobs/run_job.py`, `jobs/sync.py` |
| Live monitor | `progress.json` each epoch (actual batch); `jobs/monitor.py` → `registry/live/live.json`; **Live tab** on the history page | `training/train_yolo.py`, `jobs/monitor.py`, history page |
| Seeds | `run_experiments.py --seed` (default 42) | `training/run_experiments.py` |
| Review shipping | `apply_review_edits.py` applies **only reviewed images** by default (`--all-edits` = old rule); version.json `reviewed_images` / scope / `supersedes`; "verified N/M" chip | [LABEL_CHANGES.md](LABEL_CHANGES.md), [REVIEW_TOOL.md](REVIEW_TOOL.md) |
| Scoring note | v5 F5 test mask mAP50-95 0.708 / 0.508 is Ultralytics val with `overlap_mask=False` (as trained); the default carves nested layers → EV 0.476 | `models/models.yaml` |

Next (user to rank):
- the user's real review → dataset from `lv_*` (`dataset_from_lv.py`, not built) → Roboflow push;
- desktop setup; a `CryoAI-laptop` backup plan;
- backlog: a cloud GPU worker (Lambda API: launch → bundle → run → copy `d/<job>` back → terminate; hour and cost caps).

## Spec draft: jobs and lineage (not built)
**Problem**
- The history page's status board mixes entity types into one set of lanes, and doesn't say what a job is doing.
- The tree is single-parent, rooted at a Roboflow export.
- The real flow branches and loops: model → auto-labels → revise → new labels → train.

**Most important question:** which labeled image versions trained, validated and tested each model.

**Concepts**
- **Label set:** labels on a set of images at one moment. Kinds: Roboflow version (export), local edits (`lv_*`), auto-labels (model predictions).
- **Dataset:** label set(s) + a frozen split.
- **Model:** from a training run on a dataset.
- **Roboflow pulls:** our read-only downloads of current Roboflow labels. They are the base that edits apply to. They are shown as a detail of a label set, not as their own node (open to change).
- **Job:** a batch of images with a type and a state. Batches are split or merged by need: one session, one image type, or one goal.

**Data change**
- Each record gets `inputs: [{id, role}]`. Roles: `base`, `edits`, `predictions`, `train_data`, `init_weights`, `images`.
- This makes a DAG: several parents, still child → parent only.
- Importers fill it from the existing fields.
- New entities: `job`, `prediction_set`.

**Job states:** Ready → In progress → Done, plus Blocked (with a reason) and Cancelled.

| Type | Done when | Output | Progress from |
|---|---|---|---|
| Annotate from scratch | all labeled + reviewed | label set | **both** Roboflow (pull counts) and the app (`edits/`), per a `where` field |
| Revise auto-labels | all reviewed, then applied | `lv_*` | app `edits/` |
| Train | run returned | model | returned run folder |
| Check model | scored | scores | `batches.jsonl` |
| Process raw (later) | images exist | images | session MANIFEST |

**Views**
1. Queue: one column per job type. Replaces the status board.
2. Model page: dataset → label set(s) → per-split counts → images, each with the label set its labels came from.
3. Lineage graph: left → right columns. Shows the **current lineage by default**, with a history toggle.
4. Image lookup.

**Phases**
1. Entities + `inputs` + backfill + queue + model page.
2. Lineage graph + image lookup.

The page stays read-only; tools and the CLI write.

**Known-answer check:** the v5 F5 model page shows `roboflow - 20261028 cryoai v5 S1` and its split counts.
