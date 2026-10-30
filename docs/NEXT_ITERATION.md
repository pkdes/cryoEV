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
| User OK | Copy `CryoAI-laptop/` outputs to Box (after a Box Drive restart) |
| User | Confirm the canonical legacy model ids `v0_alllayer_y8n_20260713` and `v0_yifei_y8n_20251201`. The version number and Yifei's day were guessed; the old ids are aliases |
| Build | Job queue + model page (spec below, phase 1), then the lineage graph + image lookup (phase 2) |
| Build | Review tool UI: U4 (big Accept/Reject on prediction-only objects; editing vertices accepts) + the user's tweaks |
| Build | The 6 "suggested changes to files owned by others" in [TRACKING_CONTRACT.md](TRACKING_CONTRACT.md): `inference_check/1` lines, `supersedes` in version.json, data.js `id`/`base`/`model`/`parent`, a `job_bundle/1` line + `bundle_id`, `run.json`, `models.yaml` keys |
| Build | Move the hardcoded `C:\Users\pujan\…` defaults to config/env (the repo is a public fork) |
| Data gaps | JOB.json date format not parsed; grid `replaced_by` for 19 renumbered grids; 28 pre-manifest sessions not in the master CSV |
| Later | AU remote access (Cloudflare Tunnel + Access), SAM, process-raw jobs, the paused 20261027 sampling report |

Decided:
- datasets may be built straight from a local `lv_*` (no Roboflow push first);
- only the user promotes a model (v5 F5 = production).

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
