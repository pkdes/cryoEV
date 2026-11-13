# cryoEV: agent orientation

Automated detection and characterization of extracellular vesicles and particles (EVs/EPs) in cryo-EM micrographs, for the Carney lab. **Active R&D.** Several approaches are being explored in parallel: YOLO instance segmentation, geometric multilayer analysis, size and morphology profiling, human-in-the-loop annotation, and self-supervised embedding clustering. Don't assume any one of them is "the" method.

**Read `README.md` first.** It covers the approaches, the quick start, the repo map and the gotchas.

## Before changing anything
- **Code only.** Data, training runs and the experiment log live in a sibling `../CryoAI/` directory, which is not in git. If it exists, read `../CryoAI/CLAUDE.md` and `../CryoAI/MANIFEST.md` for full history. If it doesn't exist, only `inference/predict_models.py` works standalone.
- **Exported models** are in `models/`. Each model's inference `imgsz` lives in `models/models.yaml`; always use it.
- **`docs/ARCHITECTURE.md`** has the dependency diagram. **`CLEANUP_CANDIDATES.md`** records what was cleaned up (pre-cleanup state is git tag `pre-cleanup`) and the remaining candidates. Check it before building on an older script.

## Conventions
- **Parameterize existing scripts; don't duplicate them.** Add `--dataset`/`--model` style arguments instead of dataset-specific copies. No new hard-coded absolute paths: data roots come from `data_utils/config.py` (env `CRYOEV_LAPTOP_ROOT`, `CRYOEV_CRYOAI_ROOT`, `CRYOEV_REGISTRY`, `CRYOEV_PROCESSING_ROOT`; default = the sibling folders above the repo).
- **Run inference once, analyze many times.** Downstream analysis should read prediction files (`<model_id>/predictions/*.txt`, via `train_yolo.load_predictions_yolo_format()`) rather than loading a model.
- **Run long jobs in the background** with output logged to a file the user can follow, instead of polling repeatedly.
- **Don't reorder `classify_roles()`** in `analysis/multilayer_containment.py`: the containment checks must stay in their current order.
- **Multi-class datasets** (`--class-scheme ev_nonev*`): class 0 is still EV. Containment and role analysis assume EV-only polygons: pass `multilayer_role_analysis.py --ev-classes 0` (`0,2` with Undetermined).
- **Don't re-test settled training choices:** `overlap_mask=False`, `imgsz=1440`, augmentation on. `rect=True` is open again: it silently disables mosaic/mixup, and `--rect 0` scored best on v5 (2026-10-29).
- **On this Windows PC, commit memory (page file) limits training, not VRAM.** Don't run evaluation beside a training run. See the README gotchas.
- **Git remote naming:** on the lab desktop the remote is named `CarneyMLdesktop`; a fresh clone calls it `origin`. Check with `git remote -v` before pushing.

## Agents (Claude sessions and other automations)
**Read [`docs/AGENTS.md`](docs/AGENTS.md) first:** the `python -m cryoev` CLI (status, find / show / lineage, pull, review progress, feedback, register, predict new), identity (`CRYOEV_FOR` / `CRYOEV_VIA`; after go-live, credited as "claude for pujan"), and recipes for putting data in, analysing, pulling down, the review loop and the training loop. Hard rules:
- **Keep `CRYOEV_VIA` UNSET until the new registry code (with the `FORMAT.json` `min_reader` guard) is deployed on every machine that runs registry code.** Until then identify the person only (`--who` / `CRYOEV_FOR`).
- **Use the tools' own commands; never write records by hand or edit `*.jsonl`.** The registry is append-only; the importers are the safety net (`python -m cryoev register`).
- **Dry run first.** `cryoev` writes nothing without `--do`; every `--do` is logged to `<CryoAI-laptop>/logs/agent_actions.jsonl` (Admin, Recent actions: "laptop CLI (unverified)").
- **Never automated:** Roboflow uploads / relabels (the user's OK each time), setting production / promoting a model (user only), `git push`, sending work to the desktop, anything under Box.
- **Blind audits stay blind:** never open a `_work_*` review folder or read review decisions while a package is not applied; `cryoev review progress` gives counts only.
- **Pushing:** `.githooks/pre-push` (enabled with `git config core.hooksPath .githooks`) scans outgoing commits for keys and private names; never `--no-verify` without the user. The repo is public: codenames only.
- **Ship a review:** `annotation/apply_review_edits.py` applies **only images marked reviewed** by default; `--all-edits` only when the user asks. Count EVs with the `classes:` map (class 0 only for multi-class models), never raw `n_detections`.
- **Small score gaps are noise:** v5 F5-config repeats spread EV mask mAP50-95 ±0.008 (models.yaml `test_seed_spread`).
- **Score segmentation as trained:** Ultralytics `val` needs `overlap_mask=False` for our models (models.yaml). The default carves nested membrane layers into rings and under-reports EV mask mAP (v5 F5: 0.476 vs 0.708).
- **The laptop is home base** (registry, review servers, job queue). The A6000 desktop is sometimes on; nothing may depend on it.
- **After code changes:** `python tools/test_gate.py --affected`.
