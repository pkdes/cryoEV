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
- **Use the tools' own commands; don't write records by hand.** Commands are in the README (registry row) and `docs/JOBS.md`:
  - a new job: `registry/registry.py job new ...`; status: `registry.py set-status ...`;
  - training: `training/make_job_bundle.py`, then `jobs/sync.py local` (laptop) or `push` / `pull` (desktop);
  - after changing labels, packages or runs by hand: `registry.py import`, then `registry/history_page.py`.
- **The importers are the safety net.** Work done outside the tools (a run started by hand, a new review package) shows up on the next `registry.py import`; never edit `*.jsonl` directly.
- **Never automated:** the Roboflow push (the user's OK each time), promoting a model to production (user only), anything under Box.
- **The laptop is home base** (registry, review servers, job queue). The A6000 desktop is sometimes on; nothing may depend on it.

