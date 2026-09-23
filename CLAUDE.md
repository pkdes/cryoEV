# cryoEV: agent orientation

Automated detection and characterization of extracellular vesicles and particles (EVs/EPs) in cryo-EM micrographs, for the Carney lab. **Active R&D.** Several approaches are being explored in parallel: YOLO instance segmentation, geometric multilayer analysis, size and morphology profiling, human-in-the-loop annotation, and self-supervised embedding clustering. Don't assume any one of them is "the" method.

**Read `README.md` first.** It covers the approaches, the quick start, the repo map and the gotchas.

## Before changing anything
- **Code only.** Data, training runs and the experiment log live in a sibling `../CryoAI/` directory, which is not in git. If it exists, read `../CryoAI/CLAUDE.md` and `../CryoAI/MANIFEST.md` for full history. If it doesn't exist, only `inference/predict_models.py` works standalone.
- **Exported models** are in `models/`. Each model's inference `imgsz` lives in `models/models.yaml`; always use it.
- **`docs/ARCHITECTURE.md`** has the dependency diagram. **`CLEANUP_CANDIDATES.md`** records what was cleaned up (pre-cleanup state is git tag `pre-cleanup`) and the remaining candidates. Check it before building on an older script.

## Conventions
- **Parameterize existing scripts; don't duplicate them.** Add `--dataset`/`--model` style arguments instead of dataset-specific copies. No new hard-coded absolute paths; compute them relative to the repo (`Path(__file__)...`).
- **Run inference once, analyze many times.** Downstream analysis should read prediction files (`<model_id>/predictions/*.txt`, via `train_yolo.load_predictions_yolo_format()`) rather than loading a model.
- **Run long jobs in the background** with output logged to a file the user can follow, instead of polling repeatedly.
- **Don't reorder `classify_roles()`** in `analysis/multilayer_containment.py`: the containment checks must stay in their current order.
- **Don't re-test settled training choices:** `overlap_mask=False`, `imgsz=1440` + `rect=True`, augmentation on.
- **Git remote naming:** on the lab desktop the remote is named `CarneyMLdesktop`; a fresh clone calls it `origin`. Check with `git remote -v` before pushing.
