# cryoEV

Automated analysis of **extracellular vesicles and particles (EVs/EPs) in cryo-EM micrographs**. Carney lab, Wang–Carney collaboration.

## Goal and motivation

Characterizing EVs and EPs from cryo-EM today means counting, outlining and measuring objects by hand, one micrograph at a time. That is slow, hard to scale across samples and conditions, and subjective at the margins: faint vesicles, overlapping objects, multilayer membranes. This project aims to automate it: **find every EV/EP in a micrograph, then measure and classify it** (size, shape, membrane structure and more) reproducibly across large image sets.

**Status: active R&D.** Several approaches are being explored in parallel, and some are more mature than others. This README will narrow as methods settle.

### Approaches in use

| Approach | What it gives | Where |
|---|---|---|
| **Instance segmentation (YOLOv8-seg)** | Per-object polygon masks with confidence scores. The current models are single-class detectors trained with *every membrane layer annotated as its own polygon* | `training/`, `models/`, `inference/predict_models.py` |
| **Geometric post-hoc analysis** | Multilayer structure is recovered *after* detection: a polygon that contains other polygons is multilayer, and the number it contains is its layer count. Matches the annotators' own multilayer labels at F1 ≈ 0.996 on ground truth | `analysis/multilayer_*.py`, `analysis/layer_count_stats.py` |
| **Size and morphology profiling** | Ellipse fits, equivalent diameter, aspect ratio, circularity, solidity, and size distributions compared across samples | `analysis/morphology.py`, `inference/batch_size_profile.py` |
| **Human-in-the-loop annotation** | Model predictions used as first-pass labels, then corrected in a GUI, to grow the training set | `annotation/` |
| **Self-supervised embeddings + clustering** | Frozen DINOv2/v3 embeddings of object crops, clustered without labels, to look for structure nobody annotated. Inference-only; a separate workflow | `embedding/` (own README) |

**Tried earlier (code removed; recoverable from git tag `pre-cleanup`):**
- U-Net semantic segmentation.
- A dedicated "multilayer EV" model class. It was abandoned because recall plateaued around 0.39 for lack of training examples, and the geometric approach above replaced it.

## Repo vs. data

This repo holds **code and a few exported model weights only**. Datasets, training runs, cached predictions and the experiment log live in a sibling `CryoAI/` directory, which is not in git:

```
<parent>/
├── CryoEV Github/   <- this repo
└── CryoAI/          <- data, training outputs, EXPERIMENT_LOG.csv, CLAUDE.md, MANIFEST.md
```

The pipeline scripts find the data directory at `Path(__file__).parent.parent.parent / "CryoAI"`. On a machine without `CryoAI/`, **only the inference path (`inference/predict_models.py`) works standalone**. Training and most `analysis/` scripts expect a `--dataset` folder under `CryoAI/training outputs/`. If `CryoAI/` is present, its `CLAUDE.md` and `MANIFEST.md` hold the full project history and directory map.

## Quick start: run models on new images

```bash
pip install -r requirements.txt            # add: -r requirements-annotation.txt for the annotation GUI
python inference/predict_models.py --images <image_dir> --out <out_dir>                     # all registered models
python inference/predict_models.py --images <image_dir> --out <out_dir> --models v3_run1_20260824 --device cpu
```
Options: `--models all|id1,id2`, `--conf 0.25`, `--iou 0.7`, `--device cuda|cpu`.

Outputs:
| Path | Contents |
|---|---|
| `<out>/<model_id>/predictions/<image>.txt` | One detection per line, in the prediction file format below |
| `<out>/<model_id>/overlays/<image>.png` | Predicted outlines drawn on the image, with the count |
| `<out>/summary.csv` | image × model → `n_detections`, `mean_conf` |

### Registered models ([`models/`](models/README.md), [`models.yaml`](models/models.yaml))
| model_id | Detects | imgsz | Notes |
|---|---|---|---|
| `v3_run1_20260824` | every membrane layer | 1440 | **Current best.** 51 training images; test P=0.81, R=0.86 |
| `allLayer_run14_20260713` | every membrane layer | 1440 | Same recipe, 26 training images; a baseline for how much the extra data helped |
| `yifei_202512` | whole EVs only | 1024 | Legacy detector, >100 training images, different annotation convention |

To add a model, put `best.pt` in `models/<model_id>/` and add an entry to `models.yaml`. No code changes are needed.

### Prediction file format
`class_id confidence x1 y1 x2 y2 ...`, with coordinates normalized to [0, 1]. It's the YOLO polygon label format plus a confidence column. This is the **handoff format between inference and analysis**: run inference once, then point analysis scripts at the prediction files instead of rerunning the model. Read one with `training/train_yolo.py::load_predictions_yolo_format()`.

## Gotchas for anyone picking this up

- **Each model has its own `imgsz`**, stored in `models.yaml`. Running a model at the wrong size silently degrades its results.
- **Inputs should be 8-bit images** (JPEG/PNG, 1440×1024 native), like the Roboflow exports the models were trained on. Convert raw 16-bit or MRC data first.
- **The Yifei model's output can't be used for layer counting.** It outlines whole EVs only, so containment analysis doesn't apply to it. Its raw detection counts are nonetheless similar to v3's.
- **Validation metrics in `models.yaml` can't be compared across models.** Each model was scored on a different validation set.
- **`classify_roles()`** in `analysis/multilayer_containment.py` is deliberately ordered: "contained by something" is checked before "contains something". Don't reorder it.
- **Settled training choices; don't re-test these:**
  - `overlap_mask=False`: mask mAP 0.614 vs 0.511.
  - `imgsz=1440` with `rect=True`: matches the native resolution.
  - Augmentation on: turning it off causes severe overfitting.
  - Model size: y8s/y8m tied y8n on v3 (51 train images), but y8m clearly wins on v4 (100 train images).
- **Known v3 error patterns:**
  - Over-detection on holey-carbon grid texture.
  - Occasional false detections on blank ice.
  - Missed small or faint vesicles.

## Repo map

**Active pipeline** (run in this order; each takes `--dataset` / `--source-name` / `--model-run-dir`. Several default to the older 20260713 campaign, so pass `--dataset` explicitly):
| # | Script | Does |
|:-:|---|---|
| 1 | `data_utils/prepare_all_layer_singleclass.py` | Roboflow COCO export → YOLO labels, all layer categories collapsed to class "EV". Raises an error on unknown categories |
| 2 | `training/run_experiments.py` | Training campaign; `--single-run` uses the known-best config. Writes metrics, overlays and the prediction files |
| 3 | `analysis/multilayer_containment.py` | Validates the containment geometry on ground truth. Home of `classify_roles()` |
| 4 | `analysis/multilayer_role_analysis.py` | Multilayer / inner-layer / standalone roles, predicted vs. ground truth |
| 5 | `analysis/layer_count_stats.py` | Layer-count distributions, predicted vs. ground truth |

**Support:**
| Path | Does |
|---|---|
| `inference/predict_models.py` | Multi-model inference on any image folder (see Quick start) |
| `training/train_yolo.py` | Core training, evaluation, Hungarian matching, and prediction file read/write functions |
| `analysis/rank_predictions.py` | Ranks images by errors (FP + FN), to decide which overlays to look at |
| `analysis/confidence_threshold_sweep.py` | Precision/recall/F1 vs. confidence threshold, computed from the prediction files (no GPU) |
| `analysis/morphology.py` | Ellipse fitting and shape descriptors |
| `training/monitor_run.py`, `notify.py` | Live training monitor, completion notifications |
| `embedding/` | DINO embedding + clustering workflow; see `embedding/README.md`. Planned to move to its own fork |

**How it fits together:** [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) has a dependency diagram. **Cleanup history and remaining candidates:** [`CLEANUP_CANDIDATES.md`](CLEANUP_CANDIDATES.md).

## Training (requires `CryoAI/`)

```bash
python data_utils/prepare_all_layer_singleclass.py \
    --source-name "roboflow - 20260824 cryoai v3" --output-name roboflow_20260824_cryoai_v3_singleclass
python training/run_experiments.py --dataset roboflow_20260824_cryoai_v3_singleclass --single-run
python analysis/multilayer_role_analysis.py --dataset roboflow_20260824_cryoai_v3_singleclass \
    --source-name "roboflow - 20260824 cryoai v3" --model-run-dir "<run folder name>"
```
Each run appends a row to `CryoAI/EXPERIMENT_LOG.csv`. Long runs should go in the background with output logged to a file.

---

## Reference: size and morphology profiling

`inference/batch_size_profile.py` measures every accepted object in a folder of images. It reads prediction files and doesn't load a model, so run inference first:

```bash
python inference/predict_models.py --images <images> --out <pred_out> --models v3_run1_20260824
python inference/batch_size_profile.py --input-dir <images> --output-dir <profile_out> \
    --predictions-dir <pred_out>/v3_run1_20260824/predictions [--pixel-size 3.5] [--review]
python -m analysis.aggregate_size_results --morphology-csv <profile_out>/morphology_all.csv \
    --per-image-csv <profile_out>/per_image_summary.csv --output-dir <agg_out>      # per-sample plots
```
Any folder of `<stem>.txt` prediction files works, for example a training run's `predictions/test`. To compare models, profile each model's prediction folder.

Morphology metrics per object:
| Metric | Description |
|---|---|
| `equivalent_diameter` | Diameter of a circle with the same area |
| `major_axis` / `minor_axis` | Fitted ellipse axes |
| `aspect_ratio` | major / minor (1.0 = circular) |
| `circularity` | 4π·area / perimeter² (1.0 = perfect circle) |
| `solidity` | area / convex hull area |

`--pixel-size` (nm/px) reports physical units instead of pixels.

**Confidence review:** detections at or above the acceptance threshold (default 0.5) are accepted automatically. Below it they're rejected automatically, unless `--review` is passed. With `--review`, a matplotlib pop-up shows each object (circled in red) with **Accept / Reject / Exit** buttons.

## Label review in the browser (active)

Reviews ground-truth labels against cached model predictions, to fix annotation errors before retraining. It is a single HTML page: a desktop polygon Editor, and a swipe mode for phones to accept or reject false positives and false negatives, assign a class and reshape outlines. It works offline from a folder or zip (export/import JSON), or from a server that several reviewers can use at once.

```bash
python annotation/build_review_package.py --source-name "roboflow - 20260924 cryoai v4" \
    --dataset roboflow_20260924_cryoai_v4_singleclass --model-run-dir <run> --output-name cryoai_v4_review
python annotation/review_server.py --package "../CryoAI/annotation review/cryoai_v4_review"   # prints ?k= link
```
Put the server behind a tunnel to share it; it only binds to 127.0.0.1. Each image's saved decisions go to `<package>/edits/<file>.json`, merged per decision (newest wins). The page template is `annotation/review_tool/index.html`.

## Reference: human-in-the-loop annotation (not in active use)

Uses model predictions as first-pass polygons, then lets you correct them in a GUI. Requires `pip install -r requirements-annotation.txt`.

```bash
# Recommended: seed from prediction files (no GPU; model only used for images without a prediction file)
python -m annotation.hitl_annotator --input-images <images> --output-dir <out> \
    --predictions-dir <pred_out>/v3_run1_20260824/predictions --save-auto-labels
# Live inference (default model: models/v3_run1_20260824, imgsz 1440)
python -m annotation.hitl_annotator --input-images <images> --output-dir <out> --device cuda --save-auto-labels
python -m annotation.hitl_annotator --gui --device cuda --save-auto-labels     # folder/file pickers
python -m annotation.hitl_annotator --ui napari --gui --device cuda            # older Napari editor
python -m annotation.hitl_launcher                                             # small GUI launcher (live inference)
```
The default editor is a side-by-side reviewer with synchronized zoom and pan. It has buttons for **Add / Replace / Delete / Save & Next / Skip / Quit**, and keyboard shortcuts `S` (save), `K` (skip) and `Q` (quit).

Outputs:
- `reviewed/images/` and `reviewed/labels/`: YOLO polygon labels for training.
- `auto/labels/`: raw model predictions, written when `--save-auto-labels` is passed.
- `stats/`: session log, model-vs-review precision/recall/F1 (Hungarian IoU matching), and morphology.
