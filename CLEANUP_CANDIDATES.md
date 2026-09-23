# Cleanup candidates and planned refactors

Inventory taken 2026-09-23; revised the same day with the user's decisions. Nothing listed here has been changed yet.
"Refs" means how many other files in the repo mention the module; 0 refs is normal for a standalone command-line script, so it doesn't mean dead code on its own.

## Planned refactors: run from the prediction files, not live inference
**Shared pattern.** Run inference once with `inference/predict_models.py` (or `training/run_experiments.py`). It writes `<model_id>/predictions/<image>.txt` (`class conf x1 y1 ...`, normalized). Downstream tools then read those files instead of loading a model. That means no GPU or weights are needed for analysis, one inference pass serves many analyses, and any registered model's output can be profiled side by side.

Functions to reuse:
- `training/train_yolo.py::load_predictions_yolo_format()`: prediction file → masks and confidences
- `analysis/morphology.py::analyze_instances()` and `save_morphology_csv()`: masks/polygons → morphology records and CSV
- `annotation/io.py::read_yolo_polygon_labels()`: already reads this polygon format for the annotation tool

### R1. Size and morphology profiling (`inference/batch_size_profile.py`)
Today it calls `inference.predict_with_review()` for each image, which runs the model through `extract_instances_yolo()`. Change it to "read prediction file → `analyze_instances()`", keeping the confidence threshold and the optional `--review` step. `analysis/aggregate_size_results.py` and `visualization/compare_size_profiles.py` stay as they are, because they read the resulting CSVs.

### R2. Annotation tool (`annotation/hitl_annotator.py`, `hitl_launcher.py`)
Today it calls `training.train_yolo.load_predictions_from_model()` for each image (`hitl_annotator.py:1687`), so every session needs weights and ideally a GPU. Add a `--predictions-dir` option that seeds the initial polygons from the prediction files, with live inference as the fallback when it isn't given. This also removes the need for the tool's image-rescaling step (`_rescale_image_for_inference`), because the prediction files are already in original-image coordinates. While there, fix the hard-coded default paths (`hitl_annotator.py:36`, `hitl_launcher.py:19`, `generate_overlays.py:155`) to point to `models/v3_run1_20260824/best.pt` and paths computed relative to the repo.

## Active: keep
Current all-layer single-class pipeline and its support code:
`data_utils/prepare_all_layer_singleclass.py`, `training/run_experiments.py`, `training/train_yolo.py` (functions only, see section 6),
`training/monitor_run.py`, `training/notify.py`, `training/conf_threshold_sweep.py`,
`analysis/multilayer_containment.py`, `analysis/multilayer_containment_predictions.py`, `analysis/multilayer_role_analysis.py`, `analysis/layer_count_stats.py`,
`analysis/rank_predictions.py`, `analysis/confidence_threshold_sweep.py`, `analysis/morphology.py`,
`inference/predict_models.py`, `inference/perf_log.py`, `models/`, `embedding/` (moving to its own fork later).

## 1. Structural issues: fix first
- **`training/__init__.py` imports `train_unet`.** Every `from training.X import ...` therefore also loads the U-Net stack: `train_unet`, `datasets/`, `transforms/` and `segmentation_models_pytorch`. For example, `inference/predict_models.py` needs `segmentation_models_pytorch` installed even though it never uses it. Fix: empty `training/__init__.py`, or remove the U-Net code entirely (section 2).
- **`requirements.txt`** mixes three groups:
  - Core inference/analysis dependencies. Keep these.
  - U-Net dependencies (`segmentation-models-pytorch`, `albumentations`). Drop these along with section 2.
  - Annotation GUI dependencies (`napari`, `PyQt5`). Keep, since the tool is kept, but move them to an optional `requirements-annotation.txt` so machines that only run inference don't need them.

## 2. U-Net era (superseded by YOLO): remove together
| File | Notes |
|---|---|
| `training/train_unet.py` | U-Net/FPN/DeepLab training; hard-coded Yifei paths in `__main__` |
| `datasets/cryo_instance_dataset.py`, `datasets/__init__.py` | Used only by `train_unet` |
| `transforms/cryo_transforms.py`, `transforms/__init__.py` | Used only by `datasets/` and `train_unet` |
| `visualization/training_curves.py`, `visualization/__init__.py` | U-Net training-curve plots |

## 3. Multilayer-class era (superseded by the all-layer method)
| File | Notes |
|---|---|
| `training/run_multilayer_train_size_ablation.py` | 2-class ablation; hard-coded old paths |
| `data_utils/coco_to_yolo.py` | Old converter that re-splits the data; hard-coded `C:\Users\pujan` paths. Replaced by `prepare_all_layer_singleclass.py` |
| `data_utils/create_subset.py` | Data-scaling experiment helper; could be reused for a future ablation |
| `training/run_overlays.py` | Hard-wired to `roboflow20260604_stratified` |
| `training/run_gt_overlays.py` | One-off GT overlay script, old dataset |
| `training/run_gt_overlays_20260713_singleclass.py` | One-off GT overlays for the 0713 campaign; generalize it with `--dataset` or delete it |
| `training/monitor_campaign.py` | Older campaign monitor; `monitor_run.py` does this job now |

## 4. Yifei-era inference and size profiling: keep and refactor (R1)
| File | Decision |
|---|---|
| `inference/batch_size_profile.py` | **Keep, refactor (R1).** Size and morphology profiling is still wanted |
| `analysis/aggregate_size_results.py`, `visualization/compare_size_profiles.py` | **Keep as is.** Downstream of R1; they read CSVs |
| `inference/inference.py` | **Partly keep.** The review and decision functions (`interactive_review_objects`, `_show_review_popup`, `save_review_decisions`) are kept for R1's optional review. Removable after R1: the `__main__` block with Yifei's paths, `extract_instances_yolo()` (a duplicate of `train_yolo.load_predictions_with_classes_from_model()`; `cross_grid_comparison.py` also uses it, so switch that import first) and the unused `create_overlay_image()`. `predict_with_review()` is either rewritten around the prediction files or removed. |
| `analysis/cross_grid_comparison.py` | **Keep as a one-off, no action.** PLL-film experiment with hard-coded paths. Possibly a generic version later |
| `visualization/export_yolo_poster_figures.py` | **Keep as a one-off, no action.** Takes command-line arguments, so it still works |

## 5. Human-in-the-loop annotation tool: keep, not in active use, refactor (R2)
`annotation/hitl_annotator.py`, `hitl_launcher.py`, `generate_overlays.py`, `io.py`. `generate_overlays.py` and `io.py` already work from label files. The model dependency is only in `hitl_annotator.py`. See R2.

## 6. Already archived or trivial
- `data_utils/_archived/` (`clean_label.py`, `split_dataset.py`): Yifei paths; can be deleted.
- `data_utils/clean_filename.py`: strips the `.rf.<hash>` part of Roboflow filenames; hard-coded Yifei path.
- `data_utils/enhance_and_tile_images.py`, `harvest_raw_images.py`: general image-prep utilities; they take command-line arguments. Keep them if raw data prep is still needed.
- `training/train_yolo.py` lines ~1360–1410: the `__main__` demo block with `C:\Users\pujan` paths. Delete the block and keep the functions.
