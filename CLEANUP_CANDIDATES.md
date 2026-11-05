# Cleanup log and remaining candidates

Inventory taken 2026-09-23. Most of it was done the same day on the `legacy-cleanup` branch. The state before the cleanup is tagged **`pre-cleanup`**, so any deleted file can be recovered with `git show pre-cleanup:<path>`. The dependency map before and after is in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Principle: run inference once, analyze many times
`inference/predict_models.py` (or `training/run_experiments.py`) writes `<model_id>/predictions/<image>.txt` (`class conf x1 y1 ...`, normalized). Downstream tools read those files instead of loading a model. That means no GPU or weights are needed, one inference pass serves many analyses, and any registered model can be profiled side by side.

Readers, all in `training/train_yolo.py`:
- `load_predictions_yolo_format()`: masks, confidences and class IDs.
- `load_prediction_polygons()`: float polygons in pixel coordinates, plus confidences.
- `polygon_to_mask()`: rasterizes a float polygon, rounding its vertices.

## Done (2026-09-23)
| Item | What changed |
|---|---|
| Package `__init__` side effects | `training/`, `visualization/` and `data_utils/` now have empty `__init__.py` files. Before this, every `training.*` import loaded the U-Net stack and `segmentation_models_pytorch`. |
| Requirements | U-Net dependencies dropped. `napari` and `PyQt5` moved to `requirements-annotation.txt`. `pyyaml` added. |
| U-Net stack removed | `train_unet.py`, `datasets/`, `transforms/`, `visualization/training_curves.py` |
| Multilayer-class-era scripts removed | `run_multilayer_train_size_ablation.py`, `coco_to_yolo.py`, `run_overlays.py`, `run_gt_overlays*.py`, `monitor_campaign.py` |
| Archived and trivial code removed | `data_utils/_archived/`, `clean_filename.py`. Also `train_yolo.main()`, a demo driven by a CONFIG dict with hard-coded `C:\Users\pujan` paths (about 440 lines). |
| **R1: profiling** | `batch_size_profile.py --predictions-dir` reads prediction files, with no model. `inference.predict_with_review()` was replaced by `review_and_measure()` (review plus morphology only). The unused `create_overlay_image()` and the hard-coded `__main__` block were removed. Checked against the old live path on 5 dense test images: 185/185 objects, mean diameter 55.19 vs 55.20, circularity 0.903 vs 0.904. |
| **R2: annotation tool** | `hitl_annotator.py --predictions-dir` seeds polygons from the prediction files. The model is now only a fallback for images that have no prediction file, and it's optional. Default model is `models/v4_y8m_20260924` (a repo-relative path; v3 until 2026-10-28) in both the annotator and the launcher. The `--imgsz` default is now 1440. `generate_overlays.py --root` no longer defaults to an old path. |
| Morphology plot | Non-finite aspect ratios are skipped. A degenerate sliver (minor axis 0) used to crash the plot at the end of an annotation session. |

## Done later
| Item | What changed |
|---|---|
| **`predict_models.py` / `export_predictions_for_split()` reloaded the model for every image** | Done in 72cedec: `_YOLO_CACHE` keeps one `YOLO` per weights path. Since 2026-11-05 `load_predictions_from_model()` uses the same cache and takes `max_det` (300 by default); `calculate_segmentation_metrics()` already passed `max_det=` to it and raised `TypeError`. |
| **R3: `analysis/multilayer_containment_predictions.py`** (partly) | `--predictions-dir` reads cached prediction files (kept at conf >= `--conf`); the constants are now arguments with the old values as defaults. The live path is still the default. On v5 F5 / S1 valid (26 images) the cached and live paths gave identical TP/FP/FN for every image, and the refactored live path matched the old script exactly. Run_14's own images and COCO export are not on the laptop, so its default setup was not compared. Making the cached path the default is open (user decision; tied to the vertex-truncation item below). |
| **Multi-class counts (#29)** | `models.yaml` `classes:` (v5: `{0: EV, 1: non-EV}`), read by `inference/model_classes.py`. `size_profile_from_predictions` and `add_layer_roles` keep the EV class only for a multi-class model, before measuring and before role assignment. Single-class outputs are byte-identical (checked on two v4 sessions). |

## Remaining candidates (not done)
| Item | Notes |
|---|---|
| **R3: `analysis/multilayer_containment_predictions.py`** | Reads prediction files with `--predictions-dir` (see "Done later"), but live inference is still the default. Open: switch the default once the vertex-truncation item is decided. |
| **`load_predictions_yolo_format()` truncates vertices with `int()`** | This roughens mask edges. Measured on 185 objects, circularity comes out 0.868 instead of 0.903. IoU and area are barely affected. Rounding, as `polygon_to_mask()` does, would fix it. It was left as is because rank, layer-count and role analyses use this function, and changing it would shift previously reported numbers slightly. Decide before the next reported analysis. |
| Launcher GUI | `hitl_launcher.py` has no field for `--predictions-dir`; it's available only from the command line. Add one if the launcher gets used again. |
| One-offs, kept as they are | `analysis/cross_grid_comparison.py` (PLL-film experiment with hard-coded paths; the only remaining user of `inference.extract_instances_yolo()`) and `visualization/export_yolo_poster_figures.py`. A generic version of the cross-grid comparison may come later. |
| Kept utilities | `data_utils/create_subset.py` (data-scaling ablations), `enhance_and_tile_images.py`, `harvest_raw_images.py` |
