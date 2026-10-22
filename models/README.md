# Exported EV models

Trained YOLOv8-seg models for inference on new cryo-EM micrographs. Details and provenance are in [`models.yaml`](models.yaml).

| model_id | Detects | Size | imgsz |
|---|---|---|---|
| `v4_y8m_20260924` | every membrane layer (**current best**) | medium, 27M params | 1440 |
| `v4_y8n_20260924` | every membrane layer (light option) | nano, 3.3M params | 1440 |
| `v3_run1_20260824` | every membrane layer (superseded) | nano | 1440 |
| `allLayer_run14_20260713` | every membrane layer (26-image baseline) | nano | 1440 |
| `yifei_202512` | whole EVs only (legacy) | nano | 1024 |

## Accuracy vs. compute (v4 models)
Accuracy is on the same 50 held-out v4 test images. Multilayer and inner-layer scores use containment 0.99 / size ratio 0.75. Runtime is measured per image with `inference/benchmark_models.py`: read, predict, masks to numpy, no overlays. It was run on the lab PC (RTX A6000; AMD Ryzen, 32 threads) on v4 test images, a mix of 1440×1024 and 5760×4092.

| | `v4_y8n_20260924` | `v4_y8m_20260924` |
|---|---|---|
| Parameters / weights file | 3.3M / 7 MB | 27.2M / 55 MB |
| Compute per 1440 px pass | 58 GFLOPs | 530 GFLOPs |
| Test F1 at conf 0.25 (best cutoff) | 0.826 (0.843 at 0.45) | 0.853 (0.866 at 0.50) |
| Multilayer / inner-layer F1 | 0.734 / 0.732 | 0.839 / 0.820 |
| GPU time per image (median / mean) | 0.07 / 0.10 s | 0.07 / 0.10 s |
| Peak GPU memory | 1.5 GB | 1.9 GB |
| CPU time per image (32 threads, 10 images) | 0.16 s | 0.55 s |
| 100 images, GPU / this CPU | ~10 s / ~16 s | ~10 s / ~55 s |

**Which model to use:**
- **Default to `v4_y8m_20260924`.** On a GPU it is as fast as nano, because image loading and full-resolution mask extraction dominate the time, not the network. It fits on any GPU with 4 GB or more.
- **Use `v4_y8n_20260924` on a CPU-only machine with large batches.** It is about 3.5x faster on CPU, at the cost of about 0.03 detection F1 and about 0.1 multilayer F1. The CPU timings above are from a 32-thread desktop; a laptop CPU will be several times slower (not measured).

## Usage
```bash
pip install -r requirements.txt
python inference/predict_models.py --images <image_dir> --out <out_dir> --models v4_y8m_20260924
python inference/predict_models.py --images <image_dir> --out <out_dir> --models v4_y8n_20260924 --device cpu
python inference/predict_models.py --images <image_dir> --out <out_dir>              # every registered model
python inference/benchmark_models.py --images <image_dir> --out <bench_dir>        # time the models on your hardware
```
Outputs go to `<out_dir>/<model_id>/predictions/*.txt` (the standard cache format, readable by `analysis/` scripts), `<out_dir>/<model_id>/overlays/*.png`, and `<out_dir>/summary.csv`.

The prediction folder is the input for the downstream tools, none of which needs the model again:
- Size and morphology profile: `python inference/batch_size_profile.py --input-dir <image_dir> --predictions-dir <out_dir>/<model_id>/predictions --output-dir <profile_out>`
- Annotation seeded from predictions: `python -m annotation.hitl_annotator --input-images <image_dir> --output-dir <ann_out> --predictions-dir <out_dir>/<model_id>/predictions`

See the main [README](../README.md) and [`docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md).

## Caveats when comparing
- **Validation mAP numbers are not comparable across models.** Each model was scored on a different val set.
- **Yifei's model outlines whole EVs only.** It does not segment inner membrane layers, so multilayer and layer-count analysis (`classify_roles()`) only applies to the all-layer models. Raw detection counts can still be similar across models: on the 99 v3 test images the means were 25.1 per image (Yifei) and 24.5 (v3). The models disagree about *which* objects they find. Its masks are also blockier in crowded clusters.
- **Inputs should look like the training data:** 8-bit JPEG/PNG exports of the micrographs (1440×1024 native). Raw 16-bit or MRC data needs to be converted first.
- Yifei's training images may overlap with the v3 dataset. Keep this in mind when evaluating on v3 test images.

To add a model, drop its `best.pt` in `models/<model_id>/` and add an entry to `models.yaml`.
