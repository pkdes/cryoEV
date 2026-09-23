# Exported EV models

Three trained YOLOv8n-seg models for inference on new cryo-EM micrographs. Details and provenance are in [`models.yaml`](models.yaml).

| model_id | Detects | imgsz |
|---|---|---|
| `v3_run1_20260824` | every membrane layer (current best) | 1440 |
| `allLayer_run14_20260713` | every membrane layer (26-image baseline) | 1440 |
| `yifei_202512` | whole EVs only (legacy) | 1024 |

## Usage
```bash
pip install -r requirements.txt
python inference/predict_models.py --images <image_dir> --out <out_dir>              # all models
python inference/predict_models.py --images <image_dir> --out <out_dir> --models v3_run1_20260824 --device cpu
```
Outputs go to `<out_dir>/<model_id>/predictions/*.txt` (the standard cache format, readable by `analysis/` scripts), `<out_dir>/<model_id>/overlays/*.png`, and `<out_dir>/summary.csv`.

## Caveats when comparing
- **Validation mAP numbers are not comparable across models.** Each model was scored on a different val set.
- **Yifei's model outlines whole EVs only.** It does not segment inner membrane layers, so multilayer and layer-count analysis (`classify_roles()`) only applies to the two all-layer models. Raw detection counts can still be similar across models: on the 99 v3 test images the means were 25.1 per image (Yifei) and 24.5 (v3). The models disagree about *which* objects they find. Its masks are also blockier in crowded clusters.
- **Inputs should look like the training data:** 8-bit JPEG/PNG exports of the micrographs (1440×1024 native). Raw 16-bit or MRC data needs to be converted first.
- Yifei's training images may overlap with the v3 dataset. Keep this in mind when evaluating on v3 test images.

To add a model, drop its `best.pt` in `models/<model_id>/` and add an entry to `models.yaml`.
