# cryoEV

Instance segmentation of extracellular vesicles in cryo-EM micrographs.

Uses **YOLO instance segmentation** (via `ultralytics`) for direct per-object polygon masks, followed by **ellipse fitting and morphology analysis** to quantify vesicle size and shape.

Includes a **confidence-based interactive review** system that lets you inspect and accept/reject low-confidence detections before saving final results.

## Installation

```bash
pip install -r requirements.txt
```

## Project Structure

```
cryoEV/
├── README.md
├── requirements.txt
├── data_utils/
│   ├── clean_filename.py        # Remove .rf.<hash> artifacts from filenames
│   ├── clean_label.py           # Remap / filter YOLO class IDs
│   └── split_dataset.py         # Random train/val/test split
├── datasets/
│   └── cryo_instance_dataset.py # PyTorch Dataset for YOLO-polygon labels
├── transforms/
│   └── cryo_transforms.py       # Albumentations pipelines (light/standard/heavy)
├── training/
│   ├── train_unet.py            # UNet/FPN/DeepLabV3+ training loop
│   └── train_yolo.py            # YOLO training + evaluation
├── inference/
│   └── inference.py             # YOLO inference, review, and morphology analysis
├── analysis/
│   └── morphology.py            # Ellipse fitting and shape descriptors
└── visualization/
    └── training_curves.py       # Loss/IoU/F1 curve plotting
```

## Data Preparation

### Expected folder layout

```
dataset/
├── images/
│   ├── sample_001.jpg
│   └── ...
└── labels/
    ├── sample_001.txt    # YOLO polygon format
    └── ...
```

Each label file uses **YOLO polygon format**: one line per object, `class_id x1 y1 x2 y2 ...` with normalised coordinates in [0, 1].

### Cleaning and splitting

```bash
# Remove .rf.<hash> from filenames exported by Roboflow
python -m data_utils.clean_filename

# Remap all classes to 0 and remove unwanted classes
python -m data_utils.clean_label

# Split into train / val / test
python -m data_utils.split_dataset
```

Edit the paths and config dicts inside each script before running.

## Training

```bash
python -m training.train_yolo
```

See `training/train_yolo.py` for configuration options including YOLO model size, epochs, and evaluation thresholds.

### Visualise training curves

```bash
python -m visualization.training_curves results/experiment_name
```

## Inference

```bash
python -m inference.inference
```

Edit the `CONFIG` dict in `inference/inference.py` to set your model path, test data directory, and output directory.

### What it produces

For each test image:

| Output file | Description |
|---|---|
| `<name>_mask.png` | Combined binary mask of accepted detections |
| `<name>_decisions.csv` | Per-object accept/reject decisions with bounding boxes |
| `<name>_morphology.csv` | Per-vesicle shape measurements (see below) |
| `<name>_ellipses.png` | Image with fitted ellipses overlaid |
| `<name>_morphology_distributions.png` | Histograms of size and shape descriptors |

Plus combined outputs across all images: `morphology_all.csv` and `morphology_distributions_all.png`.

### Morphology metrics

Each detected vesicle is measured by fitting an ellipse and computing:

| Metric | Description |
|---|---|
| `equivalent_diameter` | Diameter of a circle with the same area |
| `major_axis` / `minor_axis` | Fitted ellipse axes |
| `aspect_ratio` | major / minor (1.0 = circular) |
| `circularity` | 4*pi*area / perimeter^2 (1.0 = perfect circle) |
| `solidity` | area / convex hull area |

Set `pixel_size` in the config (e.g., `3.5` for 3.5 nm/px) to report measurements in physical units instead of pixels.

### Confidence threshold

Objects with confidence >= `confidence_threshold` are automatically accepted. Objects below are either rejected automatically (`skip_review=True`) or presented for manual review (`skip_review=False`).

### Interactive review

Set `skip_review=False` to manually review below-threshold detections. A matplotlib popup will appear showing:

- **Left panel**: original image (clean, no annotations).
- **Right panel**: overlay with all detection masks — green for objects above the threshold, yellow for objects below — and a red circle highlighting the object under review.
- **Buttons**: Accept / Reject / Exit. Exit stops the entire inference process immediately.

### Python API

```python
from inference.inference import predict_with_review

masks, confidences, decisions, morphology, exited = predict_with_review(
    image_path='test_image.png',
    output_dir='results/',
    yolo_model_path='weights/best.pt',
    pixel_size=3.5,          # nm/px, or None for pixel units
    confidence_threshold=0.5,
    skip_review=False,       # False = popup review, True = auto-reject below threshold
)
```

### Key functions

| Function | Module | Description |
|---|---|---|
| `extract_instances_yolo()` | `inference.inference` | YOLO mask extraction with per-object confidence |
| `interactive_review_objects()` | `inference.inference` | Popup-based accept/reject for low-confidence objects |
| `predict_with_review()` | `inference.inference` | End-to-end: predict, review, analyse, save |
| `analyze_instances()` | `analysis.morphology` | Compute morphology descriptors for a list of masks |
| `fit_ellipse()` | `analysis.morphology` | Fit an ellipse to a single instance mask |
| `plot_morphology_distributions()` | `analysis.morphology` | Plot histograms of size/shape metrics |

## Human-In-The-Loop Polygon Annotation

Use model predictions as first-pass labels, then edit polygons manually in a rich UI.

### Launch annotator

```bash
python -m annotation.hitl_annotator \
    --input-images "path/to/new_images" \
    --output-dir "path/to/hitl_output" \
    --model-path "C:/Users/ML-2619/Desktop/Pujan Cryo/cryo-ev pipeline/Model Training by Yifei/round_2/results_yolov8_heavy_augmentation/training/vesicle_instance_seg_v2/weights/best.pt" \
    --device cuda \
    --save-auto-labels
```

Or open folder/file pickers for the paths:

```bash
python -m annotation.hitl_annotator --gui --device cuda --save-auto-labels
```

### Editor controls

By default, the annotator now opens a **simple side-by-side reviewer**:

- **Left panel**: image with polygon overlay
- **Right panel**: raw image only
- zoom and pan stay **synchronized** across both panels
- click a polygon to select it
- use buttons for **Add**, **Replace**, **Delete**, **Save & Next**, **Skip**, and **Quit**

Keyboard shortcuts still work:
- `S` = save corrected labels and continue
- `K` = skip current image
- `Q` = quit the session

If wanted, the older Napari editor can still be launched explicitly with:

```bash
python -m annotation.hitl_annotator --ui napari --gui --device cuda --save-auto-labels
```

### Outputs

- Reviewed dataset for future training:
    - `reviewed/images/*.png|jpg|tif`
    - `reviewed/labels/*.txt` (YOLO polygon format)
- Optional first-pass model labels:
    - `auto/labels/*.txt`
- Session + quality metrics:
    - `stats/annotation_session.csv`
    - `stats/model_vs_review_metrics.csv`
    - `stats/morphology_reviewed_all.csv`

`model_vs_review_metrics.csv` reports object-level precision/recall/F1, matched IoU, and count error using IoU-based Hungarian matching between model predictions and reviewed labels.
