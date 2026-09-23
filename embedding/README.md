# `embedding/` — frozen self-supervised clustering of EV objects

A **separate, inference-only workflow** from the YOLO training pipeline. Nothing here
trains, fine-tunes, or reads a label as input.

**The question:** do frozen self-supervised embeddings cluster cryo-EM EV/EP objects into
meaningful groups *without* using labels? Everything the YOLO pipeline groups today is a
group a human annotator already drew. This tests whether the data has structure we haven't
labelled.

Labels are **held out entirely** — recorded in the manifest, read only by `evaluate.py`.

## Decision criteria

| Outcome | Reading | Next |
|---|---|---|
| Clusters track held-out labels (ARI well above the permutation baseline) | It works | Feature extraction + figure design |
| Clusters track acquisition session instead | Batch effect, **not** failure | Per-session standardization, then re-cluster |
| Clusters track neither | Frozen embeddings insufficient | Continued pretraining, or the MAE route |

## Running it

```bash
PY="../CryoAI/.venv-new/Scripts/python.exe"

$PY embedding/crops.py --force        # ~3 min   -> GATE B
$PY embedding/normalize.py --force    # ~1 min   -> GATE C
$PY embedding/embed.py --force        # ~2 min   -> GATE D
$PY embedding/cluster.py              # ~2 min
$PY embedding/evaluate.py             #  ~2 min  -> GATE E
```

Watch progress and GPU in a second terminal:

```bash
$PY embedding/monitor.py --watch
```

## Variants

Every script takes `--variant NAME`, which suffixes all of its outputs so an alternative run
sits **beside** the baseline instead of overwriting it. `evaluate.py --compare-to` then
tabulates the two side by side.

The `masked` variant blanks everything outside each object's polygon:

```bash
$PY embedding/crops.py     --variant masked --mask-objects   # ~4 min
$PY embedding/normalize.py --variant masked
$PY embedding/embed.py     --variant masked
$PY embedding/cluster.py   --variant masked
$PY embedding/evaluate.py  --variant masked --compare-to baseline
```

**Why it exists.** The baseline run found that 3 of 4 clusters sorted on what the vesicle
*sits on* — carbon-film edge, clean ice, cluttered field — rather than on the vesicle itself.
DINO was trained on photographs, where the background is usually the most informative thing
in frame; it has no way to know that here the background is the part to ignore. Masking
removes that information outright, so the model can only respond to the object.

Fill is the **median of the object's own interior**, not black: a zero fill would introduce a
hard high-contrast silhouette edge, which is simply a different strong artifact to cluster on.
The mask is dilated a few pixels first (`--mask-dilate`, default 3) because the annotator drew
the polygon *on* the membrane — cutting exactly at the outline would clip the structure we
most want the model to see.

## Steps

| # | Script | Reads | Writes |
|:-:|---|---|---|
| 1 | `crops.py` | Roboflow COCO GT | `crops/*.png` (uint16), `manifest.parquet` |
| 2 | `normalize.py` | crops | `crops_norm.npy` `[N,224,224]` fp16, `normalize_meta.json` |
| 3 | `embed.py` | `crops_norm.npy` | `embeddings.npy` `[N,1024]`, `embed_meta.json` |
| 4 | `cluster.py` | `embeddings.npy` | `umap_nn*.npy`, `clusters.parquet` |
| 5 | `evaluate.py` | manifest + clusters | `evaluation/scores.csv`, plots, `RESULTS_BRIEF.md` |

Outputs land in `../CryoAI/embedding outputs/` — code is git-tracked, data is not, matching
the split described in `CryoAI/CLAUDE.md`.

**`embed.py` is the expensive step.** Steps 4–5 read only its cached `.npy`, so re-clustering
with new hyperparameters never re-embeds. Steps 1–3 refuse to overwrite without `--force`.

## Three design decisions that matter

**Pixel scale is normalized first.** Every `*Jul30*` micrograph is 5760×4092; everything else
is 1440×1024. Without the 4× downsample, DINO separates the two magnifications trivially and
we'd be clustering the microscope, not the biology.

**Crop windows shift inward rather than pad.** A window 2× the object's long side overhangs
the micrograph border for edge objects. Padding invents a high-contrast synthetic texture that
DINO clusters on — and since edge objects are unevenly distributed across grids, that artifact
masquerades as exactly the batch effect we're testing for. Sliding the window back inside
keeps the object fully visible on real pixels; it just ends up off-center, which costs a ViT
essentially nothing. 695 of 696 affected crops are fixed this way.

**Normalization is locked to one reference image, not per-crop.** Per-crop normalization would
rescale every object to the same intensity range, erasing the contrast and density differences
that might carry the signal. `normalize_qc.png` plots per-session histograms: if they all
overlap perfectly, per-crop normalization has leaked in.

## Model

Default `facebook/dinov2-large` (ungated, ViT-L/14, 1024-d CLS).

`facebook/dinov3-vitl16-pretrain-lvd1689m` is **gated**. To use it: accept the license on the
HF model page, create a read token, `setx HF_TOKEN <token>`, then

```bash
$PY embedding/embed.py --model facebook/dinov3-vitl16-pretrain-lvd1689m --force
```

Crops and normalization are model-independent, so switching costs only steps 3–5 (~5 min).

## Environment

Uses `CryoAI/.venv-new` — the same env as training. `transformers`, `umap-learn`, and
`pyarrow` were added; a dry-run confirmed nothing was downgraded (numba 0.67 is compatible
with the existing numpy 2.3.5, and torch/ultralytics are untouched).

These three are **not** in the repo's `requirements.txt`. On a fresh machine, install them on top of it:
`pip install -r requirements.txt transformers umap-learn pyarrow`.

## Reused from the main pipeline

- `classify_roles()`, `find_containment()`, `polygon_to_mask()` — `analysis/multilayer_containment.py`
- `CATEGORY_NAME_TO_CLASS` — `data_utils/prepare_all_layer_singleclass.py` (raises loudly on an unmapped category, same as the converter)
- `get_gpu_info()` — `training/monitor_run.py`
