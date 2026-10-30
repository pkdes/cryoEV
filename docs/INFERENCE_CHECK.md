# Inference checks and GPU job bundles

## Inference check: score a model on a new session
A stratified sample of a new session's tiles is reviewed in `web_qc`, the model is scored against
the review, and the corrected labels are kept as test data for that session. All commands run from
the repo root.

```bash
# 1. stage: sample + batch folder (existing predictions, or omit --predictions-dir to run the model on the sample only)
python inference/inference_check.py stage --tiles "<session>/5. Model + labeling/model_input" \
    --model v5_evnonev_y8m_20261029 --predictions-dir "<session>/5. Model + labeling/predictions/v5_evnonev_y8m_20261029/predictions" \
    --n 24 --seed 1
# 2. review: run the web_qc command that stage prints (it includes --classes .../classes_roboflow.yaml)
# 3. score: export + metrics + RESULT.md, and a "scored" line in batches.jsonl
python inference/inference_check.py score <batch_dir> --reviewer <name>
python inference/inference_check.py list                                  # latest status per batch
python inference/inference_check.py log <batch_dir> --status reviewed --reviewer <name>   # optional manual status
```

| Option | Default | Notes |
|---|---|---|
| `--out` | `<repo>/../CryoAI-laptop/inference_checks/` | Batch root. Refused if it lies inside the session (8-digit) folder |
| `--model` | `v5_evnonev_y8m_20261029` | Must match `--predictions-dir`: a path naming another registered model is refused |
| `--n`, `--seed` | 24, 1 | Same seed + same tiles give the same sample |
| `--manifest` | `<tiles>/MANIFEST.csv`, else `<tiles>_MANIFEST.csv` | `grid`, `montage` and `nm_per_px` columns; parsed from file names if missing |
| `--field-table` | `<tiles>/../sampling/field_table.csv` if present | Defocus from `mdoc_defocus_um`, matched by `output_name` or `tile`. `none` turns it off |
| `--defocus-bins` | 3 | Equal-count bins over the session (dz1 = most negative); `na` = no defocus value |
| `--classes` | `roboflow` | `roboflow`, `default` (`classes.yaml`) or a path |
| `--conf` | the model's registered conf | `conf:` key in `models.yaml`, else "use conf X" in its note, else its best-F1 conf |
| `--device` | `cuda` | Only used when there's no `--predictions-dir` |

**Sampling.** Greedy and seeded. Each pick is the tile that best balances, in order:
1. any grid × defocus stratum not yet sampled;
2. tiles per grid;
3. tiles per defocus bin within the grid;
4. tiles per montage;
5. tiles per montage position (`rXcY`).

Result:
- with n ≥ the number of strata, every stratum is sampled;
- grids get equal shares (a small grid gives all it has);
- a montage gets a second tile only after every montage in its stratum has one.

Only tiles that have a prediction file are eligible; `BATCH.json` counts the rest.

**Batch folder** `<out>/<batch_id>/` (default id `<session>_<model>_<YYYYmmdd-HHMMSS>`):
| Path | Contents |
|---|---|
| `images/` | Hard links to the tiles (copies if linking fails). Don't edit them in place: a hard link is the source file |
| `predictions/` | Copies of the sampled prediction files (`class conf x1 y1 …`) |
| `p/` | `predict_models.py` output, only when the model was run here |
| `sample.csv` | order, image, grid, montage, position, defocus_um, defocus_bin, stratum, n_pred, n_pred_conf<c>, source |
| `BATCH.json` | Every setting: sources, conf and where it came from, strata (sampled and empty), defocus bin ranges, commands |
| `review/` | web_qc review JSONs. After `score`: `export/` (corrected labels: YOLO-seg + COCO with Roboflow names) and `metrics/` |
| `RESULT.md` | One page: P/R/F1 at the registered conf, by threshold, by stratum, per-class counts |

**Classes.** `annotation/web_qc/classes_roboflow.yaml` holds the lab's Roboflow classes (EV, Low conf EV, MV,
inner, Incomplete, Overlapping, VLDL, non-EV):
- `is_ev` follows the training scheme: the first six count as EV, VLDL and non-EV don't.
- Model class 0 gets EV / MV / inner from `classify_roles()` via `role_to_class`.
- The optional key `model_class_to_class: {1: non-EV}` seeds model class 1 as non-EV. Those outlines also stay out of role
  assignment, EV P/R/F1, `precision_by_conf` and `morphology_model`; they are still listed in `confusion.csv`.
- `classes.yaml` has no such key and stays the web_qc default. Existing reviews export byte-identically
  (26 export files md5-checked on a fixture; the timestamped html/xlsx/pdf were not compared).
- With `classes.yaml`, a v5 non-EV prediction is seeded and scored as EV, as before. So use the Roboflow file for multi-class models.

### `batches.jsonl` (append-only; one JSON object per line)
Every line repeats the batch fields, so the **last line per `batch_id`** is the current record. Earlier lines are never rewritten.

| Field | Meaning |
|---|---|
| `batch_id` | Batch folder name under the root |
| `created` | Time the batch was staged (the same on every line) |
| `session` | First 8-digit folder name above `--tiles`, or `--session` |
| `tiles_dir` | Absolute path of the source tiles |
| `model` | `model_id` in `models/models.yaml` |
| `predictions_dir` | Source of the predictions: the `--predictions-dir` given, or `<batch>/p/<model>/predictions` |
| `n`, `seed` | Tiles staged, sampler seed |
| `strata` | `{"grid=2/defocus_bin=dz1": 4, …}`: tiles per stratum |
| `status` | `staged` (by `stage`), `reviewed` (by `log`), `scored` (by `score`) |
| `reviewer` | Name given with `--reviewer`, or null |
| `cryoEV_commit` | Commit that wrote **this line** (`-dirty` if tracked files were modified) |
| `logged` | Time of this line |
| `metrics` | `scored` only: `conf`, `n_done`, `precision`, `recall`, `f1`, `per_class` (reviewer's class → count) |
| `result`, `note` | `scored`: path of RESULT.md; `log --note`: free text |

## GPU job bundle for the shared desktop
```bash
python training/make_job_bundle.py --dataset-dir "<CryoAI>/training outputs/<dataset>" \
    -- --single-run --model-size m --batch 8 --rect 0 --patience 300 --weights models/v4_y8m_20260924/best.pt
```
- **Output.** Writes `<out>/<dataset>__job_<time>.zip` and `…_README.md` (default `<repo>/../CryoAI-laptop/job_bundles/`).
  The README has the exact PowerShell steps for the desktop:
  1. unzip into `CryoAI\training outputs\`;
  2. update `cryoEV` to the bundling commit;
  3. start `run_experiments.py` in the background with a log.

  It also lists what to send back: the run folder, the logs and the new `EXPERIMENT_LOG.csv` row. Nothing on the desktop is automated.
- **Arguments.** Everything after `--` goes to `run_experiments.py`. Flags it doesn't define are refused, and `--dataset` is set by the bundle.
  For `--weights`, a file tracked in the repo is used from the desktop's own checkout; any other file is packed into `weights/`.
- **`dataset.yaml` paths.** The absolute `path:` line (e.g. `C:/Users/ML-2619/...`) is commented out, and
  `train`/`val`/`test` become relative. With no `path:`, Ultralytics uses the yaml's own folder as the root. A relative
  `path:` would resolve against the working directory, so it is removed rather than made relative. `JOB.json` in
  the zip lists the changes, the arguments and an md5 for every image and label.

## Tests
`python -m unittest tests.test_inference_check -v` (no GPU, about 15 s):
- the sampler;
- a real `stage` on 20261027, skipped if the session folder is absent;
- `score` on a synthetic review with a known answer (`tests/qc_fixture.py`);
- the default-classes control;
- a bundle round trip.
