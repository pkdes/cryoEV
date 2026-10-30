# Tracking contract v1: ids, records, links, statuses

One id per thing, and the links from images to label versions, review batches, inference checks, datasets, runs and models.
This extends [LABEL_CHANGES.md](LABEL_CHANGES.md) (contract v1: edit records → `lv_*` → `changes.jsonl`) and doesn't change any of its fields.
Registry style follows `batches.jsonl` ([INFERENCE_CHECK.md](INFERENCE_CHECK.md)):
- append-only JSONL, with one record per line;
- each line repeats the whole record, so the **last line per `id` is current**;
- lines are never rewritten.

- Schemas: `registry/schemas/<entity>.json` (draft 2020-12). Samples: `registry/samples/<entity>.jsonl`.
- Checker: `python registry/validate.py [--jsonl f ...] [--report out.json] [--verbose]`.
  - It is read-only and refuses a `--report` path under `C:\Users\pujan\Box\` or `Cryo-image-processing\`.
  - Tests: `python -m unittest discover -s registry/tests`.
- Every entity record carries `schema: "<entity>/1"`, `id` and `logged` (ISO time of the line). Optional: `recorded_by` (`<tool> <git sha>` or `backfill:<importer> <sha>`), `note`.
  `label_change/1` is the existing `changes.jsonl` row. It is unchanged and has no envelope.

## Ids
| Entity | Id format | Example | Comes from |
|---|---|---|---|
| image | **stem** = `image_key(name)` = `rf_key(stem_of_name(name))`: the extension is joined by `_`, and runs of spaces/hyphens become `-` | `20260126-MDA-MB-KH__2_square143_hole73_0_hm_png`, `0004_Jul30_jpg` | Already in use as the key: pull `extra.name`, `changes.jsonl.image`, tracking CSVs, and export file names before `.rf.<hash>`. Session tiles: MANIFEST `output_name`. Helpers: `registry/ids.py` |
| session | `YYYYMMDD` or `YYYYMMDD-N` (second session that day) | `20261027`, `20260827-2` | Folder in `Cryo-image-processing`. Pre-manifest sessions use the tracker "Session date" and record `date_basis` |
| grid | inventory **Grid#** (≥ 4 digits) | `2602` | Tracker "Cryo Grids" `Grid#` = master `inventory_id`. The scope cassette slot is not an id: it is `session_grid` on image/session |
| roboflow_image | Roboflow id (20 alphanumeric) | `rLVhwdZNhLMotG5BPhpj` | Pull `extra.roboflow_id`, upload MANIFEST `roboflow_id` |
| pull | `roboflow_pull_<YYYYMMDD_HHMM>` | `roboflow_pull_20261029_1811` | Folder name (`CryoAI-laptop/`) |
| roboflow_export | Folder name under `CryoAI/training images/` | `roboflow - 20261028 cryoai v5 S1` | Roboflow version download, or a re-split of one (`derived_from`) |
| label_version | `lv_<YYYYMMDD_HHMM>` | `lv_20261030_0909` | LABEL_CHANGES.md (unchanged) |
| review_package (review batch) | data.js `id`. **Must equal the folder name and be unique.** Parts are `<parent>_part<N>` | `prelipo_v5F5_review_v2_part3` | `build_review_package.py` |
| inference_check | `batch_id` = `<session>_<model>_<YYYYmmdd-HHMMSS>` | `20261027_v5_evnonev_y8m_20261029_20261030-090300` | `inference_check.py stage` |
| dataset | `training outputs/` folder name (= `run_experiments.py --dataset`) | `roboflow_20261028_cryoai_v5_S1_evnonev` | Dataset build step |
| job_bundle | zip stem `<dataset>__job_<YYYYmmdd-HHMMSS>` | `…_evnonev__job_20261030-140000` | `make_job_bundle.py` |
| run | Run folder name `<YYYY-MM-DD_HHMMSS>_y<v><size>_<imgsz>_<batch>b_<epochs>ep_Run_<N>` (probes: `…_probe_…`) | `2026-10-29_054810_y8m_1440_8b_300ep_Run_15` | `run_experiments.py`. The config can be parsed from the name (`ids.parse_run`) |
| model | `models.yaml` key. New ones use `v<N>_<tag>_<arch>_<YYYYMMDD>` | `v5_evnonev_y8m_20261029` | `models/models.yaml` |
| person | `people.json` id | `pujan` | Roster (LABEL_CHANGES.md) |

## Records: required fields (beyond `schema`, `id`, `logged`)
| Record | Required | Notable optional |
|---|---|---|
| `image/1` | `session`, `grid`, `rf_id` (each may be null) | `rf_name`, `output_name`, `session_date_basis`, `session_grid`, `montage`, `position`, `nm_per_px`, `md5`, `duplicate_of` |
| `session/1` | `date`, `date_basis` (folder/tracker/inferred), `status` | `instrument`, `grids`, `session_grids` (slot → Grid#), `box_folder`, `tracker_status` |
| `grid/1` | `sample` | `legacy_ids`, `replaced_by`, `prep_date`, `project`, `grid_type`, `replicate`, `storage`, `discarded`, `imaged` |
| `roboflow_image/1` | `stem`, `project` | `split`, `tags`, `uploaded`, `upload_batch`, `first_seen_pull`, `deleted` |
| `pull/1` | `path`, `project`, `created`, `n_images`, `coco_sha256` | `image_set` |
| `roboflow_export/1` | `path`, `counts` | `rf_version`, `derived_from`, `split_method`, `split_file`, `classes` |
| `label_version/1` | the version.json fields from LABEL_CHANGES.md, plus `status` | `labels_sha256`, `supersedes`, `superseded_by`, `pushed_to` |
| `review_package/1` | `path`, `kind` (review/sandbox/control/part), `base`, `base_kind` (pull/export), `n_images`, `built`, `status` | `parent`, `parts`, `model`, `run`, `predictions_dir`, `flag_conf`, `reviewers`, `n_edit_records` |
| `inference_check/1` | a `batches.jsonl` line (`batch_id`, `created`, `session`, `model`, `n`, `status`) with `id` = `batch_id` | everything else in batches.jsonl |
| `dataset/1` | `created`, `source` {kind: label_version or roboflow_export, id}, `split` {method, file, sha256}, `counts`, `classes`, `images_sha256`, `status` | `label_version`, `export`, `class_map`, `excluded_sessions` |
| `job_bundle/1` | `dataset`, `created`, `cryoEV_commit`, `run_experiments_args`, `zip_path`, `status` | `weights_init`, `ultralytics_bundled_on`, `zip_sha256` |
| `run/1` | `dataset`, `config` {arch, imgsz, batch, epochs}, `status` | `job_bundle` (null + `job_bundle_gap`), `machine`, `cryoEV_commit`, `metrics`, `experiment_log_timestamp` |
| `model/1` | `run` (nullable), `weights`, `imgsz`, `status` | `dataset`, `weights_sha256`, `classes`, `conf`, `init_from`, `scores` (inference_check ids) |

- `images_sha256` = sha256 of the sorted `<split>\t<stem>\n` lines. It is the frozen image list. The split file is kept next to it.

## Links
| From | → To | Field |
|---|---|---|
| image | session, grid, roboflow_image | `session`, `grid`, `rf_id` |
| session | grids | `grids`, `session_grids` |
| grid | renumbered grid | `replaced_by` / `legacy_ids` (tracker "Grid ID Changes") |
| pull | roboflow_images | its `current.coco.json` (`extra.roboflow_id`) |
| review_package | pull (or older: export), model/run, parent | `base`, `model`/`run`, `parent` |
| edit record | review_package, image | wrapper `pkg`, `file` → `image_key(file)` (LABEL_CHANGES.md) |
| label_version | pull, review_package, previous version | `base`, `pkg`, `supersedes` |
| label_change row | label_version, image, roboflow_image, person | `label_version`, `image`, `rf_id`, `who` |
| inference_check | session, model | `session`, `model`. Its `export/` labels are test data for that session |
| dataset | label_version **or** roboflow_export, split list | `source`, `split.file` + `split.sha256`, `images_sha256` |
| job_bundle | dataset, init weights | `dataset`, `weights_init` |
| run | dataset, job_bundle, config | `dataset`, `job_bundle`, `config` |
| model | run, dataset, init model, scores | `run`, `dataset`, `init_from`, `scores` |

- "What trained on what" means model → run → dataset → label_version / export → images (`split.file`).
- "What's being relabeled" means images with an edit record in an open package, or a row in a label version that isn't pushed yet.

## Statuses (queue and history view)
| Page lane | Entity: status | Set by |
|---|---|---|
| queued | dataset `frozen`; job_bundle `bundled`/`sent`; run `queued` | Dataset build, `make_job_bundle.py`, a manual line |
| staged | inference_check `staged`; review_package `built` | `inference_check.py stage`, `build_review_package.py` |
| in review | review_package `in_review` (≥ 1 edit record) | Derived from `edits/`, or a manual line |
| reviewed | review_package `reviewed`; inference_check `reviewed` | Manual line / `inference_check.py log` |
| applied | label_version `created` → `qc_passed`; review_package `applied` | `apply_review_edits.py`, then QC |
| pushed | label_version `pushed` (+ `pushed_to`) | Push step (needs the user's OK) |
| trained | run `running` → `finished`/`failed`; dataset `trained`; job_bundle `returned` | Importer of returned run folders |
| scored | inference_check `scored`; model `candidate` → `production` / `retired` | `inference_check.py score`, the user |
| (closed) | review_package `superseded`/`abandoned`; label_version `superseded`; session `excluded` | Manual line |

- An image's stage is **derived**, not stored: imaged → uploaded → in review → relabeled → pushed → in dataset → trained on.

## Existing artifacts → contract (validator on real data, 2026-10-30)
| Artifact | Maps to | Found / linked | What's missing |
|---|---|---|---|
| `roboflow_pull_20261029_1811` | pull + 170 roboflow_image | 1/1, 170/170 | Project name isn't stored in the folder |
| `tracking/v2/*.csv` + pull + session MANIFESTs + upload MANIFESTs | image | 846 / 338 | 504 no inventory Grid#; 58 no session date; 44 session year inferred; `blind_NN` upload copies not linked to their source tile |
| master CSV + tracker "Imaging Sessions" + folders | session | 37 / 6 | 28 pre-manifest sessions are not in the master CSV; 20261021 and 20261027 are not in the tracker sheet yet |
| tracker "Cryo Grids" + master | grid | 161 / 142 | 19 renumbered (need `replaced_by`); 8 master rows have no inventory id |
| `CryoAI/training images/*` | roboflow_export | 8 / 3 | 5 are empty local placeholders (the COCO is on Box/desktop) |
| `labels/lv_*` | label_version | 2 / 2 | `lv_20261030_0846` superseded (SUPERSEDED.md). Both changes.jsonl files: 122/122 rows valid |
| `review/*/data.js`, `CryoAI/annotation review/*` | review_package | 13 / 2 | 11 have pre-v1 edit records (no `pkg`/`fmt`/`rx`); id `prelipo_v5F5_review` is used by 2 folders (the SANDBOX copy); no status is recorded anywhere |
| `inference_checks/batches.jsonl` | inference_check | 1 / 1 | No `schema` on lines (mapped as legacy) |
| `training outputs/*` + models.yaml | dataset / run | 2/0, 3/0 | No label_version (trained from exports); no job bundles; Run_1/Run_2 folders and all `args.yaml` are not local |
| `models/models.yaml` | model | 6 / 3 | 3 have no `source_run` (v3_run1, allLayer_run14, yifei); 2 legacy ids; no status or weights sha256 |
| `job_bundles/` | job_bundle | 0 | None made yet |

**Gaps that can't be backfilled from local data:**
- Run → job bundle for every existing run. They predate bundles, so set `job_bundle_gap: "pre-bundle run"`.
- Dataset → label_version for every existing dataset. They were trained from Roboflow exports.
- Image lists for datasets before v5: only on Box/desktop, and only if those exports still exist.
- The run for `v3_run1_20260824`, `allLayer_run14_20260713` and `yifei_202512`. Config for Run_1/Run_2 needs `EXPERIMENT_LOG.csv` from the desktop.
- Roboflow ids of images deleted before the first pull. Per-image label history between exports. Who edited in Roboflow (`editors.csv` is mostly empty).
- `pkg`/`rx` on pre-v1 edit records (the package comes from the folder). `who_auth` (no logins).
- Session dates for the 58 `UNKNOWN` pre-manifest images. Grid# for pre-manifest grids.
- Review-package statuses before now. Only edits-present / applied can be derived.

## Lipo/VLDL (session 20261012)
- It belongs to another session/project. Its images are in the v5 exports and S1 (`split_assignment.csv` batch `20261012 …`), and its own session folder exists.
- The contract covers them like any session, but **nothing about them is written** here, in samples or in reports. The validator skips them and only counts `excluded`.

## Registry (v1.1, `registry/registry.py`)
- **One file per entity:** `CryoAI-laptop/registry/<entity>.jsonl` (local, not Box). Writes go through `data_utils/path_guard.py`.
- **Last means file order**, not `logged`. `current()` → `{id: record}`, `history(id)` → lines oldest first.
- **Single writer.** Owner logs (`inference_checks/batches.jsonl`, `labels/lv_*/version.json`) stay the source of truth and are read by the importers, not copied.
  `inference_check` and `label_version` lines are pointers (`owner` = `<batches.jsonl>#<batch_id>` / `<version.json>`) plus status lines.
  The rest of the record is derived from the owner file at read time.
- **Explicit beats derived.** A `set-status` line (with `status_by`, a `people.json` id) wins over any derived status, and the importers keep it.
- Model `production` is set only by `registry.py set-status <id> production --who pujan`.
- **Image keys.** Legacy keys (in a pull or `tracking/v2`) stay as they are. New session tiles (only in a session MANIFEST) are `<session>__<key>`.
  Generic upload/labeling copies (`blind_NN`) are `<session>_<key>` (e.g. `20261027_blind_03_png`), with `copy_of` from `*selection_PRIVATE.csv` when it names a registered tile. Files are not renamed.
  The importer refuses an existing key that arrives with a different session, and a stem that a second Roboflow id claims. It reports them and writes nothing for them.
- **Model ids** may contain uppercase (`^[A-Za-z0-9][A-Za-z0-9_]*$`). This applies to `model.id`, `init_from`, `review_package.model` and `inference_check.model`.
  Legacy ids become `aliases` of a canonical `v0_…` id: `allLayer_run14_20260713` → `v0_alllayer_y8n_20260713`, `yifei_202512` → `v0_yifei_y8n_20251201` (the day is unknown). `models.yaml` keys are unchanged.
- **Child → parent links only.** Removed: `model.scores`, `review_package.parts`, `label_version.superseded_by` (the new version keeps `supersedes`), `session.grids`, `grid.legacy_ids`, `image.rf_id`/`roboflow_project` (now `roboflow_image.stem` → image).
  Grid and session records hold only an id and links.
- Datasets may link straight to a local `lv_*` (`source.kind: label_version`), with no Roboflow push needed. `roboflow_export` has an optional `label_version`.
- Dataset `images_sha256` / `split.sha256` may be null, with an `images_sha256_gap` / `split.sha256_gap` reason. Never use fake all-zero hashes.
- `SANDBOX*` review packages are left out of the registry. `label_change/1` status also allows `duplicate-dropped`, which `apply_review_edits.py` writes.

## Open questions (user)
Answered on 2026-10-30, see the Registry section: 1 (per-entity files), 2 (session-prefixed ids + `copy_of`), 3 (SANDBOX excluded), 4 (local lv allowed), 5 (pujan sets production; v5 F5 = production), 6 (aliases).

1. Where does the registry live? Proposal: `CryoAI-laptop/registry/<entity>.jsonl` (local, not Box).
2. Blind and labeling upload copies (`blind_01.png` …) have generic names that will collide across sessions. Either give them session-prefixed names, or add a `copy_of` link (the mapping is now only in a PRIVATE selection file).
3. `SANDBOX_prelipo_v5F5_review` reuses the id `prelipo_v5F5_review`. Should it be renamed, or should SANDBOX folders be left out of the registry?
4. Should future datasets always come from a pushed label version (via a re-export), or may they be built straight from `lv_*/labels.coco.json`?
5. Who promotes a model to `production`, and is `candidate` the right state for v5 F5 now?
6. Legacy model ids (`allLayer_run14_20260713`, `yifei_202512`): keep them as they are, or alias them?

## Suggested changes to files owned by others (not made)
- `inference/inference_check.py`: add `"schema": "inference_check/1"` (and `id` = `batch_id`) to each `batches.jsonl` line.
- `annotation/apply_review_edits.py`: when a version replaces an older one, write `supersedes` into the new version.json. Keep SUPERSEDED.md, but also log a `label_version` status line.
- `annotation/build_review_package.py`:
  - data.js `id` = folder name;
  - explicit `base` (pull id) and `model` id fields instead of the free-text `run`;
  - `parent` on part packages.
- `training/make_job_bundle.py`: append a `job_bundle/1` line, and put `bundle_id` into JOB.json.
- `training/run_experiments.py`: write `run.json` (`run/1`: dataset, bundle id, commit, config) into each run folder.
- `models/models.yaml`: add `run`, `dataset`, `status` and `weights_sha256` keys.
