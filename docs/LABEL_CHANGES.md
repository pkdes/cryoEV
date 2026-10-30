# Label changes: edit records → label versions → change log (contract v1)

How review edits become a new label set, and what is recorded about each change. Piece 2 (the full
ids/batches/datasets/runs contract) extends this file; it does not replace these fields.

## Inputs
| Input | What | Where |
|---|---|---|
| Base labels | One pinned Roboflow pull, read-only. `g<id>` decision keys are **local ids from this pull** | `CryoAI-laptop/roboflow_pull_<YYYYMMDD_HHMM>/current.coco.json` |
| Review package | `data.js` (id, images, gt, pr) built from that base | `CryoAI-laptop/review/<package>/` |
| Edit records | One file per image, written by `annotation/review_server.py` | `<package>/edits/<file>.json` |
| Roster | `annotation/people.json`: `{id, name, aliases}` | repo |

- Edits are applied to their own base pull only. Never key edits against a newer pull.
- Before a push, re-pull and diff per image against the base. Any drift on an edited image → stop.

## Edit record file (`fmt` 1)
```
{"file": <data.js file>, "pkg": <data.js id>, "fmt": 1, "edit": {...}, "rx": {<dec key>|"gt": server ms}}
```
- `edit`: the record the tool saves (`dec`, `t`, `modified`, `gt`, `gtT`, `reviewed`, `reviewedT`). Its keys and
  values are listed in [REVIEW_TOOL.md](REVIEW_TOOL.md). `review_server.effective_labels()` is the reference applier.
- `t` values are the tool's **logical clock** (never earlier than what they replace), not wall time.
  `rx` is when a change reached the server (wall time, ms). Older files have no `pkg`/`fmt`/`rx`.
- A decision may carry `seed` (`september:<v4 file>:<v4 key>`): carried over from the September review by
  `resolve_review_edits.py`, with its September `who` and `t`. It is not new review work.
- `merge_edits()` rebuilds `edit` from fixed keys. New per-record data goes in the wrapper, not inside `edit`.

## Apply rule
- An image is applied when its record has any decision or a polygon list (`modified`).
- `reviewed` does not gate applying. It is logged per row as a status.
- Zero edits → output labels equal the base (same annotations, same ids).

## Label version
- id: `lv_<YYYYMMDD_HHMM>`, written to `CryoAI-laptop/labels/lv_<YYYYMMDD_HHMM>/`. Never overwritten.
- Folder: `labels.coco.json`, `changes.jsonl`, `conflicts.csv`, `version.json`.
- `version.json`: `{id, base, pkg, edits_dir, n_images_applied, counts_by_op, class_groups_sha256, people_sha256,
  tool: "apply_review_edits.py <git sha>", created}`.
- COCO objects: internal fields (`u, x, lc, seed, src, t`) are stripped; bbox and area are recomputed from the
  outline; coordinates are clamped to the image. An object with several rings stays one annotation.
- Annotation ids: kept for surviving base objects; new objects get `max(base id) + 1, …`.

## Change log row (`changes.jsonl`, one JSON object per line)
| Field | Meaning |
|---|---|
| `schema` | `"label_change/1"` |
| `label_version`, `base`, `pkg` | Output version id, base pull folder name, data.js id |
| `image`, `rf_id` | Image stem (key everywhere), Roboflow id from the base COCO |
| `dec_key` | `g<id>`, `p<idx>`, `n<uid>`, or `gt` for a polygon-list item with no decision |
| `op` | `keep`, `reclass`, `reshape`, `reclass+reshape`, `delete`, `add` (accepted prediction), `new` (drawn), `reject` (prediction refused: no label change, counted for model scoring) |
| `ann_old`, `ann_new` | Base annotation id / output annotation id (null when none) |
| `cls_from`, `cls_to` | Class before / after (null when none) |
| `who_raw`, `who` | Name as saved / roster id (`unknown` if no alias matches) |
| `who_auth` | Reserved: identity verified by a login (null until there is one) |
| `t`, `rx` | Tool logical time / server receive time (ms, null if absent) |
| `seed` | September seed key, or null |
| `reviewed` | The image's reviewed flag |
| `status` | `applied`, or `conflict` with `reason` (row also in `conflicts.csv`, nothing written for it) |

- Conflicts are not written: the same outline twice with different classes, a `g<id>` missing from the base,
  a prediction index out of range, an outline that fails to clamp to a valid polygon.
- A class not in the base categories (e.g. VLDL on a pre-lipo image) is added as a category and flagged in
  `version.json`. Pushing it to Roboflow needs the user's OK.
- Rerunning on the same inputs gives the same `labels.coco.json` hash and no duplicate rows.
