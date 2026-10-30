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
| `status` | `applied`; `conflict` with `reason` (nothing written for it); or `duplicate-dropped` with `reason` (a same-class copy of an outline that is kept). Both are also in `conflicts.csv`. An applied row may carry a warning in `reason` |

- Conflicts are not written: the same outline twice with different classes, a `g<id>` missing from the base,
  a prediction index out of range (including `p-1`), an outline that fails to clamp to a valid polygon. A
  conflict on a base label leaves it as in the base.
- Same outline = mask IoU >= 0.9 at full resolution (`--dup-iou`). Outlines linked at that IoU form a group:
  - classes differ: every change in the group is a conflict;
  - same class, one label with an id (base or reshaped): it stays, each add/new is `duplicate-dropped`;
  - same class, only adds/new: the earliest (`t`, then key order) stays, the rest are `duplicate-dropped`;
  - same class, two labels with ids, one of them changed: the change is a conflict (a label is never dropped).
- Pairs of base labels that already overlap at >= 0.9 are not flagged; `version.json` lists them as `base_duplicates`.
- Near duplicate: an add/new over a label of another class at IoU >= 0.5 is applied, with a warning
  (`conflicts.csv` `severity: warn`).
- Outlines are compared numerically (within 1e-9, so `0` equals `0.0`): a decision that restates the outline is a `keep`.
- `conflicts.csv` columns: `severity` (`conflict`, `dropped`, `warn`), `image`, `rf_id`, `file`, `dec_key`, `op`,
  `reason`, `ann_old`, `cls_from`, `cls_to`, `who_raw`, `who`, `t`, `seed`.
- Not logged (no label change, no verdict): `null` undos, and an `n<uid>`/`p<k>` removed again; counted in
  `version.json` `skipped`. An old Editor-list item with no key is logged with `dec_key` `gt` (op `new`, or the
  change it made to a label); its `seed` comes from the item.
- The run refuses: a package whose labels differ from the base (per image the same ids; per label the same
  class and an outline within 0.5 px), a `data.js` `source` that is neither `--base` nor a byte-identical copy,
  duplicate image names, a record whose `pkg` is another package, an `--out-root` under Box or
  Cryo-image-processing (any spelling: `data_utils/path_guard.py`), and an existing `lv_` folder.
- A class not in the base categories (e.g. VLDL on a pre-lipo image) is added as a category and flagged in
  `version.json`. Pushing it to Roboflow needs the user's OK.
- Rerunning on the same inputs gives the same `labels.coco.json` hash and no duplicate rows.
