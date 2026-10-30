# Label review tool

One HTML page (`annotation/review_tool/index.html`) for reviewing labels against model predictions:
- a desktop **Editor** for polygons;
- **cards** ("eVHarmony") for phones, in two modes: flagged-object triage, and class review.

`annotation/build_review_package.py` builds a package: `index.html`, `data.js` and `images/`. The package works offline from a folder or zip, using Export / Import JSON. It can also be served by `annotation/review_server.py` to several reviewers at once, with each image's edits saved to `<package>/edits/<file>.json`.

## Demo: laptop, then a phone over Tailscale

1. On the laptop (fixture package; use a real package path instead to review real data):
   ```bash
   python annotation/tests/make_fixture.py
   python annotation/review_server.py --package annotation/tests/fixture_review
   ```
   Open the printed `[OK] Link: http://127.0.0.1:8765/?k=<key>`.
   - The server binds to `127.0.0.1` by default.
   - The key is printed only in that startup line, and request logs show `k=***`. It is also kept in `<package>/.review_key`.
2. Phone, with Tailscale installed and signed in to the same tailnet on both devices. Pick one option:
   - **Keep the server local (preferred).** Run `tailscale serve --bg 8765` on the laptop, then open `https://<laptop>.<tailnet>.ts.net/?k=<key>` on the phone. Only tailnet devices can reach it, over HTTPS. Stop it with `tailscale serve reset`.
   - **Bind to the Tailscale address.** Restart with `--host <IP from tailscale ip -4>`, e.g. `--host 100.101.102.103`, and open `http://100.101.102.103:8765/?k=<key>` on the phone. Allow Python through Windows Firewall when asked.
   - Don't use `--host 0.0.0.0`: it also opens the port to the LAN. The server warns if you do.
3. On the phone, add `&who=<name>` to the link (it names the reviewer) and `&mode=classes` (it opens class review). Example: `…/?k=<key>&who=Ann&mode=classes`. The `?k=` link sets a 30-day cookie.
4. Stop the server with Ctrl+C.

## Classes
The class list always follows the Roboflow names, with hotkeys **1–8**: `EV, Low conf EV, MV, inner, Incomplete, Overlapping, VLDL, non-EV`.
- The order is fixed whatever order `data.js` `categories` uses.
- All 8 are always offered. Any that are missing from the package's `categories` are marked as **new class**: dashed and faded, with a tooltip and "(new class)" in the legend.
  - Example: the September v4 and pre-lipo v5 F5 packages have no VLDL.
  - Labels can still be set to such a class. **`apply_review_edits.py` must add a missing category to the COCO `categories` before writing those annotations.**
- Any other category in `data.js` follows as 9 and up.
- The colours are fixed per class.

**Hotkey change.** Class keys used to follow each package's own category order (the Editor), or a short fixed list (the cards). They are now the same everywhere:

| Key | Old Editor (v4 package: alphabetical) | Old cards | **Now (Editor and cards)** |
|:-:|---|---|---|
| 1 | EV | EV | **EV** |
| 2 | Incomplete | Low conf EV | **Low conf EV** |
| 3 | Low conf EV | non-EV | **MV** |
| 4 | MV | MV | **inner** |
| 5 | Overlapping | inner | **Incomplete** |
| 6 | inner | – | **Overlapping** |
| 7 | non-EV | – | **VLDL** |
| 8 | – | – | **non-EV** |

Most moved keys: cards 3 (was non-EV, now MV) and 5 (was inner); Editor 2, 3, 5 and 7.

## Editor (desktop)
| Action | How |
|---|---|
| Reclass the selection | `1`–`8`, or a class chip under the image. This also sets the **current class** for new shapes |
| Accept a selected prediction | `A` (current class), a class key, or "Accept as `<predicted class>`" |
| New shape, freehand | `D`, then drag around the object. The stroke is simplified (RDP, about 1 screen px) |
| New shape, tap-vertex | `V`, then tap points. Tap the first point, press `Enter` or press "Close shape" to finish. `Z` removes the last point |
| Redraw the selected outline | `E` or "Redraw outline", then draw with `D` or `V`. Class, id and decision key are kept |
| Move / add / delete a vertex | Drag a handle / double-click an edge / double-click a handle |
| Pan / zoom | Drag empty space. In Draw / Vertex: right, middle or Alt-drag, or two fingers. Wheel / pinch; `F` fits |
| Other | `Del` delete · `Z` undo · `R` reviewed + next · `N`/`P` next/prev · `T` predictions · `H` all outlines · `Esc` cancel |

Class keys depend on the mode:
- **Select:** the key reclasses the selection (or accepts a selected prediction as that class) and sets the current class.
- **Draw / Vertex, new shape:** the key only sets the class for the next shape.
- **Redraw:** the key reclasses the object being redrawn, at once. The redraw then continues, and the new outline keeps that class.

The selection and the Redraw target follow the object's decision key, not its position in the list. So a pull from the server mid-edit can't move them onto another object.
- Outlines drawn by older versions have no key. They are tracked by shape, and get an `n<uid>` key at their first Editor change.

**Queue** (left panel):
- Every image, with its status: unreviewed, `•` edited, `✓` reviewed.
- Flag counts, in their flag colours: labels with no prediction (`fg`), predictions with no label (`fp`), class disagreements (`fx`), unmatched Low conf EV (`flc`).
- Split, and 👤 *who* has the image or one of its cards open (from `/api/claims`). The Editor claims `<file>|editor` and renews it every 40 s.
- Filters by split and status (unreviewed / edited / not reviewed / reviewed).
- Sorts by most flags, package order, or name.
  - "Most flags" ranks on `fg + fp + fx`, with ties in package order.
  - `flc` is shown but doesn't count toward the rank.

**Flags** on the image (legend in the right panel). A flag stops being drawn once its object has a decision.

| Flag | Drawn as | Field |
|---|---|---|
| Prediction with no label | Thick yellow dashes | `pr[k].u` |
| Label with no prediction | Magenta halo | `gt[i].u` |
| Class disagrees with the matched object | Orange dotted halo (labels), orange dots (predictions) | `x: true` (optional) |
| Low conf EV to confirm | White dashed halo | `gt[i].lc: true` (optional) |

When a prediction is selected, the selection box shows its class, `pr[k].c` (optional).

## Cards (phone)
Narrow screens open the cards automatically. "Editor" goes back. The mode menu has two modes:
- **flagged objects** (triage, as before):
  - Unmatched predictions: swipe right = add as EV, left = not an object.
  - Unmatched labels: right = keep, left = delete.
  - Up = later. Class chips. ✏️ reshapes the outline.
- **class review** (no outline editing):
  - Shows one label with the model's class: the overlapping prediction with the best box IoU (≥ 0.3), so treat it as approximate.
  - Tap a big class button: the label is set to that class and the next card opens.
  - Swipe right or ✓ = keep the class. Swipe left / up or ↑ = next, with no decision.
  - Scope "class flags" = labels with `x`, then `lc` or class Low conf EV. Scope "all labels" = every label.
  - Keys: `1`–`8`, `Enter`/`→` keep, `←`/`↑`/`N` next, `Z` undo.

## `data.js` (from build_review_package.py)
`window.REVIEW_DATA = {id, source, run, created, categories: [...], images: [...]}`. Each image:
- Required: `{file, split, w, h, src, gt: [{id, c, p, u?}], pr: [{s, p, u?}], fg, fp}`.
- Optional, newer packages only: `pr[k].c` (predicted class name), `x: true` on gt/pr items, `lc: true` on gt items, image-level `fx`, `flc`.
- `p` is a list of rings. Each ring is a flat `[x1, y1, x2, y2, …]` normalized to [0, 1].

The tool runs on packages without the optional fields: those flags and counts are simply absent.

## Edit record (`<package>/edits/<file>.json` = `{"file", "edit"}`)
```
edit = {dec: {key: {v, t, who, p?}}, gt?, gtT?, reviewed, reviewedT, modified, t}
```
**Decisions** (`dec`): one per object.
- `t` is ms since the epoch; `who` is the reviewer name.
- `p` is the object's own outline (list of normalized rings). It is present whenever the object was reshaped or redrawn.

| Key | Values | Meaning |
|---|---|---|
| `g<label id>` (COCO annotation id) | `keep:<class>` [+ `p`] | Keep the label with this class (and outline `p`, if given) |
| | `del` | Delete the label |
| `p<prediction index>` | `add:<class>` [+ `p`] | Add the prediction as a label (outline `p`, else the prediction's) |
| | `no` | Not an object |
| `n<uid>` (**new**: an outline drawn in the Editor) | `new:<class>` + `p` (always) | Add this drawn outline as a label |
| | `del` | Deleted again |
| any | `null` | Undone. A label goes back to the package's version; an added object is dropped |

- Redraw and vertex edits create **no new value type**. They write `keep:<class>` + `p` for a label, or `add:<class>` + `p` for an accepted prediction, as card reshapes already did.
- Since this version, every Editor change also writes that object's decision: reclass, redraw, vertex move / add / delete, new shape. So an Editor change merges per object, like card decisions.
- Old records stored bare strings (`"keep:MV"`). They are read as `{v, t: record t, who: ""}`.

**Polygon list** `gt` + `gtT` (kept for compatibility):
- The Editor still saves the full list of labels as it stands, with `gtT` as its time. Items are `{id, c, p, src?, t}`, where `src` is the `p…`/`n…` key of an added object.
- If present, the list replaces the package labels as the base. Decisions apply on top of it.
- **Per-object time `t`** (lists saved by this version): `t` is when the item was last changed.
  - The object's decision applies whenever its `t` ≥ the item's `t`. Class, outline `p` (else the list's), `del`/`no`, and `null` (back to the package label, or dropped if added) all follow this rule.
  - So a per-object change survives a merge with anyone's newer list. Example: A redraws label 1 offline, then B deletes label 2. B's newer list still has label 1's old outline, but that item's `t` predates A's decision, so A's outline wins.
- **Items without `t`** (lists from before this version, such as the September records) keep the old rule: the class always applies, and a decision's `p` only if the decision is newer than `gtT`. On the September edits, old and new tool give byte-identical labels.
- When such a list is saved again, every item gets a `t`. It is the newer of the list's `gtT` and the item's decision `t`, so the labels don't change; the tests check this on all 33 September records.
- A record with `gt` but no `gtT` counts only if `modified` (then `gtT = t`). Otherwise the list is ignored.

**Merge** (`review_server.merge_edits`, mirrored by `mergeEdit()` in the page):
- Per decision key, the newest `t` wins (a tie keeps the saved copy).
- `gt` / `gtT`: the newest `gtT` wins as a whole list. With per-object `t`, this no longer overrides newer per-object decisions.
- `reviewed`: the newest `reviewedT` wins. `modified`: OR. `t`: max.
- **Clocks:** a new entry gets `t` = max(now, the entry it replaces + 1, its list item's `t` + 1, `gtT` + 1). So editing a value that came from a device whose clock runs ahead still wins. Edits made at the same moment on two devices with skewed clocks still resolve by clock: newest `t` wins.

**Undo** (Editor `Z`, card ↺) reverts only entries this browser tab wrote since the step being undone.
- If someone else's decision arrived in between, on any key, it stays.
- A label with no decision before the step gets `null` when it was the package's version, or else an explicit `keep:<class>` (+ `p`) restating how it stood. A stale full list elsewhere can't bring the undone change back.

**Validation** (`review_server.valid_edit`). A PUT gets a 400 and is never stored when it has:
- `NaN`/`Infinity`;
- an unknown decision key;
- a `new:` without `p`;
- a ring with < 3 points or a coordinate outside [0, 1] (± 1e-6);
- a `gt` item not shaped like a `data.js` label (`c` string, `p` rings, `id` int or null, `src` `p…`/`n…`, `t` number).

The page clamps coordinates. If the server rejects one image, the page shows a banner and still syncs the others.

**Apply without the tool:** `review_server.effective_labels(image, edit)` returns the labels the page shows, as `[{id, c, p, src}]`.
- The tests check that it matches the page on the fixture, the September edits and R1's pre-lipo package.
- The planned `apply_review_edits.py` should use it, and add categories missing from the export; see Classes.

Known limit: older items without a decision key or `t` still follow the whole-list rule. That means September-era list items until their first edit, and keyless hand-drawn outlines until the Editor keys them. Two offline reviewers who both change such an item: the newer list wins.

## Tests
```bash
python -m unittest discover -s annotation/tests -v
```
- `test_review_server.py`, standard library only: a live server on a free 127.0.0.1 port.
  - API: ping, PUT, a merge by timestamp, the new / redraw decisions, claims (and their cap), access-key rejection.
  - Hardening: rejected bad edits and bad values, never stored; bad Content-Length; idle-socket timeout; key redacted from logs; default host.
  - Records: an offline reshape survives a newer list; the real September edits merge and apply, read-only, and change some labels.
- `test_review_tool_browser.py`, Playwright with headless Edge or Chrome (skipped without one):
  - script syntax (no node on the lab PC, so the browser parses it);
  - class order, and the new-class mark (fixture, and R1's pre-lipo package);
  - queue sort (`flc` excluded from the rank) and filters;
  - reclass, accept, freehand, tap-vertex, redraw and vertex drag;
  - two reviewers: offline redraw vs a later Editor save; undo vs someone's newer decision; selection and Redraw through a pull;
  - class review at 400 px with touch;
  - page labels == `effective_labels()`, on the fixture and on both real packages. Re-saving the September records changes no labels.
- Real packages are copied to a temp folder and never written.
  - September v4, searched in order: `$CRYOEV_REVIEW_PACKAGE`, then `CryoAI/annotation review/cryoai_v4_review` beside the repo, then the Box copy.
  - Pre-lipo, searched in order: `$CRYOEV_PRELIPO_PACKAGE`, then `CryoAI-laptop/review/prelipo_v5F5_review`.
- `REVIEW_TEST_SHOTS=<dir>` keeps phone screenshots.
