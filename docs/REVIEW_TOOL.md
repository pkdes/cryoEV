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
   Open the printed `http://127.0.0.1:8765/?k=<key>` link. The key is kept in `<package>/.review_key`.
2. Phone, with Tailscale installed and signed in to the same tailnet on both devices. Pick one option:
   - **Keep the server local (preferred).** Run `tailscale serve --bg 8765` on the laptop, then open `https://<laptop>.<tailnet>.ts.net/?k=<key>` on the phone. Only tailnet devices can reach it, over HTTPS. Stop it with `tailscale serve reset`.
   - **Bind to all interfaces.** Restart the server with `--host 0.0.0.0`. It prints `Tailscale link (phone): http://100.x.y.z:8765/?k=<key>`. Allow Python through Windows Firewall on private networks. This also exposes the port on the LAN, still protected by the key.
3. On the phone, add `&who=<name>` to the link (it names the reviewer) and `&mode=classes` (it opens class review). Example: `…/?k=<key>&who=Ann&mode=classes`. The `?k=` link sets a 30-day cookie.
4. Stop the server with Ctrl+C.

## Classes
The class list always follows the Roboflow names, with hotkeys **1–8**: `EV, Low conf EV, MV, inner, Incomplete, Overlapping, VLDL, non-EV`.
- The order is fixed whatever order `data.js` `categories` uses. Old packages list categories alphabetically and lack VLDL; VLDL is still offered.
- Any other category in `data.js` follows as 9 and up.
- The colours are fixed per class.

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

In Draw / Vertex mode, a class key only sets the class for the next shape. It doesn't reclass the selection.

**Queue** (left panel):
- Every image, with its status: unreviewed, `•` edited, `✓` reviewed.
- Flag counts, in their flag colours: labels with no prediction (`fg`), predictions with no label (`fp`), class disagreements (`fx`), unmatched Low conf EV (`flc`).
- Split, and 👤 *who* has the image or one of its cards open (from `/api/claims`). The Editor claims `<file>|editor` and renews it every 40 s.
- Filters by split and status (unreviewed / edited / not reviewed / reviewed).
- Sorts by most flags (all four counts), package order, or name.

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
| any | `null` | Undone: ignore |

- Redraw and vertex edits create **no new value type**. They write `keep:<class>` + `p` for a label, or `add:<class>` + `p` for an accepted prediction, as card reshapes already did.
- Since this version, every Editor change also writes that object's decision: reclass, redraw, vertex move / add / delete, new shape. So an Editor change merges per object, like card decisions.
- Old records stored bare strings (`"keep:MV"`). They are read as `{v, t: record t, who: ""}`.

**Polygon list** `gt` + `gtT` (kept for compatibility):
- The Editor still saves the full list of labels as it stands (`gt`, items `{id, c, p, src?}`; `src` = the `p…`/`n…` key of an added one), and `gtT` is its time.
- If present, the list replaces the package labels as the base. Decisions apply on top of it, and a decision's `p` applies only if the decision is newer than `gtT`.
- A record with `gt` but no `gtT` counts only if `modified` (then `gtT = t`). Otherwise the list is ignored.

**Merge** (`review_server.merge_edits`, mirrored by `mergeEdit()` in the page):
- Per decision key, the newest `t` wins (a tie keeps the saved copy).
- `gt` / `gtT`: the newest `gtT` wins as a whole list.
- `reviewed`: the newest `reviewedT` wins. `modified`: OR. `t`: max.
- A decision's `t` is always above its record's `gtT`, so it applies on top of the list even with clock skew.

**Apply without the tool:** `review_server.effective_labels(image, edit)` returns the labels the page shows, as `[{id, c, p, src}]`. The tests check that it matches the page on the fixture and on all September edits. The planned `apply_review_edits.py` should use it.

Known limit: two reviewers editing *different* objects' polygons in the Editor, both offline, each save a full `gt` list, and the newer list wins. Their per-object decisions still apply on top if they are newer than that list, so in practice this is only a problem for legacy records without decisions.

## Tests
```bash
python -m unittest discover -s annotation/tests -v
```
- `test_review_server.py`, standard library only: a live server on a free 127.0.0.1 port. Covers ping, PUT, a merge by timestamp, the new / redraw decisions, rejected bad edits, claims, and access-key rejection. Also merges and applies the real September edits, read-only, and checks that they change some labels.
- `test_review_tool_browser.py`, Playwright with headless Edge or Chrome (skipped without one):
  - script syntax (no node on the lab PC, so the browser parses it);
  - class order;
  - queue sort and filter;
  - reclass, accept, freehand, tap-vertex, redraw and vertex drag;
  - class review at 400 px with touch;
  - page labels == `effective_labels()`, on the fixture and on the September edits.
- Real package search order: `$CRYOEV_REVIEW_PACKAGE`, then `CryoAI/annotation review/cryoai_v4_review` beside the repo, then the Box copy. It is copied to a temp folder and never written.
- `REVIEW_TEST_SHOTS=<dir>` keeps phone screenshots.
