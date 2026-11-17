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
3. On the phone, open the link. Optional: `&mode=classes` opens class review, and `&who=<name>` names the reviewer without asking. Example: `…/?k=<key>&mode=classes`. The `?k=` link sets a 30-day cookie.
   - The cookie is `review_key_<port>` (the server's listen port). A browser sends a host's cookies to every port, so with one name the real review (8766) and the sandbox (8765), each with its own key, logged each other out.
   - The old `review_key` cookie is still accepted when it holds this server's key, so current reviewers stay signed in. Only the new name is set.
   - The first visit asks **Who is reviewing?** (see [Reviewer name](#reviewer-name)). The name is then remembered on that device.
4. Stop the server with Ctrl+C.

**Sandbox.** `--sandbox`, or a package folder named `SANDBOX_…`, puts a fixed amber bar on every view: "SANDBOX — edits here don't count toward the real review". The page moves down by the bar's height, so it covers no control.
- The flag reaches the page through `data.js`: the server appends `window.REVIEW_SERVER = {"sandbox": true};` when it serves it. The file on disk is unchanged, and a real review's `data.js` is served byte for byte.
- Example: `python annotation/review_server.py --package <SANDBOX_copy> --port 8765`.
- **Drafts are kept apart.** A `SANDBOX_` copy has the real package's id, so the page stores sandbox drafts under `cryoev-review:<id>:sandbox` (from the server's flag). A real package keeps the exact old key, `cryoev-review:<id>`, so reviewers' unsaved drafts are still picked up. Before this, a real review on the same origin as the sandbox would have uploaded sandbox drafts as its own.

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

## Reviewer name
- A picker lists the roster from `GET /api/people` (`annotation/people.json`). It is asked once per browser and kept in localStorage (`cryoev-review:who`).
- **An edit is never saved without a name.** Any edit (Editor or cards) opens the picker first; once a name is picked, that edit goes through. "Not now" lets you look around without editing.
- Saved `who` = the roster **id** (e.g. `pujan`), shown as the roster name. "Other…" takes free text, saved as typed. It must fit what the server accepts (`review_server.valid_who`: letters, digits, spaces, `. _ -`, at most 40); otherwise Save stays off and a message says why. A `?who=` or stored name that doesn't fit is ignored and the picker asks.
- If the card or the Editor selection changes while the picker is open, the edit that asked is dropped rather than applied to the new object.
- No claims are made until there is a name.
- Records saved offline by an older version can hold entries with an empty `who`. Before uploading them, the page asks for a name and writes it into those entries only. Entries already on the server, which may be someone else's, are left alone, and `t` doesn't change.
- If `/api/people` fails, or the page is opened from a folder or zip, the picker asks for a typed name (required).
- `&who=<name>` in the link, or a name typed into the old prompt, becomes the roster id when it matches an id, name or alias (any case, e.g. `Puj` → `pujan`). Otherwise it is kept as given.
- The current name shows in the Editor header and the cards' filter row (👤). Tap it to change.
- Claims (`/api/claim?who=`) use the same value.
- On the server, the `who` of a request's new decisions and of its claim goes through one function, `review_server._request_identity(handler, claimed)`. Today it returns the picker's value unchanged. Behind the planned portal it is where a verified identity goes: an `X-CJ-User` header, trusted only after the gateway secret is checked. That is not implemented, and no header is read yet.

**Escaping:** strings from edit records and claims (who, seed, a decision's class) are HTML-escaped wherever the page builds HTML. The server also refuses names with markup.

## Deep links
`?img=<file>&obj=<decision key>&mode=editor` opens the Editor with that image loaded and the object selected and centered. Opening it directly or from a bookmark works the same.
- `obj` is a decision key: `g<id>`, `p<idx>` (a prediction, or the label it became once accepted) or `n<uid>`. `img` may also be the file's stem.
- Each card's **Open in editor ↗** is such a link. It keeps the page's `k=` (so it also works on a device without the cookie) and drops `who=`. Tapping it opens the Editor in place, and Back returns to the same card. Long-press it to copy or bookmark it.
- After it opens, `img`, `obj` and `mode` are removed from the address bar (`replaceState`, `k=` stays), so a reload doesn't jump back to the object.
- If the object is gone (deleted or undone), the Editor centres where it was and says so. A prediction is drawn even if the confidence slider was above its score (the slider is lowered to it).
- The server's access-key check is unchanged: without `k=` or the cookie, the link gets 403.

## September-carried decisions ("Sept")
Display only. A decision with `seed` (`september:<v4 file>:<v4 key>`, written by `resolve_review_edits.py`) was carried over from the September review.
- Editor: a small **Sept** tag above the label's outline (the header's "Sept" checkbox hides them), a line in the selection box, and a legend entry.
- Queue: images with seeded decisions show **Sept n**.
- Cards (user decision, 2026-10-30): a seeded decision counts as **not yet checked**. The cards include every seeded object: in class review, seeded labels, accepted predictions (`add:`) and drawn outlines (`new:`); in triage, any seeded `g`/`p`/`n`. They come after the image's flagged cards, and they count in "n labels to check".
  - The card shows the **Sept** tag and the September decision in blue: "Sept: keep as non-EV", "Sept: add MV", "Sept: new inner", "Sept: delete".
  - ✓ / ♥ / swipe right confirms it: the same value and outline are written again, fresh, credited to the reviewer.
  - A class pick (same or different class) writes a fresh decision. An accepted prediction or drawn outline keeps its September outline.
- The tool never writes `seed`. A reviewer's new decision on that object has none, so it is resolved and its tag goes. Undoing that decision restores its value but not its `seed`.

## Editor (desktop)
| Action | How |
|---|---|
| Reclass the selection | `1`–`8`, or a class chip under the image. This also sets the **current class** for new shapes |
| Accept a selected prediction | `A` or the green **Accept as `<predicted class>`** button: its predicted class (`pr[k].c`; packages without it: the current class). A class key accepts it as that class |
| Reject a selected prediction | `X` or the red **Reject** button: writes `no`. `X` on an accepted prediction rejects it again (as `Del`) |
| Edit a prediction's outline | Drag a handle, click its outline (adds a point) or right-click a handle (deletes it), on a selected prediction that isn't accepted (undecided or rejected). That **accepts** it: `add:<predicted class>` + `p` = the edited outline, in one undo step. A message "Accepted (edited) — Undo" offers to undo it (also `Z`) |
| New shape, freehand | `D`, then drag around the object. The stroke is simplified (RDP, about 1 screen px) |
| New shape, tap-vertex | `V`, then tap points. Tap the first point, press `Enter` or press "Close shape" to finish. `Z` removes the last point |
| Redraw the selected outline | `E` or "Redraw outline", then draw with `D` or `V`. Class, id and decision key are kept |
| Move / add / delete a vertex | Drag a handle (the outline follows live) / click the selected outline / right-click a handle (phone: double-tap) |
| Pan / zoom | Drag empty space. In Draw / Vertex: right, middle or Alt-drag, or two fingers. Wheel / pinch; `F` fits |
| Next / previous object | `Tab` / `Shift+Tab` (round 16a #25b): selects the next shown object on this image (labels and not-yet-accepted predictions, in reading order of their centres) and centres it, as "Open in editor" does. Select mode only; while a button or field has the focus, Tab moves the focus as usual. Object sets keep their own Tab |
| Other | `Del` delete · `Z` undo · `R` reviewed + next · `N`/`P` next/prev · `T` predictions · `H` all outlines · `Esc` cancel |

A selected prediction shows its state (not decided / accepted as … / rejected, with who) and big **Accept** / **Reject** buttons under it. U4 (2026-11-02) changed `A`: it used to accept as the current class; it now uses the predicted class, and "Accept as `<current class>`" is a second button when the two differ.

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
| Prediction with no label, undecided | Thick yellow **dash-dot**. The dash-dot pattern stays with Flags off (thin, cyan) until the prediction has a decision | `pr[k].u` |
| Label with no prediction | Magenta halo | `gt[i].u` |
| Class disagrees with the matched object | Orange dotted halo (labels), orange dots (predictions) | `x: true` (optional) |
| Low conf EV to confirm | White dashed halo | `gt[i].lc: true` (optional) |

When a prediction is selected, the selection box shows its class, `pr[k].c` (optional).

## Image contrast (☼, round 16a, display only)
Every view has a **☼** (◐ stays Low conf) button in its top bar (Editor header, Quick check bar, object-set bar). It opens a small panel:
**Brightness**, **Contrast**, **Gamma** sliders, **Auto**, **Smooth** and **Reset**.
- **Auto** stretches the darkest 0.5 % to black and the brightest 0.5 % to white, measured over what is on screen: the
  Editor's current view (zoom in and the levels follow the region), or the card's crop. Brightness / contrast / gamma
  apply on top. Off by default; once on it stays on.
- **Smooth** (r16c: a slider, 0-3, 0 = off) softens the grain: a Gaussian blur whose sigma is in full-resolution
  pixels (the item crops); the 2x image gets sigma / 2 and the 1440 base image sigma / 4, so the look matches. It runs
  on the cached copy (the canvas `blur()` filter, else a separable kernel of radius ceil(3 sigma)), once per image and
  strength; on a page that can't read pixels, the screen-space canvas filter; hidden where neither works. Saved
  r16a settings with Smooth on read as 1.0.
  r17b: the / 2 and / 4 apply only when a finer level exists (the image has `hi` or the item `chi`). An image that
  is the finest there is (e.g. a 1440 px upload) gets sigma = Smooth in its own pixels (Smooth 2 = sigma 2 px), so
  Smooth is visible there; sets with `hi` / `chi` are unchanged (`vcSigma`; `test_review_r16b_browser.py`
  `SmoothLevelTests`). The panel tooltip says "in the image's finest pixels".
- Settings are per reviewer, in this browser (localStorage `cryoev-review:contrast:<who>`). Reset = the image as
  delivered. The button is lit while anything is set.
- **Display only:** the page draws a lookup-table copy of the image (an offscreen canvas at the image's natural size,
  capped at 16 Mpx) into the image's own rectangle. W / H, outlines, pixel rules, decisions, PUT bodies and exports
  are the same with it on or off (`test_review_r16a_browser.py` checks the bodies and the export).
- A page opened from a file (a zip copy) can't read the image's pixels: there the panel offers brightness and
  contrast only (a canvas filter), and says so.
- The lookup-table copy is rebuilt only when the settings, the image or Auto's levels change; a redraw of the same
  view (a save, a pull) reuses it.

## Sharper images (2x, round 16a stage 2)
`tools/add_hires.py` adds, per image, `images/<stem>.hi.jpg` (2880 px wide, grayscale, q90) and `"hi"`, `"hi_w"`,
`"hi_h"` in `data.js` (keys added only; the server's `images/` allow-list serves the file with the same key gate).
- **Drawing:** the page loads an image's 2x file lazily (never preloaded; one per view; dropped when the view moves on)
  and, once loaded, draws it into the base image's rectangle. The Editor loads it once the view shows more than 1
  screen pixel per image pixel; Quick check, object and tree cards load it for the card shown. A 2x file whose shape
  doesn't match the base image is not used.
- **Unchanged:** W / H and every pixel rule (simplify, minimum area, zoom limits, focus span, the cards' zoom steps)
  use the base image; outlines are normalized. Decisions, PUT bodies and exports are the same with the 2x image or
  without (`test_review_r16a_browser.py` `HiEditorTests`, `HiObjectCardTests`).
- **Contrast** is computed from the pixels drawn (the 2x image once it is drawn).
- **nm/px:** Image info's "nm/px here" is the level drawn (`nm_per_px x W / 2x width`, "sharper image"). Cards show
  no nm/px or scale readout (a blind set's cards never do).
- **Mixed and old sets:** images without `hi` (older `data.js`, or images add_hires skipped) draw the base image as
  before.
- **Caching:** `multi_review_server.py` serves `data.js` with `Cache-Control: no-store`, so keys added later reach the
  next page load (`review_server.py`, and the :8766 audit on it, are unchanged: `max-age=86400`).

## Cards (phone)
Narrow screens open the cards automatically. "Editor" goes back.
- **Header:** short image name (`<sample> · square<n> hole<m>`, else the file stem; full name on hover), "object n of m" for that image, "card p of N", and the order.
- **Order:** images by most flags first (`fg + fp + fx`, ties in package order), the same rank as the queue. Within an image: as before (triage: unmatched predictions by confidence, then unmatched labels; class review: `x`, then `lc` / Low conf EV). Everyone starts at the top. Claims keep two reviewers off the same card. Previously each reviewer started at a hash-based offset.
- **The object** has corner brackets and a thick outline with a dark halo. Other objects are drawn faint.
- **auto-next** (default **off**, remembered in localStorage `cryoev-review:autoNext`). It applies to class picks: the class buttons, the triage class chips and keys `1`–`8`.
  - Off: the card stays. The chosen class is highlighted, and a green "✓ class …" line shows the card's decision. **Next →** (or `N`, ↑, swipe up; in class review also swipe left) moves on, with no decision. A server pull doesn't move you off a card you just picked on.
  - On: a pick moves to the next card, as before.
  - Swipes, ✓ / ♥ / ✕ and "Save shape" always move on. After a pick on the card they **only** move on: they never overwrite the pick. On a prediction card, Accept's label and hover text then name the picked class ("Accept ✓ as MV"; round 16a #24).
- **Mouse (round 16a #25a):** on a computer, dragging the card with the mouse **pans** the (zoomed) image; it never decides. Decide with the buttons or keys. A plain click still steps the zoom, press-and-hold (without moving) still peeks without outlines, the wheel zooms. The pan resets with the card or the zoom step. Touch and pen keep swipe-to-decide.
- **Two reviewers:** when the server refuses a card's claim (someone else has it), the page moves on. Cards poll every 5 s for claims and decisions, and leave a card that is now someone else's or was decided elsewhere. The card you just picked on is never left this way. The screen re-renders only when something on it changed.

The mode menu has two modes:
- **flagged objects** (triage, as before):
  - Unmatched predictions: big green **✓ Accept as `<class>`** and red **✕ Reject** buttons replace ♥ / ✕. Accept (also `A`, `Enter`, →, swipe right) = add as its predicted class (EV when the package has no `pr[k].c`; before U4 always EV). Reject (also `X`, ←, swipe left) = not an object. The class chips offer the other classes; ✏️ + Save shape adds it as the predicted class with the new outline. September-seeded and already-picked cards keep the round buttons.
  - Unmatched labels: right = keep, left = delete.
  - Up = later. Class chips. ✏️ reshapes the outline.
- **class review** (no outline editing):
  - Shows one label with the model's class: the overlapping prediction with the best box IoU (≥ 0.3), so treat it as approximate.
  - Tap a big class button: the label is set to that class. With auto-next on, the next card opens.
  - Swipe right or ✓ = keep the class. Swipe left / up or ↑ = next, with no decision.
  - Scope "class flags" = labels with `x`, then `lc` or class Low conf EV. Scope "all labels" = every label.
  - Keys: `1`–`8`, `Enter`/`→`/`A` keep, `←`/`↑`/`N` next, `Z` undo.
- Triage keys: `1`–`8` class, `A`/`Enter`/`→` accept or keep, `X`/`←`/`Del` reject or delete, `↑`/`S`/`N` later, `E` reshape (in reshape: `R` redraw, `Enter` save, `Esc` cancel), `Z` rewind, `Space` zoom, `O`/`H` outlines, `Esc` Editor.

## `data.js` (from build_review_package.py)
`window.REVIEW_DATA = {id, source, run, created, categories: [...], images: [...]}`. Each image:
- Required: `{file, split, w, h, src, gt: [{id, c, p, u?}], pr: [{s, p, u?}], fg, fp}`.
- Optional, newer packages only: `pr[k].c` (predicted class name), `x: true` on gt/pr items, `lc: true` on gt items, image-level `fx`, `flc`.
- `p` is a list of rings. Each ring is a flat `[x1, y1, x2, y2, …]` normalized to [0, 1].

The tool runs on packages without the optional fields: those flags and counts are simply absent.

Optional top level: `cand_conf` (prediction packages): the starting value of the Editor's prediction slider and the cards' confidence filter. Without it they start at 0.45 and 0.35, as before.

## Edit record (`<package>/edits/<file>.json` = `{"file", "edit"}`)
```
edit = {dec: {key: {v, t, who, p?, server_t?}}, gt?, gtT?, reviewed, reviewedT, modified, t}
```
**Decisions** (`dec`): one per object.
- `t` is ms since the epoch on the reviewer's device (the client time, kept as written); `who` is the reviewer: a roster id from `people.json`, or a name typed under "Other" (older records: any free text).
- `server_t` (since TODO #53) is the server's receive time in ms, stamped when the server stores a decision the page sent. It orders the merge. Records saved before it have none.
- `pending: true` exists only in the page (localStorage, PUT bodies, exports): a decision made there that the server hasn't stamped yet. The server drops it when it stamps `server_t`, and never stores it.
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
| `i:reject` (the image) | `reject:<reason>` | Reject the whole image (round 7; reasons below). It drops out of results; no training effect |
| `i:note` (the image) | `<text>` (≤ 1000 characters) | A note on the image |
| `t:<objkey>` (`objkey` = the object's own key: `g<id>`, `p<k>`, `n<uid>`) | `lowconf` [+ `note`] | The **Low conf** tag (+ an optional note ≤ 300) |
| | `plain` + `note` | A note on the object, without the tag |

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

**Merge** (`review_server.merge_edits`, mirrored by `mergeEdit()` / `pickDec()` in the page):
- **Per decision key, the latest received wins** (TODO #53). Before, the newest client `t` won, so a device whose clock ran ahead silently beat everyone's later saves.
  - Order: `server_t`. **A decision without `server_t` (saved before it existed) counts its `t` instead.**
  - The page marks each new decision `pending`. A pending decision beats any other copy, and the server stamps it `server_t` = max(receive time, the replaced entry's order + 1). So a new decision always sorts above what it replaced, even an old entry whose `t` is in the future.
  - The same decision (equal apart from `server_t` / `pending` / `who`) keeps the stamped copy: the server's reply replaces the page's pending one. A re-sent copy is not stamped again.
  - Otherwise: the larger order wins; a tie keeps the saved copy.
- **Old records are unchanged.** Entries without `pending` are never stamped, so records saved before `server_t`, and PUTs from a page still open on the old version, merge byte-identical to the old rule (tests: `test_server_time_and_scope.py`).
- **Offline edits:** they are received when the device reconnects, so they win over decisions others made on those objects in the meantime (the old rule decided by device clock).
- `gt` / `gtT`: the newest `gtT` wins as a whole list. Per object, a decision with `server_t` (or `pending`) applies over its list item whatever the item's `t`. Every Editor change writes its object's decision, so a larger item `t` only means a clock ran ahead. Items with a decision without `server_t` keep the `t` rule below.
- `reviewed`: the newest `reviewedT` wins. `modified`: OR. `t`: max.
- **Clocks:** a new entry gets `t` = max(now, the entry it replaces + 1, its list item's `t` + 1, `gtT` + 1), as before.
- **Still on device clocks:** `gtT` (which list is the base) and `reviewedT` (the reviewed flag). A device far ahead can still win the reviewed toggle.

**Undo** (Editor `Z`, card ↺) reverts only entries this browser tab wrote since the step being undone.
- If someone else's decision arrived in between, on any key, it stays.
- A label with no decision before the step gets `null` when it was the package's version, or else an explicit `keep:<class>` (+ `p`) restating how it stood. A stale full list elsewhere can't bring the undone change back.
- The polygon list goes back to the step's copy too (with a strictly newer `gtT`), unless another reviewer's newer list has replaced this tab's since. Before U4 it never did, so `Z` after an Editor delete or accept changed the decision but left the label as it was.

**Validation** (`review_server.valid_edit`). A PUT gets a 400 and is never stored when it has:
- `NaN`/`Infinity`;
- an unknown decision key;
- a `new:` without `p`;
- a ring with < 3 points or a coordinate outside [0, 1] (± 1e-6);
- a `gt` item not shaped like a `data.js` label (`c` string, `p` rings, `id` int or null, `src` `p…`/`n…`, `t` number);
- an `i:` / `t:` decision that isn't exactly `i:reject`, `i:note` or `t:(g<digits>|p<digits>|n<[0-9a-z]+>)`, is a
  bare string, has a field other than `v`, `t`, `who`, `server_t`, `pending`, `note` (so no `p`), a key longer than 64,
  a reason not in the list, a note that isn't a string within its cap (1000 / 300), or a `v` / `note` that isn't
  valid UTF-8 (a lone surrogate such as `\ud800` would crash every later writer of the package).

The page clamps coordinates. If the server rejects one image, the page shows a banner and still syncs the others.

**Apply without the tool:** `review_server.effective_labels(image, edit)` returns the labels the page shows, as `[{id, c, p, src}]`.
- The tests check that it matches the page on the fixture, the September edits and R1's pre-lipo package.
- The planned `apply_review_edits.py` should use it, and add categories missing from the export; see Classes.

Known limit: older items without a decision key or `t` still follow the whole-list rule. That means September-era list items until their first edit, and keyless hand-drawn outlines until the Editor keys them. Two offline reviewers who both change such an item: the newer list wins.

## Reject, notes, low conf (round 7)
New image and object state are **decisions on keys of their own** (`i:reject`, `i:note`, `t:<objkey>`; table above), never
extra fields on a class decision. So they merge per key (the newest received wins), carry `who` and `server_t`, and
pass through older pages, exports and imports untouched. A class change on another device leaves an object's `t:` as it is.

- **Capability gate.** `GET /api/ping` answers `{"ok": true, "caps": ["i", "t", ...]}` (round 12 added `m`, `g`; round 14 `a`, object-set attribute decisions). The page shows the reject, note and
  Low conf controls only when the caps are there. A server started on older code answers `{"ok": true}` and would
  refuse the new keys with a 400, so the page never offers them; the same for a page opened from a file. The multi
  server reads `index.html` from disk on every request but validates with the code it started with: restart the
  review servers in the same step as the code moves.
- **Reject image** (Editor side panel "Image", or **⚑ Image** on a card): the reason chips (`thick ice`, `empty`,
  `contaminated`, `crystalline ice`, `broken carbon`, `aggregates`, `out of focus / drift`, `other`; the gallery's
  tags plus focus) and the image note in one dialog. Save writes only what was changed in the dialog, compared with
  the values when it opened: a reject or note that arrived from another device meanwhile is never overwritten. A rejected image gets a red **Rejected** badge in the image list,
  a ✕ status, the **Rejected** status filter and count, and the reason in the header. Quick check and class review
  skip its cards (they come back when it is un-rejected); the spot check leaves it out. Its outlines stay.
- **Image note** (✎ in the list, the text in the side panel). A note alone doesn't make the image "edited".
- **Low conf tag** on the selected counted object (labels, accepted candidates, drawn outlines): **◐ Low conf** (**L**)
  turns it on / off, with a note box under it. It is drawn as a solid amber double line plus a small "low conf" tag at
  the top right, unlike the dashed / dotted flag halos (`lc` long grey dashes, `x` orange dots). On an unaccepted candidate the button reads **◐ Accept as low
  conf**: `p<k>` `add:<cls>` and `t:p<k>` `lowconf` in one step, undone together (Editor **Z**, card ↺). On cards the
  ◐ button (**L**) does the same, and lists in the key legend. A keyless legacy outline gets its key first.
- **Low conf EV** stays a class (old data, the lab package) and reads as EV + low conf: its ◐ shows on, and turning it
  off writes `t:` null (or `plain` with a note) **plus** `keep:EV`. In the class picker it is labelled "(= EV + Low
  conf)"; picking it still writes the class, and turns a note-only (`plain`) tag on that object into `lowconf` (note
  kept) in the same step, so class and tag agree. Class review ranks tagged labels with the Low conf EV flags.
- **Text** (reasons, notes) is set with `textContent` / `title` only, never as HTML.
- **Undo:** `Z` / ↺ undo the tag like any object decision. The image's reject and note are never part of an object
  undo step: change them in their dialog (the **Not rejected** chip un-rejects). The Editor's help says so.

**Downstream**
- `reviewed.json` (`reviewed/1`, additive, each key **only when set**, so files without the new decisions are
  byte-identical to before; readers use `.get`): per image `rejected: true` + `reject_reason`, `note`; a rejected
  image has `n_candidates_open: 0`. Per object `low_conf: true` (the tag, or the Low conf EV class: such a label
  stops being low conf only when it is turned into EV) and `note`.
- `is_edited` ignores the `i:` keys (a note or a rejection alone leaves the image on the model path) and counts `t:`
  (else the tags would never reach results). The portal's "checked" count = edited, Reviewed or rejected.
- `effective_labels()` / `effGt()` pass `low_conf` / `note` through from `t:<key>`.
- `apply_review_edits.py`: the `i:` / `t:` keys are never counted as ops (a note "no" is not a removal). **The one COCO
  rule**, applied before the diff: EV + tag -> category Low conf EV (training unchanged; the change row is a reclass
  with `dec_key` `t:<key>` when the tag alone changed it); any other class keeps its class with
  `attributes.low_conf = true`; a tag note goes in `attributes.note`. Images are never dropped. `version.json` adds
  `rejected_images` (image, reason, reviewed, who_raw, who, note) + `n_rejected`; `low_conf`, the applied images'
  objects that are low conf (the tag, or the Low conf EV class: the same rule as `reviewed.json`) + `n_low_conf`; and
  `object_notes`, the other objects with a tag note. Their rows: image, key (the object's decision key), ann (its id
  in the output, null if not written), cls (its category there), tagged (`t:` is `lowconf`), note. A record with only
  `i:` keys is a record without changes; a `t:` tag counts as a change.

## Image info and sample groups (round 12)
Two caps more in `/api/ping` (`["i", "t", "m", "g"]`): **m** = `GET /api/meta`, **g** = `GET /api/groups`,
`POST /api/group` and the `grp` field on `i:reject`. A server on older code lacks them, so the page hides these
controls (the same gate as round 7). Both routes are in `review_server.make_handler`, so the single server and the
multi server (and the portal's `/review/<id>/` proxy) serve them with the same key check as every other route.

**Image info** (`annotation/review_meta.py`). Read at request time from folders that already exist, cached by each
file's mtime, never written: the package's `prediction_package.json`, the set's `images.csv` / `request.json`, the
tiles' `MANIFEST.csv`, and for a portal conversion its `convert.json` (recipe, `params_effective`: clip %, bandpass,
denoise; variant, GPU), `acquisition.json` (per montage: defocus, magnification, pixel size, dose, exposure, time) and
`montages/layout.json` (row / col). A set cut from bigger images (`source_manifest.csv`) gives the parent image and
x / y; a set made from the processing repo's own tile folder gets the session's acquisition summary from the
registry. Montage and row / col fall back to the tile name (`export_detector_tiles.training_name`:
`g<grid>_<label>__<montage>_r<row>c<col>_t<nn>__...`). Anything not found is left out and the page says **not
recorded**; a broken file never fails the request. Only names are sent, never local paths. Lab / compare packages
get the split only (plus what file names give).
- Editor side panel **Image info** (collapsible): sample, montage, tile, defocus, magnification, pixel size, dose,
  exposure, acquired, session, conversion, processing (recipe and its parameters), the session's defocus range, the
  file name; a set's note and provenance steps.
- Quick check: one line under the card position, e.g. `grid10_1 · row 0, col 1 (piece 5) · defocus -4.48 µm`.

**Sample switcher** (prediction packages). The list's first filter reads **All samples**, each sample with its image
count (and "rejected" when the sample is). Picking one shows only that sample and moves the Editor into it when the
open image is elsewhere; N / P then step through it. The side panel's **◀ Sample / Sample ▶** go to the previous / next
sample. Sort **Sample · montage · tile** orders by sample, montage (natural order), row, column, with a header per
sample and per montage (size, ✎ note, Rejected). With **All samples** and the other sorts, nothing changed.

**Groups** (`annotation/review_groups.py`; side panel **Sample**; prediction packages only, the route answers 404 for a
lab / compare package): **Comment on sample /
montage...**, **Reject sample / montage...** (a reason is required, an optional detail; the dialog says how many images it
rejects and how many already rejected stay as they are), **Undo sample / montage reject**.
- A group reject is the image reject on every image of the group: each gets `i:reject` `reject:<reason>` with `who`,
  `server_t` and `grp` = the group id (`["sample","<split>"]` or `["montage","<split>","<montage>"]`), merged like a PUT
  from that reviewer. Results, apply and the spot check treat them as any rejected image. Images already rejected are
  left alone (one rejected by hand keeps its own reason). Rejecting again with another reason updates the group's images.
- **Undo** touches only images whose `i:reject` still carries that group's id: they go back to not rejected, or to the
  other group reject that still covers them (the sample around a montage, or the montage inside a sample). Images
  rejected by hand, or changed by hand since, keep their state.
- A group comment is one note per group (text, who, time); empty removes it. Every action is logged.
- **Concurrency:** the action runs under the package write lock that every image save takes (one writer process per
  package), so it never interleaves with another reviewer's group action or save. Images open on another reviewer's
  screen are changed too; the answer lists them and the page says they see it on their next sync (<= 20 s). A later
  decision of theirs on such an image wins as usual (per-key merge).
- **Storage:** group notes and reject states in `edits/_groups.json` (`review_groups/1`). It is **not an image record**:
  every reader of `edits/*.json` skips it (`review_groups.record_files` / `is_record_file`: `/api/edits`,
  `apply_review_edits`, `resolve_review_edits`, the registry importers and validator, `strip_decoys`' count) or reads
  records by name / by their `file` key (the portal's progress and card counters, `reviewed.json`, the build's seed
  and part copies), so no count moves. It is backed up with the edits and is part of the apply snapshot's sha256.
- **Downstream:** `image_flags` / `reviewed.json` add `reject_scope` (`sample` | `montage`) to a group-rejected image;
  `reviewed.json` has `sample_groups` (notes and rejects by real sample label) when there are any, and
  `results.compute` copies it to that sample's `group` in `results.json` (not the CSVs). The portal pages don't show
  it yet.

## Shipping a partial review
1. Review images in the tool and press **R** on each one you have checked (also when you changed nothing).
2. Run `python annotation/apply_review_edits.py --base <pull>/current.coco.json --package <review package>`.
   - By default it applies only images marked reviewed (`--all-edits` applies everything, reviewed or not).
   - The run prints e.g. "reviewed only: 45 of 160 images; 12 edited images held back (not reviewed)".
3. The new `lv_<stamp>` lists `reviewed_images` and `held_back_images`, and `supersedes` the previous lv of the same
   package and base. The history page shows it as "verified 45/160".

Repeat at any time; each new lv supersedes the last. Unreviewed images keep their base labels until someone
reviews them. Rules: [LABEL_CHANGES.md](LABEL_CHANGES.md#apply-rule).

## Prediction packages (CryoEV portal)
A user's prediction set (`<CryoAI-laptop>/predict/<ps>/`) becomes a package `review/pred_<ps>/` for this same tool. The contract is in BUILD_CONTRACTS.md, section 3.
```bash
python -m annotation.prediction_package build --ps <CryoAI-laptop>/predict/<ps> [--review-root <dir>] [--owner <id>]
python -m annotation.multi_review_server [--root <CryoAI-laptop>/review] [--port 8767]
python -m annotation.prediction_package reviewed --pkg <review>/pred_<ps>     # -> <pkg>/reviewed.json
python -m annotation.prediction_package spotcheck --pkg <review>/pred_<ps> [--n 10] [--seed 0] [--mode representative|prioritised]
```
- **Labels** are the predictions at conf ≥ 0.50. A label nobody touches counts as present, so the reviewer only deletes, reclasses or reshapes. A stub COCO of them is kept in `labels.coco.json` (EV = 0, non-EV = 1).
- **Candidates** are the predictions at 0.25 ≤ conf < 0.50, shown as unmatched predictions (`u`, image flag `fp`) to accept or reject.
  - `data.js` sets `cand_conf: 0.25`, so the Editor's prediction slider and the cards' confidence filter start at 0.25 and every candidate shows. Packages without `cand_conf` (every other package) keep the old defaults: slider 0.45, cards 0.35.
  - Phone: cards in **flagged objects** mode. For MV / inner calls, use **class review** with scope **all labels**.
- `categories` lists all 8 tool classes, so none is marked "(new class)".
- `split` is the sample label, reduced to `[A-Za-z0-9 _.-]`; the real label stays in `prediction_package.json`. The page now also escapes file, split, categories and run wherever it builds HTML.
- `images.csv` rows must be image `<nnnn>`, file `images/<nnnn>.png|jpg|jpeg`.
- The other files:
  - `prediction_package.json`: per image, its size, sample label, and the label ids and candidate confidences.
  - `review_package.json`: the registry line (`kind: "prediction"`, `owner`, `prediction_set`, `base_kind: "prediction_set"`). It is written last.
  - `.review_key`: kept when the package is rebuilt.
- The package is built in `.building_*` and renamed into place when complete. A package that already has saved edits is never rebuilt, because the edits point at its label ids.
- **reviewed.json** (`reviewed/1` + the section 3 amendment) is built with `review_server.effective_labels`.
  - Per image, `reviewed` is the Reviewed flag. `edited` is true when the image has any saved decision or an edited label list.
  - **An edited or reviewed image reports its labels as the tool shows them.** The phone cards never press Reviewed, so their decisions count through `edited`. Any other image reports the model's predictions at ≥ 0.50.
  - `reviewer_label`: the class the object stands at (the reviewer's pick, or the model's class for an object nobody changed). It is null on images that report the model's predictions.
  - `reviewer_class`: EV, MV and inner map to themselves; Low conf EV, Incomplete and Overlapping map to EV; VLDL and non-EV map to nonEV. Any other label gives null and is counted in the top-level `unknown_labels`.
  - `cls`: 0 for EV / MV / inner, 1 for nonEV. With `reviewer_class` null, it is the model object's class (0 for a drawn outline).
  - `source`: `"model"` for an outline as the model drew it, `"human"` for one drawn or reshaped by a reviewer.
  - `conf`: the model's value, kept on a reshaped outline; `null` on a drawn one.
  - `reviewed_by`: the **authors of decisions**, i.e. the `who` of the image's decisions. The tool doesn't record who pressed Reviewed, so an image marked Reviewed with no decisions has `[]`.
- **Spot check:** a seeded subset written to `<pkg>/spotcheck.json` with its `mode` and `rule`.
  - `representative` (default): a uniform draw within each sample, with n split in proportion to sample size and at least 1 per sample. When n is smaller than the number of samples, n samples are chosen at random.
  - `prioritised`: round-robin over the samples, each pick weighted by 1 + the image's number of candidates.
- **Multi-package server** (port 8767) serves `/<pkg id>/…`, each package through `review_server.make_handler`. `review_server.py` is wrapped, not changed.
  - It serves only complete `pred_*` folders (`data.js` + `review_package.json`), so it never writes the edits of a package that the :8766 server has open. A folder is served only under its exact name, so a case or 8.3 alias can't become a second package.
  - Each package has its own key (`.review_key`; a package without one is not served) and its own cookie, `review_key_<port>_<pkg id>`.
  - The key is checked before anything is made for the package: no handler, cache entry or `edits/` folder for a request without it.
  - **An unknown id and a wrong key both get the same 403,** so ids can't be enumerated. A malformed id, or a path with `..`, `\` or `:`, gets 400. `/` gets 404. `/<id>` with its key redirects to `/<id>/`.
  - A package built after start is found on its first request. A rebuilt `data.js` or a new key is picked up on the next one.

## Object sets (rounds 14-15)
An object set (`review/pred_obj_<slug>_<4hex>/`, `data.js` `D.kind = "objects"`) asks a label schema's questions about
chosen objects instead of reviewing whole images. Built with `python -m annotation.object_set build` (or the portal's
Start a review -> Object set); the package layout and the blind rule are in `annotation/object_set.py`'s docstring.

- **Caps:** `/api/ping` caps are `["i", "t", "m", "g", "a"]`. `a` = the server takes attribute decisions; without it
  (an older server, a page opened from a file) the card shows the questions but says answers save only on the review
  server.
- **Attribute decisions** are keys of their own on the item's image record: `a:<obj>:<attr>@<rid>` (`i:a:<attr>@<rid>`
  for an image-level question), value `{v, t, who, server_t}`. The server checks each against `schema.json` (kind,
  options, text length), takes them only from a signed-in reviewer of the set (`X-CJ-User` from the portal; `@<rid>`
  is overwritten with that identity), at most 64 new ones per PUT (the page sends bigger batches in chunks), and
  answers each reviewer with **only their own** answers (rater independence). Undo = `v: null`.
- **Help lines:** a schema question may carry `help` (plain text, at most 200 characters); the card shows it under the
  question, as text.
- **Instructions:** the maker's plain-text set instructions (`object_set instructions <pkg> --text ... --do`, or at
  build): an "Instructions" button in the top bar (`?` toggles, Esc closes), opened by itself once per browser and
  text. `data.js` `D.instructions`; absent = no button.
- **Whole-tree cards (round 15):** `D.mode = "tree"` (built with `object_set build --tree` from a prediction set whose
  Results ran the `nesting_tree` step). One card per nested EV: every outline of the tree is numbered on the image (1 =
  the outer vesicle, the rest in a seeded order), with one block per node holding the questions that apply to it
  (`applies_to`: `root` = node 1, `child` = the others, `all`). For `ls_lamellarity_tree_v1`: node 1 "Type"
  (Single / ML / MV / ML+MV / Unsure, keys 1-4, 0); every other node "What is it?" (Layer of parent `l`, Inner vesicle
  `v`, Not a membrane `x`, Duplicate outline `d`), an optional "If inner vesicle: its type", and a note (`n`). Tab =
  next node; keys answer the current node; a tap on a block makes it current. An item is done when every assigned
  reviewer answered every required question of every node. No role, unit, call or measurement is in `data.js`; the
  export derives each reviewer's tree (`derived_parent_n`, `derived_membranes`, `derived_inner`).
- **Close set:** Reviews hub -> Close (the set's maker or an admin, typing the set's name), or
  `python -m annotation.object_set close <pkg> --by <id> --do --registry <dir>`. One writer for both
  (`object_set.close_set`): `review_package.json` `closed: true`, `closed_by`, `closed_t` under the meta lock, `data.js`
  `D.closed = true` (only that key), then the registry status line `reviewed` (`status_by` = the closer). It is
  final. The server then answers every PUT carrying a new attribute decision with 409 `{"error": "set closed"}`
  (re-checked under the write lock); other keys (notes, reject) still save. The card says the set is closed and sends
  nothing: at load from `D.closed`, or on that 409. **Any other 409** (the package open in another review server) is
  not a close: the answer is kept and retried, as before. Exports leave out any answer stamped after `closed_t`.
- **Blind columns:** a whole-tree set's hidden `nesting_tree` columns are in its export only after Close. A one-object
  set's blind columns are in its export (to the maker) before Close too: unchanged by rule zero (the live set).

## eVHarmony (round 17, `D.app = "evharmony"`)
A joke dating-app skin over a one-question object set, so lab members can label vesicle candidates. It is on only when
the set's `data.js` says `D.app = "evharmony"` on a one-object set (never from a URL parameter); every other set is
unchanged.

- **Build** (the prediction set is a parameter; `ps_20261113_c547fe` was only the first test set):
  ```bash
  python tools/evh_items.py --ps <CryoAI-laptop>/predict/<ps> --out evh_items.csv [--lo 0.25 --hi 0.5 --gold 40 --seed 17]
  python -m annotation.object_set build --ps <CryoAI-laptop>/predict/<ps> --items evh_items.csv       --schema tools/evh/ls_evh_match_v1.json --line-col line --slug evh --title "eVHarmony beta"       --app evharmony --first-items 40 [--review-root <dir>] --do
  ```
  `evh_items.py` writes every candidate (`lo <= conf < hi`, any class) with columns image, line, conf, gold; the first
  `--gold` rows are a confidence-stratified pick (one seeded pick per equal-count conf bin). `--first-items N` keeps
  the first N CSV rows first in the order for everyone (shuffled among themselves), then the rest (shuffled):
  `items.json` `first_items`. They become **gold** once the set's owner answers them.
- **Schema** `ls_evh_match_v1` (a file, `tools/evh/ls_evh_match_v1.json`, not a built-in, so the registry's built-in
  list is unchanged): `match` (required: yes = real vesicle, no = not a vesicle, unsure = "It's complicated"; keys
  y / n / u) and `consent` (yes / no, optional).
- **Consent** is the `consent` answer on the set's **first item** (`a:<items[0].obj>:consent@<rid>`): an ordinary
  attribute decision, so it is credited, only the reviewer sees their own, and the export carries it as
  `consent@<rid>` (on that item's row). No server change. Absent or no = not counted.
- **Page:** a splash once per browser ("Create your profile" = a nickname kept in this browser, display only; the
  identity stays the signed-in / chosen reviewer); the brand header (`BRANDS`, tap the logo to cycle); a bio quip on
  each card; a size line (`~87 nm`, the outline's area-equivalent diameter from the base image's nm/px in
  `/api/meta`) only when nm/px is known and the set has no blind columns. Swipe right / left / up (touch, pen), the
  three buttons, or → y / ← n / ↑ u answer `match` through the usual attribute save and the next open card comes; a
  mouse drag pans; Backspace undoes. Hidden: the item list, item id, title and reviewer chips (Instructions and ☼
  stay). ☼ starts on Auto + Smooth 2 when the viewer has no saved contrast setting (display only).
- **Reveal:** after the reviewer's 10th answer: "Plot twist: you just reviewed 10 vesicle candidates for the CryoEV
  model. A few cards had known answers. Count my answers? [Yes] [No]", then "Keep swiping". Esc = later: offered again
  after the next answer until answered.
- **Scoring:** `python -m annotation.object_set export --pkg <set> --out evh.csv`, then
  `python tools/evh_score.py --csv evh.csv --gold <owner id> [--out scores.csv]`: per reviewer the gold items
  answered, unsure (excluded, reported), agreement %, consent, and pass (>= 80 % on >= 20 scored gold). Answers
  without consent = yes are never used.
- **Not in v1:** a leaderboard (the server returns each reviewer only their own answers).
- **Save status (r17b):** eVHarmony hides the sync chip, so a failing save (any non-OK PUT, e.g. 403 "you are not a
  reviewer of this set", or offline) shows a fixed red banner `#oSaveErr`: the server's `{"error"}` text and how many
  answers wait. Answers stay dirty and in localStorage and retry every 15 s (and on reload / `online`); the banner goes
  once a sync pass uploads everything. On every object set the foot line says "Saved ✓" only when nothing is pending
  (any image), "Not saved · N answers waiting" while saves fail; the plain card's sync chip says "not saved" with the
  server's text as its tooltip.
- **v2 (r17c, schema `tools/evh/ls_evh_match_v2.json`):** `match` = yes (Match →) / no (Nope ←) / dense ("Small &
  dense" ↓, keys d / ↓; possibly lipoproteins, kept as its own answer) / unsure (It's complicated ↑): v1's value names
  are kept, so v1 answers carry over. Overlapping vesicles count as Match. Optional `kind` (single / ml / mv / inner),
  asked only after a Match: the card stays and the "What is it?" chips replace the buttons (keys 1-4; 0 / Enter / Esc
  = skip); a pick saves `a:<obj>:kind@<rid>` and the next card comes; skip = unanswered; a swipe or answer key while
  the chips are up re-answers the card (Match again = skip) and moves on; Backspace undoes the Match. The page turns
  these on from the schema itself (a `dense` option, a `kind` question): a v1 set keeps its three buttons, and ↓ / d do
  nothing there.
- **How to judge (r17c):** on a v2 set or a set with examples, a sheet opened from the splash ("How to judge"), the
  ? button in the header or the ? key (Esc closes): one section per answer with its example crops and captions
  (Match: Single / Multilayer / Has vesicles inside / Inside another), then the set's instructions as "Guidelines".
  It replaces the Instructions button and its auto-open there (v1 sets: unchanged). Examples: `data.js`
  `D.examples = [{src, cat, caption}]` (`src` must be `images/ex_<id>.jpg`: review_server serves only `index.html`,
  `data.js` and `images/`), written with
  `python tools/evh_examples.py <pkg> --from <crops dir> --pick S2,M5,... [--cat ID=cat] [--caption ID=text] --do`
  (also `<pkg>/examples.json` [{file, cat, caption}]; only the `examples` key of data.js changes; `--clear`). The crops
  dir has `examples.csv` (id, crop file, category, provenance). Use crops that are not items of the set.
- **Balanced gold (r17c):** an always-yes player scored 89 % on the first set's gold (25 yes / 3 no), so v2 sets add
  known Nopes:
  ```bash
  python tools/evh_items.py --ps <ps> --out evh_items.csv --gold 0 --decoys 20 --derive <new ps folder> \
      --low 0.10 --answered <old pred_obj_evh_*> --who <owner id>
  python -m annotation.object_set build --ps <new ps folder> --items evh_items.csv --schema tools/evh/ls_evh_match_v2.json \
      --line-col line --slug evh --title "eVHarmony beta" --app evharmony --first-items <printed N> \
      --owner <id> --reviewers a,b --instructions-file guidelines.txt [--review-root <dir>] --do
  python tools/evh_carry.py --from <old pred_obj_evh_*> --to <new set> --who <owner id> --do
  python -m annotation.object_set export --pkg <new set> --out evh.csv
  python tools/evh_score.py --csv evh.csv --gold <owner id> --decoys evh_items_decoys.csv --control
  ```
  - **Decoys:** a derived copy of the prediction set (new ps id; images, images.csv byte-copied; every model line
    byte-identical and in place) with N polygon lines appended: a real candidate's polygon (vertex list, size, pixel
    staircase), flipped / transposed at random, moved by whole pixels onto a free place of one of the set's images
    (touching no prediction of any conf and no other decoy, 3 px margin, inside the image), conf and class drawn from
    the candidates. The list lives only in `<items>_decoys.csv` next to the items CSV (and `evh_derived.json` in the
    derived folder, which nothing serves): no items CSV column, nothing in data.js, items.json or `/api/meta` says
    decoy (tested). Leftover tell: object keys g<n> follow line order, so a decoy is the last shown object of its
    image (the page shows no conf or line).
  - **--low L:** up to `--n-low` candidates with L <= conf < `--lo` (a set predicted at conf 0.25 has none: reported).
  - **First block:** the owner's previously answered objects (`--answered --who`) + decoys + low, gold=1, then the
    rest; build with `--first-items` = the printed count.
  - **Scoring:** decoys are known "no" without an owner answer (an owner "yes" on one is reported as a conflict);
    `dense` and `unsure` are excluded from agreement and reported (`n_dense`, `n_unsure`); per reviewer `n_decoys`,
    `decoy_no` and the `kind_*` counts (on items they called yes). `--control` scores made-up always-yes / always-no /
    replay reviewers on the same gold: exit 3 unless both lazy ones fail and the replay passes. On the c547fe rebuild
    (28 owner yes/no + 20 decoys): always-yes 52.1 % fail, always-no 47.9 % fail, replay 100 % pass; without decoys
    always-yes passed at 89.3 %.
  - **Carry-over** (`tools/evh_carry.py`, dry run unless --do): the reviewer's `match` (and `consent`) decisions move
    to the object with the same (image, line) in the new set, every field kept ({v, t, who, server_t}); consent goes to
    the new set's first item. Only into keys the new set doesn't have; values its schema refuses are skipped. Writes
    the review_server record format and `_attr_log.jsonl` rows; provenance in `<new>/carried_answers.json` (decisions
    have no field for it). Run it before the set is served.
- **View only (r17b):** through the portal, `multi_review_server` adds `answer: true|false` to
  `window.REVIEW_IDENTITY` for a package with a schema (`may_answer`: review_server's PUT rule, reviewers in
  `review_package.json` when non-empty, else the people roster; read per page load). `answer: false` → the
  `#oViewOnly` banner, `can()` is false (keys, buttons and swipes save nothing; in eVHarmony right / left still move
  to the next / previous card). No flag (an older server, a file, no schema) → as before.

## Tests
```bash
python -m unittest discover -s annotation/tests -v
```
- `test_prediction_package.py` runs on the golden fixtures (`tests/fixtures/cj_golden`, read only, checked unchanged).
  - Labels vs candidates; the `images.csv` checks; the temp-folder build.
  - Control: with no edits, the objects are the prediction files at ≥ 0.50.
  - The golden review of 0002, through the multi server's HTTP API, equals `reviewed.json`.
  - Card decisions with no Reviewed press give `edited: true, reviewed: false`; the `reviewer_class` map.
  - The multi server: per-package keys and cookies, one 403 for an unknown id and a wrong key, aliases, nothing made without the key, incomplete packages, 400, a late package, and the CLI.
  - Both spot-check modes.
- `test_prediction_package_browser.py` (Playwright):
  - `cand_conf` sets the slider and the cards' filter, and the 0.30 candidate shows in the Editor and the cards.
  - Without it, the controls are exactly as before.
  - Markup in file, split, category or run shows as text.
  - Against the previous page, the `cand_conf` and escaping tests fail and the defaults test passes.
- `test_review_server.py`, standard library only: a live server on a free 127.0.0.1 port.
  - API: ping, PUT, a merge by timestamp, the new / redraw decisions, claims (and their cap), access-key rejection.
  - Sandbox and cookies: the flag in `data.js` for `--sandbox` and a `SANDBOX_` package (a real package's `data.js` byte-identical); two servers with different keys and one cookie jar both stay signed in; the legacy cookie accepted only with this server's key. The browser tests check the bar at 400 and 1280 px (`SandboxBarTests`).
  - Hardening: rejected bad edits and bad values, never stored; bad Content-Length; idle-socket timeout; key redacted from logs; default host.
  - Records: an offline reshape survives a newer list; the real September edits merge and apply, read-only, and change some labels.
- `test_server_time_and_scope.py` (TODO #53 and the sandbox key):
  - a later save beats a decision from a device a day ahead: through the API, with an Editor list, and in two browsers;
  - an old entry with a future `t` is replaced for good;
  - old-format records (fixtures) merge byte-identical to a frozen copy of the old `merge_edits`, by pair, chain and re-sent copy, and through a live server, including an old page's PUTs;
  - `mergeEdit()` equals `merge_edits()`;
  - `_request_identity` decides every saved and claimed `who`;
  - sandbox drafts on one origin are never uploaded by the real review, and the real key is unchanged.
- `test_review_tool_browser.py`, Playwright with headless Edge or Chrome (skipped without one):
  - script syntax (no node on the lab PC, so the browser parses it);
  - class order, and the new-class mark (fixture, and R1's pre-lipo package);
  - queue sort (`flc` excluded from the rank) and filters;
  - reclass, accept, freehand, tap-vertex, redraw and vertex drag;
  - U4: `A`/`X` and the big buttons in the Editor and on prediction cards; a vertex drag / delete / add on an unaccepted prediction writes `add:<cls>` + `p`, with the message's Undo and `Z`; `Z` after `Del`; records saved before U4 (bare strings, a list without per-item `t`, `add:`+`p`, `no`, `null`, `new:`) load byte-identical and apply as `effective_labels()`;
  - two reviewers: offline redraw vs a later Editor save; undo vs someone's newer decision; selection and Redraw through a pull;
  - class review at 400 px with touch;
  - QC fixes (`QcFixTests`): a swipe / ♥ / ✕ / Enter after a pick only moves on (M1); markup in who / seed / class / claims renders as text (M3); Other limited to the server's characters; a retry dropped when the card changed; no claims without a name; blank-who offline entries named before upload; a deep link stripped after it opens;
  - two phones (`TwoReviewerTests`): never on the same card, and a card decided by one is left by the other (M2);
  - September cards (`SeptCardTests`): seeded keep / delete / add / new cards in class review and triage, confirm and pick write fresh decisions, counts;
  - phone review (`PhoneReviewTests`, 400 px touch and 1280 px desktop): card header, flag order and object outline (U1); auto-next off by default, stays on a picked card through a pull, remembered when on (U2); deep links opened directly at both sizes, for a label, a prediction and a missing object, in-place open + Back, and 403 without the key (U3); name picker from `/api/people`, required before card and Editor edits, roster id saved and claimed, Other, alias mapping, typed fallback when `/api/people` fails (U5); Sept tags and their removal on a new decision (U7);
  - page labels == `effective_labels()`, on the fixture and on both real packages. Re-saving the September records changes no labels.
- Real packages are copied to a temp folder and never written.
  - September v4, searched in order: `$CRYOEV_REVIEW_PACKAGE`, then `CryoAI/annotation review/cryoai_v4_review` beside the repo, then the Box copy.
  - Pre-lipo, searched in order: `$CRYOEV_PRELIPO_PACKAGE`, then `CryoAI-laptop/review/prelipo_v5F5_review`.
- `test_review_r16a_browser.py` (round 16a): contrast on vs off gives the same PUT bodies and export (control: the
  contrast was really applied); Auto brightens a faint image and follows a zoomed-in region, Reset restores it
  exactly, settings per reviewer and remembered; a tainted canvas falls back without errors; object and tree cards
  draw with contrast and the tree card keeps Tab; #24 Accept names the picked class (fun on and off); #25a a mouse
  drag pans (no decision, zoom or peek), touch and pen still swipe; #25b Editor Tab / Shift+Tab, not on a focused
  control. Stage 2: PUT bodies the same with the 2x image drawn or not (Editor: accept, vertex drag, reclass,
  freehand; object cards: answers), with a control that it was drawn; a mixed set and old images; nm/px of the level
  drawn; no nm/px on blind cards; Smooth on (in the contrast PUT / export test), Reset, hidden when unsupported; no
  table rebuild on redraws. `make_fixture.build_hi` and `object_card_fixture.build(hi=...)` make the 2x fixtures.
- `test_review_reject_lowconf.py` (round 7): `valid_edit` for the new keys, `/api/ping` caps, the HEAD (00a1921)
  `valid_edit` refusing them; two devices merged per key; a class change keeping the tag; `effective_labels`,
  `is_edited`, `is_rejected`, `_is_checked`, the spot check; `reviewed.json` fields only when set; `apply_review_edits`
  (a note "no" is no removal, the COCO rule, `version.json`, a tag-only record).
- `test_review_reject_lowconf_browser.py` (Playwright): the page against the HEAD server hides every new control and
  sends no new key; reject + note, badge, filter, counts, Quick check skipping; XSS strings inert; Accept as low conf
  and its one-step undo (Editor and cards); the toggle and note, the Low conf EV alias; export -> import round trip;
  screenshots at 1280 / 390 / 360 px, light and dark.
- `test_review_meta_groups.py` (round 12): tile names, `package_meta` (full conversion, missing / broken files,
  upload set, registry session, lab package, no paths), the routes and their key check, group reject / undo /
  comment (hand rejections survive undo, nested sample / montage, reason change, claims, portal identity, bad
  bodies), the lock under concurrent reviewers, `_groups.json` ignored by every edits reader, results.
- `test_review_meta_groups_browser.py` (Playwright): the info panel's defocus and "not recorded", the sample switcher
  and sort, reject / undo of a sample with a hand-rejected image, the Quick check line, phone width, an old server.
- `REVIEW_TEST_SHOTS=<dir>` keeps phone screenshots.
