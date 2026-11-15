# CryoEV changelog

Newest first. One entry per version: what lab users will notice, and which parts changed. The portal shows this page
at /changes ("What's new" next to the version stamp at the foot of every page).

How to add an entry: put it at the top as `## YYYY-MM-DD HH:MM · Short title`, then 2-8 plain bullets, then a
**Components:** line (portal / review page / registry / CLI). When the review page changes, also bump `PAGE_VERSION`
near the top of annotation/review_tool/index.html.

## 2026-11-15 10:40 · Round 15: whole-tree cards, help lines, Close set

- **Whole-tree cards:** an object set can show one card per vesicle tree (the outer EV and every outline inside it, numbered on the image). For each inner outline you answer "layer of the outer EV", "inner vesicle" (and its type), "not a membrane" or "duplicate". Tab moves between outlines on a laptop. Start one from Reviews → Start a review → Object set → "Whole tree per card".
- **Counting rule on the cards:** count the outer EV's own level only: concentric membranes = the outer membrane plus its own layers; each inner vesicle counts as 1; anything deeper goes in the note.
- **Help lines:** each question can carry a one-line definition under it (new schemas `ls_lamellarity_v2` and `ls_lamellarity_tree_v1`).
- **Lamellarity tree step on Results:** a new analysis step (`nesting_tree` v1) splits nested outlines into layers vs inner vesicles, measured at 1440 px; it reproduces the poster numbers exactly (134/134 items, 399/399 outlines). Adding it adds the nesting step automatically.
- **Close set:** the set's maker or an admin can close an object set (final; type the set name to confirm). Answers are switched off and the blind banner goes away; exports open to everyone with access.
- **Fixes:** large sets save in chunks; step names on Results no longer break mid-word; the CLI dataset copy checks free disk space first.
- **Components:** portal / review page 2026-11-14.4 / registry (closed object sets, tree sets) / CLI (`annotation.object_set build --tree`, `close`).

## 2026-11-14 15:45 · Set instructions on object cards

- **Set instructions:** an object set can carry its maker's instructions (how to count, what to note). Every card shows an "ⓘ Instructions" button in the top bar that opens them in a small panel (phone and laptop; `?` opens and closes it, Esc closes). The panel opens by itself the first time you open the set in a browser, and again if the instructions change.
- **Setting them:** `python -m annotation.object_set instructions <set folder> --text "..."` (or `--file`, `--clear`), a dry run until `--do`; `build` takes `--instructions` / `--instructions-file`. Only the instructions change in the set; sets without them look exactly as before.
- **Components:** review page 2026-11-14.2 / CLI (`annotation.object_set instructions`, `build --instructions`).

## 2026-11-14 15:30 · Round 14: object sets with your own label questions, class maps, analysis steps

- **Object sets:** review a chosen list of objects (for example nested EVs) as phone-friendly cards that ask a label schema's questions: type, membrane count, inner vesicles, "outlines wrong", note. Every tap saves. Each reviewer's answers are kept separately and nobody sees anyone else's. An admin starts one from Reviews → Start a review → Object set.
- **Blind sets:** measured columns (sizes, gaps) never reach the reviewer's browser; the export joins them back. Exports are for admins and the set's maker until the set is closed.
- **Label schemas:** versioned lists of classes and questions (Data → schema pages, read only). Today's fixed classes are "legacy v1"; old MV shows as "MV · stack outline", the new scheme's as "MV multivesicular".
- **Collapse classes:** Train → Advanced makes a dataset with fewer classes from a class map (e.g. Low conf EV → EV), shows the counts first, and offers an identical existing dataset instead of a copy.
- **Analysis steps on Results:** Results lists the steps it ran (roles v1 by default, the same numbers as before) and offers objects.csv, one row per outline with its layer count; admins can add the nesting step (membrane gaps, parent and depth per outline).
- **Components:** portal / review page 2026-11-14.1 / registry (label_schema kind, object sets, adopted prediction sets) / CLI (`cryoev schema register`, `jobs.predict_set adopt`, `annotation.object_set`).

## 2026-11-13 22:45 · Round 13: one Reviews page, start and assign reviews, list filters

- **Reviews** (one menu entry, replaces Label review / My reviews / Sandbox): "Needs your review" first (most recent first), then all reviews by kind (predictions, label sets, audits, practice). Finished and applied reviews are hidden until you change the status filter. Continue opens Quick check if cards are open, else the Editor.
- **Switch packages without leaving the tool:** a bar above the review tool with "← All reviews", the current package and a list of your other reviews. (Not on the running audit's page until it ends.)
- **Assign reviewers:** the owner or an admin adds lab members to a review; they see it under "Needs your review".
- **Start a review:** from a finished prediction set (everyone, own sets), or, for admins, revise a label set or build a blind audit from the web, with a progress page. Revised label sets can then be Applied.
- **Results and My analyses:** filter by person (agent-made on/off), date, model, status and search; sort; group by person, imaging session, source images, model or sample.
- Blind audits show counts only to everyone but admins; an audit's owner can't decide its cards.

**Components:** portal, review servers (settings), compare_package, registry (schema, importer)

## 2026-11-13 15:00 · Review page 2026-11-13.1: image info, sample switcher, sample / montage reject

- **Image info** in the Editor's side panel: sample, montage, tile, defocus, magnification, pixel size, dose, when it was taken, and how it was processed (recipe, contrast clip, bandpass, denoise). Anything the lab's folders don't have says "not recorded". Quick check shows montage, tile and defocus under the card.
- **Sample switcher:** "All samples" or one sample, each with its image count; ◀ / ▶ Sample in the side panel. New sort "Sample · montage · tile" with a header per sample and montage.
- **Comment on or reject a whole sample or montage** (side panel "Sample"): a reason is required and the dialog says how many images it rejects. Images rejected one by one keep their own reason; "Undo sample reject" brings back only the images the group reject took out.
- Results leave out group-rejected images like any rejected image; the group's note and reason are kept with the sample in the results file.

**Components:** review page, review servers (restart to enable), results

## 2026-11-13 09:30 · Round 11b: notes and provenance on prediction sets

- **Prediction sets can say how their images were made:** `predict new` takes `--note` (free text) and `--provenance` (a small JSON file: what the images derive from, the processing steps, tiling, script and git sha). The tile manifest is kept with the set as `source_manifest.csv`.
- **Data pages** show "Note", "How the images were made" and "Derived from" (linked to the dataset, label set or model) on such sets. Sets without them look exactly as before.
- **Sets an agent makes from the laptop CLI** (`python -m cryoev predict new` in a Claude session) now show "Made by: claude for <you>", as API-made sets do, on every later status too.

**Components:** CLI, portal, registry

## 2026-11-13 08:45 · Round 11: agent API writes, the Apply button, faster and tidier code

- **Agent API, read + write:** when you make a token on **My account**, choose **Read only** or **Read + write** (write tokens last at most 30 days). A write token lets your agent upload images, start predictions, compute results and send feedback, and nothing else: no admin, review, Apply or training. Limits: 60 writes an hour and one open agent upload per person, and agent-started prediction sets count against the active-set cap whatever your role. Everything it makes is stamped "<agent> for <you>" [AI], including on the prediction set ("Made by" on Data). An agent's upload no longer cancels your own browser upload.
- **Apply button (admins):** after "Preview apply", type the package name to confirm. The portal makes the new label set from exactly the previewed decisions, records only that label set in the registry, and updates the history page. It refuses if the decisions or the package changed since the preview, if the package was already applied, or if another label set already builds on the same base. A failed registry step shows a Retry button.
- **Behind the scenes:** the registry code is now a normal Python package (no more loading by file path), the Data module is split into 7 smaller files (pages byte-identical), there is one audit helper, the test helpers are shared, and everyday test runs skip browser tests when no UI file changed (about 9.7 → 3.5 min for registry or jobs changes).

**Components:** portal, registry, jobs, CLI, tests

## 2026-11-13 07:00 · Round 10: agent API (read-only), apply preview, eval jobs, sturdier tests

- **Agent API (read-only, tailnet for now):** on **My account** you can make a token for your own AI agent (shown once, up to 90 days, revocable). The agent can read data, lineage, your predictions and results status, and job summaries, with your own permissions, capped at advanced. Its visits show in Recent actions as "<agent> for <you>" [AI]. Admins see and revoke everyone's tokens in People. See docs/API.md.
- **Label sets:** a replaced label set now links to its replacement ("Replaced by").
- **Apply preview (admins):** for review packages built on a label set, "Preview apply" shows the change report from a scratch run. Nothing is written to the labels; the real apply stays a command for now.
- **Desktop eval jobs (code only):** a new job type runs the test-set metrics (model.val, overlap_mask off) on the desktop and writes a metrics file. It is off until switched on at the desktop.
- **Behind the scenes:** the production model is no longer re-promoted by imports after you change it; tests run one at a time per machine and no longer flake on timing.

**Components:** portal, registry, jobs, CLI

## 2026-11-12 19:43 · Review page 2026-11-12.2 (blind-audit package)

- Editor: the same point eraser as Quick check. Select points (click; shift-click to add; Shift-drag a box), then "Delete N points" or the Delete key. Outlines keep at least 3 points; Esc clears the selection, Z undoes.
- The review page shows its own version in its header.

**Components:** review page

## 2026-11-12 19:30 · Round 9 platform (preview, not live yet)

- **Jobs** is a left-to-right flow board (raw uploads, converted, in review, datasets, training); every card opens its work, and a strip on top shows what each training machine is doing.
- **Data** pages list the label classes, trace a label set back through its lineage to the imaging sessions it came from, and show lanes (automatic, human, revised) with arrows between them.
- **Admin > Recent actions** logs every portal action, sign-ins, and actions run from the command line.
- A new **About** page (link at the foot of every page) and this **What's new** page.
- An agent command line (`python -m cryoev`): actions an assistant runs for you show as "claude for <your name>".
- Sign in with Google is offered as an option.
- The version stamp at the foot of each page now also shows the review page's own version on review pages.

**Components:** portal, registry, CLI

## 2026-11-12 19:08 · Review page update (blind-audit package only)

- Quick check: a fast double tap no longer decides the next card before you see it, and Back returns to exactly the last card.
- Quick check: a list of the cards (side column; a drawer on phones) to jump back to any card.
- Pinch to zoom on phones and tablets.
- Quick check: select several outline points (drag a box, or shift-click) and delete them together.
- A Fun-mode switch (settings): off gives plain wording, no pop-ups. Saved decisions are identical either way.
- Only the blind-audit review package got this page; the portal itself did not change.

**Components:** review page

## 2026-11-11 18:00 · Review tool fixes (rounds 8b and 8c)

- The Back button returns to the image you came from.
- Zoom goes closer.
- A clear "Saved" state, and **Show original** to compare with the labels before your edits.
- Resetting an outline restores the right shape.
- Low conf EV labels are handled correctly.
- Fill (tinting the selected object) is off by default; the outline alone keeps the object visible.

**Components:** review page
