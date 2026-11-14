# CryoEV changelog

Newest first. One entry per version: what lab users will notice, and which parts changed. The portal shows this page
at /changes ("What's new" next to the version stamp at the foot of every page).

How to add an entry: put it at the top as `## YYYY-MM-DD HH:MM · Short title`, then 2-8 plain bullets, then a
**Components:** line (portal / review page / registry / CLI). When the review page changes, also bump `PAGE_VERSION`
near the top of annotation/review_tool/index.html.

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
