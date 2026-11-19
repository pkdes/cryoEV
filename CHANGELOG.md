# CryoEV changelog

Newest first. One entry per version: what lab users will notice, and which parts changed. The portal shows this page
at /changes ("What's new" next to the version stamp at the foot of every page).

How to add an entry: put it at the top as `## YYYY-MM-DD HH:MM · Short title`, then 2-8 plain bullets, then a
**Components:** line (portal / review page / registry / CLI). When the review page changes, also bump `PAGE_VERSION`
near the top of annotation/review_tool/index.html.

## 2026-11-19 00:30 · Three new fun mode names

- **Fun mode names:** cryonder is now Bubble, and Fringe, OkCryo and Carbon Meets Vesicle join eVHarmony and Gridr. Tap the logo to cycle through all six. Only the names, taglines and colours change; answers work the same.
- **CryoEV teaser:** the sorting line now reads single, multilayer, non-EV particles and more.
- **Components:** review page

## 2026-11-18 23:30 · eVHarmony guide and follow-up fixes

- **Single or Multilayer no longer jumps ahead:** after a Match, picking the kind keeps the card so Outline off can still be tapped. Tap next, press Enter or swipe to move on.
- **How to judge:** Match now says plainly that it means a vesicle (an EV). Nope leads with ice contamination and shows an ice example.
- **Example pictures:** the example object has a clearer outline.
- **CryoEV teaser:** after the consent question everyone (guests too) gets a What is this for link. It opens a short coming soon page about CryoEV inside eVHarmony instead of the portal.
- **Components:** review page

## 2026-11-18 22:00 · eVHarmony sharper cards and its own app

- **Its own app:** an eVHarmony set opens without the CryoEV bar or version stamp, with its own name and a heart icon in the browser tab. Same link as before; guests still land straight on their set. Every other review set keeps the CryoEV bar.
- **Meet the real app:** lab members see a Meet the real app: CryoEV link in the reveal once they answer Count my answers. Guests don't see it.
- **Crisper pictures:** cards with no finer picture behind them are softened less by default (Smooth 1 instead of 2); cards with a finer picture keep Smooth 2. Your own Smooth setting still wins once you change it.
- **Closer look:** a card zooms in further on its vesicle (up to 4x when a sharper picture exists, 2x otherwise).
- **No more box around the sharp crop:** on desktop, the sharper close-up of the vesicle is shaded exactly like the picture around it, so no lighter square shows.
- Pictures on object cards are always scaled with the browser's best quality, and the multi-set review server marks images so proxies and data savers leave them uncompressed.
- For set makers: `object_set build --quality --export-cols --src-use --max-items`, `tools/add_hires.py` item crops up to 2560 px (none coarser than the 2x picture), `tools/evh_items.py --images-of --exclude --answered-attr --head`, `tools/evh_carry.py --map`, `tools/evh_score.py --src`.
- **Components:** portal / review page 2026-11-18.6 / review server (multi) / CLI (`object_set`, `tools/add_hires.py`, `tools/evh_items.py`, `tools/evh_carry.py`, `tools/evh_score.py`).

## 2026-11-18 18:30 · eVHarmony guide in short bullets

- **How to judge is quicker to read:** each answer is a few short bullets instead of a paragraph, with the example images as before.
- **Tips** (contrast, zoom, outline on or off) as bullets at the end of the guide.
- The set's full guidelines text is folded under Full guidelines in the guide.
- Practice cards zoom in further on phones only; on a wide screen they keep the usual framing.
- **Components:** review page 2026-11-18.3.

## 2026-11-18 12:00 · eVHarmony practice cards before the reveal

- **Three practice cards first:** a new player gets a sense of the swiping on three cards with known answers. Practice answers are never saved.
- **Then the reveal:** a short note that these are real vesicle candidates and their swipes help annotate them for CryoEV, the How to judge guide, and the usual Count my answers question.
- **Redo or start:** after that, Redo the 3 practice cards (each shows our answer and why) or Start for real. The practice cards don't count.
- Players who already answered Count my answers skip the practice. Sets without practice cards, and every other review set, work as before.
- For set makers: `tools/evh_practice.py` moves the chosen cards out of the queue and out of the scored export.
- **Components:** review page 2026-11-18.2 / CLI (`tools/evh_practice.py`).

## 2026-11-18 08:50 · eVHarmony v3: three answers and optional follow-ups

- **Back to three answers:** Match, Nope, It's complicated. Small and dense is no longer its own answer (no swipe down); it is now one of the reasons under It's complicated, which covers anything not clearly yes or no.
- **After a Match** (optional, one tap): Single or Multilayer (any part of a multilayer counts, its outer ring or an inner layer), and an Outline off toggle for the right object with an outline that needs correcting.
- **After It's complicated** (optional, one tap): why: Faint bilayer, Small and dense, Cut off at edge, or Other. Skip, or just swipe on, when you like.
- **How to judge** now shows examples for Match (Single, Multilayer), Nope and It's complicated, without the label numbers on the pictures, and the guidelines are updated to match.
- Sets built with the v3 questions (`tools/evh/ls_evh_match_v3.json`) work this way; v1 and v2 sets and every other review set look and work as before. Your earlier answers carry over to a rebuilt set.
- **Components:** review page 2026-11-18.1 / CLI (`tools/evh_score.py` reports kind, outline_off and why per person; `tools/evh_examples.py --caption` repeatable).

## 2026-11-17 14:55 · eVHarmony v2: Small and dense, What is it, How to judge

- **A fourth answer, "Small & dense"** (swipe down, or the button): small, uniformly dark particles without a clear bilayer, possibly lipoproteins. It is kept as its own answer so it can be studied later.
- **After a Match, "What is it?"**: one tap for Single, Multilayer, Has vesicles inside or Inside another, then the next card comes. Skip it, or just swipe on, when you can't tell.
- **How to judge:** a sheet with example pictures for each answer and the guidelines, from the welcome screen ("How to judge") and the **?** button at the top.
- These apply to new eVHarmony sets built with the v2 questions (`tools/evh/ls_evh_match_v2.json`). The current eVHarmony set and every other review set work as before.
- For set makers: some first cards can be known "Nope" cards placed on empty background, so the answers can be checked fairly (`tools/evh_items.py --decoys`, `tools/evh_score.py --decoys --control`); earlier answers move to a rebuilt set with `tools/evh_carry.py`; example pictures go in with `tools/evh_examples.py`.
- **Components:** review page 2026-11-17.3 / CLI (`tools/evh_items.py --decoys --derive --low --answered`, `tools/evh_carry.py`, `tools/evh_examples.py`, `tools/evh_score.py --decoys --control`).

## 2026-11-17 14:28 · eVHarmony save status, Smooth on low-res sets

- **Saves that fail now say so.** When the server refuses your answers (for example "not a reviewer") or can't be reached, eVHarmony shows a red banner with the server's reason and how many answers are waiting. Keep the page open: it keeps retrying, and your answers are kept in this browser until they upload.
- "Saved ✓" at the bottom of every object set now means the server has all your answers, not just the current card's. Until then it says "Saving…" or "Not saved · N answers waiting".
- **View only:** if you open an object set you're not a reviewer of (for example as an admin through the portal), the cards show "View only: you're not a reviewer of this set" and swipes, keys and buttons don't record answers. You can still look through the cards.
- **Smooth works on sets without sharper images.** On sets whose images are already the finest available (no 2x image or item crop), Smooth is now measured in those images' pixels, so Smooth 2 visibly softens the grain. Sets with sharper images look the same as before.
- **Components:** review page 2026-11-17.2 / review server for portal sign-in (`multi_review_server` tells the page whether you may answer).

## 2026-11-17 14:10 · Round 17: eVHarmony beta, guest sign-in

- **eVHarmony:** a joke dating-app look for one kind of object set: swipe right on a real vesicle, left on anything else, up when it's complicated. Each swipe is an ordinary answer, credited to you; the first cards are the same for everyone.
- After your 10th answer it tells you what you were doing and asks whether your answers may count (Yes / No). You can keep swiping either way.
- Low-contrast cards in eVHarmony start with ☼ Auto and Smooth on, until you change them.
- Every other review set looks and works as before.
- **Guest sign-in (off until configured):** people with an email at a domain the admin allows can sign in as a guest. Guests see only review sets opened to guests, and can only answer that set's questions; they can't change labels, notes or anyone else's answers.
- **Components:** review page 2026-11-17.1 / portal (guest role, `python -m portal.guests open|close|list`) / CLI (`annotation.object_set build --app evharmony --first-items N`, `tools/evh_items.py`, `tools/evh_score.py`).

## 2026-11-16 16:40 · Smooth strength slider

- **Smooth** is now a 0–3 slider (0 = off) in the ☼ panel. At full resolution about 1.5–2.5 shows membranes best. An earlier "Smooth on" setting loads as 1.
- Full-resolution card crops are capped at 1536 px (only large items shrink, which are already big on screen), so contrast and Auto work on every card and phones use less memory.
- **Components:** review page 2026-11-16.4 / CLI (`tools/add_hires.py --item-crops`, 1536 px cap).

## 2026-11-16 16:05 · Full-resolution crops on object cards

- **Object and tree cards** now draw each item's crop at the original's full resolution (5760 px wide, 0.37 nm/px), so small EVs are no longer magnified 4–11× on a phone. Added to the ML/MV set (145 of 155 items; 10 sit on images uploaded at 1440 px). Outlines and answers are unchanged.
- `python tools/add_hires.py <object set> --item-crops --do` adds them to other object sets, with the same alignment check.
- **Components:** review page 2026-11-16.3 / CLI (`tools/add_hires.py --item-crops`).

## 2026-11-16 15:10 · Round 16a: sharper review images, contrast controls, review fixes

- **Sharper images:** sets can now carry a 2× copy of each image (2880 px, 0.74 nm/px instead of 1.47). The review page draws it when you zoom in the Editor and always on cards. Outlines and answers are unchanged. Added to the ML/MV object set (44 of 49 images) and the HEK spot-check (32 of 32); `python tools/add_hires.py <set> [--source <folder>] --do` adds them to other sets after an alignment check.
- **Image contrast (☼):** brightness, contrast, gamma, Auto (stretches what you're looking at) and Smooth (light denoise) in the Editor, Quick check and object/tree cards. Only how the image looks on your screen changes; your choices are remembered in your browser.
- **Quick check:** after you pick a class, Accept names that class (#24). With a mouse, dragging the card pans it instead of deciding (#25).
- **Editor:** Tab / Shift+Tab steps through the objects in the image (#25).
- **Reviews list:** shows images next to cards ("1 of 5 cards · 32 images") (#26).
- **Components:** review page 2026-11-16.2 / review server (data.js always fresh) / portal (Reviews list) / CLI (`tools/add_hires.py`).

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
