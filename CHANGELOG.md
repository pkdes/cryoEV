# CryoEV changelog

Newest first. One entry per version: what lab users will notice, and which parts changed. The portal shows this page
at /changes ("What's new" next to the version stamp at the foot of every page).

How to add an entry: put it at the top as `## YYYY-MM-DD HH:MM · Short title`, then 2-8 plain bullets, then a
**Components:** line (portal / review page / registry / CLI). When the review page changes, also bump `PAGE_VERSION`
near the top of annotation/review_tool/index.html.

## 2026-11-12 19:30 · Round 9 platform (preview, not live yet)

- **Jobs** is a flow board: click any job to see its details, with a strip showing what each machine is doing.
- **Data** pages list the label classes, trace a label set back through its lineage to the imaging sessions it came from, and show lanes (automatic, human, revised) with arrows between them.
- **Admin > Recent actions** logs every portal action, sign-ins, and actions run from the command line.
- A new **About** page (link at the foot of every page) and this **What's new** page.
- An agent command line (`python -m cryoev`): actions an assistant runs for you show as "claude for <your name>".
- Sign in with Google is offered as an option.
- The version stamp at the foot of each page now also shows the review page's own version on review pages.

**Components:** portal, registry, CLI

## 2026-11-12 19:08 · Review page update (blind-audit package only)

- Back no longer jumps two images on a quick double tap.
- The image list shows as cards.
- Pinch to zoom on phones and tablets.
- Quick check has a point eraser.
- A Fun-mode switch.
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
