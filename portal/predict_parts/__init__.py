"""
The parts of the Predict module (portal/modules/predict.py), split out in round 7 (P7) with no behaviour change.
They live outside portal/modules/ so the module loader (discover()) sees only predict.py.

  common     settings and limits, Refusal, the roots, small readers and times, HTML helpers
  sets       models, a session's tiles, reading sets, background work, a set's status
  estimate   how long an analysis takes
  raw        processing repo + recipes, raw form fields, GPU check, raw session folders, raw file checks
  disk       free disk and the upload disk holds (Reservation)
  runner     raw conversion records and the runner (convert, stitch, register, sweep, watcher), conversion status
  create     the sources people may pick, checks, background creates, kept tiles, session plan / run, audit, rename
  uploads    the multipart stream, the upload sessions API, declared files, the start-up sweep
  pages      the pages of sets (My analyses, New analysis, session and upload forms, a set, preparing)
  convpages  the raw pages (raw upload form, a conversion, raw sessions) and the conversion settings

Each part imports only the parts above it in this list. predict.py keeps PREFIX, ROLES, the page script and
routes() with its handlers, and re-exports every name the module had before the split (P.<name> reads keep working).

Patching: a test patches a name on the part that DEFINES it (mock.patch.object(DISK, "disk_free"), not P). Names the
tests patch are used by the other parts and the handlers only through that part (COMMON.MAX_FILES, DISK.disk_free,
CREATE.check_active, ...), never through a `from ... import` binding of their own, so one patch reaches every caller.
Collaborators are used the same way (PS.create, PSRC.check_folder). portal/tests/test_predict_patch_guard.py checks
every patch in portal/tests against this rule.
"""
