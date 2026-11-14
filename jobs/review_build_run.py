#!/usr/bin/env python3
"""The detached runner of a portal review build (round 13, Start a review): portal/review_build.py run().

    python -m jobs.review_build_run --spec <portal state>/review_build/<name>/spec.json

Started by portal/review_build.py start() with no console window, so it survives a portal restart. It holds
<review root>/.building/build.lock for its whole life (one build at a time across both portal doors), and for a
model build <predict root>/predict.lock too (as jobs/predict_local.py), then runs annotation/compare_package.py's
build() IN this process (no child: killing the runner stops any GPU work with it), building under
<review root>/.building/<name>/pkg and renaming the checked package into the review root at the end. Progress: <review root>/.building/<name>/status.json (allowlisted phases). The raw log stays in the portal's
state folder (<state>/review_build/<name>/build.log).
Exit: 0 done, 1 failed, 2 usage, 3 another build holds the lock.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:                     # run as a script: the repo root, so `portal` and `jobs` import
    sys.path.insert(0, str(REPO))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", type=Path, required=True)
    a = ap.parse_args(argv)
    if not a.spec.is_file():
        print(f"no spec file: {a.spec}", file=sys.stderr)
        return 2
    from portal import review_build
    return review_build.run(a.spec)


if __name__ == "__main__":
    sys.exit(main())
