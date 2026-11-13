"""cryoev: the agent CLI for the CryoEV platform (round 9). Run it from the repo root: `python -m cryoev --help`.

A thin layer over the repo's own tools (the registry, jobs/predict_set.py, portal/feedback_store.py, ...): every
command calls an existing function or forwards argv to an existing main(); no logic is re-implemented here.
Guide: docs/AGENTS.md.
"""
