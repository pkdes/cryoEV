"""cryoev: the agent CLI for the CryoEV platform. Guide: docs/AGENTS.md. Run from the repo root.

Read (never write; --json on each):
  python -m cryoev status [--share \\\\host\\cj]          services, desktop jobs, review packages (counts), open feedback
  python -m cryoev find <text> [--entity E] [--status S]   registry ids containing <text>
  python -m cryoev show <id>                               the registry record (= registry/registry.py show)
  python -m cryoev lineage <id>                            the chain back to the imaging sessions (the Data page's)
  python -m cryoev review progress <package>               cards decided / total; counts only (blind-audit rule)
  python -m cryoev feedback list                           the portal's feedback reports

Write (a DRY RUN that writes nothing unless --do; each --do is logged to <CryoAI-laptop>/logs/agent_actions.jsonl):
  python -m cryoev pull labels|predictions|results|dataset <id> --out DIR [--do]   copy out + sha256 manifest
  python -m cryoev feedback status <n> "<status>" [--do]                           set a report's status
  python -m cryoev feedback reply <n> "<text>" [--do]                              reply to a report
  python -m cryoev register [--do]                         registry import, then the history page is regenerated
  python -m cryoev predict new <jobs/predict_set.py create args> [--do]            a new prediction set

Who: --who, else $CRYOEV_FOR; the agent: $CRYOEV_VIA (Claude Code sets both). Never automated (refused, exit 3):
Roboflow uploads / relabels, production / promotion, git push, desktop training sends, anything under Box.
Exit codes: 0 ok; 1 not found (unknown id / package, no match) or done but its result line could not be logged;
2 refused or failed (bad arguments, a guard, an error); 3 refused by rule (never automated, blind audit).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

if __package__ in (None, ""):                       # `python cryoev/__main__.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "cryoev"

from cryoev import common as CM  # noqa: E402
from cryoev.commands import feedback, predict, pull, read, register, review, status  # noqa: E402

_ROBOFLOW = ("Roboflow uploads, relabels and deletions are never automated: the user runs annotation/roboflow_*.py "
             "himself, each time, after annotation/roboflow_snapshot.py.")
_PROD = ("Setting production / promoting a model is the user's call only: he types `python registry/registry.py "
         "set-status <model> production --who pujan` himself (the registry refuses it when CRYOEV_VIA is set).")
_PUSH = ("git push is never automated: the user pushes after reading the pre-push scan (.githooks/pre-push).")
_SEND = ("Sending work to the desktop (jobs/sync.py push / push-predict / local) is the user's call each time: an "
         "agent prepares the bundle (training/make_job_bundle.py) and stops there.")
_BOX = "Nothing under Box is read or written by agents."
NEVER = {"roboflow": _ROBOFLOW, "upload": _ROBOFLOW, "relabel": _ROBOFLOW, "rf": _ROBOFLOW,
         "promote": _PROD, "production": _PROD, "prod": _PROD,
         "push": _PUSH, "git": _PUSH,
         "train": _SEND, "send": _SEND, "sync": _SEND, "queue": _SEND,
         "box": _BOX}
GROUPS = {"predict", "registry", "model", "models", "job", "jobs", "dataset", "labels", "review"}


def never_problem(argv) -> str | None:
    """A refusal text when the command looks like a never-automated action, else None. Checks the command word
    (and its parts: push-predict, set-production) and, for group words, the word after it (predict push)."""
    words = [w for w in argv if not w.startswith("-")][:2]
    if not words:
        return None
    checks = [words[0]] + ([words[1]] if len(words) > 1 and words[0].lower() in GROUPS else [])
    for n, w in enumerate(checks):
        parts = re.split(r"[-_\s]+", w.lower())
        for part in parts:
            if part in NEVER:
                why = _SEND if part == "push" and (n or "predict" in parts) else NEVER[part]   # predict push = a send
                return f"refused: `{w}` is never automated. {why} See {CM.GUIDE}, \"Never automated\"."
    return None


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m cryoev", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False)
    sub = ap.add_subparsers(dest="cmd", required=True, metavar="command")
    for m in (status, read, pull, review, feedback, register, predict):
        m.add(sub)
    return ap


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    bad = never_problem(argv)
    if bad:
        print(bad, file=sys.stderr)
        return CM.EXIT_REFUSED
    extra = []
    if argv[:2] == ["predict", "new"]:              # the rest goes to jobs/predict_set.py create unchanged
        extra, own = predict.split_argv(argv[2:])
        argv = ["predict", "new", *own]
    a = parser().parse_args(argv)
    a.extra = extra
    try:
        return a.func(a)
    except CM.Refused as e:
        print(e, file=sys.stderr)
        return CM.EXIT_REFUSED
    except CM.CliError as e:
        print(e, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
