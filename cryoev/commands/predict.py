"""predict new <args>: a new prediction set through `jobs/predict_set.py create <args>` (the args go there unchanged;
`python -m jobs.predict_set create --help` lists them).

Dry run (default): predict_set's own --dry-run: every check create makes (images by content, nm/px, sample labels,
the model, the registry record), then the plan; nothing is written. --do: create for real (the set folder under
<CryoAI-laptop>/predict and a prediction_set registry line, status queued). Running the set is a separate step
(`python -m jobs.predict_local --ps <id>`, docs/JOBS.md "Predict jobs"). Paths under Box are refused.
"""
from __future__ import annotations

from cryoev import common as CM

PATH_FLAGS = ("--from-dir", "--manifest", "--only", "--registry", "--root")
OWN = {"--do": 0, "--who": 1}                       # this CLI's flags (number of values) taken out of the args


def add(sub):
    p = sub.add_parser("predict", help="prediction sets: new (dry run unless --do)")
    s = p.add_subparsers(dest="pcmd", required=True)
    n = s.add_parser("new", help="jobs/predict_set.py create <args> (its --help lists them)")
    CM.write_flags(n)
    n.set_defaults(func=new)


def split_argv(argv) -> tuple:
    """-> (args for predict_set create, this CLI's own flags)."""
    rest, own, i = [], [], 0
    argv = list(argv)
    while i < len(argv):
        w = argv[i]
        key = w.split("=", 1)[0]
        if key in OWN:
            k = OWN[key] if "=" not in w else 0
            own += argv[i:i + 1 + k]
            i += 1 + k
            continue
        rest.append(w)
        i += 1
    return rest, own


def _paths(args) -> list:
    out = []
    for i, w in enumerate(args):
        key, eq, val = w.partition("=")
        if key in PATH_FLAGS:
            v = val if eq else (args[i + 1] if i + 1 < len(args) else None)
            if v:
                out.append(v)
    return out


def new(a) -> int:
    from jobs import predict_set as PS
    args = list(a.extra)
    if "--dry-run" in args:
        args.remove("--dry-run")
    for p in _paths(args):
        problem = CM.box_problem(p)
        if problem:
            raise CM.Refused(f"refused: {problem}. Nothing under Box is read or written by agents ({CM.GUIDE}).")
    safe = {"argv": args[:40]}                       # flags and local paths/labels only; nothing secret

    def plan():
        print("python -m jobs.predict_set create " + " ".join(args) + " --dry-run")
        rc, _ = CM.call_main(PS.main, ["create", *args, "--dry-run"])
        return rc

    def act(who, via):
        rc, text = CM.call_main(PS.main, ["create", *args], capture=True)
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        ps = lines[-1] if rc == 0 and lines and PS.PS_RE.match(lines[-1]) else None
        return rc, {"ps": ps}

    return CM.run_write(a, "prediction set create", "prediction set (new)", safe, plan, act)
