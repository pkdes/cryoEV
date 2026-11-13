"""predict new <args>: a new prediction set through `jobs/predict_set.py create <args>` (the args go there unchanged;
`python -m jobs.predict_set create --help` lists them).

Dry run (default): predict_set's own --dry-run: every check create makes (images by content, nm/px, sample labels,
the model, the registry record), then the plan; nothing is written. --do: create for real (the set folder under
<CryoAI-laptop>/predict and a prediction_set registry line, status queued). Running the set is a separate step
(`python -m jobs.predict_local --ps <id>`, docs/JOBS.md "Predict jobs"). Paths under Box are refused.
"""
from __future__ import annotations

from cryoev import common as CM

READ_SOURCES = ("--from-dir", "--manifest", "--only")   # read only: Box refused; Cryo-image-processing allowed
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


def check_args(args) -> list:
    """The user's args for `predict_set create`, normalised and checked: every option must be an EXACT create option
    (predict_set's own parser; abbreviations such as --from or --man are refused, since argparse would expand them);
    `--dry-run` is dropped (only this CLI adds it); every value is refused under Box, and the values of the flags
    that write (--root, --registry, and anything that is not a read source) under any protected root
    (data_utils/path_guard.py). -> the args to forward. Raises Refused (Box) or CliError."""
    from jobs import predict_set as PS
    opts = PS.create_options()
    out, i = [], 0
    while i < len(args):
        w = args[i]
        if not w.startswith("-"):
            if CM.box_problem(w):
                raise CM.Refused(f"refused: {CM.box_problem(w)}. Nothing under Box is read or written by agents.")
            raise CM.CliError(f"refused: unexpected argument {w!r} (create takes options only)")
        key, eq, val = w.partition("=")
        if key == "--dry-run" and not eq:
            i += 1
            continue
        if key not in opts or key in ("-h", "--help"):
            raise CM.CliError(f"refused: {key} is not an exact `jobs/predict_set.py create` option (no abbreviations; "
                              f"see python -m jobs.predict_set create --help)")
        if opts[key] == 0:
            if eq:
                raise CM.CliError(f"refused: {key} takes no value")
            out.append(w)
            i += 1
            continue
        if eq:
            value, take = val, 1
        elif i + 1 < len(args):
            value, take = args[i + 1], 2
        else:
            raise CM.CliError(f"refused: {key} needs a value")
        problem = CM.box_problem(value)
        if problem:
            raise CM.Refused(f"refused: {key} {problem}. Nothing under Box is read or written by agents ({CM.GUIDE}).")
        if key not in READ_SOURCES:
            problem = CM.path_guard.protected_problem(value)
            if problem:
                raise CM.CliError(f"refused: {key} {problem}")
        out += args[i:i + take]
        i += take
    return out


def new(a) -> int:
    from jobs import predict_set as PS
    args = check_args(list(a.extra))
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
