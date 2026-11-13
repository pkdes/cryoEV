"""review progress <package>: cards decided / total and images reviewed / total, COUNTS only.

Blind-audit rule: a compare package's answer key lives in <review>/_work_<package>/ (decoys.json), and who decided
what is the reviewers' business until the package is applied. So this command
  - takes a package NAME under <CryoAI-laptop>/review, never a path, and refuses any `_work_*` name;
  - prints only counts from portal/titles.py (scan_packages / package_progress), never the decisions, never who;
  - refuses --decisions while the package is not applied (exit 3); after it is applied the decisions are in the
    label version (`python -m cryoev pull labels <lv>`), never read here.
"""
from __future__ import annotations

from cryoev import common as CM

KEYS = ("cards", "decided", "images", "reviewed")


def add(sub):
    p = sub.add_parser("review", help="review packages: progress counts (read)")
    s = p.add_subparsers(dest="rcmd", required=True)
    pr = s.add_parser("progress", help="cards decided / total, images reviewed / total (counts only)")
    pr.add_argument("package", help="package folder name under <CryoAI-laptop>/review")
    pr.add_argument("--decisions", action="store_true",
                    help="asks for decision contents: refused while the package is not applied")
    pr.add_argument("--json", action="store_true")
    pr.set_defaults(func=progress)


def _check_name(name: str) -> None:
    parts = name.replace("\\", "/").split("/")
    if any(p.lower().startswith("_work") for p in parts):
        raise CM.Refused(f"refused: {name} is a `_work_*` folder (a blind audit's answer key): agents never read "
                         f"it. See {CM.GUIDE}, \"Golden rules\".")
    if len(parts) != 1 or name in (".", "..") or not name.strip() or ":" in name:
        raise CM.CliError(f"give a package folder name under <CryoAI-laptop>/review, not a path: {name!r}")


def progress(a) -> int:
    from portal import titles as TT
    _check_name(a.package)
    root = CM.laptop_root() / "review"
    pdir = root / a.package
    if not (pdir / "data.js").is_file():
        print(f"no review package {a.package} under {root}")
        return 1
    reg, rr = CM.registry(), CM.registry_root()
    records = {e: reg.current(e, rr) for e in ("label_version", "review_package", "pull")}
    row = next((p for p in TT.scan_packages(root, records) if p["id"] == a.package), None)
    if row is None:                                   # prediction packages, parts: not in the lab scan
        prog = TT.package_progress(pdir)
        if prog is None:
            print(f"{a.package}: data.js not readable")
            return 1
        row = {**prog, "status": "unknown", "applied_by": None}
    out = {"package": a.package, "status": row["status"], **{k: row[k] for k in KEYS},
           "applied_by": row.get("applied_by")}
    if a.decisions:
        if out["status"] != "applied":
            raise CM.Refused(f"refused: {a.package} is {out['status']}, not applied: decision contents stay closed "
                             f"while a review is open (a blind audit may be running). Counts only.")
        raise CM.CliError(f"{a.package} is applied as {out['applied_by']}: read its decisions from that label "
                          f"version (python -m cryoev pull labels {out['applied_by']} --out <dir>), not here")
    if a.json:
        CM.emit_json(out)
    else:
        print(f"{a.package}: {out['status']}; cards decided {out['decided']}/{out['cards']}; images reviewed "
              f"{out['reviewed']}/{out['images']}" + (f"; applied as {out['applied_by']}" if out["applied_by"] else ""))
    return 0
