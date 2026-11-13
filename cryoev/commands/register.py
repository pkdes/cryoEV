"""register: bring the registry up to date with work done outside the tools, then regenerate the history page.

Dry run: the importer's own dry run (registry/importers.py Importer.scan + write(dry_run=True), = `registry.py import
--dry-run`: what it would append), plus the PRODUCTION restatement the real import would make (importers.PRODUCTION:
when the production model's current line no longer says production, the import re-appends the user's recorded
decision, `set-status <model> production --who <user>`, never stamped with an agent), plus the history-page command
that would follow (its output is not checked in a dry run).
--do: `registry/registry.py import` (append-only), then `registry/history_page.py`, which REGENERATES
<registry>/history.html (the whole file is rewritten).
"""
from __future__ import annotations

from cryoev import common as CM


def add(sub):
    p = sub.add_parser("register", help="registry import + regenerate the history page (dry run unless --do)")
    CM.write_flags(p)
    p.set_defaults(func=run)


def _history():
    from registry import history_page
    return history_page


def _importers():
    from registry import importers
    return importers


def production_restatement(importers, reg, root, imp) -> str | None:
    """The production model the real import would restate (importers._write's last step), or None."""
    prod_id, _ = importers.PRODUCTION
    cur = reg.current(root=root).get(prod_id)
    if cur is not None:
        return prod_id if cur.get("status") != "production" else None
    return prod_id if any(e == "model" and r.get("id") == prod_id for e, r in imp.out) else None


def run(a) -> int:
    root = CM.registry_root()
    page = root / "history.html"

    def plan():
        I, reg = _importers(), CM.registry()
        print(f"1. python registry/registry.py --root {root} import --dry-run:")
        imp = I.Importer(root).scan()
        appended = imp.write(dry_run=True)
        prod = production_restatement(I, reg, root, imp)
        if prod:
            appended["model (set-status production)"] += 1
        cur = reg.current(root=root)
        by = I.Counter((v["schema"].split("/")[0], v.get("status") or "-") for v in cur.values())
        I.print_result({"appended": appended, "refused": imp.refused, "examples": dict(imp.examples),
                        "excluded": imp.excluded, "current": by, "dry_run": True})
        if prod:
            print(f"   would restate production for {prod} as recorded by the user ({I.PRODUCTION[1]}): one "
                  f"set-status line, never stamped with an agent")
        print(f"2. then: python registry/history_page.py  -> regenerates {page} (rewritten whole; outputs not "
              f"checked in a dry run)")
        return 0

    def act(who, via):
        rc, text = CM.call_main(CM.registry().main, ["--root", str(root), "import"], capture=True)
        appended = next((ln for ln in text.splitlines() if ln.startswith("appended")), "")
        if rc:
            return rc, {"import": appended or f"exit {rc}", "history_page": "not regenerated (import failed)"}
        hrc, _ = CM.call_main(_history().main, ["--registry", str(root), "--out", str(page)])
        print(f"history page regenerated: {page}" if hrc == 0 else f"history page NOT regenerated (exit {hrc})")
        return hrc, {"import": appended[:200], "history_page": "regenerated" if hrc == 0 else f"failed (exit {hrc})"}

    return CM.run_write(a, "registry import", "registry", {}, plan, act)
