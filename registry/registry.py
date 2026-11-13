"""Script shim: `python registry/registry.py <command> ...` runs registry.store.main() (the module moved to
registry/store.py in round 11, when registry/ became a package). Code imports `from registry import store`.

Run by path, Python puts registry/ first on sys.path, where this file would shadow the package: it is dropped and
the repo root added before the import. Loaded by file path (portal/modules/data.py until the round 11 merge
re-points it), it re-exports store's names; the functions are store's own, so they share its state.
"""
import sys

if __name__ == "__main__":     # run by path: registry/ off sys.path (it would shadow the package), the repo on it
    import os as _os
    _here = _os.path.dirname(_os.path.abspath(__file__))
    sys.path[:] = [p for p in sys.path if _os.path.normcase(_os.path.abspath(p or ".")) != _os.path.normcase(_here)]
    sys.path.insert(0, _os.path.dirname(_here))

from registry import store as _store  # noqa: E402

if __name__ == "__main__":
    sys.exit(_store.main())
globals().update({k: v for k, v in vars(_store).items() if not (k.startswith("__") and k.endswith("__"))})
