"""
Shim: the attribution helpers moved to data_utils/actor.py (round 10, TODO #61) so they import as a normal package
everywhere. This file keeps the old ways working: `from registry import actor` and loads by file path (round 11: nothing
puts registry/ on sys.path any more, so the bare `import actor` is gone; new code imports data_utils.actor). Every name of the real module is re-exported, private ones
(e.g. _people) included; registry/tests/test_actor.py checks the two expose the same names.
"""
import sys as _sys
from pathlib import Path as _Path

_ROOT = str(_Path(__file__).resolve().parent.parent)
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

from data_utils import actor as _impl  # noqa: E402

globals().update({k: v for k, v in vars(_impl).items() if not (k.startswith("__") and k.endswith("__"))})
