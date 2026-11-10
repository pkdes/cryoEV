"""
Predict, free disk and the upload disk holds (Reservation: bytes an upload may still write, per drive).

Split out of portal/modules/predict.py (round 7, P7; no behaviour change); predict.py re-exports every name.
Names the tests patch are used from other parts only as <PART>.<name> (portal/predict_parts/__init__.py).
"""
from __future__ import annotations

import os
import secrets
import shutil
import threading
from pathlib import Path

from portal.predict_parts.common import Refusal
from portal.predict_parts import common as COMMON


# ---------------------------------------------------------------- free disk and the upload disk holds
def disk_free(path) -> int:
    return shutil.disk_usage(path).free


_RESERVED: dict = {}                     # upload id -> bytes it may still write (its Content-Length minus written)
_RES_DEV: dict = {}                      # upload id -> the drive (st_dev) its hold is on
_RES_LOCK = threading.Lock()


def _dev(path):
    """The volume a path is (or would be) on; None when it can't be told."""
    try:
        return os.stat(_existing(path)).st_dev
    except OSError:
        return None


def _held(dev) -> int:
    """Bytes the uploads in flight may still write on this drive. Callers hold _RES_LOCK."""
    return sum(n for k, n in _RESERVED.items() if _RES_DEV.get(k) == dev)


class Reservation:
    """Disk space held for one upload while it streams. check() refuses (507) when the free space minus what every
    upload in flight may still write would drop below FREE_MARGIN_BYTES. A chunked upload (no Content-Length)
    holds nothing ahead; the checks during the upload still see the space it has used."""

    def __init__(self, dest: Path, length: int):
        self.dest, self.key = dest, secrets.token_hex(8)
        self.dev = _dev(dest)
        with _RES_LOCK:
            _RESERVED[self.key] = max(0, int(length or 0))
            _RES_DEV[self.key] = self.dev

    def written(self, n: int):
        with _RES_LOCK:
            if self.key in _RESERVED:
                _RESERVED[self.key] = max(0, _RESERVED[self.key] - n)

    def check(self):
        free = disk_free(self.dest)
        with _RES_LOCK:
            held = _held(self.dev)
        if free - held < COMMON.FREE_MARGIN_BYTES:
            raise Refusal("Not enough disk space on the server for this upload. Nothing was kept. Ask the admin.",
                          507, tech=f"{free} bytes free, {held} held by uploads in flight, "
                                    f"margin {COMMON.FREE_MARGIN_BYTES}")

    def release(self):
        with _RES_LOCK:
            _RESERVED.pop(self.key, None)
            _RES_DEV.pop(self.key, None)


def _existing(p: Path) -> Path:
    p = Path(p)
    while not p.exists() and p.parent != p:
        p = p.parent
    return p
