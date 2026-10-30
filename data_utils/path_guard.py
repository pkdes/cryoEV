"""
One guard for "never write under these folders" (Box, Cryo-image-processing), shared by
inference/inference_check.py and annotation/apply_review_edits.py. Standard library only.

protected_problem(path, roots) refuses a path when, after normalization, it is a protected root or lies
under one. It closes the holes the plain abspath + string-prefix check had on Windows:
  - \\\\?\\ and \\\\.\\ prefixes (\\\\?\\C:\\... is stripped; \\\\.\\ device paths and \\\\?\\UNC\\ are refused)
  - UNC paths and admin shares (\\\\localhost\\C$\\..., \\\\127.0.0.1\\c$\\...): refused outright, they can
    name a local folder by another spelling
  - trailing spaces / dots in a component (Windows drops them: "Box \\x" is "Box\\x")
  - case and / vs \\ (normcase)
  - links, junctions and 8.3 short names: the deepest existing ancestor is resolved with realpath, and
    each existing ancestor is compared with each existing root by os.path.samefile
"""
from __future__ import annotations

import ntpath
import os
from pathlib import Path

PROTECTED_ROOTS = (Path.home() / "Desktop" / "Cryo-image-processing", Path.home() / "Box")


def _strip_prefix(s: str) -> tuple[str, str | None]:
    """-> (path, problem). Drops \\\\?\\ before a drive; refuses device and UNC forms."""
    s = str(s)
    if s.startswith(("\\\\?\\", "//?/")):
        rest = s[4:]
        if rest[:4].upper() in ("UNC\\", "UNC/"):
            return s, "is a UNC path (\\\\?\\UNC\\...)"
        s = rest
    if s.startswith(("\\\\.\\", "//./")):
        return s, "is a device path (\\\\.\\...)"
    if os.name == "nt" and s.startswith(("\\\\", "//")):
        return s, "is a UNC / network path (e.g. an admin share); write to a local drive path"
    return s, None


def normalize(path) -> str:
    """Absolute, case-folded, separator-normalized, with Windows' trailing space/dot trimming per component."""
    s, _ = _strip_prefix(str(path))
    s = os.path.abspath(s)
    if os.name == "nt":
        drive, rest = ntpath.splitdrive(s)
        parts = [p.rstrip(" .") if p not in (".", "..") else p for p in rest.replace("/", "\\").split("\\")]
        s = os.path.abspath(drive + "\\".join(parts))
    return os.path.normcase(s)


def _under(p: str, r: str) -> bool:
    return p == r or p.startswith(r.rstrip("\\/") + os.sep)


def _existing_ancestors(s: str):
    p = Path(s)
    for a in (p, *p.parents):
        try:
            if a.exists():
                yield a
        except OSError:
            continue


def protected_problem(path, roots=PROTECTED_ROOTS) -> str | None:
    """Why `path` may not be written to (it is, or is under, a protected root), or None."""
    raw, problem = _strip_prefix(str(path))
    if problem:
        return f"{path} {problem}"
    p = normalize(raw)
    ancestors = list(_existing_ancestors(p))
    real = None
    if ancestors:  # resolve links / junctions / short names on the part that exists, keep the rest as typed
        deepest = ancestors[0]
        tail = os.path.relpath(p, os.path.normcase(os.path.abspath(deepest))) if str(deepest) != p else ""
        real = normalize(os.path.join(os.path.realpath(deepest), tail) if tail and tail != "." else os.path.realpath(deepest))
    for root in roots:
        r = normalize(root)
        rr = normalize(os.path.realpath(r)) if os.path.exists(r) else r
        for cand in {p, real} - {None}:
            if _under(cand, r) or _under(cand, rr):
                return f"{path} is under {root}, which is read-only"
        if os.path.exists(r):
            for a in ancestors:
                try:
                    if os.path.samefile(a, r):
                        return f"{path} is under {root}, which is read-only"
                except OSError:
                    continue
    return None
