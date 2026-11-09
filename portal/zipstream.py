"""
Streamed zip files for the portal (BUILD_CONTRACTS.md section 13; Gallery downloads, later the Phase 2 exports).

    stream_zip(entries) -> iterator[bytes]
        entries: iterable of (arcname, source); source is a path (str / Path: read in 1 MiB pieces), bytes, or an
        iterable of bytes. Nothing is written to disk and the whole archive is never held in memory: `zipfile` writes
        into an unseekable sink (so every member gets a data descriptor), and the sink is drained after every piece.
        - Images (png jpg jpeg gif webp zip gz) are ZIP_STORED, everything else ZIP_DEFLATED.
        - Members of unknown size (iterables) and files of 1 GiB or more are written with zip64 extensions; the
          central directory switches to zip64 by itself when the archive passes 4 GiB or 65535 members.
        - Arcnames are sanitised (`safe_arcname`): never absolute, no `..`, no drive letters, no control or
          Windows-reserved characters, duplicates get " (2)".
        - Paths are checked (is a file) when stream_zip is CALLED, so a missing file is an exception before the first
          byte, not a truncated archive. A file that disappears while streaming ends the stream with that error.
    safe_arcname(name) -> str
"""
from __future__ import annotations

import os
import re
import time
import zipfile
from pathlib import Path

PIECE = 1024 * 1024
FLUSH = 256 * 1024
ZIP64_AT = 1 << 30
STORED_EXT = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp", ".zip", ".gz", ".mrc"})
BAD = re.compile(r'[\x00-\x1f<>:"|?*\\]')
RESERVED = frozenset({"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10))})
MAX_PART = 120


def _part(p: str) -> str:
    p = BAD.sub("_", p).strip(" .")
    if not p:
        return ""
    if p.split(".", 1)[0].strip().lower() in RESERVED:           # CON, con.txt, nul.tar.gz: the part before the FIRST dot
        p = "_" + p
    if len(p) > MAX_PART:
        base, ext = os.path.splitext(p)
        p = base[:MAX_PART - len(ext[:12])] + ext[:12]
    return p


def safe_arcname(name) -> str:
    """A zip member name that can't escape the folder it is extracted into."""
    parts = [_part(p) for p in str(name).replace("\\", "/").split("/")]
    parts = [p for p in parts if p and p not in (".", "..") and set(p) != {"."}]
    return "/".join(parts) or "file"


class _Sink:
    """The unseekable file `zipfile` writes into: collects bytes until the generator drains them."""

    def __init__(self):
        self.buf = bytearray()

    def write(self, b) -> int:
        self.buf += b
        return len(b)

    def flush(self):
        pass

    def take(self) -> bytes:
        out = bytes(self.buf)
        self.buf.clear()
        return out


def _prepare(entries):
    out, seen = [], set()
    for arc, src in entries:
        name = safe_arcname(arc)
        base, n = name, 1
        while name.lower() in seen:
            n += 1
            stem, ext = os.path.splitext(base)
            name = f"{stem} ({n}){ext}"
        seen.add(name.lower())
        size, mtime = None, time.time()
        if isinstance(src, (str, os.PathLike)):
            src = Path(src)
            st = src.stat()                                   # raises now when the file isn't there
            if not src.is_file():
                raise FileNotFoundError(str(src))
            size, mtime = st.st_size, st.st_mtime
        elif isinstance(src, (bytes, bytearray, memoryview)):
            src, size = bytes(src), len(src)
        out.append((name, src, size, mtime))
    return out


def _pieces(src):
    if isinstance(src, Path):
        with open(src, "rb") as f:
            while True:
                b = f.read(PIECE)
                if not b:
                    return
                yield b
    elif isinstance(src, bytes):
        for i in range(0, len(src), PIECE):
            yield src[i:i + PIECE]
    else:
        for b in src:
            yield bytes(b)


def _generate(prepared):
    sink = _Sink()
    zf = zipfile.ZipFile(sink, "w", allowZip64=True)
    try:
        for name, src, size, mtime in prepared:
            tm = time.localtime(max(mtime, 315532800))        # zip dates start in 1980
            zi = zipfile.ZipInfo(name, date_time=tm[:6])
            ext = os.path.splitext(name)[1].lower()
            zi.compress_type = zipfile.ZIP_STORED if ext in STORED_EXT else zipfile.ZIP_DEFLATED
            zi.external_attr = 0o644 << 16
            with zf.open(zi, "w", force_zip64=size is None or size >= ZIP64_AT) as w:
                for piece in _pieces(src):
                    w.write(piece)
                    if len(sink.buf) >= FLUSH:
                        yield sink.take()
            if sink.buf:
                yield sink.take()
    finally:
        zf.close()                                            # writes the central directory into the sink
    tail = sink.take()
    if tail:
        yield tail


def stream_zip(entries):
    """-> iterator of bytes: the zip file, streamed. See the module docstring."""
    return _generate(_prepare(entries))
