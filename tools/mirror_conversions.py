"""Mirror finished raw-tile conversions to the desktop share ("queue-share"), verified, never deleting there.

  python tools/mirror_conversions.py --source <raw_tiles dir> --dest <share conversions dir>
                                     [--min-free-gb 20] [--dry-run] [--json] [--copier auto|robocopy|shutil]

--dest comes from the caller (the nightly .bat), never from the repo. For each conversion folder under --source
whose status.json says state "done": copy everything except input/ (raw uploads are large and temporary) into
<dest>/<cid>/ with copy-only semantics (jobs.common.robocopy_argv: never /MIR, /PURGE or /MOV; nothing on dest is
ever deleted). Then verify: every tile's md5 against the MANIFEST md5 column, and every copied file's md5 against
its source. The result goes to <conv>/mirror.json (atomic write, in the SOURCE folder):
  {"dest_rel","mirrored_at","n_files","bytes","md5_ok","error", + "fingerprint","previous"}
dest_rel is a folder name relative to --dest only (no host or share path).

Changed after a previous mirror (retried conversion: different convert.json / tile md5s): any file that differs
from what is already on dest sends the whole conversion to a new folder <cid>__<YYYYMMDD-HHMMSS>; the old folder is
untouched. Two mutable files are exempt: gallery/**/*.jsonl (append only: updated in place when the dest copy is a
byte prefix of the source) and status.json (copied once; later source edits such as `registered` are not re-mirrored).
Unchanged conversions are skipped by a size+mtime fingerprint stored in mirror.json (no md5 reads on dest).
Free-space floor: below --min-free-gb on dest, stop and record the error.
One summary line per conversion; exit 0 on success or nothing to do, 1 on any failure.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from jobs import common  # noqa: E402

SKIP_TOP = {"input", "mirror.json"}          # input/ = raw uploads (large, temporary); mirror.json = this tool's own record
ONCE_ONLY = {"status.json"}                  # copied if absent, never compared or overwritten
MIRROR = "mirror.json"


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def write_json_atomic(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def list_files(conv):
    """{posix relpath: Path} of everything to mirror (all but input/ and mirror.json)."""
    out = {}
    for p in sorted(conv.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(conv).as_posix()
        if rel.split("/")[0] in SKIP_TOP or rel.endswith(".tmp"):
            continue
        out[rel] = p
    return out


def fingerprint(files):
    h = hashlib.sha1()
    for rel, p in files.items():
        if rel in ONCE_ONLY:
            continue
        st = p.stat()
        h.update(f"{rel}\0{st.st_size}\0{st.st_mtime_ns}\n".encode())
    return h.hexdigest()


def is_append_only(rel):
    return rel.startswith("gallery/") and rel.endswith(".jsonl")


def is_prefix(small, big):
    """True when file `small` is a byte prefix of file `big`."""
    with open(small, "rb") as a, open(big, "rb") as b:
        while True:
            x = a.read(1 << 20)
            if not x:
                return True
            if b.read(len(x)) != x:
                return False


def check_tiles(conv):
    """Source tiles vs MANIFEST md5. Returns an error string or None."""
    import csv
    man = next((p for p in (conv / "tiles" / "MANIFEST.csv", conv / "MANIFEST.csv") if p.is_file()), None)
    if man is None:
        return "no MANIFEST.csv"
    with open(man, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    bad = []
    for r in rows:
        name, want = r.get("output_name", ""), (r.get("md5") or "").strip().lower()
        t = conv / "tiles" / name
        if not name or not t.is_file():
            bad.append(f"{name or '?'}: missing")
        elif want and md5(t) != want:
            bad.append(f"{name}: md5 differs from MANIFEST")
    return f"source tiles failed MANIFEST check: {'; '.join(bad[:3])}" + (f" (+{len(bad) - 3} more)" if len(bad) > 3 else "") if bad else None


def plan(files, dest_dir):
    """(to_copy, conflicts): files missing or growable on dest vs files that differ and may not be overwritten."""
    to_copy, conflicts = [], []
    for rel, src in files.items():
        dst = dest_dir / rel
        if not dst.is_file():
            to_copy.append(rel)
        elif rel in ONCE_ONLY:
            continue
        elif dst.stat().st_size == src.stat().st_size and md5(dst) == md5(src):
            continue
        elif is_append_only(rel) and is_prefix(dst, src):
            to_copy.append(rel)
        else:
            conflicts.append(rel)
    return to_copy, conflicts


def copy_set(files, dest_dir, rels, copier):
    by_dir = {}
    for rel in rels:
        parent, _, name = rel.rpartition("/")
        by_dir.setdefault(parent, []).append(name)
    for parent, names in sorted(by_dir.items()):
        src_dir = files[f"{parent}/{names[0]}" if parent else names[0]].parent
        common.copy_files(src_dir, dest_dir / parent if parent else dest_dir, names, copier)


def free_gb(path):
    return shutil.disk_usage(path).free / 1e9


def mirror_one(conv, dest, min_free_gb, dry_run, copier, now):
    cid = conv.name
    files = list_files(conv)
    total = sum(p.stat().st_size for p in files.values())
    res = {"cid": cid, "n_files": len(files), "bytes": total, "action": "", "dest_rel": None,
           "copied": 0, "md5_ok": None, "error": None}
    prev = read_json(conv / MIRROR) or {}
    fp = fingerprint(files)
    if prev.get("md5_ok") is True and prev.get("fingerprint") == fp and prev.get("dest_rel"):
        res.update(action="skipped (unchanged)", dest_rel=prev["dest_rel"], md5_ok=True)
        return res
    base = prev.get("dest_rel") or cid
    dest_rel = base
    try:
        err = check_tiles(conv)
        if err:
            raise RuntimeError(err)
        to_copy, conflicts = plan(files, dest / base)
        if conflicts:
            dest_rel = f"{cid}__{now:%Y%m%d-%H%M%S}"
            if (dest / dest_rel).exists():
                raise RuntimeError(f"dated folder already exists: {dest_rel}")
            to_copy, conflicts = list(files), []
            res["action"] = f"changed since {base}: new folder {dest_rel}"
        else:
            res["action"] = "new" if not (dest / base).exists() else "updated"
        need = sum(files[r].stat().st_size for r in to_copy)
        res.update(dest_rel=dest_rel, copied=len(to_copy), copy_bytes=need)
        if dry_run:
            res["action"] = "dry-run: " + res["action"]
            return res
        if not dest.is_dir():
            raise RuntimeError(f"dest folder missing: {dest}")
        free = free_gb(dest)
        if free - need / 1e9 < min_free_gb:
            raise RuntimeError(f"free space on dest {free:.1f} GB minus {need / 1e9:.2f} GB to copy is below the "
                               f"floor of {min_free_gb:g} GB")
        copy_set(files, dest / dest_rel, to_copy, copier)
        bad = [rel for rel, src in files.items()
               if (rel not in ONCE_ONLY or rel in to_copy)
               and (not (dest / dest_rel / rel).is_file() or md5(dest / dest_rel / rel) != md5(src))]
        if bad:
            raise RuntimeError(f"md5 mismatch after copy: {', '.join(bad[:3])}" + (f" (+{len(bad) - 3} more)" if len(bad) > 3 else ""))
        res["md5_ok"] = True
        rec = {"dest_rel": dest_rel, "mirrored_at": now.isoformat(timespec="seconds"), "n_files": len(files),
               "bytes": total, "md5_ok": True, "error": None, "fingerprint": fp}
        hist = list(prev.get("previous") or [])
        if dest_rel != base and base not in hist:
            hist.append(base)
        if hist:
            rec["previous"] = hist
        write_json_atomic(conv / MIRROR, rec)
    except (RuntimeError, OSError) as e:
        res.update(md5_ok=False, error=str(e), action=res["action"] or "failed")
        if not dry_run:
            rec = {"dest_rel": prev.get("dest_rel"), "mirrored_at": now.isoformat(timespec="seconds"),
                   "n_files": len(files), "bytes": total, "md5_ok": False, "error": str(e)}
            if prev.get("previous"):
                rec["previous"] = prev["previous"]
            try:
                write_json_atomic(conv / MIRROR, rec)
            except OSError:
                pass
    return res


def run(source, dest, min_free_gb=20, dry_run=False, copier="auto", now=None):
    source, dest = Path(source), Path(dest)
    now = now or datetime.now()
    if not source.is_dir():
        raise RuntimeError(f"source not found: {source}")
    out = []
    for conv in sorted(p for p in source.iterdir() if p.is_dir()):
        st = read_json(conv / "status.json")
        if not isinstance(st, dict) or st.get("state") != "done":
            continue
        out.append(mirror_one(conv, dest, min_free_gb, dry_run, copier, now))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source", required=True, type=Path, help="raw_tiles folder (read only except mirror.json)")
    ap.add_argument("--dest", required=True, type=Path, help="conversions folder on the share (from the caller)")
    ap.add_argument("--min-free-gb", type=float, default=20)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--json", action="store_true", help="print a JSON summary instead of lines")
    ap.add_argument("--copier", default="auto", choices=["auto", "robocopy", "shutil"])
    a = ap.parse_args(argv)
    try:
        results = run(a.source, a.dest, a.min_free_gb, a.dry_run, a.copier)
    except (RuntimeError, OSError) as e:
        print(f"mirror_conversions: FAILED {e}", file=sys.stderr)
        return 1
    failed = [r for r in results if r["error"]]
    if a.json:
        print(json.dumps({"results": results, "failed": len(failed)}, indent=1))
    else:
        for r in results:
            tail = f"ERROR {r['error']}" if r["error"] else f"{r['copied']} copied" if r["action"] != "skipped (unchanged)" else ""
            print(f"{r['cid']}: {r['action']}, {r['n_files']} files {r['bytes'] / 1e6:.1f} MB -> {r['dest_rel']} {tail}".rstrip())
        if not results:
            print("mirror_conversions: no finished conversions, nothing to do")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
