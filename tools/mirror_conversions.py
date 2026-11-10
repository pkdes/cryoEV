"""Mirror finished raw-tile conversions to the desktop share ("queue-share"), verified, never deleting there.

  python tools/mirror_conversions.py --source <raw_tiles dir> --dest <share conversions dir>
                                     [--min-free-gb 20] [--dry-run] [--json]

--dest comes from the caller (the nightly .bat), never from the repo. For each conversion folder under --source
whose status.json says state "done": copy everything except input/ (raw uploads are large and temporary) into
<dest>/<cid>/. One copier for local and UNC paths (no robocopy: its file-list filter and command-line limit were
unsafe): every file is copied to <name>.mirror-part, the part's md5 is checked against the source, then os.replace
puts it in place. Nothing on dest is ever deleted. The result goes to <conv>/mirror.json (atomic write, in the
SOURCE folder): {"dest_rel","mirrored_at","n_files","bytes","md5_ok","error","last_ok_at","last_ok_dest_rel",
"tiles_unverified"?,"fingerprint","previous"?}. dest_rel is a folder name relative to --dest only.
Each tile's md5 is also checked against the MANIFEST md5 column on the source side first. A <dest_rel>/
mirror_verified.json (md5 of every file) is written on the share.

Changed since a previous mirror (a file differs from what is on dest, e.g. a CX re-stitch changes
montages/stitch.json): the whole conversion goes to a new folder <cid>__<YYYYMMDD-HHMMSS>, built as
<name>.partial and renamed after verify; the old folder is untouched. Exempt: gallery/**/*.jsonl (append only,
updated in place when the share copy is a byte prefix) and status.json, session.json, convert.log, convert.out
("latest wins": overwritten in place, left out of the fingerprint). Our own *.mirror-part / *.partial leftovers of a
crashed run are overwritten or resumed, never treated as conflicts. Unchanged conversions are skipped from a
size+mtime fingerprint. A share that is not reachable/writable, or a run already in progress (source/.mirror.lock),
exits 1 without touching any mirror.json. Free space below --min-free-gb on dest: stop, record the error.
Verify limit: md5 is read back through the SMB client, which may serve cached data. Exit 0 on success or nothing to
do, 1 on any failure; one summary line per conversion.
"""
import argparse
import csv
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from data_utils.oslock import acquire_lock  # noqa: E402
from jobs import common  # noqa: E402
from jobs.predict_set import write_json_atomic  # noqa: E402

SKIP_TOP = {"input", "mirror.json"}          # input/ = raw uploads; mirror.json = this tool's own record
LATEST_WINS = {"status.json", "session.json", "convert.log", "convert.out"}   # overwritten in place, not fingerprinted
SKIP_SUFFIX = (".tmp", ".lock", ".mirror-part", ".partial")
MIRROR = "mirror.json"
VERIFIED = "mirror_verified.json"
MTIME_TOL_NS = 2_000_000_000                 # FAT / SMB timestamp granularity


def md5(path):
    return common.md5_file(path)


def lp(path):
    """Windows long-path form for dest-side opens when the path is long."""
    s = str(path)
    if os.name != "nt" or len(s) <= 240 or s.startswith("\\\\?\\"):
        return Path(path)
    s = os.path.abspath(s)
    return Path("\\\\?\\UNC\\" + s[2:] if s.startswith("\\\\") else "\\\\?\\" + s)


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def list_files(conv):
    """{posix relpath: Path} of everything to mirror."""
    out = {}
    for p in sorted(conv.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(conv).as_posix()
        if rel.split("/")[0] in SKIP_TOP or rel.endswith(SKIP_SUFFIX):
            continue
        out[rel] = p
    return out


def fingerprint(files):
    import hashlib
    h = hashlib.sha1()
    for rel, p in files.items():
        if rel in LATEST_WINS:
            continue
        st = p.stat()
        h.update(f"{rel}\0{st.st_size}\0{st.st_mtime_ns}\n".encode())
    return h.hexdigest()


def is_append_only(rel):
    return rel.startswith("gallery/") and rel.endswith(".jsonl")


def is_prefix(small, big):
    with open(small, "rb") as a, open(big, "rb") as b:
        while True:
            x = a.read(1 << 20)
            if not x:
                return True
            if b.read(len(x)) != x:
                return False


def check_tiles(conv):
    """Source tiles vs MANIFEST md5 -> (error or None, unverified). unverified = no row carries an md5."""
    man = next((p for p in (conv / "tiles" / "MANIFEST.csv", conv / "MANIFEST.csv") if p.is_file()), None)
    if man is None:
        return "no MANIFEST.csv", False
    with open(man, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    bad, any_md5 = [], False
    for r in rows:
        name, want = r.get("output_name", ""), (r.get("md5") or "").strip().lower()
        t = conv / "tiles" / name
        if not name or not t.is_file():
            bad.append(f"{name or '?'}: missing")
        elif want:
            any_md5 = True
            if md5(t) != want:
                bad.append(f"{name}: md5 differs from MANIFEST")
    err = None
    if bad:
        err = f"source tiles failed MANIFEST check: {'; '.join(bad[:3])}" + (f" (+{len(bad) - 3} more)" if len(bad) > 3 else "")
    return err, bool(rows) and not any_md5


def same_stat(src, dst):
    a, b = src.stat(), dst.stat()
    return a.st_size == b.st_size and abs(a.st_mtime_ns - b.st_mtime_ns) <= MTIME_TOL_NS


def plan(files, dest_dir):
    """(to_copy, conflicts) vs a dest folder. Unchanged = size and mtime equal (copy2 keeps mtime)."""
    to_copy, conflicts = [], []
    for rel, src in files.items():
        dst = lp(dest_dir / rel)
        if not dst.is_file():
            to_copy.append(rel)
        elif same_stat(src, dst):
            continue
        elif rel in LATEST_WINS:
            to_copy.append(rel)
        elif is_append_only(rel) and is_prefix(dst, src):
            to_copy.append(rel)
        else:
            conflicts.append(rel)
    return to_copy, conflicts


def copy_one(src, dst):
    """Copy to <dst>.mirror-part, check its md5 against the source, then os.replace into place. Returns the md5."""
    src, dst = Path(src), lp(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_name(dst.name + ".mirror-part")
    want = md5(src)
    shutil.copy2(src, part)
    got = md5(part)
    if got != want:
        try:
            part.unlink()
        except OSError:
            pass
        raise RuntimeError(f"md5 mismatch after copy: {dst.name}")
    os.replace(part, dst)
    return want


def free_gb(path):
    return shutil.disk_usage(path).free / 1e9


def check_share(dest):
    dest = Path(dest)
    if not dest.is_dir():
        raise RuntimeError(f"dest folder not reachable: {dest}")
    probe = dest / f".mirror_probe.{os.getpid()}"
    try:
        probe.write_text("x")
        probe.unlink()
    except OSError as e:
        raise RuntimeError(f"dest folder not writable: {dest} ({e})")


def mirror_one(conv, dest, min_free_gb, dry_run, now):
    cid = conv.name
    files = list_files(conv)
    total = sum(p.stat().st_size for p in files.values())
    res = {"cid": cid, "n_files": len(files), "bytes": total, "action": "", "dest_rel": None,
           "copied": 0, "md5_ok": None, "error": None, "tiles_unverified": False}
    prev = read_json(conv / MIRROR) or {}
    fp = fingerprint(files)
    if prev.get("md5_ok") is True and prev.get("fingerprint") == fp and prev.get("dest_rel"):
        res.update(action="skipped (unchanged)", dest_rel=prev["dest_rel"], md5_ok=True)
        return res
    base = prev.get("last_ok_dest_rel") or prev.get("dest_rel") or cid
    try:
        err, unverified = check_tiles(conv)
        if err:
            raise RuntimeError(err)
        res["tiles_unverified"] = unverified
        final_dir = dest / base
        if lp(final_dir).is_dir():
            to_copy, conflicts = plan(files, final_dir)
            if conflicts:
                final_name = f"{cid}__{now:%Y%m%d-%H%M%S}"
                if lp(dest / final_name).exists():
                    raise RuntimeError(f"dated folder already exists: {final_name}")
                stale = sorted(d.name for d in dest.glob(f"{cid}__*.partial"))     # resume a crashed dated build
                stage_name = stale[-1] if stale else final_name + ".partial"
                stage_to_copy, stage_conf = plan(files, dest / stage_name)
                if stage_conf:
                    stage_name = final_name + ".partial"
                    stage_to_copy = list(files)
                to_copy, target, final_rel = stage_to_copy, dest / stage_name, final_name
                res["action"] = f"changed since {base} ({', '.join(conflicts[:2])}): new folder {final_name}"
            else:
                target, final_rel = final_dir, base
                res["action"] = "updated"
        else:
            stage_name = base + ".partial"
            to_copy, conflicts = plan(files, dest / stage_name)
            if conflicts:
                raise RuntimeError(f"{stage_name} holds files that differ from the source: {', '.join(conflicts[:3])}")
            target, final_rel = dest / stage_name, base
            res["action"] = "new"
        need = sum(files[r].stat().st_size for r in to_copy)
        res.update(dest_rel=final_rel, copied=len(to_copy), copy_bytes=need)
        if dry_run:
            res["action"] = "dry-run: " + res["action"]
            return res
        free = free_gb(dest)
        if free - need / 1e9 < min_free_gb:
            raise RuntimeError(f"free space on dest {free:.1f} GB minus {need / 1e9:.2f} GB to copy is below the "
                               f"floor of {min_free_gb:g} GB")
        digests = {}
        for rel in to_copy:
            digests[rel] = copy_one(files[rel], target / rel)
        for rel, src in files.items():            # md5 list for the share; unchanged files hash from the local source
            if rel not in digests:
                digests[rel] = md5(src)
        write_json_atomic(lp(target / VERIFIED), {
            "schema": "mirror_verified/1", "dest_rel": final_rel, "verified_at": now.isoformat(timespec="seconds"),
            "files": dict(sorted(digests.items()))})
        if target != dest / final_rel:
            os.replace(lp(target), lp(dest / final_rel))
        res["md5_ok"] = True
        stamp = now.isoformat(timespec="seconds")
        rec = {"dest_rel": final_rel, "mirrored_at": stamp, "n_files": len(files), "bytes": total, "md5_ok": True,
               "error": None, "last_ok_at": stamp, "last_ok_dest_rel": final_rel, "fingerprint": fp}
        if unverified:
            rec["tiles_unverified"] = True
        hist = list(prev.get("previous") or [])
        if final_rel != base and base not in hist:
            hist.append(base)
        if hist:
            rec["previous"] = hist
        write_record(conv, rec, res)
    except Exception as e:  # one bad conversion must not stop the night
        res.update(md5_ok=False, error=str(e), action=res["action"] or "failed")
        if not dry_run:
            ok_at = prev.get("last_ok_at") or (prev.get("mirrored_at") if prev.get("md5_ok") else None)
            ok_rel = prev.get("last_ok_dest_rel") or (prev.get("dest_rel") if prev.get("md5_ok") else None)
            rec = {"dest_rel": prev.get("dest_rel"), "mirrored_at": now.isoformat(timespec="seconds"),
                   "n_files": len(files), "bytes": total, "md5_ok": False, "error": str(e),
                   "last_ok_at": ok_at, "last_ok_dest_rel": ok_rel}
            if prev.get("previous"):
                rec["previous"] = prev["previous"]
            write_record(conv, rec, res)
    return res


def write_record(conv, rec, res):
    """mirror.json after a verified copy must not turn into md5_ok False: retry, then log and carry on."""
    for i in range(3):
        try:
            write_json_atomic(conv / MIRROR, rec)
            return
        except OSError as e:
            if i == 2:
                res["record_error"] = str(e)
                print(f"mirror_conversions: WARNING could not write {conv / MIRROR}: {e}", file=sys.stderr)
            else:
                time.sleep(0.5 * (i + 1))


def run(source, dest, min_free_gb=20, dry_run=False, now=None):
    source, dest = Path(source), Path(dest)
    now = now or datetime.now()
    if not source.is_dir():
        raise RuntimeError(f"source not found: {source}")
    if not dry_run:
        check_share(dest)
    out = []
    for conv in sorted(p for p in source.iterdir() if p.is_dir()):
        st = read_json(conv / "status.json")
        if not isinstance(st, dict) or st.get("state") != "done":
            continue
        out.append(mirror_one(conv, dest, min_free_gb, dry_run, now))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source", required=True, type=Path, help="raw_tiles folder (read only except mirror.json)")
    ap.add_argument("--dest", required=True, type=Path, help="conversions folder on the share (from the caller)")
    ap.add_argument("--min-free-gb", type=float, default=20)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--json", action="store_true", help="print a JSON summary instead of lines")
    a = ap.parse_args(argv)
    lock = None
    try:
        if not a.source.is_dir():
            raise RuntimeError(f"source not found: {a.source}")
        if not a.dry_run:
            lock = acquire_lock(a.source / ".mirror.lock", wait=0)
            if lock is None:
                raise RuntimeError("already running (source/.mirror.lock is held)")
        results = run(a.source, a.dest, a.min_free_gb, a.dry_run)
    except (RuntimeError, OSError) as e:
        print(f"mirror_conversions: FAILED {e}", file=sys.stderr)
        return 1
    finally:
        if lock is not None:
            lock.close()
    failed = [r for r in results if r["error"]]
    if a.json:
        print(json.dumps({"results": results, "failed": len(failed)}, indent=1))
    else:
        for r in results:
            tail = f"ERROR {r['error']}" if r["error"] else f"{r['copied']} copied" if r["action"] != "skipped (unchanged)" else ""
            flag = " [tiles unverified: no md5 in MANIFEST]" if r.get("tiles_unverified") else ""
            print(f"{r['cid']}: {r['action']}, {r['n_files']} files {r['bytes'] / 1e6:.1f} MB -> {r['dest_rel']} {tail}{flag}".rstrip())
        if not results:
            print("mirror_conversions: no finished conversions, nothing to do")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
