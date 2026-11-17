"""
eVHarmony "How to judge" examples (round 17c): chosen example crops -> an eVHarmony object set.

    python tools/evh_examples.py <pred_obj_*> --from <crops dir> --pick S2,M5,N1,... [--cat S2=single,...]
        [--caption S2="text",...] [--do]
    python tools/evh_examples.py <pred_obj_*> --clear [--do]

<crops dir>/examples.csv lists the candidate crops, one row each: `id` (e.g. S2), the crop's file (`file`, `crop`,
`path`, `png` or `crop_file` column, relative to the dir; else <id>.jpg / <id>.png), the category (`cat` or
`category`) and a one-line `caption` (optional; default the category's words, e.g. "multilayer (ML)"); any other
columns are provenance and stay in that file. Categories: single, ml (multilayer), mv (has vesicles inside), inner
(inside another), nope, dense (small & dense), unsure (it's complicated). A category may also be a label such as
"Match - multilayer (ML)" (matched by its words) or empty: then the id's letter decides (S single, M ml, V mv,
I inner, N nope, D dense, C / U unsure). --cat / --caption override the CSV per id.

Writes (only with --do), in the order picked:
  <pkg>/images/ex_<id>.jpg   the crop as a JPEG (longer side <= 480 px), served by review_server's images/ rule
  <pkg>/examples.json        [{file, cat, caption}] (file = images/ex_<id>.jpg)
  <pkg>/data.js              D.examples = [{src, cat, caption}] (only that key changes: every other byte of data.js
                             stays, as object_set instructions does; under the package's .meta_locks lock)
The page's sheet shows them under their answer. Use crops that are NOT items of the set (never gold).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from annotation import object_set as O  # noqa: E402
from annotation import prediction_package as pp  # noqa: E402

CATS = ("single", "ml", "mv", "inner", "nope", "dense", "unsure")
ALIAS = {"match": "single", "multilayer": "ml", "no": "nope", "complicated": "unsure", "inside": "inner",
         "small_dense": "dense", "lipoprotein": "dense"}
ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
CAPTION_MAX = 160
SIDE = 480
FILE_COLS = ("file", "crop", "path", "png", "crop_file")
WORDS = (("vesicles inside", "mv"), ("(mv)", "mv"), ("multilayer", "ml"), ("(ml)", "ml"), ("inner", "inner"),
         ("inside another", "inner"), ("single", "single"), ("nope", "nope"), ("not a vesicle", "nope"),
         ("dense", "dense"), ("lipoprotein", "dense"), ("complicated", "unsure"), ("unsure", "unsure"))
LETTER = {"S": "single", "M": "ml", "V": "mv", "I": "inner", "N": "nope", "D": "dense", "C": "unsure", "U": "unsure"}


def _cat(v: str, i: str = "") -> str:
    t = (v or "").strip().lower()
    t = ALIAS.get(t, t)
    if t in CATS:
        return t
    for w, c in WORDS:
        if w in t:
            return c
    if not t and i[:1] in LETTER:
        return LETTER[i[:1]]
    raise ValueError(f"{i}: category {v!r} is not one of {', '.join(CATS)}")


def _default_caption(v: str) -> str:
    t = " ".join((v or "").split())
    for sep in (" – ", " - ", " — "):
        if sep in t:
            t = t.split(sep, 1)[1]
    return t[:1].upper() + t[1:]


def _caption(t) -> str:
    t = " ".join(str(t or "").split())
    if any(ord(c) < 32 for c in t) or len(t) > CAPTION_MAX:
        raise ValueError(f"caption {t[:40]!r}...: at most {CAPTION_MAX} characters, one line")
    return t


def _pairs(arg) -> dict:
    out = {}
    for x in [y for y in (arg or "").split(",") if y.strip()]:
        k, _, v = x.partition("=")
        out[k.strip()] = v.strip().strip('"')
    return out


def read_source(src) -> dict:
    """{id: {file: Path, cat, caption}} from <src>/examples.csv."""
    src = Path(src)
    f = src / "examples.csv"
    if not f.is_file():
        raise FileNotFoundError(f"{f} not found")
    out = {}
    with open(f, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            i = (r.get("id") or "").strip()
            if not i:
                continue
            p = next((src / r[c].strip() for c in FILE_COLS if (r.get(c) or "").strip()), None)
            if p is None:
                p = next((src / f"{i}{e}" for e in (".jpg", ".png", ".jpeg") if (src / f"{i}{e}").is_file()), src / f"{i}.jpg")
            cat = r.get("cat") or r.get("category") or ""
            out[i] = {"file": p, "cat": cat, "caption": r.get("caption") or _default_caption(cat)}
    return out


def plan(pkg, src, pick, cats=None, captions=None) -> list:
    pkg = Path(pkg)
    data, _kw, _tail = O._split_datajs((pkg / "data.js").read_bytes())
    if data.get("kind") != "objects":
        raise ValueError(f"{pkg.name}: not an object set")
    rows = read_source(src)
    out, seen = [], set()
    for i in pick:
        if not ID_RE.match(i) or i in seen:
            raise ValueError(f"bad or repeated id {i!r}")
        seen.add(i)
        if i not in rows:
            raise ValueError(f"{i}: not in {Path(src) / 'examples.csv'}")
        r = rows[i]
        if not r["file"].is_file():
            raise FileNotFoundError(f"{i}: {r['file']} not found")
        out.append({"id": i, "from": r["file"], "file": f"images/ex_{i}.jpg",
                    "cat": _cat((cats or {}).get(i) or r["cat"], i), "caption": _caption((captions or {}).get(i, r["caption"]))})
    return out


def write(pkg, picks: list, lock_wait: float = 5.0):
    pkg = Path(pkg)
    for p in picks:
        with Image.open(p["from"]) as im:
            im = im.convert("RGB") if im.mode not in ("L", "RGB") else im.copy()
        im.thumbnail((SIDE, SIDE), Image.LANCZOS)
        im.save(pkg / p["file"], "JPEG", quality=85)
    ex = [{"file": p["file"], "cat": p["cat"], "caption": p["caption"]} for p in picks]
    lk = pp.meta_lock(pkg.parent, pkg.name, lock_wait)
    try:
        djs = pkg / "data.js"
        raw = djs.read_bytes()
        data, kw, tail = O._split_datajs(raw)
        before = dict(data)
        before.pop("examples", None)
        if ex:
            data["examples"] = [{"src": e["file"], "cat": e["cat"], "caption": e["caption"]} for e in ex]
        else:
            data.pop("examples", None)
        rest = dict(data)
        rest.pop("examples", None)
        if rest != before:
            raise ValueError("internal: data.js would change beyond `examples`; refused")
        O._write_atomic(djs, O.DATAJS_PREFIX + json.dumps(data, **kw).encode("utf-8") + tail)
        O._write_atomic(pkg / "examples.json", (json.dumps(ex, indent=1, ensure_ascii=False) + "\n").encode("utf-8"))
    finally:
        lk.close()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pkg", type=Path)
    ap.add_argument("--from", dest="src", type=Path)
    ap.add_argument("--pick", default="")
    ap.add_argument("--cat", default="", help="id=category,... overrides")
    ap.add_argument("--caption", default="", help='id="caption",... overrides')
    ap.add_argument("--clear", action="store_true")
    ap.add_argument("--do", action="store_true")
    a = ap.parse_args(argv)
    try:
        if a.clear:
            picks = []
        else:
            if not a.src or not a.pick:
                ap.error("--from and --pick are needed (or --clear)")
            picks = plan(a.pkg, a.src, [x.strip() for x in a.pick.split(",") if x.strip()], _pairs(a.cat),
                         _pairs(a.caption))
    except (ValueError, OSError) as e:
        print(f"[evh_examples] {e}", file=sys.stderr)
        return 2
    by = {}
    for p in picks:
        by[p["cat"]] = by.get(p["cat"], 0) + 1
    print(f"{a.pkg.name}: {len(picks)} examples (" + ", ".join(f"{c} {by[c]}" for c in CATS if c in by) + ")")
    if not a.do:
        print("[dry run] nothing written; add --do")
        return 0
    write(a.pkg, picks)
    print(f"[OK] {a.pkg / 'examples.json'}, images/ex_*.jpg, data.js examples")
    return 0


if __name__ == "__main__":
    sys.exit(main())
