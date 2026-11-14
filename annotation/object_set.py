"""
Object review sets (round 14, contract C4): a review package of chosen objects, labelled with a label schema's
attributes (labels/schemas.py) instead of whole images.

  build   a prediction set + an items CSV of (image, prediction line | poster particle id) -> <review root>/pred_obj_<slug>_<4hex>/
  export  the set's attribute decisions -> CSV (one column per attribute and reviewer, agree_<attr>, the blind
          columns joined back) and COCO (attributes on each annotation)
  import  read an exported CSV back (checked against the set's schema): the round trip

    python -m annotation.object_set build --ps <CryoAI-laptop>/predict/<ps> --items <items.csv> --schema <ls id | file>
        [--image-col image] (--line-col line | --particle-col particle_id) [--blind-cols c1,c2]
        [--outline-mode target|target+contained] [--slug <name>] [--seed N] [--owner <id>] [--reviewers a,b]
        [--review-root <dir>] [--do [--register --registry <dir>]]
    python -m annotation.object_set export --pkg <pkg> --out <file.csv> [--coco <file.json>]
    python -m annotation.object_set import --pkg <pkg> --csv <file.csv>

build is a dry run (checks, then the plan) unless --do; the registry line is written only with --do --register and an
explicit --registry. Built in a temp folder (.building_*) and renamed into place, so a half-built set is never served.

The package (a prediction package restricted to the listed objects; served as pred_ by multi_review_server):
  data.js        D.kind = "objects", D.schema (labels/schemas.to_page), D.outline_mode, D.items = [{id, image, obj,
                 crop[, also]}] in a seeded shuffled order: id an opaque 5-hex id, image the data.js image stem
                 (<nnnn>), obj the object's package key (g<id>), crop [x0, y0, x1, y1] normalized to the image,
                 also (target+contained only) the keys of the objects inside the target. D.images carry only the
                 listed objects (+ contained ones in target+contained mode) as labels; no candidates.
  schema.json    the label schema (review_server checks attribute decisions against it; never served)
  items.json     server-only: every items CSV column per item (the blind ones too), the line, the seed
  prediction_package.json, labels.coco.json, review_package.json (last: "complete"), .review_key, images/
Blind rule: data.js, the images and every HTTP response hold no items CSV value except the item's id, image, obj
and crop. A blind column may not be the image / line / particle column, and may not hold a value the page or
/api/meta shows anyway (a sample label, an image's file or original name): refused at build.

Object key (contract C5): (image, 0-based line index in the prediction .txt, counting every line, blank ones too).
--particle-col maps a poster particle_id to its line through <ps>/adopted/particle_lines.csv (written by
`jobs/predict_set.py adopt`, checked there against the poster's own mask areas).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import re
import secrets
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from annotation import build_review_package as brp  # noqa: E402
from annotation import object_meta as om  # noqa: E402
from annotation import prediction_package as pp  # noqa: E402
from annotation import review_server as rs  # noqa: E402
from labels import schemas as S  # noqa: E402

TOOL = "annotation/object_set.py"
PREFIX = "pred_obj_"
SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_]{0,29}$")
ITEMS = "items.json"
SCHEMA_FILE = rs.SCHEMA_FILE
OUTLINE_MODES = ("target", "target+contained")
CONTAIN = 0.99                    # target+contained: >= 99 % of the object's pixels inside the target (poster rule)
CROP_PAD, CROP_MIN_1440 = 1.6, 60  # crop side = 1.6 x the target's longer side, >= 60 px at 1440 wide (poster)
MAX_ITEMS = 2000
COL_MAX = 64
PARTICLES_CSV = Path("adopted") / "particle_lines.csv"   # jobs/predict_set.py adopt


class ObjectSetError(ValueError):
    pass


# ---------------------------------------------------------------- prediction lines (contract C5)
def read_lines(path: Path) -> list:
    """One entry per line of a prediction .txt (splitlines(): blank lines count): None for a line that is no
    polygon of >= 3 points, else {line, k, sr, p}: class id, raw conf, one normalized ring (rounded as
    build_review_package.load_predictions rounds it)."""
    out = []
    for i, ln in enumerate(Path(path).read_text().splitlines()):
        v = ln.split()
        if len(v) < 8 or (len(v) - 2) % 2:
            out.append(None)
            continue
        try:
            ring = [round(float(x), 5) for x in np.asarray(v[2:], dtype=np.float32)]
            out.append({"line": i, "k": int(v[0]), "sr": float(v[1]), "p": [ring]})
        except ValueError:
            out.append(None)
    return out


def _mask(ring, w, h):
    im = Image.new("L", (w, h), 0)
    ImageDraw.Draw(im).polygon([(ring[i] * w, ring[i + 1] * h) for i in range(0, len(ring), 2)], fill=1)
    return np.asarray(im, dtype=bool)


def contained(objs: dict, target: int, w: int, h: int) -> list:
    """Lines of the objects inside the target (>= CONTAIN of their pixels, smaller than it), at w x h."""
    tm = _mask(objs[target]["p"][0], w, h)
    ta = int(tm.sum())
    out = []
    for ln, o in objs.items():
        if ln == target:
            continue
        m = _mask(o["p"][0], w, h)
        a = int(m.sum())
        if 0 < a < ta and int((m & tm).sum()) / a >= CONTAIN:
            out.append(ln)
    return sorted(out)


def crop_box(ring, w, h) -> list:
    """[x0, y0, x1, y1] normalized: a square of CROP_PAD x the target's longer side (>= CROP_MIN_1440 px at 1440
    wide), centred on it, moved (not shrunk) to stay inside the image where it fits."""
    xs, ys = [ring[i] * w for i in range(0, len(ring), 2)], [ring[i] * h for i in range(1, len(ring), 2)]
    side = max(max(max(xs) - min(xs), max(ys) - min(ys)) * CROP_PAD, CROP_MIN_1440 * w / 1440)
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2

    def span(c, size):
        s = min(side, size)
        lo = min(max(c - s / 2, 0), size - s)
        return lo, lo + s
    x0, x1 = span(cx, w)
    y0, y1 = span(cy, h)
    return [round(x0 / w, 5), round(y0 / h, 5), round(x1 / w, 5), round(y1 / h, 5)]


# ---------------------------------------------------------------- inputs
def read_items_csv(path) -> tuple:
    raw = Path(path).read_bytes()
    text = raw.decode("utf-8-sig")
    rows = list(csv.DictReader(io.StringIO(text)))
    cols = list(csv.DictReader(io.StringIO(text)).fieldnames or [])
    return cols, rows, hashlib.sha256(raw).hexdigest()


def load_schema(schema, registry_root=None) -> dict:
    p = Path(str(schema))
    if p.suffix == ".json" and p.is_file():
        return S.load_file(p, registry_root)
    return S.load(str(schema), registry_root)


def _particle_lines(ps_dir: Path) -> dict:
    """{(ps image, particle_id): line} from <ps>/adopted/particle_lines.csv (only rows whose area matched)."""
    p = Path(ps_dir) / PARTICLES_CSV
    if not p.is_file():
        raise ObjectSetError(f"--particle-col needs {p} (written by `python -m jobs.predict_set adopt`)")
    out = {}
    with open(p, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r.get("area_match") == "1":
                out[(r["image"], str(r["particle_id"]).strip())] = int(r["line"])
    return out


def resolve(ps_dir, rows, cols, image_col, line_col=None, particle_col=None, ps_rows=None) -> tuple:
    """Each items CSV row -> (ps image id, line); [problems]. The image column may hold the ps image id (<nnnn>),
    its file name, or its original name (with or without extension)."""
    problems = []
    if image_col not in cols:
        problems.append(f"no image column {image_col!r} in the items CSV (columns: {', '.join(cols)})")
    key_col = line_col or particle_col
    if (line_col is None) == (particle_col is None):
        problems.append("give exactly one of --line-col or --particle-col")
    elif key_col not in cols:
        problems.append(f"no column {key_col!r} in the items CSV")
    if problems:
        return [], problems
    names = {}
    for r in ps_rows:
        for n in (r["image"], Path(r["file"]).name, r.get("orig_name") or "", Path(r.get("orig_name") or "").stem):
            if n:
                names.setdefault(n, r["image"])
    plines = _particle_lines(ps_dir) if particle_col else {}
    out, seen = [], {}
    for n, r in enumerate(rows, 2):
        img = names.get((r.get(image_col) or "").strip())
        if img is None:
            problems.append(f"row {n}: image {r.get(image_col)!r} is not in the prediction set")
            continue
        val = (r.get(key_col) or "").strip()
        if particle_col:
            line = plines.get((img, val))
            if line is None:
                problems.append(f"row {n}: particle {val!r} of image {img} has no checked line "
                                f"({PARTICLES_CSV.as_posix()})")
                continue
        else:
            if not val.isdigit():
                problems.append(f"row {n}: line {val!r} is not a whole number >= 0")
                continue
            line = int(val)
        if (img, line) in seen:
            problems.append(f"row {n}: the same object as row {seen[(img, line)]} ({img}, line {line})")
            continue
        seen[(img, line)] = n
        out.append((img, line))
    return out, problems


# ---------------------------------------------------------------- build
def plan(ps_dir, items_csv, schema, image_col="image", line_col=None, particle_col=None, blind_cols=(),
         outline_mode="target", registry_root=None) -> dict:
    """Every check build makes, nothing written. -> the plan (raises ObjectSetError with every problem)."""
    ps_dir = Path(ps_dir)
    req, ps_id, model, ps_rows, pred_dir = pp._ps_inputs(ps_dir)
    try:
        doc = load_schema(schema, registry_root)
    except S.SchemaError as e:
        raise ObjectSetError(str(e)) from e
    if not doc.get("object_attrs"):
        raise ObjectSetError(f"{doc['id']} has no object attributes: nothing to label on an object card")
    if outline_mode not in OUTLINE_MODES:
        raise ObjectSetError(f"outline mode must be one of {OUTLINE_MODES}")
    cols, rows, sha = read_items_csv(items_csv)
    problems = []
    if not rows:
        problems.append("the items CSV has no rows")
    if len(rows) > MAX_ITEMS:
        problems.append(f"{len(rows)} items: at most {MAX_ITEMS}")
    if len(set(cols)) != len(cols) or any(not c or len(c) > COL_MAX for c in cols):
        problems.append(f"items CSV columns must be unique names of 1-{COL_MAX} characters")
    blind = list(dict.fromkeys(blind_cols or ()))
    for c in blind:
        if c not in cols:
            problems.append(f"blind column {c!r} is not in the items CSV")
        elif c in (image_col, line_col, particle_col):
            problems.append(f"blind column {c!r} is the object's key: it can't be hidden")
    if problems:
        raise ObjectSetError("refused:\n  - " + "\n  - ".join(problems))
    keys, problems = resolve(ps_dir, rows, cols, image_col, line_col, particle_col, ps_rows)
    shown = set()
    for r in ps_rows:
        shown |= {r["image"], r["file"], Path(r["file"]).name, r.get("orig_name") or "",
                  Path(r.get("orig_name") or "").stem, r.get("sample_label") or "",
                  pp._display_label(r.get("sample_label"))}
    shown.discard("")
    for c in blind:
        hit = sorted({(r.get(c) or "").strip() for r in rows} & shown)
        if hit:
            problems.append(f"blind column {c!r} holds values the page shows anyway (sample labels or image names, "
                            f"e.g. {hit[0]!r}): not blind; drop it from --blind-cols")
    by_image = {}
    lines_of = {}
    for (img, line), r in zip(keys, rows) if not problems else ():
        if img not in lines_of:
            lines_of[img] = read_lines(pred_dir / f"{img}.txt") if (pred_dir / f"{img}.txt").is_file() else None
        objs = lines_of[img]
        if objs is None:
            problems.append(f"image {img}: no predictions file")
        elif line >= len(objs) or objs[line] is None:
            problems.append(f"image {img}: line {line} is not a polygon in its predictions file")
        else:
            by_image.setdefault(img, []).append(line)
    if problems:
        raise ObjectSetError(f"refused ({len(problems)} problem(s)):\n  - " + "\n  - ".join(problems[:30]))
    return {"ps_dir": ps_dir, "ps_id": ps_id, "model": model, "req": req, "ps_rows": ps_rows, "pred_dir": pred_dir,
            "schema": doc, "cols": cols, "rows": rows, "sha256": sha, "keys": keys, "blind": blind,
            "outline_mode": outline_mode, "lines_of": lines_of, "by_image": by_image,
            "image_col": image_col, "key_col": line_col or particle_col, "items_name": Path(items_csv).name,
            "key_kind": "line" if line_col else "particle"}


def build(ps_dir, items_csv, schema, review_root, image_col="image", line_col=None, particle_col=None,
          blind_cols=(), outline_mode="target", slug=None, seed=None, owner=None, reviewers=(),
          registry_root=None, width=1440, quality=72, built_by="cli") -> Path:
    """Build the object set (module doc) under review_root. -> its folder. Registering is separate (register())."""
    P = plan(ps_dir, items_csv, schema, image_col, line_col, particle_col, blind_cols, outline_mode, registry_root)
    owner = owner or P["req"].get("owner")
    if not pp.OWNER_RE.match(owner or ""):
        raise ObjectSetError(f"bad owner {owner!r}")
    revs = list(dict.fromkeys(reviewers or ()))
    bad = [r for r in revs if not pp.REVIEWER_ID_RE.fullmatch(r)]
    if bad:
        raise ObjectSetError(f"bad reviewer ids {bad}")
    slug = slug or P["schema"]["name"]
    if not SLUG_RE.match(slug):
        raise ObjectSetError(f"slug {slug!r} must match {SLUG_RE.pattern}")
    seed = secrets.randbits(32) if seed is None else int(seed)
    review_root = Path(review_root)
    for _ in range(50):
        pid = f"{PREFIX}{slug}_{secrets.token_hex(2)}"
        if not (review_root / pid).exists():
            break
    else:
        raise ObjectSetError("no free package id")
    final = review_root / pid
    tmp = review_root / f".building_{pid}_{secrets.token_hex(4)}"
    try:
        _build_into(tmp, final, P, owner, revs, seed, width, quality, built_by)
        if final.exists():
            raise FileExistsError(f"{final} appeared during the build")
        tmp.rename(final)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
    print(f"[OK] {pid} -> {final}")
    return final


def _build_into(out, final, P, owner, reviewers, seed, width, quality, built_by):
    (out / "images").mkdir(parents=True)
    doc, mode = P["schema"], P["outline_mode"]
    print(f"[steps] object set: {len(P['keys'])} objects on {len(P['by_image'])} images; schema {doc['id']}; "
          f"outlines {mode}; images grayscale JPEG <= {width} px, q{quality}; order shuffled (seed in {ITEMS})")
    rows_by_image = {r["image"]: r for r in P["ps_rows"]}
    coco = {"info": {"description": f"object set from prediction set {P['ps_id']}, model {P['model']}"},
            "categories": [{"id": k, "name": n} for k, n in enumerate(pp.PRED_CLASSES)], "images": [],
            "annotations": []}
    images, meta_images, key_of, also_of, size_of, ann_id = [], {}, {}, {}, {}, 0
    for n, img in enumerate(sorted(P["by_image"]), 1):
        r = rows_by_image[img]
        src = P["ps_dir"] / r["file"]
        with Image.open(src) as im0:
            w, h = im0.size
            im = im0.convert("L")
        if im.width > width:
            im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
        jpg = f"images/{img}.jpg"
        im.save(out / jpg, "JPEG", quality=quality)
        size_of[img] = (w, h)
        objs = {o["line"]: o for o in P["lines_of"][img] if o is not None}
        targets = sorted(P["by_image"][img])
        show = set(targets)
        if mode == "target+contained":
            for t in targets:
                also_of[(img, t)] = contained(objs, t, im.width, im.height)
                show |= set(also_of[(img, t)])
        anns, lab_meta = [], []
        for ln in sorted(show):
            o = objs[ln]
            ann_id += 1
            key_of[(img, ln)] = f"g{ann_id}"
            seg = pp._flat_px(o["p"][0], w, h)
            xs, ys = seg[0::2], seg[1::2]
            anns.append({"id": ann_id, "image_id": n, "category_id": o["k"] if o["k"] in (0, 1) else 0,
                         "segmentation": [seg], "iscrowd": 0, "score": o["sr"], "line": ln,
                         "bbox": [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]})
            lab_meta.append({"id": ann_id, "cls": o["k"], "conf": o["sr"], "line": ln})
        coco["images"].append({"id": n, "file_name": Path(src).name, "width": w, "height": h})
        coco["annotations"] += anns
        gt = brp.gt_items(anns, dict(enumerate(pp.PRED_CLASSES)), w, h)
        file = Path(src).name
        images.append({"file": file, "split": pp._display_label(r.get("sample_label")), "w": w, "h": h, "src": jpg,
                       "gt": gt, "pr": [], "fg": 0, "fp": 0})
        meta_images[img] = {"file": file, "w": w, "h": h, "sample_label": r.get("sample_label"),
                            "labels": lab_meta, "candidates": []}

    rng = random.Random(seed)
    order = list(range(len(P["keys"])))
    rng.shuffle(order)
    used, items, pub = set(), [], []
    for k in order:
        img, ln = P["keys"][k]
        while True:
            iid = f"{rng.getrandbits(20):05x}"
            if iid not in used:
                used.add(iid)
                break
        w, h = size_of[img]
        crop = crop_box(P["lines_of"][img][ln]["p"][0], w, h)
        obj = key_of[(img, ln)]
        it = {"id": iid, "image": img, "obj": obj, "crop": crop}
        if mode == "target+contained":
            it["also"] = [key_of[(img, c)] for c in also_of[(img, ln)]]
        pub.append(it)
        items.append({**it, "line": ln, "row": dict(P["rows"][k])})

    built = pp._now()
    pid = final.name
    data = {"id": pid, "kind": "objects", "prediction_set": P["ps_id"], "model": P["model"], "owner": owner,
            "source": f"prediction_set:{P['ps_id']}", "run": P["model"], "created": built,
            "categories": pp.PRED_CLASSES + pp.EXTRA_CLASSES, "schema": S.to_page(doc), "outline_mode": mode,
            "items": pub, "images": images}
    brp.write_packages(out, data, part_size=0, zip_it=False)
    (out / SCHEMA_FILE).write_bytes(S.canonical_bytes(doc))
    pp._write_json(out / ITEMS, {
        "schema": "object_items/1", "package": pid, "seed": seed, "source": {
            "file": P["items_name"], "sha256": P["sha256"]},
        "columns": P["cols"], "image_col": P["image_col"], "key_col": P["key_col"], "key_kind": P["key_kind"],
        "blind_cols": P["blind"], "outline_mode": mode, "label_schema": doc["id"], "items": items})
    pp._write_json(out / "labels.coco.json", coco, indent=None)
    pp._write_json(out / pp.META, {"schema": "prediction_package/1", "kind": "objects", "package": pid,
                                   "prediction_set": P["ps_id"], "model": P["model"], "owner": owner,
                                   "label_conf": pp.LABEL_CONF, "cand_conf": pp.CAND_CONF, "built": built,
                                   "ps_dir": str(Path(P["ps_dir"]).resolve()), "images": meta_images})
    (out / ".review_key").write_text(secrets.token_urlsafe(18), encoding="utf-8")
    pp._write_json(out / pp.RECORD, registry_record(final, data, built, doc, P, owner, reviewers, built_by))  # last
    print(f"[OK] {pid}: {len(pub)} objects on {len(images)} images")


def registry_record(pkg, data, built, doc, P, owner, reviewers, built_by="cli") -> dict:
    """The review_package/1 line of an object set (C4; the schema id is `label_schema`: `schema` is the envelope)."""
    return {"schema": "review_package/1", "id": data["id"], "logged": built, "recorded_by": TOOL,
            "path": str(Path(pkg).resolve()), "kind": "objects", "parent": None, "base": P["ps_id"],
            "base_kind": "prediction_set", "model": P["model"], "run": None, "label_schema": doc["id"],
            "n_items": len(data["items"]), "n_images": len(data["images"]), "blind_cols": list(P["blind"]),
            "outline_mode": P["outline_mode"], "reviewers": list(reviewers), "owner": owner, "built_by": built_by,
            "built": built, "tool": TOOL, "status": "built", "prediction_set": P["ps_id"]}


def register(pkg, registry_root) -> dict:
    """Append the set's review_package line (its review_package.json) to an explicitly given registry."""
    from registry import store
    rec = json.loads((Path(pkg) / pp.RECORD).read_text(encoding="utf-8"))
    return store.append(rec, root=Path(registry_root), new_only=True)


# ---------------------------------------------------------------- export / import
def read_items(pkg) -> dict:
    return json.loads((Path(pkg) / ITEMS).read_text(encoding="utf-8"))


def read_schema(pkg) -> dict:
    return json.loads((Path(pkg) / SCHEMA_FILE).read_text(encoding="utf-8"))


def _enc(v) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def _dec(v: str, a: dict):
    if v == "":
        return None
    if a.get("kind") == "bool":
        if v not in ("true", "false"):
            raise ObjectSetError(f"{a['id']}: {v!r} is not true / false")
        return v == "true"
    if not S.value_ok(a, v):
        raise ObjectSetError(f"{a['id']}: {v!r} is not a value of this attribute")
    return v


def table(pkg) -> dict:
    """The export as data: {columns, rows (dicts of typed values)}. Columns (C4): item, image, obj, line,
    <attr>@<rid> (schema order x reviewers sorted), agree_<attr>, reviewers, last_t, history_n, schema_id,
    schema_version, <blind cols>."""
    pkg = Path(pkg)
    items, doc, data = read_items(pkg), read_schema(pkg), om.read_datajs(pkg)
    dec = om.decisions(pkg, data)
    hist = om.history(pkg)
    attrs = [a["id"] for a in doc.get("object_attrs") or []]
    rids = sorted({k.split("@", 1)[1] for d in dec.values() for k in d} | set(om.package_reviewers(pkg)))
    blind = list(items.get("blind_cols") or [])
    cols = (["item", "image", "obj", "line"] + [f"{a}@{r}" for a in attrs for r in rids]
            + [f"agree_{a}" for a in attrs] + ["reviewers", "last_t", "history_n", "schema_id", "schema_version"]
            + blind)
    n_hist = {}
    file_of = {Path(i["file"]).stem: i["file"] for i in data["images"]}
    for h in hist:
        p = rs.parse_attr_key(h.get("key"))
        if p and p[0] is not None:
            n_hist[(Path(str(h.get("file"))).stem, p[0])] = n_hist.get((Path(str(h.get("file"))).stem, p[0]), 0) + 1
    rows = []
    for it in items["items"]:
        d = dec.get(it["id"], {})
        row = {"item": it["id"], "image": it["image"], "obj": it["obj"], "line": it["line"]}
        who = set()
        last = None
        for a in attrs:
            vals = []
            for r in rids:
                x = d.get(f"{a}@{r}")
                v = x.get("v") if isinstance(x, dict) else None
                row[f"{a}@{r}"] = v
                if x is not None:
                    t = x.get("server_t") if rs._num(x.get("server_t")) else x.get("t")
                    last = t if last is None or (rs._num(t) and t > last) else last
                if v is not None:
                    vals.append(v)
                    who.add(r)
            row[f"agree_{a}"] = None if len(vals) < 2 else (len(set(map(_enc, vals))) == 1)
        row["reviewers"] = ";".join(sorted(who))
        row["last_t"] = int(last) if rs._num(last) else None
        row["history_n"] = n_hist.get((it["image"], it["obj"]), 0) if it["image"] in file_of else 0
        row["schema_id"], row["schema_version"] = doc["id"], doc["version"]
        for c in blind:
            row[c] = it["row"].get(c, "")
        rows.append(row)
    return {"columns": cols, "rows": rows, "schema": doc}


def to_csv(tab: dict) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(tab["columns"])
    for r in tab["rows"]:
        w.writerow([_enc(r.get(c)) for c in tab["columns"]])
    return buf.getvalue()


def import_csv(path_or_text, schema: dict) -> dict:
    """An exported CSV -> the same table (typed values), checked against the schema: the attribute columns'
    values, agree_* (true / false / empty), schema_id and version. Raises ObjectSetError."""
    text = Path(path_or_text).read_text(encoding="utf-8-sig") if isinstance(path_or_text, Path) else path_or_text
    rd = csv.reader(io.StringIO(text))
    cols = next(rd, None)
    if not cols or cols[:4] != ["item", "image", "obj", "line"]:
        raise ObjectSetError("not an object set export (first columns item, image, obj, line)")
    attr = {a["id"]: a for a in schema.get("object_attrs") or []}
    rows = []
    for n, vals in enumerate(rd, 2):
        if len(vals) != len(cols):
            raise ObjectSetError(f"row {n}: {len(vals)} values for {len(cols)} columns")
        r = {}
        for c, v in zip(cols, vals):
            if c == "line" or c in ("last_t", "history_n", "schema_version"):
                r[c] = int(v) if v != "" else None
            elif c.startswith("agree_") and c[6:] in attr:
                r[c] = None if v == "" else _dec(v, {"id": c, "kind": "bool"})
            elif "@" in c and c.split("@", 1)[0] in attr:
                r[c] = _dec(v, attr[c.split("@", 1)[0]])
            else:
                r[c] = v
        if r.get("schema_id") != schema.get("id") or r.get("schema_version") != schema.get("version"):
            raise ObjectSetError(f"row {n}: schema {r.get('schema_id')} v{r.get('schema_version')} is not "
                                 f"{schema.get('id')} v{schema.get('version')}")
        rows.append(r)
    return {"columns": cols, "rows": rows, "schema": schema}


def coco(pkg) -> dict:
    """labels.coco.json's targets with the decisions: attributes {"<attr>@<rid>": v} and attributes_schema on
    each item's annotation (plus item and line). Blind columns are not in it."""
    pkg = Path(pkg)
    tab, items = table(pkg), read_items(pkg)
    base = json.loads((pkg / "labels.coco.json").read_text(encoding="utf-8"))
    doc = tab["schema"]
    by_key = {}
    data = om.read_datajs(pkg)
    img_id = {Path(i["file_name"]).stem: i["id"] for i in base["images"]}
    for a in base["annotations"]:
        by_key[(a["image_id"], a["id"])] = a
    attrs = [c for c in tab["columns"] if "@" in c]
    anns = []
    for it, row in zip(items["items"], tab["rows"]):
        a = dict(by_key[(img_id[it["image"]], int(it["obj"][1:]))])
        a["item"] = it["id"]
        a["attributes"] = {c: row[c] for c in attrs if row[c] is not None}
        a["attributes_schema"] = {"id": doc["id"], "version": doc["version"]}
        anns.append(a)
    return {"info": {"description": f"object set {data['id']} with attributes ({doc['id']})"},
            "categories": base["categories"], "images": base["images"],
            "annotations": sorted(anns, key=lambda a: a["id"])}


# ---------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="items CSV + prediction set -> object set (dry run unless --do)")
    b.add_argument("--ps", required=True, type=Path)
    b.add_argument("--items", required=True, type=Path)
    b.add_argument("--schema", required=True, help="label schema id (built-in or registered) or a .json file")
    b.add_argument("--image-col", default="image")
    g = b.add_mutually_exclusive_group(required=True)
    g.add_argument("--line-col")
    g.add_argument("--particle-col")
    b.add_argument("--blind-cols", default="", help="comma-separated items CSV columns kept off the page")
    b.add_argument("--outline-mode", choices=OUTLINE_MODES, default="target")
    b.add_argument("--slug")
    b.add_argument("--seed", type=int)
    b.add_argument("--owner")
    b.add_argument("--reviewers", default="", help="comma-separated portal ids")
    b.add_argument("--review-root", type=Path)
    b.add_argument("--registry", type=Path, help="registry folder (only with --register; never a default)")
    b.add_argument("--do", action="store_true", help="build it (default: a dry run that writes nothing)")
    b.add_argument("--register", action="store_true", help="with --do: append the review_package line to --registry")
    e = sub.add_parser("export", help="decisions -> CSV (+ COCO)")
    e.add_argument("--pkg", required=True, type=Path)
    e.add_argument("--out", required=True, type=Path)
    e.add_argument("--coco", type=Path)
    i = sub.add_parser("import", help="read an exported CSV back, checked against the set's schema")
    i.add_argument("--pkg", required=True, type=Path)
    i.add_argument("--csv", required=True, type=Path)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "build":
            blind = [c.strip() for c in a.blind_cols.split(",") if c.strip()]
            revs = [c.strip() for c in a.reviewers.split(",") if c.strip()]
            if a.register and not (a.do and a.registry):
                raise ObjectSetError("--register needs --do and an explicit --registry")
            if not a.do:
                P = plan(a.ps, a.items, a.schema, a.image_col, a.line_col, a.particle_col, blind, a.outline_mode,
                         a.registry)
                print(f"DRY RUN, nothing written: {len(P['keys'])} objects on {len(P['by_image'])} images of "
                      f"{P['ps_id']}, schema {P['schema']['id']}, blind {P['blind'] or 'none'}, outlines "
                      f"{P['outline_mode']}. Run again with --do to build"
                      + (" (and --register --registry <dir> to record it)." if not a.register else "."))
                return 0
            if a.review_root is None:
                from data_utils import config
                a.review_root = config.laptop_root() / "review"
            pkg = build(a.ps, a.items, a.schema, a.review_root, a.image_col, a.line_col, a.particle_col, blind,
                        a.outline_mode, a.slug, a.seed, a.owner, revs, a.registry)
            if a.register:
                register(pkg, a.registry)
                print(f"[OK] registered {pkg.name} in {a.registry}")
        elif a.cmd == "export":
            tab = table(a.pkg)
            a.out.write_text(to_csv(tab), encoding="utf-8", newline="")
            print(f"[OK] {a.out}: {len(tab['rows'])} items, {len(tab['columns'])} columns")
            if a.coco:
                a.coco.write_text(json.dumps(coco(a.pkg)), encoding="utf-8")
                print(f"[OK] {a.coco}")
        else:
            tab = import_csv(a.csv, read_schema(a.pkg))
            print(f"[OK] {a.csv}: {len(tab['rows'])} items, schema {tab['schema']['id']} v{tab['schema']['version']}")
    except (ObjectSetError, ValueError, OSError, S.SchemaError) as ex:
        print(f"[object_set] {ex}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
