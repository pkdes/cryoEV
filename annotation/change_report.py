"""
Change report for a label version: what applying a review package did to the labels, for a person to check
before the label version is used (the apply preview; docs: round 8c Track 2 / V4).

    python annotation/change_report.py --lv <lv folder> --images <package>/images --out <report folder>
        [--title "..."] [--max-side 900]

Reads <lv>/version.json, changes.jsonl, conflicts.csv, labels.coco.json and the base COCO named in version.json
(base_path). Read-only on all of them. Writes <out>/index.html and <out>/img/<n>.jpg: one side-by-side picture
per changed image, base labels on the left, new labels on the right, outlines coloured by what happened:

  green    added: a model find accepted          magenta  new: drawn by hand
  orange   reshaped (left: the old outline)      blue     reclassed (class changed, outline kept)
  white    unchanged

Images come from --images, named <extra.name>.jpg (the review package's images/ folder). Outlines are COCO
pixels of the label image (width/height in the COCO) and are scaled to the package copy.
Every text is escaped. Refuses an existing --out folder.
"""
from __future__ import annotations

import argparse
import collections
import csv
import html
import json
from pathlib import Path

from PIL import Image, ImageDraw

COL = {"add": (46, 204, 64), "new": (230, 60, 220), "reshape": (255, 150, 30), "reclass": (60, 140, 255),
       "reclass+reshape": (255, 150, 30), "same": (235, 235, 235)}
OP_WORD = {"add": "added (model find accepted)", "new": "drawn new", "reclass": "reclassed",
           "reshape": "reshaped", "reclass+reshape": "reclassed + reshaped", "delete": "deleted",
           "keep": "kept as is", "reject": "model find rejected (no label change)"}


def load_coco(path):
    c = json.loads(Path(path).read_text(encoding="utf-8"))
    cats = {x["id"]: x["name"] for x in c["categories"]}
    imgs = {}
    for im in c["images"]:
        name = (im.get("extra") or {}).get("name") or Path(im["file_name"]).stem
        imgs[im["id"]] = {"name": name, "w": im["width"], "h": im["height"],
                          "split": (im.get("extra") or {}).get("split"), "anns": {}}
    for a in c["annotations"]:
        if a["image_id"] in imgs:
            imgs[a["image_id"]]["anns"][a["id"]] = (cats.get(a["category_id"], "?"), a.get("segmentation") or [])
    return {v["name"]: v for v in imgs.values()}


def draw(img, rec, scale, colour_of, width):
    d = ImageDraw.Draw(img)
    for aid, (_cls, seg) in rec["anns"].items():
        col, wd = colour_of(aid)
        for ring in seg if isinstance(seg, list) else []:
            pts = [(ring[i] * scale, ring[i + 1] * scale) for i in range(0, len(ring) - 1, 2)]
            if len(pts) >= 3:
                d.line(pts + [pts[0]], fill=col, width=wd if col != COL["same"] else max(1, width - 1))


def build(lv: Path, images: Path, out: Path, title: str | None, max_side: int, edits_sha: str | None = None) -> dict:
    lv, images, out = Path(lv), Path(images), Path(out)
    ver = json.loads((lv / "version.json").read_text(encoding="utf-8"))
    if edits_sha:
        ver["edits_sha256"] = edits_sha
    rows = [json.loads(x) for x in (lv / "changes.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    conflicts = []
    if (lv / "conflicts.csv").exists():   # closed at once: the portal's preview prunes this folder later (Windows)
        with (lv / "conflicts.csv").open(encoding="utf-8") as fh:
            conflicts = list(csv.DictReader(fh))
    base = load_coco(ver["base_path"])
    new = load_coco(lv / "labels.coco.json")
    out.mkdir(parents=True, exist_ok=False)
    (out / "img").mkdir()

    applied = [r for r in rows if r.get("status") == "applied"]
    by_img = collections.defaultdict(list)
    for r in applied:
        by_img[r["image"]].append(r)
    changed_imgs = sorted(n for n, rs in by_img.items() if any(r["op"] not in ("keep", "reject") for r in rs))

    sections, missing = [], []
    for i, name in enumerate(changed_imgs, 1):
        rs = by_img[name]
        b, a = base.get(name), new.get(name)
        src = images / f"{name}.jpg"
        if not (b and a and src.exists()):
            missing.append(name)
            continue
        im = Image.open(src).convert("RGB")
        k = min(1.0, max_side / max(im.size))
        im = im.resize((round(im.size[0] * k), round(im.size[1] * k)))
        scale = im.size[0] / a["w"]
        wd = max(2, round(im.size[0] / 300))
        old_op = {r["ann_old"]: r["op"] for r in rs if r.get("ann_old") is not None and r["op"] != "keep"}
        new_op = {r["ann_new"]: r["op"] for r in rs if r.get("ann_new") is not None and r["op"] != "keep"}
        left, right = im.copy(), im.copy()
        draw(left, b, scale, lambda aid: (COL.get(old_op.get(aid), COL["same"]), wd + 1) if aid in old_op
             else (COL["same"], wd), wd)
        draw(right, a, scale, lambda aid: (COL.get(new_op.get(aid), COL["same"]), wd + 1) if aid in new_op
             else (COL["same"], wd), wd)
        gap = 12
        pair = Image.new("RGB", (im.size[0] * 2 + gap, im.size[1]), (20, 20, 20))
        pair.paste(left, (0, 0))
        pair.paste(right, (im.size[0] + gap, 0))
        fn = f"img/{i:02d}.jpg"
        pair.save(out / fn, quality=82)
        cnt = collections.Counter(r["op"] for r in rs if r["op"] != "keep")
        recl = collections.Counter(f'{r["cls_from"]} → {r["cls_to"]}' for r in rs
                                   if r["op"] in ("reclass", "reclass+reshape"))
        adds = collections.Counter(r["cls_to"] for r in rs if r["op"] in ("add", "new"))
        facts = ", ".join(f"{n} {html.escape(OP_WORD.get(op, op))}" for op, n in cnt.most_common())
        extra = []
        if adds:
            extra.append("added as: " + ", ".join(f"{html.escape(c)} {n}" for c, n in adds.most_common()))
        if recl:
            extra.append("class changes: " + ", ".join(f"{html.escape(c)} ({n})" for c, n in recl.most_common()))
        sections.append(
            f'<section class="im" id="i{i}"><h3><span class="n">{i}</span> {html.escape(name)}'
            f' <span class="sp">{html.escape(str(a.get("split") or ""))}</span></h3>'
            f'<p class="f">{facts}{"<br>" + "; ".join(extra) if extra else ""}</p>'
            f'<a href="{fn}" target="_blank" rel="noopener"><img src="{fn}" loading="lazy" '
            f'alt="Before and after labels for {html.escape(name)}"></a>'
            f'<div class="cap"><span>Before ({len(b["anns"])} objects)</span><span>After ({len(a["anns"])} objects)'
            f'</span></div></section>')

    ops = ver.get("counts_by_op", {})
    to_cls = collections.Counter(r["cls_to"] for r in applied if r["op"] in ("add", "new"))
    recl_all = collections.Counter(f'{r["cls_from"]} → {r["cls_to"]}' for r in applied
                                   if r["op"] in ("reclass", "reclass+reshape"))
    split_n = collections.Counter(new[n]["split"] for n in changed_imgs if n in new)
    trow = "".join(f"<tr><td>{html.escape(OP_WORD.get(k, k))}</td><td>{v}</td></tr>"
                   for k, v in sorted(ops.items(), key=lambda x: -x[1]) if v)
    crow = "".join(f"<tr><td>{html.escape(k)}</td><td>{v}</td></tr>" for k, v in to_cls.most_common())
    rrow = "".join(f"<tr><td>{html.escape(k)}</td><td>{v}</td></tr>" for k, v in recl_all.most_common())
    conf_rows = [r for r in rows if r.get("status") in ("conflict", "duplicate-dropped")]
    cfl = "".join(f'<li>{html.escape(r["image"])} {html.escape(r.get("dec_key") or "")}: {html.escape(r["status"])}'
                  f' ({html.escape(str(r.get("reason") or ""))})</li>' for r in conf_rows)
    t = title or f'Label changes: {ver.get("pkg")} → new labels on {ver.get("base")}'
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Label Change Report</title>
<style>
:root{{--bg:#f4f6f8;--card:#fff;--fg:#17202a;--mut:#5a6672;--line:#dbe1e7;--acc:#1d5f99}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#10161c;--card:#18212a;--fg:#e6ecf1;
--mut:#9aa8b5;--line:#2a3642;--acc:#7db8ec;color-scheme:dark}}}}
:root[data-theme="dark"]{{--bg:#10161c;--card:#18212a;--fg:#e6ecf1;--mut:#9aa8b5;--line:#2a3642;--acc:#7db8ec;
color-scheme:dark}}
body{{background:var(--bg);color:var(--fg);font:15px/1.5 "Segoe UI",system-ui,sans-serif;margin:0;padding:20px 16px 48px}}
main{{max-width:1180px;margin:0 auto;display:grid;gap:16px}}
h1{{font-size:1.45rem;margin:0}} h2{{font-size:1rem;margin:0 0 8px}} h3{{font-size:.95rem;margin:0;overflow-wrap:anywhere}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px;min-width:0}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:16px}}
table{{border-collapse:collapse;width:100%;font-size:.9rem}} td{{padding:3px 6px;border-bottom:1px solid var(--line)}}
td:last-child{{text-align:right;font-variant-numeric:tabular-nums}}
.mut,.sp,.cap{{color:var(--mut)}} .sp{{font-weight:400;font-size:.8rem;margin-left:6px}}
.n{{display:inline-block;min-width:1.8em;color:var(--acc)}}
.legend{{display:flex;flex-wrap:wrap;gap:6px 16px;font-size:.88rem}}
.legend i{{display:inline-block;width:22px;height:4px;border-radius:2px;vertical-align:middle;margin-right:6px}}
.im img{{width:100%;height:auto;border-radius:6px;display:block;margin-top:6px}}
.f{{margin:4px 0 0;font-size:.88rem}} .cap{{display:flex;justify-content:space-around;font-size:.8rem}}
.kv{{display:flex;flex-wrap:wrap;gap:4px 18px;font-size:.9rem}} .kv b{{font-weight:600}}
code{{font-size:.82rem;overflow-wrap:anywhere}}
</style></head><body><main>
<header><h1>{html.escape(t)}</h1>
<p class="mut">Nothing in this report has been applied to the real labels yet. Check it, then say go.</p></header>
<section class="card"><div class="kv">
<span>Objects: <b>{ver.get("n_annotations_base")} → {ver.get("n_annotations_out")}</b></span>
<span>Images changed: <b>{len(changed_imgs)}</b> of {ver.get("n_package_images")}
({", ".join(f"{html.escape(str(k))} {v}" for k, v in sorted(split_n.items(), key=lambda x: str(x[0])))})</span>
<span>Decisions by: <b>{html.escape(", ".join(sorted({str(r.get("who") or "?") for r in applied})))}</b></span>
<span>Test images touched: <b>{split_n.get("test", 0)}</b></span></div>
<p class="mut" style="margin:8px 0 0;font-size:.82rem">Base <code>{html.escape(str(ver.get("base")))}</code> ·
package <code>{html.escape(str(ver.get("pkg")))}</code> · decisions copy sha256
<code>{html.escape(str(ver.get("edits_sha256") or ""))}</code> · labels sha256
<code>{html.escape(str(ver.get("labels_sha256", ""))[:16])}…</code></p></section>
<div class="grid">
<section class="card"><h2>Changes</h2><table>{trow}</table></section>
<section class="card"><h2>Added objects by class</h2><table>{crow}</table>
<p class="mut" style="font-size:.82rem;margin:6px 0 0">An EV you tagged low conf is saved as “Low conf EV”.</p></section>
<section class="card"><h2>Class changes</h2><table>{rrow or '<tr><td class="mut">none</td><td></td></tr>'}</table></section>
</div>
<section class="card"><h2>Not applied ({len(conf_rows)})</h2><ul style="margin:0;padding-left:18px;font-size:.88rem">
{cfl or '<li class="mut">none</li>'}</ul></section>
<section class="card"><h2>Per image</h2><div class="legend">
<span><i style="background:rgb{COL["add"]}"></i>added (model find)</span>
<span><i style="background:rgb{COL["new"]}"></i>drawn new</span>
<span><i style="background:rgb{COL["reshape"]}"></i>reshaped</span>
<span><i style="background:rgb{COL["reclass"]}"></i>reclassed</span>
<span><i style="background:#bbb"></i>unchanged</span></div>
<p class="mut" style="font-size:.82rem;margin:6px 0 0">Left: labels before. Right: labels after. Click a picture for full size.</p></section>
{"".join(sections)}
{f'<p class="mut">No picture for: {html.escape(", ".join(missing))}</p>' if missing else ""}
</main></body></html>"""
    (out / "index.html").write_text(doc, encoding="utf-8")
    return {"images": len(sections), "missing": missing, "changed_images": len(changed_imgs)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lv", type=Path, required=True, help="label version folder (applied, or a scratch apply)")
    ap.add_argument("--images", type=Path, required=True, help="the review package's images/ folder")
    ap.add_argument("--out", type=Path, required=True, help="report folder to create")
    ap.add_argument("--title")
    ap.add_argument("--max-side", type=int, default=900)
    ap.add_argument("--edits-sha256", help="shown on the report (the frozen decisions copy's fingerprint)")
    a = ap.parse_args()
    res = build(a.lv, a.images, a.out, a.title, a.max_side, a.edits_sha256)
    print(f"[OK] {res['images']} of {res['changed_images']} changed images drawn -> {a.out / 'index.html'}")
    if res["missing"]:
        print(f"[warn] no picture: {res['missing']}")


if __name__ == "__main__":
    main()
