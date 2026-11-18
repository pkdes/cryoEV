"""Round 17c eVHarmony tools on synthetic data only (no real package): tools/evh_items.py decoys / --low / --answered,
tools/evh_carry.py, tools/evh_score.py decoys / dense / kind / --control, tools/evh_examples.py.

  - decoys: appended lines (every real line byte-identical and in place), inside the image, touching no prediction
    (any conf) and no other decoy, a candidate's shape (vertex count, size) and a candidate's conf / class; the
    decoy list only in <items>_decoys.csv; the built set's data.js, items.json and /api/meta say nothing of decoys;
  - carry: the owner's answers land on the same (image, line) of the new set, consent on its first item, every field
    kept; a second run carries nothing; the source set is unchanged;
  - score: decoys are known Nopes; dense and unsure excluded; kind counted; the control: on an unbalanced gold
    (mostly yes) always-yes passes and --control says FAILED; with decoys always-yes fails and a replay passes;
  - examples: images/ex_<id>.jpg + examples.json + data.js `examples` (no other byte of data.js changes).

    python -m unittest tools/tests/test_evh_r17c.py -v
"""
import csv
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))
import evh_carry  # noqa: E402
import evh_examples  # noqa: E402
import evh_items  # noqa: E402
import evh_practice  # noqa: E402
import evh_score  # noqa: E402
from annotation import object_set as O  # noqa: E402
from annotation import review_server as rs  # noqa: E402

V1 = REPO / "tools" / "evh" / "ls_evh_match_v1.json"
V2 = REPO / "tools" / "evh" / "ls_evh_match_v2.json"
V3 = REPO / "tools" / "evh" / "ls_evh_match_v3.json"
W, H = 300, 220


def square(cx, cy, r):
    """An 8-vertex polygon in whole pixels (as the model's)."""
    return [(cx - r, cy - r), (cx, cy - r - 2), (cx + r, cy - r), (cx + r + 2, cy), (cx + r, cy + r), (cx, cy + r + 2),
            (cx - r, cy + r), (cx - r - 2, cy)]


def make_ps(root: Path) -> Path:
    ps = root / "ps_20990101_abcdef"
    (ps / "images").mkdir(parents=True)
    pred = ps / "p" / "m1_test" / "predictions"
    pred.mkdir(parents=True)
    (ps / "request.json").write_text(json.dumps({"ps": ps.name, "model": "m1_test", "owner": "admin1",
                                                 "name": "test set", "provenance": {"steps": ["x"]}}))
    (ps / "status.json").write_text(json.dumps({"state": "done"}))
    with open(ps / "images.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "file", "orig_name", "sample_label", "nm_per_px", "source"])
        for i in (1, 2, 3):
            Image.new("L", (W, H), 120).save(ps / "images" / f"{i:04d}.png")
            w.writerow([f"{i:04d}", f"images/{i:04d}.png", f"x{i}.png", "S", "1.5", "upload"])
            lines = []
            for k in range(6):                    # conf 0.9 .. 0.15, sorted down like the model's
                cx, cy, r = 30 + 45 * k, 60 if k % 2 else 150, 10 + k
                conf = [0.9, 0.7, 0.45, 0.35, 0.28, 0.15][k]
                xy = " ".join(f"{x / W:.6f} {y / H:.6f}" for x, y in square(cx, cy, r))
                lines.append(f"{k % 2} {conf:.4f} {xy}")
            (pred / f"{i:04d}.txt").write_text("\n".join(lines) + "\n")
    return ps


def poly_mask(xy, grow=0):
    im = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(im)
    d.polygon(xy, fill=1)
    if grow:
        d.line(xy + xy[:1], fill=1, width=2 * grow + 1)
    return np.asarray(im, dtype=bool)


def datajs(pkg) -> dict:
    return json.loads((Path(pkg) / "data.js").read_text(encoding="utf-8").split("=", 1)[1].strip().rstrip(";"))


class R17c(unittest.TestCase):
    def setUp(self):
        self.t = Path(tempfile.mkdtemp(prefix="evh17c_"))
        self.addCleanup(shutil.rmtree, self.t, True)
        self.ps = make_ps(self.t)

    def items(self, *extra, out="items.csv"):
        rc = evh_items.main(["--ps", str(self.ps), "--out", str(self.t / out), "--gold", "0", *extra])
        self.assertEqual(rc, 0)
        return list(csv.DictReader(io.StringIO((self.t / out).read_text(encoding="utf-8"))))

    def test_decoys_low_and_no_marker(self):
        before = {f.name: f.read_bytes() for f in (self.ps / "p" / "m1_test" / "predictions").glob("*.txt")}
        rows = self.items("--decoys", "6", "--derive", str(self.t / "derived"), "--low", "0.10", "--n-low", "2")
        self.assertEqual(list(rows[0]), ["image", "line", "conf", "gold"])         # no decoy column
        dec = list(csv.DictReader(io.StringIO((self.t / "items_decoys.csv").read_text(encoding="utf-8"))))
        self.assertEqual(len(dec), 6)
        cands = evh_items.candidates(self.ps, 0.25, 0.5, with_cls=True)
        self.assertEqual(len(cands), 9)
        dpred = self.t / "derived" / "p" / "m1_test" / "predictions"
        for f, b in before.items():                                                # source untouched, lines kept
            self.assertEqual((self.ps / "p" / "m1_test" / "predictions" / f).read_bytes(), b)
            self.assertTrue((dpred / f).read_bytes().startswith(b))
        src_shapes = {}
        for img, ln, _c, _k in cands:
            v = (self.ps / "p" / "m1_test" / "predictions" / f"{img}.txt").read_text().splitlines()[ln].split()
            xy = [(round(float(v[i]) * W), round(float(v[i + 1]) * H)) for i in range(2, len(v), 2)]
            xs, ys = [p[0] for p in xy], [p[1] for p in xy]
            src_shapes[(img, ln)] = (len(xy), sorted((max(xs) - min(xs), max(ys) - min(ys))))
        confs, classes = {f"{c[2]:.4f}" for c in cands}, {c[3] for c in cands}
        for d in dec:
            lines = (dpred / f"{d['image']}.txt").read_text().splitlines()
            v = lines[int(d["line"])].split()
            self.assertGreaterEqual(int(d["line"]), 6)
            self.assertIn(v[1], confs)
            self.assertIn(int(v[0]), classes)
            xy = [(float(v[i]) * W, float(v[i + 1]) * H) for i in range(2, len(v), 2)]
            self.assertTrue(all(0 <= x < W and 0 <= y < H for x, y in xy))
            xs, ys = [p[0] for p in xy], [p[1] for p in xy]
            self.assertEqual((len(xy), sorted((round(max(xs) - min(xs)), round(max(ys) - min(ys))))),
                             src_shapes[(d["src_image"], int(d["src_line"]))])
            m = poly_mask(xy)
            for k, other in enumerate(lines):                                      # no overlap with anything
                if k == int(d["line"]):
                    continue
                o = other.split()
                oxy = [(float(o[i]) * W, float(o[i + 1]) * H) for i in range(2, len(o), 2)]
                self.assertFalse((m & poly_mask(oxy)).any(), (d, k))
        # --low: the 0.15 lines (3 images x 1), 2 picked; first block = decoys + low
        self.assertEqual(sum(1 for r in rows if r["conf"] == "0.1500"), 2)
        self.assertEqual(len(rows), 9 + 6 + 2)
        self.assertEqual([r["gold"] for r in rows], ["1"] * 8 + ["0"] * 9)
        self.assertEqual({(r["image"], int(r["line"])) for r in rows[:6]}, {(d["image"], int(d["line"])) for d in dec})
        # build: nothing in the package or /api/meta says decoy
        req = json.loads((self.t / "derived" / "request.json").read_text(encoding="utf-8"))
        self.assertNotEqual(req["ps"], "ps_20990101_abcdef")
        self.assertTrue((self.t / "derived" / evh_items.DERIVED).is_file())
        pkg = O.build(self.t / "derived", self.t / "items.csv", str(V2), self.t / "review", line_col="line", slug="evh",
                      seed=3, owner="admin1", app="evharmony", first_items=8)
        for f in ("data.js", "items.json", "review_package.json", "prediction_package.json"):
            self.assertNotIn("decoy", (pkg / f).read_text(encoding="utf-8").lower(), f)
        self.assertNotIn("decoy", json.dumps(rs.package_meta(pkg)).lower())
        d = datajs(pkg)
        self.assertEqual(len(d["items"]), 17)
        self.assertEqual({tuple(sorted(it)) for it in d["items"]}, {("crop", "id", "image", "obj")})
        self.assertEqual({tuple(sorted(g)) for im in d["images"] for g in im["gt"]}, {("c", "id", "p")})
        # the same inputs give the same bytes
        self.items("--decoys", "6", "--derive", str(self.t / "derived2"), "--low", "0.10", "--n-low", "2", out="b.csv")
        self.assertEqual((self.t / "b.csv").read_bytes(), (self.t / "items.csv").read_bytes())
        self.assertEqual((self.t / "b_decoys.csv").read_bytes(), (self.t / "items_decoys.csv").read_bytes())

    def answer(self, pkg, who, picks, consent=None):
        """Write answers as review_server stores them: {(image, line): value}, consent on items[0]."""
        its = O.read_items(pkg)["items"]
        by = {(i["image"], int(i["line"])): i for i in its}
        recs = {}
        t = 1794950000000
        for (img, ln), v in picks.items():
            it = by[(img, ln)]
            t += 1000
            recs.setdefault(f"{img}.png", {})[f"a:{it['obj']}:match@{who}"] = {"v": v, "t": t, "who": who, "server_t": t + 7}
        if consent:
            it = its[0]
            recs.setdefault(f"{it['image']}.png", {})[f"a:{it['obj']}:consent@{who}"] = {"v": consent, "t": t + 5,
                                                                                         "who": who, "server_t": t + 9}
        (pkg / "edits").mkdir(exist_ok=True)
        for f, dec in recs.items():
            (pkg / "edits" / f"{f}.json").write_text(json.dumps({"file": f, "pkg": pkg.name, "fmt": 1, "edit": {
                "reviewed": False, "modified": False, "dec": dec, "t": max(d["t"] for d in dec.values())}, "rx": {}}))
        return recs

    def test_carry_and_answered_first(self):
        self.items(out="old.csv")
        old = O.build(self.ps, self.t / "old.csv", str(V1), self.t / "review", line_col="line", slug="evh", seed=1,
                      owner="admin1", reviewers=["admin1"], app="evharmony")
        picks = {("0001", 2): "yes", ("0002", 3): "unsure", ("0003", 4): "no"}
        recs = self.answer(old, "admin1", picks, consent="yes")
        snap = {f.name: f.read_bytes() for f in (old / "edits").iterdir()}
        rows = self.items("--decoys", "3", "--derive", str(self.t / "derived"), "--answered", str(old), "--who", "admin1")
        self.assertEqual({(r["image"], int(r["line"])) for r in rows[:3]}, set(picks))
        new = O.build(self.t / "derived", self.t / "items.csv", str(V3), self.t / "review", line_col="line",
                      slug="evh", seed=2, owner="admin1", reviewers=["admin1"], app="evharmony", first_items=6)
        P = evh_carry.plan(old, new, "admin1")
        self.assertEqual(sum(len(v) for v in P["put"].values()), 4)
        self.assertEqual(evh_carry.main(["--from", str(old), "--to", str(new), "--who", "admin1", "--do"]), 0)
        its = O.read_items(new)["items"]
        by = {(i["image"], int(i["line"])): i for i in its}
        olddec = {k: d for f in recs.values() for k, d in f.items()}
        for (img, ln), v in picks.items():
            rec = json.loads((new / "edits" / f"{img}.png.json").read_text(encoding="utf-8"))
            d = rec["edit"]["dec"][f"a:{by[(img, ln)]['obj']}:match@admin1"]
            oit = next(i for i in O.read_items(old)["items"] if (i["image"], int(i["line"])) == (img, ln))
            self.assertEqual(d, olddec[f"a:{oit['obj']}:match@admin1"])            # v, t, who, server_t kept
            self.assertEqual(rec["fmt"], rs.RECORD_FMT)
        first = its[0]
        rec = json.loads((new / "edits" / f"{first['image']}.png.json").read_text(encoding="utf-8"))
        self.assertEqual(rec["edit"]["dec"][f"a:{first['obj']}:consent@admin1"]["v"], "yes")
        self.assertEqual(rs.check_attrs(rec["edit"], O.read_schema(new)), None)
        log = (new / "edits" / rs.ATTR_LOG).read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(log), 4)
        prov = json.loads((new / evh_carry.PROV).read_text(encoding="utf-8"))
        self.assertEqual(prov["runs"][0]["n"], 4)
        tab = O.table(new)
        self.assertEqual(sorted(r["match@admin1"] for r in tab["rows"] if r["match@admin1"]), ["no", "unsure", "yes"])
        # again: nothing new; the source untouched
        P2 = evh_carry.plan(old, new, "admin1")
        self.assertEqual((sum(len(v) for v in P2["put"].values()), len(P2["exists"])), (0, 4))
        self.assertEqual({f.name: f.read_bytes() for f in (old / "edits").iterdir()}, snap)

    def test_score_decoys_dense_kind_and_control(self):
        cols = ["item", "image", "obj", "line", "match@own", "match@r1", "match@lazy", "kind@r1", "consent@r1"]
        own = ["yes"] * 18 + ["no"] * 2                     # unbalanced: always-yes would pass on this alone
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(cols)
        decoys = set()
        for k in range(40):
            dec = k >= 20
            if dec:
                decoys.add(("0001", k))
            truth = "no" if dec else own[k]
            r1 = "dense" if k in (0, 1) else "unsure" if k == 2 else truth
            w.writerow([f"{k:05x}", "0001", f"g{k + 1}", k, "" if dec else own[k], r1, "yes",
                        "ml" if r1 == "yes" and k < 6 else "single" if k == 30 else "", "yes" if k == 0 else ""])
        rows, revs = evh_score.read_export(buf.getvalue())
        no_dec = {r["reviewer"]: r for r in evh_score.score(rows, revs, "own")}
        self.assertTrue(no_dec["lazy"]["pass"])                # the r17 gate's flaw, reproduced
        _ctl, ok = evh_score.control(rows, "own")
        self.assertFalse(ok)
        res = {r["reviewer"]: r for r in evh_score.score(rows, revs, "own", decoys=decoys)}
        self.assertEqual((res["lazy"]["n_scored"], res["lazy"]["agree_pct"], res["lazy"]["pass"]), (40, 45.0, False))
        r1 = res["r1"]
        self.assertEqual((r1["n_gold"], r1["n_dense"], r1["n_unsure"], r1["n_scored"], r1["agree_pct"]),
                         (40, 2, 1, 37, 100.0))
        self.assertEqual((r1["n_decoys"], r1["decoy_no"]), (20, 20))
        self.assertEqual((r1["kind_ml"], r1["kind_single"]), (3, 0))  # k 3-5 yes + ml; k 30 is a decoy (no)
        ctl, ok = evh_score.control(rows, "own", decoys=decoys)
        self.assertTrue(ok)
        self.assertEqual([(c["reviewer"], c["pass"]) for c in ctl],
                         [("ctl_always_yes", False), ("ctl_always_no", False), ("ctl_replay", True)])
        with tempfile.TemporaryDirectory() as t:
            e, d = Path(t) / "e.csv", Path(t) / "d.csv"
            e.write_text(buf.getvalue(), encoding="utf-8")
            d.write_text("image,line,conf,cls,src_image,src_line,flip\n" + "".join(f"0001,{k},0.3,0,0001,1,0\n"
                                                                                   for k in range(20, 40)))
            self.assertEqual(evh_score.main(["--csv", str(e), "--gold", "own", "--decoys", str(d), "--control"]), 0)
            self.assertEqual(evh_score.main(["--csv", str(e), "--gold", "own", "--control"]), 3)

    def test_examples(self):
        self.items(out="x.csv")
        pkg = O.build(self.ps, self.t / "x.csv", str(V2), self.t / "review", line_col="line", slug="evh", seed=1,
                      owner="admin1", app="evharmony", instructions="Judge the outline.")
        src = self.t / "crops"
        src.mkdir()
        with open(src / "examples.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["id", "category", "package", "note", "crop_file"])
            for i, c in (("S1", "Match – single vesicle"), ("M1", "Match – multilayer (ML)"),
                         ("D1", "Small & dense (possible lipoprotein)"), ("C1", "It's complicated"), ("N1", "")):
                Image.new("RGB", (900, 600), (90, 90, 90)).save(src / f"{i}.jpg")
                w.writerow([i, c, "pred_obj_other_1234", "", f"{i}.jpg"])
        raw = (pkg / "data.js").read_bytes()
        self.assertEqual(evh_examples.main([str(pkg), "--from", str(src), "--pick", "M1,S1,D1,C1,N1",
                                            "--caption", "S1=One membrane"]), 0)
        self.assertEqual((pkg / "data.js").read_bytes(), raw)                     # dry run
        self.assertEqual(evh_examples.main([str(pkg), "--from", str(src), "--pick", "M1,S1,D1,C1,N1",
                                            "--caption", "S1=One membrane", "--do"]), 0)
        ex = json.loads((pkg / "examples.json").read_text(encoding="utf-8"))
        self.assertEqual([(e["file"], e["cat"]) for e in ex],
                         [("images/ex_M1.jpg", "ml"), ("images/ex_S1.jpg", "single"), ("images/ex_D1.jpg", "dense"),
                          ("images/ex_C1.jpg", "unsure"), ("images/ex_N1.jpg", "nope")])
        self.assertEqual(ex[1]["caption"], "One membrane")
        self.assertEqual(ex[0]["caption"], "Multilayer (ML)")
        with Image.open(pkg / "images" / "ex_M1.jpg") as im:
            self.assertEqual(max(im.size), evh_examples.SIDE)
        d, d0 = datajs(pkg), json.loads(raw.decode("utf-8").split("=", 1)[1].strip().rstrip(";"))
        self.assertEqual([x["src"] for x in d.pop("examples")], [e["file"] for e in ex])
        self.assertEqual(d, d0)                                                    # only `examples` changed
        self.assertEqual(evh_examples.main([str(pkg), "--clear", "--do"]), 0)
        self.assertEqual((pkg / "data.js").read_bytes(), raw)
        self.assertEqual(evh_examples.main([str(pkg), "--from", str(src), "--pick", "Z9", "--do"]), 2)

    def test_practice(self):
        """r17d: picked items leave data.js items and items.json (so the export) for `practice`; nothing else in
        data.js changes; the first item (consent) and bad answers / kinds are refused; a dry run writes nothing."""
        self.items(out="x.csv")
        pkg = O.build(self.ps, self.t / "x.csv", str(V3), self.t / "review", line_col="line", slug="evh", seed=1,
                      owner="admin1", app="evharmony")
        its = O.read_items(pkg)["items"]
        a, b = its[2], its[4]
        self.answer(pkg, "admin1", {(a["image"], int(a["line"])): "yes"}, consent="yes")
        raw = (pkg / "data.js").read_bytes()
        picks = [f"{a['image']}:{a['line']}=yes/single:One ring", f"{b['image']}:{b['line']}=no"]
        self.assertEqual(evh_practice.main([str(pkg), *sum((["--pick", x] for x in picks), [])]), 0)
        self.assertEqual((pkg / "data.js").read_bytes(), raw)                     # dry run
        bad = (f"{its[0]['image']}:{its[0]['line']}=yes", f"{b['image']}:{b['line']}=maybe",
               f"{b['image']}:{b['line']}=no/single", "9999:1=yes", f"{b['image']}:{b['line']}=no:<b>")
        for x in bad:
            self.assertEqual(evh_practice.main([str(pkg), "--pick", x, "--do"]), 2, x)
        self.assertEqual(evh_practice.main([str(pkg), *sum((["--pick", x] for x in picks), []), "--do"]), 0)
        d, d0 = datajs(pkg), json.loads(raw.decode("utf-8").split("=", 1)[1].strip().rstrip(";"))
        pr = d.pop("practice")
        self.assertEqual([(x["id"], x["answer"], x.get("kind"), x.get("note")) for x in pr],
                         [(a["id"], "yes", "single", "One ring"), (b["id"], "no", None, None)])
        self.assertEqual(d["items"], [x for x in d0["items"] if x["id"] not in (a["id"], b["id"])])
        d.pop("items"), d0.pop("items")
        self.assertEqual(d, d0)                                                    # only items / practice changed
        ij = O.read_items(pkg)
        self.assertEqual([x["id"] for x in ij["items"]], [x["id"] for x in its if x["id"] not in (a["id"], b["id"])])
        self.assertEqual([x["id"] for x in ij["practice"]], [a["id"], b["id"]])
        tab = O.table(pkg)
        self.assertEqual(len(tab["rows"]), len(its) - 2)                           # never in the export (gold)
        self.assertNotIn(a["id"], {r["item"] for r in tab["rows"]})
        rec = json.loads((pkg / "review_package.json").read_text(encoding="utf-8"))
        self.assertEqual((rec["n_items"], rec["n_practice"]), (len(its) - 2, 2))
        self.assertEqual(evh_practice.main([str(pkg), "--pick", picks[1], "--do"]), 2)   # already has practice

    def test_practice_from_other_package(self):
        """r17d v2: @ picks copy the source image as images/px_<hex>.jpg and put the card (with its own image record)
        in D.practice only: D.items, D.images and the export unchanged; the source package byte-identical."""
        self.items(out="x.csv")
        pkg = O.build(self.ps, self.t / "x.csv", str(V3), self.t / "review", line_col="line", slug="evh", seed=1,
                      owner="admin1", app="evharmony")
        src = O.build(self.ps, self.t / "x.csv", str(V3), self.t / "other", line_col="line", slug="mlmv", seed=2,
                      owner="admin1")
        snap = {f.relative_to(src): f.read_bytes() for f in src.rglob("*") if f.is_file()}
        sd = datajs(src)
        img = sd["images"][0]["file"].rsplit(".", 1)[0]
        obj = f"g{sd['images'][0]['gt'][0]['id']}"
        d0, rows0 = datajs(pkg), len(O.table(pkg)["rows"])
        picks = ["--pick", f"@{img}:{obj}=yes/multilayer:Two rings", "--pick", f"@{img}@0.5,0.5,0.05=no:Ice"]
        self.assertEqual(evh_practice.main([str(pkg), *picks]), 2)                 # no --from-pkg
        self.assertEqual(evh_practice.main([str(pkg), "--from-pkg", str(src), "--pick", f"@{img}:g99999=yes"]), 2)
        self.assertEqual(evh_practice.main([str(pkg), "--from-pkg", str(src), *picks]), 0)   # dry run
        self.assertEqual(datajs(pkg), d0)
        self.assertEqual(sorted(f.name for f in (pkg / "images").glob("px_*")), [])
        self.assertEqual(evh_practice.main([str(pkg), "--from-pkg", str(src), *picks, "--do"]), 0)
        d = datajs(pkg)
        pr = d.pop("practice")
        self.assertEqual(d, d0)                                                    # items and images unchanged
        self.assertEqual(len(O.table(pkg)["rows"]), rows0)
        self.assertEqual([(x["answer"], x.get("kind"), x["note"], x["obj"]) for x in pr],
                         [("yes", "multilayer", "Two rings", "g1"), ("no", None, "Ice", "g1")])
        for x in pr:
            self.assertRegex(x["im"]["src"], r"^images/px_[0-9a-f]{8}\.jpg$")
            self.assertEqual(x["image"], x["im"]["src"][7:-4])
            self.assertTrue((pkg / x["im"]["src"]).is_file())
            self.assertEqual((x["im"]["split"], len(x["im"]["gt"])), ("", 1))
            self.assertNotIn(src.name, json.dumps(x))                             # no source name in data.js
        self.assertEqual(pr[0]["im"]["gt"][0]["p"][0], sd["images"][0]["gt"][0]["p"][0])
        ring = pr[1]["im"]["gt"][0]["p"][0]
        self.assertAlmostEqual(min(ring[0::2]), 0.45, places=3)
        prov = O.read_items(pkg)["practice"]
        self.assertEqual([(x["from"], x["image"], x["obj"]) for x in prov], [(src.name, img, obj), (src.name, img, None)])
        self.assertEqual({f.relative_to(src): f.read_bytes() for f in src.rglob("*") if f.is_file()}, snap)


class FollowUps(unittest.TestCase):
    def test_v3_follow_up_counts(self):
        cols = ["item", "image", "obj", "line", "match@own", "match@r1", "kind@r1", "outline_off@r1", "why@r1"]
        data = [("yes", "yes", "single", "yes", ""), ("no", "yes", "multilayer", "", ""), ("yes", "yes", "multilayer", "yes", ""),
                ("no", "unsure", "", "", "dense"), ("no", "unsure", "", "", "faint"), ("yes", "no", "single", "yes", "edge")]
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(cols)
        for k, (o, a, kd, ol, wy) in enumerate(data):
            w.writerow([f"{k:05x}", "0001", f"g{k + 1}", k, o, a, kd, ol, wy])
        rows, revs = evh_score.read_export(buf.getvalue())
        r = evh_score.score(rows, revs, "own", min_gold=1)[0]
        # kind / outline_off only after yes (row 5 is "no"), why only after unsure
        self.assertEqual((r["kind_single"], r["kind_multilayer"], r["outline_off"]), (1, 2, 2))
        self.assertEqual((r["why_dense"], r["why_faint"], r["why_edge"]), (1, 1, 0))
        self.assertEqual((r["n_unsure"], r["n_scored"], r["n_agree"]), (2, 4, 2))
        self.assertIn("why_dense", evh_score.to_csv([r]).splitlines()[0])


class SchemaV2(unittest.TestCase):
    def test_v2_keeps_v1_values(self):
        from labels import schemas as S
        v1, v2 = S.load_file(V1), S.load_file(V2)
        o1 = {o["v"] for o in S.attr(v1, "match")["options"]}
        o2 = {o["v"] for o in S.attr(v2, "match")["options"]}
        self.assertTrue(o1 <= o2)
        self.assertEqual(o2 - o1, {"dense"})
        self.assertEqual([o["v"] for o in S.attr(v2, "kind")["options"]], ["single", "ml", "mv", "inner"])
        self.assertFalse(S.attr(v2, "kind").get("required", False))
        self.assertEqual(S.attr(v2, "consent"), S.attr(v1, "consent"))
        v3 = S.load_file(V3)
        self.assertEqual([o["v"] for o in S.attr(v3, "match")["options"]], ["yes", "no", "unsure"])
        self.assertEqual([o["v"] for o in S.attr(v3, "kind")["options"]], ["single", "multilayer"])
        self.assertEqual([o["v"] for o in S.attr(v3, "why")["options"]], ["faint", "dense", "edge", "other"])
        self.assertIn("yes", [o["v"] for o in S.attr(v3, "outline_off")["options"]])
        self.assertEqual(S.attr(v3, "consent"), S.attr(v1, "consent"))


if __name__ == "__main__":
    unittest.main()
