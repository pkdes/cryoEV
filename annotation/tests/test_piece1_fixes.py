"""
Tests for the piece-1 scripts: resolve_review_edits.py, build_review_package.py (flags, parts) and
backfill_tracking.py, on small synthetic fixtures (squares on a 1000 x 1000 image), plus read-only checks
on the real outputs when they exist.

  python -m unittest annotation.tests.test_piece1 -v          (from the repo root)
  python -m unittest discover -s annotation/tests -p "test_piece1.py" -v

PIECE1_SRC=<folder> runs the same tests against another copy of the scripts (e.g. the pre-fix version),
to show which tests a fix made pass.
"""
import csv
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
SRC = Path(os.environ.get("PIECE1_SRC") or REPO / "annotation")
sys.path.insert(0, str(REPO / "training"))
sys.path.insert(0, str(REPO / "annotation"))  # review_server for a pre-fix copy that doesn't carry one
REAL_SEEDS = [p / "CryoAI-laptop" / "september_edits" / "v2" / "seed_edits" for p in REPO.parents]
REAL_PULL = [p / "CryoAI-laptop" for p in REPO.parents]
W = H = 1000


def load(name):
    spec = importlib.util.spec_from_file_location(f"piece1_{name}", SRC / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(SRC))
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(SRC))
    return mod


def sq(x, y, s):
    """Normalized square ring list."""
    return [[x, y, x + s, y, x + s, y + s, x, y + s]]


def run(args, cwd=None):
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(REPO / "training"), str(SRC)])}
    r = subprocess.run([sys.executable, *map(str, args)], capture_output=True, text=True, env=env, cwd=cwd)
    if r.returncode:
        raise AssertionError(f"exit {r.returncode}\n{r.stdout}\n{r.stderr}")
    return r.stdout


class ResolveFixture:
    """A September-style package + edits + a current COCO in a temp folder; run the resolver's CLI."""

    def __init__(self, tmp):
        self.tmp = Path(tmp)
        self.images, self.edits, self.current, self.dups, self.preds = [], {}, [], {}, {}

    def image(self, stem, gt=(), pr=()):
        file = f"{stem}.rf.abc.jpg"
        self.images.append({"file": file, "split": "train", "w": W, "h": H, "src": "", "fg": 0, "fp": 0,
                            "gt": [{"id": i, "c": c, "p": p} for i, c, p in gt],
                            "pr": [{"s": 0.9, "p": p} for p in pr]})
        return file

    def edit(self, file, dec, **extra):
        self.edits[file] = {"dec": {k: {"v": v, "t": 1000 + n, "who": "test", **({"p": p} if p else {})}
                                    for n, (k, (v, p)) in enumerate(dec.items())}, "t": 2000, "modified": True,
                            "reviewed": False, **extra}

    def now(self, stem, labels):
        self.current.append((stem, labels))

    def resolve(self, seed=False):
        pkg = self.tmp / "pkg"
        (pkg / "edits").mkdir(parents=True, exist_ok=True)
        (pkg / "data.js").write_text("window.REVIEW_DATA = " + json.dumps({"images": self.images}) + ";\n")
        for n, (f, e) in enumerate(self.edits.items()):
            (pkg / "edits" / f"{n}.json").write_text(json.dumps({"file": f, "edit": e}))
        cats, imgs, anns = {}, [], []
        for i, (stem, labels) in enumerate(self.current, 1):
            imgs.append({"id": i, "file_name": f"{stem}.png", "width": W, "height": H, "extra": {"name": stem}})
            for c, p in labels:
                cid = cats.setdefault(c, len(cats) + 1)
                anns.append({"id": len(anns) + 1, "image_id": i, "category_id": cid,
                             "segmentation": [[v * W for v in p[0]]]})
        coco = self.tmp / "current.coco.json"
        coco.write_text(json.dumps({"images": imgs, "annotations": anns,
                                    "categories": [{"id": v, "name": k} for k, v in cats.items()]}))
        dup = self.tmp / "dups.csv"
        with open(dup, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["stem", "duplicate_of"])
            w.writerows(self.dups.items())
        out = self.tmp / "out"
        args = [SRC / "resolve_review_edits.py", "--package", pkg, "--current-coco", coco, "--dup-csv", dup, "--out", out]
        if seed:
            pd = self.tmp / "preds"
            pd.mkdir(exist_ok=True)
            for stem, polys in self.preds.items():
                (pd / f"{stem}.txt").write_text("".join("0 0.9000 " + " ".join(f"{v:.6f}" for v in p[0]) + "\n"
                                                        for p in polys))
            args += ["--pred-dir", pd, "--seed-dir", out / "seeds"]
        run(args)
        with open(out / "resolution.csv", newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        return rows, out

    @staticmethod
    def status(rows, stem, key):
        hit = [r["status"] for r in rows if r["stem"] == stem and r["key"] == key]
        assert len(hit) == 1, (stem, key, [(r["stem"], r["key"]) for r in rows])
        return hit[0]


class TestResolve(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.fx = ResolveFixture(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    # MUST 1: adds on one image are compared with each other
    def test_duplicate_adds_collapse_or_conflict(self):
        prs = [sq(0.1, 0.1, 0.1), sq(0.102, 0.1, 0.1), sq(0.5, 0.5, 0.1), sq(0.502, 0.5, 0.1)]
        f = self.fx.image("A_png", pr=prs)
        self.fx.edit(f, {"p0": ("add:EV", None), "p1": ("add:EV", None), "p2": ("add:EV", None),
                         "p3": ("add:MV", None)})
        self.fx.now("A_png", [])
        rows, _ = self.fx.resolve()
        same = sorted(ResolveFixture.status(rows, "A_png", k) for k in ("p0", "p1"))
        self.assertEqual(same, ["already-done", "applied-ready"])
        self.assertEqual(ResolveFixture.status(rows, "A_png", "p2"), "conflict")
        self.assertEqual(ResolveFixture.status(rows, "A_png", "p3"), "conflict")

    def test_editor_copy_of_an_add_collapses(self):
        """The square237_hole31 pattern: an accepted prediction plus a src-less Editor-list copy of it."""
        p = sq(0.2, 0.2, 0.1)
        f = self.fx.image("B_png", gt=[(1, "EV", sq(0.7, 0.7, 0.1))], pr=[p])
        self.fx.edit(f, {"p0": ("add:EV", None)}, gt=[{"id": 1, "c": "EV", "p": sq(0.7, 0.7, 0.1)},
                                                      {"id": None, "c": "EV", "p": sq(0.201, 0.2, 0.1)}], gtT=900)
        self.fx.now("B_png", [("EV", sq(0.7, 0.7, 0.1))])
        rows, _ = self.fx.resolve()
        adds = sorted((r["key"], r["status"]) for r in rows if r["op"] == "add")
        self.assertEqual(adds, [("editor", "already-done"), ("p0", "applied-ready")])

    def test_no_two_seeded_outlines_overlap(self):
        prs = [sq(0.1, 0.1, 0.1), sq(0.102, 0.1, 0.1), sq(0.5, 0.5, 0.1)]
        f = self.fx.image("C_png", pr=prs)
        self.fx.edit(f, {"p0": ("add:EV", None), "p1": ("add:EV", None), "p2": ("add:EV", None)})
        self.fx.now("C_png", [])
        self.fx.preds["C_png"] = prs
        _, out = self.fx.resolve(seed=True)
        rr = load("resolve_review_edits")
        for rec in (json.loads(p.read_text()) for p in (out / "seeds").glob("*.json")):
            polys = [d["p"] for d in rec["edit"]["dec"].values() if d.get("p")]
            for i, a in enumerate(polys):
                for b in polys[i + 1:]:
                    self.assertLess(rr.iou(a, b, W, H), 0.5)

    # MUST 2: keep/keep with different classes on a duplicate and its twin
    def test_twin_keep_keep_different_class_is_conflict(self):
        p = sq(0.3, 0.3, 0.1)
        fd = self.fx.image("4_x_png", gt=[(1, "Low conf EV", p)])
        ft = self.fx.image("T__4_x_png", gt=[(2, "EV", p)])
        self.fx.edit(fd, {"g1": ("keep:Low conf EV", None)})
        self.fx.edit(ft, {"g2": ("keep:EV", None)})
        self.fx.dups["4_x_png"] = "T__4_x_png"
        self.fx.now("4_x_png", [("Low conf EV", p)])
        self.fx.now("T__4_x_png", [("EV", p)])
        rows, _ = self.fx.resolve()
        self.assertEqual(ResolveFixture.status(rows, "4_x_png", "g1"), "conflict")

    # SHOULD 3: a duplicate's delete of a label only its own copy has
    def test_dup_delete_only_in_own_copy_needs_human(self):
        p = sq(0.6, 0.1, 0.1)
        fd = self.fx.image("4_y_png", gt=[(1, "EV", p)])
        self.fx.image("T__4_y_png", gt=[])
        self.fx.edit(fd, {"g1": ("del", None)})
        self.fx.dups["4_y_png"] = "T__4_y_png"
        self.fx.now("4_y_png", [("EV", p)])
        self.fx.now("T__4_y_png", [])
        rows, _ = self.fx.resolve()
        self.assertEqual(ResolveFixture.status(rows, "4_y_png", "g1"), "needs-human")

    # SHOULD 4: a delete is applied-ready only while the label still has the September class
    def test_delete_of_reclassed_label_is_conflict(self):
        p = sq(0.4, 0.4, 0.1)
        f = self.fx.image("D_png", gt=[(1, "EV", p)])
        self.fx.edit(f, {"g1": ("del", None)})
        self.fx.now("D_png", [("MV", p)])
        rows, _ = self.fx.resolve()
        self.assertEqual(ResolveFixture.status(rows, "D_png", "g1"), "conflict")

    # SHOULD 5: an add inside an existing label (IoU < 0.5) needs a human
    def test_add_inside_label_needs_human(self):
        small = sq(0.42, 0.42, 0.05)
        f = self.fx.image("E_png", pr=[small])
        self.fx.edit(f, {"p0": ("add:EV", None)})
        self.fx.now("E_png", [("EV", sq(0.4, 0.4, 0.1))])
        rows, _ = self.fx.resolve()
        self.assertEqual(ResolveFixture.status(rows, "E_png", "p0"), "needs-human")

    # SHOULD 6: the same label rules as review_server.effective_labels(), e.g. drawn n<uid> outlines
    def test_drawn_outline_is_an_add(self):
        f = self.fx.image("F_png")
        self.fx.edit(f, {"nabc123": ("new:MV", sq(0.1, 0.6, 0.1))})
        self.fx.now("F_png", [])
        rows, _ = self.fx.resolve()
        self.assertEqual([(r["op"], r["to"], r["status"]) for r in rows], [("add", "MV", "applied-ready")])

    # SHOULD 7: the twin operation that overlaps most decides, and its own status counts
    def test_twin_best_overlap_decides(self):
        p = sq(0.3, 0.3, 0.1)
        fd = self.fx.image("4_z_png", gt=[(1, "EV", p)])
        ft = self.fx.image("T__4_z_png", gt=[(2, "MV", sq(0.32, 0.3, 0.1)), (3, "EV", p)])
        self.fx.edit(fd, {"g1": ("keep:EV", None)})
        self.fx.edit(ft, {"g2": ("del", None), "g3": ("keep:EV", None)})  # first-listed overlap differs
        self.fx.dups["4_z_png"] = "T__4_z_png"
        self.fx.now("4_z_png", [("EV", p)])
        self.fx.now("T__4_z_png", [("MV", sq(0.32, 0.3, 0.1)), ("EV", p)])
        rows, _ = self.fx.resolve()
        self.assertEqual(ResolveFixture.status(rows, "4_z_png", "g1"), "already-done")

    def test_twin_same_call_but_unresolved_needs_human(self):
        p = sq(0.3, 0.3, 0.1)
        fd = self.fx.image("4_w_png", gt=[(1, "EV", p)])
        ft = self.fx.image("T__4_w_png", gt=[(2, "EV", p)])
        self.fx.edit(fd, {"g1": ("keep:EV", None)})
        self.fx.edit(ft, {"g2": ("keep:EV", None)})
        self.fx.dups["4_w_png"] = "T__4_w_png"
        self.fx.now("4_w_png", [("EV", p)])
        self.fx.now("T__4_w_png", [])          # the twin's label is gone: its keep is an orphan
        rows, _ = self.fx.resolve()
        self.assertEqual(ResolveFixture.status(rows, "T__4_w_png", "g2"), "orphan")
        self.assertEqual(ResolveFixture.status(rows, "4_w_png", "g1"), "needs-human")


class TestFlags(unittest.TestCase):
    def setUp(self):
        self.B = load("build_review_package")
        self.groups = self.B.load_class_groups()

    # SHOULD 8a: the conf cut uses the raw confidence
    def test_conf_compared_before_rounding(self):
        with tempfile.TemporaryDirectory() as t:
            f = Path(t) / "x.txt"
            f.write_text("0 0.4996 " + " ".join(f"{v:.6f}" for v in sq(0.1, 0.1, 0.1)[0]) + "\n")
            pr = self.B.load_predictions(f, ["EV", "non-EV"])
        self.assertEqual(pr[0]["s"], 0.5)                       # shown rounded
        self.B.flag_image([], pr, W, H, self.groups, 0.5)
        self.assertNotIn("u", pr[0])                            # but 0.4996 < 0.50: not flagged

    # SHOULD 8b: label classes missing from class_groups.yaml are reported
    def test_unknown_class_warns(self):
        gt = [{"id": 1, "c": "Weird", "p": sq(0.1, 0.1, 0.1)}]
        with self.assertWarns(UserWarning):
            self.B.flag_image(gt, [], W, H, self.groups, 0.5)

    # SHOULD 11: negative control: an unchanged image gains no flags, a change flags only its own object
    def test_unchanged_gains_nothing_and_change_is_local(self):
        def image():
            gt = [{"id": i, "c": "EV", "p": sq(0.1 + 0.2 * i, 0.1, 0.1)} for i in range(3)]
            pr = [{"s": 0.9, "sr": 0.9, "k": 0, "c": "EV", "p": sq(0.1 + 0.2 * i, 0.1, 0.1)} for i in range(3)]
            return gt, pr
        g0, p0 = image()
        base = self.B.flag_image(g0, p0, W, H, self.groups, 0.5)
        g1, p1 = image()
        self.assertEqual(self.B.flag_image(g1, p1, W, H, self.groups, 0.5), base)
        self.assertEqual((g0, p0), (g1, p1))
        g2, p2 = image()
        g2 = [g for g in g2 if g["id"] != 1]
        fg, fp, fx, flc = self.B.flag_image(g2, p2, W, H, self.groups, 0.5)
        self.assertEqual((fg, fp, fx, flc), (base[0], base[1] + 1, base[2], base[3]))
        self.assertEqual([p.get("u") for p in p2], [None, 1, None])
        self.assertFalse(any(g.get("u") or g.get("x") for g in g2))


class TestParts(unittest.TestCase):
    # SHOULD 10: edits/ goes into the part folders only when asked
    def test_parts_get_edits_only_with_flag(self):
        B = load("build_review_package")
        with tempfile.TemporaryDirectory() as t:
            out = Path(t) / "pkg"
            (out / "images").mkdir(parents=True)
            (out / "edits").mkdir()
            imgs = []
            for n in range(2):
                (out / "images" / f"i{n}.jpg").write_bytes(b"x")
                (out / "edits" / f"i{n}.json").write_text(json.dumps({"file": f"f{n}", "edit": {}}))
                imgs.append({"file": f"f{n}", "src": f"images/i{n}.jpg", "gt": [], "pr": []})
            B.write_packages(out, {"id": "pkg", "images": imgs}, 1, zip_it=False)
            self.assertFalse((Path(t) / "pkg_part1" / "edits").exists())


class TestBackfill(unittest.TestCase):
    # SHOULD 9: MANIFEST-style columns, honest label_source, inferred dates, unread exports
    def test_columns_and_unread(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            root = t / "exports"
            cur, hist, broken = root / "cur", root / "hist", root / "broken"
            for d in (cur / "train", hist / "train", broken / "train"):
                d.mkdir(parents=True)
            coco = {"images": [{"id": 1, "file_name": "Jul30_a_jpg.rf.123.jpg", "width": 10, "height": 10}],
                    "annotations": [], "categories": [{"id": 1, "name": "EV"}]}
            for d in (cur, hist):
                (d / "train" / "_annotations.coco.json").write_text(json.dumps(coco))
            (broken / "train" / "_annotations.coco.json").write_text("\0" * 50)
            (cur / "split_assignment.csv").write_text("image,batch,split\nJul30_a_jpg,Jul30,train\n")
            (cur / "SPLIT_REPORT.md").write_text("# none\n")
            rf = t / "rf.json"
            rf.write_text(json.dumps([{"id": "X1", "name": "Jul30_a.jpg", "split": "train", "tags": [],
                                       "created": 1762367409628, "width": 10, "height": 10}]))
            sm = t / "sessions.csv"
            sm.write_text("batch,session_date,session_date_basis,grid_id,sample_label,tracker_note\n"
                          "Jul30,2025-07-30,inferred: year from Roboflow upload,,,\n")
            out = t / "track.csv"
            run([SRC / "backfill_tracking.py", "--rf-images", rf, "--exports-root", root, "--current", "cur",
                 "--history", "hist", "--session-map", sm, "--out", out])
            with open(out, newline="", encoding="utf-8") as fh:
                row = next(csv.DictReader(fh))
        for col in ("roboflow_batch", "split_role"):
            self.assertIn(col, row)
            self.assertEqual(row[col], "")
        self.assertEqual(row["label_source"], "unknown (pre-MANIFEST)")
        self.assertTrue(row.get("session_date_basis", "").startswith("inferred"))
        self.assertEqual(row["in: broken"], "unread")
        self.assertEqual(row["in: hist"], "train (0)")


@unittest.skipUnless(any(p.is_dir() for p in REAL_SEEDS), "september_edits/v2 seeds not built here")
class TestRealSeeds(unittest.TestCase):
    """MUST 1 on the real data: no two seeded outlines on one image overlap at IoU >= 0.5."""

    def test_real_seeds_do_not_overlap(self):
        rr = load("resolve_review_edits")
        seeds = next(p for p in REAL_SEEDS if p.is_dir())
        pull = sorted(next(p for p in REAL_PULL if p.is_dir()).glob("roboflow_pull_*"))[-1]
        cur = rr.index_coco(json.loads((pull / "current.coco.json").read_text(encoding="utf-8")))
        by_file = {img["file_name"]: s for s, img in cur.items()}
        records = {by_file[json.loads(p.read_text(encoding="utf-8"))["file"]]: json.loads(p.read_text(encoding="utf-8"))
                   for p in seeds.glob("*.json")}
        self.assertTrue(records)
        self.assertEqual(rr.seeded_overlaps(records, cur), [])


if __name__ == "__main__":
    warnings.simplefilter("default")
    unittest.main()
