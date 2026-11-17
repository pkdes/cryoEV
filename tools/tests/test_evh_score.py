"""tools/evh_score.py and tools/evh_items.py (round 17 eVHarmony), on synthetic data only.

Controls on a balanced gold (20 yes / 20 no): an always-yes reviewer and a random reviewer FAIL; a replay of the gold
answers scores 100 % and passes. Plus: unsure excluded and reported, consent (absent = no), too few gold = fail, the
items maker's conf-stratified gold rows first, and object_set build's --first-items / --app on a synthetic set.

    python -m unittest tools/tests/test_evh_score.py -v
"""
import csv
import io
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))
import evh_items  # noqa: E402
import evh_score  # noqa: E402


def export(rows_by_rid: dict, n: int, consent: dict | None = None) -> str:
    """An object_set export's shape: item, image, obj, line, match@<rid>..., consent@<rid>..., agree_*..."""
    rids = sorted(rows_by_rid)
    cols = (["item", "image", "obj", "line"] + [f"match@{r}" for r in rids] + [f"consent@{r}" for r in rids]
            + ["agree_match", "agree_consent", "reviewers", "last_t", "history_n", "schema_id", "schema_version"])
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(cols)
    for k in range(n):
        row = {"item": f"{k:05x}", "image": "0001", "obj": f"g{k + 1}", "line": k, "schema_id": "ls_evh_match_v1",
               "schema_version": 1}
        for r in rids:
            row[f"match@{r}"] = rows_by_rid[r][k] or ""
            row[f"consent@{r}"] = (consent or {}).get(r, "") if k == 0 else ""
        w.writerow([row.get(c, "") for c in cols])
    return buf.getvalue()


class Score(unittest.TestCase):
    def setUp(self):
        rnd = random.Random(5)
        self.gold = ["yes"] * 20 + ["no"] * 20
        rnd.shuffle(self.gold)
        self.n = 40

    def run_score(self, revs, consent=None, **kw):
        rows, rids = evh_score.read_export(export({"owner": self.gold, **revs}, self.n, consent))
        return {r["reviewer"]: r for r in evh_score.score(rows, rids, "owner", **kw)}

    def test_controls(self):
        rnd = random.Random(11)
        res = self.run_score({"always_yes": ["yes"] * 40, "random": [rnd.choice(["yes", "no"]) for _ in range(40)],
                              "replay": list(self.gold)}, consent={"replay": "yes"})
        self.assertEqual(res["always_yes"]["agree_pct"], 50.0)
        self.assertFalse(res["always_yes"]["pass"])
        self.assertLess(res["random"]["agree_pct"], 80)
        self.assertFalse(res["random"]["pass"])
        self.assertEqual(res["replay"]["agree_pct"], 100.0)
        self.assertTrue(res["replay"]["pass"])
        self.assertEqual(res["replay"]["consent"], "yes")
        self.assertEqual(res["always_yes"]["consent"], "no")          # absent = no
        self.assertNotIn("owner", res)

    def test_unsure_excluded_and_minimum(self):
        a = list(self.gold)
        for k in range(15):
            a[k] = "unsure"
        res = self.run_score({"a": a, "few": list(self.gold[:19]) + [None] * 21}, consent={"a": "no"})
        self.assertEqual((res["a"]["n_gold"], res["a"]["n_unsure"], res["a"]["n_scored"]), (40, 15, 25))
        self.assertEqual(res["a"]["agree_pct"], 100.0)
        self.assertTrue(res["a"]["pass"])
        self.assertEqual(res["a"]["consent"], "no")
        self.assertEqual(res["few"]["n_scored"], 19)
        self.assertFalse(res["few"]["pass"])                         # 100 % on 19 < 20 gold

    def test_gold_unsure_not_gold(self):
        g = list(self.gold)
        g[0] = "unsure"
        rows, rids = evh_score.read_export(export({"owner": g, "b": ["yes"] * 40}, 40))
        r = evh_score.score(rows, rids, "owner")[0]
        self.assertEqual(r["n_gold"], 39)

    def test_cli(self):
        with tempfile.TemporaryDirectory() as t:
            f, out = Path(t) / "e.csv", Path(t) / "s.csv"
            f.write_text(export({"owner": self.gold, "replay": list(self.gold)}, 40, {"replay": "yes"}), encoding="utf-8")
            self.assertEqual(evh_score.main(["--csv", str(f), "--gold", "owner", "--out", str(out)]), 0)
            rows = list(csv.DictReader(io.StringIO(out.read_text(encoding="utf-8"))))
            self.assertEqual(rows[0]["agree_pct"], "100.0")
            self.assertEqual(rows[0]["pass"], "1")
            self.assertEqual(evh_score.main(["--csv", str(f), "--gold", "nobody"]), 2)


class Items(unittest.TestCase):
    def test_stratified(self):
        cands = [(f"{k % 5 + 1:04d}", k, 0.25 + 0.25 * k / 200) for k in range(200)]
        a = evh_items.stratified(cands, 40, 17)
        self.assertEqual(a, evh_items.stratified(cands, 40, 17))      # reproducible
        self.assertEqual(len(set(a)), 40)
        confs = sorted(c[2] for c in a)
        for k, c in enumerate(confs):                                 # one per conf bin
            self.assertTrue(cands[k * 5][2] <= c <= cands[k * 5 + 4][2])
        text = evh_items.items_csv(cands, 40, 17)
        rows = list(csv.DictReader(io.StringIO(text)))
        self.assertEqual(len(rows), 200)
        self.assertEqual([r["gold"] for r in rows], ["1"] * 40 + ["0"] * 160)


class BuildFlags(unittest.TestCase):
    """object_set build --first-items / --app on a synthetic prediction set (no real package)."""

    def make_ps(self, root: Path) -> Path:
        from PIL import Image
        ps = root / "ps_20990101_abcdef"
        (ps / "images").mkdir(parents=True)
        pred = ps / "p" / "m1_test" / "predictions"
        pred.mkdir(parents=True)
        (ps / "request.json").write_text(json.dumps({"ps": ps.name, "model": "m1_test", "owner": "admin1"}))
        with open(ps / "images.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["image", "file", "orig_name", "sample_label", "nm_per_px", "source"])
            for i in (1, 2):
                Image.new("L", (200, 200), 120).save(ps / "images" / f"{i:04d}.png")
                w.writerow([f"{i:04d}", f"images/{i:04d}.png", f"x{i}.png", "S", "1.5", "upload"])
                lines = []
                for k in range(8):
                    cx, cy = 0.1 + 0.1 * k, 0.5
                    lines.append(f"0 {0.3 + 0.02 * k:.3f} {cx - .04} {cy - .04} {cx + .04} {cy - .04} {cx + .04} {cy + .04} "
                                 f"{cx - .04} {cy + .04}")
                (pred / f"{i:04d}.txt").write_text("\n".join(lines) + "\n")
        return ps

    def test_first_items_and_app(self):
        from annotation import object_set as O
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            ps = self.make_ps(t)
            items = t / "items.csv"
            items.write_text(evh_items.items_csv(evh_items.candidates(ps), 5, 3), encoding="utf-8")
            rows = list(csv.DictReader(io.StringIO(items.read_text(encoding="utf-8"))))
            self.assertEqual(len(rows), 16)
            schema = REPO / "tools" / "evh" / "ls_evh_match_v1.json"
            gold = {(r["image"], int(r["line"])) for r in rows[:5]}
            orders = []
            for seed in (1, 2):
                pkg = O.build(ps, items, str(schema), t / "review", line_col="line", slug="evh", seed=seed,
                              owner="admin1", title="eVHarmony beta", app="evharmony", first_items=5)
                its = O.read_items(pkg)
                self.assertEqual(its["first_items"], 5)
                self.assertEqual({(i["image"], i["line"]) for i in its["items"][:5]}, gold)
                self.assertTrue(all(i["row"]["gold"] == "1" for i in its["items"][:5]))
                orders.append([(i["image"], i["line"]) for i in its["items"][:5]])
                d = json.loads((pkg / "data.js").read_text(encoding="utf-8").split("=", 1)[1].strip().rstrip(";"))
                self.assertEqual(d["app"], "evharmony")
                self.assertEqual(json.loads((pkg / "review_package.json").read_text())["app"], "evharmony")
            # without the flags: no app / first_items keys, and the same seed gives the order of before
            plain = O.build(ps, items, str(schema), t / "review", line_col="line", slug="plain", seed=1,
                            owner="admin1")
            d = json.loads((plain / "data.js").read_text(encoding="utf-8").split("=", 1)[1].strip().rstrip(";"))
            self.assertNotIn("app", d)
            self.assertNotIn("first_items", O.read_items(plain))
            rnd = random.Random(1)
            order = list(range(16))
            rnd.shuffle(order)
            self.assertEqual([(i["image"], i["line"]) for i in O.read_items(plain)["items"]],
                             [(rows[k]["image"], int(rows[k]["line"])) for k in order])
            with self.assertRaises(O.ObjectSetError):
                O.build(ps, items, str(schema), t / "review", line_col="line", slug="x", seed=1, owner="admin1",
                        first_items=17)


if __name__ == "__main__":
    unittest.main()
