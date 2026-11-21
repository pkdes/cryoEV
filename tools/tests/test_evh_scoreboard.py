"""
tools/evh_scoreboard.py (round 17p): who is listed (consent yes + a dating name), who counts in `others`, who never
counts (owner, --exclude, *-test, consent no, no swipes), what a swipe is (match values only; follow-ups and undone
answers don't count), name cleaning and hiding, ranking and --top, and data.js: only `scoreboard` changes, written
only when the board changed, nothing written on a dry run.

    python -m unittest discover -s tools/tests -p test_evh_scoreboard.py -v
"""
import io
import json
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(HERE))
import evh_items  # noqa: E402
import evh_scoreboard as SB  # noqa: E402
from annotation import object_set as O  # noqa: E402
from test_evh_r17c import make_ps  # noqa: E402

V4 = REPO / "tools" / "evh" / "ls_evh_match_v4.json"

# who: (match values in item order, consent, nick)
PEOPLE = {
    "admin1": (["yes"] * 5, "yes", "TheOwner"),               # the set's owner: never
    "ann-test": (["yes"] * 6, "yes", "Tester"),               # *-test: never
    "ann": (["yes", "no", "unsure", None], "yes", "Bilayer​Babe‮"),   # 3 (an undone answer doesn't count)
    "bob": (["no", "no"], "yes", "BOB"),                       # name = an id: others
    "cat": (["yes"] * 4, "yes", "cat@example.org"),            # name with @: others
    "dan": (["yes"], "no", "Dan"),                             # consent no: never
    "eve": (["no", "yes"], None, "Eve"),                       # no consent yet: others
    "rev-f": (["yes"] * 5, "yes", "Fay"),                      # 5: first
    "gus": ([], "yes", "Gus"),                                 # no swipes: never
}


class Scoreboard(unittest.TestCase):
    def setUp(self):
        self.t = Path(tempfile.mkdtemp(prefix="evhsb_"))
        self.addCleanup(shutil.rmtree, self.t, True)
        ps = make_ps(self.t)
        self.assertEqual(evh_items.main(["--ps", str(ps), "--out", str(self.t / "items.csv"), "--gold", "0"]), 0)
        self.pkg = O.build(ps, self.t / "items.csv", str(V4), self.t / "review", line_col="line", slug="evh", seed=3,
                           owner="admin1", app="evharmony", first_items=0)
        its = O.read_items(self.pkg)["items"]
        recs, t = {}, 1794950000000
        for who, (vals, consent, nick) in PEOPLE.items():
            for it, v in zip(its, vals):
                t += 10
                d = recs.setdefault(f"{it['image']}.png", {})
                d[f"a:{it['obj']}:match@{who}"] = {"v": v, "t": t, "who": who, "server_t": t + 1}
                if v == "yes":                                 # a follow-up never counts as a swipe
                    d[f"a:{it['obj']}:kind@{who}"] = {"v": "single", "t": t, "who": who, "server_t": t + 1}
            d0 = recs.setdefault(f"{its[0]['image']}.png", {})
            if consent:
                d0[f"a:{its[0]['obj']}:consent@{who}"] = {"v": consent, "t": t, "who": who, "server_t": t + 2}
            d0[f"a:{its[0]['obj']}:nick@{who}"] = {"v": nick, "t": t, "who": who, "server_t": t + 3}
        (self.pkg / "edits").mkdir(exist_ok=True)
        for f, dec in recs.items():
            (self.pkg / "edits" / f"{f}.json").write_text(json.dumps({"file": f, "pkg": self.pkg.name, "fmt": 1, "edit": {
                "reviewed": False, "modified": False, "dec": dec, "t": t}, "rx": {}}), encoding="utf-8")

    def run_tool(self, *args):
        out = io.StringIO()
        with redirect_stdout(out):
            rc = SB.main([str(self.pkg), *args])
        return rc, out.getvalue()

    def test_board_rules(self):
        cat, bob = SB.anon_name(self.pkg.name, "cat"), SB.anon_name(self.pkg.name, "bob")   # r17s: unusable name -> Anonymous
        self.assertRegex(cat, r"^Anonymous #[0-9a-f]{4}$")
        self.assertNotEqual(cat, bob)
        F, A, C, B = ({"name": "Fay", "swipes": 5}, {"name": "BilayerBabe", "swipes": 3}, {"name": cat, "swipes": 4},
                      {"name": bob, "swipes": 2})
        self.assertEqual(SB.board(self.pkg), {"rows": [F, C, A, B], "others": 1})            # eve: no consent yet
        self.assertEqual(SB.board(self.pkg, hide=["fay"]), {"rows": [C, A, B], "others": 2})
        self.assertEqual(SB.board(self.pkg, exclude=["ann"]), {"rows": [F, C, B], "others": 1})
        self.assertEqual(SB.board(self.pkg, top=1), {"rows": [F], "others": 4})

    def test_unnamed_guest_listed(self):
        p = self.pkg / "edits"
        for f in p.glob("*.json"):                                 # ann's name gone: listed as Anonymous, same swipes
            j = json.loads(f.read_text(encoding="utf-8"))
            j["edit"]["dec"].pop(next((k for k in j["edit"]["dec"] if k.endswith(":nick@ann")), None), None)
            f.write_text(json.dumps(j), encoding="utf-8")
        rows = SB.board(self.pkg)["rows"]
        self.assertIn({"name": SB.anon_name(self.pkg.name, "ann"), "swipes": 3}, rows)
        self.assertNotIn("ann", json.dumps(rows))

    def test_writes_only_scoreboard_and_only_on_change(self):
        djs = self.pkg / "data.js"
        before = djs.read_bytes()
        rc, out = self.run_tool()                                  # dry run: prints, writes nothing
        self.assertEqual(rc, 0)
        self.assertIn("Fay", out)
        self.assertEqual(djs.read_bytes(), before)
        rc, out = self.run_tool("--do")
        self.assertIn("written", out)
        self.assertNotIn("Fay", out)                               # --do never prints names
        a, b = O._split_datajs(before)[0], O._split_datajs(djs.read_bytes())[0]
        sb = b.pop("scoreboard")
        self.assertEqual(a, b)                                     # nothing else changed
        self.assertEqual([r["name"] for r in sb["rows"]][::2], ["Fay", "BilayerBabe"])
        self.assertEqual(sb["others"], 1)
        text = djs.read_text(encoding="utf-8")
        for secret in ("admin1", "ann-test", '"bob"', "cat@", "Dan", "Eve", "Gus", "TheOwner", "Tester"):
            self.assertNotIn(secret, text.split('"scoreboard"', 1)[1], secret)
        mid = djs.read_bytes()
        rc, out = self.run_tool("--do")                            # unchanged board: no write
        self.assertIn("unchanged", out)
        self.assertEqual(djs.read_bytes(), mid)
        rc, out = self.run_tool("--do", "--hide", "Fay")           # a change: written again
        self.assertIn("written", out)
        self.assertNotIn("Fay", [r["name"] for r in O._split_datajs(djs.read_bytes())[0]["scoreboard"]["rows"]])

    def test_clean_name(self):
        self.assertEqual(SB.clean_name("  A​‮b\x07  c "), "Ab c")
        self.assertEqual(SB.clean_name("x" * 40), "x" * 24)
        self.assertEqual(SB.clean_name(None), "")


if __name__ == "__main__":
    unittest.main()
