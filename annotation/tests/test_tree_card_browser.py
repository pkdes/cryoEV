"""
Browser tests (Playwright, headless Edge or Chrome; skipped without one) for round 15's review page changes
(annotation/review_tool/index.html, contract K4), on hand-made packages and a stand-in API (object_card_fixture.py):
  - the whole-tree card (D.mode "tree"): one block per node with the questions that apply to it (applies_to root /
    child), numbered badges at the node outlines, Tab / keys on the current node, a tap on a block makes it current,
    answers saved per node object, done = every required question of every node, only this reviewer's answers;
  - saves in chunks of at most 64 new attribute decisions (a 30-node card answered offline);
  - a closed set (409 from the server): the card says so and stops saving;
  - help lines under questions (K2 `help`), escaped;
  - rule zero: a set without D.mode (the live set) behaves as before; a byte copy of the live set's data.js and
    schema.json renders and saves (skipped where the live set is not on this machine; the source is only read).

  python -m unittest discover -s annotation/tests -p test_tree_card_browser.py -v
"""
import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import object_card_fixture as F  # noqa: E402
from test_object_card_browser import _Browser  # noqa: E402

sys.path.append(str(Path(__file__).resolve().parents[2] / "tests"))
from cryoev_testkit import browser  # noqa: E402

def _live():
    """The live set's folder (<CryoAI-laptop>/review/<id>, data_utils.config; CJ_LIVE_OBJSET overrides), or None."""
    if os.environ.get("CJ_LIVE_OBJSET"):
        return Path(os.environ["CJ_LIVE_OBJSET"])
    try:
        from data_utils import config
        root = config.laptop_root(required=False)
    except Exception:                                   # noqa: BLE001  no config: the test skips
        root = None
    return Path(root) / "review" / "pred_obj_ml_mv_round1_8092" if root else Path("__no_live_set__")


LIVE = _live()
READY = "window.__review && __review.obj() && __review.state().server"


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


class _Tree(_Browser):
    def serve_tree(self, *, big=False, width=1280, height=800, **srv_kw):
        d = Path(tempfile.mkdtemp(prefix="treecard_")) / "pred_obj_tree_fixture_ab12"
        data = F.build_tree(d, big=big)
        srv = F.FakeServer(d, **srv_kw).start()
        self.addCleanup(srv.close)
        pg = self.page(srv.url + "?who=rev1", width=width, height=height)
        pg.wait_for_function(READY, timeout=15000)
        return data, srv, pg

    def goto_item(self, pg, n):
        pos = self.obj(pg)["pos"]
        key = "ArrowRight" if n >= pos else "ArrowLeft"
        for _ in range(abs(n - pos)):
            pg.keyboard.press(key)
        self.assertEqual(self.obj(pg)["pos"], n)

    def tree(self, pg):
        return self.obj(pg)["tree"]


@browser
class TreeCardTests(_Tree):
    def test_blocks_badges_keys_and_saves(self):
        data, srv, pg = self.serve_tree()
        n = next(i for i, it in enumerate(data["items"]) if len(it["nodes"]) == 5)
        self.goto_item(pg, n)
        it = data["items"][n]
        o = self.obj(pg)
        self.assertEqual(o["mode"], "tree")
        t = o["tree"]
        self.assertEqual([x["n"] for x in t["nodes"]], [1, 2, 3, 4, 5])
        self.assertEqual(t["nodes"][0]["attrs"], ["type", "note"])                     # applies_to root / all
        self.assertTrue(all(x["attrs"] == ["relation", "inner_type", "note"] for x in t["nodes"][1:]))
        heads = pg.eval_on_selector_all("#oAttrs .oNode > h2", "hs => hs.map(h => h.textContent)")
        self.assertEqual(len(heads), 5)
        self.assertTrue(heads[0].startswith("1Outer vesicle"))
        for nd, h in zip(it["nodes"][1:], heads[1:]):
            self.assertTrue(h.startswith(f"{nd['n']}Inside{nd['parent_n']}"), h)
        # help lines (K2): under the root's Type and every child's relation question, as text
        helps = pg.eval_on_selector_all("#oAttrs .oHelp", "ps => ps.map(p => p.textContent)")
        self.assertEqual(len(helps), 5)
        self.assertEqual(helps[0], F.HELP_TYPE)
        # "Layer of parent" names the node's parent: "Layer of ④"
        child = it["nodes"][1]
        circ = chr(0x245F + child["parent_n"])
        lbl = pg.eval_on_selector_all('#oAttrs .oNode[data-k="1"] h3[data-attr="relation"] ~ .oBtns button',
                                      "bs => bs.map(b => b.firstChild.textContent)")
        self.assertEqual(lbl[0], f"Layer of {circ}")
        # one badge per node, none covering another (concentric layers share a centroid)
        pg.wait_for_function("__review.obj().tree.badges.length === 5", timeout=10000)
        b = self.tree(pg)["badges"]
        self.assertEqual(sorted(x["n"] for x in b), [1, 2, 3, 4, 5])
        for i in range(len(b)):
            for j in range(i + 1, len(b)):
                self.assertGreaterEqual(((b[i]["x"] - b[j]["x"]) ** 2 + (b[i]["y"] - b[j]["y"]) ** 2) ** .5, 22, b)
        self.assertEqual(sorted(self.obj(pg)["drawn"]), sorted(x["obj"] for x in it["nodes"]))
        # keys act on the current node: 2 = ML on the root; Tab; l = layer on node 2; 2 on a child: not asked there
        self.assertEqual(self.tree(pg)["node"], 0)
        pg.keyboard.press("2")
        pg.keyboard.press("Tab")
        self.assertEqual(self.tree(pg)["node"], 1)
        self.assertTrue(pg.eval_on_selector('#oAttrs .oNode[data-k="1"]', "e => e.classList.contains('cur')"))
        pg.keyboard.press("l")
        pg.keyboard.press("2")
        self.settle(pg)
        t = self.tree(pg)
        self.assertEqual(t["nodes"][0]["mine"]["type"], "ML")
        self.assertEqual(t["nodes"][1]["mine"], {"relation": "layer", "inner_type": None, "note": None})
        dec = srv.records[it["image"] + ".png"]["dec"]
        self.assertEqual({k: v["v"] for k, v in dec.items()},
                         {f"a:{it['nodes'][0]['obj']}:type@rev1": "ML", f"a:{child['obj']}:relation@rev1": "layer"})
        # Shift+Tab back; a tap on a block makes it current (its badge turns amber: state node)
        pg.keyboard.press("Shift+Tab")
        self.assertEqual(self.tree(pg)["node"], 0)
        pg.click('#oAttrs .oNode[data-k="3"] > h2')
        self.assertEqual(self.tree(pg)["node"], 3)
        # n focuses the CURRENT node's note
        pg.keyboard.press("n")
        self.assertEqual(pg.evaluate("document.activeElement.dataset.node"), str(it["nodes"][3]["n"]))
        pg.keyboard.type("two")
        pg.keyboard.press("Enter")
        self.settle(pg)
        self.assertEqual(self.tree(pg)["nodes"][3]["mine"]["note"], "two")
        # done = every required question of every node; Backspace undoes the last one
        self.assertFalse(pg.evaluate("document.getElementById('oNext').classList.contains('on')"))
        for k in range(2, 5):
            pg.click(f'#oAttrs .oNode[data-k="{k}"] h3[data-attr="relation"] ~ .oBtns button >> nth=1')
        self.settle(pg)
        self.assertEqual(self.obj(pg)["done"], 1)
        self.assertTrue(all(x["done"] for x in self.tree(pg)["nodes"]))
        self.assertTrue(pg.evaluate("document.getElementById('oNext').classList.contains('on')"))
        pg.keyboard.press("Backspace")
        self.settle(pg)
        self.assertEqual(self.obj(pg)["done"], 0)
        # ← → move items (the node goes back to the root); Enter = next unfinished
        pg.keyboard.press("ArrowRight" if n + 1 < len(data["items"]) else "ArrowLeft")
        self.assertEqual(self.tree(pg)["node"], 0)
        self.assertEqual(self.errors, [])

    def test_only_own_answers(self):
        data, srv, pg = self.serve_tree()
        it = data["items"][self.obj(pg)["pos"]]
        root = it["nodes"][0]["obj"]
        srv.records[it["image"] + ".png"] = {"dec": {f"a:{root}:type@rev2": {"v": "single", "t": 5, "who": "rev2",
                                                                               "server_t": 5}}}
        pg.evaluate("__review.pull()")
        pg.wait_for_timeout(200)
        self.assertIsNone(self.tree(pg)["nodes"][0]["mine"]["type"])
        self.assertEqual(pg.eval_on_selector_all("#oAttrs button.on", "bs => bs.length"), 0)

    def test_keys_line_and_phone_layout(self):
        data, srv, pg = self.serve_tree(width=390, height=844)
        self.assertEqual(pg.eval_on_selector("#oKeys", "e => e.textContent"),
                         "Tab = next node · ← → items · Enter = next unfinished")
        self.assertLessEqual(pg.evaluate("document.documentElement.scrollWidth"), 390)
        rights = pg.eval_on_selector_all("#obj .otbar > *, #oAttrs .oNode, #oAttrs .oNode > h2 > *",
                                         "es => es.filter(e => e.offsetParent).map(e => e.getBoundingClientRect().right)")
        self.assertTrue(rights and max(rights) <= 390, rights)

    def test_closed_set_stops_saving(self):
        data, srv, pg = self.serve_tree(closed=True)
        pg.keyboard.press("2")
        pg.wait_for_function("__review.obj().closed === true", timeout=8000)
        self.assertFalse(self.obj(pg)["can"])
        self.assertIn("closed", pg.inner_text("#oOff"))
        self.assertTrue(pg.eval_on_selector_all("#oAttrs .oNode button", "bs => bs.every(b => b.disabled)"))
        n = len(srv.puts)
        pg.evaluate("window.dispatchEvent(new Event('online'))")
        pg.wait_for_timeout(600)
        self.assertEqual(len(srv.puts), n)                                         # nothing sent again
        self.assertEqual(srv.records, {})


@browser
class ChunkedSaveTests(_Tree):
    def test_30_node_card_saves_in_chunks(self):
        """QC 9: a 30-node tree answered offline is more than the server's 64 per PUT; the page sends chunks."""
        data, srv, pg = self.serve_tree(big=True, fail=True)
        it = data["items"][0]
        self.assertEqual(len(it["nodes"]), 30)
        self.assertEqual(self.obj(pg)["pos"], 0)
        # every child: Inner vesicle + its type Single; the root: ML; 8 notes -> 1 + 29 * 2 + 8 = 67 decisions
        pg.evaluate("""() => {
          const q = s => document.querySelector(s);
          const blocks = () => [...document.querySelectorAll('#oAttrs .oNode')];
          const n = blocks().length;
          q('#oAttrs .oNode[data-k="0"] h3[data-attr="type"] ~ .oBtns button:nth-child(2)').click();
          for (let k = 1; k < n; k++) {
            q(`#oAttrs .oNode[data-k="${k}"] h3[data-attr="relation"] ~ .oBtns button:nth-child(2)`).click();
            q(`#oAttrs .oNode[data-k="${k}"] h3[data-attr="inner_type"] ~ .oBtns button:nth-child(1)`).click();
          }
          for (let k = 0; k < 8; k++) {
            const inp = q(`#oAttrs .oNode[data-k="${k}"] input[data-attr="note"]`);
            inp.value = 'note ' + k; inp.dispatchEvent(new Event('change'));
          }
        }""")
        file = it["image"] + ".png"
        pend = pg.evaluate(f"Object.entries(__review.edits()[{json.dumps(file)}].dec).filter(([k, d]) => k.startsWith('a:') && d.pending).length")
        self.assertEqual(pend, 67)
        self.assertNotIn(file, srv.records)
        srv.fail = False
        pg.evaluate("window.dispatchEvent(new Event('online'))")
        self.settle(pg)
        puts = [rec for f, rec in srv.puts if f == file]
        sizes = [sum(1 for k, d in rec["dec"].items() if k.startswith("a:") and d.get("pending")) for rec in puts]
        self.assertTrue(sizes and max(sizes) <= 64, sizes)
        self.assertGreaterEqual(len(puts), 2)
        self.assertEqual(sum(sizes), 67)                                           # each decision sent once
        dec = srv.records[file]["dec"]
        self.assertEqual(len(dec), 67)
        self.assertEqual(dec[f"a:{it['nodes'][0]['obj']}:type@rev1"]["v"], "ML")
        self.assertTrue(all(dec[f"a:{nd['obj']}:relation@rev1"]["v"] == "inner_vesicle" for nd in it["nodes"][1:]))
        self.assertEqual(self.obj(pg)["done"], 1)
        n = len(srv.puts)
        pg.wait_for_timeout(1200)
        self.assertEqual(len(srv.puts), n)                                         # nothing resent

    def test_64_or_fewer_is_one_put_of_the_whole_record(self):
        """The path of before: a record with 64 or fewer new decisions goes as it is, in one PUT."""
        data, srv, pg = self.serve_tree(fail=True)
        pg.keyboard.press("2")
        pg.keyboard.press("Tab")
        pg.keyboard.press("v")
        it = data["items"][self.obj(pg)["pos"]]
        file = it["image"] + ".png"
        local = pg.evaluate(f"JSON.stringify(__review.edits()[{json.dumps(file)}])")
        srv.fail, before = False, len(srv.puts)
        pg.evaluate("window.dispatchEvent(new Event('online'))")
        self.settle(pg)
        puts = srv.puts[before:]
        self.assertEqual(len([1 for f, _ in puts if f == file]), 1)
        self.assertEqual(next(rec for f, rec in puts if f == file), json.loads(local))


@browser
class RuleZeroTests(_Tree):
    """Rule zero: without D.mode the object card is the one of before; help lines are additive."""

    def test_legacy_set_has_no_tree_parts(self):
        d, srv = self.serve(F.build)
        pg = self.page(srv.url + "?who=rev1")
        pg.wait_for_function(READY, timeout=15000)
        o = self.obj(pg)
        self.assertNotIn("mode", o)
        self.assertNotIn("tree", o)
        self.assertEqual(pg.eval_on_selector_all("#oAttrs .oNode, #oAttrs .oHelp", "e => e.length"), 0)
        self.assertEqual(pg.eval_on_selector_all("#oAttrs h3 > span:first-child", "els => els.map(e => e.textContent)"),
                         ["Type", "Concentric membranes", "Inner vesicles", "Outlines wrong", "Note"])
        self.assertEqual(pg.eval_on_selector("#oKeys", "e => e.textContent"),
                         "← → move · Enter = next unlabelled · Backspace = undo · i = outline on/off")
        pg.keyboard.press("Tab")                                                    # not a key of the one-object card
        self.assertEqual(o["keys"], self.obj(pg)["keys"])

    def test_help_line_on_the_one_object_card(self):
        def build(f):
            data = F.build(f)
            data["schema"] = json.loads(json.dumps(data["schema"]))       # F.SCHEMA itself stays untouched
            data["schema"]["object_attrs"][0]["help"] = "Count the outer <b>EV</b> only."
            (f / "data.js").write_text("window.REVIEW_DATA = " + json.dumps(data) + ";\n", encoding="utf-8")
        d, srv = self.serve(build)
        pg = self.page(srv.url + "?who=rev1")
        pg.wait_for_function(READY, timeout=15000)
        self.assertEqual(pg.eval_on_selector_all("#oAttrs .oHelp", "ps => ps.map(p => p.textContent)"),
                         ["Count the outer <b>EV</b> only."])
        self.assertEqual(pg.eval_on_selector_all("#oAttrs .oHelp b", "e => e.length"), 0)
        self.assertEqual(pg.eval_on_selector("#oAttrs .oHelp", "p => p.previousElementSibling.dataset.attr"), "type")

    @unittest.skipUnless((LIVE / "data.js").is_file() and (LIVE / "schema.json").is_file(),
                         "the live set is not on this machine")
    def test_live_set_byte_copy_renders_and_saves(self):
        """The live set's data.js + schema.json (a byte copy; the source is only read) on this page: the card
        renders as the one-object card with the schema's rows, and a tap saves."""
        before = {n: md5(LIVE / n) for n in ("data.js", "schema.json")}
        d = Path(tempfile.mkdtemp(prefix="liveset_")) / LIVE.name
        data = F.copy_live(d, LIVE)
        self.assertEqual({n: md5(d / n) for n in ("data.js", "schema.json")}, before)   # byte-identical copy
        self.assertNotIn("mode", data)
        srv = F.FakeServer(d).start()
        self.addCleanup(srv.close)
        pg = self.page(srv.url + "?who=rev1")
        pg.wait_for_function(READY, timeout=15000)
        o = self.obj(pg)
        self.assertNotIn("tree", o)
        self.assertEqual(o["n"], len(data["items"]))
        self.assertEqual(o["attrs"], [a["id"] for a in data["schema"]["object_attrs"]])
        self.assertEqual(pg.eval_on_selector_all("#oAttrs .oRow", "e => e.length"), len(o["attrs"]))
        self.assertEqual(pg.eval_on_selector_all("#oAttrs .oNode", "e => e.length"), 0)
        self.assertTrue(pg.is_visible("#obj") and not pg.is_visible("#app"))
        if pg.is_visible("#oInstr"):
            pg.click("#oInstrX")
        pg.keyboard.press("1")
        self.settle(pg)
        it = self.obj(pg)["item"]
        files = [im["file"] for im in data["images"] if im["file"].rsplit(".", 1)[0] == it["image"]
                 or im["file"] == it["image"]]
        self.assertEqual(srv.records[files[0]]["dec"][f"a:{it['obj']}:type@rev1"]["v"], "single")
        self.assertEqual(self.errors, [])
        self.assertEqual({n: md5(LIVE / n) for n in ("data.js", "schema.json")}, before)   # source untouched


if __name__ == "__main__":
    unittest.main()
