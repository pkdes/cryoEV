"""
Browser tests (Playwright, headless Edge or Chrome; skipped without one) for round 14's review page changes
(annotation/review_tool/index.html):
  - the OBJECT SET CARD (D.kind "objects", contracts C2-C4) on a hand-made package and a stand-in API
    (object_card_fixture.py): attribute rows from D.schema.object_attrs, every tap saves an a:<obj>:<attr>@<rid>
    decision, keys, undo, done / progress, the blind banner, the "a" cap gate, only this reviewer's answers shown,
    escaped schema text, attribute keys never make an image "edited";
  - schema-driven classes (D.schema.classes: order, colours, keys) on an ordinary package;
  - the legacy path: a package without D.schema keeps the old class order, legend and chip text.

  python -m unittest discover -s annotation/tests -p test_object_card_browser.py -v
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_fixture  # noqa: E402
import object_card_fixture as F  # noqa: E402
from test_review_tool_browser import launch, sync_playwright  # noqa: E402

sys.path.append(str(Path(__file__).resolve().parents[2] / "tests"))
from cryoev_testkit import browser  # noqa: E402

OLD_ORDER = ["EV", "Low conf EV", "MV", "inner", "Incomplete", "Overlapping", "VLDL", "non-EV"]


class _Browser(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if sync_playwright is None:
            raise unittest.SkipTest("playwright not installed")
        cls.pw = sync_playwright().start()
        cls.browser = launch(cls.pw)
        if cls.browser is None:
            cls.pw.stop()
            raise unittest.SkipTest("no Edge / Chrome for Playwright")

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.pw.stop()

    def serve(self, build, caps=("i", "t", "a"), who=None):
        d = Path(tempfile.mkdtemp(prefix="objcard_")) / "pred_obj_fixture_ab12"
        build(d)
        srv = F.FakeServer(d, caps=caps, who=who).start()
        self.addCleanup(srv.close)
        return d, srv

    def page(self, url, width=1280, height=800):
        ctx = self.browser.new_context(viewport={"width": width, "height": height})
        self.addCleanup(ctx.close)
        pg = ctx.new_page()
        self.errors = []
        pg.on("pageerror", lambda e: self.errors.append(str(e)))
        pg.goto(url)
        return pg

    def obj(self, pg):
        return pg.evaluate("__review.obj()")

    def settle(self, pg):
        pg.wait_for_function("__review.state().dirty === 0", timeout=10000)


@browser
class ObjectCardTests(_Browser):
    def open_card(self, caps=("i", "t", "a"), **kw):
        d, srv = self.serve(lambda f: F.build(f, **kw), caps=caps)
        pg = self.page(srv.url + "?who=rev1")
        pg.wait_for_function("window.__review && __review.obj() && __review.state().server", timeout=15000)
        return d, srv, pg

    def test_rows_keys_and_saves(self):
        d, srv, pg = self.open_card()
        o = self.obj(pg)
        self.assertEqual(o["attrs"], ["type", "membranes", "inner", "outline_wrong", "note"])
        self.assertTrue(o["can"] and o["blind"])
        self.assertFalse(pg.is_visible("#app"))
        self.assertTrue(pg.is_visible("#oBlind"))
        self.assertEqual(pg.inner_text("#oPos"), "1 / 6")
        self.assertRegex(pg.inner_text("#oItem"), r"^item [0-9a-f]{4}$")
        heads = pg.eval_on_selector_all("#oAttrs h3 > span:first-child", "els => els.map(e => e.textContent)")
        self.assertEqual(heads, ["Type", "Concentric membranes", "Inner vesicles", "Outlines wrong", "Note"])
        pg.keyboard.press("2")
        pg.keyboard.press("d")
        pg.click("#oAttrs h3[data-attr=inner] + .oBtns button >> nth=0")
        pg.keyboard.press("o")
        self.settle(pg)
        it = self.obj(pg)["item"]
        self.assertEqual(self.obj(pg)["mine"], {"type": "ML", "membranes": "3", "inner": "0", "outline_wrong": True,
                                                "note": None})
        file = it["image"] + ".png"
        dec = srv.records[file]["dec"]
        self.assertEqual({k: v["v"] for k, v in dec.items()},
                         {f"a:{it['obj']}:type@rev1": "ML", f"a:{it['obj']}:membranes@rev1": "3",
                          f"a:{it['obj']}:inner@rev1": "0", f"a:{it['obj']}:outline_wrong@rev1": True})
        self.assertEqual(self.obj(pg)["done"], 1)
        self.assertTrue(pg.evaluate("document.getElementById('oNext').classList.contains('on')"))
        # attribute decisions never make the image "edited" (C3)
        self.assertEqual(pg.evaluate(f"__review.statusOf(__review.D.images.find(im => im.file === {json.dumps(file)}))"),
                         "unreviewed")
        # the same choice again clears it; Backspace puts it back
        pg.keyboard.press("2")
        self.settle(pg)
        self.assertIsNone(self.obj(pg)["mine"]["type"])
        pg.keyboard.press("Backspace")
        self.settle(pg)
        self.assertEqual(self.obj(pg)["mine"]["type"], "ML")
        # note: n focuses it, Enter saves; capped at the schema's max
        pg.keyboard.press("n")
        pg.keyboard.type("two layers <b>x</b>")
        pg.keyboard.press("Enter")
        self.settle(pg)
        self.assertEqual(self.obj(pg)["mine"]["note"], "two layers <b>x</b>")
        # navigation: → / ← / Enter = next not done
        pg.keyboard.press("ArrowRight")
        self.assertEqual(self.obj(pg)["pos"], 1)
        pg.keyboard.press("ArrowLeft")
        pg.keyboard.press("Enter")
        self.assertEqual(self.obj(pg)["pos"], 1)
        pg.keyboard.press("i")
        self.assertFalse(self.obj(pg)["outlines"])
        self.assertEqual(pg.inner_text("#oOutl"), "Outlines off · i")
        self.assertEqual(self.errors, [])

    def test_no_a_cap_no_saves(self):
        d, srv, pg = self.open_card(caps=("i", "t"))
        self.assertFalse(self.obj(pg)["can"])
        self.assertTrue(pg.is_visible("#oOff"))
        self.assertTrue(pg.eval_on_selector_all("#oAttrs button", "bs => bs.every(b => b.disabled)"))
        pg.keyboard.press("2")
        pg.wait_for_timeout(300)
        self.assertEqual(srv.puts, [])
        self.assertIsNone(self.obj(pg)["mine"]["type"])

    def test_not_blind_no_banner(self):
        d, srv, pg = self.open_card(blind=False)
        self.assertFalse(pg.is_visible("#oBlind"))

    def test_other_reviewers_answers_never_shown(self):
        d, srv, pg = self.open_card()
        it = self.obj(pg)["item"]
        srv.records[it["image"] + ".png"] = {"dec": {f"a:{it['obj']}:type@rev2": {"v": "single", "t": 5, "who": "rev2",
                                                                                     "server_t": 5}}}
        pg.evaluate("__review.pull()")
        pg.wait_for_timeout(200)
        self.assertIsNone(self.obj(pg)["mine"]["type"])
        self.assertNotIn("single", pg.eval_on_selector_all("#oAttrs button.on", "bs => bs.map(b => b.textContent).join()"))
        self.assertEqual(self.obj(pg)["done"], 0)

    def test_schema_text_is_escaped(self):
        def build(f):
            data = F.build(f)
            data["schema"]["object_attrs"][0]["label"] = "<img src=x onerror=window.__pwn=1>"
            data["schema"]["object_attrs"][0]["options"][0]["label"] = "<b>bold</b>"
            (f / "data.js").write_text("window.REVIEW_DATA = " + json.dumps(data) + ";\n", encoding="utf-8")
        d, srv = self.serve(build)
        pg = self.page(srv.url + "?who=rev1")
        pg.wait_for_function("window.__review && __review.obj()", timeout=15000)
        self.assertIsNone(pg.evaluate("window.__pwn || null"))
        self.assertEqual(pg.eval_on_selector_all("#oAttrs img, #oAttrs b", "e => e.length"), 0)
        self.assertIn("<b>bold</b>", pg.inner_text("#oAttrs"))

    def test_phone_layout(self):
        d, srv = self.serve(F.build)
        pg = self.page(srv.url + "?who=rev1", width=390, height=844)
        pg.wait_for_function("window.__review && __review.obj() && __review.state().server", timeout=15000)
        sw = pg.evaluate("document.documentElement.scrollWidth")
        self.assertLessEqual(sw, 390)
        pg.click("#oListBtn")
        self.assertEqual(pg.eval_on_selector_all("#oItems button", "b => b.length"), 6)
        pg.click("#oItems button[data-n='3']")
        self.assertEqual(self.obj(pg)["pos"], 3)


@browser
class SchemaClassesTests(_Browser):
    def test_schema_classes_colours_keys(self):
        def build(f):
            make_fixture.build(f)
            t = (f / "data.js").read_text(encoding="utf-8")
            data = json.loads(t.split("=", 1)[1].strip().rstrip(";"))
            data["schema"] = {"id": "ls_test_v1", "classes": [{"name": "EV", "color": "#112233", "key": "e"},
                                                               {"name": "non-EV", "color": "red;x", "key": "q"}]}
            (f / "data.js").write_text("window.REVIEW_DATA = " + json.dumps(data) + ";\n", encoding="utf-8")
        d, srv = self.serve(build, caps=("i", "t"))
        pg = self.page(srv.url + "?who=tester&mode=editor")
        pg.wait_for_function("window.__review && __review.state().server && __review.state().imgReady", timeout=15000)
        cats = pg.evaluate("__review.CATS")
        self.assertEqual(cats[:2], ["EV", "non-EV"])
        self.assertNotIn("Low conf EV", cats[:2])
        pg.evaluate("__review.render()")
        chips = pg.eval_on_selector_all("#chipBar button", "bs => bs.map(b => b.textContent)")
        self.assertEqual(chips[:2], ["e EV", "q non-EV"])
        self.assertEqual(pg.eval_on_selector("#legend span", "s => s.style.background"), "rgb(17, 34, 51)")
        pg.keyboard.press("q")
        self.assertEqual(pg.evaluate("__review.state().curCls"), "non-EV")
        self.assertIsNone(pg.evaluate("__review.obj()"))


@browser
class LegacyClassesTests(_Browser):
    """No D.schema: the class order, legend, chips and keys are exactly the old ones."""

    def test_old_classes(self):
        d, srv = self.serve(make_fixture.build, caps=("i", "t"))
        pg = self.page(srv.url + "?who=tester&mode=editor")
        pg.wait_for_function("window.__review && __review.state().server && __review.state().imgReady", timeout=15000)
        self.assertEqual(pg.evaluate("__review.CATS")[:8], OLD_ORDER)
        pg.evaluate("__review.render()")
        chips = pg.eval_on_selector_all("#chipBar button", "bs => bs.map(b => b.textContent)")
        self.assertEqual(chips[:8], [f"{k + 1} {c}" for k, c in enumerate(OLD_ORDER)])
        legend = pg.eval_on_selector_all("#legend > div", "ds => ds.map(d => d.textContent)")
        self.assertEqual([x.replace(" (new class)", "") for x in legend[:8]], [f"{k + 1}. {c}" for k, c in enumerate(OLD_ORDER)])
        pg.keyboard.press("3")
        self.assertEqual(pg.evaluate("__review.state().curCls"), "MV")
        self.assertIsNone(pg.evaluate("__review.obj()"))
        self.assertTrue(pg.is_visible("#app"))


if __name__ == "__main__":
    unittest.main()
