"""
Browser tests (Playwright, headless Edge or Chrome; skipped without one) for round 17's eVHarmony skin on the
object set card (annotation/review_tool/index.html, D.app === "evharmony"), on a hand-made set and the stand-in API
(object_card_fixture.py; nothing here touches a real package or server):
  - flag off: the plain object card, no eVHarmony element shown, contrast default unchanged;
  - a touch swipe right / left / up and the keys write a:<obj>:match@<rid> = yes / no / unsure; a mouse drag pans;
  - the reveal after the 10th answer, consent saved as a:<items[0].obj>:consent@<rid>, not offered again;
  - the size line: shown with nm/px and no blind columns; none when blind or nm/px is missing;
  - the splash once per browser (nickname display only); default contrast Auto + Smooth 2 without a saved setting;
  - phone 390 x 844, light and dark, without page errors;
  - r17b: saves the server refuses (403 "not a reviewer"): the #oSaveErr banner with the server's text and the
    pending count, "Not saved" (never "Saved ✓"); the drafts survive a reload and upload once the server accepts;
  - r17c (schema ls_evh_match_v2): the 4th answer (button, swipe down, d / ↓) saves "dense"; after a Match the "What
    is it?" chips (pick = kind saved + next card; skip = nothing saved + next; a swipe = re-answer + next); the "How to
    judge" sheet from the splash and the ? button (sections, example crops, the instructions); v1 sets and the flag
    off unchanged (3 buttons, ↓ / d answer nothing, no sheet, no chips);
  - r17c v3 (ls_evh_match_v3): 3 answers (↓ / d nothing); after a Match the kind chips (Single / Multilayer) and the
    "Outline off" toggle; after It's complicated the "why" chips; the sheet without a Small & dense section.
  - r17d (D.practice): the practice cards first (no attribute PUT for them), the reveal after the 3rd (message, How
    to judge, consent saved; Esc doesn't close it), Redo (feedback "Our answer: Match · Single — note", then the
    real queue) or Start for real (the real queue at once); a reviewer with consent already skips the practice; a
    set without D.practice keeps the reveal at 10.

  python -m unittest discover -s annotation/tests -p test_evh_browser.py -v
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import object_card_fixture as F  # noqa: E402
from test_review_tool_browser import launch, sync_playwright, touch_swipe  # noqa: E402

sys.path.append(str(Path(__file__).resolve().parents[2] / "tests"))
from cryoev_testkit import browser  # noqa: E402

REPO = HERE.parents[1]
EVH_SCHEMA = json.loads((REPO / "tools" / "evh" / "ls_evh_match_v1.json").read_text(encoding="utf-8"))
EVH_V2 = json.loads((REPO / "tools" / "evh" / "ls_evh_match_v2.json").read_text(encoding="utf-8"))
EVH_V3 = json.loads((REPO / "tools" / "evh" / "ls_evh_match_v3.json").read_text(encoding="utf-8"))
GUIDE_TEXT = "What you're judging: the highlighted outline. Unsure is useful; don't guess."
META = {"schema": "review_meta/1", "package": {}, "images": {"imgA.png": {"nm_per_px": 1.5}, "imgB.png": {"nm_per_px": 1.5}}}
SAMPLE = "Zed Person 2023 sample"
SPLASH_SEEN = "try { localStorage.setItem('cryoev-review:evh:splash', '1'); } catch (e) {}"


PRACTICE = ((3, "yes", "single", "A round ring: one vesicle."), (5, "no", "", "No membrane ring."),
            (7, "yes", "", "Another clear ring."))


def build_evh(folder, *, blind=False, app="evharmony", schema=None, examples=None, instructions=None, practice=False):
    """The fixture's two images with every outline as an item (11 items: 10 labels + 1 prediction), the eVHarmony
    schema (v1 unless given), D.app (app=None: the flag off); r17c: D.examples (copies of imgA.png as
    images/ex_<id>.jpg) and D.instructions when given."""
    data = F.build(folder, blind=blind)
    sys.path.insert(0, str(REPO))
    from labels import schemas as S
    items = []
    for im in data["images"]:
        stem = im["file"].rsplit(".", 1)[0]
        objs = [f"g{g['id']}" for g in im["gt"]] + [f"p{k}" for k in range(len(im["pr"]))]
        for o in objs:
            items.append({"id": f"{len(items) + 1:05x}", "image": stem, "obj": o, "crop": [0, 0, 1, 1]})
    data["items"], data["schema"], data["title"] = items, S.to_page(schema or EVH_SCHEMA), "eVHarmony beta"
    if examples:
        import shutil
        for x in examples:
            if x["src"].startswith("images/"):
                shutil.copyfile(Path(folder) / "images" / "imgA.png", Path(folder) / x["src"])
        data["examples"] = examples
    if instructions:
        data["instructions"] = instructions
    if practice:                            # r17d: items 3, 5, 7 leave the queue as practice cards (tools/evh_practice.py)
        data["practice"] = [{**items[n], "answer": a, **({"kind": k} if k else {}), "note": t} for n, a, k, t in PRACTICE]
        data["items"] = [it for n, it in enumerate(items) if n not in {x[0] for x in PRACTICE}]
    for im in data["images"]:
        im["split"] = SAMPLE                # a sample label that names a person: never in an eVHarmony bio
    if app:
        data["app"] = app
    (Path(folder) / "data.js").write_text("window.REVIEW_DATA = " + json.dumps(data) + ";\n", encoding="utf-8")
    return data


@browser
class EvhTests(unittest.TestCase):
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

    def open(self, *, blind=False, app="evharmony", meta=META, splash_seen=True, width=1280, height=800,
             scheme="light", init=None, schema=None, examples=None, instructions=None, practice=False, consent=None):
        d = Path(tempfile.mkdtemp(prefix="evh_")) / "pred_obj_evh_ab12"
        data = build_evh(d, blind=blind, app=app, schema=schema, examples=examples, instructions=instructions,
                         practice=practice)
        srv = F.FakeServer(d, caps=("i", "t", "m", "a"), meta=meta).start()
        if consent:                         # r17d: this reviewer answered consent before (on the first item)
            it0 = data["items"][0]
            srv.records[it0["image"] + ".png"] = {"dec": {f"a:{it0['obj']}:consent@rev1": {
                "v": consent, "t": 1, "who": "rev1", "server_t": 1}}}
        self.addCleanup(srv.close)
        ctx = self.browser.new_context(viewport={"width": width, "height": height}, color_scheme=scheme,
                                       has_touch=width < 600)
        self.addCleanup(ctx.close)
        if splash_seen:
            ctx.add_init_script(SPLASH_SEEN)
        if init:
            ctx.add_init_script(init)
        pg = ctx.new_page()
        self.errors = []
        pg.on("pageerror", lambda e: self.errors.append(str(e)))
        pg.goto(srv.url + "?who=rev1")
        pg.wait_for_function("window.__review && __review.obj() && __review.state().server", timeout=15000)
        return data, srv, pg

    def obj(self, pg):
        return pg.evaluate("__review.obj()")

    def settle(self, pg):
        pg.wait_for_function("__review.state().dirty === 0", timeout=10000)

    def next_card(self, pg, pos):
        pg.wait_for_function(f"__review.obj().pos !== {pos} || __review.obj().done === __review.obj().n", timeout=5000)

    def match_of(self, srv, it, attr="match"):
        d = srv.records.get(it["image"] + ".png", {}).get("dec", {})
        x = d.get(f"a:{it['obj']}:{attr}@rev1")
        return x and x.get("v")

    # ---- r17c: schema v2
    def open_v2(self, **kw):
        ex = [{"src": f"images/ex_{i}.jpg", "cat": c, "caption": f"{c} example {i}"}
              for i, c in (("S1", "single"), ("M1", "ml"), ("N1", "nope"), ("N2", "nope"), ("D1", "dense"),
                           ("C1", "unsure"), ("Bad", "nonsense"))]
        ex.append({"src": "https://example.org/x.jpg", "cat": "single", "caption": "outside the package"})
        return self.open(schema=EVH_V2, examples=kw.pop("examples", ex), instructions=kw.pop("instructions", GUIDE_TEXT),
                         **kw)

    def test_v2_four_answers_and_swipe_down(self):
        _, srv, pg = self.open_v2(width=390, height=844)
        labels = pg.eval_on_selector_all("#oAttrs .evhBtns button", "els => els.map(e => [e.dataset.v, e.textContent])")
        self.assertEqual([v for v, _ in labels], ["no", "yes", "unsure", "dense"])
        self.assertIn("Small & dense", dict(labels)["dense"])
        self.assertTrue(self.obj(pg)["evh"]["dense"])
        self.assertLessEqual(pg.evaluate("document.documentElement.scrollWidth"), 390)
        b = pg.locator("#ocv").bounding_box()
        x, y = b["x"] + b["width"] / 2, b["y"] + b["height"] / 2
        want = []
        o = self.obj(pg)
        touch_swipe(pg, x, y - 60, 10, 170, sel="#ocv")             # swipe down
        self.next_card(pg, o["pos"])
        want.append((o["item"], "dense"))
        for k in ("d", "ArrowDown"):
            o = self.obj(pg)
            pg.keyboard.press(k)
            self.next_card(pg, o["pos"])
            want.append((o["item"], "dense"))
        o = self.obj(pg)
        pg.click("#oAttrs .evhBtns .dense")
        self.next_card(pg, o["pos"])
        want.append((o["item"], "dense"))
        for k, v in (("ArrowLeft", "no"), ("ArrowUp", "unsure")):     # the v1 answers still work
            o = self.obj(pg)
            pg.keyboard.press(k)
            self.next_card(pg, o["pos"])
            want.append((o["item"], v))
        self.settle(pg)
        for it, v in want:
            self.assertEqual(self.match_of(srv, it), v, it)
        self.assertEqual(self.obj(pg)["done"], 6)
        self.assertEqual(self.errors, [])

    def test_v2_kind_chips_after_match(self):
        _, srv, pg = self.open_v2()
        o = self.obj(pg)
        pg.keyboard.press("ArrowRight")                               # Match: the card stays, the chips come
        pg.wait_for_selector("#oAttrs .evhKind", state="visible", timeout=5000)
        o1 = self.obj(pg)
        self.assertEqual((o1["pos"], o1["evh"]["kindFor"]), (o["pos"], o["item"]["id"]))
        self.assertEqual(pg.locator("#oAttrs .evhBtns").count(), 0)
        chips = pg.eval_on_selector_all("#oAttrs .evhKind button", "els => els.map(e => e.textContent)")
        self.assertEqual(chips, ["Single", "Multilayer", "Has vesicles inside", "Inside another", "skip →"])
        pg.click("#oAttrs .evhKind button[data-v='ml']")              # pick: saved, the next card
        self.next_card(pg, o["pos"])
        self.assertIsNone(self.obj(pg)["evh"]["kindFor"])
        o2 = self.obj(pg)                                             # Match, then skip: no kind saved
        pg.click("#oAttrs .evhBtns .like")
        pg.wait_for_selector("#oAttrs .evhKind", state="visible", timeout=5000)
        pg.click("#oAttrs .evhKind button.skip")
        self.next_card(pg, o2["pos"])
        o3 = self.obj(pg)                                             # Match, then the 3 key
        pg.keyboard.press("y")
        pg.wait_for_selector("#oAttrs .evhKind", state="visible", timeout=5000)
        pg.keyboard.press("3")
        self.next_card(pg, o3["pos"])
        o4 = self.obj(pg)                                             # Match, then a swipe left: re-answered no
        pg.keyboard.press("ArrowRight")
        pg.wait_for_selector("#oAttrs .evhKind", state="visible", timeout=5000)
        b = pg.locator("#ocv").bounding_box()
        touch_swipe(pg, b["x"] + b["width"] / 2, b["y"] + b["height"] / 2, -170, 0, sel="#ocv")
        self.next_card(pg, o4["pos"])
        o5 = self.obj(pg)                                             # Match, then Match again: = skip
        pg.keyboard.press("ArrowRight")
        pg.wait_for_selector("#oAttrs .evhKind", state="visible", timeout=5000)
        pg.keyboard.press("ArrowRight")
        self.next_card(pg, o5["pos"])
        self.settle(pg)
        got = [(self.match_of(srv, x["item"]), self.match_of(srv, x["item"], "kind")) for x in (o, o2, o3, o4, o5)]
        self.assertEqual(got, [("yes", "ml"), ("yes", None), ("yes", "mv"), ("no", None), ("yes", None)])
        self.assertEqual(self.obj(pg)["done"], 5)
        o6 = self.obj(pg)                                             # Backspace on the chips undoes the Match
        pg.keyboard.press("ArrowRight")
        pg.wait_for_selector("#oAttrs .evhKind", state="visible", timeout=5000)
        pg.keyboard.press("Backspace")
        pg.wait_for_selector("#oAttrs .evhBtns", state="visible", timeout=5000)
        self.assertIsNone(self.obj(pg)["mine"]["match"])
        self.assertEqual(self.obj(pg)["pos"], o6["pos"])
        self.assertEqual(self.errors, [])

    def test_v2_how_to_judge_sheet(self):
        _, _, pg = self.open_v2(splash_seen=False, width=390, height=844)
        self.assertTrue(pg.is_visible("#evhSplash"))
        self.assertIn("down on small dense particles", pg.inner_text("#evhSplash"))
        self.assertFalse(pg.is_visible("#oInstr"))                     # the sheet holds the text: no auto-open
        self.assertFalse(pg.is_visible("#oInfoBtn"))
        pg.click("#evhHow")
        self.assertTrue(pg.is_visible("#evhGuide"))
        self.assertTrue(self.obj(pg)["evh"]["guide"])
        secs = pg.eval_on_selector_all("#evhGdBody section h3", "els => els.map(e => e.textContent)")
        self.assertEqual(secs, ["Match →", "Nope ←", "Small & dense ↓", "It's complicated ↑"])
        figs = pg.eval_on_selector_all("#evhGdBody figure", "els => els.map(e => [e.dataset.cat, "
                                       "e.querySelector('img').getAttribute('src'), e.textContent])")
        self.assertEqual([f[0] for f in figs], ["single", "ml", "nope", "nope", "dense", "unsure"])   # bad cat / src dropped
        self.assertTrue(all(f[1].startswith("images/ex_") for f in figs))
        self.assertEqual(figs[1][2], "ml example M1")
        self.assertEqual(self.obj(pg)["evh"]["examples"], 6)
        pg.wait_for_function("[...document.querySelectorAll('#evhGdBody img')].every(i => i.complete && i.naturalWidth > 0)",
                             timeout=5000)
        self.assertIn("Unsure is useful", pg.inner_text("#evhGdBody .guide"))
        pg.keyboard.press("Escape")                                   # closes the sheet, back on the splash
        self.assertFalse(pg.is_visible("#evhGuide"))
        self.assertTrue(pg.is_visible("#evhSplash"))
        pg.click("#evhGo")
        pg.click("#evhHelp")                                          # the ? button
        self.assertTrue(pg.is_visible("#evhGuide"))
        pg.keyboard.press("y")                                        # keys answer nothing behind the sheet
        pg.wait_for_timeout(300)
        self.assertEqual(self.obj(pg)["done"], 0)
        pg.click("#evhGdX")
        self.assertFalse(pg.is_visible("#evhGuide"))
        pg.keyboard.press("?")
        self.assertTrue(pg.is_visible("#evhGuide"))
        self.assertEqual(self.errors, [])

    def test_v1_and_flag_off_unchanged_by_v2(self):
        _, srv, pg = self.open(instructions=GUIDE_TEXT, splash_seen=False)    # a v1 set with instructions
        self.assertFalse(pg.is_visible("#evhHow"))
        self.assertTrue(pg.is_visible("#oInstr"))                      # auto-opened once, as before
        pg.click("#evhGo")
        pg.keyboard.press("Escape")
        o = self.obj(pg)
        self.assertFalse(o["evh"]["dense"])
        self.assertFalse(o["evh"]["guideOn"])
        self.assertFalse(pg.is_visible("#evhHelp"))
        self.assertTrue(pg.is_visible("#oInfoBtn"))
        self.assertEqual(pg.eval_on_selector_all("#oAttrs .evhBtns button", "els => els.map(e => e.dataset.v)"),
                         ["no", "unsure", "yes"])
        for k in ("d", "ArrowDown"):
            pg.keyboard.press(k)
        b = pg.locator("#ocv").bounding_box()
        touch_swipe(pg, b["x"] + b["width"] / 2, b["y"] + b["height"] / 2 - 60, 10, 170, sel="#ocv")
        pg.wait_for_timeout(500)
        self.assertEqual((self.obj(pg)["pos"], self.obj(pg)["done"]), (o["pos"], 0))
        pg.keyboard.press("ArrowRight")                                # v1 Match: no chips, the next card at once
        self.next_card(pg, o["pos"])
        self.assertEqual(pg.locator("#oAttrs .evhKind").count(), 0)
        self.assertEqual(self.errors, [])
        # flag off with the v2 schema: the plain card, its three questions, no eVHarmony element
        _, _, pg2 = self.open_v2(app=None)
        self.assertNotIn("app", self.obj(pg2))
        heads = pg2.eval_on_selector_all("#oAttrs h3 > span:first-child", "els => els.map(e => e.textContent)")
        self.assertEqual(heads, ["Match?", "What is it?", "Count my answers"])
        for sel in ("#oEvh", "#evhGuide", "#evhSplash", "#evhHelp"):
            self.assertFalse(pg2.is_visible(sel), sel)
        pg2.keyboard.press("d")                                        # the schema's own key: an ordinary tap
        pg2.wait_for_function("__review.obj().mine.match === 'dense'", timeout=5000)

    def test_flag_off_is_the_plain_card(self):
        _, srv, pg = self.open(app=None)
        o = self.obj(pg)
        self.assertNotIn("app", o)
        self.assertEqual(o["attrs"], ["match", "consent"])
        heads = pg.eval_on_selector_all("#oAttrs h3 > span:first-child", "els => els.map(e => e.textContent)")
        self.assertEqual(heads, ["Match?", "Count my answers"])
        for sel in ("#oEvh", "#oBio", "#evhSplash", "#evhReveal", "#oMatch"):
            self.assertFalse(pg.is_visible(sel), sel)
        self.assertFalse(pg.evaluate("document.getElementById('obj').classList.contains('evh')"))
        for sel in ("#oListBtn", "#oItem", "#oWho", "#oTitle"):
            self.assertTrue(pg.is_visible(sel), sel)
        self.assertEqual(pg.evaluate("__review.state().vc"), {**pg.evaluate("__review.state().vc"), "auto": False,
                                                              "smooth": 0})
        pg.keyboard.press("y")          # the schema's own key: an ordinary attribute tap, no swipe / fly-off
        self.settle(pg)
        self.assertEqual(self.obj(pg)["mine"]["match"], "yes")
        self.assertEqual(self.obj(pg)["pos"], 0)
        self.assertEqual(self.errors, [])

    def test_swipes_and_keys_write_match(self):
        data, srv, pg = self.open()
        self.assertTrue(pg.is_visible("#oEvh"))
        self.assertFalse(pg.is_visible("#oListBtn"))
        b = pg.locator("#ocv").bounding_box()
        x, y = b["x"] + b["width"] / 2, b["y"] + b["height"] / 2
        want = []
        for dx, dy, v in ((160, 0, "yes"), (-160, 0, "no"), (0, -160, "unsure")):
            o = self.obj(pg)
            touch_swipe(pg, x, y, dx, dy, sel="#ocv")
            self.next_card(pg, o["pos"])
            want.append((o["item"], v))
        for k, v in (("ArrowRight", "yes"), ("n", "no"), ("ArrowUp", "unsure")):
            o = self.obj(pg)
            pg.keyboard.press(k)
            self.next_card(pg, o["pos"])
            want.append((o["item"], v))
        self.settle(pg)
        for it, v in want:
            self.assertEqual(self.match_of(srv, it), v, it)
        keys = {k for f in srv.records.values() for k in f["dec"]}
        self.assertTrue(all(k.startswith("a:") and k.endswith(":match@rev1") for k in keys), keys)
        self.assertEqual(self.obj(pg)["done"], 6)
        # a pen swipe answers too; a short touch drag snaps back and saves nothing
        o = self.obj(pg)
        touch_swipe(pg, x, y, 40, 0, sel="#ocv")
        pg.wait_for_timeout(300)
        self.assertEqual(self.obj(pg)["pos"], o["pos"])
        touch_swipe(pg, x, y, 160, 0, sel="#ocv", kind="pen")
        self.next_card(pg, o["pos"])
        self.settle(pg)
        self.assertEqual(self.match_of(srv, o["item"]), "yes")
        # a mouse drag pans the image and answers nothing
        o = self.obj(pg)
        pg.mouse.move(x, y)
        pg.mouse.down()
        pg.mouse.move(x + 150, y + 20, steps=6)
        pg.mouse.up()
        pg.wait_for_timeout(300)
        o2 = self.obj(pg)
        self.assertEqual(o2["pos"], o["pos"])
        self.assertEqual(o2["evh"]["pan"], [150, 20])
        self.assertIsNone(o2["mine"]["match"])
        self.assertEqual(self.errors, [])

    def test_reveal_at_ten_and_consent(self):
        data, srv, pg = self.open()
        first = data["items"][0]
        for n in range(9):
            o = self.obj(pg)
            pg.keyboard.press("y" if n % 2 else "n")
            self.next_card(pg, o["pos"])
            self.assertFalse(pg.is_visible("#evhReveal"), n)
        pg.keyboard.press("y")
        pg.wait_for_selector("#evhReveal", state="visible", timeout=5000)
        self.assertEqual(self.obj(pg)["done"], 10)
        self.assertIsNone(self.obj(pg)["evh"]["practice"])          # r17d: no D.practice = the reveal at 10, as before
        self.assertIn("Plot twist: you just reviewed 10 vesicle candidates for the CryoEV model. A few cards had "
                      "known answers. Count my answers?", pg.inner_text("#evhRvTx"))
        self.assertTrue(pg.is_visible("#evhRvYes") and pg.is_visible("#evhRvNo"))
        self.assertFalse(pg.is_visible("#evhRvKeep"))   # "Keep swiping" only after the answer
        pg.keyboard.press("y")            # keys don't answer behind the reveal
        pg.wait_for_timeout(300)
        self.assertEqual(self.obj(pg)["done"], 10)
        pg.click("#evhRvYes")
        self.settle(pg)
        self.assertTrue(pg.is_visible("#evhRvKeep"))
        dec = srv.records[first["image"] + ".png"]["dec"]
        self.assertEqual(dec[f"a:{first['obj']}:consent@rev1"]["v"], "yes")
        self.assertEqual(self.obj(pg)["evh"]["consent"], "yes")
        pg.click("#evhRvKeep")
        self.assertFalse(pg.is_visible("#evhReveal"))
        o = self.obj(pg)
        pg.keyboard.press("y")            # the 11th answer: not offered again
        self.next_card(pg, o["pos"])
        pg.wait_for_timeout(300)
        self.assertFalse(pg.is_visible("#evhReveal"))
        self.assertEqual(self.errors, [])

    def test_reveal_reoffered_when_unanswered(self):
        _, srv, pg = self.open()
        for n in range(10):
            o = self.obj(pg)
            pg.keyboard.press("n")
            self.next_card(pg, o["pos"])
        pg.wait_for_selector("#evhReveal", state="visible", timeout=5000)
        pg.keyboard.press("Escape")       # later
        self.assertFalse(pg.is_visible("#evhReveal"))
        self.assertIsNone(self.obj(pg)["evh"]["consent"])
        pg.keyboard.press("y")
        pg.wait_for_selector("#evhReveal", state="visible", timeout=5000)
        pg.click("#evhRvNo")
        self.settle(pg)
        self.assertEqual(self.obj(pg)["evh"]["consent"], "no")

    def test_size_line(self):
        _, _, pg = self.open()
        pg.wait_for_function("__review.obj().evh.size !== null", timeout=5000)
        o = self.obj(pg)
        # imgA g1 is a 28-gon of radius 110 px: ~2 x 110 x 1.5 nm, a little under for the polygon
        self.assertTrue(300 <= o["evh"]["size"] <= 330, o["evh"]["size"])
        self.assertIn(f"~{o['evh']['size']} nm", pg.inner_text("#oBio"))
        _, _, pg2 = self.open(blind=True)          # blind: never a size, though nm/px is known
        pg2.wait_for_timeout(500)
        self.assertIsNone(self.obj(pg2)["evh"]["size"])
        self.assertNotIn(" nm", pg2.inner_text("#oBio"))
        self.assertTrue(pg2.inner_text("#oBio").strip())   # the bio is still there
        _, _, pg3 = self.open(meta=None)           # no nm/px (no /api/meta): no size
        pg3.wait_for_timeout(500)
        self.assertIsNone(self.obj(pg3)["evh"]["size"])
        self.assertNotIn(" nm", pg3.inner_text("#oBio"))

    def test_splash_once_nickname_and_contrast_default(self):
        _, _, pg = self.open(splash_seen=False)
        self.assertTrue(pg.is_visible("#evhSplash"))
        self.assertIn("Find your perfect membrane match", pg.inner_text("#evhSplash"))
        self.assertEqual(pg.evaluate("__review.state().vc.auto"), True)
        self.assertEqual(pg.evaluate("__review.state().vc.smooth"), 2)
        pg.fill("#evhNick", "Bilayer<b>Babe")
        pg.click("#evhGo")
        self.assertFalse(pg.is_visible("#evhSplash"))
        self.assertEqual(self.obj(pg)["evh"]["nick"], "BilayerbBabe")
        self.assertIn("BilayerbBabe", pg.inner_text("#oEvhNick"))
        self.assertEqual(self.obj(pg)["evh"]["splash"], False)
        self.assertEqual(pg.evaluate("__review.state().who"), "rev1")   # identity unchanged
        pg.reload()
        pg.wait_for_function("window.__review && __review.obj()", timeout=15000)
        self.assertFalse(pg.is_visible("#evhSplash"))
        # a saved contrast setting wins over the eVHarmony default
        _, _, pg2 = self.open(init="try { localStorage.setItem('cryoev-review:contrast:rev1', "
                                    "JSON.stringify({b: 0, c: 10, g: 1, auto: false, smooth: 0})); } catch (e) {}")
        vc = pg2.evaluate("__review.state().vc")
        self.assertEqual((vc["auto"], vc["smooth"], vc["c"]), (False, 0, 10))

    def test_framing_and_bio_without_sample(self):
        for width, height in ((390, 844), (1280, 800)):
            data, _, pg = self.open(width=width, height=height)
            pg.wait_for_function("__review.obj().evh.zoom !== null", timeout=5000)
            cw = pg.evaluate("document.getElementById('ocv').getBoundingClientRect().width")
            seen = set()
            for k in range(len(data["items"])):
                o = self.obj(pg)
                z = o["evh"]["zoom"]
                self.assertLessEqual(z, 1.25 + 1e-9)        # never > 2.5 native px per screen px at DPR 2
                bio = pg.inner_text("#oBio")
                self.assertNotIn("Zed", bio)
                self.assertNotIn("Person", bio)
                seen.add(bio)
                if o["item"]["image"] == "imgA" and o["item"]["obj"] == "g1":   # a 220 px circle
                    d = 220 * z
                    self.assertLessEqual(d, cw / 3 + 2)          # about 1/3 of the card, or less when capped
                    self.assertTrue(z == 1.25 or abs(d - cw / 3) < 0.15 * cw, (z, d, cw))
                if k < len(data["items"]) - 1:
                    pg.click("#oNext")
                    pg.wait_for_function(f"__review.obj().pos === {k + 1}", timeout=5000)
            self.assertGreater(len(seen), 1)
            self.assertEqual(self.errors, [])

    def test_phone_light_and_dark(self):
        for scheme in ("light", "dark"):
            _, _, pg = self.open(width=390, height=844, scheme=scheme)
            for sel in ("#oEvh", "#ocv", "#oBio", "#oAttrs .evhBtns"):
                self.assertTrue(pg.is_visible(sel), (scheme, sel))
            self.assertLessEqual(pg.evaluate("document.documentElement.scrollWidth"), 390)
            hot = pg.evaluate("getComputedStyle(document.getElementById('obj')).getPropertyValue('--hot').trim()")
            self.assertEqual(hot, "#2dd4bf" if scheme == "dark" else "#0f766e")
            self.assertEqual(self.errors, [], scheme)

    def test_refused_saves_show_banner_then_upload(self):
        shots = os.environ.get("R17B_SHOTS")
        _, srv, pg = self.open()
        srv.refuse = "not a reviewer"
        self.assertFalse(pg.is_visible("#oSaveErr"))
        want = []
        for _ in range(3):
            o = self.obj(pg)
            pg.keyboard.press("ArrowRight")
            self.next_card(pg, o["pos"])
            want.append(o["item"])
        pg.wait_for_selector("#oSaveErr", state="visible", timeout=8000)
        txt = pg.inner_text("#oSaveErr")
        self.assertIn("aren't being saved", txt)
        self.assertIn("the server said: not a reviewer", txt)
        self.assertIn("3 answers waiting", txt)
        self.assertEqual(pg.get_attribute("#oSaved", "data-state"), "error")
        self.assertNotIn("Saved ✓", pg.inner_text("#oSaved"))
        self.assertIn("3 answers waiting", pg.inner_text("#oSaved"))
        self.assertGreater(srv.refused, 0)
        self.assertEqual(srv.records, {})
        if shots:
            pg.screenshot(path=str(Path(shots) / "r17b_evh_not_saved.png"))
        # a reload keeps the drafts (localStorage) and still says so
        pg.reload()
        pg.wait_for_function("window.__review && __review.obj() && __review.state().server", timeout=15000)
        pg.wait_for_selector("#oSaveErr", state="visible", timeout=8000)
        self.assertIn("3 answers waiting", pg.inner_text("#oSaveErr"))
        # the server accepts: the drafts upload, the banner goes, "Saved ✓"
        srv.refuse = None
        pg.evaluate("dispatchEvent(new Event('online'))")
        self.settle(pg)
        pg.wait_for_selector("#oSaveErr", state="hidden", timeout=5000)
        self.assertEqual(pg.inner_text("#oSaved"), "Saved ✓")
        for it in want:
            self.assertEqual(self.match_of(srv, it), "yes", it)
        if shots:
            pg.screenshot(path=str(Path(shots) / "r17b_evh_saved.png"))
        self.assertEqual(self.errors, [])

    def test_view_only_when_not_a_reviewer(self):
        """r17b: REVIEW_IDENTITY.answer === false (multi_review_server: not a reviewer): a View only banner; swipes,
        keys and buttons save nothing, right / left still browse. Control: the same identity without the flag answers."""
        ident = "window.REVIEW_IDENTITY = {who: 'rev1', locked: true%s};"
        for app in ("evharmony", None):
            _, srv, pg = self.open(app=app, init=ident % ", answer: false")
            self.assertTrue(pg.is_visible("#oViewOnly"), app)
            self.assertIn("not a reviewer", pg.inner_text("#oViewOnly"))
            o = self.obj(pg)
            self.assertTrue(o["viewOnly"])
            self.assertFalse(o["can"])
            pg.keyboard.press("y")
            if app:
                pg.keyboard.press("ArrowRight")
                pg.wait_for_function(f"__review.obj().pos !== {o['pos']}", timeout=5000)
                b = pg.locator("#ocv").bounding_box()
                touch_swipe(pg, b["x"] + b["width"] / 2, b["y"] + b["height"] / 2, 160, 0, sel="#ocv")
                pg.wait_for_timeout(400)
            else:
                pg.locator("#oAttrs button").first.dispatch_event("click")
                pg.click("#oNext")
                pg.wait_for_function(f"__review.obj().pos !== {o['pos']}", timeout=5000)
            pg.wait_for_timeout(1200)                     # a save would have gone by now (800 ms)
            self.assertEqual(srv.puts, [], app)
            self.assertEqual(self.obj(pg)["done"], 0)
            self.assertEqual(self.errors, [])
        _, srv, pg = self.open(init=ident % "")               # control: no flag (an older server): answers as before
        self.assertFalse(pg.is_visible("#oViewOnly"))
        self.assertFalse(self.obj(pg)["viewOnly"])
        pg.keyboard.press("ArrowRight")
        self.settle(pg)
        self.assertEqual(len([k for f in srv.records.values() for k in f["dec"]]), 1)

    def test_refused_saves_plain_card_not_saved(self):
        _, srv, pg = self.open(app=None)
        srv.refuse = "not a reviewer"
        pg.keyboard.press("y")
        pg.wait_for_function("document.getElementById('oSaved').dataset.state === 'error'", timeout=8000)
        self.assertNotIn("Saved ✓", pg.inner_text("#oSaved"))
        self.assertFalse(pg.is_visible("#oSaveErr"))              # the plain card has its sync chip instead
        self.assertIn("not saved", pg.inner_text("#oSync"))
        self.assertEqual(pg.get_attribute("#oSync", "title"), "The server said: not a reviewer")
        srv.refuse = None
        pg.evaluate("dispatchEvent(new Event('online'))")
        self.settle(pg)
        pg.wait_for_function("document.getElementById('oSaved').dataset.state === 'saved'", timeout=5000)
        self.assertEqual(self.errors, [])

    # ---- r17c v3
    def open_v3(self, **kw):
        ex = [{"src": f"images/ex_{i}.jpg", "cat": c, "caption": cap} for i, c, cap in (
            ("S3", "single", "Single vesicle"), ("M4", "ml", "Any part of a multilayer counts"),
            ("N5", "nope", "Ice"), ("C5", "unsure", "Maybe a bilayer, hard to tell"))]
        return self.open(schema=EVH_V3, examples=ex, instructions=GUIDE_TEXT, **kw)

    def test_v3_three_answers_and_follow_ups(self):
        _, srv, pg = self.open_v3(width=390, height=844)
        self.assertEqual(pg.eval_on_selector_all("#oAttrs .evhBtns button", "els => els.map(e => e.dataset.v)"),
                         ["no", "unsure", "yes"])
        self.assertFalse(self.obj(pg)["evh"]["dense"])
        o0 = self.obj(pg)
        pg.keyboard.press("d")                                        # no 4th answer: d / ↓ / swipe down do nothing
        pg.keyboard.press("ArrowDown")
        b = pg.locator("#ocv").bounding_box()
        touch_swipe(pg, b["x"] + b["width"] / 2, b["y"] + b["height"] / 2 - 60, 10, 170, sel="#ocv")
        pg.wait_for_timeout(400)
        self.assertEqual((self.obj(pg)["pos"], self.obj(pg)["done"]), (o0["pos"], 0))
        # Match: kind chips + the Outline off toggle; toggle, then Multilayer -> both saved, the next card
        o = self.obj(pg)
        pg.keyboard.press("ArrowRight")
        pg.wait_for_selector("#oAttrs .evhKind", state="visible", timeout=5000)
        self.assertEqual(self.obj(pg)["evh"]["fuV"], "yes")
        self.assertEqual(pg.eval_on_selector_all("#oAttrs .evhKind button", "els => els.map(e => e.textContent)"),
                         ["Single", "Multilayer", "✎ Outline off", "skip →"])
        pg.click("#oAttrs .evhKind button[data-attr='outline_off']")
        self.assertEqual(pg.get_attribute("#oAttrs .evhKind button[data-attr='outline_off']", "aria-pressed"), "true")
        self.assertEqual(pg.inner_text("#oAttrs .evhKind button.skip"), "next →")
        self.assertEqual(self.obj(pg)["pos"], o["pos"])               # a toggle doesn't move on
        pg.click("#oAttrs .evhKind button[data-v='multilayer']")
        self.next_card(pg, o["pos"])
        # Match, o (toggle), Enter: outline_off only
        o2 = self.obj(pg)
        pg.keyboard.press("y")
        pg.wait_for_selector("#oAttrs .evhKind", state="visible", timeout=5000)
        pg.keyboard.press("o")
        pg.keyboard.press("Enter")
        self.next_card(pg, o2["pos"])
        # Match, 1 = Single
        o3 = self.obj(pg)
        pg.keyboard.press("y")
        pg.wait_for_selector("#oAttrs .evhKind", state="visible", timeout=5000)
        pg.keyboard.press("1")
        self.next_card(pg, o3["pos"])
        # It's complicated: the why chips; pick Small and dense
        o4 = self.obj(pg)
        pg.keyboard.press("ArrowUp")
        pg.wait_for_selector("#oAttrs .evhKind.why", state="visible", timeout=5000)
        self.assertEqual(pg.eval_on_selector_all("#oAttrs .evhKind button", "els => els.map(e => e.textContent)"),
                         ["Faint bilayer", "Small and dense", "Cut off at edge", "Other", "skip →"])
        pg.click("#oAttrs .evhKind button[data-v='dense']")
        self.next_card(pg, o4["pos"])
        # It's complicated, then a swipe left: re-answered no, no why
        o5 = self.obj(pg)
        pg.keyboard.press("u")
        pg.wait_for_selector("#oAttrs .evhKind.why", state="visible", timeout=5000)
        touch_swipe(pg, b["x"] + b["width"] / 2, b["y"] + b["height"] / 2, -170, 0, sel="#ocv")
        self.next_card(pg, o5["pos"])
        # Nope: no follow-ups, the next card at once
        o6 = self.obj(pg)
        pg.keyboard.press("n")
        self.next_card(pg, o6["pos"])
        self.settle(pg)
        got = [tuple(self.match_of(srv, x["item"], a) for a in ("match", "kind", "outline_off", "why"))
               for x in (o, o2, o3, o4, o5, o6)]
        self.assertEqual(got, [("yes", "multilayer", "yes", None), ("yes", None, "yes", None), ("yes", "single", None, None),
                               ("unsure", None, None, "dense"), ("no", None, None, None), ("no", None, None, None)])
        self.assertLessEqual(pg.evaluate("document.documentElement.scrollWidth"), 390)
        self.assertEqual(self.errors, [])

    def test_v3_how_to_judge(self):
        _, _, pg = self.open_v3(splash_seen=False)
        self.assertIn("up when it's complicated", pg.inner_text("#evhSplash"))
        self.assertNotIn("down", pg.inner_text("#evhSplash"))
        pg.click("#evhHow")
        secs = pg.eval_on_selector_all("#evhGdBody section h3", "els => els.map(e => e.textContent)")
        self.assertEqual(secs, ["Match →", "Nope ←", "It's complicated ↑"])
        txt = pg.inner_text("#evhGdBody")
        for t in ("any part of a multilayer counts", "Outline off", "Ice contamination", "Small and dense",
                  "Anything not clearly yes or no", "Maybe a bilayer, hard to tell"):
            self.assertIn(t, txt)
        self.assertEqual(pg.eval_on_selector_all("#evhGdBody h4", "els => els.map(e => e.textContent)"),
                         ["Single", "Multilayer"])
        self.assertEqual(self.errors, [])


    # ---- r17d: practice cards
    def attr_puts(self, srv):
        return [k for _f, rec in srv.puts for k in (rec.get("dec") or {}) if k.startswith("a:")]

    def practice_answer(self, pg, key):
        k = self.obj(pg)["evh"]["practice"]["k"]
        pg.keyboard.press(key)
        pg.wait_for_function(f"(p => p.k !== {k} || p.done || p.fb)(__review.obj().evh.practice)", timeout=5000)

    def to_reveal(self, pg, keys=("y", "n", "y")):
        for key in keys:
            self.practice_answer(pg, key)
        pg.wait_for_selector("#evhReveal", state="visible", timeout=5000)

    def test_practice_then_reveal_then_redo(self):
        data, srv, pg = self.open_v3(practice=True, width=390, height=844)
        p = self.obj(pg)["evh"]["practice"]
        self.assertEqual((p["on"], p["k"], p["n"], p["card"]), (True, 0, 3, data["practice"][0]["id"]))
        self.assertEqual(pg.inner_text("#oPos"), "Practice 1 / 3")
        self.assertFalse(pg.is_visible("#evhReveal"))
        self.practice_answer(pg, "ArrowRight")                     # no follow-up chips on a practice card
        self.assertEqual(self.obj(pg)["evh"]["practice"]["card"], data["practice"][1]["id"])
        self.assertFalse(pg.is_visible("#oAttrs .evhKind"))
        b = pg.locator("#ocv").bounding_box()
        touch_swipe(pg, b["x"] + b["width"] / 2, b["y"] + b["height"] / 2, -170, 0, sel="#ocv")   # a swipe left = Nope
        pg.wait_for_function("__review.obj().evh.practice.k === 2", timeout=5000)
        pg.keyboard.press("Backspace")                             # no undo in practice
        self.assertEqual(self.obj(pg)["evh"]["practice"]["k"], 2)
        self.practice_answer(pg, "u")
        pg.wait_for_selector("#evhReveal", state="visible", timeout=5000)
        p = self.obj(pg)["evh"]["practice"]
        self.assertEqual((p["done"], p["answers"]), (True, ["yes", "no", "unsure"]))
        self.assertEqual(self.attr_puts(srv), [])                  # nothing of the practice was sent
        self.assertIn("practice cards", pg.inner_text("#evhRvTx"))
        self.assertIn("real vesicle candidate", pg.inner_text("#evhRvTx"))
        self.assertTrue(pg.is_visible("#evhRvGuide"))
        self.assertEqual(pg.eval_on_selector_all("#evhRvGuide section h3", "els => els.map(e => e.textContent)"),
                         ["Match →", "Nope ←", "It's complicated ↑"])
        self.assertIn(GUIDE_TEXT, pg.inner_text("#evhRvGuide"))
        self.assertIn("Count my answers?", pg.inner_text("#evhRvQ"))
        self.assertFalse(pg.is_visible("#evhRvPr"))                # Redo / Start only after the consent answer
        pg.keyboard.press("Escape")                                # the practice reveal stays
        pg.keyboard.press("y")
        self.assertTrue(pg.is_visible("#evhReveal"))
        self.assertLessEqual(pg.evaluate("document.documentElement.scrollWidth"), 390)
        pg.click("#evhRvYes")
        self.settle(pg)
        first = data["items"][0]
        self.assertEqual(self.match_of(srv, first, "consent"), "yes")
        self.assertEqual(self.obj(pg)["evh"]["consent"], "yes")
        self.assertTrue(pg.is_visible("#evhRvPr"))
        self.assertIn("The practice cards don't count.", pg.inner_text("#evhRvPr"))
        self.assertEqual(pg.inner_text("#evhRvRedo"), "Redo the 3 practice cards")
        pg.click("#evhRvRedo")
        self.assertFalse(pg.is_visible("#evhReveal"))
        p = self.obj(pg)["evh"]["practice"]
        self.assertEqual((p["on"], p["redo"], p["k"], p["card"]), (True, True, 0, data["practice"][0]["id"]))
        self.practice_answer(pg, "n")                              # wrong: the card stays, our answer shows
        pg.wait_for_selector("#oAttrs .evhFb", state="visible", timeout=5000)
        self.assertEqual(self.obj(pg)["evh"]["practice"]["k"], 0)
        fb = pg.inner_text("#oAttrs .evhFb")
        self.assertIn("You: Nope", fb)
        self.assertIn("Our answer: Match · Single — A round ring: one vesicle.", fb)
        pg.click("#oAttrs .evhFb button")
        pg.wait_for_function("__review.obj().evh.practice.k === 1 && !__review.obj().evh.practice.fb", timeout=5000)
        self.practice_answer(pg, "n")                              # right
        self.assertIn("✓ You: Nope", pg.inner_text("#oAttrs .evhFb"))
        self.assertIn("Our answer: Nope — No membrane ring.", pg.inner_text("#oAttrs .evhFb"))
        pg.keyboard.press("Enter")
        pg.wait_for_function("__review.obj().evh.practice.k === 2 && !__review.obj().evh.practice.fb", timeout=5000)
        self.practice_answer(pg, "y")
        self.assertIn("Our answer: Match — Another clear ring.", pg.inner_text("#oAttrs .evhFb"))  # no kind given
        self.assertEqual(pg.inner_text("#oAttrs .evhFb button"), "Start for real →")
        pg.click("#oAttrs .evhFb button")
        pg.wait_for_function("!__review.obj().evh.practice.on", timeout=5000)
        o = self.obj(pg)
        self.assertEqual((o["pos"], o["item"]["id"]), (0, first["id"]))   # the real queue from its first card
        self.assertFalse(pg.is_visible("#evhReveal"))
        self.assertEqual([k for k in self.attr_puts(srv) if ":match@" in k], [])
        pg.keyboard.press("n")                                     # now an answer saves
        self.next_card(pg, 0)
        self.settle(pg)
        self.assertEqual(self.match_of(srv, first), "no")
        self.assertFalse(pg.is_visible("#evhReveal"))
        self.assertEqual(self.errors, [])

    def test_practice_start_for_real(self):
        data, srv, pg = self.open_v3(practice=True)
        self.to_reveal(pg)
        pg.click("#evhRvNo")
        self.settle(pg)
        self.assertEqual(self.match_of(srv, data["items"][0], "consent"), "no")
        self.assertIn("won't be counted", pg.inner_text("#evhRvQ"))
        pg.click("#evhRvReal")
        pg.wait_for_function("!__review.obj().evh.practice.on", timeout=5000)
        o = self.obj(pg)
        self.assertEqual((o["pos"], o["item"]["id"], o["n"]), (0, data["items"][0]["id"], len(data["items"])))
        self.assertFalse(pg.is_visible("#evhReveal"))
        self.assertNotIn("Practice", pg.inner_text("#oPos"))
        self.assertEqual([k for k in self.attr_puts(srv) if ":consent@" not in k], [])
        self.assertEqual(self.errors, [])

    def test_practice_skipped_with_consent(self):
        data, srv, pg = self.open_v3(practice=True, consent="yes")
        pg.wait_for_function("!__review.obj().evh.practice.on", timeout=5000)
        o = self.obj(pg)
        self.assertEqual((o["item"]["id"], o["evh"]["consent"]), (data["items"][0]["id"], "yes"))
        self.assertNotIn("Practice", pg.inner_text("#oPos"))
        pg.keyboard.press("n")
        self.next_card(pg, o["pos"])
        self.settle(pg)
        self.assertEqual(self.match_of(srv, data["items"][0]), "no")
        self.assertFalse(pg.is_visible("#evhReveal"))
        self.assertEqual(self.errors, [])


if __name__ == "__main__":
    unittest.main()
