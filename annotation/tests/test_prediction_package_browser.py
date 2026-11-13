"""
Browser tests (Playwright, headless Edge or Chrome; skipped without one) for the two review-tool changes made for
prediction packages (annotation/review_tool/index.html):
  - data.js cand_conf (set by annotation/prediction_package.py to 0.25): the Editor's prediction slider and the
    cards' confidence filter start there, so every candidate shows; a package without it (every real package)
    keeps the old defaults (slider 0.45, cards 0.35, the same four filter options);
  - data-derived strings (file, split, categories, run) are escaped where the page builds HTML.

  python -m unittest discover -s annotation/tests -v
"""
import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))
import make_fixture  # noqa: E402
from test_review_server import LiveServer, load_data  # noqa: E402
import test_review_tool_browser as trb  # noqa: E402  (module: its TestCases must not be collected twice)

from annotation import prediction_package as pp  # noqa: E402
from test_prediction_package import PKG, PS  # noqa: E402

OLD_OPTIONS = ["0.25", "0.35", "0.45", "0.6"]  # the cards' filter options before cand_conf
CONTROLS = """() => ({slider: document.getElementById('conf').value, sliderMin: document.getElementById('conf').min,
    out: document.getElementById('confOut').textContent, filter: document.getElementById('triConf').value,
    options: [...document.getElementById('triConf').options].map(o => o.value)})"""


sys.path.append(str(Path(__file__).resolve().parents[2] / "tests"))
from cryoev_testkit import browser  # noqa: E402  (the gate skips these when no UI file changed)


def with_cand_conf(v):
    """An init script that sets REVIEW_DATA.cand_conf as data.js assigns it (data.js on disk unchanged)."""
    def setup(context, page):
        page.add_init_script("Object.defineProperty(window, 'REVIEW_DATA', {configurable: true, "
                             "get() { return this.__rd; }, set(d) { d.cand_conf = %s; this.__rd = d; }});" % json.dumps(v))
    return setup


@browser
class RealPackageDefaultsTests(trb.Base):
    """The fixture is a real-package data.js (no cand_conf): the controls are exactly as before."""

    def test_defaults_unchanged_without_cand_conf(self):
        page = self.open()
        self.assertNotIn("cand_conf", load_data(self.srv.pkg))
        self.assertEqual(page.evaluate(CONTROLS), {"slider": "0.45", "sliderMin": "0.25", "out": "0.45",
                                                   "filter": "0.35", "options": OLD_OPTIONS})
        self.assertEqual(self.errors, [])

    def test_cand_conf_not_among_the_options(self):
        page = self.open(setup=with_cand_conf(0.3))
        self.assertEqual(page.evaluate(CONTROLS), {"slider": "0.3", "sliderMin": "0.25", "out": "0.30",
                                                   "filter": "0.3", "options": ["0.3"] + OLD_OPTIONS})
        page = self.open(setup=with_cand_conf(0.2))
        c = page.evaluate(CONTROLS)
        self.assertEqual((c["slider"], c["sliderMin"], c["filter"]), ("0.2", "0.2", "0.2"))


WORDS = """() => ({gt: document.getElementById('lblGtWrap').innerText.trim(), gtTip: document.getElementById('lblGtWrap').title,
    pr: document.getElementById('lblPrWrap').innerText.trim(), prTip: document.getElementById('lblPrWrap').title,
    help: document.querySelector('.help').innerText, flags: document.getElementById('flagLegend').innerText,
    btn: document.getElementById('toTri').innerText, btnTip: document.getElementById('toTri').title,
    title: document.title, keys: document.getElementById('triKeys').innerText,
    keysShown: !document.getElementById('triKeys').hidden && getComputedStyle(document.getElementById('triKeys')).display !== 'none'})"""
BRANDS = ("eVHarmony", "Gridr", "cryonder")


def with_fields(**kw):
    def setup(context, page):
        page.add_init_script("Object.defineProperty(window, 'REVIEW_DATA', {configurable: true, "
                             "get() { return this.__rd; }, set(d) { Object.assign(d, %s); this.__rd = d; }});" % json.dumps(kw))
    return setup


@browser
class LabelingPackageWordsTests(trb.Base):
    """Control: the lab's labeling package (no kind, no cand_conf) keeps the old words and the brand button."""

    def test_old_words_and_brand_button(self):
        page = self.open()
        w = page.evaluate(WORDS)
        self.assertEqual(w["gt"], "Labels")
        self.assertTrue(w["pr"].startswith("Predictions ≥"), w["pr"])
        self.assertEqual((w["gtTip"], w["prTip"]), ("", ""))
        self.assertIn("toggle predictions", w["help"])
        self.assertIn("Selected prediction:", w["help"])
        self.assertIn("delete label", w["help"])
        self.assertIn("label with no prediction", w["flags"])
        self.assertIn("Low conf EV label", w["flags"])
        self.assertIn("Sept", w["flags"])
        self.assertIn("Older packages have only the first three", w["flags"])
        self.assertIn("prediction", w["flags"])
        self.assertIn("Predictions:", page.inner_text("#progress"))
        page.evaluate("__review.render()")
        self.assertRegex(page.inner_text("#selBox"), r"\d+ labels, \d+ predictions shown")
        self.assertTrue(page.is_visible("#showSept"))
        self.assertIn(w["btn"], BRANDS)
        self.assertFalse(page.evaluate("document.getElementById('toTri').classList.contains('quick')"))
        page.click("#toTri")
        self.assertTrue(page.evaluate("document.getElementById('triKeys').hidden"))
        self.assertEqual(self.errors, [])


@browser
class PredictionWordsTests(trb.Base):
    """Prediction package: Counted / Unsure wording, Quick check button, key legend in the cards."""

    def test_wording_button_and_legend(self):
        page = self.open(setup=with_fields(kind="prediction", cand_conf=0.3, label_conf=0.5))
        w = page.evaluate(WORDS)
        self.assertEqual(w["gt"], "Counted")
        self.assertTrue(w["pr"].startswith("Unsure, not counted ≥"), w["pr"])
        self.assertIn("≥50% plus what you accepted or drew", w["gtTip"])
        self.assertIn("EVs counted, non-EVs counted separately, inner layers excluded.", w["gtTip"])
        self.assertEqual(w["prTip"], "Model outlines between 30% and 50%. Not counted unless you accept them.")
        self.assertIn("toggle unsure", w["help"])
        self.assertIn("unsure outline, undecided: not counted until you accept it", w["flags"])
        for gone in ("label with no prediction", "class disagrees", "Low conf EV label", "Sept", "Older packages"):
            self.assertNotIn(gone, w["flags"])
        self.assertIn("Selected unsure outline:", w["help"])
        self.assertIn("delete counted outline", w["help"])
        self.assertFalse(page.is_visible("#showSept"))
        self.assertTrue(w["btn"].startswith("Quick check"), w["btn"])
        self.assertEqual(w["btnTip"], "Go through unsure outlines one at a time: accept or reject")
        self.assertEqual(w["title"], "fixture_review")  # the package (D.title || D.id), no longer the fixed "EV Label Review"
        page.click("#toTri")
        w = page.evaluate(WORDS)
        self.assertIn(w["title"], BRANDS)  # document.title is still the brand
        # L: the Low conf key (round 7), listed because this server has the t cap (/api/ping caps)
        self.assertEqual(w["keys"], "→ / A accept · ← / X reject · ↑ / S skip · L accept as low conf / low conf · B back · Z undo · Esc close")
        self.assertTrue(w["keysShown"])
        page.select_option("#triMode", "cls")
        self.assertEqual(page.inner_text("#triKeys"), "1–8 pick class · → / A keep · ← / ↑ / S next · L low conf · B back · Z undo · Esc close")
        self.assertTrue(page.evaluate("document.getElementById('toTri').innerText").startswith("Quick check"))
        self.assertEqual(self.errors, [])

    def test_logo_cycle_keeps_quick_check(self):
        page = self.open(setup=with_fields(kind="prediction", cand_conf=0.3, label_conf=0.5))
        page.click("#toTri")
        before = page.evaluate("document.title")
        page.click("#logo")
        w = page.evaluate(WORDS)
        self.assertTrue(w["btn"].startswith("Quick check"), w["btn"])
        self.assertIn(w["title"], BRANDS)
        self.assertNotEqual(w["title"], before)
        self.assertEqual(self.errors, [])

    def test_fallback_ranges_and_hidden_on_touch(self):
        page = self.open(setup=with_fields(kind="prediction"))
        self.assertIn("between 25% and 50%", page.evaluate(WORDS)["prTip"])
        phone = self.open("&who=tester", setup=with_fields(kind="prediction"), **trb.PhoneTests.PHONE)
        self.assertTrue(phone.evaluate("!document.getElementById('tri').hidden"))
        self.assertFalse(phone.evaluate(WORDS)["keysShown"])  # coarse pointer: legend not displayed
        self.assertEqual(self.errors, [])


@browser
class PredictionPackageTests(trb.Base):
    """A package built by prediction_package.build from the golden set: the 0.30 candidate (D on 0001) shows."""
    card, file_of, wait_card = trb.PhoneReviewTests.card, trb.PhoneReviewTests.file_of, trb.PhoneReviewTests.wait_card

    @classmethod
    def make_server(cls):
        return LiveServer(build=lambda out: pp.build(PS, out.parent), name=PKG)

    def test_candidates_shown_in_editor_and_cards(self):
        self.assertEqual(load_data(self.srv.pkg)["cand_conf"], 0.25)
        page = self.open()
        self.assertEqual(page.evaluate(CONTROLS), {"slider": "0.25", "sliderMin": "0.25", "out": "0.25",
                                                   "filter": "0.25", "options": OLD_OPTIONS})
        self.go(page, "0001.png")
        self.assertIn("4 counted, 1 unsure shown", page.inner_text("#selBox"))  # D, 0.30: hidden at 0.45
        phone = self.open("&who=tester", **trb.PhoneTests.PHONE)
        self.wait_card(phone)
        queue = [(self.file_of(c), c["key"]) for c in self.state(phone)["tri"]["queue"]]
        self.assertIn(("0001.png", "p0"), queue)  # D, 0.30: left out at the old 0.35 default
        self.assertEqual(self.errors, [])


@browser
class EscapingTests(trb.Base):
    """Markup in a package's file name, split, category and run is shown as text, never parsed."""
    FILE, SPLIT, CAT, RUN = ("<img src=x onerror=window.__xss=1>.png", "<b id=xsplit>S</b>", "<i id=xcat>c</i>",
                             "<img src=y onerror=window.__xss2=1>")

    @classmethod
    def make_server(cls):
        def build(out):
            data = make_fixture.build(out)
            extra = dict(data["images"][2], file=cls.FILE, split=cls.SPLIT)  # zero flags: listed last
            # new lists: data["categories"] is make_fixture.CATEGORIES itself, shared with every other fixture build
            data = dict(data, images=data["images"] + [extra], categories=data["categories"] + [cls.CAT], run=cls.RUN)
            (out / "data.js").write_text("window.REVIEW_DATA = " + json.dumps(data) + ";\n", encoding="utf-8")
        return LiveServer(build=build)

    def test_markup_is_text(self):
        page = self.open()
        self.assertIsNone(page.evaluate("window.__xss || window.__xss2 || null"))
        self.assertIsNone(page.query_selector("#xsplit, #xcat"))
        items = page.inner_text("#items")
        self.assertIn(self.FILE, items)
        self.assertIn(self.SPLIT, items)
        self.assertIn(self.CAT, page.inner_text("#legend"))
        self.assertIn(self.RUN, page.inner_text("#progress"))
        self.go(page, self.FILE)
        self.assertIsNone(page.evaluate("window.__xss || window.__xss2 || null"))
        self.assertEqual(self.errors, [])


if __name__ == "__main__":
    unittest.main()
