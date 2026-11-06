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


def with_cand_conf(v):
    """An init script that sets REVIEW_DATA.cand_conf as data.js assigns it (data.js on disk unchanged)."""
    def setup(context, page):
        page.add_init_script("Object.defineProperty(window, 'REVIEW_DATA', {configurable: true, "
                             "get() { return this.__rd; }, set(d) { d.cand_conf = %s; this.__rd = d; }});" % json.dumps(v))
    return setup


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
        self.assertIn("4 labels, 1 predictions shown", page.inner_text("#selBox"))  # D, 0.30: hidden at 0.45
        phone = self.open("&who=tester", **trb.PhoneTests.PHONE)
        self.wait_card(phone)
        queue = [(self.file_of(c), c["key"]) for c in self.state(phone)["tri"]["queue"]]
        self.assertIn(("0001.png", "p0"), queue)  # D, 0.30: left out at the old 0.35 default
        self.assertEqual(self.errors, [])


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
