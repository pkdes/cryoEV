"""
The review package picker (round 9, R2; portal/modules/reviews.py): card progress (the Quick check queue, not images),
access by owner or `reviewers`, "made from" without local paths, lab_ packages once the portal route takes them,
the multi server's excluded package never listed, and the rendered row.

  python -m unittest portal.tests.test_reviews_picker -v
"""
import json
import os
import re
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "annotation" / "tests"))
import make_fixture  # noqa: E402
from annotation import multi_review_server as MRS  # noqa: E402
from portal import app as APP  # noqa: E402
from portal.modules import Ctx  # noqa: E402
from portal.modules import reviews as REV  # noqa: E402

PRED = "pred_ps_20261112_abc123"
LAB = "lab_compare_fixture_20261112"
PRED_ONLY_RE = re.compile(r"^pred_ps_\d{8}_[0-9a-f]{6}\Z")                                     # app.py at main
LAB_RE = re.compile(r"^(?:pred_ps_\d{8}_[0-9a-f]{6}|lab_[A-Za-z0-9][A-Za-z0-9_.-]{0,100})\Z")  # the app.py change (report)


def user(uid, role="basic"):
    return {"id": uid, "role": role}


def build(root: Path, name: str, meta: dict) -> Path:
    pkg = root / name
    make_fixture.build(pkg)
    (pkg / "review_package.json").write_text(json.dumps(meta), encoding="utf-8")
    (pkg / ".review_key").write_text("k" * 20, encoding="utf-8")
    return pkg


def put(pkg: Path, file: str, dec: dict):
    (pkg / "edits").mkdir(exist_ok=True)
    (pkg / "edits" / f"{file}.json").write_text(json.dumps({"file": file, "edit": {"t": 1, "dec": dec}}), encoding="utf-8")
    os.utime(pkg / "edits", (time.time() + 5, time.time() + 5))  # the cache follows the folder's mtime


class PickerTests(unittest.TestCase):
    def setUp(self):
        for k in [k for k in os.environ if k.startswith("CRYOEV_") or k == MRS.EXCLUDE_ENV]:
            self.addCleanup(os.environ.__setitem__, k, os.environ.pop(k))
        self.tmp = Path(tempfile.mkdtemp(prefix="cj_picker_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.root = self.tmp / "review"
        self.root.mkdir()
        self.pred = build(self.root, PRED, {"kind": "prediction", "owner": "bas", "model": "v5_model", "n_images": 4})
        self.lab = build(self.root, LAB, {
            "kind": "compare", "owner": "pujan", "reviewers": ["bas"], "n_images": 4,
            "base": "C:\\Users\\someone\\CryoAI-laptop\\labels\\lv_20261030_0909\\labels.coco.json",
            "compare": {"a": {"kind": "labels", "coco": "C:\\Users\\someone\\x\\lv_20261030_0909\\labels.coco.json"},
                        "b": {"kind": "model", "weights": "models\\v5_evnonev_y8m_20261029\\best.pt"}}})
        self.cfg = SimpleNamespace(review_root=self.root, predict_root=None)

    def rows(self, who, only_mine=True, regex=LAB_RE):
        with mock.patch.object(APP, "PRED_ID_RE", regex):
            return {r["pkg"]: r for r in REV.list_packages(self.cfg, who, Ctx.can_see, only_mine)}

    def test_card_progress_counts_cards_not_images(self):
        # fixture, conf 0.35 (no cand_conf): B p1 (.52, u), g202, g203; A g104, g105 -> 5 cards
        self.assertEqual(REV.card_progress(self.pred), (0, 5))
        put(self.pred, "fixB_old.png", {"p1": {"v": "no", "t": 1}, "g202": {"v": None, "t": 2},       # undone: open
                                        "g999": {"v": "del", "t": 1}})                                # not a card
        put(self.pred, "fixA_new.png", {"g104": "keep:EV", "g105": {"v": "keep:EV", "t": 1, "seed": "september:x"}})
        self.assertEqual(REV.card_progress(self.pred), (2, 5))  # p1 and g104; a September seed is not decided
        put(self.pred, "fixB_old.png", {"i:reject": {"v": "reject:empty", "t": 3}})  # a rejected image: all its cards
        self.assertEqual(REV.card_progress(self.pred), (4, 5))

    def test_cand_conf_threshold(self):
        data = make_fixture.build(self.tmp / "x")
        self.assertEqual(sum(map(len, REV.package_cards(dict(data, cand_conf=0.6)).values())), 4)   # p1 (.52) drops
        self.assertEqual(sum(map(len, REV.package_cards(dict(data, cand_conf=0.25)).values())), 5)

    def test_unreadable_data_js(self):
        (self.pred / "data.js").write_text("window.REVIEW_DATA = {};", encoding="utf-8")
        self.assertEqual(REV.card_progress(self.pred), (None, None))

    def test_access_owner_reviewer_admin(self):
        self.assertEqual(set(self.rows(user("bas"))), {PRED, LAB})          # owner of one, reviewer of the other
        self.assertEqual(self.rows(user("bas"))[LAB]["role"], "reviewer: you")
        self.assertEqual(self.rows(user("bas"))[PRED]["role"], "yours")
        self.assertEqual(set(self.rows(user("other"))), set())                # neither: no trace
        self.assertEqual(set(self.rows(user("other"), only_mine=False)), set())
        self.assertEqual(set(self.rows(user("pujan"))), {LAB})
        self.assertEqual(set(self.rows(user("adm", "admin"))), set())         # admin Mine: only their own
        self.assertEqual(set(self.rows(user("adm", "admin"), only_mine=False)), {PRED, LAB})
        self.assertFalse(REV.may_open(None, {"owner": "bas"}, Ctx.can_see))
        self.assertFalse(REV.may_open(user("bas"), {"owner": "x", "reviewers": "bas"}, Ctx.can_see))  # not a list

    def test_lab_listed_only_once_the_route_takes_it(self):
        self.assertEqual(set(self.rows(user("bas"), regex=PRED_ONLY_RE)), {PRED})  # main's portal route: pred_ only

    def test_excluded_package_never_listed(self):
        os.environ[MRS.EXCLUDE_ENV] = LAB
        self.assertEqual(set(self.rows(user("bas"))), {PRED})

    def test_made_from_has_no_paths(self):
        r = self.rows(user("bas"))
        self.assertEqual(r[LAB]["made_from"], "lv_20261030_0909 vs v5_evnonev_y8m_20261029 · 4 images")
        self.assertEqual(r[PRED]["made_from"], "v5_model predictions · 4 images")
        self.assertEqual(REV._folder_name("D:\\a\\roboflow - x\\test\\_annotations.coco.json"), "roboflow - x")

    def test_render(self):
        put(self.lab, "fixA_new.png", {"g104": {"v": "keep:EV", "t": 1}})
        html = REV.render(sorted(self.rows(user("bas")).values(), key=lambda r: r["pkg"]), False, False)
        self.assertIn("<b>1 of 5 cards</b> decided", html)
        self.assertIn("checked 1 of 4 images", html)
        self.assertIn(f'href="/review/{LAB}/"', html)
        self.assertIn("reviewer: you", html)
        self.assertNotIn("Users", html)
        self.assertNotIn("<th>Owner</th>", html)
        self.assertIn("<th>Owner</th>", REV.render(list(self.rows(user("a", "admin"), False).values()), True, True))


if __name__ == "__main__":
    unittest.main()
