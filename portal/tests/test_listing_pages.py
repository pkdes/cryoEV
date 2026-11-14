"""
Round 13 B2 (#22 + links) through a real portal: Results and Predict "My analyses" share portal/listing.py's rows
(filters, sort, Group by from the URL), each page keeps its own can_see behaviour (a basic user sees exactly the sets
they saw before, and nothing of anyone else's: no other owner, model or private parent in the HTML), every link into
a review package goes to /reviews/go/<pkg>, the Jobs board hides who / via on compare and audit packages, and the
Data page reads Reviewers from the package's own review_package.json.

Temp roster / registry / laptop root only (CRYOEV_LAPTOP_ROOT points at the temp folder; never the real ones).

  python -m unittest portal.tests.test_listing_pages -v
"""
import csv
import json
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cjtest as T  # noqa: E402
from portal import auth as A  # noqa: E402
from portal.modules import data as D  # noqa: E402
from portal.modules import jobs as J  # noqa: E402
from portal import titles as TT  # noqa: E402

XSS = '<img src=x onerror=alert(1)>'
S1 = "ps_20261110_a00001"        # bas, session 20261106-2, 5 samples, has a review package
S2 = "ps_20261111_a00002"        # bas, upload, derived from S1, made by claude for bas
S3 = "ps_20261112_a00003"        # bas, derived from OTHER (someone else's private set)
OTHER = "ps_20261109_b00001"     # other, model v4
ADMIN_SET = "ps_20261108_c00001"  # tester (admin), session 20261012
LAB_REV, LAB_AUDIT = "lab_revise_x", "lab_compare_audit_x"
ADMIN_NAME = "Session 20261012 · 8 Nov 2026, 08:00"


def wjson(p: Path, doc):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc), encoding="utf-8")


def make_set(root, ps, owner, created, *, name=None, session=None, via=None, model="v5", parent=None,
             samples=("S1",)):
    req = {"ps": ps, "owner": owner, "model": model, "created": created,
           "source": {"kind": "session", "session": session} if session else {"kind": "upload"}}
    if name:
        req["name"] = name
    if via:
        req["owner_via"] = via
    if parent:
        req["provenance"] = {"derived_from": [{"kind": "prediction_set", "id": parent}]}
    d = root / ps
    wjson(d / "request.json", req)
    wjson(d / "status.json", {"state": "done", "n_done": len(samples) * 2, "n_total": len(samples) * 2})
    with open(d / "images.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "file", "sample_label", "nm_per_px"])
        for k in range(len(samples) * 2):
            w.writerow([f"{k + 1:04d}", f"images/{k + 1:04d}.png", samples[k % len(samples)], "1.47"])


def row_ids(page: str) -> list:
    return re.findall(r'<li class="cj-list-row[^"]*" data-id="([^"]+)">', page)


def groups(page: str) -> dict:
    """{group heading: [row ids]} from the <details> groups."""
    out = {}
    for name, body in re.findall(r'<details class="cj-list-group" open id="cj-list-g\d+"><summary>'
                                 r'<span class="cj-list-gname cj-wrap">(.*?)</span>.*?</summary>(.*?)</details>',
                                 page, re.S):
        out[name] = row_ids(body)
    return out


class Pages(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = T.Env()
        cls.tmp = Path(tempfile.mkdtemp(prefix="cj_r13b2_"))
        lap = cls.tmp / "laptop"
        cls.pred, cls.review = lap / "predict", lap / "review"
        cls.pred.mkdir(parents=True)
        cls.review.mkdir(parents=True)
        make_set(cls.pred, S1, "bas", "2026-11-10T09:47:31", name="Bas session set", session="20261106-2",
                 samples=("EV-013-2", "14K2-A", "14K2-B", "HEK", "GFP"))
        make_set(cls.pred, S2, "bas", "2026-11-11T11:43:59", name="Bas 2x test " + XSS, via="claude", parent=S1,
                 samples=("HEK",))
        make_set(cls.pred, S3, "bas", "2026-11-12T16:52:37", name="Bas from other", parent=OTHER,
                 samples=("SKOV-3",))
        make_set(cls.pred, OTHER, "other", "2026-11-09T12:30:01", name="Other secret root", model="v4_secret",
                 samples=("Secret sample",))
        make_set(cls.pred, ADMIN_SET, "tester", "2026-11-08T08:00:00", session="20261012", samples=("26119 - LHD",))
        wjson(cls.review / f"pred_{S1}" / "review_package.json",
              {"id": f"pred_{S1}", "kind": "prediction", "owner": "bas", "prediction_set": S1,
               "reviewers": ["alyssia"]})
        wjson(cls.review / LAB_REV / "review_package.json",
              {"id": LAB_REV, "kind": "compare", "purpose": "revise", "reviewers": ["alyssia", "au1"]})
        wjson(cls.review / LAB_AUDIT / "review_package.json",
              {"id": LAB_AUDIT, "kind": "compare", "reviewers": ["au2"]})
        T.TK.write_registry(cls.env.registry, {"review_package": [
            {"schema": "review_package/1", "id": f"pred_{S1}", "kind": "prediction", "prediction_set": S1,
             "owner": "bas", "status": "built", "reviewers": ["stale"], "logged": "2026-11-10T10:00:00"},
            {"schema": "review_package/1", "id": LAB_REV, "kind": "compare", "status": "built",
             "reviewers": ["stale"], "logged": "2026-11-10T10:00:00"},
            {"schema": "review_package/1", "id": LAB_AUDIT, "kind": "compare", "status": "built",
             "reviewers": ["stale"], "logged": "2026-11-10T10:00:00"}]}, samples=False)
        cls.patch = T.TK.env_patch(strip_all=True, CRYOEV_LAPTOP_ROOT=str(lap))
        cls.patch.start()
        D._files_cache.clear()
        D._owner_cache.clear()
        D._registry_cache.clear()
        keys = {"admin": cls.env.key, "bas": A.add_key(cls.env.roster, "bas", "basic"),
                "other": A.add_key(cls.env.roster, "other", "basic"),
                "nobody": A.add_key(cls.env.roster, "nobody", "basic")}
        cfg = cls.env.config()
        cfg.predict_root, cfg.review_root = cls.pred, cls.review
        cls.p = T.Portal(cfg)
        cls.tok = {k: T.login(cls.p.port, v) for k, v in keys.items()}

    @classmethod
    def tearDownClass(cls):
        cls.p.stop()
        cls.patch.stop()
        cls.env.cleanup()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def page(self, path, who="admin") -> str:
        st, _h, body = T.request(self.p.port, "GET", path, T.cookie(self.tok[who]))
        self.assertEqual(st, 200, path)
        return body.decode("utf-8")

    # ---- Results
    def test_results_admin_default_and_groups(self):
        page = self.page("/results/")
        self.assertEqual(row_ids(page), [S3, S2, S1, OTHER, ADMIN_SET])          # newest first, every set
        self.assertIn('id="cj-list-person"', page)
        self.assertNotIn("<details", page)                                       # group none by default
        self.assertIn('id="cj-list-agents" name="agents" value="1" checked', page)
        self.assertNotIn(XSS, page)
        g = groups(self.page("/results/?group=source"))
        self.assertEqual(sorted(g), ["Source: Bas session set", "Source: Other secret root", f"Source: {ADMIN_NAME}"])
        self.assertEqual(g["Source: Bas session set"], [S2, S1])
        self.assertEqual(g["Source: Other secret root"], [S3, OTHER])            # the admin may see the parent
        self.assertEqual(groups(self.page("/results/?group=session")),
                         {"upload · 2026-11-12": [S3], "upload · 2026-11-11": [S2], "20261106-2": [S1],
                          "upload · 2026-11-09": [OTHER], "20261012": [ADMIN_SET]})
        self.assertEqual(groups(self.page("/results/?group=person"))["bas"], [S3, S2, S1])   # agent-made under bas
        g = groups(self.page("/results/?group=sample"))
        self.assertEqual(g["Mixed (more than 3 samples)"], [S1])
        self.assertEqual(g["HEK"], [S2])

    def test_results_filters_and_sorts(self):
        self.assertEqual(row_ids(self.page("/results/?agents=0")), [S3, S1, OTHER, ADMIN_SET])
        self.assertEqual(row_ids(self.page("/results/?agents=0&agents=1")), [S3, S2, S1, OTHER, ADMIN_SET])
        self.assertEqual(row_ids(self.page("/results/?person=bas&agents=0")), [S3, S1])
        self.assertEqual(row_ids(self.page("/results/?from=2026-11-10&to=2026-11-11")), [S2, S1])
        self.assertEqual(row_ids(self.page("/results/?model=v4_secret")), [OTHER])
        self.assertEqual(row_ids(self.page("/results/?q=secret")), [OTHER])
        self.assertEqual(row_ids(self.page("/results/?sort=size")), [S1, S3, S2, OTHER, ADMIN_SET])   # 10, then 2s
        self.assertEqual(row_ids(self.page("/results/?sort=name")), [S2, S3, S1, OTHER, ADMIN_SET])
        self.assertEqual(row_ids(self.page("/results/?status=ready")), [])        # nothing computed in this fixture
        none = self.page("/results/?status=ready")
        self.assertIn("No analyses match these filters.", none)
        self.assertEqual(row_ids(self.page("/results/?status=none")), [S3, S2, S1, OTHER, ADMIN_SET])
        junk = self.page("/results/?group=%3Cx%3E&sort=evil&person=%3Cb%3E&from=nope")      # strict fallbacks
        self.assertEqual(row_ids(junk), [S3, S2, S1, OTHER, ADMIN_SET])
        self.assertNotIn("<x>", junk)
        self.assertNotIn("<b>", junk.split("<main", 1)[1].replace("<b class=", ""))

    def test_results_basic_user_sees_only_their_own(self):
        page = self.page("/results/", "bas")
        self.assertEqual(row_ids(page), [S3, S2, S1])                             # exactly as before: can_see
        self.assertNotIn('id="cj-list-person"', page)                            # one person: no person filter
        for leak in (OTHER, "b00001", "Other secret", "v4_secret", "Secret sample", ADMIN_SET, "tester", ">other<"):
            self.assertNotIn(leak, page)
        g = groups(self.page("/results/?group=source", "bas"))
        self.assertEqual(g, {"Source: Bas from other": [S3], "Source: Bas session set": [S2, S1]})
        src = self.page("/results/?group=source", "bas")
        for leak in (OTHER, "Other secret"):
            self.assertNotIn(leak, src)
        self.assertEqual(row_ids(self.page("/results/?person=other", "bas")), [])   # a filter never widens
        self.assertNotIn(OTHER, self.page("/results/?person=other&group=person", "bas"))
        self.assertIn("claude for bas", page)
        empty = self.page("/results/", "nobody")                                 # owns nothing: unchanged page
        self.assertIn("No analyses yet", empty)
        self.assertNotIn("cj-list-form", empty)
        self.assertNotIn("ps_2026", empty)

    def test_results_other_users_view_of_the_shared_chain(self):
        page = self.page("/results/?group=source", "other")
        self.assertEqual(row_ids(page), [OTHER])
        self.assertNotIn(S3, page)

    # ---- Predict
    def test_predict_my_analyses_same_rows(self):
        page = self.page("/predict/", "bas")
        self.assertIn("My analyses", page)
        self.assertEqual(row_ids(page), [S3, S2, S1])
        self.assertIn(f'href="/reviews/go/pred_{S1}"', page)                     # the Review link goes to go
        self.assertNotIn(f'href="/review/pred_{S1}/"', page)
        self.assertIn('action="/predict/"', page)
        for leak in (OTHER, "Other secret", "v4_secret"):
            self.assertNotIn(leak, page)
        self.assertEqual(groups(self.page("/predict/?group=source", "bas")),
                         {"Source: Bas from other": [S3], "Source: Bas session set": [S2, S1]})
        self.assertEqual(row_ids(self.page("/predict/?sort=name", "bas")), [S2, S3, S1])
        adm = self.page("/predict/")
        self.assertIn("All analyses", adm)
        self.assertEqual(row_ids(adm), [S3, S2, S1, OTHER, ADMIN_SET])
        self.assertEqual(row_ids(self.page("/predict/?status=failed")), [])
        self.assertEqual(row_ids(self.page("/predict/?status=done&person=other")), [OTHER])

    def test_predict_set_page_and_api_link_to_go(self):
        page = self.page(f"/predict/{S1}/", "bas")
        self.assertIn(f'href="/reviews/go/pred_{S1}"', page)
        api = json.loads(self.page("/predict/api/mine", "bas"))
        hrefs = {i["ps"]: i["review_href"] for i in api["items"]}
        self.assertEqual(hrefs, {S1: f"/reviews/go/pred_{S1}", S2: None, S3: None})

    # ---- Data
    def test_data_reviewers_from_disk_and_go_links(self):
        t = self.page(f"/data/images/pred_{S1}")
        self.assertRegex(t, r"Reviewers.*?alyssia")
        self.assertNotIn("stale", t)
        self.assertIn(f'href="/reviews/go/pred_{S1}"', t)
        rev = self.page(f"/data/images/{LAB_REV}")
        self.assertIn("alyssia, au1", rev)
        self.assertNotIn("stale", rev)
        self.assertIn(f'href="/reviews/go/{LAB_REV}"', rev)                     # lab_ packages open through go
        audit = self.page(f"/data/images/{LAB_AUDIT}")
        self.assertNotIn("au2", audit)                                           # an audit lists no reviewers
        self.assertNotIn("stale", audit)
        self.assertIn(f'href="/reviews/go/{LAB_AUDIT}"', audit)                  # go decides who may open it


class JobsBlind(unittest.TestCase):
    def test_audit_purpose_and_compare_hide_who(self):
        self.assertTrue(J.is_blind({"kind": "compare", "meta": {}}))
        self.assertTrue(J.is_blind({"kind": "review", "meta": {"purpose": "audit"}}))
        self.assertTrue(J.is_blind({"kind": "review", "meta": {"kind": "compare", "purpose": "revise"}}))
        self.assertFalse(J.is_blind({"kind": "review", "meta": {}}))
        self.assertFalse(J.is_blind({"kind": "review"}))

    def test_review_packages_rows(self):
        rows = [{"id": "lab_a", "title": "A", "kind": "review", "cards": 2, "decided": 1, "images": 1,
                 "reviewed": 0, "last": None, "who": "bob", "via": "claude", "status": "in_progress",
                 "applied_by": None, "meta": {"purpose": "audit"}},
                {"id": "lab_b", "title": "B", "kind": "review", "cards": 2, "decided": 1, "images": 1,
                 "reviewed": 0, "last": None, "who": "carol", "via": None, "status": "in_progress",
                 "applied_by": None, "meta": {}}]
        orig = TT.scan_packages
        TT.scan_packages = lambda root, records: [dict(r) for r in rows]
        try:
            got = {r["id"]: r for r in J.review_packages(SimpleNamespace(review_root=Path("x"), registry=None))}
        finally:
            TT.scan_packages = orig
        self.assertEqual((got["lab_a"]["who"], got["lab_a"]["via"]), (None, None))
        self.assertEqual(got["lab_b"]["who"], "carol")                           # control


if __name__ == "__main__":
    unittest.main()
