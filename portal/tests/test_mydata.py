"""
My data (portal/modules/mydata.py, round 17r): a basic account's Data page. It shows only the image sets Predict
already offers that person (a set with a problem never shows; an unknown key is the same 404), the person's own
analyses (not someone else's), thumbnails by index only (out of range = 404), names escaped; the nav gives basic
"Data" -> /mydata/ and admin / advanced the lab's /data/.

Reuses test_predict_module.Base (temp roster, sources, predict root; never the real ones).

    python -m unittest discover -s portal/tests -p test_mydata.py -v
"""
import html
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cjtest as T  # noqa: E402
from test_predict_module import XSS, Base  # noqa: E402


class MyData(Base):
    def test_roles_and_nav(self):
        for who in ("basic", "advanced", "admin"):
            self.page("/mydata/", who)
        man = json.loads(self.manifest.read_text(encoding="utf-8"))
        md = next(m for m in man["modules"] if m["id"] == "mydata")
        self.assertEqual((md["roles"], md["name"], md["route"], md["nav"]), (["basic"], "Data", "/mydata/", True))
        home_b, home_a = self.page("/", "basic"), self.page("/", "admin")
        self.assertIn('href="/mydata/"', home_b)
        self.assertNotIn('href="/data/"', home_b)
        self.assertIn('href="/data/"', home_a)
        self.assertNotIn('href="/mydata/"', home_a)
        st, _h, _b = self.get("/data/", "basic")                       # the lab catalog stays closed to basic
        self.assertEqual(st, 403)

    def test_index_lists_only_usable_sets(self):
        body = self.page("/mydata/", "basic")
        self.assertIn("EV prep A", body)
        self.assertNotIn(XSS, body)                                    # escaped
        self.assertIn(html.escape(XSS), body)
        self.assertIn("Plain tiles", body)
        self.assertNotIn("Gone", body)                                 # a source with a problem never shows
        self.assertIn(f'href="/predict/new/session/{self.key_a}/"', body)
        self.assertIn(f'src="/mydata/thumb/{self.key_a}/0.jpg"', body)
        self.assertIn("2 samples", body)

    def test_set_page_and_thumbs(self):
        body = self.page(f"/mydata/set/{self.key_a}/", "basic")
        self.assertIn("1.4724 nm/px", body)
        self.assertIn("S1", body)
        self.assertIn("S2", body)
        st, h, img = self.get(f"/mydata/thumb/{self.key_a}/0.jpg", "basic")
        self.assertEqual((st, T.hget(h, "content-type")), (200, "image/jpeg"))
        self.assertTrue(img.startswith(b"\xff\xd8"))
        for path in (f"/mydata/thumb/{self.key_a}/99.jpg", f"/mydata/thumb/{self.key_a}/-1.jpg",
                     "/mydata/thumb/0000000000/0.jpg", "/mydata/set/0000000000/", "/mydata/set/..%2F..%2F/"):
            st, _h, _b = self.get(path, "basic")
            self.assertEqual(st, 404, path)

    def test_own_analyses_only(self):
        mine = self.fake_set("bas", "done", n_done=3)
        theirs = self.fake_set("bas2", "done", n_done=3)
        body = self.page("/mydata/", "basic")
        self.assertIn(f'href="/predict/{mine}/"', body)
        self.assertNotIn(theirs, body)
        self.assertIn(f'href="/predict/{theirs}/"', self.page("/mydata/", "other"))


if __name__ == "__main__":
    unittest.main()
