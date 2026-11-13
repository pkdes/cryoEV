"""
/review/<lab_ id>/ through the portal (round 9, R2; portal/app.py pred_mount + portal/modules/reviews.serving): a lab /
compare package opens for its owner and its listed `reviewers`, and is proxied to the multi server with that
package's key and the reviewer's id; everything else is the one 404. Served only as the multi server serves it:
lab_ needs $CJ_REVIEW_PREFIXES, and the single-package review server's package (services config), $CJ_REVIEW_EXCLUDE
and the portal's own real-review package (cfg.review_package) are never opened. The picker lists the same set.

Temp roster, temp review root, a dummy multi server upstream, a temp services config (never the real one).

  python -m unittest portal.tests.test_review_lab_route -v
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cjtest as T  # noqa: E402
from test_shell_pred import MultiDummy, hdrs  # noqa: E402
from portal import auth as A  # noqa: E402
from annotation import multi_review_server as MRS  # noqa: E402
sys.path.insert(0, str(T.REPO / "annotation" / "tests"))
import make_fixture  # noqa: E402

LAB = "lab_compare_route_20261112"
SVC = "lab_compare_single_20261112"     # the single-package review server's package (services config)
ENVX = "lab_compare_envx_20261112"      # $CJ_REVIEW_EXCLUDE
OWN = "lab_compare_own_20261112"        # the portal's own real-review package (cfg.review_package)


def build(root: Path, name: str, reviewers=("u-basic",)):
    pkg = root / name
    make_fixture.build(pkg)
    (pkg / "review_package.json").write_text(json.dumps({"kind": "compare", "owner": "u-adv",
                                                         "reviewers": list(reviewers), "n_images": 4}))
    (pkg / ".review_key").write_text(f"key-{name}", encoding="utf-8")
    return pkg


class LabRoute(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._env = {k: os.environ.pop(k) for k in list(os.environ) if k.startswith(("CRYOEV_", "CJ_"))}
        cls.up = MultiDummy()
        cls.e = T.Env()
        cls.root = cls.e.tmp / "review"
        cls.root.mkdir()
        for n in (LAB, SVC, ENVX, OWN):
            build(cls.root, n)
        cls.svc = cls.e.tmp / "services.json"
        cls.svc.write_text(json.dumps({"services": [{"name": "review", "argv": [
            "{python}", "annotation/review_server.py", "--package", str(cls.root / SVC), "--port", "8766"]}]}))
        os.environ.update({"CRYOEV_SERVICES_CONFIG": str(cls.svc), MRS.PREFIX_ENV: "pred_,lab_", MRS.EXCLUDE_ENV: ENVX})
        cls.keys = {"basic": A.add_key(cls.e.roster, "u-basic", "basic"), "other": A.add_key(cls.e.roster, "u-other", "basic"),
                    "adv": A.add_key(cls.e.roster, "u-adv", "advanced")}
        cls.p = T.Portal(cls.e.config(pred_upstream=cls.up.url, review_root=cls.root, review_package=cls.root / OWN))
        cls.tok = {w: T.login(cls.p.port, k) for w, k in cls.keys.items()}

    @classmethod
    def tearDownClass(cls):
        cls.p.stop()
        cls.up.stop()
        cls.e.cleanup()
        for k in ("CRYOEV_SERVICES_CONFIG", MRS.PREFIX_ENV, MRS.EXCLUDE_ENV):
            os.environ.pop(k, None)
        os.environ.update(cls._env)

    def get(self, who, path):
        st, _, body = T.request(self.p.port, "GET", path, T.cookie(self.tok[who]))
        return st, body

    def test_listed_reviewer_opens_and_is_the_reviewer_upstream(self):
        self.assertEqual(self.get("basic", f"/review/{LAB}/")[0], 200)      # the shell
        n = len(self.up.seen)
        st, _ = self.get("basic", f"/review/{LAB}/app/api/ping")
        self.assertEqual(st, 200)
        rec = self.up.seen[n]
        self.assertTrue(rec["path"].startswith(f"/{LAB}/"), rec["path"])
        self.assertEqual(hdrs(rec)["x-cj-user"], "u-basic")
        self.assertEqual(hdrs(rec)["x-review-key"], f"key-{LAB}")
        self.assertEqual(self.get("adv", f"/review/{LAB}/app/api/ping")[0], 200)     # the owner

    def test_unlisted_basic_user_404(self):
        n = len(self.up.seen)
        for path in (f"/review/{LAB}/", f"/review/{LAB}/app/api/ping"):
            self.assertEqual(self.get("other", path)[0], 404, path)
        self.assertEqual(len(self.up.seen), n)                               # nothing reached the multi server

    def test_excluded_ids_404(self):
        n = len(self.up.seen)
        for pid in (SVC, ENVX, OWN):                                         # services config, env, own package
            for path in (f"/review/{pid}/", f"/review/{pid}/app/api/ping"):
                self.assertEqual(self.get("basic", path)[0], 404, path)
                self.assertEqual(self.get("adv", path)[0], 404, path)
        self.assertEqual(len(self.up.seen), n)

    def test_malformed_and_wrong_case_ids_404(self):
        n = len(self.up.seen)
        for pid in (f"{LAB}.", "lab_a..b", LAB.upper(), "LAB" + LAB[3:], LAB + "x"):
            for path in (f"/review/{pid}/", f"/review/{pid}/app/api/ping"):
                self.assertEqual(self.get("basic", path)[0], 404, path)
        self.assertEqual(len(self.up.seen), n)

    def test_lab_needs_the_prefix(self):
        os.environ.pop(MRS.PREFIX_ENV)
        try:
            self.assertEqual(self.get("basic", f"/review/{LAB}/")[0], 404)  # default: pred_ only
            self.assertNotIn(LAB.encode(), self.get("basic", "/reviews/")[1])
        finally:
            os.environ[MRS.PREFIX_ENV] = "pred_,lab_"

    def test_unreadable_services_config_closes_lab(self):
        good = self.svc.read_text()
        self.svc.write_text("{broken")
        try:
            self.assertEqual(self.get("basic", f"/review/{LAB}/")[0], 404)
            self.assertNotIn(LAB.encode(), self.get("basic", "/reviews/")[1])
        finally:
            self.svc.write_text(good)
            os.utime(self.svc, None)
        self.assertEqual(self.get("basic", f"/review/{LAB}/")[0], 200)

    def test_picker_lists_the_same_set(self):
        st, body = self.get("basic", "/reviews/")
        self.assertEqual(st, 200)
        self.assertIn(f'href="/review/{LAB}/"'.encode(), body)
        for pid in (SVC, ENVX, OWN):
            self.assertNotIn(pid.encode(), body)
        self.assertNotIn(LAB.encode(), self.get("other", "/reviews/")[1])


if __name__ == "__main__":
    unittest.main()
