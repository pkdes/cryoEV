"""
Apply preview module (round 10, V4; portal/modules/applyreview.py), fixtures only: a temp CryoAI-laptop with a label
set (labels/lv_*/labels.coco.json), review packages built on it (and on an export), a compare package, a `_work_`
folder, a pred_ package, a temp services config and a temp registry. Never the real data or ~/.cryoev.

  - eligibility: label-set base yes (the control); export base no; `_work_` present / compare: not even listed;
    named by a review_server.py --package in the services config no; pred_ no; nothing reviewed no; an unreadable
    services config makes nothing eligible (fail closed);
  - a preview writes only under <state>/apply_preview (everything else hashed before / after, the registry included);
  - non-admins get 403 on every route; an unmapped who (portal id not in people.json) is refused and writes nothing;
  - the report is served (CSP, pictures) and path traversal is refused; old previews are pruned.
"""
import hashlib
import json
import os
import re
import shutil
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from urllib.parse import urlencode

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cjtest as T  # noqa: E402
from portal import auth as A  # noqa: E402
from portal.modules import applyreview as M  # noqa: E402

sys.path.insert(0, str(T.REPO / "annotation" / "tests"))
import test_apply_review_edits as tar  # noqa: E402  (module import: its TestCases are not collected twice)

LV = "lv_20261101_1200"


def tree_hash(root: Path, skip: Path) -> dict:
    out = {}
    for p in sorted(Path(root).rglob("*")):
        if p.is_file() and skip not in p.parents:
            out[str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def make_package(review: Path, name: str, source: Path, reviewed=True, images=False, meta=None):
    """A copy of the synthetic package as <review>/<name>, its data.js naming `source`."""
    tmp = review.parent / "_build" / name
    _base, pkg, _edits = tar.build_synthetic(tmp)
    d = json.loads((pkg / "data.js").read_text(encoding="utf-8").split("=", 1)[1].strip().rstrip(";"))
    d["id"], d["source"] = name, str(source)
    (pkg / "data.js").write_text("window.REVIEW_DATA = " + json.dumps(d) + ";\n", encoding="utf-8")
    for f in (pkg / "edits").glob("*.json"):
        rec = json.loads(f.read_text(encoding="utf-8"))
        if "pkg" in rec:
            rec["pkg"] = name
        if not reviewed:
            rec["edit"]["reviewed"] = False
        f.write_text(json.dumps(rec), encoding="utf-8")
    if images:
        from PIL import Image
        (pkg / "images").mkdir()
        for n in ("a_png", "b_png", "c_png"):
            Image.new("RGB", (tar.W, tar.H), (90, 90, 90)).save(pkg / "images" / f"{n}.jpg", quality=70)
    if meta:
        (pkg / "review_package.json").write_text(json.dumps(meta), encoding="utf-8")
    shutil.move(str(pkg), str(review / name))
    return review / name


class World:
    def __init__(self):
        self.env = T.Env()
        self.tmp = self.env.tmp
        self.lap = self.tmp / "laptop"
        self.review = self.lap / "review"
        self.review.mkdir(parents=True)
        base, _p, _e = tar.build_synthetic(self.tmp / "_seed")
        self.base = self.lap / "labels" / LV / "labels.coco.json"
        self.base.parent.mkdir(parents=True)
        shutil.copy(base, self.base)
        export = self.lap / "roboflow_export_20261101" / "current.coco.json"
        export.parent.mkdir(parents=True)
        shutil.copy(base, export)
        make_package(self.review, "lab_ok", self.base, images=True)                 # the control: eligible
        make_package(self.review, "lab_export", export)                              # export base
        make_package(self.review, "lab_served", self.base)                           # served live
        make_package(self.review, "lab_unrev", self.base, reviewed=False)            # nothing reviewed
        make_package(self.review, "lab_work", self.base)                             # has a _work_ folder
        (self.review / "_work_lab_work").mkdir()
        (self.review / "_work_lab_work" / "decoys.json").write_text("SECRET-ANSWER-KEY", encoding="utf-8")
        make_package(self.review, "cmp_audit", self.base, meta={"kind": "compare", "base": str(self.base)})
        make_package(self.review, "pred_x", self.base)
        shutil.rmtree(self.review.parent / "_build")
        shutil.rmtree(self.tmp / "_seed")
        self.services = self.tmp / "services.json"
        self.services.write_text(json.dumps({"services": [
            {"name": "review", "argv": ["{python}", "annotation/review_server.py", "--package",
                                        str(self.review / "lab_served"), "--port", "8766"]},
            {"name": "portal", "argv": ["{python}", "-m", "portal.app", "--package", "lab_ok_not_a_review_server"]},
        ]}), encoding="utf-8")
        self.keys = {"tester": self.env.key, "pujan": A.add_key(self.env.roster, "pujan", "admin"),
                     "adv": A.add_key(self.env.roster, "adv", "advanced")}

    def cfg(self):
        return self.env.config(review_root=self.review)

    def cleanup(self):
        self.env.cleanup()


class Eligibility(unittest.TestCase):
    def setUp(self):
        self.w = World()
        self.addCleanup(self.w.cleanup)
        p = mock.patch.dict(os.environ, {"CRYOEV_SERVICES_CONFIG": str(self.w.services)})
        p.start()
        self.addCleanup(p.stop)

    def rows(self):
        served, problem = M.served_packages()
        return {r["id"]: r for r in M.packages(self.w.cfg(), {}, served, problem)}

    def test_rules(self):
        self.assertEqual(M.served_packages(), ({"lab_served"}, None))
        rows = self.rows()
        self.assertTrue(rows["lab_ok"]["eligible"], rows["lab_ok"]["why"])           # control
        self.assertEqual(rows["lab_ok"]["base"], self.w.base)
        self.assertNotIn("lab_export", rows)                                          # export base: not a label set
        self.assertNotIn("lab_work", rows)                                            # _work_: excluded entirely
        self.assertNotIn("cmp_audit", rows)                                           # compare: excluded entirely
        self.assertNotIn("pred_x", rows)                                              # prediction package
        self.assertFalse(rows["lab_served"]["eligible"])
        self.assertIn("reviewed live", rows["lab_served"]["why"][0])
        self.assertFalse(rows["lab_unrev"]["eligible"])
        self.assertIn("reviewed", rows["lab_unrev"]["why"][0])
        self.assertEqual(sorted(rows), ["lab_ok", "lab_served", "lab_unrev"])

    def test_services_config(self):
        self.w.services.write_text("{not json", encoding="utf-8")                    # unreadable: fail closed
        rows = self.rows()
        self.assertFalse(any(r["eligible"] for r in rows.values()))
        self.w.services.unlink()                                                      # no config: nothing served
        self.assertEqual(M.served_packages(), (set(), None))
        self.assertTrue(self.rows()["lab_served"]["eligible"])
        for argv, want in ((["py", "annotation\\review_server.py", "--package=C:\\x\\review\\pk1\\"], {"pk1"}),
                           (["py", "annotation/multi_review_server.py", "--package", "/r/pk2"], set()),
                           ("py annotation/review_server.py --package /r/pk3", {"pk3"})):
            self.w.services.write_text(json.dumps({"services": [{"name": "s", "argv": argv}]}), encoding="utf-8")
            self.assertEqual(M.served_packages()[0], want, argv)

    def test_people_and_command(self):
        sys.path.insert(0, str(T.REPO / "annotation"))
        import apply_review_edits as ar
        people = M.load_people()
        lookup = ar.roster_lookup(people)
        for name in ("pujan", "pujan-web", "Pujan", " AU1 ", "au1", "alyssia", "tester", ""):
            want = lookup(name)
            self.assertEqual(M.people_id(name), None if want == "unknown" else want, name)
        self.assertEqual(M.people_id("pujan-web"), "pujan")
        self.assertIsNone(M.people_id("alyssia"))
        cmd = M.cli_command(self.rows()["lab_ok"], "pujan")
        self.assertIn(f'--base "{self.w.base}"', cmd)
        self.assertIn(f'--out-root "{self.w.lap / "labels"}"', cmd)                   # not the CLI default labels/labels
        self.assertTrue(cmd.endswith("--who pujan"))

    def test_prune(self):
        root = M.scratch_root(self.w.cfg())
        now = datetime(2026, 11, 12, tzinfo=timezone.utc)
        old = root / "lab_ok" / (now - timedelta(days=8)).strftime("%Y%m%dT%H%M%SZ")
        new = root / "lab_ok" / (now - timedelta(days=6)).strftime("%Y%m%dT%H%M%SZ")
        other = root / "lab_ok" / "not_a_preview"
        for d in (old, new, other):
            d.mkdir(parents=True)
        M.prune(root, now)
        self.assertEqual((old.exists(), new.exists(), other.exists()), (False, True, True))


class Portal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.w = World()
        cls.envp = mock.patch.dict(os.environ, {"CRYOEV_SERVICES_CONFIG": str(cls.w.services)})
        cls.envp.start()
        cls.p = T.Portal(cls.w.cfg())
        cls.tok = {u: T.login(cls.p.port, k) for u, k in cls.w.keys.items()}

    @classmethod
    def tearDownClass(cls):
        cls.p.stop()
        cls.envp.stop()
        cls.w.cleanup()

    def get(self, user, path):
        return T.request(self.p.port, "GET", path, T.cookie(self.tok[user]) if user else {})

    def post(self, user, path, fields):
        h = {**T.cookie(self.tok[user]), "Origin": self.p.origin, "Content-Type": "application/x-www-form-urlencoded"}
        return T.request(self.p.port, "POST", path, h, urlencode(fields).encode())

    def form_token(self, user):
        st, _, body = self.get(user, "/apply/")
        self.assertEqual(st, 200)
        m = re.search(r'name="t" value="([0-9a-f]+)"', body.decode())
        return m.group(1) if m else ""

    def preview(self, user="pujan", pkg="lab_ok"):
        return self.post(user, "/apply/preview", {"t": self.form_token(user), "pkg": pkg})

    def test_1_index(self):
        st, _, body = self.get("pujan", "/apply/")
        page = body.decode()
        self.assertEqual(st, 200)
        self.assertIn("Preview apply", page)
        self.assertIn("lab_ok", page)
        self.assertIn("lab_served", page)                                       # listed as not ready, with the reason
        self.assertIn("reviewed live", page)
        for hidden in ("lab_work", "cmp_audit", "pred_x", "lab_export", "SECRET-ANSWER-KEY"):
            self.assertNotIn(hidden, page)
        self.assertIn("python annotation/apply_review_edits.py", page)
        self.assertEqual(page.count('name="pkg"'), 1)                            # a button for the eligible one only

    def test_2_non_admin_403(self):
        for method, path in (("GET", "/apply/"), ("GET", "/apply/run/lab_ok/20261112T000000Z"),
                             ("GET", "/apply/view/lab_ok/20261112T000000Z/report/index.html")):
            st, _, _ = self.get("adv", path)
            self.assertEqual(st, 403, path)
        st, _, _ = self.post("adv", "/apply/preview", {"t": "x", "pkg": "lab_ok"})
        self.assertEqual(st, 403)
        st, _, _ = self.get("pujan", "/apply/")                                 # control: an admin gets in
        self.assertEqual(st, 200)

    def test_3_unmapped_who_refused(self):
        before = tree_hash(self.w.tmp, self.w.env.state)
        st, _, body = self.get("tester", "/apply/")
        self.assertIn("is not in", body.decode())
        self.assertNotIn("Preview apply</button>", body.decode())               # no button for an unmapped id
        tok = re.search(r'name="t" value="([0-9a-f]+)"', body.decode())
        st, _, body = self.post("tester", "/apply/preview", {"t": tok.group(1) if tok else "x", "pkg": "lab_ok"})
        if tok:
            self.assertEqual(st, 403)
            self.assertIn("is not in annotation/people.json", body.decode())
        self.assertFalse((M.scratch_root(self.w.cfg()) / "lab_ok").exists())    # nothing ran (test_5 runs later)
        self.assertEqual(tree_hash(self.w.tmp, self.w.env.state), before)

    def test_4_refusals(self):
        for pkg in ("lab_served", "lab_unrev", "lab_work", "cmp_audit", "pred_x", "lab_export", "../x"):
            st, _, _ = self.preview(pkg=pkg)
            self.assertEqual(st, 404, pkg)
        st, _, _ = self.post("pujan", "/apply/preview", {"t": "wrong", "pkg": "lab_ok"})
        self.assertEqual(st, 403)

    def test_5_preview_writes_only_scratch_and_serves_report(self):
        scratch = M.scratch_root(self.w.cfg())
        state_before = {p for p in self.w.env.state.rglob("*") if p.is_file()}
        before = tree_hash(self.w.tmp, self.w.env.state)
        st, h, _ = self.preview()
        self.assertEqual(st, 303)
        loc = T.hget(h, "location")
        m = re.fullmatch(r"/apply/run/lab_ok/(\d{8}T\d{6}Z(?:-\d+)?)", loc)
        self.assertTrue(m, loc)
        ts = m.group(1)
        self.assertEqual(tree_hash(self.w.tmp, self.w.env.state), before)       # labels, packages, registry untouched
        new = {p for p in self.w.env.state.rglob("*") if p.is_file()} - state_before
        self.assertTrue(new)
        for p in new:
            self.assertTrue(scratch in p.parents or p.parent.name == "audit", p)
        folder = scratch / "lab_ok" / ts
        info = json.loads((folder / "preview.json").read_text(encoding="utf-8"))
        self.assertNotIn("error", info, info.get("error"))
        self.assertEqual((info["who"], info["scope"], info["base_lv"]), ("pujan", "reviewed_only", LV))
        ver = json.loads((folder / "labels" / info["lv"] / "version.json").read_text(encoding="utf-8"))
        self.assertEqual((ver["created_by"], ver["created_via"], ver["review_scope"]), ("pujan", None, "reviewed_only"))
        self.assertEqual(ver["n_held_back"], 1)                                  # b.png is edited but not reviewed
        self.assertTrue(ver["edits_dir"].startswith(str(folder)))               # the frozen copy, not the package
        st, _, body = self.get("pujan", loc)
        self.assertEqual(st, 200)
        page = body.decode()
        self.assertIn("Open the change report", page)
        self.assertIn("--who pujan", page)
        st, h, body = self.get("pujan", f"/apply/view/lab_ok/{ts}/report/index.html")
        self.assertEqual(st, 200)
        self.assertIn("Label changes", body.decode())
        self.assertIn("default-src 'none'", T.hget(h, "content-security-policy"))
        img = re.search(r'src="(img/\d+\.jpg)"', body.decode())
        self.assertTrue(img)
        st, h, body = self.get("pujan", f"/apply/view/lab_ok/{ts}/report/{img.group(1)}")
        self.assertEqual((st, T.hget(h, "content-type"), body[:2]), (200, "image/jpeg", b"\xff\xd8"))
        st, _, _ = self.get("pujan", f"/apply/view/lab_ok/{ts}/labels/{info['lv']}/changes.jsonl")
        self.assertEqual(st, 200)
        # path traversal and odd names: refused, never another file
        (self.w.env.state / "secret.json").write_text('{"s": "TOPSECRET"}', encoding="utf-8")
        bad = [f"/apply/view/lab_ok/{ts}/../../../secret.json", f"/apply/view/lab_ok/{ts}/report/..%2F..%2F..%2Fsecret.json",
               f"/apply/view/lab_ok/{ts}/%2e%2e/%2e%2e/%2e%2e/secret.json", f"/apply/view/..%2F..%2Fsecret.json/{ts}/x.json",
               f"/apply/view/lab_ok/{ts}/report/index.htm",
               f"/apply/view/lab_ok/{ts}/report/.hidden.html", "/apply/view/lab_ok/notats/report/index.html",
               f"/apply/view/lab_ok/{ts}/C:%5Cx.json", f"/apply/view/lab_ok/{ts}/report//index.html"]
        for path in bad:
            st, _, body = self.get("pujan", path)
            self.assertIn(st, (400, 404), path)
            self.assertNotIn(b"TOPSECRET", body, path)
        for raw in (f"GET /apply/view/lab_ok/{ts}/..\\..\\..\\secret.json HTTP/1.1\r\n",):
            resp = T.raw(self.p.port, (raw + f"Host: 127.0.0.1\r\nCookie: {A.COOKIE}={self.tok['pujan']}\r\n"
                                       "Connection: close\r\n\r\n").encode())
            self.assertIn(T.raw_status(resp), (400, 404))
            self.assertNotIn(b"TOPSECRET", resp)
        (self.w.env.state / "secret.json").unlink()


if __name__ == "__main__":
    unittest.main()
