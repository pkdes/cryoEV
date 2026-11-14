"""
Readable titles, "Used by / Waiting on it" and the Jobs "Review packages" section (portal/titles.py,
portal/modules/data.py, portal/modules/jobs.py): known-answer fixtures in a temp registry dict and a temp review root
(a compare package with edits, a lab labeling package, an applied package, a prediction package, a private decoys
file), visibility (advanced never gets the admin-only section; no private prediction id leaks), escaping.

  python -m unittest portal.tests.test_vis -v
"""
import contextlib
import importlib
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
from portal import titles as TT  # noqa: E402
from portal.modules import data as D  # noqa: E402
from portal.modules import jobs as J  # noqa: E402

DR = importlib.import_module("portal.modules.data.routes")   # the routes part (D.routes is the function)

LV = "lv_20261030_0909"
LV0 = "lv_20261030_0846"
PULL = "roboflow_pull_20261029_1811"
XSS = "<script>alert(1)</script>"
AUDIT = "Audit: v5 at conf 0.55 vs Roboflow pull 29 Oct, edited 30 Oct"
LV_TITLE = "Roboflow pull 29 Oct + Lab labeling: lab_pkg_applied · 28 images changed · 30 Oct"


@contextlib.contextmanager
def fake_registry(recs):
    """read_registry answers `recs`, patched where it is looked up at call time: on the package (jobs.py reads
    DATA.read_registry) and on the routes part (the Data routes' own lookup), the two places one patch of data.py
    reached before the round 11 split (portal/tests/test_data_patch_guard.py)."""
    def fake(root, now=None):
        return recs, None
    with mock.patch.object(D, "read_registry", fake), mock.patch.object(DR, "read_registry", fake):
        yield


def records():
    r = {e: {} for e in D.ENTITIES}
    r["pull"][PULL] = {"id": PULL, "created": "2026-10-29T18:11"}
    r["label_version"][LV0] = {"id": LV0, "base": PULL, "created": "2026-10-30T08:46"}
    r["label_version"][LV] = {"id": LV, "base": PULL, "pkg": "lab_pkg_applied", "n_images_applied": 28,
                              "created": "2026-10-30T09:09", "supersedes": LV0}
    r["label_version"]["lv_20261101_1000"] = {"id": "lv_20261101_1000", "base": LV, "created": "2026-11-01T10:00",
                                              "n_images_applied": 1}
    r["dataset"]["ds1"] = {"id": "ds1", "label_version": LV}
    r["model"]["m1"] = {"id": "m1", "dataset": "ds1"}
    r["model"]["m2"] = {"id": "m2", "dataset": "other"}
    r["review_package"]["lab_pkg_applied"] = {"id": "lab_pkg_applied", "kind": "review"}
    return r


def data_js(pkg_id, source, images):
    d = {"id": pkg_id, "kind": "compare", "source": source, "images": images}
    return "window.REVIEW_DATA = " + json.dumps(d) + ";"


def image(name, gt_u, pr_u, extra_gt=()):
    return {"file": name, "gt": [{"id": i, "c": "EV", "u": 1} for i in gt_u] + [{"id": i, "c": "EV"} for i in extra_gt],
            "pr": [{"c": "EV", "s": 0.9, **({"u": 1} if u else {})} for u in pr_u]}


def edit(name, keys, reviewed=False, who="alice", t=1000):
    return {"file": name, "edit": {"dec": {k: {"v": "keep", "t": t, "who": who} for k in keys},
                                   "reviewed": reviewed}}


def put(path: Path, text: str):
    path.write_text(text, encoding="utf-8")


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.root = self.tmp / "laptop" / "review"
        self.root.mkdir(parents=True)
        src = f"C:\\x\\labels\\{LV}\\labels.coco.json"
        # compare package: cards g1 g2 p0 (a.png) and g5 (b.png) = 4; decided g1, p0 (+ "t:p0" and a new-box key)
        cmp_dir = self.root / "lab_compare_x"
        (cmp_dir / "edits").mkdir(parents=True)
        put(cmp_dir / "review_package.json", json.dumps({
            "id": "lab_compare_x", "kind": "compare", "base": src, "base_kind": "label_version",
            "compare": {"a": {"kind": "labels", "coco": src}, "b": {"kind": "model"}, "label": "v5 at conf 0.55"}}))
        put(cmp_dir / "data.js", data_js("lab_compare_x", src, [
            image("a.png", [1, 2], [True, False], extra_gt=[3]), image("b.png", [5], [])]))
        put(cmp_dir / "edits" / "a.png.json",
            json.dumps(edit("a.png", ["g1", "p0", "t:p0", "newbox1"], reviewed=True, who="bob", t=2000)))
        put(cmp_dir / "decoys.json", '{"SECRET": 1}')
        (self.root / "_work_lab_compare_x").mkdir()
        put(self.root / "_work_lab_compare_x" / "decoys.json", '{"SECRET": 1}')
        # lab labeling package, no edits: ready (base found through data.js source)
        lab = self.root / "lab_pkg_ready"
        (lab / "edits").mkdir(parents=True)
        put(lab / "data.js", data_js("lab_pkg_ready", src, [image("c.png", [7], [True])]))
        # applied package
        ap = self.root / "lab_pkg_applied"
        (ap / "edits").mkdir(parents=True)
        put(ap / "data.js", data_js("lab_pkg_applied", "C:\\x\\roboflow_pull_20261029_1811\\c.json",
                                    [image("d.png", [9], [])]))
        put(ap / "edits" / "d.png.json", json.dumps(edit("d.png", ["g9"])))
        # a prediction package (private) and a part: never listed
        pr = self.root / "pred_ps_20261101_aaaaaa"
        (pr / "edits").mkdir(parents=True)
        put(pr / "review_package.json", json.dumps({"kind": "prediction", "owner": "carol"}))
        put(pr / "data.js", data_js("pred_ps_20261101_aaaaaa", "x", [image("e.png", [1], [])]))
        part = self.root / "lab_part1"
        part.mkdir()
        put(part / "review_package.json", json.dumps({"kind": "part", "parent": "lab_pkg_ready"}))
        put(part / "data.js", data_js("lab_part1", src, [image("c.png", [7], [])]))
        self.recs = records()
        TT._cache.clear()
        D._files_cache.clear()
        D._owner_cache.clear()
        env = mock.patch.dict(os.environ, {"CRYOEV_LAPTOP_ROOT": str(self.tmp / "laptop")})
        env.start()
        self.addCleanup(env.stop)


class TitleTests(Base):
    def test_label_title_known_answer(self):
        meta = lambda pid: {"kind": "review"}  # noqa: E731
        self.assertEqual(TT.label_title(self.recs, LV, meta), LV_TITLE)
        self.assertEqual(TT.label_title(self.recs, LV0, meta), "Roboflow pull 29 Oct · 30 Oct")

    def test_label_title_recursive_short_base(self):
        self.assertEqual(TT.label_title(self.recs, "lv_20261101_1000"),
                         "Roboflow pull 29 Oct, edited 30 Oct · 1 image changed · 1 Nov")

    def test_label_title_without_facts_falls_back(self):
        self.assertEqual(TT.label_title({"label_version": {"lv_x": {}}}, "lv_x"), "lv_x")

    def test_package_titles_by_kind(self):
        meta = json.loads((self.root / "lab_compare_x" / "review_package.json").read_text())
        self.assertEqual(TT.package_title(self.recs, "lab_compare_x", meta), AUDIT)
        self.assertEqual(TT.package_title(self.recs, "lab_pkg_ready", {}), "Lab labeling: lab_pkg_ready")
        self.assertEqual(TT.package_title(self.recs, "pred_ps_1", {"kind": "prediction"}, lambda p: "My run"), "My run")
        self.assertEqual(TT.package_title(self.recs, "pred_ps_1", {"kind": "prediction"}), "Prediction review")


    def test_base_title_unknown_ref_never_a_path(self):
        recs = {"label_version": {"lv_x": {"base": r"C:\Users\someone\roboflow - X\train\a.json", "created": "2026-10-30T09:00"}}}
        t = TT.label_title(recs, "lv_x")
        self.assertEqual(t, "Earlier labels · 30 Oct")
        self.assertNotIn("Users", t)
        self.assertEqual(TT._base_title({}, "/some/path", 0), "Earlier labels")
        self.assertEqual(TT._base_title({}, PULL, 0), "Roboflow pull 29 Oct")  # a pull id stays readable

    def test_read_json_caps_size(self):
        f = self.tmp / "big.json"
        f.write_text('{"a": "' + "x" * (TT.MAX_JSON_BYTES + 10) + '"}', encoding="utf-8")
        self.assertIsNone(TT.read_json(f))
        g = self.tmp / "ok.json"
        g.write_text('{"a": 1}', encoding="utf-8")
        self.assertEqual(TT.read_json(g), {"a": 1})
        self.assertIsNone(TT.read_json(self.tmp / "missing.json"))
        put(self.root / "lab_pkg_ready" / "review_package.json", f.read_text(encoding="utf-8"))  # scan survives it
        self.assertIn("lab_pkg_ready", [p["id"] for p in TT.scan_packages(self.root, self.recs)])

    def test_scan_packages_cached_for_the_ttl(self):
        clock = [1000.0]
        with mock.patch.object(TT.time, "monotonic", lambda: clock[0]):
            first = TT.scan_packages(self.root, self.recs)
            self.assertEqual(len(first), 3)
            first[0]["title"] = "mutated"  # a caller's edit must not reach the cache
            new = self.root / "lab_pkg_new"
            (new / "edits").mkdir(parents=True)
            put(new / "data.js", data_js("lab_pkg_new", "x", [image("n.png", [1], [])]))
            with mock.patch.object(TT, "_scan_packages", side_effect=AssertionError("re-scanned")):
                again = TT.scan_packages(self.root, self.recs)  # 5 s later: served from the cache
                clock[0] += 5
                again = TT.scan_packages(self.root, self.recs)
            self.assertEqual(len(again), 3)
            self.assertNotEqual(again[0]["title"], "mutated")
            clock[0] += TT.SCAN_TTL
            self.assertEqual(len(TT.scan_packages(self.root, self.recs)), 4)  # expired: the new package shows
            other = dict(self.recs, label_version={})  # different records: not the same cache entry
            self.assertEqual(len(TT.scan_packages(self.root, other)), 4)


class ProgressTests(Base):
    def test_progress_counts(self):
        p = TT.package_progress(self.root / "lab_compare_x")
        self.assertEqual((p["cards"], p["decided"], p["images"], p["reviewed"], p["who"]), (4, 2, 2, 1, "bob"))
        self.assertIsNotNone(p["last"])

    def test_scan_statuses_and_exclusions(self):
        by = {p["id"]: p for p in TT.scan_packages(self.root, self.recs)}
        self.assertEqual(sorted(by), ["lab_compare_x", "lab_pkg_applied", "lab_pkg_ready"])
        self.assertEqual(by["lab_compare_x"]["status"], "in_progress")
        self.assertEqual(by["lab_compare_x"]["waiting"], 2)
        self.assertEqual(by["lab_compare_x"]["base_lv"], LV)
        self.assertEqual(by["lab_pkg_ready"]["status"], "ready")
        self.assertEqual(by["lab_pkg_ready"]["base_lv"], LV)
        self.assertEqual((by["lab_pkg_applied"]["status"], by["lab_pkg_applied"]["applied_by"],
                          by["lab_pkg_applied"]["waiting"]), ("applied", LV, 0))

    def test_cache_follows_edits_folder_mtime(self):
        d = self.root / "lab_pkg_ready"
        self.assertEqual(TT.package_progress(d)["decided"], 0)
        put(d / "edits" / "c.png.json", json.dumps(edit("c.png", ["g7", "p0"])))
        self.assertEqual(TT.package_progress(d)["decided"], 2)

    def test_decoys_never_read(self):
        opened = []
        real = Path.read_text

        def spy(self_, *a, **k):
            opened.append(self_.name)
            return real(self_, *a, **k)
        with mock.patch.object(Path, "read_text", spy):
            TT.scan_packages(self.root, self.recs)
            with fake_registry(self.recs):
                J.review_packages(SimpleNamespace(review_root=self.root, registry=None))
        self.assertNotIn("decoys.json", opened)
        self.assertIn("data.js", opened)


class DataPageTests(Base):
    def viewer(self, role="admin", uid="admin1"):
        return D.Viewer(lambda user, owner: user["role"] == "admin" or owner == user["id"],
                        {"id": uid, "role": role})

    def test_label_detail_title_used_by_waiting(self):
        d = D.detail(self.recs, "labels", LV, self.viewer(), 1)
        self.assertEqual(d["title"], LV_TITLE)
        self.assertEqual([m["id"] for m in d["models"]], ["m1"])
        self.assertEqual([m["id"] for m in d["datasets"]], ["ds1"])
        self.assertEqual([(w["id"], w["n"]) for w in d["waiting"]], [("lab_compare_x", 2)])
        self.assertEqual(d["waiting"][0]["title"], AUDIT)
        html = D.render_detail(d, None)
        self.assertIn("Waiting on it", html)
        self.assertIn("2 decisions not applied yet", html)
        self.assertIn("Models trained from those datasets", html)
        self.assertIn(f'<p class="cj-mut cj-vis-id">{LV}</p>', html)

    def test_applied_package_not_waiting_and_other_lv_empty(self):
        self.assertEqual(D.detail(self.recs, "labels", LV0, self.viewer(), 1)["waiting"], [])
        self.recs["label_version"]["lv_z"] = {"id": "lv_z", "pkg": "lab_compare_x", "created": "2026-11-02T10:00"}
        self.assertEqual(D.detail(self.recs, "labels", LV, self.viewer(), 1)["waiting"], [])

    def test_list_rows_show_title_and_id(self):
        rows = {r["id"]: r for r in D.label_rows(self.recs, self.viewer())}
        self.assertTrue(rows[LV]["title"].startswith("Roboflow pull 29 Oct + "))
        self.assertIn(f'<span class="cj-mut cj-vis-id">{LV}</span>', D.render_rows([rows[LV]]))

    def test_review_package_set_titles_and_privacy(self):
        self.recs["review_package"]["lab_compare_x"] = {"id": "lab_compare_x", "kind": "compare"}
        self.recs["review_package"]["pred_ps_20261101_aaaaaa"] = {
            "id": "pred_ps_20261101_aaaaaa", "kind": "prediction", "owner": "carol"}
        mine = self.viewer("advanced", "dave")
        rows = D.image_set_rows(self.recs, mine)
        ids = [r["id"] for r in rows]
        self.assertIn("lab_compare_x", ids)
        self.assertNotIn("pred_ps_20261101_aaaaaa", ids)
        self.assertNotIn("pred_ps_20261101_aaaaaa", json.dumps(D.search(self.recs, "pred_ps", mine)))
        self.assertEqual(next(r["title"] for r in rows if r["id"] == "lab_compare_x"), AUDIT)
        carol = D.image_set_rows(self.recs, self.viewer("advanced", "carol"))
        self.assertIn("pred_ps_20261101_aaaaaa", [r["id"] for r in carol])

    def test_private_package_not_in_waiting_for_others(self):
        meta = self.root / "lab_compare_x" / "review_package.json"
        m = json.loads(meta.read_text())
        m["owner"] = "erin"
        put(meta, json.dumps(m))
        D._files_cache.clear()
        d = D.detail(self.recs, "labels", LV, self.viewer("advanced", "dave"), 1)
        self.assertEqual(d["waiting"], [])
        self.assertEqual(len(D.detail(self.recs, "labels", LV, self.viewer("advanced", "erin"), 1)["waiting"]), 1)

    def test_private_package_never_named_in_label_title(self):
        self.recs["label_version"]["lv_priv"] = {"id": "lv_priv", "base": PULL, "pkg": "pred_ps_20261101_aaaaaa",
                                                 "created": "2026-11-02T10:00", "n_images_applied": 1}
        self.recs["review_package"]["pred_ps_20261101_aaaaaa"] = {
            "id": "pred_ps_20261101_aaaaaa", "kind": "prediction", "owner": "carol"}
        dave, carol = self.viewer("advanced", "dave"), self.viewer("advanced", "carol")
        d = D.detail(self.recs, "labels", "lv_priv", dave, 1)
        rows = json.dumps(D.label_rows(self.recs, dave))
        self.assertEqual(d["title"], "Roboflow pull 29 Oct · 1 image changed · 2 Nov")
        self.assertNotIn("pred_ps_20261101_aaaaaa", json.dumps(d) + rows)
        self.assertIn("Prediction review", D.detail(self.recs, "labels", "lv_priv", carol, 1)["title"])

    def test_escaping(self):
        self.recs["label_version"][LV]["pkg"] = XSS
        self.recs["label_version"][LV]["base"] = XSS
        html = D.render_detail(D.detail(self.recs, "labels", LV, self.viewer(), 1), None)
        self.assertNotIn(XSS, html)
        self.assertIn("&lt;script&gt;", html)
        self.assertNotIn(XSS, D.render_rows(D.label_rows(self.recs, self.viewer())))


class JobsTests(Base):
    def test_summary_admin_vs_advanced(self):
        cfg = SimpleNamespace(review_root=self.root, registry=None)
        with fake_registry(self.recs):
            adm = J.summary(cfg, {"id": "a", "role": "admin"}, lambda u, o: True)
            adv = J.summary(cfg, {"id": "b", "role": "advanced"}, lambda u, o: o == "b")
        self.assertIsNone(adv["pkgs"])
        html_adv = "".join(J.render(adv).values())
        self.assertNotIn("Review packages", html_adv)
        self.assertNotIn("lab_compare_x", html_adv)
        by = {p["id"]: p for p in adm["pkgs"]}
        self.assertEqual(sorted(by), ["lab_compare_x", "lab_pkg_applied", "lab_pkg_ready"])
        self.assertNotIn("pred_ps_20261101_aaaaaa", json.dumps(adm["pkgs"]))
        c = by["lab_compare_x"]
        self.assertEqual((c["decided"], c["cards"], c["reviewed"], c["images"], c["who"], c["via"], c["status"]),
                         (2, 4, 1, 2, None, None, "in_progress"))     # round 13: no who / via on a compare
        self.assertNotIn("bob", json.dumps(adm["pkgs"]))
        self.assertEqual(by["lab_pkg_applied"]["who"], "alice")       # control: a lab labeling package keeps who
        self.assertEqual(by["lab_pkg_applied"]["applied_by"], LV)
        rendered = J.render(adm)
        html = rendered["finished"] + rendered["review"]                # in review moved up; applied stays below
        self.assertIn("Applied review packages", rendered["finished"])
        self.assertNotIn("lab_pkg_applied", rendered["review"])
        self.assertIn("2 of 4 cards decided", rendered["review"])
        self.assertIn("1 of 2 images reviewed", rendered["review"])
        self.assertIn("last edit", html)
        self.assertIn(f"/data/labels/{LV}", html)
        self.assertNotIn("SECRET", html)
        self.assertNotIn("pred_ps_20261101_aaaaaa", html)
        self.assertNotIn("bob", "".join(rendered.values()))           # who decided the compare's cards: never
        self.assertIn("by alice", rendered["finished"])                # control

    def test_render_escapes(self):
        row = {"id": XSS, "title": XSS, "kind": "review", "cards": 1, "decided": 0, "images": 1, "reviewed": 0,
               "last": XSS, "who": XSS, "status": "applied", "applied_by": XSS}
        html = J.render_packages([row])
        self.assertNotIn(XSS, html)
        self.assertIn("&lt;script&gt;", html)


if __name__ == "__main__":
    unittest.main()
