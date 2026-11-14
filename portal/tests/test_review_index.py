"""
The one review list (round 13, portal/review_index.py) and the reviewers write (portal/modules/reviews.py
write_reviewers): may_open before any row is built, the field allowlist, audits as counts only, core / excluded /
dot-folder / unservable folders dropped, the status order, needs_you, the switcher groups, titles.py's blind mask and
dot-folder skip, and the locked atomic write (two writers + a reader holding the file).

Temp fixture roots only (a temp review root, services config, predict root and registry); never CryoAI-laptop.

  python -m unittest portal.tests.test_review_index -v
"""
import itertools
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "annotation" / "tests"))
sys.path.append(str(REPO / "tests"))
import cryoev_testkit as TK  # noqa: E402
import make_fixture  # noqa: E402
from annotation import multi_review_server as MRS  # noqa: E402
from portal import listing  # noqa: E402
from portal import review_index as RI  # noqa: E402
from portal import titles as TT  # noqa: E402
from portal.modules import Ctx  # noqa: E402
from portal.modules import reviews as REV  # noqa: E402

PRED_A = "pred_ps_20261112_aaaaa1"      # bas's own
PRED_B = "pred_ps_20261112_bbbbb2"      # other's, bas is a reviewer
PRED_C = "pred_ps_20261112_ccccc3"      # other's, finished (every card decided)
REV_LAB = "lab_revise_fixture_20261112"  # compare, purpose revise: a label set review (applied)
AUDIT = "lab_compare_fixture3_20261111"  # the portal's own real-review package (core), a blind audit
AUDIT2 = "lab_compare_served_20261110"   # a served blind audit owned by audowner
DECOYS = 7777                            # a decoy count that must never leave review_package.json
FIELDS = set(RI.FIELDS)


def user(uid, role="basic"):
    return {"id": uid, "role": role}


def build(root: Path, name: str, meta: dict, key=True) -> Path:
    pkg = root / name
    make_fixture.build(pkg)
    (pkg / "review_package.json").write_text(json.dumps(meta), encoding="utf-8")
    if key:
        (pkg / ".review_key").write_text("k" * 20, encoding="utf-8")
    return pkg


_TICK = itertools.count()


def put(pkg: Path, file: str, dec: dict, t=None):
    (pkg / "edits").mkdir(exist_ok=True)
    f = pkg / "edits" / f"{file}.json"
    f.write_text(json.dumps({"file": file, "edit": {"t": 1, "dec": dec}}), encoding="utf-8")
    if t is not None:
        os.utime(f, (t, t))
    t2 = time.time() + 5 + next(_TICK)                         # the caches follow the folder's mtime (a step per call:
    os.utime(pkg / "edits", (t2, t2))                           # time.time() can repeat within a clock tick)


def decide_all(pkg: Path):
    for file, keys in REV.package_cards(json.loads((pkg / "data.js").read_text(encoding="utf-8")
                                                   .split("=", 1)[1].strip().rstrip(";"))).items():
        put(pkg, file, {k: {"v": "keep:EV", "t": 1} for k in keys})


class World(unittest.TestCase):
    """A fixture review root: served pred_/lab_ packages, the core audit (+ its _work_ folder), a dot-folder, an
    unservable folder and a keyless package."""

    def setUp(self):
        self.enterContext(TK.env_patch(drop=("CJ_*",)))
        REV._cache.clear()
        REV._cards.clear()
        TT._cache.clear()
        self.tmp = Path(tempfile.mkdtemp(prefix="cj_rindex_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.root = self.tmp / "review"
        self.root.mkdir()
        self.audit = build(self.root, AUDIT, {
            "kind": "compare", "owner": "audowner", "reviewers": ["audrev"], "n_images": 4, "built": "2026-11-11T09:00",
            "decoys": {"n": DECOYS, "frac": 0.1}, "base": "C:\\Users\\someone\\CryoAI-laptop\\labels\\lv_20261030_0909",
            "compare": {"a": {"coco": "C:\\Users\\someone\\x\\lv_20261030_0909\\labels.coco.json"},
                        "b": {"kind": "model", "weights": "models\\secret_model_x\\best.pt"}}})
        work = self.root / f"_work_{AUDIT}"
        work.mkdir()
        (work / "decoys.json").write_text(json.dumps({"n": DECOYS}), encoding="utf-8")
        self.svc = self.tmp / "services.json"   # the single-package server serves the audit; lab_ readable
        self.svc.write_text(json.dumps({"services": [{"name": "review", "argv": [
            "py", "annotation/review_server.py", "--package", str(self.audit), "--port", "8766"]}]}), encoding="utf-8")
        os.environ["CRYOEV_SERVICES_CONFIG"] = str(self.svc)
        os.environ[MRS.PREFIX_ENV] = "pred_,lab_"
        self.pred_a = build(self.root, PRED_A, {"kind": "prediction", "owner": "bas", "model": "v5", "n_images": 4,
                                                "built": "2026-11-12T10:00"})
        self.pred_b = build(self.root, PRED_B, {"kind": "prediction", "owner": "other", "reviewers": ["bas"],
                                                "model": "v5", "n_images": 4, "built": "2026-11-12T11:00",
                                                "owner_via": "claude"})
        self.pred_c = build(self.root, PRED_C, {"kind": "prediction", "owner": "other", "reviewers": ["bas"],
                                                "model": "v5", "n_images": 4, "built": "2026-11-12T09:00"})
        self.lab = build(self.root, REV_LAB, {
            "kind": "compare", "purpose": "revise", "owner": "adm", "reviewers": ["bas"], "n_images": 4,
            "compare": {"a": {"coco": "x/lv_20261030_0909/labels.coco.json"}, "b": {"weights": "models/v5m/best.pt"}}})
        self.audit2 = build(self.root, AUDIT2, {"kind": "compare", "owner": "audowner", "reviewers": [],
                                                "n_images": 4, "decoys": {"n": DECOYS}})
        build(self.root, ".building_pred_ps_20261112_ddddd4", {"kind": "prediction", "owner": "bas"})
        build(self.root, "prelipo_v5F5_review_v2", {"kind": "review", "owner": "bas"})
        build(self.root, "pred_ps_20261112_eeeee5", {"kind": "prediction", "owner": "bas"}, key=False)
        self.records = {"label_version": {"lv_20261111_1409": {"pkg": REV_LAB, "status": "current"}}}
        self.cfg = SimpleNamespace(review_root=self.root, predict_root=self.tmp / "predict", registry=None,
                                   review_package=self.audit, manifest=None)
        p = mock.patch.object(RI, "_records", lambda cfg: self.records)
        p.start()
        self.addCleanup(p.stop)

    def rows(self, who, **kw):
        return {r["id"]: r for r in RI.list_reviews(self.cfg, who, Ctx.can_see, **kw)}


class ListTests(World):
    def test_basic_sees_only_own_and_assigned(self):
        for show_all in (False, True):                                     # ?show=all can't widen a basic list
            got = self.rows(user("bas"), show_all=show_all)
            self.assertEqual(set(got), {PRED_A, PRED_B, PRED_C, REV_LAB}, show_all)
        self.assertEqual(self.rows(user("bas"))[PRED_A]["role"], "yours")
        self.assertEqual(self.rows(user("bas"))[PRED_B]["role"], "reviewer")
        self.assertEqual(self.rows(user("nobody")), {})

    def test_admin_mine_all_core_sandbox(self):
        mine = self.rows(user("adm", "admin"))
        self.assertEqual(set(mine), {REV_LAB, AUDIT, RI.SANDBOX_ID})        # own + core + sandbox
        every = self.rows(user("adm", "admin"), show_all=True)
        self.assertEqual(set(every), {PRED_A, PRED_B, PRED_C, REV_LAB, AUDIT, AUDIT2, RI.SANDBOX_ID})
        self.assertNotIn(RI.SANDBOX_ID, self.rows(user("adm", "admin"), show_all=True, access_mode=True))
        for r in every.values():
            self.assertEqual(set(r), FIELDS)                               # the allowlist, nothing else

    def test_dropped_folders(self):
        every = json.dumps(RI.list_reviews(self.cfg, user("adm", "admin"), Ctx.can_see, show_all=True))
        for gone in (".building_", "prelipo_", "eeeee5", "_work_"):
            self.assertNotIn(gone, every)
        self.assertEqual(sum(1 for r in json.loads(every) if r["id"] == AUDIT), 1)   # core listed once

    def test_audit_counts_only(self):
        r = self.rows(user("adm", "admin"), show_all=True)[AUDIT]
        self.assertEqual((r["kind"], r["blind"], r["owner"], r["reviewers"], r["via"]), ("audit", True, None, [], None))
        self.assertEqual(r["open_url"], "/review/")
        self.assertFalse(r["can_assign"])
        self.assertEqual(r["made_from"], "Blind comparison · 4 images")
        self.assertEqual((r["decided"], r["cards"]), REV.card_progress(self.audit))
        text = json.dumps(r)
        for leak in (str(DECOYS), "audowner", "audrev", "secret_model_x", "Users", "_work_", "decoys"):
            self.assertNotIn(leak, text)
        r2 = self.rows(user("audowner"))[AUDIT2]                           # its owner: no needs_you, no owner shown
        self.assertEqual((r2["role"], r2["needs_you"], r2["owner"], r2["can_assign"]), ("yours", False, None, True))

    def test_may_open_runs_before_any_row_is_built(self):
        seen = []
        real = REV.card_progress
        with mock.patch.object(REV, "card_progress", side_effect=lambda d: seen.append(Path(d).name) or real(d)):
            self.rows(user("bas"))
        self.assertEqual(set(seen), {PRED_A, PRED_B, PRED_C, REV_LAB})      # never another package, never the audit
        seen.clear()
        with mock.patch.object(REV, "card_progress", side_effect=lambda d: seen.append(Path(d).name) or real(d)):
            self.rows(user("adm", "admin"), show_all=True)
        self.assertIn(AUDIT, seen)                                          # control: the counter is really hooked

    def test_core_never_for_non_admin_even_if_listed_reviewer(self):
        meta = json.loads((self.audit / "review_package.json").read_text())
        meta["reviewers"] = ["bas"]
        (self.audit / "review_package.json").write_text(json.dumps(meta))
        self.assertNotIn(AUDIT, self.rows(user("bas"), show_all=True))

    def test_status_order(self):
        rows = self.rows(user("bas"))
        self.assertEqual(rows[REV_LAB]["status"], "applied")               # a label set was built from it
        self.assertEqual(rows[PRED_A]["status"], "not_started")
        put(self.pred_a, "fixA_new.png", {"g104": {"v": "keep:EV", "t": 1}})
        decide_all(self.pred_c)
        rows = self.rows(user("bas"))
        self.assertEqual(rows[PRED_A]["status"], "in_progress")
        self.assertEqual(rows[PRED_C]["status"], "finished")
        self.assertEqual(rows[PRED_C]["mode"], "editor")
        self.assertEqual(rows[PRED_A]["mode"], "cards")
        self.records["review_package"] = {PRED_C: {"status": "in_review", "status_by": "adm"},
                                          REV_LAB: {"status": "reviewed", "status_by": "adm"},
                                          PRED_A: {"status": "applied"}}       # no status_by: not explicit
        rows = self.rows(user("bas"))
        self.assertEqual(rows[PRED_C]["status"], "in_progress")             # explicit beats finished
        self.assertEqual(rows[REV_LAB]["status"], "finished")              # explicit beats applied
        self.assertEqual(rows[PRED_A]["status"], "in_progress")

    def test_needs_you_order_and_rules(self):
        decide_all(self.pred_c)                                             # finished: not needed
        put(self.pred_b, "fixA_new.png", {"g104": {"v": "keep:EV", "t": 1}}, t=time.time() - 3600)
        put(self.pred_a, "fixA_new.png", {"g104": {"v": "keep:EV", "t": 1}}, t=time.time() - 60)
        need = RI.needs_you(list(self.rows(user("bas")).values()))
        self.assertEqual([r["id"] for r in need], [PRED_A, PRED_B])        # newest activity first; applied not needed
        self.assertEqual(RI.needs_you([]), [])

    def test_revise_with_a_work_folder_is_an_audit(self):
        self.assertEqual(self.rows(user("bas"))[REV_LAB]["kind"], "label_review")
        (self.root / f"_work_{REV_LAB}").mkdir()                            # B4's rule: purpose revise AND no _work_
        r = self.rows(user("bas"))[REV_LAB]
        self.assertEqual((r["kind"], r["blind"], r["owner"], r["reviewers"]), ("audit", True, None, []))
        meta = json.loads((self.audit2 / "review_package.json").read_text())
        self.assertNotIn("purpose", meta)                                   # no purpose at all: an audit
        self.assertEqual(self.rows(user("audowner"))[AUDIT2]["kind"], "audit")

    def test_non_admin_audit_title_names_no_set(self):
        """fixer 3: a basic reviewer's audit row is "Blind audit · <built date>" / "Blind comparison · N images":
        the hub and strip HTML never hold either compared set (A's label set, B's model or label set, a meta title
        naming them); the admin keeps the descriptive title (control: the names really are in the package)."""
        from portal import app as APP
        pid = "lab_compare_blindx_20261112"
        a_lv, b_lv, model = "lv_20261031_0707", "lv_20261029_0606", "v5_secretmodel_y8m"
        build(self.root, pid, {"kind": "compare", "purpose": "audit", "owner": "adm", "reviewers": ["bas"],
                               "n_images": 4, "built": "2026-11-12T10:00",
                               "title": f"Audit {a_lv} vs {model}", "base": f"x/{a_lv}",
                               "compare": {"a": {"id": a_lv, "coco": f"x/{a_lv}/labels.coco.json"},
                                           "b": {"kind": "model", "id": model,
                                                 "weights": f"models/{model}/best.pt", "label": b_lv}}})
        (self.root / f"_work_{pid}").mkdir()
        secret = (a_lv, b_lv, model, "secretmodel")
        rows = RI.list_reviews(self.cfg, user("bas"), Ctx.can_see)
        r = next(x for x in rows if x["id"] == pid)
        self.assertEqual((r["kind"], r["title"], r["made_from"]),
                         ("audit", "Blind audit · 12 Nov 2026", "Blind comparison · 4 images"))
        hub = REV.render_hub(rows, user=user("bas"), params={}, flash_html="")
        strip = APP.render_strip(rows, pid, RI.switcher_groups(rows, pid))
        self.assertIn("Blind audit · 12 Nov 2026", hub)
        self.assertIn("Blind audit · 12 Nov 2026", strip)
        for text in (hub, strip, json.dumps(rows)):
            for s_ in secret:
                self.assertNotIn(s_, text)
        adm = self.rows(user("adm", "admin"), show_all=True)[pid]
        self.assertTrue(any(s_ in adm["title"] for s_ in secret), adm["title"])        # control: admins see it
        self.assertEqual(adm["made_from"], "Blind comparison · 4 images")
        self.assertEqual(RI._blind_title(None), "Blind audit")

    def test_package_being_built_is_skipped(self):
        """A portal build writes data.js, images and .review_key before review_package.json: not listed yet."""
        d = self.root / "pred_ps_20261112_fffff6"
        make_fixture.build(d)
        (d / ".review_key").write_text("k" * 20, encoding="utf-8")
        lab = self.root / "lab_revise_building_20261113"
        make_fixture.build(lab)
        (lab / ".review_key").write_text("k" * 20, encoding="utf-8")
        (self.root / ".building" / lab.name).mkdir(parents=True)          # review_build.py's status folder
        (self.root / ".building" / lab.name / "status.json").write_text("{}", encoding="utf-8")
        every = self.rows(user("adm", "admin"), show_all=True)
        self.assertNotIn(d.name, every)
        self.assertNotIn(lab.name, every)
        scanned = {p["id"] for p in TT.scan_packages(self.root, {})}
        self.assertNotIn(lab.name, scanned)
        self.assertNotIn(".building", scanned)
        self.assertIn(AUDIT2, scanned)                                      # control: a built one is listed
        TT._cache.clear()
        shutil.rmtree(self.root / ".building")                             # an older build_review_package.py folder
        self.assertIn(lab.name, {p["id"] for p in TT.scan_packages(self.root, {})})   # (no json ever) stays listed

    def test_titles_and_made_from(self):
        (self.cfg.predict_root / PRED_A[5:]).mkdir(parents=True)
        (self.cfg.predict_root / PRED_A[5:] / "request.json").write_text(json.dumps(
            {"name": "My <b>EV</b> run", "source": {"kind": "session", "session": "20261106-2"}, "owner_via": "claude"}))
        rows = self.rows(user("bas"))
        self.assertEqual(rows[PRED_A]["title"], "My <b>EV</b> run")         # escaped by the page, not here
        self.assertEqual(rows[PRED_A]["made_from"], "v5 predictions · 4 images · Session 20261106-2")
        self.assertEqual(rows[PRED_A]["via"], "claude")                     # from request.json
        self.assertEqual(rows[PRED_B]["via"], "claude")                     # from review_package.json
        self.assertEqual(rows[REV_LAB]["title"], "Revise lv_20261030_0909 against v5m")
        self.assertEqual(rows[REV_LAB]["kind"], "label_review")

    def test_facets_from_listing_for_prediction_rows_only(self):
        calls, hiders = [], []

        def fake(ps_dir, request, *, viewer=None, hidden=None):
            calls.append((Path(ps_dir).name, viewer["id"], callable(hidden)))
            hiders.append(hidden)
            return {"model": "v5", "person": "x"}

        with mock.patch.object(listing, "facets", fake):                    # hidden is the viewer's real rule
            self.rows(user("bas"))
            self.assertTrue(hiders and all(h is hiders[0] for h in hiders))  # built once per list
            self.assertTrue(hiders[0]("prediction_set", "ps_20261101_ffffff"))   # someone else's private set
            hiders.clear()
            self.rows(user("adm", "admin"))                                 # admin Mine: no prediction rows,
            self.assertEqual(hiders, [])                                    # so no facets are built at all
            self.rows(user("adm", "admin"), show_all=True)
            self.assertFalse(hiders[0]("prediction_set", "ps_20261101_ffffff"))  # control: an admin sees it
            calls.clear()

        with mock.patch.object(listing, "facets", fake):
            rows = self.rows(user("adm", "admin"), show_all=True)
            self.assertEqual(rows[PRED_A]["facets"], {"model": "v5", "person": "x"})
            self.assertEqual(rows[AUDIT]["facets"], {})
            self.assertEqual(rows[REV_LAB]["facets"], {})
            self.assertEqual({c[0] for c in calls}, {PRED_A[5:], PRED_B[5:], PRED_C[5:]})
            self.assertTrue(all(c[2] for c in calls))
            calls.clear()
            self.rows(user("adm", "admin"), show_all=True, with_facets=False)
            self.assertEqual(calls, [])
        with mock.patch.object(listing, "facets", side_effect=RuntimeError("boom")):
            self.assertEqual(self.rows(user("bas"))[PRED_A]["facets"], {})  # a facet failure never hides the row

    def test_excluded_and_unreadable_config(self):
        os.environ[MRS.EXCLUDE_ENV] = REV_LAB
        self.assertNotIn(REV_LAB, self.rows(user("bas")))
        os.environ.pop(MRS.EXCLUDE_ENV)
        self.svc.write_text("{nope")                                        # unreadable: no lab_ at all
        self.assertEqual(set(self.rows(user("adm", "admin"), show_all=True)) & {REV_LAB, AUDIT2}, set())

    def test_switcher_groups(self):
        rows = list(self.rows(user("adm", "admin"), show_all=True).values())
        for r in rows:
            r["needs_you"] = r["id"] == PRED_B
        groups = RI.switcher_groups(rows, PRED_C, per_group=1)
        self.assertEqual(groups[0][0], "Needs you")
        self.assertEqual([r["id"] for r in groups[0][1]], [PRED_B])
        heads = [g[0] for g in groups]
        self.assertEqual(heads, ["Needs you", "Prediction reviews", "Label set reviews", "Blind audits", "Practice"])
        pred = dict(groups)["Prediction reviews"]
        self.assertIn(PRED_C, [r["id"] for r in pred])                     # the current one kept past the cap
        self.assertNotIn(PRED_B, [r["id"] for r in pred])                  # not twice


class TitlesTests(World):
    def test_scan_skips_dot_folders_and_masks_who_on_blind(self):
        put(self.audit2, "fixA_new.png", {"g104": {"v": "keep:EV", "t": 1, "who": "audrev", "via": "claude"}})
        put(self.lab, "fixA_new.png", {"g104": {"v": "keep:EV", "t": 2, "who": "bas"}})
        (self.root / ".meta_locks").mkdir()
        by = {p["id"]: p for p in TT.scan_packages(self.root, {})}
        self.assertFalse(any(k.startswith(".") for k in by))
        self.assertEqual((by[AUDIT2]["who"], by[AUDIT2]["via"]), (None, None))
        self.assertEqual(by[REV_LAB]["who"], "bas")                         # a revise compare is not blind
        self.assertTrue(TT.is_blind({"kind": "compare"}))
        self.assertFalse(TT.is_blind({"kind": "compare", "purpose": "revise"}))
        self.assertEqual(TT.applied_map({"label_version": {"lv_1": {"pkg": "p", "status": "superseded"},
                                                           "lv_2": {"pkg": "p"}}}), {"p": "lv_2"})


class WriteTests(World):
    ROSTER = {"bas", "other", "other2", "audowner", "adm"} | {f"p{i}" for i in range(25)}

    def meta(self, pkg):
        return json.loads((pkg / "review_package.json").read_text(encoding="utf-8"))

    def test_changes_only_reviewers(self):
        before = self.meta(self.pred_b)
        self.assertEqual(REV.write_reviewers(self.root, self.pred_b, ["other2"], ["bas"], allowed=self.ROSTER),
                         (["other2"], ["bas"]))
        after = self.meta(self.pred_b)
        self.assertEqual(after.pop("reviewers"), ["other2"])
        before.pop("reviewers")
        self.assertEqual(after, before)
        self.assertTrue((self.root / ".meta_locks" / f"{PRED_B}.lock").exists())
        self.assertEqual([p.name for p in self.pred_b.glob("*.tmp")], [])
        self.assertEqual(REV.write_reviewers(self.root, self.pred_b, ["other2"], [], allowed=self.ROSTER), ([], []))

    def test_refusals(self):
        with self.assertRaisesRegex(REV.ReviewersError, "roster"):
            REV.write_reviewers(self.root, self.pred_b, ["ghost"], allowed=self.ROSTER)
        with self.assertRaisesRegex(REV.ReviewersError, "audit's owner"):
            REV.write_reviewers(self.root, self.audit2, ["audowner"], allowed=self.ROSTER)
        self.assertEqual(REV.write_reviewers(self.root, self.lab, ["adm"], allowed=self.ROSTER), (["adm"], []))  # revise
        with self.assertRaisesRegex(REV.ReviewersError, "at most 20"):
            REV.write_reviewers(self.root, self.pred_a, [f"p{i}" for i in range(21)], allowed=self.ROSTER)
        self.assertNotIn("reviewers", self.meta(self.pred_a))               # nothing written on a refusal

    def test_busy_lock(self):
        from data_utils.oslock import acquire_lock
        (self.root / ".meta_locks").mkdir()
        held = acquire_lock(self.root / ".meta_locks" / f"{PRED_A}.lock")   # another process's lock (same API)
        try:
            if os.name == "nt":                                             # msvcrt locks are per handle: refused
                with self.assertRaisesRegex(REV.ReviewersError, "right now"):
                    REV.write_reviewers(self.root, self.pred_a, ["other"], allowed=self.ROSTER, lock_wait=0)
        finally:
            held.close()
        self.assertEqual(REV.write_reviewers(self.root, self.pred_a, ["other"], allowed=self.ROSTER), (["other"], []))

    def test_race_two_writers_and_a_reader(self):
        """Two threads add different people while a reader holds review_package.json open: both land (lock +
        re-read), nothing else changes, no temp file is left."""
        before = self.meta(self.pred_a)
        f = self.pred_a / "review_package.json"
        opened, errors = threading.Event(), []

        def reader():
            with open(f, "rb"):
                opened.set()
                time.sleep(0.4)                                             # under the ~1 s replace retry

        def writer(pid):
            try:
                REV.write_reviewers(self.root, self.pred_a, [pid], allowed=self.ROSTER)
            except Exception as e:                                          # noqa: BLE001
                errors.append(e)

        r = threading.Thread(target=reader)
        r.start()
        opened.wait(5)
        ws = [threading.Thread(target=writer, args=(p,)) for p in ("other", "other2")]
        for w in ws:
            w.start()
        for t in ws + [r]:
            t.join(15)
        self.assertEqual(errors, [])
        after = self.meta(self.pred_a)
        self.assertEqual(sorted(after.pop("reviewers")), ["other", "other2"])
        self.assertEqual(after, before)
        self.assertEqual(list(self.pred_a.glob("*.tmp")), [])

    @unittest.skipUnless(os.name == "nt", "only Windows refuses to replace an open file")
    def test_reader_holding_too_long_gives_a_clear_error(self):
        f = self.pred_a / "review_package.json"
        before = f.read_bytes()
        with open(f, "rb"):
            with self.assertRaisesRegex(REV.ReviewersError, "open elsewhere"):
                REV.write_reviewers(self.root, self.pred_a, ["other"], allowed=self.ROSTER, retry_s=0.2)
        self.assertEqual(f.read_bytes(), before)
        self.assertEqual(list(self.pred_a.glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
