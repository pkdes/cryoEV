"""
Jobs board (round 9, portal/modules/jobs.py): the data-flow columns, the one link helper href_for, empty states,
the attribution label ("claude for pujan" + AI tag) with the owner's privacy mask, one fragment per column for the
30 s refresh, and the "finished, not recorded" check. Temp roster / registry / live.json / raw + predict roots only
(cjtest.Env); CRYOEV_* cleared in setUp so an agent's own environment never leaks into a fixture.

  python -m unittest portal.tests.test_jobs_board -v
"""
import json
import os
import re
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cjtest as T  # noqa: E402
from portal import auth as A  # noqa: E402
from portal.modules import data as D  # noqa: E402
from portal.modules import jobs as J  # noqa: E402

SESS_RAW, SESS_DONE, SESS_PRED = "20261101", "20261102", "20261103"
CID_RAW, CID_DONE, CID_PRED = f"{SESS_RAW}_aaaaaa", f"{SESS_DONE}_bbbbbb", f"{SESS_PRED}_cccccc"
PS = "ps_20261103_abcdef"
PKG = f"pred_{PS}"
PKG_OTHER = "pred_ps_20261104_bbbbbb"
DS, LV = "ds_20261101_aaaa", "lv_20261101_1000"
HREF = re.compile(r'class="cj-jobs-open" href="([^"]*)"')


def jl(root: Path, entity: str, recs):
    (root / f"{entity}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")


def wjson(p: Path, doc):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc), encoding="utf-8")


def make_roots(tmp: Path):
    raw, pred = tmp / "raw", tmp / "predict"
    for cid, sess, state, n in ((CID_RAW, SESS_RAW, "running", None), (CID_DONE, SESS_DONE, "done", 12),
                                (CID_PRED, SESS_PRED, "done", 30)):
        d = raw / "raw_tiles" / cid
        wjson(d / "convert.json", {"schema": "cj_convert/1", "id": cid, "owner": "tester", "session": sess, "recipe": "default",
                                   "created": "2026-11-11T09:00:00"})
        wjson(d / "status.json", {"state": state, "n_tiles": n})
    wjson(pred / PS / "request.json", {"ps": PS, "owner": "tester", "source": {"kind": "session", "session": SESS_PRED},
                                       "created": "2026-11-11T10:00:00"})
    wjson(pred / PS / "status.json", {"state": "done", "n_done": 30, "n_total": 30})
    return raw, pred


def registry(root: Path):
    jl(root, "review_package", [
        {"schema": "review_package/1", "id": PKG, "kind": "prediction", "prediction_set": PS, "status": "in_review",
         "owner": "tester", "model": "v5", "n_images": 30, "via": "claude", "logged": "2026-11-11T10:05:00"},
        {"schema": "review_package/1", "id": PKG_OTHER, "kind": "prediction", "status": "built",
         "owner": "otherperson", "via": "claude", "logged": "2026-11-11T10:06:00"}])
    jl(root, "label_version", [{"schema": "label_version/1", "id": LV, "status": "created",
                                "logged": "2026-11-01T10:00:00", "created": "2026-11-01T10:00:00"}])
    jl(root, "dataset", [{"schema": "dataset/1", "id": DS, "label_version": LV, "status": "frozen",
                          "logged": "2026-11-01T11:00:00"},
                         {"schema": "dataset/1", "id": "ds_20261031_old", "label_version": LV, "status": "retired",
                          "logged": "2026-10-31T11:00:00"}])
    jl(root, "run", [{"schema": "run/1", "id": f"r{i}", "dataset": DS, "status": "finished"} for i in (1, 2)])

    def job(id_, name, typ, where, status, **kw):
        return {"schema": "job/1", "id": id_, "logged": "2026-11-11T08:00:00", "name": name, "type": typ,
                "where": where, "status": status, **kw}
    jl(root, "job", [
        job("job_20261111_revise_mine", "Review my tiles", "revise", "app", "in_progress", who="tester", via="claude"),
        job("job_20261111_predict_theirs", "Predict client-x", "predict", "desktop", "ready", who="otherperson",
            via="claude"),
        job("job_20261111_c1_a", "C1 merged", "train", "desktop", "ready", who="tester"),
        job("job_20261111_c1_b", "C1 separate", "train", "desktop", "in_progress", who="tester"),
        job("job_20261111_c1_ok", "C1 recorded", "train", "desktop", "done", who="tester"),
        job("job_20261111_train_np", "Train not pushed", "train", "desktop", "ready", who="tester")])


def live_c1():
    """A desktop queue whose three C1 jobs finished (two pulled) while the registry still says ready / running."""
    fins = [{"job": j, "status": "done", "reason": None, "ended": J.datetime.now().isoformat(timespec="seconds"),
             "run": "r", "pulled": p, "metrics": None} for j, p in (("job_20261111_c1_a", True),
                                                                      ("job_20261111_c1_b", False),
                                                                      ("job_20261111_c1_ok", True))]
    q = {"share": "\\\\gpu-desktop\\cj", "reach": "ok", "detail": None,
         "watcher": {"display": "alive", "state": "running", "detail": None, "job": None, "age_s": 20, "stale": False,
                     "machine": "gpu-desktop"}, "jobs": [], "running": None, "finished": fins}
    return T.live_doc(queues=[q])


class Board(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = T.Env()
        cls.adv_key = A.add_key(cls.env.roster, "adv", "advanced")
        cls.raw, cls.pred = make_roots(cls.env.tmp)
        registry(cls.env.registry)
        cls.env.write_live(live_c1())
        cls.p = T.Portal(cls.env.config(review_root=cls.env.tmp / "review"))
        cls.tok = {"admin": T.login(cls.p.port, cls.env.key), "advanced": T.login(cls.p.port, cls.adv_key)}

    @classmethod
    def tearDownClass(cls):
        cls.p.stop()
        cls.env.cleanup()

    def setUp(self):
        keep = {k: v for k, v in os.environ.items() if not k.startswith("CRYOEV_")}
        patch = mock.patch.dict(os.environ, {**keep, "CJ_RAW_ROOT": str(self.raw), "CJ_PREDICT_ROOT": str(self.pred)},
                                clear=True)
        patch.start()
        self.addCleanup(patch.stop)
        J._reg_cache.clear()
        D._registry_cache.clear()

    def get(self, path, role="admin"):
        return T.request(self.p.port, "GET", path, T.cookie(self.tok[role]))

    def page(self, role="admin") -> str:
        st, _h, body = self.get("/jobs/", role)
        self.assertEqual(st, 200)
        return body.decode("utf-8")

    def col(self, key, role="admin") -> str:
        m = re.search(r'<section class="cj-jobs-sec[^"]*" id="cj-jobs-s-%s"[^>]*>(.*?)</section>' % key,
                      self.page(role), re.S)
        self.assertTrue(m, key)
        return m.group(1)

    # ---- the columns
    def test_columns_from_fixture(self):
        raw, conv, rev = self.col("raw"), self.col("converted"), self.col("review")
        self.assertIn(f"Session {SESS_RAW}", raw)
        self.assertIn("converting", raw)
        self.assertNotIn(SESS_DONE, raw)
        self.assertIn(f"Session {SESS_DONE}", conv)                    # converted, no prediction set for it
        self.assertIn("12 tiles", conv)
        self.assertNotIn(SESS_PRED, conv)                              # it has a set: not "no predictions yet"
        self.assertIn(PKG, rev)                                        # review packages moved up into column 3
        self.assertIn("in review", rev)
        self.assertIn("Review my tiles", rev)                          # labeling / review jobs live here too
        ds = self.col("datasets")
        self.assertIn(DS, ds)
        self.assertIn("used by 2 runs", ds)
        self.assertNotIn("ds_20261031_old", ds)                        # retired: not "ready to train"
        tr = self.col("training")
        self.assertIn("Train not pushed", tr)                          # registry train job, not on a queue
        self.assertNotIn("C1 merged", tr)                              # on the queue's finished list: attention only
        self.assertNotIn("Review my tiles", tr)

    def test_every_card_link_is_href_for_and_works(self):
        page = self.page()
        hrefs = HREF.findall(page)
        self.assertGreaterEqual(len(hrefs), 7)
        want = {J.href_for("conversion", CID_RAW), J.href_for("conversion", CID_DONE), J.href_for("dataset", DS),
                J.href_for("review_package", PKG), J.href_for("job", "x", "revise"), J.href_for("training", "x")}
        self.assertTrue(want <= set(hrefs), want - set(hrefs))
        self.assertEqual(self.col("review").count("<article"), len(HREF.findall(self.col("review"))))
        for h in hrefs:
            self.assertTrue(h.startswith("/") and not h.startswith("//"), h)
        for h in (J.href_for("conversion", CID_DONE), J.href_for("dataset", DS), "/train/", "/jobs/"):
            st, _h, _b = self.get(h)
            self.assertEqual(st, 200, h)                               # routes that exist here; /reviews/ is E's

    def test_href_for_kinds(self):
        self.assertEqual(J.href_for("prediction_set", PS), f"/predict/{PS}/")
        self.assertEqual(J.href_for("conversion", CID_RAW), f"/predict/convert/{CID_RAW}/")
        self.assertEqual(J.href_for("dataset", DS), f"/data/dataset/{DS}")
        self.assertEqual(J.href_for("label_set", LV), f"/data/labels/{LV}")
        self.assertEqual(J.href_for("training", "job_1"), "/train/")
        self.assertEqual(J.href_for("run", "r1"), "/train/")
        self.assertEqual(J.href_for("review_package", PKG), f"/review/{PKG}/app/")
        self.assertEqual(J.href_for("review_package", "lab_compare_x"), "/reviews/")
        self.assertEqual(J.href_for("dataset", "a b/c"), "/data/dataset/a%20b%2Fc")      # quoted
        for bad in (None, "", 5):
            self.assertIsNone(J.href_for("dataset", bad))
        self.assertIsNone(J.href_for("nonsense", "x"))

    def test_conversion_scan_is_cached_across_viewers(self):
        from portal.predict_parts import sets as SETS
        J._scan_cache.clear()
        cfg, calls = self.env.config(), []
        real = SETS.list_sets
        with mock.patch.object(SETS, "list_sets", lambda *a, **k: (calls.append(1), real(*a, **k))[1]):
            a = J.summary(cfg, {"id": "tester", "role": "admin"}, lambda u, o: True)
            b = J.summary(cfg, {"id": "adv", "role": "advanced"}, lambda u, o: o == "adv")
        self.assertEqual(len(calls), 1)                                # one scan serves both viewers
        self.assertTrue(a["flow"]["raw"])
        self.assertEqual(b["flow"]["raw"], [])                         # visibility applied after the cache

    # ---- attribution
    def test_agent_label_and_tag(self):
        rev = self.col("review")
        self.assertIn("by claude for tester", rev)
        self.assertIn('class="cj-jobs-ai"', rev)
        self.assertEqual(J.who_html("pujan", None), "pujan")           # control: a person acting directly, no tag
        self.assertEqual(J.who_html("pujan", "claude"),
                         'claude for pujan <span class="cj-jobs-ai" title="done by an AI agent for this person">AI'
                         "</span>")
        self.assertEqual(J.who_html("<b>x</b>", None), "&lt;b&gt;x&lt;/b&gt;")

    def test_privacy_mask_hides_agent_label_too(self):
        adv = self.page("advanced")
        for text in ("otherperson", "Predict client-x", PKG_OTHER, "job_20261111_predict_theirs"):
            self.assertNotIn(text, adv, text)
        adm = self.page("admin")                                       # control: the admin sees them, labelled
        self.assertIn("claude for otherperson", adm)
        self.assertIn(PKG_OTHER, adm)
        s = json.dumps(J.summary(self.env.config(), {"id": "adv", "role": "advanced"},
                                 lambda u, o: o == u["id"])["flow"])
        self.assertNotIn("otherperson", s)

    def test_hidden_item_has_no_via_no_link(self):
        it = J._item("training", None, J.PRIVATE_NAME, note="Details hidden.", hidden=True)
        html = J.card_html(it)
        self.assertNotIn("<a ", html)
        self.assertNotIn("cj-jobs-ai", html)

    # ---- fragments
    def test_fragment_ids(self):
        page = self.page()
        js = J.JS
        for key in J.SECTIONS:
            elem = "cj-jobs-head" if key == "head" else f"cj-jobs-s-{key}"
            self.assertIn(f'id="{elem}"', page, key)
            self.assertIn(f"{key}: '{elem}'", js, key)                 # the refresh swaps it
        self.assertEqual(set(J.COLUMNS) | {"head", "hardware", "attention", "finished"}, set(J.SECTIONS))
        st, _h, body = self.get("/jobs/api/summary?html=1")
        frags = json.loads(body)["html"]
        self.assertEqual(set(frags), set(J.SECTIONS))
        self.assertIn(f"Session {SESS_RAW}", frags["raw"])             # one fragment per column
        self.assertNotIn(f"Session {SESS_RAW}", frags["converted"])
        self.assertNotIn("innerHTML =", js.replace("t.innerHTML = frag[k]", ""))   # JS builds no HTML from data

    # ---- finished, not recorded
    def test_finished_not_recorded(self):
        s = J.summary(self.env.config(), {"id": "tester", "role": "admin"}, lambda u, o: True)
        un = {a["id"]: a for a in s["attention"] if a["kind"] == "unrecorded"}
        self.assertEqual(sorted(un), ["job_20261111_c1_a", "job_20261111_c1_b"])
        self.assertNotIn("job_20261111_c1_ok", un)                     # control: registry already says done
        self.assertEqual(un["job_20261111_c1_a"]["title"], "Finished, not recorded")
        self.assertIn("results are on the laptop", un["job_20261111_c1_a"]["reason"])
        self.assertNotIn("on the laptop", un["job_20261111_c1_b"]["reason"])   # finished, not pulled
        page = self.page()
        self.assertIn("Finished, not recorded", self.col("attention"))
        self.assertIn("2 finished, not recorded", self.col("training"))
        self.assertIn("not recorded", page)


class Empty(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = T.Env()
        (cls.env.tmp / "raw").mkdir()
        (cls.env.tmp / "predict").mkdir()
        cls.env.write_live(T.live_doc(queues=[]))
        cls.p = T.Portal(cls.env.config(review_root=cls.env.tmp / "review"))
        cls.tok = T.login(cls.p.port, cls.env.key)

    @classmethod
    def tearDownClass(cls):
        cls.p.stop()
        cls.env.cleanup()

    def setUp(self):
        keep = {k: v for k, v in os.environ.items() if not k.startswith("CRYOEV_")}
        patch = mock.patch.dict(os.environ, {**keep, "CJ_RAW_ROOT": str(self.env.tmp / "raw"),
                                             "CJ_PREDICT_ROOT": str(self.env.tmp / "predict")}, clear=True)
        patch.start()
        self.addCleanup(patch.stop)
        J._reg_cache.clear()
        D._registry_cache.clear()

    def test_every_column_is_a_dashed_card_with_a_next_action(self):
        st, _h, body = T.request(self.p.port, "GET", "/jobs/", T.cookie(self.tok))
        page = body.decode("utf-8")
        self.assertEqual(st, 200)
        for key in J.COLUMNS:
            m = re.search(r'<section class="cj-jobs-sec[^"]*" id="cj-jobs-s-%s"[^>]*>(.*?)</section>' % key, page,
                          re.S)
            frag = m.group(1)
            self.assertEqual(frag.count("cj-jobs-dash"), 1, key)
            self.assertEqual(frag.count("<article"), 0, key)
            self.assertIn(f'<a href="{J.EMPTY_NEXT[key][1]}">{J.EMPTY_NEXT[key][0]} &rarr;</a>', frag, key)
        self.assertNotIn("not recorded", page)                         # control: nothing flagged on an empty fixture


if __name__ == "__main__":
    unittest.main()
