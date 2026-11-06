"""
Data module's Lab graph (portal/modules/data.py lab_graph, /data/lineage/graph/, /data/api/lineage/graph,
/data/data.js): History's lineage graph for the whole lab, filtered per viewer BEFORE graph_model runs.

  - known answer: for an admin the nodes and edges equal registry/history_page.py's own graph (same keys, columns,
    faded flags, parts, subtitles, edges);
  - leaks: another person's prediction set, its pred_ package, their Roboflow export, a revise job named after the
    hidden set and a job that only outputs to the hidden package never reach an advanced viewer (JSON and page HTML,
    string search), while the admin control sees every one;
  - fade-out: an export whose only child is a hidden set is faded exactly as if the child were not in the registry;
  - whitelist: every node has exactly GRAPH_NODE_KEYS;
  - basic: 403 on every graph path, and the function-level filter for a basic viewer;
  - ?focus: a visible key gives its index, a hidden one gives nothing and is not echoed;
  - the page in a browser (Playwright, Edge channel): 390 and 1280 px, light and dark, the graph scrolls inside its
    box and never the page, a box opens its Lineage page, the focused box is highlighted, no inline script.
    Screenshots go to $CJ_SHOT_DIR when set.

Temp registry from registry/samples plus fixture rows (cjtest.Env); never the real one.

  python -m unittest discover -s portal/tests -v
"""
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from urllib.parse import quote

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cjtest as T  # noqa: E402
from portal import auth as A  # noqa: E402
from portal.modules import Ctx  # noqa: E402
from portal.modules import data as D  # noqa: E402

sys.path.insert(0, str(T.REPO / "registry"))
import history_page as hp  # noqa: E402

try:
    from playwright.sync_api import sync_playwright
except ImportError:  # pragma: no cover
    sync_playwright = None

SAMPLES = T.REPO / "registry" / "samples"
MODEL = "m_graph_20261106"
LONELY = "rf_lonely_20261106"                       # visible export whose only child is the hidden set
HID_PS = "ps_20261106_aaaa11"                       # another person's prediction set
HID_PKG = "pred_ps_20261106_aaaa11"                 # ... its prediction package
HID_RF = "rf_theirs_20261106"                       # ... their Roboflow export
HID_JOB = "job_20261106_revise_ps_20261106_aaaa11"  # revise job named after the hidden set (inputs all visible)
OUT_JOB = "job_20261106_annotate_out"               # names the hidden package only in its outputs
LV_HID = "lv_20261106_fromhidden"                   # label version built from the hidden package: stays
OWN_PS, OWN_PKG = "ps_20261106_bbbb22", "pred_ps_20261106_bbbb22"   # the advanced user's own
OWN_PART = "own_part1_20261106"                     # a lab part of the user's own prediction package
LAB_PART = "lab_part1_20261106"                     # a lab part (kind part) of the HIDDEN prediction package
NOTED = "rf_noted_20261106"                         # visible export whose note / status quote the hidden set
SHORT_RF, SHORT_RF2, SHORT_DS = "rf_x", "rf_x_v2", "ds_from_rf_x_v2"   # QC: another's short-id export vs the lab's
SHORT_JOB_OK, SHORT_JOB_HID = "job_build_rf_x_v2", "job_from_rf_x"       # ... a job on rf_x_v2 / one naming rf_x
LATE_RF = "rf_owner_later_20261106"                 # first line owned by another, last line has no owner
LATE_PS = "ps_20261106_cccc33"                      # adv's, but the last line has no owner (and no request.json)
HISTORY_ONLY = "rf_history_only_20261106"           # in a file History reads and Data does not
HIDDEN_STRINGS = (HID_PS, HID_PKG, HID_RF, HID_JOB, OUT_JOB, "aaaa11", LAB_PART, SHORT_JOB_HID, '"rf_x"', LATE_PS,
                  HISTORY_ONLY)
SAMPLE_PS = "ps_20261105_abc123"                    # the samples' prediction set (another owner, so hidden too)


def fixture_rows(with_hidden_ps=True) -> dict:
    ds = "roboflow_20261028_cryoai_v5_S1_evnonev"
    rows = {
        "model": [{"schema": "model/1", "id": MODEL, "logged": "2026-11-06T09:00:00", "status": "candidate",
                   "inputs": [{"entity": "dataset", "id": ds, "role": "train_data"}]}],
        "roboflow_export": [
            {"schema": "roboflow_export/1", "id": LONELY, "logged": "2026-11-06T09:00:00", "status": "imported",
             "note": "a lonely export"},
            {"schema": "roboflow_export/1", "id": HID_RF, "logged": "2026-11-06T09:00:00", "owner": "otherperson",
             "status": "imported", "note": "their export"},
            {"schema": "roboflow_export/1", "id": NOTED, "logged": "2026-11-06T09:00:00",
             "status": "waiting_on_" + HID_PS, "note": "copied from " + HID_PS},
            {"schema": "roboflow_export/1", "id": SHORT_RF, "logged": "2026-11-06T09:00:00", "owner": "otherperson",
             "status": "imported"},
            {"schema": "roboflow_export/1", "id": SHORT_RF2, "logged": "2026-11-06T09:00:00", "status": "imported",
             "derived_from": SHORT_RF},
            {"schema": "roboflow_export/1", "id": LATE_RF, "logged": "2026-11-06T09:00:00", "owner": "otherperson",
             "status": "imported"},
            {"schema": "roboflow_export/1", "id": LATE_RF, "logged": "2026-11-06T09:30:00", "status": "imported"}],
        "dataset": [{"schema": "dataset/1", "id": SHORT_DS, "logged": "2026-11-06T09:10:00", "status": "built",
                     "inputs": [{"entity": "roboflow_export", "id": SHORT_RF2, "role": "source"}]}],
        "prediction_set": [
            {"schema": "prediction_set/1", "id": OWN_PS, "logged": "2026-11-06T10:00:00", "owner": "adv",
             "model": MODEL, "status": "done", "inputs": [{"entity": "model", "id": MODEL, "role": "model"}]},
            {"schema": "prediction_set/1", "id": LATE_PS, "logged": "2026-11-06T10:00:00", "owner": "adv",
             "model": MODEL, "status": "running"},
            {"schema": "prediction_set/1", "id": LATE_PS, "logged": "2026-11-06T10:30:00", "status": "done"}],
        "review_package": [
            {"schema": "review_package/1", "id": HID_PKG, "logged": "2026-11-06T10:00:00", "kind": "prediction",
             "owner": "otherperson", "prediction_set": HID_PS, "model": MODEL, "n_images": 4, "status": "in_review"},
            {"schema": "review_package/1", "id": OWN_PKG, "logged": "2026-11-06T10:00:00", "kind": "prediction",
             "owner": "adv", "prediction_set": OWN_PS, "model": MODEL, "n_images": 2, "status": "in_review"},
            {"schema": "review_package/1", "id": LAB_PART, "logged": "2026-11-06T10:05:00", "kind": "part",
             "parent": HID_PKG, "n_images": 2, "status": "in_review"},
            {"schema": "review_package/1", "id": OWN_PART, "logged": "2026-11-06T10:05:00", "kind": "part",
             "parent": OWN_PKG, "n_images": 1, "status": "in_review"}],
        "job": [
            {"schema": "job/1", "id": HID_JOB, "logged": "2026-11-06T11:00:00", "type": "revise", "status": "ready",
             "inputs": [{"entity": "model", "id": MODEL, "role": "predictions"}], "outputs": []},
            {"schema": "job/1", "id": OUT_JOB, "logged": "2026-11-06T11:00:00", "type": "annotate",
             "status": "ready", "inputs": [], "outputs": [{"entity": "review_package", "id": HID_PKG}]},
            {"schema": "job/1", "id": SHORT_JOB_OK, "logged": "2026-11-06T11:00:00", "type": "annotate",
             "status": "ready", "inputs": [{"entity": "roboflow_export", "id": SHORT_RF2, "role": "base"}]},
            {"schema": "job/1", "id": SHORT_JOB_HID, "logged": "2026-11-06T11:00:00", "type": "annotate",
             "status": "ready", "inputs": [{"entity": "roboflow_export", "id": SHORT_RF, "role": "base"}]}],
        "label_version": [
            {"schema": "label_version/1", "id": LV_HID, "logged": "2026-11-06T12:00:00", "pkg": HID_PKG,
             "status": "active", "n_images_applied": 3,
             "inputs": [{"entity": "review_package", "id": HID_PKG, "role": "edits"},
                        {"entity": "model", "id": MODEL, "role": "checked against " + HID_PS}]}]}
    if with_hidden_ps:
        rows["prediction_set"].append(
            {"schema": "prediction_set/1", "id": HID_PS, "logged": "2026-11-06T10:00:00", "owner": "otherperson",
             "model": MODEL, "status": "done",
             "inputs": [{"entity": "roboflow_export", "id": LONELY, "role": "source"},
                        {"entity": "model", "id": MODEL, "role": "model"}]})
    return rows


def write_registry(root: Path, with_hidden_ps=True):
    root.mkdir(parents=True, exist_ok=True)
    extra = fixture_rows(with_hidden_ps)
    for f in SAMPLES.glob("*.jsonl"):
        lines = f.read_text(encoding="utf-8").splitlines()
        lines += [json.dumps(r) for r in extra.get(f.stem, [])]
        (root / f.name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    for ent, recs in extra.items():
        if not (root / f"{ent}.jsonl").exists():
            (root / f"{ent}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    (root / "zz_extra.jsonl").write_text(json.dumps(       # History reads every *.jsonl; Data only <entity>.jsonl
        {"schema": "roboflow_export/1", "id": HISTORY_ONLY, "logged": "2026-11-06T09:00:00", "status": "imported"})
        + "\n", encoding="utf-8")
    clear_caches()


def write_disk(laptop: Path):
    """The user's own prediction set's request.json: its owner, which a part of its package inherits."""
    req = laptop / "predict" / OWN_PS / "request.json"
    req.parent.mkdir(parents=True, exist_ok=True)
    req.write_text(json.dumps({"owner": "adv"}), encoding="utf-8")


def clear_caches():
    for c in (D._registry_cache, D._files_cache, D._package_cache, D._owner_cache, D._graph_cache):
        c.clear()


def viewer(uid, role):
    return D.Viewer(Ctx.can_see, {"id": uid, "role": role})


ADMIN, ADV, BASIC = viewer("tester", "admin"), viewer("adv", "advanced"), viewer("bas", "basic")
REVERSE_SHORT = {v: k for k, v in D.GRAPH_SHORT.items()}


def node_key(n):
    return REVERSE_SHORT.get(n["kind"], n["kind"]) + "|" + n["title"]


def by_title(g):
    return {n["title"]: n for n in g["nodes"]}


class Roots:
    """Temp data roots, so nothing reads the real laptop / processing folders."""

    @classmethod
    def set_roots(cls, tmp: Path):
        cls.saved_env = {k: os.environ.get(k) for k in ("CRYOEV_LAPTOP_ROOT", "CRYOEV_PROCESSING_ROOT")}
        for k, name in (("CRYOEV_LAPTOP_ROOT", "CryoAI-laptop"), ("CRYOEV_PROCESSING_ROOT", "Cryo-image-processing")):
            (tmp / name).mkdir(parents=True, exist_ok=True)
            os.environ[k] = str(tmp / name)

    @classmethod
    def restore_roots(cls):
        for k, v in cls.saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ---------------------------------------------------------------- function level
class Function(Roots, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="cj_graph_"))
        cls.set_roots(cls.tmp)
        cls.reg = cls.tmp / "registry"
        cls.reg_absent = cls.tmp / "registry_no_child"
        write_registry(cls.reg)
        write_registry(cls.reg_absent, with_hidden_ps=False)
        write_disk(cls.tmp / "CryoAI-laptop")

    @classmethod
    def tearDownClass(cls):
        cls.restore_roots()
        shutil.rmtree(cls.tmp, ignore_errors=True)
        clear_caches()

    def graph(self, v, reg=None, focus=None):
        reg = reg or self.reg
        records, error = D.read_registry(reg)
        self.assertIsNone(error)
        return D.lab_graph(D.with_disk_packages(records), v, reg, focus)

    def test_admin_equals_history_graph(self):
        changes = []
        cur, his, st = hp.read_registry(hp.input_files([self.reg]), changes)
        ref = hp.build_model(cur, his, st, [str(self.reg)], changes)["graph"]
        g = self.graph(ADMIN)
        self.assertEqual(g["cols"], ref["cols"])
        self.assertEqual([[node_key(n), n["col"], n["hist"], n["part"], n["sub"]] for n in g["nodes"]], ref["nodes"])
        self.assertEqual(g["edges"], ref["edges"])
        self.assertGreater(len(ref["edges"]), 5)                 # the fixture is not trivially empty
        keys = {node_key(n) for n in g["nodes"]}
        for k in ("prediction_set|" + HID_PS, "review_package|" + HID_PKG, "roboflow_export|" + HID_RF,
                  "job|" + HID_JOB, "job|" + OUT_JOB, "label_version|" + LV_HID):
            self.assertIn(k, keys)                              # control: the admin sees every one

    def test_advanced_never_sees_hidden_records(self):
        g = self.graph(ADV)
        text = json.dumps(g)
        for s in HIDDEN_STRINGS + (SAMPLE_PS, "otherperson", "their export"):
            self.assertNotIn(s, text)
        admin_text = json.dumps(self.graph(ADMIN))
        for s in HIDDEN_STRINGS + (SAMPLE_PS,):
            self.assertIn(s, admin_text)                        # control
        t = by_title(g)
        self.assertIn(OWN_PS, t)                                # their own analysis stays
        self.assertIn(OWN_PKG, t)
        self.assertIn(LV_HID, t)                                # built from a hidden set: the node stays ...
        i = list(t).index(LV_HID)
        im = list(t).index(MODEL)                               # ... its edge from the hidden package goes; the
        self.assertEqual([e for e in g["edges"] if e[1] == i], [[im, i, "input"]])     # model's stays (role blanked)
        ia = [node_key(n) for n in self.graph(ADMIN)["nodes"]].index("label_version|" + LV_HID)
        self.assertTrue([e for e in self.graph(ADMIN)["edges"] if e[1] == ia])     # control: the admin has the edge

    def test_hidden_set_does_not_shape_the_fade(self):
        adv = by_title(self.graph(ADV))[LONELY]
        absent = by_title(self.graph(ADV, self.reg_absent))[LONELY]
        self.assertEqual(adv, absent)
        self.assertEqual(adv["hist"], 1)                        # nothing visible was built from it
        self.assertEqual(by_title(self.graph(ADMIN))[LONELY]["hist"], 0)      # control: the admin sees its child
        self.assertEqual(by_title(self.graph(ADMIN, self.reg_absent))[LONELY]["hist"], 1)

    def test_node_whitelist(self):
        for v in (ADMIN, ADV, BASIC):
            g = self.graph(v)
            self.assertEqual(set(g), {"schema", "cols", "nodes", "edges", "focus", "error"})
            for n in g["nodes"]:
                self.assertEqual(tuple(n), D.GRAPH_NODE_KEYS)
            for e in g["edges"]:
                self.assertEqual((len(e), type(e[0]), type(e[1]), type(e[2])), (3, int, int, str))

    def test_urls_lead_to_lineage_pages(self):
        t = by_title(self.graph(ADMIN))
        self.assertEqual(t[MODEL]["url"], "/data/lineage/model/m_graph_20261106/")
        self.assertEqual(t[LV_HID]["url"], "/data/lineage/labels/lv_20261106_fromhidden/")
        self.assertEqual(t[LONELY]["url"], "/data/lineage/images/rf_lonely_20261106/")
        self.assertEqual(t[HID_PKG]["url"], "/data/lineage/images/pred_ps_20261106_aaaa11/")
        self.assertIsNone(t[HID_JOB]["url"])                    # jobs and pulls have no Data page
        self.assertIsNone(t["roboflow_pull_20261029_1811"]["url"])

    def test_basic_viewer_filter(self):
        g = self.graph(BASIC)
        text = json.dumps(g)
        for s in HIDDEN_STRINGS + (OWN_PS, OWN_PKG, SAMPLE_PS):
            self.assertNotIn(s, text)
        self.assertIn(LV_HID, by_title(g))
        own = self.graph(viewer("adv", "basic"))                # a basic user still sees their own analyses
        self.assertIn(OWN_PS, by_title(own))
        self.assertNotIn(HID_PS, json.dumps(own))

    def test_focus(self):
        g = self.graph(ADV, focus="model|" + MODEL)
        self.assertEqual(g["nodes"][g["focus"]]["title"], MODEL)
        for key in ("prediction_set|" + HID_PS, "review_package|" + HID_PKG, "job|" + HID_JOB, "model|nope"):
            g = self.graph(ADV, focus=key)
            self.assertIsNone(g["focus"])
            self.assertNotIn("aaaa11", json.dumps(g))
        self.assertIsNotNone(self.graph(ADMIN, focus="prediction_set|" + HID_PS)["focus"])   # control

    def test_missing_registry(self):
        g = D.lab_graph(D.read_registry(None)[0], ADMIN, None)
        self.assertEqual((g["nodes"], g["edges"]), ([], []))
        self.assertTrue(g["error"])
        g = D.lab_graph(D.read_registry(None)[0], ADMIN, self.tmp / "nope")
        self.assertEqual(g["error"], "registry folder not found")

    def in_tree(self, v, kind, eid, reg=None):
        records, _ = D.read_registry(reg or self.reg)
        return D.lineage_tree(D.with_disk_packages(records), v, kind, eid) is not None

    def test_short_hidden_id_does_not_hide_the_labs_records(self):
        g = self.graph(ADV)
        t = by_title(g)
        for eid in (SHORT_RF2, SHORT_DS, SHORT_JOB_OK):
            self.assertIn(eid, t)                               # the lab's rf_x_v2 and what was built from it
        self.assertNotIn(SHORT_RF, t)                           # another person's rf_x
        self.assertNotIn(SHORT_JOB_HID, t)                      # a job naming rf_x (exact value, a whole word)
        self.assertTrue(self.in_tree(ADV, "images", SHORT_RF2))         # graph and tree agree
        self.assertTrue(self.in_tree(ADV, "dataset", SHORT_DS))
        self.assertFalse(self.in_tree(ADV, "images", SHORT_RF))
        i2 = list(t).index(SHORT_RF2)
        self.assertEqual([e for e in g["edges"] if e[1] == i2], [])        # its derived_from edge to rf_x goes
        a = by_title(self.graph(ADMIN))
        for eid in (SHORT_RF, SHORT_RF2, SHORT_DS, SHORT_JOB_OK, SHORT_JOB_HID):
            self.assertIn(eid, a)                               # control
        quotes = D.hidden_matcher({"rf_x", "", "ps_20261106_aaaa11"})
        self.assertEqual([quotes(s) for s in ("rf_x", "from rf_x.", "rf_x_v2", "xrf_x", "", "pred_ps_20261106_aaaa11")],
                         [True, True, False, False, False, True])

    def test_visibility_follows_datas_last_line(self):
        g = by_title(self.graph(ADV))
        self.assertIn(LATE_RF, g)                               # last line has no owner: visible, as in the tree
        self.assertTrue(self.in_tree(ADV, "images", LATE_RF))
        self.assertNotIn(LATE_PS, g)                            # a private set with no known owner: admins only
        self.assertFalse(self.in_tree(ADV, "images", LATE_PS))
        self.assertNotIn(HISTORY_ONLY, g)                       # Data has no record: unknown owner, admins only
        a = by_title(self.graph(ADMIN))
        self.assertIn(LATE_PS, a)
        self.assertIn(HISTORY_ONLY, a)                          # admin parity with History

    def test_part_of_a_hidden_prediction_package(self):
        g = by_title(self.graph(ADV))
        self.assertNotIn(LAB_PART, g)
        self.assertFalse(self.in_tree(ADV, "images", LAB_PART))
        self.assertIn(OWN_PART, g)                              # a part of their own package (owner via request.json)
        self.assertTrue(self.in_tree(ADV, "images", OWN_PART))
        self.assertIn(LAB_PART, by_title(self.graph(ADMIN)))    # control
        self.assertTrue(self.in_tree(ADMIN, "images", LAB_PART))

    def test_note_status_and_role_quoting_a_hidden_id_are_blanked(self):
        n = by_title(self.graph(ADV))[NOTED]
        self.assertEqual((n["sub"], n["status"]), ("", ""))
        a = self.graph(ADMIN)
        self.assertEqual(by_title(a)[NOTED]["sub"], "copied from " + HID_PS)      # control: the admin sees the note
        self.assertIn("checked against " + HID_PS, [e[2] for e in a["edges"]])
        self.assertIn("input", [e[2] for e in self.graph(ADV)["edges"]])

    def test_graph_built_once_per_registry_state_and_group(self):
        clear_caches()
        hpm = D._history_module()
        real, calls = hpm.build_model, []
        hpm.build_model = lambda *a: (calls.append(1), real(*a))[1]
        try:
            self.graph(ADV)
            self.graph(ADV)
            self.graph(viewer("adv", "basic"))                  # same hidden records: same group
            self.assertEqual(len(calls), 1)
            self.graph(ADMIN)
            self.assertEqual(len(calls), 2)
            f = self.reg / "model.jsonl"
            saved = f.read_bytes()
            try:
                with open(f, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"schema": "model/1", "id": "m_late", "logged": "2026-11-06T13:00:00"}) + "\n")
                D._registry_cache.clear()
                self.assertIn("m_late", by_title(self.graph(ADMIN)))
                self.assertEqual(len(calls), 3)
            finally:
                f.write_bytes(saved)
                clear_caches()
        finally:
            hpm.build_model = real

    def test_registry_files_untouched(self):
        before = {f.name: f.read_bytes() for f in self.reg.iterdir()}
        self.graph(ADMIN)
        self.graph(ADV)
        self.assertEqual({f.name: f.read_bytes() for f in self.reg.iterdir()}, before)


# ---------------------------------------------------------------- through the portal
class Base(Roots, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = T.Env()
        cls.set_roots(cls.env.tmp)
        cls.adv_key = A.add_key(cls.env.roster, "adv", "advanced")
        cls.bas_key = A.add_key(cls.env.roster, "bas", "basic")
        write_registry(cls.env.registry)
        write_disk(cls.env.tmp / "CryoAI-laptop")
        cls.env.write_live(T.live_doc())
        cls.p = T.Portal(cls.env.config())
        cls.tok = {"admin": T.login(cls.p.port, cls.env.key), "advanced": T.login(cls.p.port, cls.adv_key),
                   "basic": T.login(cls.p.port, cls.bas_key)}

    @classmethod
    def tearDownClass(cls):
        cls.p.stop()
        cls.restore_roots()
        cls.env.cleanup()
        clear_caches()

    def get(self, path, role="admin"):
        return T.request(self.p.port, "GET", path, T.cookie(self.tok[role]))


class Http(Base):
    def test_basic_is_403(self):
        for path in ("/data/lineage/graph/", "/data/lineage/graph", "/data/api/lineage/graph",
                     "/data/api/lineage/graph?focus=model%7C" + MODEL, "/data/data.js"):
            with self.subTest(path=path):
                st, _, body = self.get(path, "basic")
                self.assertEqual(st, 403)
                self.assertNotIn(MODEL.encode(), body)
                self.assertEqual(self.get(path, "advanced")[0], 200)       # control
        self.assertEqual(T.request(self.p.port, "GET", "/data/api/lineage/graph")[0], 401)

    def test_advanced_json_and_page_do_not_leak(self):
        for path in ("/data/api/lineage/graph", "/data/lineage/graph/",
                     "/data/api/lineage/graph?focus=" + quote("prediction_set|" + HID_PS, safe=""),
                     "/data/lineage/graph/?focus=" + quote("review_package|" + HID_PKG, safe="")):
            with self.subTest(path=path):
                st, _, body = self.get(path, "advanced")
                self.assertEqual(st, 200)
                text = body.decode("utf-8")
                for s in HIDDEN_STRINGS + (SAMPLE_PS, "otherperson"):
                    self.assertNotIn(s, text)
        st, h, body = self.get("/data/api/lineage/graph?focus=" + quote("prediction_set|" + HID_PS, safe=""),
                               "advanced")
        self.assertEqual(T.hget(h, "cache-control"), "no-store")
        g = json.loads(body)
        self.assertIsNone(g["focus"])
        self.assertIn(OWN_PS, body.decode("utf-8"))
        admin = self.get("/data/api/lineage/graph", "admin")[2].decode("utf-8")
        for s in HIDDEN_STRINGS:
            self.assertIn(s, admin)                             # control

    def test_page_frame_and_script(self):
        page = self.get("/data/lineage/graph/")[2].decode("utf-8")
        main = page.split('<main class="cj-main">', 1)[1].split("</main>", 1)[0]
        self.assertIn('<script src="/data/data.js" defer></script>', main)
        self.assertNotIn("<script>", main)
        self.assertIn('id="cj-data-graph-canvas"', main)
        self.assertNotIn(MODEL, main)                           # no record text in the HTML: the script loads it
        st, h, js = self.get("/data/data.js")
        self.assertEqual(st, 200)
        self.assertTrue(T.hget(h, "content-type").startswith("text/javascript"))
        self.assertIn(b"cj-data-graph-canvas", js)
        css = self.get("/cj/cj.css")[2].decode("utf-8")
        self.assertIn(".cj-data-graph-node{", css)

    def test_links_from_landing_and_lineage(self):
        self.assertIn('href="/data/lineage/graph/"', self.get("/data/")[2].decode("utf-8"))
        page = self.get("/data/lineage/model/" + MODEL + "/", "advanced")[2].decode("utf-8")
        self.assertIn('href="/data/lineage/graph/?focus=model%7C' + MODEL + '"', page)
        page = self.get("/data/lineage/images/" + OWN_PKG + "/", "advanced")[2].decode("utf-8")
        self.assertIn('href="/data/lineage/graph/?focus=review_package%7C' + OWN_PKG + '"', page)

    def test_api_equals_function(self):
        g = json.loads(self.get("/data/api/lineage/graph?focus=model%7C" + MODEL, "advanced")[2])
        records, _ = D.read_registry(self.env.registry)
        self.assertEqual(g, D.lab_graph(D.with_disk_packages(records), ADV, self.env.registry, "model|" + MODEL))
        self.assertEqual(g["nodes"][g["focus"]]["title"], MODEL)


def launch(p):
    for kw in ({"channel": "msedge"}, {"channel": "chrome"}, {}):
        try:
            return p.chromium.launch(headless=True, **kw)
        except Exception:
            continue
    return None


class Browser(Base):
    @classmethod
    def setUpClass(cls):
        if sync_playwright is None:
            raise unittest.SkipTest("playwright not installed")
        super().setUpClass()
        cls.own_shots = not os.environ.get("CJ_SHOT_DIR")
        cls.shots = Path(os.environ.get("CJ_SHOT_DIR") or tempfile.mkdtemp(prefix="cj_shots_"))
        cls.shots.mkdir(parents=True, exist_ok=True)
        cls.pw = sync_playwright().start()
        cls.browser = launch(cls.pw)
        if cls.browser is None:
            cls.tearDownClass()
            raise unittest.SkipTest("no Edge / Chrome for Playwright")

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "browser", None):
            cls.browser.close()
        if getattr(cls, "pw", None):
            cls.pw.stop()
        super().tearDownClass()
        if cls.own_shots:
            shutil.rmtree(cls.shots, ignore_errors=True)

    def session(self, width, scheme, key):
        ctx = self.browser.new_context(viewport={"width": width, "height": 900}, color_scheme=scheme)
        origin = self.p.origin
        rec = {"foreign": [], "errors": [], "console": []}
        ctx.route("**/*", lambda r: (r.continue_() if r.request.url.startswith(origin + "/")
                                     else (rec["foreign"].append(r.request.url), r.abort())))
        page = ctx.new_page()
        page.on("pageerror", lambda e: rec["errors"].append(str(e)))
        page.on("console", lambda m: m.type == "error" and rec["console"].append(m.text))
        page.goto(f"{origin}/login?k={key}")
        return ctx, page, rec

    def open_graph(self, page, query=""):
        page.goto(self.p.origin + "/data/lineage/graph/" + query)
        page.wait_for_selector(".cj-data-graph-node")
        page.wait_for_load_state("networkidle")

    def test_graph_at_two_widths_and_schemes(self):
        for who, key in (("admin", self.env.key), ("advanced", self.adv_key)):
            for scheme in ("light", "dark"):
                for width in (390, 1280):
                    ctx, page, rec = self.session(width, scheme, key)
                    try:
                        with self.subTest(who=who, width=width, scheme=scheme):
                            self.open_graph(page)
                            sw, iw = page.evaluate("[document.documentElement.scrollWidth, window.innerWidth]")
                            self.assertLessEqual(sw, iw, "the page scrolls sideways")
                            wrap = page.evaluate("(w => [w.scrollWidth, w.clientWidth])"
                                                 "(document.getElementById('cj-data-graph-wrap'))")
                            if width == 390:
                                self.assertGreater(wrap[0], wrap[1], "the graph box should scroll inside")
                            self.assertEqual(page.locator("script:not([src])").count(), 0)
                            self.assertGreater(page.locator("svg#cj-data-graph-svg path").count(), 4)
                            # analyses start hidden; the checkbox shows them
                            own = page.locator(f'.cj-data-graph-node[title="predictions: {OWN_PS}"]')
                            self.assertEqual(own.count(), 0)
                            page.check("#cj-data-graph-an")
                            page.check("#cj-data-graph-hist")
                            self.assertEqual(own.count(), 1)
                            text = page.content()
                            if who == "admin":
                                for s in (HID_PS, HID_PKG, HID_RF, HID_JOB, OUT_JOB):
                                    self.assertIn(s, text)              # control: all drawn for the admin
                            else:
                                for s in HIDDEN_STRINGS:
                                    self.assertNotIn(s, text)
                            page.uncheck("#cj-data-graph-an")
                            page.uncheck("#cj-data-graph-hist")
                            page.screenshot(path=str(self.shots / f"graph_{who}_{width}_{scheme}.png"),
                                            full_page=True)
                        self.assertEqual((rec["foreign"], rec["errors"], rec["console"]), ([], [], []))
                    finally:
                        ctx.close()

    def test_click_opens_lineage_and_focus_highlights(self):
        ctx, page, rec = self.session(1280, "light", self.adv_key)
        try:
            self.open_graph(page, "?focus=" + quote("prediction_set|" + OWN_PS, safe=""))
            focused = page.locator(".cj-data-graph-focus")
            self.assertEqual(focused.count(), 1)
            self.assertIn(OWN_PS, focused.text_content())
            self.assertTrue(page.is_checked("#cj-data-graph-an"))         # its column was revealed for it
            page.screenshot(path=str(self.shots / "graph_advanced_focus.png"), full_page=True)
            self.open_graph(page, "?focus=" + quote("prediction_set|" + HID_PS, safe=""))
            self.assertEqual(page.locator(".cj-data-graph-focus").count(), 0)
            self.assertNotIn("aaaa11", page.content())
            page.click(f'a.cj-data-graph-node[title$="{MODEL}"]')
            page.wait_for_load_state("networkidle")
            self.assertTrue(page.url.endswith("/data/lineage/model/" + MODEL + "/"), page.url)
            self.assertIn("Lineage of " + MODEL, page.text_content("main"))
            page.click('main a[href^="/data/lineage/graph/"]')
            page.wait_for_selector(".cj-data-graph-focus")
            self.assertIn(MODEL, page.locator(".cj-data-graph-focus").text_content())
            self.assertEqual((rec["foreign"], rec["errors"], rec["console"]), ([], [], []))
        finally:
            ctx.close()


if __name__ == "__main__":
    unittest.main()
