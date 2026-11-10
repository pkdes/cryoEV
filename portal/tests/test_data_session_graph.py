"""
Data module, the Graph of a session's and a grid's Lineage page (portal/modules/data.py source_graph, source_links;
/data/api/lineage/graph?focus=session|<id> or grid|<id>&scope=item; /data/data.js).

Known answers on a fixture registry: session S has grids G1 and G2 and one upload conversion C (owner A);
  G1's images -> label version LV (its change log) -> dataset D -> model M -> prediction set P (owner A);
  S's image with no grid -> Roboflow export EXS (its image list); prediction set P2 (owner A) of S, run on C, and its
  package PKG2; another person's prediction set HP of S and their package HPKG listing a G1 image (hidden from A and B);
  G2's images are used by nothing; an unrelated session T with grid GT and its own dataset DT (image list).
  - S's graph has S, G1, G2, LV, D, M, EXS; P, P2, PKG2 and C only for A and the admin; HP / HPKG only for the admin
    (the control); nothing of T. G2's graph is G2 and S, no LV. G1's graph has no EXS, no P2, no G2.
  - every node except a grid's sessions is directed-reachable from the focus over the new edges plus the lab graph's
    (part links included), and the sessions reach the grid;
  - user B (advanced): no trace of P, P2, PKG2, HP, HPKG or C -- B's graph is byte-identical to B's graph on the same
    registry WITHOUT those records (the strongest "not even a count or edge" check), admin as the control;
  - node whitelist, no path or uploader in the JSON; unknown / hidden focus -> empty, focus null, no echo;
  - descendants capped at 150 with `more` and no count; the link index is built once per registry state and group;
  - through the portal: basic 403, the API equals the function, the page has the toggle (Graph by default), no inline
    script; in a browser (Playwright, Edge channel) at 390 and 1280 px, light and dark. Screenshots go to
    $CJ_SHOT_DIR when set.

Temp folders only; never the real registry.

  python -m unittest portal.tests.test_data_session_graph -v
"""
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import quote

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cjtest as T  # noqa: E402
import test_data_graph as TG  # noqa: E402
from portal import auth as A  # noqa: E402
from portal.modules import Ctx  # noqa: E402
from portal.modules import data as D  # noqa: E402

try:
    from playwright.sync_api import sync_playwright
except ImportError:  # pragma: no cover
    sync_playwright = None

S, T_SID = "20991201", "20991202"
G1, G2, GT = "29901", "29902", "29903"
LV, DS, MOD = "lv_20991201_sg", "ds_20991201_sg", "m_20991201_sg"
EXS, DT = "rf_20991201_sgexport", "ds_20991202_other"
P, P2, PKG2 = "ps_20991201_aaaa01", "ps_20991201_aaaa02", "pred_ps_20991201_aaaa02"
HP, HPKG = "ps_20991201_cccc03", "pred_ps_20991201_cccc03"
CID = "20991201_abc123"
PRIVATE = (P, P2, PKG2, HP, HPKG, CID, "aaaa01", "aaaa02", "cccc03", "abc123")
OF_T = (T_SID, GT, DT)


def rec(entity, rid, **kw):
    return {"schema": entity + "/1", "id": rid, "logged": "2099-12-01T09:00:00", **kw}


def ins(*triples):
    return [{"entity": e, "id": i, "role": r} for e, i, r in triples]


def rows(tmp: Path, private=True) -> dict:
    lv_dir = tmp / "labels" / LV
    out = {
        "session": [rec("session", S, date="2099-12-01", status="processed", session_grids={"1": G1, "2": G2},
                        conversions=[{"id": CID, "uploaded_by": "usera", "n_tiles": 9}]),
                    rec("session", T_SID, date="2099-12-02", status="processed", session_grids={"1": GT})],
        "grid": [rec("grid", G1, sample_label="EV prep one"), rec("grid", G2, sample_label="EV prep two"),
                 rec("grid", GT, sample_label="Other prep")],
        "image": [rec("image", "img_s1_a_png", session=S, grid=G1, session_grid="1"),
                  rec("image", "img_s1_b_png", session=S, grid=G1, session_grid="1"),
                  rec("image", "img_s2_a_png", session=S, session_grid="2"),       # grid from the session's slot
                  rec("image", "img_s_x_png", session=S),                          # no grid
                  rec("image", "img_t_a_png", session=T_SID, grid=GT, session_grid="1")],
        "label_version": [rec("label_version", LV, owner=str(lv_dir / "version.json"), status="active")],
        "dataset": [rec("dataset", DS, status="built", inputs=ins(("label_version", LV, "source"))),
                    rec("dataset", DT, status="built", image_list={"file": "lists/dt.tsv"})],
        "model": [rec("model", MOD, status="candidate", inputs=ins(("dataset", DS, "train_data")))],
        "roboflow_export": [rec("roboflow_export", EXS, status="imported", image_list={"file": "lists/exs.tsv"})],
    }
    if private:
        out["prediction_set"] = [
            rec("prediction_set", P, owner="usera", model=MOD, status="done", source={"kind": "upload"},
                inputs=ins(("model", MOD, "model"))),
            rec("prediction_set", P2, owner="usera", model=MOD, status="done",
                source={"kind": "session", "session": S}),
            rec("prediction_set", HP, owner="userc", model=MOD, status="done",
                source={"kind": "session", "session": S})]
        out["review_package"] = [
            rec("review_package", PKG2, kind="prediction", owner="usera", prediction_set=P2, model=MOD, n_images=2,
                status="built"),
            rec("review_package", HPKG, kind="prediction", owner="userc", prediction_set=HP, n_images=1,
                status="built", path=str(tmp / "ext_review" / HPKG))]
    return out


def write_fixture(tmp: Path, reg: Path, private=True):
    """The registry (only these rows), its lists, the label version's change log, the hidden package's data.js, the
    conversion folder and P2's request.json (the conversion it ran on)."""
    reg.mkdir(parents=True, exist_ok=True)
    for ent, recs in rows(tmp, private).items():
        (reg / f"{ent}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    (reg / "lists").mkdir(exist_ok=True)
    (reg / "lists" / "exs.tsv").write_text("img_s_x_png\ttrain\n", encoding="utf-8")
    (reg / "lists" / "dt.tsv").write_text("img_t_a_png\ttrain\n", encoding="utf-8")
    lv_dir = tmp / "labels" / LV
    lv_dir.mkdir(parents=True, exist_ok=True)
    (lv_dir / "version.json").write_text(json.dumps({"n_images_applied": 1}), encoding="utf-8")
    (lv_dir / "changes.jsonl").write_text(json.dumps({"image": "img_s1_a.png"}) + "\n", encoding="utf-8")
    pk = tmp / "ext_review" / HPKG
    pk.mkdir(parents=True, exist_ok=True)
    (pk / "data.js").write_text('window.REVIEW_DATA = {"images": [{"file": "img_s1_b.png"}]};', encoding="utf-8")
    conv = tmp / "raw" / "raw_tiles" / CID
    (conv / "tiles").mkdir(parents=True, exist_ok=True)
    (conv / "convert.json").write_text(json.dumps({"schema": "cj_convert/1", "id": CID, "owner": "usera"}),
                                       encoding="utf-8")
    req = tmp / "CryoAI-laptop" / "predict" / P2 / "request.json"
    req.parent.mkdir(parents=True, exist_ok=True)
    req.write_text(json.dumps({"owner": "usera", "source": {"kind": "session", "session": S,
                                                            "dir": str(conv / "tiles")}}), encoding="utf-8")
    TG.clear_caches()


def cfg_for(tmp: Path):
    return SimpleNamespace(registry=None, raw_root=str(tmp / "raw"),
                           cj_predict_root=str(tmp / "CryoAI-laptop" / "predict"),
                           cj_raw_legacy_list="off")


def titles(g):
    return {n["title"] for n in g["nodes"]}


def reachable(g, start, reverse=False):
    pairs = [(a, b) for a, b, _ in g["edges"]] + [(n["part"], i) for i, n in enumerate(g["nodes"]) if n["part"] >= 0]
    nxt = {}
    for a, b in pairs:
        if reverse:
            a, b = b, a
        nxt.setdefault(a, []).append(b)
    seen, todo = {start}, [start]
    for i in todo:
        for j in nxt.get(i, []):
            if j not in seen:
                seen.add(j)
                todo.append(j)
    return seen


class Function(TG.Roots, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="cj_sgraph_"))
        cls.set_roots(cls.tmp)
        cls.reg = cls.tmp / "registry"
        write_fixture(cls.tmp, cls.reg)
        cls.tmp2 = cls.tmp / "absent"                       # the same registry without the private records
        cls.reg_absent = cls.tmp2 / "registry"
        write_fixture(cls.tmp2, cls.reg_absent, private=False)
        cfg = cfg_for(cls.tmp)
        cls.ADMIN = D.Viewer(Ctx.can_see, {"id": "tester", "role": "admin"}, cfg)
        cls.A = D.Viewer(Ctx.can_see, {"id": "usera", "role": "advanced"}, cfg)
        cls.B = D.Viewer(Ctx.can_see, {"id": "userb", "role": "advanced"}, cfg)

    @classmethod
    def tearDownClass(cls):
        cls.restore_roots()
        shutil.rmtree(cls.tmp, ignore_errors=True)
        TG.clear_caches()

    def graph(self, v, focus, reg=None):
        reg = reg or self.reg
        records, err = D.read_registry(reg)
        self.assertIsNone(err)
        return D.source_graph(D.with_disk_packages(records), v, reg, focus)

    # ------------------------------------------------------------ known answers
    def test_session_graph_per_viewer(self):
        mine = {S, G1, G2, LV, DS, MOD, EXS}
        want = {"admin": mine | {P, P2, PKG2, CID, HP, HPKG}, "A": mine | {P, P2, PKG2, CID}, "B": mine}
        for name, v in (("admin", self.ADMIN), ("A", self.A), ("B", self.B)):
            g = self.graph(v, "session|" + S)
            with self.subTest(viewer=name):
                self.assertEqual(titles(g), want[name])
                self.assertEqual(g["nodes"][g["focus"]]["title"], S)
                self.assertEqual(g["nodes"][g["focus"]]["kind"], "session")
                self.assertTrue(titles(g).isdisjoint(OF_T))
                self.assertEqual(g["scope"], "item")
                self.assertFalse(g["more"])
                self.assertEqual(g["cols"][:len(D.SOURCE_COLS)], list(D.SOURCE_COLS))

    def test_edges_are_the_recorded_links(self):
        g = self.graph(self.A, "session|" + S)
        t = [n["title"] for n in g["nodes"]]
        e = {(t[a], t[b]) for a, b, _ in g["edges"]}
        for pair in ((S, G1), (S, G2), (G1, LV), (LV, DS), (DS, MOD), (MOD, P), (S, EXS), (S, CID), (CID, P2),
                     (CID, PKG2)):
            self.assertIn(pair, e)
        self.assertFalse({p for p in e if p[0] == G2})            # G2's images are used by nothing
        self.assertNotIn((S, P2), e)                              # P2 hangs off the conversion it ran on
        roles = {(t[a], t[b]): r for a, b, r in g["edges"]}
        self.assertEqual(roles[(G1, LV)], "used its images")
        self.assertEqual(roles[(S, CID)], "upload")
        # admin control: the hidden package is linked from G1 (its data.js lists a G1 image), HP from the session
        ga = self.graph(self.ADMIN, "session|" + S)
        ta = [n["title"] for n in ga["nodes"]]
        ea = {(ta[a], ta[b]) for a, b, _ in ga["edges"]}
        self.assertIn((G1, HPKG), ea)
        self.assertIn((S, HP), ea)
        self.assertIn((CID, P2), ea)

    def test_grid_graphs(self):
        g2 = self.graph(self.ADMIN, "grid|" + G2)
        self.assertEqual(titles(g2), {G2, S})
        self.assertEqual(g2["nodes"][g2["focus"]]["title"], G2)
        g1 = self.graph(self.ADMIN, "grid|" + G1)
        # PKG2 is here through the lab graph's own model -> package edge (it names MOD), not through the session
        self.assertEqual(titles(g1), {G1, S, LV, DS, MOD, P, PKG2, HPKG})
        self.assertTrue(titles(g1).isdisjoint({EXS, P2, CID, G2, HP}))           # the session's other branches
        self.assertEqual(titles(self.graph(self.B, "grid|" + G1)), {G1, S, LV, DS, MOD})
        gt = self.graph(self.ADMIN, "session|" + T_SID)
        self.assertEqual(titles(gt), {T_SID, GT, DT})                              # the unrelated session's own

    def test_every_node_is_directed_reachable(self):
        for v in (self.ADMIN, self.A, self.B):
            for focus in ("session|" + S, "grid|" + G1, "grid|" + G2, "session|" + T_SID):
                g = self.graph(v, focus)
                f = g["focus"]
                down = reachable(g, f)
                up = reachable(g, f, reverse=True)
                with self.subTest(focus=focus):
                    self.assertEqual(down | up, set(range(len(g["nodes"]))))
                    self.assertEqual(up - {f}, {i for i, n in enumerate(g["nodes"]) if n["kind"] == "session"
                                                and i != f})        # ancestors: only a grid's sessions
                    for s in up - {f}:
                        self.assertTrue(set(range(len(g["nodes"]))) <= reachable(g, s) | {s})

    def test_lab_part_is_the_lab_graphs_descendants(self):
        # the lab-graph nodes in S's slice are exactly the descendants (lab edges, item_graph) of the first-level items
        g = self.graph(self.ADMIN, "session|" + S)
        lab = {n["title"] for n in g["nodes"] if n["kind"] not in ("session", "grid", "upload")}
        records = D.with_disk_packages(D.read_registry(self.reg)[0])
        union = set()
        for key in ("label_version|" + LV, "roboflow_export|" + EXS, "prediction_set|" + P2,
                    "review_package|" + PKG2, "prediction_set|" + HP, "review_package|" + HPKG):
            item = D.item_graph(D.lab_graph(records, self.ADMIN, self.reg, key))
            f = item["focus"]
            union |= {item["nodes"][i]["title"] for i in reachable(item, f)}
        self.assertEqual(lab, union)

    # ------------------------------------------------------------ visibility
    def test_b_has_no_trace_of_private_items(self):
        for focus in ("session|" + S, "grid|" + G1, "grid|" + G2):
            b = json.dumps(self.graph(self.B, focus))
            with self.subTest(focus=focus):
                for s in PRIVATE + ("usera", "userc"):
                    self.assertNotIn(s, b)
                # byte-identical to B's graph on a registry where the private records never existed
                self.assertEqual(b, json.dumps(self.graph(self.B, focus, self.reg_absent)))
        admin = json.dumps(self.graph(self.ADMIN, "session|" + S))
        for s in (P, P2, PKG2, HP, HPKG, CID):
            self.assertIn(s, admin)                                         # control
        a = json.dumps(self.graph(self.A, "session|" + S))
        for s in (HP, HPKG, "cccc03", "userc"):
            self.assertNotIn(s, a)                                          # A sees only A's own
        self.assertNotEqual(json.dumps(self.graph(self.A, "session|" + S)),
                            json.dumps(self.graph(self.A, "session|" + S, self.reg_absent)))   # the check can fail

    def test_conversion_only_for_owner_and_admin(self):
        for v, has in ((self.ADMIN, True), (self.A, True), (self.B, False)):
            g = self.graph(v, "session|" + S)
            up = [n for n in g["nodes"] if n["kind"] == "upload"]
            with self.subTest(user=v.user["id"]):
                if has:
                    self.assertEqual(up, [{"title": CID, "kind": "upload", "col": 1, "hist": 0, "part": -1,
                                           "sub": "9 tiles", "status": None, "url": f"/predict/convert/{CID}/"}])
                else:
                    self.assertEqual(up, [])
        no_cfg = D.Viewer(Ctx.can_see, {"id": "tester", "role": "admin"})       # no portal cfg: no conversions
        self.assertNotIn(CID, titles(self.graph(no_cfg, "session|" + S)))

    def test_whitelist_and_no_paths_or_people(self):
        for v in (self.ADMIN, self.A, self.B):
            for focus in ("session|" + S, "grid|" + G1):
                g = self.graph(v, focus)
                text = json.dumps(g)
                self.assertEqual(set(g), {"schema", "scope", "cols", "nodes", "edges", "focus", "more", "error"})
                for n in g["nodes"]:
                    self.assertEqual(tuple(n), D.GRAPH_NODE_KEYS)
                for s in (str(self.tmp), str(self.tmp).replace("\\", "\\\\"), "usera", "uploaded_by", "tiles\\",
                          "raw_tiles", "request.json"):
                    self.assertNotIn(s, text)
        node = {n["title"]: n for n in self.graph(self.A, "session|" + S)["nodes"]}
        self.assertEqual(node[S], {"title": S, "kind": "session", "col": 0, "hist": 0, "part": -1,
                                   "sub": "2099-12-01", "status": "processed", "url": f"/data/lineage/session/{S}/"})
        self.assertEqual(node[G1], {"title": G1, "kind": "grid", "col": 1, "hist": 0, "part": -1,
                                    "sub": "EV prep one", "status": None, "url": f"/data/lineage/grid/{G1}/"})
        self.assertEqual(node[LV]["col"], 1 + len(D.SOURCE_COLS))           # lab columns moved right

    def test_unknown_or_bad_focus_is_empty_and_not_echoed(self):
        for focus in ("session|20991299", "grid|99999", "session|", "grid|", "session", "image|img_s1_a_png",
                      None, "session|" + HP):
            for v in (self.ADMIN, self.B):
                g = self.graph(v, focus)
                with self.subTest(focus=focus):
                    self.assertEqual((g["nodes"], g["edges"], g["focus"], g["cols"], g["more"]),
                                     ([], [], None, [], False))
                    if focus and "|" in focus and focus.split("|")[1]:
                        self.assertNotIn(focus.split("|")[1], json.dumps(g))

    def test_focus_keys_and_links(self):
        records = D.read_registry(self.reg)[0]
        self.assertEqual(D.graph_focus_key(records, "session", S), "session|" + S)
        self.assertEqual(D.graph_focus_key(records, "grid", G1), "grid|" + G1)
        self.assertEqual(D.lab_graph_href("session|" + S), "/data/lineage/graph/")      # not a lab graph node
        self.assertEqual(D.lab_graph_href("model|" + MOD), "/data/lineage/graph/?focus=model%7C" + MOD)
        self.assertIsNone(D.source_focus("model|x"))
        self.assertEqual(D.source_focus("grid|" + G1), ("grid", G1))

    def test_links_built_once_per_registry_state_and_group(self):
        TG.clear_caches()
        hpm = D._history_module()
        real, calls = hpm.lookup_index, []
        hpm.lookup_index = lambda *a: (calls.append(1), real(*a))[1]
        try:
            self.graph(self.B, "session|" + S)
            self.graph(self.B, "grid|" + G1)
            self.graph(D.Viewer(Ctx.can_see, {"id": "userd", "role": "advanced"}), "session|" + S)   # same group
            n_build = len(calls)
            self.assertEqual(n_build, 2)       # lab_graph's build_model runs lookup_index once, source_links once
            self.graph(self.ADMIN, "session|" + S)
            self.assertEqual(len(calls), n_build + 2)                       # a new group
            self.graph(self.ADMIN, "grid|" + G2)
            self.assertEqual(len(calls), n_build + 2)
        finally:
            hpm.lookup_index = real

    def test_descendant_cap(self):
        tmp = self.tmp / "big"
        reg = tmp / "registry"
        reg.mkdir(parents=True)
        n = D.ITEM_DOWNSTREAM_MAX + 30
        (reg / "lists").mkdir()
        sess = rec("session", "20991203", status="processed", session_grids={"1": "29904"})
        lines = {"session": [sess], "grid": [rec("grid", "29904")], "image": [], "dataset": []}
        for i in range(n):
            lines["image"].append(rec("image", f"big_{i}_png", session="20991203", grid="29904", session_grid="1"))
            (reg / "lists" / f"d{i}.tsv").write_text(f"big_{i}_png\ttrain\n", encoding="utf-8")
            lines["dataset"].append(rec("dataset", f"ds_big_{i:03d}", status="built",
                                        image_list={"file": f"lists/d{i}.tsv"}))
        for ent, recs in lines.items():
            (reg / f"{ent}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
        g = self.graph(self.ADMIN, "session|20991203", reg)
        self.assertEqual(len(g["nodes"]), 1 + D.ITEM_DOWNSTREAM_MAX)
        self.assertIs(g["more"], True)
        self.assertNotIn(str(n), json.dumps({k: v for k, v in g.items() if k not in ("nodes", "edges")}))

    def test_registry_files_untouched(self):
        before = {f.name: f.read_bytes() for f in self.reg.rglob("*") if f.is_file()}
        for v in (self.ADMIN, self.A, self.B):
            self.graph(v, "session|" + S)
            self.graph(v, "grid|" + G1)
        self.assertEqual({f.name: f.read_bytes() for f in self.reg.rglob("*") if f.is_file()}, before)


# ---------------------------------------------------------------- through the portal
class Served(TG.Roots, unittest.TestCase):
    ENV = ("CJ_RAW_ROOT", "CJ_PREDICT_ROOT", "CJ_RAW_LEGACY_LIST")

    @classmethod
    def setUpClass(cls):
        cls.env = T.Env()
        cls.set_roots(cls.env.tmp)
        cls.saved = {k: os.environ.get(k) for k in cls.ENV}
        os.environ.update({"CJ_RAW_ROOT": str(cls.env.tmp / "raw"), "CJ_RAW_LEGACY_LIST": "off",
                           "CJ_PREDICT_ROOT": str(cls.env.tmp / "CryoAI-laptop" / "predict")})
        cls.keys = {"usera": A.add_key(cls.env.roster, "usera", "advanced"),
                    "userb": A.add_key(cls.env.roster, "userb", "advanced"),
                    "bas": A.add_key(cls.env.roster, "bas", "basic")}
        write_fixture(cls.env.tmp, cls.env.registry)
        cls.env.write_live(T.live_doc())
        cls.p = T.Portal(cls.env.config())
        cls.tok = {"admin": T.login(cls.p.port, cls.env.key)}
        cls.tok.update({u: T.login(cls.p.port, k) for u, k in cls.keys.items()})

    @classmethod
    def tearDownClass(cls):
        cls.p.stop()
        for k, v in cls.saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        cls.restore_roots()
        cls.env.cleanup()
        TG.clear_caches()

    def get(self, path, who="admin"):
        return T.request(self.p.port, "GET", path, T.cookie(self.tok[who]))

    def api(self, focus, who):
        st, h, body = self.get("/data/api/lineage/graph?scope=item&focus=" + quote(focus, safe=""), who)
        return st, h, body


class Http(Served):
    def test_api_per_user(self):
        for who, has in (("usera", True), ("userb", False), ("admin", True)):
            st, h, body = self.api("session|" + S, who)
            g = json.loads(body)
            with self.subTest(who=who):
                self.assertEqual(st, 200)
                self.assertEqual(T.hget(h, "cache-control"), "no-store")
                self.assertTrue({S, G1, G2, LV, DS, MOD, EXS} <= titles(g))
                self.assertEqual(P in titles(g), has)
                self.assertEqual(CID in titles(g), has)
                if not has:
                    for s in PRIVATE + ("usera",):
                        self.assertNotIn(s, body.decode("utf-8"))
        st, _, body = self.api("session|" + S, "bas")
        self.assertEqual(st, 403)
        self.assertNotIn(S.encode(), body)
        # without scope=item a session key is not a lab graph node: the whole lab graph, nothing focused
        whole = json.loads(self.get("/data/api/lineage/graph?focus=" + quote("session|" + S, safe=""), "userb")[2])
        self.assertIsNone(whole["focus"])
        self.assertNotIn("scope", whole)

    def test_api_equals_the_function(self):
        st, _, body = self.api("grid|" + G1, "userb")
        records, _ = D.read_registry(self.env.registry)
        v = D.Viewer(Ctx.can_see, {"id": "userb", "role": "advanced"}, self.env.config())
        self.assertEqual(json.loads(body), D.source_graph(D.with_disk_packages(records), v, self.env.registry,
                                                          "grid|" + G1))

    def test_unknown_focus_no_echo(self):
        for focus in ("session|20991299", "grid|nope_grid_x", "session|" + HP):
            st, _, body = self.api(focus, "userb")
            g = json.loads(body)
            self.assertEqual((st, g["nodes"], g["focus"]), (200, [], None))
            self.assertNotIn(focus.split("|")[1], body.decode("utf-8"))

    def test_lineage_pages_have_the_graph(self):
        for kind, eid in (("session", S), ("grid", G1)):
            page = self.get(f"/data/lineage/{kind}/{eid}/", "userb")[2].decode("utf-8")
            main = page.split('<main class="cj-main">', 1)[1].split("</main>", 1)[0]
            with self.subTest(kind=kind):
                self.assertIn('id="cj-data-view-toggle"', main)
                self.assertIn('data-default="graph"', main)
                self.assertIn(f'data-endpoint="/data/api/lineage/graph?focus={kind}%7C{eid}&amp;scope=item"', main)
                self.assertIn('<script src="/data/data.js" defer></script>', main)
                self.assertNotRegex(main, r"<script(?![^>]*\bsrc=)")
                self.assertNotIn("/data/lineage/graph/?focus=", main)       # the lab graph link focuses nothing
                for s in PRIVATE:
                    self.assertNotIn(s, page)
        item = self.get(f"/data/lineage/model/{MOD}/", "userb")[2].decode("utf-8")
        self.assertIn('id="cj-data-view-toggle"', item)                      # control: an item page, no default
        self.assertNotIn('data-default="graph"', item)
        detail = self.get(f"/data/session/{S}", "userb")[2].decode("utf-8")
        self.assertIn("Lineage: what was made from it", detail)


def launch(p):
    for kw in ({"channel": "msedge"}, {"channel": "chrome"}, {}):
        try:
            return p.chromium.launch(headless=True, **kw)
        except Exception:
            continue
    return None


class Browser(Served):
    @classmethod
    def setUpClass(cls):
        if sync_playwright is None:
            raise unittest.SkipTest("playwright not installed")
        super().setUpClass()
        cls.own_shots = not os.environ.get("CJ_SHOT_DIR")
        cls.shots = Path(os.environ.get("CJ_SHOT_DIR") or tempfile.mkdtemp(prefix="cj_sgshots_"))
        cls.shots.mkdir(parents=True, exist_ok=True)
        cls.pw = sync_playwright().start()
        cls.browser = launch(cls.pw)
        if cls.browser is None:
            cls.pw.stop()
            super().tearDownClass()
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

    def session(self, width, scheme, who):
        ctx = self.browser.new_context(viewport={"width": width, "height": 900}, color_scheme=scheme)
        origin = self.p.origin
        rec_ = {"foreign": [], "errors": [], "console": []}
        ctx.route("**/*", lambda r: (r.continue_() if r.request.url.startswith(origin + "/")
                                     else (rec_["foreign"].append(r.request.url), r.abort())))
        page = ctx.new_page()
        page.on("pageerror", lambda e: rec_["errors"].append(str(e)))
        page.on("console", lambda m: m.type == "error" and rec_["console"].append(m.text))
        page.goto(f"{origin}/login?k={self.keys[who]}")
        return ctx, page, rec_

    def names(self, page):
        return set(page.locator(".cj-data-graph-node .cj-data-graph-id").all_text_contents())

    def test_session_graph_in_a_browser(self):
        url = self.p.origin + f"/data/lineage/session/{S}/"
        for scheme in ("light", "dark"):
            for width in (390, 1280):
                ctx, page, rec_ = self.session(width, scheme, "usera")
                try:
                    with self.subTest(width=width, scheme=scheme):
                        page.goto(url)
                        page.wait_for_selector(".cj-data-graph-node")      # opens on the Graph, no click
                        page.wait_for_load_state("networkidle")
                        self.assertTrue(page.is_visible("#cj-data-graph-panel"))
                        self.assertFalse(page.is_visible("#cj-data-tree-panel"))
                        self.assertTrue(page.is_checked("#cj-data-graph-an"))          # analyses shown
                        self.assertTrue({S, G1, G2, LV, DS, MOD, EXS, CID, P, P2, PKG2} <= self.names(page))
                        self.assertEqual(page.locator(".cj-data-graph-focus").count(), 1)
                        sw, iw = page.evaluate("[document.documentElement.scrollWidth, window.innerWidth]")
                        self.assertLessEqual(sw, iw, "the page scrolls sideways")
                        self.assertEqual(page.locator("script:not([src])").count(), 0)
                        page.screenshot(path=str(self.shots / f"sg_session_{width}_{scheme}.png"), full_page=True)
                        # Tree chosen -> remembered over the default
                        page.click('[data-view="tree"]')
                        page.reload()
                        page.wait_for_load_state("networkidle")
                        self.assertTrue(page.is_visible("#cj-data-tree-panel"))
                    self.assertEqual((rec_["foreign"], rec_["errors"], rec_["console"]), ([], [], []))
                finally:
                    ctx.close()

    def test_grid_and_other_user_in_a_browser(self):
        ctx, page, rec_ = self.session(1280, "light", "userb")
        try:
            page.goto(self.p.origin + f"/data/lineage/session/{S}/")
            page.wait_for_selector(".cj-data-graph-node")
            page.wait_for_load_state("networkidle")
            got = self.names(page)
            self.assertTrue({S, G1, G2, LV, DS, MOD, EXS} <= got)
            self.assertTrue(got.isdisjoint({P, P2, PKG2, CID, HP, HPKG}))
            html_ = page.content()
            for s in PRIVATE:
                self.assertNotIn(s, html_)
            page.screenshot(path=str(self.shots / "sg_session_userb_1280_light.png"), full_page=True)
            page.goto(self.p.origin + f"/data/lineage/grid/{G1}/")
            page.wait_for_selector(".cj-data-graph-node")
            self.assertEqual(self.names(page), {G1, S, LV, DS, MOD})
            page.screenshot(path=str(self.shots / "sg_grid_userb_1280_light.png"), full_page=True)
            self.assertEqual((rec_["foreign"], rec_["errors"], rec_["console"]), ([], [], []))
        finally:
            ctx.close()


if __name__ == "__main__":
    unittest.main()
