"""
Data module, the Graph of a session's and a grid's Lineage page (portal/modules/data.py source_graph, source_links;
/data/api/lineage/graph?focus=session|<id> or grid|<id>&scope=item; /data/data.js).

Known answers on a fixture registry: session S has grids G1, G2 and GS (GS is shared with session V) and two upload
conversions, C (owner A) and C2 (owner C);
  G1's images -> label version LV (its change log) -> dataset D -> model M -> prediction set P (owner A, an analysis
  OF S); M also made prediction set Q (owner A) and its package QPKG for an unrelated session U;
  S's image with no grid -> Roboflow export EXS (its image list); S's image on a grid the registry lacks -> dataset DU;
  prediction set P2 (owner A) of S, run on C, and its package PKG2; P3 (owner A) of S, run on C2 (A can't see C2);
  another person's prediction set HP of S, their package HPKG listing a G1 image, and a revise job JOBH on HPKG whose
  list names a G1 image (all hidden from A and B); G2's images are used by nothing; V's image on GS -> dataset DV;
  an unrelated session T whose only grid GT (its sample label quotes HP's id) has dataset DT and analysis PT (owner A).
  - S's graph: S, G1, G2, GS, LV, D, M, EXS, DU; P, P2, PKG2, P3 and C for A and the admin; C2, HP, HPKG, JOBH only for
    the admin (the control); never Q / QPKG (another session's analyses through M), nothing of T, U or V.
  - U's graph has Q and QPKG; a grid of a one-grid session (GT) shows the session's analysis PT; G1 (S has 3 grids)
    shows none of S's analyses; GS's graph has S and V as ancestors and DV, S's graph does not have DV;
  - every node except a grid's sessions is directed-reachable from the focus over the new edges plus the lab graph's
    (part links included), and the sessions reach the grid;
  - user B (advanced): no trace of any private record -- B's graph is byte-identical to B's graph on the same
    registry WITHOUT those records, admin as the control;
  - the per-group link index taken from the whole-registry build equals a build on the group's filtered registry
    (admin, A, B); hidden-id blanking of session / grid subtitles;
  - node whitelist, no path or uploader in the JSON; unknown / hidden focus -> empty, focus null, no echo;
  - descendants capped at 150 with `more` and no count; the link index is built once per registry state;
  - through the portal: basic 403, the API equals the function, the page has the toggle (Graph by default), its
    empty-state text, no inline script; in a browser (Playwright, Edge channel) at 390 and 1280 px, light and dark.
    Screenshots go to $CJ_SHOT_DIR when set.

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

S, T_SID, U, V = "20991201", "20991202", "20991204", "20991205"
G1, G2, GT, GS = "29901", "29902", "29903", "29905"
LV, DS, MOD = "lv_20991201_sg", "ds_20991201_sg", "m_20991201_sg"
EXS, DT, DU, DV = "rf_20991201_sgexport", "ds_20991202_other", "ds_20991201_unreg", "ds_20991205_shared"
P, P2, PKG2, P3 = "ps_20991201_aaaa01", "ps_20991201_aaaa02", "pred_ps_20991201_aaaa02", "ps_20991201_aaaa05"
Q, QPKG = "ps_20991204_aaaa04", "pred_ps_20991204_aaaa04"
PT = "ps_20991202_aaaa06"
HP, HPKG, JOBH = "ps_20991201_cccc03", "pred_ps_20991201_cccc03", "job_20991201_revise_cccc03"
CID, CID2 = "20991201_abc123", "20991201_def456"
PRIVATE = (P, P2, PKG2, P3, Q, QPKG, PT, HP, HPKG, JOBH, CID, CID2, "aaaa0", "cccc03", "abc123", "def456")
OF_T = (T_SID, GT, DT, PT)


def rec(entity, rid, **kw):
    return {"schema": entity + "/1", "id": rid, "logged": "2099-12-01T09:00:00", **kw}


def ins(*triples):
    return [{"entity": e, "id": i, "role": r} for e, i, r in triples]


def ps(rid, owner, session=None, **kw):
    return rec("prediction_set", rid, owner=owner, model=MOD, status="done",
               source={"kind": "session", "session": session} if session else {"kind": "upload"}, **kw)


def rows(tmp: Path, private=True) -> dict:
    lv_dir = tmp / "labels" / LV
    out = {
        "session": [rec("session", S, date="2099-12-01", status="processed",
                        session_grids={"1": G1, "2": G2, "3": GS},
                        conversions=[{"id": CID, "uploaded_by": "usera", "n_tiles": 9},
                                     {"id": CID2, "uploaded_by": "userc", "n_tiles": 4}]),
                    rec("session", T_SID, date="2099-12-02", status="processed", session_grids={"1": GT}),
                    rec("session", U, date="2099-12-04", status="processed"),
                    rec("session", V, date="2099-12-05", status="processed", session_grids={"1": GS})],
        "grid": [rec("grid", G1, sample_label="EV prep one"), rec("grid", G2, sample_label="EV prep two"),
                 rec("grid", GT, sample_label="Other prep, see " + HP), rec("grid", GS, sample_label="Shared")],
        "image": [rec("image", "img_s1_a_png", session=S, grid=G1, session_grid="1"),
                  rec("image", "img_s1_b_png", session=S, grid=G1, session_grid="1"),
                  rec("image", "img_s2_a_png", session=S, session_grid="2"),       # grid from the session's slot
                  rec("image", "img_s3_a_png", session=S, grid=GS, session_grid="3"),
                  rec("image", "img_s_x_png", session=S),                          # no grid
                  rec("image", "img_s_unreg_png", session=S, grid="29999"),        # a grid the registry lacks
                  rec("image", "img_v_a_png", session=V, grid=GS, session_grid="1"),
                  rec("image", "img_t_a_png", session=T_SID, grid=GT, session_grid="1")],
        "label_version": [rec("label_version", LV, owner=str(lv_dir / "version.json"), status="active")],
        "dataset": [rec("dataset", DS, status="built", inputs=ins(("label_version", LV, "source"))),
                    rec("dataset", DT, status="built", image_list={"file": "lists/dt.tsv"}),
                    rec("dataset", DU, status="built", image_list={"file": "lists/du.tsv"}),
                    rec("dataset", DV, status="built", image_list={"file": "lists/dv.tsv"})],
        "model": [rec("model", MOD, status="candidate", inputs=ins(("dataset", DS, "train_data")))],
        "roboflow_export": [rec("roboflow_export", EXS, status="imported", image_list={"file": "lists/exs.tsv"})],
    }
    if private:
        out["prediction_set"] = [
            ps(P, "usera", S, inputs=ins(("model", MOD, "model"))),
            ps(P2, "usera", S), ps(P3, "usera", S), ps(PT, "usera", T_SID),
            ps(Q, "usera", U, inputs=ins(("model", MOD, "model"))),
            ps(HP, "userc", S)]
        out["review_package"] = [
            rec("review_package", PKG2, kind="prediction", owner="usera", prediction_set=P2, model=MOD, n_images=2,
                status="built"),
            rec("review_package", QPKG, kind="prediction", owner="usera", prediction_set=Q, model=MOD, n_images=2,
                status="built"),
            rec("review_package", HPKG, kind="prediction", owner="userc", prediction_set=HP, n_images=1,
                status="built", path=str(tmp / "ext_review" / HPKG))]
        out["job"] = [rec("job", JOBH, type="revise", status="ready", image_list={"file": "lists/jobh.txt"},
                          inputs=ins(("review_package", HPKG, "predictions")), outputs=[])]
    return out


def write_fixture(tmp: Path, reg: Path, private=True):
    """The registry (only these rows), its lists, the label version's change log, the hidden package's data.js, the
    conversion folders and the request.json of the analyses run on them."""
    reg.mkdir(parents=True, exist_ok=True)
    for ent, recs in rows(tmp, private).items():
        (reg / f"{ent}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    (reg / "lists").mkdir(exist_ok=True)
    for name, stem in (("exs.tsv", "img_s_x_png"), ("dt.tsv", "img_t_a_png"), ("du.tsv", "img_s_unreg_png"),
                       ("dv.tsv", "img_v_a_png"), ("jobh.txt", "img_s1_a_png")):
        (reg / "lists" / name).write_text(f"{stem}\ttrain\n", encoding="utf-8")
    lv_dir = tmp / "labels" / LV
    lv_dir.mkdir(parents=True, exist_ok=True)
    (lv_dir / "version.json").write_text(json.dumps({"n_images_applied": 1}), encoding="utf-8")
    (lv_dir / "changes.jsonl").write_text(json.dumps({"image": "img_s1_a.png"}) + "\n", encoding="utf-8")
    pk = tmp / "ext_review" / HPKG
    pk.mkdir(parents=True, exist_ok=True)
    (pk / "data.js").write_text('window.REVIEW_DATA = {"images": [{"file": "img_s1_b.png"}]};', encoding="utf-8")
    for cid, owner, set_id in ((CID, "usera", P2), (CID2, "userc", P3)):
        conv = tmp / "raw" / "raw_tiles" / cid
        (conv / "tiles").mkdir(parents=True, exist_ok=True)
        (conv / "convert.json").write_text(json.dumps({"schema": "cj_convert/1", "id": cid, "owner": owner}),
                                           encoding="utf-8")
        req = tmp / "CryoAI-laptop" / "predict" / set_id / "request.json"
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


def pairs_of(g):
    t = [n["title"] for n in g["nodes"]]
    return {(t[a], t[b]) for a, b, _ in g["edges"]}


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

    def records(self, reg=None):
        records, err = D.read_registry(reg or self.reg)
        self.assertIsNone(err)
        return D.with_disk_packages(records)

    def graph(self, v, focus, reg=None):
        return D.source_graph(self.records(reg), v, reg or self.reg, focus)

    # ------------------------------------------------------------ known answers
    def test_session_graph_per_viewer(self):
        mine = {S, G1, G2, GS, LV, DS, MOD, EXS, DU}
        own_a = {P, P2, PKG2, P3, CID}
        want = {"admin": mine | own_a | {CID2, HP, HPKG, JOBH}, "A": mine | own_a, "B": mine}
        for name, v in (("admin", self.ADMIN), ("A", self.A), ("B", self.B)):
            g = self.graph(v, "session|" + S)
            with self.subTest(viewer=name):
                self.assertEqual(titles(g), want[name])
                self.assertEqual(g["nodes"][g["focus"]]["title"], S)
                self.assertEqual(g["nodes"][g["focus"]]["kind"], "session")
                self.assertTrue(titles(g).isdisjoint(OF_T + (U, V, DV, Q, QPKG, "29999")))
                self.assertEqual(g["scope"], "item")
                self.assertFalse(g["more"])
                self.assertEqual(g["cols"][0], "Imaging sessions")           # the lab's own first lane
                self.assertEqual(g["cols"][D.SOURCE_AT:D.SOURCE_AT + len(D.SOURCE_COLS)], list(D.SOURCE_COLS))

    def test_edges_are_the_recorded_links(self):
        e = pairs_of(self.graph(self.A, "session|" + S))
        for pair in ((S, G1), (S, G2), (S, GS), (G1, LV), (LV, DS), (DS, MOD), (MOD, P), (S, P), (S, EXS), (S, DU),
                     (S, CID), (CID, P2), (CID, PKG2), (S, P3)):
            self.assertIn(pair, e)
        self.assertFalse({p for p in e if p[0] in (G2, GS)})      # G2's and S's GS images are used by nothing
        self.assertNotIn((S, P2), e)                              # P2 hangs off the conversion it ran on
        g = self.graph(self.A, "session|" + S)
        t = [n["title"] for n in g["nodes"]]
        roles = {(t[a], t[b]): r for a, b, r in g["edges"]}
        self.assertEqual(roles[(G1, LV)], "used its images")
        self.assertEqual(roles[(S, CID)], "upload")
        self.assertEqual(roles[(CID, P2)], "analysed")
        # admin control: the hidden package is linked from G1 (its data.js lists a G1 image), HP from the session,
        # the hidden job from G1 (its list), P3 from the conversion the admin can see
        ea = pairs_of(self.graph(self.ADMIN, "session|" + S))
        for pair in ((G1, HPKG), (S, HP), (CID, P2), (G1, JOBH), (CID2, P3)):
            self.assertIn(pair, ea)
        self.assertNotIn((S, P3), ea)

    def test_another_sessions_analyses_stay_out(self):
        # S1: M (trained on S's images) made Q / QPKG for session U: not in S's graph, in U's graph
        for v in (self.ADMIN, self.A):
            with self.subTest(user=v.user["id"]):
                self.assertTrue(titles(self.graph(v, "session|" + S)).isdisjoint({Q, QPKG}))
                gu = self.graph(v, "session|" + U)
                self.assertEqual(titles(gu), {U, Q, QPKG})
                self.assertTrue({(U, Q), (U, QPKG)} <= pairs_of(gu))
        # control: the lab graph does lead from M to Q and QPKG
        item = D.item_graph(D.lab_graph(self.records(), self.ADMIN, self.reg, "model|" + MOD))
        self.assertTrue({Q, QPKG, P, PKG2} <= titles(item))
        self.assertTrue(titles(self.graph(self.B, "session|" + U)) == {U})

    def test_grid_graphs(self):
        g2 = self.graph(self.ADMIN, "grid|" + G2)
        self.assertEqual(titles(g2), {G2, S})
        self.assertEqual(g2["nodes"][g2["focus"]]["title"], G2)
        g1 = self.graph(self.ADMIN, "grid|" + G1)
        # S has three grids: none of S's analyses here (only those of a one-grid session); the hidden package and job
        # are here for the admin because they list G1 images
        self.assertEqual(titles(g1), {G1, S, LV, DS, MOD, HPKG, JOBH})
        self.assertTrue(titles(g1).isdisjoint({EXS, P, P2, PKG2, P3, Q, QPKG, CID, G2, HP}))
        self.assertEqual(titles(self.graph(self.B, "grid|" + G1)), {G1, S, LV, DS, MOD})
        self.assertEqual(titles(self.graph(self.A, "grid|" + G1)), {G1, S, LV, DS, MOD})
        gt = self.graph(self.ADMIN, "session|" + T_SID)
        self.assertEqual(titles(gt), {T_SID, GT, DT, PT})                          # the unrelated session's own

    def test_one_grid_session_gives_its_grid_its_analyses(self):
        # N1: T imaged only GT, so GT's graph has T's analysis PT, linked from the grid
        for v, has in ((self.ADMIN, True), (self.A, True), (self.B, False)):
            g = self.graph(v, "grid|" + GT)
            with self.subTest(user=v.user["id"]):
                self.assertEqual(titles(g), {GT, T_SID, DT} | ({PT} if has else set()))
                self.assertEqual((GT, PT) in pairs_of(g), has)

    def test_shared_grid_does_not_cross_link(self):
        # S3: GS is imaged in S and V; V's image is in DV. DV is on V's and GS's graph, never on S's
        self.assertNotIn(DV, titles(self.graph(self.ADMIN, "session|" + S)))
        self.assertEqual(titles(self.graph(self.ADMIN, "session|" + V)), {V, GS, DV})
        gs = self.graph(self.ADMIN, "grid|" + GS)
        self.assertEqual(titles(gs), {GS, S, V, DV})
        self.assertTrue({(S, GS), (V, GS), (GS, DV)} <= pairs_of(gs))

    def test_image_on_an_unregistered_grid_counts_for_the_session(self):
        g = self.graph(self.ADMIN, "session|" + S)
        self.assertIn((S, DU), pairs_of(g))
        self.assertNotIn("29999", json.dumps(g))
        self.assertEqual(self.graph(self.ADMIN, "grid|29999")["nodes"], [])

    def test_own_set_on_a_conversion_the_viewer_cannot_see_falls_back_to_the_session(self):
        a = self.graph(self.A, "session|" + S)
        self.assertIn((S, P3), pairs_of(a))
        self.assertNotIn(CID2, json.dumps(a))
        self.assertIn((CID2, P3), pairs_of(self.graph(self.ADMIN, "session|" + S)))          # control

    def test_every_node_is_directed_reachable(self):
        for v in (self.ADMIN, self.A, self.B):
            for focus in ("session|" + S, "grid|" + G1, "grid|" + G2, "grid|" + GS, "grid|" + GT,
                          "session|" + T_SID, "session|" + U):
                g = self.graph(v, focus)
                f = g["focus"]
                down = reachable(g, f)
                up = reachable(g, f, reverse=True)
                with self.subTest(focus=focus):
                    self.assertEqual(down | up, set(range(len(g["nodes"]))))
                    self.assertEqual(up - {f}, {i for i, n in enumerate(g["nodes"]) if n["kind"] == "session"
                                                and i != f})        # ancestors: only a grid's sessions
                    for s in up - {f}:
                        self.assertTrue(set(range(len(g["nodes"]))) - (up - {f}) <= reachable(g, s))

    def test_lab_part_is_the_lab_graphs_descendants_less_other_sessions_analyses(self):
        g = self.graph(self.ADMIN, "session|" + S)
        lab = {n["title"] for n in g["nodes"] if n["kind"] not in ("session", "grid", "upload")}
        records = self.records()
        union = set()
        for key in ("label_version|" + LV, "roboflow_export|" + EXS, "dataset|" + DU, "prediction_set|" + P,
                    "prediction_set|" + P2, "review_package|" + PKG2, "prediction_set|" + P3, "prediction_set|" + HP,
                    "review_package|" + HPKG, "job|" + JOBH):
            item = D.item_graph(D.lab_graph(records, self.ADMIN, self.reg, key))
            union |= {item["nodes"][i]["title"] for i in reachable(item, item["focus"])}
        self.assertTrue(lab <= union)
        self.assertEqual(union - lab, {Q, QPKG})            # exactly U's analyses, reached only through M

    # ------------------------------------------------------------ visibility
    def test_b_has_no_trace_of_private_items(self):
        for focus in ("session|" + S, "grid|" + G1, "grid|" + G2, "grid|" + GS, "session|" + U):
            b = json.dumps(self.graph(self.B, focus))
            with self.subTest(focus=focus):
                for s in PRIVATE + ("usera", "userc"):
                    self.assertNotIn(s, b)
                # byte-identical to B's graph on a registry where the private records never existed
                self.assertEqual(b, json.dumps(self.graph(self.B, focus, self.reg_absent)))
        admin = json.dumps(self.graph(self.ADMIN, "session|" + S))
        for s in (P, P2, PKG2, P3, HP, HPKG, JOBH, CID, CID2):
            self.assertIn(s, admin)                                         # control
        a = json.dumps(self.graph(self.A, "session|" + S))
        for s in (HP, HPKG, JOBH, CID2, "cccc03", "userc"):
            self.assertNotIn(s, a)                                          # A sees only A's own
        self.assertNotEqual(json.dumps(self.graph(self.A, "session|" + S)),
                            json.dumps(self.graph(self.A, "session|" + S, self.reg_absent)))   # the check can fail

    def test_group_index_from_the_whole_registry_equals_a_filtered_build(self):
        # S2: source_links = build_source_links(whole registry) minus the group's hidden keys; QC's claim that this
        # equals building on the group's filtered registry, checked for each viewer (the hidden job and package
        # are in the whole-registry links, so the subtraction is doing work for A and B)
        records = self.records()
        whole = None
        for v in (self.ADMIN, self.A, self.B):
            extra = {}
            D.lab_graph(records, v, self.reg, None, extra)
            current = extra["base"][1]
            kept = {k: r for k, r in current.items() if k not in extra["drop"]}
            with self.subTest(user=v.user["id"]):
                self.assertEqual(D.source_links(self.reg, extra["base"], extra["drop"]),
                                 D.build_source_links(kept, self.reg))
                if v is self.ADMIN:
                    whole = D.build_source_links(current, self.reg)
                    self.assertFalse(extra["drop"])
                else:
                    self.assertIn("job|" + JOBH, extra["drop"])
                    self.assertNotEqual(D.source_links(self.reg, extra["base"], extra["drop"]), whole)

    def test_conversion_only_for_owner_and_admin(self):
        for v, want in ((self.ADMIN, {CID, CID2}), (self.A, {CID}), (self.B, set())):
            g = self.graph(v, "session|" + S)
            up = {n["title"]: n for n in g["nodes"] if n["kind"] == "upload"}
            with self.subTest(user=v.user["id"]):
                self.assertEqual(set(up), want)
                if CID in up:
                    self.assertEqual(up[CID], {"title": CID, "kind": "upload", "col": 1, "hist": 0, "part": -1,
                                               "sub": "9 tiles", "status": None, "url": f"/predict/convert/{CID}/",
                                               "maker": "upload"})
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
                                   "sub": "2099-12-01", "status": "processed", "url": f"/data/lineage/session/{S}/",
                                   "maker": "session"})
        self.assertEqual(node[G1], {"title": G1, "kind": "grid", "col": 1, "hist": 0, "part": -1,
                                    "sub": "EV prep one", "status": None, "url": f"/data/lineage/grid/{G1}/",
                                    "maker": "grid"})
        self.assertGreaterEqual(node[LV]["col"], D.SOURCE_AT + len(D.SOURCE_COLS))   # lab lanes moved right

    def test_box_text_quoting_a_hidden_id_is_blanked(self):
        # N4: GT's sample label quotes HP's id: blank for A and B (they can't see HP), shown to the admin
        for v, sub in ((self.A, ""), (self.B, ""), (self.ADMIN, "Other prep, see " + HP)):
            for focus in ("grid|" + GT, "session|" + T_SID):
                g = self.graph(v, focus)
                with self.subTest(user=v.user["id"], focus=focus):
                    self.assertEqual({n["title"]: n for n in g["nodes"]}[GT]["sub"], sub)

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

    def test_links_built_once_per_registry_state(self):
        TG.clear_caches()
        hpm = D._history_module()
        real, calls = hpm.lookup_index, []
        hpm.lookup_index = lambda *a: (calls.append(1), real(*a))[1]
        try:
            self.graph(self.B, "session|" + S)
            self.graph(self.B, "grid|" + G1)
            self.graph(D.Viewer(Ctx.can_see, {"id": "userd", "role": "advanced"}), "session|" + S)   # same group
            self.assertEqual(len(calls), 2)     # lab_graph's build_model runs lookup_index once, the index once
            self.graph(self.ADMIN, "session|" + S)
            self.assertEqual(len(calls), 3)     # a new group: only its lab graph; the index is per registry state
            self.graph(self.A, "grid|" + G2)
            self.assertEqual(len(calls), 4)
        finally:
            hpm.lookup_index = real

    def test_recall_makes_an_entry_the_newest(self):
        TG.clear_caches()
        D._remember("first", 1)
        for i in range(D.GRAPH_CACHE_N - 1):
            D._remember(("x", i), i)
        self.assertEqual(D._recall("first"), 1)            # used: now the newest
        D._remember("one more", 2)                         # evicts the oldest, ("x", 0), not "first"
        self.assertEqual(D._recall("first"), 1)
        self.assertIsNone(D._recall(("x", 0)))
        TG.clear_caches()

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
                self.assertNotIn(Q, titles(g))
                if not has:
                    for s in PRIVATE + ("usera",):
                        self.assertNotIn(s, body.decode("utf-8"))
        st, _, body = self.api("session|" + S, "bas")
        self.assertEqual(st, 403)
        self.assertNotIn(S.encode(), body)
        # without scope=item: the whole lab graph. #15: a session an export lists is a lab node (its lane is
        # "Imaging sessions"), so it is the one focused; a session no export lists focuses nothing
        whole = json.loads(self.get("/data/api/lineage/graph?focus=" + quote("session|" + S, safe=""), "userb")[2])
        self.assertEqual((whole["nodes"][whole["focus"]]["kind"], whole["nodes"][whole["focus"]]["title"]),
                         ("session", S))
        self.assertNotIn("scope", whole)
        other = json.loads(self.get("/data/api/lineage/graph?focus=" + quote("session|" + T_SID, safe=""),
                                    "userb")[2])
        self.assertIsNone(other["focus"])

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
                self.assertIn('data-empty="Not found or not visible to you."', main)
                self.assertIn('<script src="/data/data.js" defer></script>', main)
                self.assertNotRegex(main, r"<script(?![^>]*\bsrc=)")
                self.assertNotIn("/data/lineage/graph/?focus=", main)       # the lab graph link focuses nothing
                for s in PRIVATE:
                    self.assertNotIn(s, page)
        grid = self.get(f"/data/lineage/grid/{G1}/", "userb")[2].decode("utf-8")
        self.assertIn("Analyses appear on the session's graph", grid)
        item = self.get(f"/data/lineage/model/{MOD}/", "userb")[2].decode("utf-8")
        self.assertIn('id="cj-data-view-toggle"', item)                      # control: an item page, no default
        self.assertNotIn('data-default="graph"', item)
        self.assertNotIn(' data-empty="', item)
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
                        self.assertTrue(page.is_checked("#cj-data-graph-an"))          # its own analyses shown
                        self.assertTrue({S, G1, G2, LV, DS, MOD, EXS, CID, P, P2, PKG2, P3} <= self.names(page))
                        self.assertTrue(self.names(page).isdisjoint({Q, QPKG}))
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
            self.assertTrue(got.isdisjoint({P, P2, PKG2, P3, CID, CID2, HP, HPKG, JOBH}))
            self.assertFalse(page.is_checked("#cj-data-graph-an"))         # none of its analyses are B's
            html_ = page.content()
            for s in PRIVATE:
                self.assertNotIn(s, html_)
            page.screenshot(path=str(self.shots / "sg_session_userb_1280_light.png"), full_page=True)
            page.goto(self.p.origin + f"/data/lineage/grid/{G1}/")
            page.wait_for_selector(".cj-data-graph-node")
            self.assertEqual(self.names(page), {G1, S, LV, DS, MOD})
            page.screenshot(path=str(self.shots / "sg_grid_userb_1280_light.png"), full_page=True)
            # a one-grid session's grid: B sees none of its analyses
            page.goto(self.p.origin + f"/data/lineage/grid/{GT}/")
            page.wait_for_selector(".cj-data-graph-node")
            self.assertEqual(self.names(page), {GT, T_SID, DT})
            self.assertEqual((rec_["foreign"], rec_["errors"], rec_["console"]), ([], [], []))
        finally:
            ctx.close()

    def test_grid_with_its_sessions_analysis_in_a_browser(self):
        ctx, page, rec_ = self.session(1280, "light", "usera")
        try:
            page.goto(self.p.origin + f"/data/lineage/grid/{GT}/")
            page.wait_for_selector(".cj-data-graph-node")
            page.wait_for_load_state("networkidle")
            self.assertEqual(self.names(page), {GT, T_SID, DT, PT})
            self.assertTrue(page.is_checked("#cj-data-graph-an"))
            page.screenshot(path=str(self.shots / "sg_grid_onegrid_usera_1280_light.png"), full_page=True)
            self.assertEqual((rec_["foreign"], rec_["errors"], rec_["console"]), ([], [], []))
        finally:
            ctx.close()

    def test_empty_state_wording(self):
        # N3: a focus that comes back null on a session page reads "Not found or not visible to you."
        ctx, page, rec_ = self.session(1280, "light", "userb")
        try:
            page.route("**/data/api/lineage/graph*", lambda r: r.fulfill(
                status=200, content_type="application/json",
                body=json.dumps({"schema": "cj_data/1", "scope": "item", "cols": [], "nodes": [], "edges": [],
                                 "focus": None, "more": False, "error": None})))
            page.goto(self.p.origin + f"/data/lineage/session/{S}/")
            page.wait_for_function("document.getElementById('cj-data-graph-status').textContent.indexOf('Loading')"
                                   " < 0")
            self.assertEqual(page.text_content("#cj-data-graph-status"), "Not found or not visible to you.")
            self.assertEqual((rec_["foreign"], rec_["errors"]), ([], []))
        finally:
            ctx.close()


if __name__ == "__main__":
    unittest.main()
