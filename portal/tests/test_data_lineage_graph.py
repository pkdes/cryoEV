"""
Data module, Lineage item pages: the Tree | Graph toggle and the item graph (portal/modules/data.py item_graph,
/data/api/lineage/graph?focus=<key>&scope=item, /data/data.js).

  - known answer: every node of an item graph is directed-reachable from the focus in the FULL lab graph (ancestor
    over reverse edges, descendant over forward edges, `part` links followed), the slice is exactly that set, and the
    edges / `part` / `focus` match the full graph after the index remap;
  - mid-chain dataset: its model (descendant) and label set (ancestor) are in; a sibling dataset built from the same
    label set, that sibling's model and an unrelated model are out;
  - a review-package part shows its parent package; the package shows its parts;
  - the descendant cap (150) sets `more` and nothing else (no count); an out-of-slice parent -> -1;
  - leaks: the visibility tests of test_data_graph rerun with scope=item, admin as the control; a hidden or unknown
    focus gives an empty graph, focus null and no echo;
  - the page in a browser (Playwright, Edge channel): Tree default, toggle, remembered choice, blocked storage,
    390 and 1280 px, light and dark; no toggle on a session page. Screenshots go to $CJ_SHOT_DIR when set.

Temp registry from registry/samples plus fixture rows (cjtest.Env); never the real one.

  python -m unittest portal.tests.test_data_lineage_graph -v
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
import test_data_graph as TG  # noqa: E402
from portal.modules import data as D  # noqa: E402

try:
    from playwright.sync_api import sync_playwright
except ImportError:  # pragma: no cover
    sync_playwright = None

LS = "lv_20261107_chain"                  # the label set
DS_MID, DS_SIB, DS_OTHER = "ds_20261107_mid", "ds_20261107_sibling", "ds_20261107_other"
M_MID, M_SIB, M_OTHER = "m_20261107_mid", "m_20261107_sibling", "m_20261107_other"
LS_OTHER = "lv_20261107_other"


def row(entity, rid, status, inputs=(), **kw):
    r = {"schema": entity + "/1", "id": rid, "logged": "2026-11-07T09:00:00", "status": status}
    if inputs:
        r["inputs"] = [{"entity": e, "id": i, "role": role} for e, i, role in inputs]
    r.update(kw)
    return r


def chain_rows() -> dict:
    return {
        "label_version": [row("label_version", LS, "active", n_images_applied=3),
                          row("label_version", LS_OTHER, "active", n_images_applied=2)],
        "dataset": [row("dataset", DS_MID, "built", [("label_version", LS, "source")]),
                    row("dataset", DS_SIB, "built", [("label_version", LS, "source")]),
                    row("dataset", DS_OTHER, "built", [("label_version", LS_OTHER, "source")])],
        "model": [row("model", M_MID, "candidate", [("dataset", DS_MID, "train_data")]),
                  row("model", M_SIB, "candidate", [("dataset", DS_SIB, "train_data")]),
                  row("model", M_OTHER, "candidate", [("dataset", DS_OTHER, "train_data")])]}


BASE_WRITE = TG.write_registry        # the Lab graph test's own writer, captured before Served swaps it


def write_registry(root: Path):
    BASE_WRITE(root)
    for ent, recs in chain_rows().items():
        with (root / f"{ent}.jsonl").open("a", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
    TG.clear_caches()


def key_of(n):
    return TG.node_key(n)


def titles(g):
    return {n["title"] for n in g["nodes"]}


def reach(n, pairs, start):
    """Nodes reachable from start over pairs (a, b) read a -> b only."""
    nxt = {}
    for a, b in pairs:
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
        cls.tmp = Path(tempfile.mkdtemp(prefix="cj_lgraph_"))
        cls.set_roots(cls.tmp)
        cls.reg = cls.tmp / "registry"
        write_registry(cls.reg)
        TG.write_disk(cls.tmp / "CryoAI-laptop")
        cls.records, err = D.read_registry(cls.reg)
        assert err is None
        cls.records = D.with_disk_packages(cls.records)

    @classmethod
    def tearDownClass(cls):
        cls.restore_roots()
        shutil.rmtree(cls.tmp, ignore_errors=True)
        TG.clear_caches()

    def full(self, v):
        return D.lab_graph(self.records, v, self.reg)

    def item(self, v, key):
        return D.item_graph(D.lab_graph(self.records, v, self.reg, key))

    # ------------------------------------------------------------ known answers
    def test_every_node_is_directed_reachable_and_edges_match(self):
        for v in (TG.ADMIN, TG.ADV):
            full = self.full(v)
            keys = [key_of(n) for n in full["nodes"]]
            parts = [(n["part"], i) for i, n in enumerate(full["nodes"]) if n["part"] >= 0]
            pairs = [(a, b) for a, b, _ in full["edges"]] + parts      # a part link reads package -> part
            self.assertGreater(len(parts), 0)
            for f, key in enumerate(keys):
                with self.subTest(viewer=v.user_id if hasattr(v, "user_id") else "v", focus=key):
                    g = self.item(v, key)
                    self.assertEqual(g["nodes"][g["focus"]]["title"], full["nodes"][f]["title"])
                    down = reach(len(keys), pairs, f)
                    up = reach(len(keys), [(b, a) for a, b in pairs], f)
                    want = sorted(down | up)
                    got = [keys.index(key_of(n)) for n in g["nodes"]]
                    self.assertEqual(got, want)                         # exactly ancestors + descendants + focus
                    new = {old: k for k, old in enumerate(got)}
                    self.assertEqual(g["edges"], [[new[a], new[b], r] for a, b, r in full["edges"]
                                                  if a in new and b in new])
                    for k, old in enumerate(got):
                        n, o = g["nodes"][k], full["nodes"][old]
                        self.assertEqual(tuple(n), D.GRAPH_NODE_KEYS)
                        self.assertEqual({x: n[x] for x in n if x != "part"}, {x: o[x] for x in o if x != "part"})
                        self.assertEqual(n["part"], new.get(o["part"], -1))
                    self.assertEqual(g["scope"], "item")
                    self.assertFalse(g["more"])

    def test_mid_chain_dataset(self):
        g = self.item(TG.ADMIN, "dataset|" + DS_MID)
        t = titles(g)
        self.assertIn(M_MID, t)                                         # its model (descendant)
        self.assertIn(LS, t)                                            # its label set (ancestor)
        self.assertIn(DS_MID, t)
        self.assertNotIn(DS_SIB, t)                                     # a sibling built from the same label set
        self.assertNotIn(M_SIB, t)
        self.assertNotIn(M_OTHER, t)                                    # an unrelated model
        self.assertNotIn(DS_OTHER, t)
        # control: the full graph has all of them, and the sibling IS reachable from the label set, undirected
        full = titles(self.full(TG.ADMIN))
        for x in (DS_SIB, M_SIB, M_OTHER, DS_MID, M_MID):
            self.assertIn(x, full)
        up = self.item(TG.ADMIN, "label_version|" + LS)
        self.assertTrue({DS_MID, DS_SIB, M_MID, M_SIB} <= titles(up))   # control: from the label set both siblings show
        self.assertNotIn(M_OTHER, titles(up))

    def test_part_shows_its_parent_package_and_back(self):
        g = self.item(TG.ADV, "review_package|" + TG.OWN_PART)
        t = TG.by_title(g)
        self.assertIn(TG.OWN_PKG, t)
        order = list(t)
        self.assertEqual(g["nodes"][order.index(TG.OWN_PART)]["part"], order.index(TG.OWN_PKG))
        pk = self.item(TG.ADV, "review_package|" + TG.OWN_PKG)
        self.assertIn(TG.OWN_PART, titles(pk))

    def test_part_whose_parent_is_outside_the_slice_is_minus_one(self):
        def node(title, col=0, part=-1):
            return dict(title=title, kind="package", col=col, hist=0, part=part, sub="", status="x", url=None)
        # focus -> part by an edge: the part is a descendant, its package is not (the walk never goes up from it)
        full = {"schema": "s", "cols": ["a"], "error": None, "focus": 0,
                "nodes": [node("focus"), node("pkg"), node("part", part=1)], "edges": [[0, 2, "input"]]}
        g = D.item_graph(full)
        self.assertEqual([n["title"] for n in g["nodes"]], ["focus", "part"])
        self.assertEqual([n["part"] for n in g["nodes"]], [-1, -1])
        # focus the part itself: the package is an ancestor, so the index is remapped, not dropped
        g = D.item_graph(dict(full, focus=2))
        self.assertEqual([n["title"] for n in g["nodes"]], ["focus", "pkg", "part"])
        self.assertEqual([n["part"] for n in g["nodes"]], [-1, -1, 1])
        # focus the package: its parts are descendants, and "focus" (a parent of the part only by an edge) is not
        g = D.item_graph(dict(full, focus=1))
        self.assertEqual([n["title"] for n in g["nodes"]], ["pkg", "part"])
        self.assertEqual(g["nodes"][g["focus"]]["title"], "pkg")

    def test_descendant_cap_sets_more_without_a_count(self):
        n = D.ITEM_DOWNSTREAM_MAX + 40
        full = {"schema": "s", "cols": ["a", "b"], "error": None, "focus": 0,
                "nodes": [dict(title=f"n{i}", kind="job", col=1 if i else 0, hist=0, part=-1, sub="", status="x",
                               url=None) for i in range(n + 1)],
                "edges": [[0, i, "input"] for i in range(1, n + 1)]}
        g = D.item_graph(full)
        self.assertEqual(len(g["nodes"]), 1 + D.ITEM_DOWNSTREAM_MAX)
        self.assertIs(g["more"], True)
        self.assertEqual(set(g), {"schema", "scope", "cols", "nodes", "edges", "focus", "more", "error"})
        self.assertNotIn(str(n), json.dumps({k: v for k, v in g.items() if k not in ("nodes", "edges")}))
        small = D.item_graph(dict(full, nodes=full["nodes"][:5], edges=full["edges"][:4]))
        self.assertIs(small["more"], False)
        # ancestors are never capped
        anc = {"schema": "s", "cols": ["a"], "error": None, "focus": n,
               "nodes": [dict(title=f"n{i}", kind="job", col=0, hist=0, part=-1, sub="", status="x", url=None)
                         for i in range(n + 1)], "edges": [[i, n, "input"] for i in range(n)]}
        self.assertEqual(len(D.item_graph(anc)["nodes"]), n + 1)

    def test_cycle_through_the_focus_still_expands_downstream(self):
        def node(t):
            return dict(title=t, kind="job", col=0, hist=0, part=-1, sub="", status="x", url=None)
        full = {"schema": "s", "cols": ["a"], "error": None, "focus": 0, "nodes": [node("a"), node("b"), node("c")],
                "edges": [[0, 1, "r"], [1, 0, "r"], [1, 2, "r"]]}
        self.assertEqual([n["title"] for n in D.item_graph(full)["nodes"]], ["a", "b", "c"])

    def test_cap_counts_only_new_descendants_when_some_are_also_ancestors(self):
        def node(t):
            return dict(title=t, kind="job", col=0, hist=0, part=-1, sub="", status="x", url=None)
        n = D.ITEM_DOWNSTREAM_MAX
        # focus 0; 5 nodes that are both ancestors and descendants (0 -> a -> 0), then n+10 pure descendants
        nodes = [node("f")] + [node(f"c{i}") for i in range(5)] + [node(f"d{i}") for i in range(n + 10)]
        edges = [[0, i, "r"] for i in range(1, 6)] + [[i, 0, "r"] for i in range(1, 6)] \
            + [[0, i, "r"] for i in range(6, len(nodes))]
        g = D.item_graph({"schema": "s", "cols": ["a"], "error": None, "focus": 0, "nodes": nodes, "edges": edges})
        self.assertEqual(len(g["nodes"]), 1 + 5 + n)            # the 5 cycle nodes do not eat the cap
        self.assertTrue(g["more"])
        edges = edges[:10 + n - 1]              # fewer than the cap of pure descendants
        g = D.item_graph({"schema": "s", "cols": ["a"], "error": None, "focus": 0,
                          "nodes": nodes[:6 + n - 1], "edges": edges})
        self.assertFalse(g["more"])

    def test_part_pointing_at_itself_is_no_part(self):
        def node(t, part=-1):
            return dict(title=t, kind="package", col=0, hist=0, part=part, sub="", status="x", url=None)
        full = {"schema": "s", "cols": ["a"], "error": None, "focus": 1,
                "nodes": [node("x"), node("self", part=1)], "edges": [[0, 1, "r"]]}
        g = D.item_graph(full)
        self.assertEqual([n["title"] for n in g["nodes"]], ["x", "self"])
        self.assertEqual([n["part"] for n in g["nodes"]], [-1, -1])
        g = D.item_graph(dict(full, focus=0))
        self.assertEqual([n["part"] for n in g["nodes"]], [-1, -1])

    def test_loop_in_the_registry_terminates(self):
        full = {"schema": "s", "cols": ["a"], "error": None, "focus": 0,
                "nodes": [dict(title=t, kind="job", col=0, hist=0, part=-1, sub="", status="x", url=None)
                          for t in "abc"], "edges": [[0, 1, "r"], [1, 2, "r"], [2, 0, "r"]]}
        self.assertEqual(len(D.item_graph(full)["nodes"]), 3)

    # ------------------------------------------------------------ visibility
    def test_hidden_or_unknown_focus_is_empty_and_not_echoed(self):
        for key in ("prediction_set|" + TG.HID_PS, "review_package|" + TG.HID_PKG, "job|" + TG.HID_JOB,
                    "model|nope", "garbage", ""):
            g = self.item(TG.ADV, key)
            with self.subTest(key=key):
                self.assertEqual((g["nodes"], g["edges"], g["focus"], g["cols"], g["more"]), ([], [], None, [], False))
                self.assertNotIn("aaaa11", json.dumps(g))
                if key:
                    self.assertNotIn(key.split("|")[-1], json.dumps(g))
        g = self.item(TG.ADMIN, "prediction_set|" + TG.HID_PS)          # control: the admin sees it
        self.assertIsNotNone(g["focus"])
        self.assertIn(TG.HID_PS, json.dumps(g))

    def test_item_graphs_never_leak_hidden_records(self):
        keys = [key_of(n) for n in self.full(TG.ADMIN)["nodes"]]
        seen_by_admin = ""
        for key in keys:
            seen_by_admin += json.dumps(self.item(TG.ADMIN, key))
            for v in (TG.ADV, TG.BASIC):
                text = json.dumps(self.item(v, key))
                for s in TG.HIDDEN_STRINGS + (TG.SAMPLE_PS, "otherperson", "their export"):
                    with self.subTest(viewer=v, focus=key, s=s):
                        self.assertNotIn(s, text)
        for s in TG.HIDDEN_STRINGS:
            self.assertIn(s, seen_by_admin)                             # control: the admin's item graphs hold them
        adv = json.dumps(self.item(TG.ADV, "model|" + TG.MODEL))
        self.assertIn(TG.OWN_PS, adv)                                   # their own analysis stays downstream
        self.assertNotIn(TG.HID_PS, adv)
        self.assertIn(TG.HID_PS, json.dumps(self.item(TG.ADMIN, "model|" + TG.MODEL)))

    def test_item_slice_equals_the_filtered_graph_slice(self):
        # the slice is cut from the viewer's own filtered graph: nothing the filtered graph lacks can appear
        for v in (TG.ADV, TG.BASIC):
            full = json.dumps(self.full(v))
            for n in self.full(v)["nodes"]:
                for m in self.item(v, key_of(n))["nodes"]:
                    self.assertIn(json.dumps(m["title"]), full)


class Served(TG.Base):
    """The portal over the test registry with the chain fixture added."""

    @classmethod
    def setUpClass(cls):
        TG_write = TG.write_registry
        TG.write_registry = lambda root, with_hidden_ps=True: write_registry(root)
        try:
            super().setUpClass()
        finally:
            TG.write_registry = TG_write


class Http(Served):
    def api(self, key, role="admin", extra="&scope=item"):
        st, h, body = self.get("/data/api/lineage/graph?focus=" + quote(key, safe="") + extra, role)
        return st, h, body

    def test_basic_is_403_and_anonymous_401(self):
        path = "/data/api/lineage/graph?scope=item&focus=" + quote("model|" + M_MID, safe="")
        st, _, body = self.get(path, "basic")
        self.assertEqual(st, 403)
        self.assertNotIn(M_MID.encode(), body)
        self.assertEqual(self.get(path, "advanced")[0], 200)
        self.assertEqual(T.request(self.p.port, "GET", path)[0], 401)

    def test_api_item_scope(self):
        st, h, body = self.api("dataset|" + DS_MID, "advanced")
        self.assertEqual(st, 200)
        self.assertEqual(T.hget(h, "cache-control"), "no-store")
        g = json.loads(body)
        self.assertEqual(g["scope"], "item")
        self.assertEqual(g["nodes"][g["focus"]]["title"], DS_MID)
        self.assertTrue({LS, DS_MID, M_MID} <= titles(g))
        self.assertTrue({DS_SIB, M_SIB, M_OTHER}.isdisjoint(titles(g)))
        records, _ = D.read_registry(self.env.registry)
        want = D.item_graph(D.lab_graph(D.with_disk_packages(records), TG.ADV, self.env.registry,
                                        "dataset|" + DS_MID))
        self.assertEqual(g, want)
        # without scope the whole graph still comes back
        whole = json.loads(self.api("dataset|" + DS_MID, "advanced", "")[2])
        self.assertIn(DS_SIB, titles(whole))
        self.assertNotIn("scope", whole)

    def test_hidden_focus_no_echo_and_admin_control(self):
        for key in ("prediction_set|" + TG.HID_PS, "review_package|" + TG.HID_PKG, "model|nope"):
            st, _, body = self.api(key, "advanced")
            g = json.loads(body)
            with self.subTest(key=key):
                self.assertEqual(st, 200)
                self.assertEqual((g["nodes"], g["edges"], g["focus"]), ([], [], None))
                text = body.decode("utf-8")
                self.assertNotIn(key.split("|")[-1], text)
                self.assertNotIn("aaaa11", text)
        st, _, body = self.api("prediction_set|" + TG.HID_PS, "admin")
        self.assertIsNotNone(json.loads(body)["focus"])

    def test_advanced_item_graphs_do_not_leak(self):
        for key in ("model|" + TG.MODEL, "label_version|" + TG.LV_HID, "roboflow_export|" + TG.LONELY,
                    "review_package|" + TG.OWN_PKG, "prediction_set|" + TG.OWN_PS):
            st, _, body = self.api(key, "advanced")
            text = body.decode("utf-8")
            with self.subTest(key=key):
                self.assertEqual(st, 200)
                for s in TG.HIDDEN_STRINGS + (TG.SAMPLE_PS, "otherperson"):
                    self.assertNotIn(s, text)
        admin = self.api("model|" + TG.MODEL, "admin")[2].decode("utf-8")
        for s in (TG.HID_PS, TG.HID_PKG):
            self.assertIn(s, admin)                                     # control

    def test_lineage_page_has_toggle_and_no_inline_script(self):
        page = self.get("/data/lineage/dataset/" + DS_MID + "/", "advanced")[2].decode("utf-8")
        main = page.split('<main class="cj-main">', 1)[1].split("</main>", 1)[0]
        self.assertIn('id="cj-data-view-toggle"', main)
        self.assertIn('data-endpoint="/data/api/lineage/graph?focus=dataset%7C' + DS_MID + '&amp;scope=item"', main)
        self.assertIn('data-scope="item"', main)
        self.assertIn('<script src="/data/data.js" defer></script>', main)
        self.assertNotRegex(main, r"<script(?![^>]*\bsrc=)")
        self.assertIn('id="cj-data-tree-panel"', main)
        lab = self.get("/data/lineage/graph/")[2].decode("utf-8")       # the lab page: same script, its own attributes
        self.assertIn('data-endpoint="/data/api/lineage/graph"', lab)
        self.assertIn("data-focus-from-url", lab)
        self.assertNotIn("cj-data-view-toggle", lab)

    def test_no_toggle_on_session_or_grid_pages(self):
        for kind, eid in (("session", "20261027"), ("grid", "2602")):
            st, _, body = self.get(f"/data/lineage/{kind}/{eid}/")
            page = body.decode("utf-8")
            with self.subTest(kind=kind):
                self.assertEqual(st, 200)
                self.assertNotIn("cj-data-view-toggle", page)
                self.assertNotIn("cj-data-graph-panel", page)
        # control: an item page has it (guards against the loop above never running)
        self.assertIn("cj-data-view-toggle", self.get("/data/lineage/model/" + M_MID + "/")[2].decode("utf-8"))

    def test_session_page_exists_for_the_toggle_check(self):
        records, _ = D.read_registry(self.env.registry)
        sid = next(iter(records.get("session", {})), None)
        self.assertIsNotNone(sid)
        st, _, body = self.get("/data/lineage/session/" + quote(str(sid)) + "/")
        self.assertEqual(st, 200)
        self.assertNotIn(b"cj-data-view-toggle", body)


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
        cls.shots = Path(os.environ.get("CJ_SHOT_DIR") or tempfile.mkdtemp(prefix="cj_lshots_"))
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

    def test_toggle_in_a_browser(self):
        url = self.p.origin + "/data/lineage/dataset/" + DS_MID + "/"
        for scheme in ("light", "dark"):
            for width in (390, 1280):
                ctx, page, rec = self.session(width, scheme, self.adv_key)
                try:
                    with self.subTest(width=width, scheme=scheme):
                        page.goto(url)
                        page.wait_for_load_state("networkidle")
                        # default: Tree
                        self.assertTrue(page.is_visible("#cj-data-tree-panel"))
                        self.assertFalse(page.is_visible("#cj-data-graph-panel"))
                        self.assertEqual(page.get_attribute('[data-view="tree"]', "aria-pressed"), "true")
                        page.screenshot(path=str(self.shots / f"l_tree_{width}_{scheme}.png"), full_page=True)
                        page.click('[data-view="graph"]')
                        page.wait_for_selector(".cj-data-graph-node")
                        page.wait_for_load_state("networkidle")
                        self.assertFalse(page.is_visible("#cj-data-tree-panel"))
                        sw, iw = page.evaluate("[document.documentElement.scrollWidth, window.innerWidth]")
                        self.assertLessEqual(sw, iw, "the page scrolls sideways")
                        names = page.locator(".cj-data-graph-node .cj-data-graph-id").all_text_contents()
                        self.assertTrue({LS, DS_MID, M_MID} <= set(names), names)
                        self.assertTrue({DS_SIB, M_SIB, M_OTHER}.isdisjoint(names), names)
                        self.assertEqual(page.locator(".cj-data-graph-focus").count(), 1)
                        self.assertEqual(page.locator("script:not([src])").count(), 0)
                        self.assertEqual(page.locator("#cj-data-graph-more").is_visible(), False)
                        page.screenshot(path=str(self.shots / f"l_graph_{width}_{scheme}.png"), full_page=True)
                        # remembered: a reload opens the Graph
                        page.reload()
                        page.wait_for_selector(".cj-data-graph-node")
                        self.assertTrue(page.is_visible("#cj-data-graph-panel"))
                        self.assertEqual(page.get_attribute('[data-view="graph"]', "aria-pressed"), "true")
                        page.click('[data-view="tree"]')
                        self.assertTrue(page.is_visible("#cj-data-tree-panel"))
                        page.reload()
                        page.wait_for_load_state("networkidle")
                        self.assertTrue(page.is_visible("#cj-data-tree-panel"))
                    self.assertEqual((rec["foreign"], rec["errors"], rec["console"]), ([], [], []))
                finally:
                    ctx.close()

    def test_blocked_storage_still_works(self):
        ctx, page, rec = self.session(1280, "light", self.adv_key)
        try:
            ctx.add_init_script("Object.defineProperty(window,'localStorage',{get(){throw new Error('blocked')}})")
            page.goto(self.p.origin + "/data/lineage/dataset/" + DS_MID + "/")
            page.wait_for_load_state("networkidle")
            self.assertTrue(page.is_visible("#cj-data-tree-panel"))
            page.click('[data-view="graph"]')
            page.wait_for_selector(".cj-data-graph-node")
            self.assertEqual((rec["foreign"], rec["errors"]), ([], []))
        finally:
            ctx.close()

    def test_part_and_hidden_focus_in_a_browser(self):
        ctx, page, rec = self.session(1280, "light", self.adv_key)
        try:
            page.goto(self.p.origin + "/data/lineage/images/" + TG.OWN_PKG + "/")
            page.click('[data-view="graph"]')
            page.wait_for_selector(".cj-data-graph-node")
            self.assertIn(TG.OWN_PKG, " ".join(page.locator(".cj-data-graph-id").all_text_contents()))
            self.assertNotIn("aaaa11", page.content())
            # a part's package is an analysis ancestor: "Show analyses" starts checked so the focus has its parents
            page.goto(self.p.origin + "/data/lineage/images/" + TG.OWN_PART + "/")
            page.wait_for_selector(".cj-data-graph-node")
            self.assertTrue(page.is_checked("#cj-data-graph-an"))
            self.assertEqual(page.locator(f'.cj-data-graph-node[title$="{TG.OWN_PKG}"]').count(), 1)
            # ... and stays unchecked when no ancestor is an analysis
            page.goto(self.p.origin + "/data/lineage/dataset/" + DS_MID + "/")
            page.wait_for_selector(".cj-data-graph-node")
            self.assertFalse(page.is_checked("#cj-data-graph-an"))
            page.evaluate("localStorage.removeItem('cj-data-lineage-view')")
            # the lab page is unchanged by the new attributes
            page.goto(self.p.origin + "/data/lineage/graph/?focus=" + quote("model|" + TG.MODEL, safe=""))
            page.wait_for_selector(".cj-data-graph-focus")
            self.assertEqual((rec["foreign"], rec["errors"], rec["console"]), ([], [], []))
        finally:
            ctx.close()


if __name__ == "__main__":
    unittest.main()
