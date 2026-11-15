"""Round 15 (contracts K2 / K3, #80f / #80g): schema help + applies_to, ls_lamellarity_v2 and the tree schema,
whole-tree object sets (build from a hand-written nesting_tree objects.csv, server per node, progress, derived trees,
export / import, close), and the blind check's named-field exemptions (with the sentinel)."""
import contextlib
import csv
import hashlib
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
from annotation import object_meta as om  # noqa: E402
from annotation import object_set as OS  # noqa: E402
from annotation import prediction_package as pp  # noqa: E402
from annotation import review_server as rs  # noqa: E402
from labels import schemas as S  # noqa: E402
from test_object_sets import KEY, MultiLive, dec, golden_ps, items_csv, put  # noqa: E402
from test_review_server import load_data  # noqa: E402

UNIT_SENT, CALL_SENT, OFF_SENT, SHAPE_SENT = "UNITQZX", "MV[ML2,s]QZX", "0.987654", "0.876543"
NEST = ["n_contained", "n_direct_children", "d_outer_nm", "children", "parent_line", "depth"]
TREE = list(OS.TREE_COLS)
COLS = (["image", "line", "pkg_key", "cls", "conf", "included", "exclude_reason"] + [f"nesting.{c}" for c in NEST]
        + [f"nesting_tree.{c}" for c in TREE])
# (image, line, parent_line, role, vesicle_depth): 0001 0 > 1 > 3 (2 is a root without children; 4 is no EV)
FIXTURE = [("0001", 0, None, "outer", 0), ("0001", 1, 0, "lamella", 0), ("0001", 2, None, "outer", 0),
           ("0001", 3, 1, "inner_vesicle", 1), ("0001", 4, None, None, None),
           ("0002", 0, None, "outer", 0), ("0002", 1, 0, "inner_vesicle", 1), ("0002", 4, 1, "lamella", 1),
           ("0003", 0, None, "outer", 0), ("0003", 1, 0, "lamella", 0), ("0003", 2, 0, "inner_vesicle", 1)]
TREE_SCHEMA = "ls_lamellarity_tree_v1"


def write_results(ps: Path, fixture=FIXTURE, recipe=("roles", "nesting", "nesting_tree"), extra_rows=()):
    """<ps>/results/objects.csv + steps.json (+ results.json), as results/compute.py writes them (K1 columns)."""
    rows = []
    for img, ln, parent, role, vd in fixture:
        r = {c: "" for c in COLS}
        r.update(image=img, line="" if ln is None else ln, pkg_key="", cls=0, conf=0.9, included=1)
        r["nesting.parent_line"] = "" if parent is None else parent
        if role:
            r.update({"nesting_tree.unit": f"{UNIT_SENT}{img}{ln}", "nesting_tree.role": role,
                      "nesting_tree.vesicle_depth": vd, "nesting_tree.layer_index": 0,
                      "nesting_tree.offset_idx": OFF_SENT, "nesting_tree.shape_idx": SHAPE_SENT,
                      "nesting_tree.wide": 0})
            if role == "outer":
                r["nesting_tree.call"] = CALL_SENT
        rows.append(r)
    rows += list(extra_rows)
    d = ps / "results"
    d.mkdir(exist_ok=True)
    buf = io.StringIO()
    w = csv.DictWriter(buf, COLS, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    (d / "objects.csv").write_text(buf.getvalue(), encoding="utf-8", newline="")
    sha = hashlib.sha256(buf.getvalue().encode("utf-8")).hexdigest()
    (d / "steps.json").write_text(json.dumps({"schema": "steps/1", "recipe": [
        {"step": s, "version": 1, "params": {}, "code_sha": "x"} for s in recipe], "objects_csv_sha256": sha}),
        encoding="utf-8")
    (d / "results.json").write_text("{}\n", encoding="utf-8")
    return ps


def build_tree(root: Path, seed=7, reviewers=("au1", "au2"), ps=None, **kw):
    ps = ps or write_results(golden_ps(root))
    (root / "review").mkdir(exist_ok=True)
    with contextlib.redirect_stdout(io.StringIO()):
        return OS.build_tree(ps, TREE_SCHEMA, root / "review", seed=seed, reviewers=reviewers, owner="pujan", **kw)


def node(data, image, n):
    it = next(i for i in data["items"] if i["image"] == image)
    return next(x["obj"] for x in it["nodes"] if x["n"] == n)


# ---------------------------------------------------------------- K2
class SchemaFields(unittest.TestCase):
    def test_v2_is_v1_plus_help(self):
        v1, v2 = S.load("ls_lamellarity_v1"), S.load("ls_lamellarity_v2")
        strip = [{k: v for k, v in a.items() if k != "help"} for a in v2["object_attrs"]]
        self.assertEqual(strip, v1["object_attrs"])
        self.assertEqual(v2["classes"], v1["classes"])
        self.assertEqual((v2["supersedes"], v2["class_mapping"]), ("ls_lamellarity_v1", {"EV": "EV"}))
        helps = {a["id"]: a.get("help") for a in v2["object_attrs"]}
        self.assertEqual(helps["membranes"], "The outer membrane plus its own concentric layers. One tight layer "
                                             "inside = 2.")
        self.assertEqual(helps["inner"], "Separate vesicles directly inside; each counts as 1 whatever is inside it.")
        self.assertEqual(helps["type"], "Its own level: concentric layers make it ML, separate vesicles inside make "
                                        "it MV.")
        page = S.to_page(v2)
        self.assertEqual(page["object_attrs"][1]["help"], helps["membranes"])
        # a legacy schema's page is unchanged: no help / applies_to keys at all
        self.assertTrue(all("help" not in a and "applies_to" not in a for a in S.to_page(v1)["object_attrs"]))

    def test_tree_schema(self):
        doc = S.load(TREE_SCHEMA)
        a = {x["id"]: x for x in doc["object_attrs"]}
        self.assertEqual([x["id"] for x in doc["object_attrs"]], ["type", "relation", "inner_type", "note"])
        self.assertEqual((a["type"]["applies_to"], a["type"]["required"]), ("root", True))
        self.assertEqual([o["v"] for o in a["type"]["options"]], ["single", "ML", "MV_vesicular", "ML+MV", "unsure"])
        self.assertEqual([o["key"] for o in a["type"]["options"]], ["1", "2", "3", "4", "0"])
        self.assertEqual((a["relation"]["applies_to"], a["relation"]["required"]), ("child", True))
        # r15 integration wording: the page header says "Inside n", so the question is "What is it?"; type's MV
        # option reads "MV" with the help line saying multivesicular (values unchanged)
        self.assertEqual(a["relation"]["label"], "What is it?")
        self.assertEqual([o["label"] for o in a["type"]["options"]], ["Single", "ML", "MV", "ML+MV", "Unsure"])
        self.assertEqual([o["label"] for o in a["inner_type"]["options"]], ["Single", "ML", "MV", "ML+MV", "Unsure"])
        self.assertIn("MV (multivesicular)", a["type"]["help"])
        self.assertIn("multivesicular", a["inner_type"]["help"])
        self.assertEqual([(o["label"], o["key"]) for o in a["relation"]["options"]],
                         [("Layer of parent", "l"), ("Inner vesicle", "v"), ("Not a membrane", "x"),
                          ("Duplicate outline", "d")])
        self.assertEqual((a["inner_type"]["label"], a["inner_type"]["applies_to"], a["inner_type"]["required"]),
                         ("If inner vesicle: its type", "child", False))
        self.assertTrue(all("key" not in o for o in a["inner_type"]["options"]))
        self.assertEqual([o["v"] for o in a["inner_type"]["options"]], [o["v"] for o in a["type"]["options"]])
        self.assertEqual((a["note"]["applies_to"], a["note"]["key"], a["note"]["kind"]), ("all", "n", "text"))
        self.assertTrue(S.applies(a["type"], "root") and not S.applies(a["type"], "child"))
        self.assertTrue(S.applies(a["note"], "child") and S.applies({"id": "x"}, "child"))

    def _bad(self, change, expect):
        doc = json.loads(json.dumps(S.load("ls_lamellarity_v2")))
        change(doc)
        probs = S.validate(doc)
        self.assertTrue(any(expect in p for p in probs), probs)

    def test_refused(self):
        self._bad(lambda d: d["object_attrs"][0].update(help="x" * 201), "help")
        self._bad(lambda d: d["object_attrs"][0].update(help="a <b> tag"), "help")
        self._bad(lambda d: d["object_attrs"][0].update(help="a & b"), "help")
        self._bad(lambda d: d["object_attrs"][0].update(help=""), "help")
        self._bad(lambda d: d["object_attrs"][0].update(help="line\nbreak"), "help")
        self._bad(lambda d: d["object_attrs"][0].update(applies_to="leaf"), "applies_to")
        self._bad(lambda d: d["object_attrs"][3].update(key="Tab"), "reserved")
        doc = json.loads(json.dumps(S.load("ls_lamellarity_v2")))
        doc["object_attrs"][0]["help"] = "x" * 200
        self.assertEqual(S.validate(doc), [])


# ---------------------------------------------------------------- K3 build
class TreeBuild(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="tree_"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_shape_numbering_blind(self):
        pkg = build_tree(self.tmp)
        d = load_data(pkg)
        self.assertEqual((d["kind"], d["mode"], d["outline_mode"], d["blind"]), ("objects", "tree", "tree", True))
        self.assertEqual(d["schema"]["id"], TREE_SCHEMA)
        self.assertEqual(sorted(i["image"] for i in d["items"]), ["0001", "0002", "0003"])   # 0001 line 2: no child
        items = OS.read_items(pkg)
        self.assertEqual(items["mode"], "tree")
        for it, pub in zip(items["items"], d["items"]):
            self.assertEqual(set(pub), OS.TREE_ITEM_KEYS)
            self.assertEqual(pub["obj"], pub["nodes"][0]["obj"])
            self.assertEqual([n["n"] for n in pub["nodes"]], [1, 2, 3])
            self.assertIsNone(pub["nodes"][0]["parent_n"])
            for n in pub["nodes"]:
                self.assertEqual(set(n), {"obj", "n", "parent_n"})
            # items.json nodes: {key, n, line, parent_n, depth, row}; parents geometric (parent_line)
            by_line = {nd["line"]: nd for nd in it["nodes"]}
            for nd in it["nodes"]:
                pl = nd["row"]["nesting.parent_line"]
                self.assertEqual(nd["parent_n"], None if pl == "" else by_line[int(pl)]["n"])
            self.assertEqual([(nd["key"], nd["n"], nd["parent_n"]) for nd in it["nodes"]],
                             [(n["obj"], n["n"], n["parent_n"]) for n in pub["nodes"]])
        t1 = next(i for i in items["items"] if i["image"] == "0001")
        self.assertEqual(sorted((nd["line"], nd["depth"]) for nd in t1["nodes"]), [(0, 0), (1, 1), (3, 2)])
        # every node is one gt outline of its image, drawn once
        gt = {(Path(im["file"]).stem, f"g{g['id']}") for im in d["images"] for g in im["gt"]}
        nodes = [(i["image"], n["obj"]) for i in d["items"] for n in i["nodes"]]
        self.assertEqual(len(nodes), len(set(nodes)))
        self.assertEqual(set(nodes), gt)
        # blind: no unit, call or index value anywhere but items.json; no role / step column in data.js's items
        # (the role words are also schema vocabulary: "inner_vesicle" is a relation option, the same for every card)
        words = [UNIT_SENT, CALL_SENT, OFF_SENT, SHAPE_SENT, "MV["]
        for p in pkg.rglob("*"):
            if p.is_file() and p.name != OS.ITEMS:
                raw = p.read_bytes()
                self.assertEqual([w for w in words if w.encode() in raw], [], p.name)
        self.assertNotIn(b"nesting_tree", (pkg / "data.js").read_bytes())       # not even the column names
        pub = json.dumps(d["items"])
        self.assertEqual([w for w in ("outer", "lamella", "inner", "role", "unit", "call", "depth") if w in pub], [])
        self.assertIn(CALL_SENT, (pkg / OS.ITEMS).read_text(encoding="utf-8"))
        rec = json.loads((pkg / pp.RECORD).read_text(encoding="utf-8"))
        from registry import store
        self.assertEqual(store.check_record(rec, store.schemas()), [])
        self.assertEqual((rec["mode"], rec["n_items"], rec["n_nodes"], rec["outline_mode"]), ("tree", 3, 9, "tree"))
        self.assertIn("nesting_tree.call", rec["blind_cols"])

    def test_seeded_numbering(self):
        a, b = build_tree(self.tmp / "a", seed=3), build_tree(self.tmp / "b", seed=3)
        strip = lambda p: [(i["image"], i["nodes"]) for i in load_data(p)["items"]]  # noqa: E731
        self.assertEqual(strip(a), strip(b))
        many = {json.dumps(strip(build_tree(self.tmp / f"s{s}", seed=s))) for s in range(6)}
        self.assertGreater(len(many), 1)                      # the seed changes the order / numbering

    def test_refusals(self):
        ps = golden_ps(self.tmp)
        with self.assertRaises(OS.ObjectSetError) as cm:
            OS.tree_plan(ps, TREE_SCHEMA)
        self.assertIn("no nesting_tree results", str(cm.exception))
        write_results(ps, recipe=("roles", "nesting"))
        with self.assertRaises(OS.ObjectSetError) as cm:
            OS.tree_plan(ps, TREE_SCHEMA)
        self.assertIn("nesting v1", str(cm.exception))
        write_results(ps)
        with open(ps / "results" / "objects.csv", "a", encoding="utf-8") as fh:
            fh.write("\n")                                     # objects.csv no longer the one steps.json names
        with self.assertRaises(OS.ObjectSetError) as cm:
            OS.tree_plan(ps, TREE_SCHEMA)
        self.assertIn("sha256", str(cm.exception))
        write_results(ps)
        with self.assertRaises(OS.ObjectSetError) as cm:
            OS.tree_plan(ps, "ls_lamellarity_v1")              # no applies_to: not a tree schema
        self.assertIn("tree schema", str(cm.exception))
        with self.assertRaises(OS.ObjectSetError) as cm:
            OS.tree_plan(ps, TREE_SCHEMA, roots=[("0001", 2)])  # a root without children
        self.assertIn("not a unit root", str(cm.exception))
        P = OS.tree_plan(ps, TREE_SCHEMA, roots=[("0003", 0)])
        self.assertEqual(P["keys"], [("0003", 0)])
        write_results(ps, fixture=FIXTURE + [("0003", 7, 0, "lamella", 0)])   # line 7: no polygon
        with self.assertRaises(OS.ObjectSetError) as cm:
            OS.tree_plan(ps, TREE_SCHEMA)
        self.assertIn("not polygons", str(cm.exception))

    def test_no_line_tree_skipped(self):
        """An outline drawn on a reviewed image (no prediction line) inside 0003's tree: that tree is skipped and
        counted, the others build."""
        ps = golden_ps(self.tmp)
        extra = {c: "" for c in COLS}
        extra.update({"image": "0003", "nesting.parent_line": 1, "nesting_tree.role": "lamella",
                      "nesting_tree.vesicle_depth": 0, "nesting_tree.unit": "u"})
        write_results(ps, extra_rows=[extra])
        P = OS.tree_plan(ps, TREE_SCHEMA)
        self.assertEqual(P["skipped_no_line"], [("0003", 0)])
        self.assertEqual(sorted(k[0] for k in P["keys"]), ["0001", "0002"])

    def test_cli_dry_run_and_build(self):
        ps = write_results(golden_ps(self.tmp))
        before = sorted(p.relative_to(self.tmp) for p in self.tmp.rglob("*"))
        with contextlib.redirect_stdout(io.StringIO()) as out:
            rc = OS.main(["build", "--tree", "--ps", str(ps), "--schema", TREE_SCHEMA, "--review-root",
                          str(self.tmp / "rv")])
        self.assertEqual(rc, 0)
        self.assertIn("DRY RUN", out.getvalue())
        self.assertIn("3 whole-tree cards (9 outlines)", out.getvalue())
        self.assertEqual(sorted(p.relative_to(self.tmp) for p in self.tmp.rglob("*")), before)
        roots = items_csv(self.tmp / "roots.csv", [("0002", 0)], cols=("image", "line"))
        (self.tmp / "rv").mkdir()
        with contextlib.redirect_stdout(io.StringIO()):
            rc = OS.main(["build", "--tree", "--ps", str(ps), "--schema", TREE_SCHEMA, "--items", str(roots),
                          "--line-col", "line", "--review-root", str(self.tmp / "rv"), "--seed", "1", "--do"])
        self.assertEqual(rc, 0)
        pkg = next((self.tmp / "rv").glob("pred_obj_lamellarity_tree_*"))
        self.assertEqual([i["image"] for i in load_data(pkg)["items"]], ["0002"])
        with contextlib.redirect_stderr(io.StringIO()):     # legacy build still needs --items and a key column
            self.assertEqual(OS.main(["build", "--ps", str(ps), "--schema", "ls_lamellarity_v1"]), 2)
            self.assertEqual(OS.main(["build", "--tree", "--ps", str(ps), "--schema", TREE_SCHEMA,
                                      "--blind-cols", "x"]), 2)


# ---------------------------------------------------------------- derived trees
class Derive(unittest.TestCase):
    #   1 (root) > 2 > 3 > 5 ; 1 > 4 ; 4 > 6
    NODES = [{"n": 1, "parent_n": None}, {"n": 2, "parent_n": 1}, {"n": 3, "parent_n": 2}, {"n": 4, "parent_n": 1},
             {"n": 5, "parent_n": 3}, {"n": 6, "parent_n": 4}]

    def test_cases(self):
        L, V, X, D = "layer", "inner_vesicle", "not_membrane", "duplicate"
        self.assertIsNone(OS.derive_tree(self.NODES, {2: L, 3: L, 4: V, 5: V}))        # 6 unanswered
        d = OS.derive_tree(self.NODES, {2: L, 3: L, 4: V, 5: V, 6: L})
        self.assertEqual((d["membranes"], d["inner"]), (3, 2))     # 1+2+3 a chain; 4 and 5 inner (5 inside 3)
        self.assertEqual(d["parent"], {1: None, 2: 1, 3: 2, 4: 1, 5: 3, 6: 4})
        d = OS.derive_tree(self.NODES, {2: X, 3: L, 4: V, 5: V, 6: V})
        self.assertEqual(d["parent"], {1: None, 2: None, 3: 1, 4: 1, 5: 3, 6: 4})   # 3 moves up to the root
        self.assertEqual((d["membranes"], d["inner"]), (2, 2))     # 6 is inside an inner vesicle: not counted
        d = OS.derive_tree(self.NODES, {2: D, 3: D, 4: D, 5: L, 6: V})
        self.assertEqual(d["parent"], {1: None, 2: None, 3: None, 4: None, 5: 1, 6: 1})
        self.assertEqual((d["membranes"], d["inner"]), (2, 1))
        d = OS.derive_tree([{"n": 1, "parent_n": None}], {})
        self.assertEqual((d["membranes"], d["inner"], d["parent"]), (1, 0, {1: None}))


# ---------------------------------------------------------------- server, progress, export (K3, #80g)
class TreeServer(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="tree_srv_"))
        self.srv = MultiLive(build_tree(self.tmp))
        self.d = self.srv.data
        self.file = {Path(i["file"]).stem: i["file"] for i in self.d["images"]}

    def tearDown(self):
        self.srv.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def answer(self, image, who, rel, t=1000, type_="ML"):
        """Answer one item: the root's type, each child's relation (rel {n: value})."""
        it = next(i for i in self.d["items"] if i["image"] == image)
        dd = {f"a:{it['nodes'][0]['obj']}:type@{who}": dec(type_, who, t)}
        for nd in it["nodes"][1:]:
            if nd["n"] in rel:
                dd[f"a:{nd['obj']}:relation@{who}"] = dec(rel[nd["n"]], who, t)
        return put(self.srv, self.file[image], dd, t=t)

    def test_nodes_and_applies_to(self):
        root, child = node(self.d, "0001", 1), node(self.d, "0001", 2)
        f = self.file["0001"]
        for d_ in ({f"a:{root}:type@au1": dec("ML", "au1")}, {f"a:{child}:relation@au1": dec("layer", "au1")},
                   {f"a:{child}:inner_type@au1": dec("single", "au1")}, {f"a:{child}:note@au1": dec("x", "au1")},
                   {f"a:{root}:note@au1": dec("y", "au1")}):
            self.assertEqual(put(self.srv, f, d_)[0], 200, d_)
        for d_ in ({f"a:{child}:type@au1": dec("ML", "au1")}, {f"a:{root}:relation@au1": dec("layer", "au1")},
                   {f"a:{root}:inner_type@au1": dec("ML", "au1")}):
            code, body = put(self.srv, f, d_)
            self.assertEqual(code, 400, d_)
            self.assertIn("not asked of this node", body["error"])
        other = node(self.d, "0002", 2)                     # a node, but of another image's tree
        self.assertEqual(put(self.srv, f, {f"a:{other}:relation@au1": dec("layer", "au1")})[0], 400)

    def test_chunked_cap(self):
        """The 64-key cap per PUT stays (the page sends a big tree in chunks): 65 refused, 64 + 1 accepted."""
        it = next(i for i in self.d["items"] if i["image"] == "0003")
        f = self.file["0003"]
        many = {}
        for n in range(65):                                 # 65 distinct pending keys over the same few nodes
            many[f"a:{it['nodes'][n % 3]['obj']}:note@au{n}"] = dec(f"n{n}", "au1")
        code, body = put(self.srv, f, many)
        self.assertEqual(code, 400)
        self.assertIn("at most 64", body["error"])
        self.assertEqual(put(self.srv, f, dict(list(many.items())[:64]))[0], 200)
        self.assertEqual(put(self.srv, f, dict(list(many.items())[64:]))[0], 200)

    def test_progress_export_round_trip_two_raters(self):
        n_of = {im: {n["n"]: n["obj"] for n in next(i for i in self.d["items"] if i["image"] == im)["nodes"]}
                for im in ("0001", "0002", "0003")}
        items = OS.read_items(self.srv.pkg)

        def rel_by_line(image, by_line):
            it = next(i for i in items["items"] if i["image"] == image)
            return {nd["n"]: by_line[nd["line"]] for nd in it["nodes"] if nd["n"] != 1 and nd["line"] in by_line}
        # au1: the step's own reading; au2 disagrees on 0003 and calls 0002's inner one "not a membrane"
        self.assertEqual(self.answer("0001", "au1", rel_by_line("0001", {1: "layer", 3: "inner_vesicle"}))[0], 200)
        self.assertEqual(self.answer("0002", "au1", rel_by_line("0002", {1: "inner_vesicle", 4: "layer"}))[0], 200)
        self.assertEqual(self.answer("0003", "au1", rel_by_line("0003", {1: "layer", 2: "inner_vesicle"}))[0], 200)
        self.assertEqual(self.answer("0002", "au2", rel_by_line("0002", {1: "not_membrane", 4: "layer"}),
                                     type_="single")[0], 200)
        self.assertEqual(self.answer("0003", "au2", rel_by_line("0003", {1: "inner_vesicle"}))[0], 200)   # 1 left
        prog = om.progress(self.srv.pkg)
        self.assertEqual((prog["n_items"], prog["n_done"], prog["by_reviewer"]), (3, 1, {"au1": 3, "au2": 1}))
        self.assertEqual(self.answer("0001", "au2", rel_by_line("0001", {1: "layer", 3: "duplicate"}))[0], 200)
        self.assertEqual(om.progress(self.srv.pkg)["n_done"], 2)                      # 0003: au2 has 1 left
        tab = OS.table(self.srv.pkg)
        self.assertEqual(tab["columns"][:7], OS.TREE_HEAD)
        self.assertNotIn("nesting_tree.call", tab["columns"])                 # open set: still blind
        self.assertEqual(len(tab["rows"]), 9)
        row = {(r["image"], r["line"]): r for r in tab["rows"]}
        r1 = row[("0001", 0)]
        self.assertEqual((r1["node_n"], r1["parent_n"], r1["depth"], r1["type@au1"]), (1, None, 0, "ML"))
        self.assertEqual((r1["derived_membranes@au1"], r1["derived_inner@au1"]), (2, 1))
        self.assertEqual((r1["derived_membranes@au2"], r1["derived_inner@au2"]), (2, 0))   # au2: 3 is a duplicate
        self.assertIsNone(row[("0001", 3)]["derived_parent_n@au2"])
        r2 = row[("0002", 0)]
        self.assertEqual((r2["derived_membranes@au1"], r2["derived_inner@au1"]), (1, 1))
        self.assertEqual((r2["derived_membranes@au2"], r2["derived_inner@au2"]), (2, 0))   # 4 moves up: a layer
        self.assertEqual(row[("0002", 4)]["derived_parent_n@au2"], 1)
        self.assertIsNone(row[("0003", 0)]["derived_membranes@au2"])                        # au2 not done there
        self.assertIs(row[("0002", 1)]["agree_relation"], False)
        self.assertIs(row[("0001", 1)]["agree_relation"], True)
        self.assertIsNone(row[("0001", 0)]["relation@au1"])
        # export -> import -> export identical, through the CLI too
        text = OS.to_csv(tab)
        back = OS.import_csv(text, OS.read_schema(self.srv.pkg), items)
        self.assertEqual(OS.to_csv(back), text)
        out = self.tmp / "tree.csv"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(OS.main(["export", "--pkg", str(self.srv.pkg), "--out", str(out), "--coco",
                                      str(self.tmp / "c.json")]), 0)
            self.assertEqual(OS.main(["import", "--pkg", str(self.srv.pkg), "--csv", str(out)]), 0)
        self.assertEqual(out.read_text(encoding="utf-8"), text)
        coco = json.loads((self.tmp / "c.json").read_text(encoding="utf-8"))
        self.assertEqual(len(coco["annotations"]), 9)
        self.assertNotIn(CALL_SENT, json.dumps(coco))
        # tampered: a derived count, a value on a node it isn't asked of, a node of another item
        hdr = tab["columns"]
        rows = list(csv.reader(io.StringIO(text)))
        i_root = next(k for k, r in enumerate(rows) if r[:2] == [r1["item"], "0001"] and r[2] == "1")
        bad = [list(r) for r in rows]
        bad[i_root][hdr.index("derived_inner@au1")] = "5"
        bad2 = [list(r) for r in rows]
        bad2[i_root][hdr.index("relation@au1")] = "layer"
        bad3 = [list(r) for r in rows]
        bad3[i_root][2] = "9"
        for b in (bad, bad2, bad3):
            buf = io.StringIO()
            csv.writer(buf, lineterminator="\n").writerows(b)
            with self.assertRaises(OS.ObjectSetError):
                OS.import_csv(buf.getvalue(), OS.read_schema(self.srv.pkg), items)
        # #80g close: final, PUTs with answers -> 409, others still save, export unblinds
        with self.assertRaises(OS.ObjectSetError):
            OS.close_set(self.srv.pkg, "Bad Id", do=True)
        self.assertFalse(OS.close_set(self.srv.pkg, "pujan")["written"])        # dry run
        self.assertFalse(OS.is_closed(self.srv.pkg))
        self.assertTrue(OS.close_set(self.srv.pkg, "pujan", do=True)["written"])
        rec = json.loads((self.srv.pkg / pp.RECORD).read_text(encoding="utf-8"))
        self.assertEqual((rec["closed"], rec["closed_by"]), (True, "pujan"))
        from registry import store
        self.assertEqual(store.check_record(rec, store.schemas()), [])
        with self.assertRaises(OS.ObjectSetError):
            OS.close_set(self.srv.pkg, "pujan", do=True)                       # final
        code, body = self.answer("0003", "au2", rel_by_line("0003", {1: "inner_vesicle", 2: "layer"}))
        self.assertEqual((code, body), (409, {"error": "set closed"}))
        self.assertEqual(put(self.srv, self.file["0003"], {"i:note": dec("still fine", "au2")})[0], 200)
        k0 = f"a:{node(self.d, '0001', 1)}:type@au1"          # an echo of a saved answer (not pending) + a note
        saved = json.loads((self.srv.pkg / "edits" / f"{self.file['0001']}.json").read_text(encoding="utf-8"))
        code, body = put(self.srv, self.file["0001"], {k0: saved["edit"]["dec"][k0], "i:note": dec("n", "au1")})
        self.assertEqual(code, 200, body)
        self.assertEqual(body["edit"]["dec"][k0], saved["edit"]["dec"][k0])
        self.assertEqual(self.srv.request("GET", "/api/edits")[0], 200)
        tab2 = OS.table(self.srv.pkg)
        self.assertEqual(tab2["columns"][-len(OS.read_items(self.srv.pkg)["blind_cols"]):],
                         OS.read_items(self.srv.pkg)["blind_cols"])
        self.assertEqual(next(r for r in tab2["rows"] if r["node_n"] == 1 and r["image"] == "0001")
                         ["nesting_tree.call"], CALL_SENT)
        t2 = OS.to_csv(tab2)
        self.assertEqual(OS.to_csv(OS.import_csv(t2, OS.read_schema(self.srv.pkg), items)), t2)

    def test_blind_http(self):
        self.assertEqual(self.answer("0001", "au1", {2: "layer", 3: "layer"})[0], 200)
        bodies = {}
        for path in ("/", "/data.js", "/api/ping", "/api/meta", "/api/edits", "/api/groups", "/items.json",
                     *(f"/{im['src']}" for im in self.d["images"])):
            code, body, r = self.srv.request("GET", path)
            bodies[path] = body if isinstance(body, bytes) else json.dumps(body).encode()
        for path, b in bodies.items():
            for w in (UNIT_SENT, CALL_SENT, OFF_SENT, SHAPE_SENT, "nesting_tree", "lamella\""):
                self.assertNotIn(w.encode(), b, (path, w))


class BigTree(unittest.TestCase):
    """A 31-outline tree: built, and all of one reviewer's answers saved in chunks of <= 64, then done."""

    def test_31_nodes(self):
        tmp = Path(tempfile.mkdtemp(prefix="tree_big_"))
        try:
            ps = golden_ps(tmp)
            pred = next((ps / "p").iterdir()) / "predictions" / "0003.txt"
            lines = []
            for k in range(31):
                c, r = 0.5, 0.45 - k * 0.014
                lines.append("0 0.9 " + " ".join(f"{x:.5f}" for x in (c - r, c - r, c + r, c - r, c + r, c + r,
                                                                       c - r, c + r)))
            pred.write_text("\n".join(lines) + "\n", encoding="utf-8")
            fx = [("0003", 0, None, "outer", 0)] + [("0003", k, k - 1, "lamella", 0) for k in range(1, 31)]
            write_results(ps, fixture=fx)
            srv = MultiLive(build_tree(tmp, ps=ps, reviewers=("au1",)))
            try:
                it = srv.data["items"][0]
                self.assertEqual(len(it["nodes"]), 31)
                f = srv.data["images"][0]["file"]
                keys = {f"a:{it['nodes'][0]['obj']}:type@au1": dec("ML", "au1")}
                for nd in it["nodes"][1:]:
                    keys[f"a:{nd['obj']}:relation@au1"] = dec("layer", "au1")
                    keys[f"a:{nd['obj']}:inner_type@au1"] = dec("single", "au1")
                self.assertEqual(len(keys), 61)
                keys.update({f"a:{nd['obj']}:note@au1": dec("n", "au1") for nd in it["nodes"][:10]})
                kv = list(keys.items())
                self.assertEqual(put(srv, f, dict(kv))[0], 400)                 # 71 in one PUT
                for i in range(0, len(kv), 64):
                    self.assertEqual(put(srv, f, dict(kv[i:i + 64]))[0], 200)
                self.assertEqual(om.progress(srv.pkg)["n_done"], 1)
                row = next(r for r in OS.table(srv.pkg)["rows"] if r["node_n"] == 1)
                self.assertEqual((row["derived_membranes@au1"], row["derived_inner@au1"]), (31, 0))
            finally:
                srv.close()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------- #80f
class BlindExemptions(unittest.TestCase):
    """Only NAMED calibration / acquisition fields of /api/meta are exempt; a shown count equal to a blind value is
    still refused (the sentinel)."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="blind80f_"))
        self.ps = golden_ps(self.tmp)
        (self.tmp / "review").mkdir()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def build(self, val, n):
        from annotation import review_meta
        real = review_meta.package_meta

        def fake(pkg):
            m = real(pkg)
            for im in m["images"].values():
                im.update(nm_per_px=0.3681, defocus_um=-2.5, mag=11000, magnification=11000, pixel_A=3.681,
                          tile={"row": 41, "col": 42, "piece": 43}, n_pieces=7, grid=9)
            m["package"]["session_acquisition"] = {"defocus_um": -2.5, "target_defocus_um": -3.0,
                                                   "magnification": 11000, "pixel_size_a": 3.681, "dose": 61}
            return m
        src = items_csv(self.tmp / f"b{n}.csv", [("0001", 0, 1, val)])
        with mock.patch.object(review_meta, "package_meta", fake), contextlib.redirect_stdout(io.StringIO()):
            return OS.build(self.ps, src, "ls_lamellarity_v1", self.tmp / "review", line_col="line",
                            blind_cols=("secret",), seed=1, owner="pujan")

    def test_named_exempt_counts_refused(self):
        for n, val in enumerate(("0.3681", "-2.5", "11000", "3.681", "41", "42", "43", "-3.0")):
            self.assertTrue(self.build(val, n).is_dir(), val)          # named fields: exempt
        for n, val in enumerate(("7", "9", "61"), 20):
            with self.assertRaises(OS.ObjectSetError, msg=val) as cm:
                self.build(val, n)                                     # SENTINEL: shown counts stay compared
            self.assertIn("shows anyway", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
