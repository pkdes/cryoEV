"""
Round 15 (B3, contract K4) on a real portal, temp folders only (cjtest.Env; never CryoAI-laptop or the real registry):
  - whole-tree object sets on the Reviews hub: progress counts an item done when every assigned reviewer answered
    every required question that applies to every node (root / child), a "whole tree" tag;
  - Close set (#80g): GET /reviews/<id>/close (maker or admin; the amber warning; the hub's 404 for anyone else),
    POST with the typed name: closed / closed_by / closed_t in review_package.json, a registry status line
    "reviewed" (status_by), one "object set closed" audit line, final (a second close: 409), the hub shows it closed
    with status Finished and no Close link; a wrong name or token closes nothing;
  - Start a review -> Object set: "Cards" (one object per card / whole tree per card), tree schemas marked, the
    per-set tree counts (review_build.tree_info) or the reason plus the Results -> Change recipe link; a whole-tree
    POST builds through start_objects with tree=True and no CSV (B2's builder stubbed), refusals for a one-object
    schema and for a set without nesting_tree results;
  - #80b: progress pages say Done when the build is done; #80e: hub rows share one grid.

  python -m unittest portal.tests.test_object_sets_r15 -v
"""
import json
import os
import re
import shutil
import sys
import tempfile
import time
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cjtest as T  # noqa: E402
sys.path.insert(0, str(T.REPO / "annotation" / "tests"))
import object_card_fixture as OF  # noqa: E402
from test_reviews_hub import fake_listing  # noqa: E402
from annotation import multi_review_server as MRS  # noqa: E402
from portal import auth as A  # noqa: E402
from portal import review_build as RB  # noqa: E402
from portal import review_index as RI  # noqa: E402
from portal.modules import reviews as REV  # noqa: E402
from portal.modules import reviewstart as RS  # noqa: E402

TREE = "pred_obj_tree_check_ab12"
ONE = "pred_obj_lamellarity_cd34"
PRED_ID_R14 = re.compile(r"^(?:pred_ps_\d{8}_[0-9a-f]{6}|pred_obj_[a-z0-9][a-z0-9_]{0,40}_[0-9a-f]{4}"
                         r"|lab_[A-Za-z0-9][A-Za-z0-9_.-]{0,100})\Z")
PS_TREE = "ps_20261114_d6d6d6"        # has nesting_tree results
PS_PLAIN = "ps_20261114_a0a0a0"       # results without the tree step
TOKEN_RE = re.compile(r'name="t" value="([0-9a-f]{40})"')


def edit(pkg, file, dec):
    (pkg / "edits").mkdir(exist_ok=True)
    have = pkg / "edits" / f"{file}.json"
    old = json.loads(have.read_text(encoding="utf-8"))["edit"]["dec"] if have.is_file() else {}
    tmp = pkg / "edits" / f".{file}.tmp"
    tmp.write_text(json.dumps({"edit": {"dec": {**old, **dec}}}), encoding="utf-8")
    os.replace(tmp, have)


def write_results(ps_dir: Path, tree_step=True):
    """results/steps.json + objects.csv: image a has root 0 (children 1, 2; 2 holds 3) and a lone root 4; image b has
    root 0 with child 1; image c has nothing nested. -> 2 trees, 6 nodes, 2 images."""
    r = ps_dir / "results"
    r.mkdir(parents=True, exist_ok=True)
    recipe = [{"step": "roles", "version": 1}, {"step": "nesting", "version": 1}]
    if tree_step:
        recipe.append({"step": "nesting_tree", "version": 1})
    (r / "steps.json").write_text(json.dumps({"schema": "steps/1", "recipe": recipe}), encoding="utf-8")
    rows = [("a", 0, "", 0), ("a", 1, 0, 1), ("a", 2, 0, 1), ("a", 3, 2, 2), ("a", 4, "", 0), ("a", 9, "", ""),
            ("b", 0, "", 0), ("b", 1, 0, 1), ("c", 0, "", 0)]
    lines = ["image,line,pkg_key,cls,conf,included,exclude_reason,nesting.parent_line,nesting.depth"]
    lines += [f"{im},{ln},g{ln},0,0.9,1,,{par},{dep}" for im, ln, par, dep in rows]
    (r / "objects.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


class TreeInfoUnit(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="cj_treeinfo_"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_counts_and_reasons(self):
        ps = self.tmp / "ps_20261114_111111"
        self.assertEqual(RB.tree_info(ps)["ok"], False)                                  # no results at all
        write_results(ps)
        self.assertEqual(RB.tree_info(ps), {"ok": True, "why": "", "trees": 2, "nodes": 6, "images": 2})
        plain = self.tmp / "ps_20261114_222222"
        write_results(plain, tree_step=False)
        t = RB.tree_info(plain)
        self.assertFalse(t["ok"])
        self.assertIn("nesting tree step", t["why"])

    def test_tree_schema_flag(self):
        self.assertTrue(RB.is_tree_schema(OF.TREE_SCHEMA))
        self.assertFalse(RB.is_tree_schema(OF.SCHEMA))
        self.assertFalse(RB.is_tree_schema(None))

    def test_progress_headings_say_done(self):
        """#80b: the heading says Done once the build is done (it said Building to the end)."""
        page = RS.render_obj_progress({"name": "obj_20261114_120000_abcd", "phase": "done", "pkg": TREE,
                                       "title": "Tree check 1", "n_items": 3, "cards": "tree"})
        self.assertIn("Done: object set built", page)
        self.assertNotIn("Building object set", page)
        self.assertIn("3 trees", page)
        self.assertIn("Building object set", RS.render_obj_progress({"name": "obj_20261114_120000_abcd",
                                                                     "phase": "building", "n_items": 2}))
        self.assertIn("2 items", RS.render_obj_progress({"name": "obj_20261114_120000_abcd", "phase": "building",
                                                        "n_items": 2}))
        self.assertIn("Done: review built", RS.render_progress({"name": "x", "phase": "done", "purpose": "revise"}))

    def test_close_set_is_final(self):
        root = self.tmp / "review"
        pkg = root / TREE
        OF.build_tree(pkg)
        meta = REV.close_set(root, pkg, "maker")
        self.assertEqual((meta["closed"], meta["closed_by"]), (True, "maker"))
        self.assertRegex(meta["closed_t"], r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d[+-]\d\d:\d\d$")   # as object_set.close_set
        saved = json.loads((pkg / "review_package.json").read_text(encoding="utf-8"))
        self.assertEqual(saved, meta)
        self.assertEqual(saved["reviewers"], ["rev1"])                                    # every other key kept
        with self.assertRaises(REV.CloseError):
            REV.close_set(root, pkg, "admin1")
        self.assertEqual(json.loads((pkg / "review_package.json").read_text(encoding="utf-8"))["closed_by"], "maker")
        self.assertEqual([p.name for p in pkg.glob(".review_package.json.*")], [])        # no temp file left

    def test_tree_builds_go_to_build_tree(self):
        """The portal's builder call: tree=True -> object_set.build_tree (B2) without items_csv / tree."""
        from annotation import object_set as OS
        seen = {}
        with mock.patch.object(OS, "build_tree", lambda **kw: seen.setdefault("tree", kw) and "x", create=True),                 mock.patch.object(OS, "build", lambda **kw: seen.setdefault("one", kw) and "y"):
            RB._object_set_build(ps_dir="p", items_csv=None, schema="s", review_root="r", tree=True, slug="a",
                                 owner="o", reviewers=[], registry_root=None, built_by="portal", title="T")
            RB._object_set_build(ps_dir="p", items_csv="c.csv", schema="s", review_root="r", slug="a")
        self.assertEqual(seen["tree"], {"ps_dir": "p", "schema": "s", "review_root": "r", "slug": "a", "owner": "o",
                                        "reviewers": [], "registry_root": None, "built_by": "portal", "title": "T"})
        self.assertEqual(seen["one"]["items_csv"], "c.csv")

    def test_same_name(self):
        self.assertTrue(REV.same_name("  tree   CHECK 1 ", "Tree check 1"))
        self.assertFalse(REV.same_name("Tree check", "Tree check 1"))
        self.assertFalse(REV.same_name("", ""))


class TreeSetsPortal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stack = ExitStack()
        cls.stack.enter_context(T.TK.env_patch(drop=("CJ_*",)))
        fake_listing(cls.stack)
        from portal import app as APP
        if not APP.PRED_ID_RE.match(TREE):
            cls.stack.enter_context(mock.patch.object(APP, "PRED_ID_RE", PRED_ID_R14))
        cls.env = T.Env()
        cls.tmp = Path(tempfile.mkdtemp(prefix="cj_r15b3_"))
        cls.review, cls.predict = cls.tmp / "review", cls.tmp / "predict"
        cls.review.mkdir()
        cls.predict.mkdir()
        cls.tree = cls.review / TREE
        cls.tdata = OF.build_tree(cls.tree)
        meta = json.loads((cls.tree / "review_package.json").read_text(encoding="utf-8"))
        meta.update(owner="maker", reviewers=["rev1", "rev2"])
        meta.pop("mode")                                  # the hub must find the mode in data.js too
        (cls.tree / "review_package.json").write_text(json.dumps(meta), encoding="utf-8")
        (cls.tree / ".review_key").write_text("k" * 20, encoding="utf-8")
        cls.one = cls.review / ONE
        OF.build(cls.one)
        m1 = json.loads((cls.one / "review_package.json").read_text(encoding="utf-8"))
        m1.update(owner="maker", reviewers=["rev1"])
        (cls.one / "review_package.json").write_text(json.dumps(m1), encoding="utf-8")
        (cls.one / ".review_key").write_text("k" * 20, encoding="utf-8")
        for ps, step in ((PS_TREE, True), (PS_PLAIN, False)):
            d = cls.predict / ps
            d.mkdir()
            (d / "request.json").write_text(json.dumps({"owner": T.USER, "name": f"Run {ps[-6:]}"}), encoding="utf-8")
            (d / "status.json").write_text(json.dumps({"state": "done", "n_total": 9}), encoding="utf-8")
            write_results(d, tree_step=step)
        reg = cls.env.registry
        (reg / "defs" / "label_schema").mkdir(parents=True)
        lines = []
        for sch in (OF.SCHEMA, OF.TREE_SCHEMA):
            (reg / "defs" / "label_schema" / f"{sch['id']}.json").write_text(json.dumps(sch), encoding="utf-8")
            lines.append({"schema": "label_schema/1", "id": sch["id"], "maker": "pujan", "created": "2026-11-14T09:00:00",
                          "def": f"defs/label_schema/{sch['id']}.json"})
        T.TK.write_registry(reg, {
            "label_schema": lines,
            "review_package": [{"schema": "review_package/1", "id": TREE, "kind": "objects", "label_schema":
                                "ls_lamellarity_tree_v1", "base": PS_TREE, "base_kind": "prediction_set",
                                "n_items": len(cls.tdata["items"]), "blind_cols": [], "reviewers": ["rev1", "rev2"],
                                "owner": "maker", "built_by": "portal", "status": "built", "n_images": 3,
                                "built": "2026-11-14T10:00:00"}],
        }, samples=False)
        os.environ[MRS.PREFIX_ENV] = "pred_,lab_"
        os.environ["CJ_PREDICT_ROOT"] = str(cls.predict)
        cls.keys = {"admin": cls.env.key}
        for pid, role in (("maker", "advanced"), ("rev1", "basic"), ("rev2", "basic"), ("nobody", "basic")):
            cls.keys[pid] = A.add_key(cls.env.roster, pid, role)
        cfg = cls.env.config()
        cfg.review_root, cfg.predict_root = cls.review, cls.predict
        cls.cfg = cfg
        cls.p = T.Portal(cfg)
        cls.tok = {k: T.login(cls.p.port, v) for k, v in cls.keys.items()}

    @classmethod
    def tearDownClass(cls):
        cls.p.stop()
        cls.env.cleanup()
        shutil.rmtree(cls.tmp, ignore_errors=True)
        cls.stack.close()

    def setUp(self):
        shutil.rmtree(self.tree / "edits", ignore_errors=True)
        self.meta0 = (self.tree / "review_package.json").read_bytes()
        self.reg0 = (self.env.registry / "review_package.jsonl").read_bytes()

    def tearDown(self):
        (self.tree / "review_package.json").write_bytes(self.meta0)
        (self.env.registry / "review_package.jsonl").write_bytes(self.reg0)
        shutil.rmtree(self.tree / "edits", ignore_errors=True)

    def get(self, path, who):
        st, h, body = T.request(self.p.port, "GET", path, T.cookie(self.tok[who]))
        return st, h, body.decode()

    def post(self, path, who, body: bytes, ctype="application/x-www-form-urlencoded"):
        hdr = {**T.cookie(self.tok[who]), "Content-Type": ctype, "Origin": self.p.origin,
               "Content-Length": str(len(body))}
        st, h, out = T.request(self.p.port, "POST", path, hdr, body)
        return st, h, out.decode()

    def answer_tree(self, rid, it, skip=()):
        dec = {}
        for nd in it["nodes"]:
            attrs = ("type",) if nd["n"] == 1 else ("relation",)
            for a in attrs:
                if (nd["n"], a) not in skip:
                    dec[f"a:{nd['obj']}:{a}@{rid}"] = {"v": "ML" if a == "type" else "layer", "t": 1, "who": rid}
        edit(self.tree, f"{it['image']}.png", dec)

    # ------------------------------------------------------------ progress and the hub
    def test_tree_progress_per_node(self):
        items = self.tdata["items"]
        n = len(items)
        self.assertEqual(REV.card_progress(self.tree), (0, n))
        self.assertEqual(REV.object_mode(self.tree), "tree")
        self.assertEqual(REV.object_mode(self.one) if REV.card_progress(self.one) else None, "objects")
        big = max(items, key=lambda it: len(it["nodes"]))
        self.answer_tree("rev1", big)
        self.assertEqual(REV.card_progress(self.tree), (0, n))                 # rev2 has not answered
        self.answer_tree("rev2", big, skip={(len(big["nodes"]), "relation")})  # one child's question missing
        self.assertEqual(REV.card_progress(self.tree), (0, n))
        self.answer_tree("rev2", big)
        self.assertEqual(REV.card_progress(self.tree), (1, n))
        # an optional question (inner_type) or a note never holds an item back, and alone never finishes one
        small = min(items, key=lambda it: len(it["nodes"]))
        edit(self.tree, f"{small['image']}.png", {f"a:{nd['obj']}:note@{r}": {"v": "x", "t": 1, "who": r}
                                                  for nd in small["nodes"] for r in ("rev1", "rev2")})
        self.assertEqual(REV.card_progress(self.tree), (1, n))

    def test_hub_tree_tag_close_link(self):
        st, _, page = self.get("/reviews/?show=all&status=any", "admin")
        self.assertEqual(st, 200)
        self.assertIn('<span class="cj-rv-tag">whole tree</span>', page)
        self.assertIn(f'href="/reviews/{TREE}/close">Close set</a>', page)
        self.assertIn(f'href="/reviews/{ONE}/close">Close set</a>', page)           # every object set can close
        st, _, page = self.get("/reviews/", "maker")
        self.assertIn(f'href="/reviews/{TREE}/close"', page)
        st, _, page = self.get("/reviews/", "rev1")
        self.assertIn("Tree check 1", page)
        self.assertNotIn("/close", page)                                            # a reviewer can't close
        rows = RI.list_reviews(self.cfg, {"id": "admin1", "role": "admin"}, lambda u, o: True, show_all=True)
        row = next(r for r in rows if r["id"] == TREE)
        self.assertEqual((row["tree"], row["closed"]), (True, False))
        self.assertFalse(next(r for r in rows if r["id"] == ONE)["tree"])

    def test_hub_rows_share_one_grid(self):
        """#80e: from 900 px every hub row (the header row too) is a grid with the same tracks."""
        self.assertIn(".cj-rv-row{display:grid;grid-template-columns:", REV.STYLE)
        self.assertIn("6.5rem}", REV.STYLE)

    # ------------------------------------------------------------ Close set
    def close_form(self, who="maker", pkg=TREE):
        st, _, page = self.get(f"/reviews/{pkg}/close", who)
        return st, page

    def test_close_page_access(self):
        st, page = self.close_form("maker")
        self.assertEqual(st, 200)
        self.assertIn("Closing is final. No one can answer after this, and exports then include the hidden columns "
                      "and the rule's calls. Type the set name to confirm.", page)
        self.assertIn('name="confirm"', page)
        self.assertIn('<span class="cj-rv-tag">whole tree</span>', page)
        self.assertEqual(self.close_form("admin")[0], 200)
        for who in ("rev1", "rev2", "nobody"):
            self.assertEqual(self.close_form(who)[0], 404, who)
        self.assertEqual(self.get("/reviews/pred_obj_nope_0000/close", "admin")[0], 404)

    def test_close_writes_meta_registry_audit_and_is_final(self):
        st, page = self.close_form("maker")
        tok = TOKEN_RE.search(page).group(1)
        # a wrong name or token: nothing closed
        st, _, out = self.post(f"/reviews/{TREE}/close", "maker", f"t={tok}&confirm=Tree+check".encode())
        self.assertEqual(st, 400)
        self.assertIn("does not match", out)
        st, _, _ = self.post(f"/reviews/{TREE}/close", "maker", f"t={'0' * 40}&confirm=Tree+check+1".encode())
        self.assertEqual(st, 403)
        st, _, _ = self.post(f"/reviews/{TREE}/close", "rev1", f"t={tok}&confirm=Tree+check+1".encode())
        self.assertEqual(st, 404)
        self.assertNotIn("closed", json.loads((self.tree / "review_package.json").read_text(encoding="utf-8")))
        from registry import store
        with mock.patch.object(store, "roster", lambda path=None: {"maker", "pujan"}):
            st, h, _ = self.post(f"/reviews/{TREE}/close", "maker", f"t={tok}&confirm=tree%20check%201".encode())
        self.assertEqual(st, 303)
        self.assertTrue(T.hget(h, "location").startswith("/reviews/?f="))
        meta = json.loads((self.tree / "review_package.json").read_text(encoding="utf-8"))
        self.assertEqual((meta["closed"], meta["closed_by"]), (True, "maker"))
        self.assertIn("closed_t", meta)
        self.assertEqual(meta["reviewers"], ["rev1", "rev2"])
        rec = store.current("review_package", self.env.registry)[TREE]
        self.assertEqual((rec["status"], rec["status_by"]), ("reviewed", "maker"))
        self.assertNotIn("status_via", rec)
        from portal import audit as PA
        lines = [json.loads(x) for x in PA.audit_path(self.env.state).read_text(encoding="utf-8").splitlines() if x]
        self.assertTrue(any(x.get("action") == "object set closed" and x.get("target") == TREE
                            and x.get("detail") == {"registry": "written"} for x in lines))
        # final: the page says who closed it, a second close is refused, the hub shows it closed
        st, page = self.close_form("admin")
        self.assertIn("Closed by maker", page)
        self.assertNotIn('name="confirm"', page)
        atok = TOKEN_RE.search(self.get(f"/reviews/{TREE}/reviewers", "admin")[2]).group(1)   # the module's token
        st, _, _ = self.post(f"/reviews/{TREE}/close", "admin", f"t={atok}&confirm=Tree+check+1".encode())
        self.assertEqual(st, 409)
        st, _, hub = self.get("/reviews/?show=all&status=any", "admin")
        self.assertNotIn(f'href="/reviews/{TREE}/close"', hub)
        self.assertIn("cj-rv-closed", hub)
        row = next(r for r in RI.list_reviews(self.cfg, {"id": "admin1", "role": "admin"}, lambda u, o: True,
                                              show_all=True) if r["id"] == TREE)
        self.assertEqual((row["closed"], row["status"], row["needs_you"]), (True, "finished", False))

    def test_close_without_people_entry_still_closes(self):
        """The closer is not in people.json: the set is closed, the registry line is skipped and the flash says so."""
        st, page = self.close_form("maker")
        tok = TOKEN_RE.search(page).group(1)
        from registry import store
        with mock.patch.object(store, "roster", lambda path=None: {"pujan"}):
            st, _, _ = self.post(f"/reviews/{TREE}/close", "maker", f"t={tok}&confirm=Tree+check+1".encode())
        self.assertEqual(st, 303)
        self.assertTrue(json.loads((self.tree / "review_package.json").read_text(encoding="utf-8"))["closed"])
        self.assertEqual(store.current("review_package", self.env.registry)[TREE]["status"], "built")
        from portal import audit as PA
        lines = [json.loads(x) for x in PA.audit_path(self.env.state).read_text(encoding="utf-8").splitlines() if x]
        self.assertTrue(any(x.get("action") == "object set closed" and x.get("detail") == {"registry":
                                                                                          "not in people.json"}
                            for x in lines))

    # ------------------------------------------------------------ Start a review -> whole tree per card
    def start_form(self):
        st, _, page = self.get("/reviews/start/", "admin")
        self.assertEqual(st, 200)
        return page

    def test_start_form_cards_and_tree_counts(self):
        page = self.start_form()
        self.assertIn('<input type="radio" name="cards" value="objects" checked> One object per card', page)
        self.assertIn('<input type="radio" name="cards" value="tree"> Whole tree per card', page)
        self.assertIn('(from the nesting tree step)', page)
        self.assertIn('<option value="ls_lamellarity_tree_v1" data-tree="1">', page)
        self.assertIn('<option value="ls_lamellarity_v1">', page)
        blob = re.search(r'<script type="application/json" id="rs-obj-trees">(.*?)</script>', page).group(1)
        trees = json.loads(blob)
        self.assertEqual(trees[PS_TREE], {"ok": True, "why": "", "trees": 2, "nodes": 6, "images": 2})
        self.assertFalse(trees[PS_PLAIN]["ok"])
        self.assertIn('id="rs-obj-tree-note"', page)
        self.assertIn('class="rs-only-objects"', page)

    def multipart(self, fields):
        b = "----cjtestboundary15"
        out = bytearray()
        for k, v in fields:
            out += f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
        out += f"--{b}--\r\n".encode()
        return f"multipart/form-data; boundary={b}", bytes(out)

    def post_tree(self, fields, who="admin"):
        tok = TOKEN_RE.search(self.start_form()).group(1)
        ctype, body = self.multipart([("t", tok), ("cards", "tree")] + list(fields))
        return self.post("/reviews/start/objects", who, body, ctype)

    def test_post_whole_tree_builds_with_tree_true(self):
        got = []

        def fake_build(**kw):
            got.append(kw)
            return Path(kw["review_root"]) / "pred_obj_tree_check_2_beef"
        base = [("name", "Tree check 2"), ("schema", "ls_lamellarity_tree_v1"), ("ps", PS_TREE),
                ("reviewer", "rev1"), ("reviewer", "rev2")]
        with mock.patch.object(RB, "_object_set_build", fake_build), \
                mock.patch.object(RB, "_object_set_register", lambda pkg, reg: None):
            st, h, _ = self.post_tree(base)
            self.assertEqual(st, 303)
            loc = T.hget(h, "location")
            job = loc.rsplit("/", 1)[1]
            for _ in range(200):
                jst = RB.read_obj_status(self.cfg, job)
                if jst and jst["phase"] in ("done", "failed"):
                    break
                time.sleep(0.05)
            self.assertEqual((jst["phase"], jst["pkg"], jst["n_items"], jst["cards"]),
                             ("done", "pred_obj_tree_check_2_beef", 2, "tree"))
            kw = got[-1]
            self.assertEqual((kw["tree"], kw["items_csv"], kw["schema"], kw["ps_dir"], kw["reviewers"], kw["slug"],
                              kw["title"], kw["built_by"]),
                             (True, None, "ls_lamellarity_tree_v1", self.predict / PS_TREE, ["rev1", "rev2"],
                              "tree_check_2", "Tree check 2", "portal"))
            self.assertNotIn("blind_cols", kw)
            st, _, page = self.get(loc, "admin")
            self.assertIn("Done: object set built", page)
            self.assertIn("2 trees", page)
            n = len(got)
            for fields, why in (([x for x in base if x[0] != "schema"] + [("schema", "ls_lamellarity_v1")],
                                 "whole-tree label schema"),
                                ([x for x in base if x[0] != "ps"] + [("ps", PS_PLAIN)], "Change recipe")):
                st, _, page = self.post_tree(fields)
                self.assertEqual(st, 400, why)
                self.assertIn(why, page)
            self.assertEqual(len(got), n)
        from portal import audit as PA
        lines = [json.loads(x) for x in PA.audit_path(self.env.state).read_text(encoding="utf-8").splitlines() if x]
        self.assertTrue(any(x.get("action") == "object set built" and (x.get("detail") or {}).get("cards") == "tree"
                            and x["detail"].get("items") == 2 for x in lines))


if __name__ == "__main__":
    unittest.main()
