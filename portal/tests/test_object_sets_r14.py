"""
Round 14 (B4) on a real portal, temp folders only (cjtest.Env; never CryoAI-laptop or the real registry):
  - Reviews hub: an object set (pred_obj_*, kind "objects") is its own kind ("Object set" chip, "Object sets" tab),
    its progress counts ITEMS done by every reviewer (a: decisions in edits/), and other reviewers' partial answers
    don't count; the maker / an admin get "Export CSV";
  - GET /reviews/<id>/export.csv|json: admin and maker while open, a reviewer only once closed, everyone else and
    every non-object package the hub's 404; one "object set exported" audit line (B3's exporter is stubbed here);
  - Start a review -> Object set: the admin's form (schema, prediction set, CSV, blind columns, outlines,
    reviewers); POST /reviews/start/objects builds through review_build.start_objects (B3's builder stubbed), blinds
    every non-key column when no script ran, refuses non-admins, a bad token and bad input;
  - titles.SKIP_KINDS and the Apply page leave object sets out;
  - Data: /data/classmap/<cm> and /data/schema/<schema> (read only), the dataset page's class map link (only when
    class_map_id is recorded), the object set's image-set page and lineage (schema, reviewers);
  - registry/history_page.py: label_schema records get a "Label schemas" section; an object set links its schema.

  python -m unittest portal.tests.test_object_sets_r14 -v
"""
import json
import os
import re
import shutil
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cjtest as T  # noqa: E402
sys.path.insert(0, str(T.REPO / "annotation" / "tests"))
import make_fixture  # noqa: E402
import object_card_fixture as OF  # noqa: E402
from test_reviews_hub import fake_listing  # noqa: E402
from annotation import multi_review_server as MRS  # noqa: E402
from portal import auth as A  # noqa: E402
from portal import review_build as RB  # noqa: E402
from portal import review_index as RI  # noqa: E402
from portal import titles as TT  # noqa: E402
from portal.modules import applyreview as AR  # noqa: E402
from portal.modules import reviews as REV  # noqa: E402
from portal.modules.data import labelmeta as LM  # noqa: E402

OBJ = "pred_obj_lamellarity_ab12"
# portal/app.py PRED_ID_RE (PM-owned) must also take object sets for /review/<id>/ and the hub; until the PM's merge
# this test runs with the proposed rule (the report's snippet). After the merge APP.PRED_ID_RE already matches.
PRED_ID_R14 = re.compile(r"^(?:pred_ps_\d{8}_[0-9a-f]{6}|pred_obj_[a-z0-9][a-z0-9_]{0,40}_[0-9a-f]{4}"
                         r"|lab_[A-Za-z0-9][A-Za-z0-9_.-]{0,100})\Z")
PRED = "pred_ps_20261112_aaaaa1"
PS = "ps_20261114_abcdef"
PS_OTHER = "ps_20261114_0f0e0d"
TOKEN_RE = re.compile(r'name="t" value="([0-9a-f]{40})"')
CM = {"schema": "class_map/1", "id": "cm_ev_nonev_v1", "name": "ev_nonev", "version": 1, "maker": "builtin",
      "description": "EV and non-EV.", "source_schema": "ls_legacy_v1", "classes": ["EV", "non-EV"],
      "map": {"EV": "EV", "Low conf EV": "EV", "MV": "EV", "non-EV": "non-EV", "VLDL": "non-EV", "inner": None},
      "ev_classes": ["EV"], "supersedes": None, "rename": None}
CM2 = {**CM, "id": "cm_ev_nonev_v2", "version": 2, "supersedes": "cm_ev_nonev_v1",
       "map": {**CM["map"], "inner": "EV <b>"}}


def edit(pkg, file, dec):
    """As the review server writes a record: a temp file + os.replace (the edits folder's mtime changes)."""
    (pkg / "edits").mkdir(exist_ok=True)
    tmp = pkg / "edits" / f".{file}.tmp"
    tmp.write_text(json.dumps({"edit": {"dec": dec}}), encoding="utf-8")
    os.replace(tmp, pkg / "edits" / f"{file}.json")


def multipart(fields, files):
    b = "----cjtestboundary7"
    out = bytearray()
    for k, v in fields:
        out += f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
    for k, name, data in files:
        out += (f'--{b}\r\nContent-Disposition: form-data; name="{k}"; filename="{name}"\r\n'
                f"Content-Type: text/csv\r\n\r\n").encode() + data + b"\r\n"
    out += f"--{b}--\r\n".encode()
    return f"multipart/form-data; boundary={b}", bytes(out)


class ObjectSetsPortal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stack = ExitStack()
        cls.stack.enter_context(T.TK.env_patch(drop=("CJ_*",)))
        fake_listing(cls.stack)
        from portal import app as APP
        if not APP.PRED_ID_RE.match(OBJ):
            cls.stack.enter_context(mock.patch.object(APP, "PRED_ID_RE", PRED_ID_R14))
        cls.env = T.Env()
        cls.tmp = Path(tempfile.mkdtemp(prefix="cj_r14b4_"))
        cls.review, cls.predict = cls.tmp / "review", cls.tmp / "predict"
        cls.review.mkdir()
        cls.predict.mkdir()
        cls.obj = cls.review / OBJ
        cls.data = OF.build(cls.obj)
        meta = json.loads((cls.obj / "review_package.json").read_text(encoding="utf-8"))
        meta.update(owner="maker", reviewers=["rev1", "rev2"])
        (cls.obj / "review_package.json").write_text(json.dumps(meta), encoding="utf-8")
        (cls.obj / ".review_key").write_text("k" * 20, encoding="utf-8")
        cls.pred = cls.review / PRED
        make_fixture.build(cls.pred)
        (cls.pred / "review_package.json").write_text(json.dumps({"kind": "prediction", "owner": "maker"}), encoding="utf-8")
        (cls.pred / ".review_key").write_text("k" * 20, encoding="utf-8")
        ps = cls.predict / PS
        ps.mkdir()
        (ps / "request.json").write_text(json.dumps({"owner": T.USER, "name": "Poster run", "created": "2026-11-14"}),
                                         encoding="utf-8")
        (ps / "status.json").write_text(json.dumps({"state": "done", "n_total": 677}), encoding="utf-8")
        other = cls.predict / PS_OTHER                       # someone else's set: never offered or accepted
        other.mkdir()
        (other / "request.json").write_text(json.dumps({"owner": "maker", "name": "Maker run"}), encoding="utf-8")
        (other / "status.json").write_text(json.dumps({"state": "done", "n_total": 3}), encoding="utf-8")
        reg = cls.env.registry
        (reg / "defs" / "label_schema").mkdir(parents=True)
        (reg / "defs" / "label_schema" / "ls_lamellarity_v1.json").write_text(json.dumps(OF.SCHEMA), encoding="utf-8")
        T.TK.write_registry(reg, {
            "label_schema": [{"schema": "label_schema/1", "id": "ls_lamellarity_v1", "maker": "pujan",
                              "created": "2026-11-14T09:00:00", "def": "defs/label_schema/ls_lamellarity_v1.json"}],
            "review_package": [{"schema": "review_package/1", "id": OBJ, "kind": "objects", "label_schema":
                                "ls_lamellarity_v1", "base": PS, "base_kind": "prediction_set", "n_items": 6,
                                "blind_cols": ["d_outer_nm"], "outline_mode": "target", "reviewers": ["rev1", "rev2"],
                                "owner": "maker", "built_by": "portal", "status": "built"}],
            "dataset": [{"schema": "dataset/1", "id": "ds_cm_test", "status": "created", "classes": ["EV", "non-EV"],
                         "class_map_id": "cm_ev_nonev_v1", "class_map_sha256": "0" * 64},
                        {"schema": "dataset/1", "id": "ds_plain", "status": "created", "classes": ["EV"]}],
            "run": [{"schema": "run/1", "id": "run_cm_1", "dataset": "ds_cm_test", "status": "done"}],
        }, samples=False)
        cls.cmdir = cls.tmp / "classmaps"
        cls.cmdir.mkdir()
        for cm in (CM, CM2):
            (cls.cmdir / f"{cm['id']}.json").write_text(json.dumps(cm), encoding="utf-8")
        cls.stack.enter_context(mock.patch.object(LM, "classmap_dirs", lambda: [cls.cmdir]))
        os.environ[MRS.PREFIX_ENV] = "pred_,lab_"
        os.environ["CJ_PREDICT_ROOT"] = str(cls.predict)     # Start a review lists THESE prediction sets only
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
        shutil.rmtree(self.obj / "edits", ignore_errors=True)
        self.meta0 = (self.obj / "review_package.json").read_bytes()

    def tearDown(self):
        (self.obj / "review_package.json").write_bytes(self.meta0)
        shutil.rmtree(self.obj / "edits", ignore_errors=True)

    def get(self, path, who, raw=False):
        st, h, body = T.request(self.p.port, "GET", path, T.cookie(self.tok[who]))
        return st, h, (body if raw else body.decode())

    def answer(self, rid, items, attrs=("type", "membranes", "inner"), value="1"):
        by_file: dict = {}
        for it in items:
            for a in attrs:
                by_file.setdefault(f"{it['image']}.png", {})[f"a:{it['obj']}:{a}@{rid}"] = {"v": value, "t": 1, "who": rid}
        for f, dec in by_file.items():
            have = self.obj / "edits" / f"{f}.json"
            old = json.loads(have.read_text(encoding="utf-8"))["edit"]["dec"] if have.is_file() else {}
            edit(self.obj, f, {**old, **dec})

    # ------------------------------------------------------------ progress and the hub
    def test_progress_counts_items_done_by_every_reviewer(self):
        items = self.data["items"]
        self.assertEqual(REV.card_progress(self.obj), (0, 6))
        self.answer("rev1", items[:3])
        self.assertEqual(REV.card_progress(self.obj), (0, 6))         # rev2 hasn't answered: nothing done by all
        self.answer("rev2", items[:2])
        self.assertEqual(REV.card_progress(self.obj), (2, 6))
        self.answer("rev2", items[2:3], attrs=("type", "membranes"))  # a required attribute missing: not done
        self.assertEqual(REV.card_progress(self.obj), (2, 6))
        edit(self.obj, "imgA.png", {"a:g1:type@rev1": {"v": None}})   # cleared answers don't count (control)
        self.assertLessEqual(REV.card_progress(self.obj)[0], 2)
        # an ordinary prediction package keeps its card count (control: the round-13 rule, package_cards)
        data = json.loads((self.pred / "data.js").read_text(encoding="utf-8").split("=", 1)[1].strip().rstrip(";"))
        self.assertEqual(REV.card_progress(self.pred), (0, sum(len(v) for v in REV.package_cards(data).values())))

    def test_progress_without_reviewers_uses_who_answered(self):
        meta = json.loads(self.meta0)
        meta["reviewers"] = []
        (self.obj / "review_package.json").write_text(json.dumps(meta), encoding="utf-8")
        self.assertEqual(REV.card_progress(self.obj), (0, 6))
        self.answer("rev1", self.data["items"][:4])
        self.assertEqual(REV.card_progress(self.obj), (4, 6))

    def test_kind_and_hub(self):
        self.assertEqual(RI.kind_of(OBJ, {}), "objects")
        self.assertEqual(RI.kind_of("pred_ps_x", {"kind": "objects"}), "objects")
        self.assertEqual(RI.kind_of(PRED, {"kind": "prediction"}), "prediction")
        st, _, page = self.get("/reviews/?show=all&status=any", "admin")
        self.assertEqual(st, 200)
        self.assertIn("cj-rv-k-objects", page)
        self.assertIn(">Object set<", page)
        self.assertIn("Object sets <span", page)
        self.assertIn("0 of 6 items", page)
        self.assertIn("Object set · ls_lamellarity_v1 · 6 items", page)
        self.assertIn(f'href="/reviews/{OBJ}/export.csv"', page)
        st, _, page = self.get("/reviews/", "rev1")
        self.assertIn("Lamellarity check 1", page)
        self.assertNotIn("export.csv", page)                           # a reviewer: no export link
        self.assertNotIn("d_outer_nm", page)
        st, _, page = self.get("/reviews/", "nobody")
        self.assertNotIn(OBJ, page)
        self.assertNotIn("Object sets", page)                          # no tab for a kind the viewer has none of

    # ------------------------------------------------------------ export
    def test_export_access(self):
        calls = []

        def fake(pkg, fmt):
            calls.append((Path(pkg).name, fmt))
            return b"item,image\r\n7f3a,imgA\r\n" if fmt == "csv" else b'{"annotations": []}'
        with mock.patch.object(REV, "export_object_set", fake):
            for who in ("admin", "maker"):
                st, h, body = self.get(f"/reviews/{OBJ}/export.csv", who, raw=True)
                self.assertEqual(st, 200, who)
                self.assertEqual(body, b"item,image\r\n7f3a,imgA\r\n")
                self.assertIn("attachment", T.hget(h, "content-disposition"))
                self.assertEqual(T.hget(h, "cache-control"), "no-store")
            st, h, body = self.get(f"/reviews/{OBJ}/export.json", "admin", raw=True)
            self.assertEqual((st, body), (200, b'{"annotations": []}'))
            for who in ("rev1", "nobody"):
                self.assertEqual(self.get(f"/reviews/{OBJ}/export.csv", who)[0], 404, who)
            self.assertEqual(self.get(f"/reviews/{PRED}/export.csv", "admin")[0], 404)   # not an object set
            self.assertEqual(self.get("/reviews/pred_obj_nope_0000/export.csv", "admin")[0], 404)
            meta = json.loads(self.meta0)
            meta["closed"] = True
            (self.obj / "review_package.json").write_text(json.dumps(meta), encoding="utf-8")
            self.assertEqual(self.get(f"/reviews/{OBJ}/export.csv", "rev1")[0], 200)     # closed: reviewers too
            self.assertEqual(self.get(f"/reviews/{OBJ}/export.csv", "nobody")[0], 404)
        self.assertEqual(calls[:3], [(OBJ, "csv"), (OBJ, "csv"), (OBJ, "coco")])
        from portal import audit as PA
        lines = [json.loads(x) for x in PA.audit_path(self.env.state).read_text(encoding="utf-8").splitlines() if x]
        self.assertTrue(any(x.get("action") == "object set exported" and x.get("target") == OBJ
                            and x.get("detail") == {"format": "csv"} for x in lines))

    def test_closed_rule(self):
        self.assertFalse(REV.object_set_closed({}, {"status": "reviewed"}))            # no status_by: not explicit
        self.assertTrue(REV.object_set_closed({}, {"status": "reviewed", "status_by": "admin1"}))
        self.assertFalse(REV.object_set_closed({}, {"status": "in_review", "status_by": "admin1"}))
        self.assertTrue(REV.object_set_closed({"closed": True}, None))

    # ------------------------------------------------------------ skip lists
    def test_titles_and_apply_skip_object_sets(self):
        self.assertIn("objects", TT.SKIP_KINDS)
        self.assertTrue(AR.is_object_set(self.review, OBJ))
        self.assertFalse(AR.is_object_set(self.review, PRED))
        lab = self.review / "lab_objects_like"
        lab.mkdir()
        try:
            (lab / "review_package.json").write_text(json.dumps({"kind": "objects"}), encoding="utf-8")
            self.assertTrue(AR.is_object_set(self.review, lab.name))
            self.assertTrue(AR.excluded(self.cfg, lab.name))
        finally:
            shutil.rmtree(lab)

    # ------------------------------------------------------------ Start a review -> Object set
    def start_form(self, who="admin"):
        st, _, page = self.get("/reviews/start/", who)
        self.assertEqual(st, 200)
        return page

    def post_objects(self, who, fields, csv=b"id,image,line,d_outer_nm,gap_median_nm\nL1,imgA,0,103.5,9.6\n",
                     token=True):
        tok = TOKEN_RE.search(self.start_form()).group(1) if token is True else token
        ctype, body = multipart([("t", tok)] + list(fields), [("items", "items.csv", csv)] if csv is not None else [])
        hdr = {**T.cookie(self.tok[who]), "Content-Type": ctype, "Origin": self.p.origin,
               "Content-Length": str(len(body))}
        st, h, out = T.request(self.p.port, "POST", "/reviews/start/objects", hdr, body)
        return st, h, out.decode()

    def test_start_page_has_the_object_set_form(self):
        page = self.start_form()
        self.assertIn('href="#rs-objects"', page)
        self.assertIn('action="/reviews/start/objects" enctype="multipart/form-data"', page)
        self.assertIn('<option value="ls_lamellarity_v1">', page)
        self.assertIn(f'<option value="{PS}">Poster run · {PS} · 677 img</option>', page)
        self.assertIn('value="target+contained"', page)
        self.assertNotIn(PS_OTHER, page)
        self.assertIn("Create object set", page)
        self.assertIn("Order shuffled. Exports are for you and admins until the set is closed.", page)
        st, _, basic = self.get("/reviews/start/", "rev1")
        self.assertNotIn("rs-objects", basic)

    def test_post_builds_through_start_objects(self):
        got = []

        def fake_build(**kw):
            got.append(kw)
            return {"id": "pred_obj_lamellarity_check_2_c0de"}
        base = [("name", "Lamellarity check 2"), ("schema", "ls_lamellarity_v1"), ("ps", PS), ("outline", "target"),
                ("reviewer", "rev1"), ("reviewer", "rev2")]
        with mock.patch.object(RB, "_object_set_build", fake_build):
            st, h, _ = self.post_objects("admin", base + [("blind_ui", "1"), ("blind", "d_outer_nm")])
            self.assertEqual(st, 303)
            self.assertTrue(T.hget(h, "location").startswith("/reviews/?kind=objects&f="))
            kw = got[-1]
            self.assertEqual((kw["schema"], kw["ps"], kw["blind_cols"], kw["reviewers"], kw["outline_mode"],
                              kw["owner"], kw["name"], kw["built_by"]),
                             ("ls_lamellarity_v1", PS, ["d_outer_nm"], ["rev1", "rev2"], "target", T.USER,
                              "lamellarity_check_2", "portal"))
            self.assertEqual(kw["items"].read_bytes(), b"id,image,line,d_outer_nm,gap_median_nm\nL1,imgA,0,103.5,9.6\n")
            self.assertFalse(str(kw["items"]).startswith(str(self.review)))      # the CSV never goes under review/
            st, _, _ = self.post_objects("admin", base)                           # no script ran: every non-key column
            self.assertEqual(st, 303)
            self.assertEqual(got[-1]["blind_cols"], ["d_outer_nm", "gap_median_nm"])
            n = len(got)
            for fields, csv, why in ((base + [("blind_ui", "1"), ("blind", "image")], None, "a key column"),
                                     ([x for x in base if x[0] != "schema"] + [("schema", "ls_nope_v1")], None, "schema"),
                                     ([x for x in base if x[0] != "ps"] + [("ps", "ps_20990101_000000")], None, "ps"),
                                     ([x for x in base if x[0] != "ps"] + [("ps", PS_OTHER)], None, "not own ps"),
                                     (base + [("reviewer", "ghost")], None, "roster"),
                                     (base, b"just one line\n", "csv"),
                                     (base, b"a,b\n1,2\n", "no image column")):
                st, _, page = self.post_objects("admin", fields, **({"csv": csv} if csv is not None else {}))
                self.assertEqual(st, 400, why)
                self.assertIn("Not built:", page, why)
            self.assertEqual(len(got), n)                                          # nothing built on a refusal
            st, _, _ = self.post_objects("admin", base, token="0" * 40)
            self.assertEqual(st, 403)
            st, _, _ = self.post_objects("maker", base, token="0" * 40)
            self.assertEqual(st, 403)
            self.assertEqual(len(got), n)

    def test_builder_errors_never_show_paths(self):
        def boom(**kw):
            raise RuntimeError(r"refused: C:\Users\someone\CryoAI-laptop\predict\x missing /home/u/a/b")
        with self.assertRaises(RB.BuildError) as cm, mock.patch.object(RB, "_object_set_build", boom):
            RB.start_objects(self.cfg, T.USER, name="x", schema="ls_lamellarity_v1", ps=PS,
                             items_csv=b"image,line\nimgA,0\n", schemas=[{"id": "ls_lamellarity_v1"}],
                             sets=[{"id": PS}])
        self.assertNotIn("Users", str(cm.exception))
        self.assertNotIn("/home/u", str(cm.exception))

    # ------------------------------------------------------------ Data pages
    def test_classmap_page(self):
        st, _, page = self.get("/data/classmap/cm_ev_nonev_v1", "admin")
        self.assertEqual(st, 200)
        self.assertIn("Class map · ev_nonev", page)
        self.assertIn(">v1<", page)
        self.assertIn(">built-in<", page)
        self.assertIn('<td>inner</td><td class="cj-lm-drop">dropped</td>', page)
        self.assertIn('rowspan="3">EV</td>', page)
        self.assertIn('href="/data/dataset/ds_cm_test"', page)
        self.assertIn("runs run_cm_1", page)
        self.assertIn('href="/data/classmap/cm_ev_nonev_v2">v2</a>', page)
        st, _, page2 = self.get("/data/classmap/cm_ev_nonev_v2", "maker")
        self.assertIn("EV &lt;b&gt;", page2)                                        # escaped
        self.assertNotIn("EV <b>", page2)
        self.assertEqual(self.get("/data/classmap/cm_nope_v1", "admin")[0], 404)
        self.assertEqual(self.get("/data/classmap/..%2F..%2Fx", "admin")[0], 404)
        self.assertIn("cm_ev_nonev_v2", self.get("/data/classmap/", "admin")[2])
        self.assertEqual(self.get("/data/classmap/cm_ev_nonev_v1", "rev1")[0], 403)  # Data: admin and advanced

    def test_dataset_links_its_class_map_only_when_recorded(self):
        _, _, page = self.get("/data/dataset/ds_cm_test", "admin")
        self.assertIn('href="/data/classmap/cm_ev_nonev_v1">cm_ev_nonev_v1</a>', page)
        _, _, plain = self.get("/data/dataset/ds_plain", "admin")
        self.assertNotIn("Class map<", plain)

    def test_schema_page(self):
        st, _, page = self.get("/data/schema/ls_lamellarity_v1", "admin")
        self.assertEqual(st, 200)
        self.assertIn("Label schema · lamellarity", page)
        self.assertIn("Made by pujan · registered", page)
        self.assertIn("MV multivesicular<kbd>3</kbd>", page)
        self.assertIn("Concentric membranes", page)
        self.assertIn(f'href="/data/images/{OBJ}"', page)                            # used by (admin may see it)
        self.assertIn("Read-only this round. A new version is registered from a file with", page)
        st, _, adv = self.get("/data/schema/ls_lamellarity_v1", "maker")             # maker owns it: listed
        self.assertIn(OBJ, adv)
        self.assertEqual(self.get("/data/schema/ls_nope_v1", "admin")[0], 404)
        self.assertIn("ls_lamellarity_v1", self.get("/data/schema/", "admin")[2])

    def test_object_set_image_page_and_lineage(self):
        st, _, page = self.get(f"/data/images/{OBJ}", "admin")
        self.assertEqual(st, 200)
        self.assertIn('href="/data/schema/ls_lamellarity_v1">ls_lamellarity_v1</a>', page)
        self.assertIn(f'href="/reviews/go/{OBJ}"', page)
        st, _, lin = self.get(f"/data/lineage/images/{OBJ}/", "admin")
        self.assertEqual(st, 200)
        self.assertIn("Label schema", lin)
        self.assertIn('href="/data/schema/ls_lamellarity_v1"', lin)
        self.assertIn("rev1, rev2", lin)


class HistoryPageSchemas(unittest.TestCase):
    def test_label_schema_section_and_link(self):
        from registry import history_page as HP
        cur = {"label_schema|ls_a_v1": {"schema": "label_schema/1", "id": "ls_a_v1", "created": "2026-11-14"},
               "label_schema|ls_a_v2": {"schema": "label_schema/1", "id": "ls_a_v2", "supersedes": "ls_a_v1",
                                        "created": "2026-11-15"},
               "review_package|pred_obj_x_0001": {"schema": "review_package/1", "id": "pred_obj_x_0001",
                                                  "kind": "objects", "label_schema": "ls_a_v2"}}
        hist = {k: [v] for k, v in cur.items()}
        m = HP.build_model(cur, hist, {}, [])
        sec = next(s for s in m["sections"] if s["title"] == "Label schemas")
        self.assertEqual(sec["nodes"], [{"k": "label_schema|ls_a_v1", "c": [{"k": "label_schema|ls_a_v2", "c": []}]}])
        self.assertFalse(any(s["title"] == "Other: label_schema" for s in m["sections"]))
        links = m["recs"]["review_package|pred_obj_x_0001"]["links"]
        self.assertIn(["label schema", "label_schema|ls_a_v2"], links)
        self.assertIn(["used by (object set)", "review_package|pred_obj_x_0001"], m["recs"]["label_schema|ls_a_v2"]["links"])


class StartObjectsUnit(unittest.TestCase):
    def test_parse_items_csv(self):
        info = RB.parse_items_csv("\ufeffid,set,image_id,particle_id,d_outer_nm,children\nL1,HEK,x,7,103.5,[]\n".encode())
        self.assertEqual(info["rows"], 1)
        self.assertEqual(info["blindable"], ["set", "d_outer_nm", "children"])
        for bad in (b"", b"image\n", b"image,image\nx,y\n", "image,\u00e9t\u00e9\nx,y\n".encode(), b"\xff\xfe",
                    b"image,a b\nx,y\n"):
            with self.assertRaises(RB.BuildError):
                RB.parse_items_csv(bad)
        self.assertEqual(RB.object_slug("Lamellarity check #1 (HEK)"), "lamellarity_check_1_hek")


if __name__ == "__main__":
    unittest.main()
