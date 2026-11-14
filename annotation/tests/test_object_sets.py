"""
Round 14 (contracts C2-C4): label schemas, attribute decisions on the review server, object sets (build, blind,
export / import round trip, COCO attributes) and the attribute-only controls (is_edited, reviewed.json, apply).

Temp folders only: the golden prediction set (tests/fixtures/cj_golden) is copied; no real package is opened.

  python -m unittest discover -s annotation/tests -p test_object_sets.py -v
"""
import contextlib
import csv
import io
import json
import os
import re
import shutil
import sys
import tempfile
import threading
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
from test_review_server import LiveServer, load_data  # noqa: E402

GOLDEN = REPO / "tests" / "fixtures" / "cj_golden" / "predict" / "ps_20261105_abc123"
KEY = "test-key-123"
SENT_NUM, SENT_TXT = "987654.321", "ZQXSENTINELQZX"


def golden_ps(root: Path) -> Path:
    dst = root / GOLDEN.name
    shutil.copytree(GOLDEN, dst)
    return dst


def items_csv(path: Path, rows, cols=("image", "line", "gap_nm", "secret")):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        w.writerows(rows)
    return path


ALL10 = [("0001", 0), ("0001", 1), ("0001", 2), ("0001", 3), ("0001", 4), ("0002", 0), ("0002", 1), ("0002", 2),
         ("0003", 0), ("0003", 2)]


def build_set(root: Path, rows=None, blind=("gap_nm", "secret"), mode="target", seed=11, reviewers=("au1", "au2")):
    ps = golden_ps(root) if not (root / GOLDEN.name).exists() else root / GOLDEN.name
    rows = rows or [(i, ln, SENT_NUM, SENT_TXT) for i, ln in ALL10]
    src = items_csv(root / "items.csv", rows)
    (root / "review").mkdir(exist_ok=True)
    with contextlib.redirect_stdout(io.StringIO()):
        return OS.build(ps, src, "ls_lamellarity_v1", root / "review", line_col="line", blind_cols=blind,
                        outline_mode=mode, seed=seed, reviewers=reviewers, owner="pujan")


# ---------------------------------------------------------------- schemas (C2)
class Schemas(unittest.TestCase):
    def test_builtins_valid(self):
        self.assertEqual(S.builtin_ids(), ["ls_lamellarity_v1", "ls_legacy_v1"])
        for sid in S.builtin_ids():
            self.assertEqual(S.validate(S.load(sid)), [])

    def test_legacy_is_todays_tool(self):
        """ls_legacy_v1 = review_tool CLASS_ORDER, CLASS_COLOR, hotkeys 1-8, the Low conf tag, MV's display label."""
        page = (REPO / "annotation" / "review_tool" / "index.html").read_text(encoding="utf-8")
        order = json.loads(re.search(r"const CLASS_ORDER = [^;]*?:\s*(\[[^\]]*\]);", page, re.S).group(1))  # legacy path
        colors = dict(re.findall(r'"([^"]+)": "(#[0-9a-f]{6})"', re.search(r"const CLASS_COLOR = \{(.*?)\};", page,
                                                                          re.S).group(1)))
        doc = S.load("ls_legacy_v1")
        self.assertEqual([c["name"] for c in doc["classes"]], order)
        self.assertEqual({c["name"]: c["color"] for c in doc["classes"]}, colors)
        self.assertEqual([c["key"] for c in doc["classes"]], [str(k) for k in range(1, 9)])
        self.assertEqual(doc["tags"], [{"id": "lowconf", "label": "Low conf", "key": "l"}])
        self.assertEqual(next(c for c in doc["classes"] if c["name"] == "MV")["label"], "MV · stack outline")
        lam = S.to_page(S.load("ls_lamellarity_v1"))
        self.assertEqual(next(o for o in lam["object_attrs"][0]["options"] if o["v"] == "MV_vesicular")["label"],
                         "MV multivesicular")

    def _bad(self, change, expect):
        doc = json.loads(json.dumps(S.load("ls_lamellarity_v1")))
        change(doc)
        probs = S.validate(doc)
        self.assertTrue(any(expect in p for p in probs), probs)

    def test_refused(self):
        self._bad(lambda d: d["object_attrs"][1]["options"][0].update(key="i"), "reserved")
        self._bad(lambda d: d["object_attrs"][3].update(key="Enter"), "reserved")
        self._bad(lambda d: d["object_attrs"][3].update(key="ArrowLeft"), "reserved")
        self._bad(lambda d: d["object_attrs"][3].update(key="a"), "already used")      # = membranes "1"
        self._bad(lambda d: d["object_attrs"][3].update(key="Z"), "already used")      # case-insensitive
        self._bad(lambda d: d["object_attrs"][3].update(id="type"), "used twice")
        self._bad(lambda d: d["object_attrs"][3].update(id="class"), "not be 'class'")
        self._bad(lambda d: d["object_attrs"][3].update(id="Bad-Id"), "must match")
        self._bad(lambda d: d["classes"][0].update(color="red"), "#rrggbb")
        self._bad(lambda d: d["classes"][0].update(color="#12345g"), "#rrggbb")
        self._bad(lambda d: d["classes"][0].update(name="<b>x</b>"), "plain characters")
        self._bad(lambda d: d["object_attrs"][0].update(label="x" * 41), "plain characters")
        self._bad(lambda d: d["object_attrs"][4].update(max=301), "max")
        self._bad(lambda d: d["object_attrs"][0]["options"].append({"v": "single"}), "used twice")
        self._bad(lambda d: d.update(version=2), "does not match")
        self._bad(lambda d: d.update(supersedes="ls_legacy_v1"), "class_mapping")
        self._bad(lambda d: d["classes"].append({"name": "EV", "color": "#000000"}), "listed twice")
        self._bad(lambda d: d["classes"].append({"name": "X", "color": "#000000", "key": "1"}), "already used")

    def test_scopes_and_versions(self):
        # classes and object attributes are separate key scopes (EV "1" and type single "1" both stand)
        self.assertEqual(S.validate(S.load("ls_lamellarity_v1")), [])
        new = json.loads(json.dumps(S.load("ls_legacy_v1")))
        new.update(id="ls_legacy_v2", version=2, supersedes="ls_legacy_v1",
                   class_mapping={"Low conf EV": "EV", "MV": "MV stack", "Incomplete": "EV", "Overlapping": "EV",
                                  "VLDL": "non-EV", "inner": "inner"})
        new["classes"] = [{"name": "EV", "color": "#4ade80", "key": "1"}, {"name": "MV stack", "color": "#f87171"},
                          {"name": "inner", "color": "#f472b6"}, {"name": "non-EV", "color": "#e5e7eb"}]
        self.assertEqual(S.validate(new, S.load("ls_legacy_v1")), [])
        self.assertEqual([S.migrate(c, new) for c in ("EV", "Low conf EV", "MV", "VLDL")],
                         ["EV", "EV", "MV stack", "non-EV"])
        del new["class_mapping"]["VLDL"]
        self.assertTrue(any("leaves out" in p for p in S.validate(new, S.load("ls_legacy_v1"))))

    def test_value_ok(self):
        doc = S.load("ls_lamellarity_v1")
        self.assertTrue(S.value_ok(S.attr(doc, "type"), "ML+MV"))
        self.assertFalse(S.value_ok(S.attr(doc, "type"), "MV"))
        self.assertTrue(S.value_ok(S.attr(doc, "outline_wrong"), False))
        self.assertFalse(S.value_ok(S.attr(doc, "outline_wrong"), "false"))
        self.assertTrue(S.value_ok(S.attr(doc, "note"), "x" * 300))
        self.assertFalse(S.value_ok(S.attr(doc, "note"), "x" * 301))
        self.assertFalse(S.value_ok(S.attr(doc, "note"), "\ud800"))
        self.assertTrue(S.value_ok(S.attr(doc, "membranes"), None))


# ---------------------------------------------------------------- server (C3)
class MultiLive:
    """multi_review_server in a thread on a temp review root holding a copy of one package (key = KEY), reached as
    the portal reaches it: loopback, X-Review-Key and X-CJ-User (who=; None sends no identity header)."""

    def __init__(self, src: Path):
        from annotation import multi_review_server as mrs
        self.dir = Path(tempfile.mkdtemp(prefix="objset_multi_"))
        root = self.dir / "review"
        root.mkdir()
        self.pkg = root / Path(src).name
        shutil.copytree(src, self.pkg)
        (self.pkg / ".review_key").write_text(KEY, encoding="utf-8")
        self.data = load_data(self.pkg)
        env = {"CRYOEV_SERVICES_CONFIG": str(self.dir / "no_services.json"), "CJ_REVIEW_EXCLUDE": "",
               "CJ_REVIEW_PREFIXES": ""}
        with mock.patch.dict(os.environ, env):
            self.server = mrs.make_server(root, port=0, people=[{"id": "au1"}, {"id": "au2"}, {"id": "au3"}])
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def request(self, method, path, body=None, key=KEY, headers=None, who="au1"):
        import http.client
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        h = dict(headers or {})
        if key is not None:
            h["X-Review-Key"] = key
        if who is not None:
            h["X-CJ-User"] = who
        data = json.dumps(body).encode() if body is not None and not isinstance(body, bytes) else body
        c.request(method, f"/{self.pkg.name}{path}", body=data, headers=h)
        r = c.getresponse()
        raw = r.read()
        c.close()
        try:
            return r.status, json.loads(raw), r
        except ValueError:
            return r.status, raw, r

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        shutil.rmtree(self.dir, ignore_errors=True)


def put(srv, file, dec, key=KEY, t=1000, who=None, **extra):
    """PUT one record; on a MultiLive the identity header is `who`, else the first decision's who."""
    kw = {}
    if isinstance(srv, MultiLive):
        kw["who"] = who if who is not None else next((d.get("who") for d in dec.values()
                                                      if isinstance(d, dict) and d.get("who")), "au1")
    return srv.request("PUT", f"/api/edits?f={file}", {"t": t, "dec": dec, **extra}, key=key, **kw)[:2]


def dec(v, who, t=1000, pending=True):
    return {"v": v, "t": t, "who": who, **({"pending": True} if pending else {})}


class AttrServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="objset_srv_"))
        cls.pkg = build_set(cls.tmp)

        cls.srv = MultiLive(cls.pkg)            # attribute decisions need the portal's trusted identity (QC 9)

    @classmethod
    def tearDownClass(cls):
        cls.srv.close()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_ping_caps(self):
        self.assertEqual(self.srv.request("GET", "/api/ping")[:2], (200, {"ok": True, "caps": ["i", "t", "m", "g", "a"]}))

    def test_attr_saved_rekeyed_logged(self):
        code, body = put(self.srv, "0001.png", {"a:g1:type@mallory": dec("ML", "au1"),
                                                "a:g1:outline_wrong@au1": dec(True, "au1"),
                                                "a:g1:note@au1": dec('a, "quoted" note é', "au1")})
        self.assertEqual(code, 200, body)
        d = body["edit"]["dec"]
        self.assertNotIn("a:g1:type@mallory", d)                     # the client's rid never counts
        self.assertEqual(d["a:g1:type@au1"]["v"], "ML")
        self.assertTrue(rs._num(d["a:g1:type@au1"]["server_t"]))
        self.assertIs(d["a:g1:outline_wrong@au1"]["v"], True)
        log = (self.srv.pkg / "edits" / rs.ATTR_LOG).read_text(encoding="utf-8").splitlines()
        rows = [json.loads(x) for x in log]
        self.assertTrue({"a:g1:type@au1", "a:g1:outline_wrong@au1", "a:g1:note@au1"} <= {r["key"] for r in rows})
        self.assertEqual(set(rows[0]), {"t", "who", "key", "v", "file"})
        # GET /api/edits lists the record, never the log
        code, body = self.srv.request("GET", "/api/edits")[:2]
        self.assertEqual(code, 200)
        self.assertNotIn(rs.ATTR_LOG, json.dumps(body))
        self.assertIn("a:g1:type@au1", body["0001.png"]["dec"])

    def obj_of(self, stem):
        return [it["obj"] for it in self.srv.data["items"] if it["image"] == stem]

    def test_rater_independence(self):
        """QC 8: GET /api/edits holds only the requester's own attribute decisions; none without an identity."""
        o = self.obj_of("0003")[0]
        self.assertEqual(put(self.srv, "0003.png", {f"a:{o}:inner@au1": dec("1", "au1")})[0], 200)
        self.assertEqual(put(self.srv, "0003.png", {f"a:{o}:inner@au2": dec("2", "au2")})[0], 200)

        def keys(who):
            body = self.srv.request("GET", "/api/edits", who=who)[1]
            return {k for k in body.get("0003.png", {}).get("dec", {}) if k.startswith("a:")}
        self.assertEqual(keys("au1"), {f"a:{o}:inner@au1"})
        self.assertEqual(keys("au2"), {f"a:{o}:inner@au2"})
        self.assertEqual(keys(None), set())
        # the admin export (the files) still has both
        tab = OS.table(self.srv.pkg)
        row = next(r for r in tab["rows"] if r["obj"] == o and r["image"] == "0003")
        self.assertEqual((row["inner@au1"], row["inner@au2"]), ("1", "2"))

    def test_identity_and_reviewers(self):
        """QC 9: no trusted identity -> 403; an identity that is not one of the set's reviewers -> 403."""
        o = self.obj_of("0003")[0]
        self.assertEqual(put(self.srv, "0003.png", {f"a:{o}:type@au3": dec("ML", "au3")})[0], 403)   # not a reviewer
        code = self.srv.request("PUT", "/api/edits?f=0003.png",
                                {"t": 1, "dec": {f"a:{o}:type@au1": dec("ML", "au1")}}, who=None)[0]
        self.assertEqual(code, 403)                                                                     # no identity
        # a non-attribute decision without identity still saves
        code = self.srv.request("PUT", "/api/edits?f=0003.png", {"t": 1, "dec": {"i:note": dec("hi", "x")}},
                                who=None)[0]
        self.assertEqual(code, 200)

    def test_resend_logs_once(self):
        """QC 10: the same pending decision sent twice is one history row."""
        o = self.obj_of("0002")[-1]
        k = f"a:{o}:membranes@au1"
        log = self.srv.pkg / "edits" / rs.ATTR_LOG
        for _ in range(2):
            self.assertEqual(put(self.srv, "0002.png", {k: dec("2", "au1", t=4242)})[0], 200)
        rows = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(sum(r["key"] == k for r in rows), 1)

    def test_only_items_objects_and_cap(self):
        """QC 11: a: only on the objects of that image's items; at most 64 new ones per PUT."""
        other = self.obj_of("0001")[0]                                  # an item, but of another image
        code, body = put(self.srv, "0002.png", {f"a:{other}:type@au1": dec("ML", "au1")})
        self.assertEqual(code, 400, body)
        self.assertIn("objects", body["error"])
        many = {f"a:g{n}:type@au1": dec("ML", "au1") for n in range(1, 66)}
        code, body = put(self.srv, "0001.png", many)
        self.assertEqual(code, 400)
        self.assertIn("at most 64", body["error"])

    def test_echo_of_saved_copy_dropped(self):
        put(self.srv, "0002.png", {"a:g6:type@au2": dec("single", "au2")})
        saved = json.loads((self.srv.pkg / "edits" / "0002.png.json").read_text(encoding="utf-8"))["edit"]["dec"]
        forged = dict(saved["a:g6:type@au2"], v="ML", server_t=9e15)      # not pending: an echo, here altered
        code, body = put(self.srv, "0002.png", {"a:g6:type@au2": forged})
        self.assertEqual(code, 200)
        self.assertEqual(body["edit"]["dec"]["a:g6:type@au2"], saved["a:g6:type@au2"])

    def test_refused(self):
        o = self.obj_of("0003")[0]
        bad = [
            {f"a:{o}:nosuch@au1": dec("x", "au1")},                           # unknown attribute
            {f"a:{o}:type@au1": dec("MV", "au1")},                            # not an option
            {f"a:{o}:outline_wrong@au1": dec("true", "au1")},                 # bool as text
            {f"a:{o}:note@au1": dec("x" * 301, "au1")},                       # text cap
            {f"a:{o}:type@au1": {"v": "ML", "t": 1, "who": "au1", "pending": True, "note": "x"}},   # extra field
            {"a:x1:type@au1": dec("ML", "au1")},                              # bad object key
            {"a:n" + "a" * 90 + ":type@au1": dec("ML", "au1")},               # key over 96
            {"i:a:type@au1": dec("ML", "au1")},                               # no image attributes in this schema
            {f"a:{o}:type@<b>": dec("ML", "au1")},                            # rid not a name
        ]   # (a decision without who: the portal identity stamps it, so that is no refusal here)
        before = (self.srv.pkg / "edits" / rs.ATTR_LOG).read_bytes() if (self.srv.pkg / "edits" / rs.ATTR_LOG).exists() else b""
        for d in bad:
            self.assertEqual(put(self.srv, "0003.png", d)[0], 400, d)
        # a lone surrogate (JSON "\ud800") in a text value
        raw = ('{"t": 1, "dec": {"a:%s:note@au1": {"v": "\\ud800", "t": 1, "who": "au1", "pending": true}}}'
               % o).encode()
        rec = self.srv.pkg / "edits" / "0003.png.json"
        had = rec.read_bytes() if rec.exists() else None
        self.assertEqual(self.srv.request("PUT", "/api/edits?f=0003.png", raw)[0], 400)
        self.assertEqual(rec.read_bytes() if rec.exists() else None, had)
        after = (self.srv.pkg / "edits" / rs.ATTR_LOG).read_bytes() if (self.srv.pkg / "edits" / rs.ATTR_LOG).exists() else b""
        self.assertEqual(before, after)

    def test_valid_edit_shapes(self):
        self.assertTrue(rs.valid_edit({"t": 1, "dec": {"a:g1:type@au1": {"v": None, "t": 1}}}))
        self.assertFalse(rs.valid_edit({"t": 1, "dec": {"a:g1:type@au1": "ML"}}))      # never a bare string
        self.assertFalse(rs.valid_edit({"t": 1, "dec": {"a:g1:type": {"v": "ML", "t": 1}}}))  # no rid
        self.assertEqual(rs.parse_attr_key("a:p3:inner@j.doe"), ("p3", "inner", "j.doe"))
        self.assertEqual(rs.parse_attr_key("i:a:quality@au1"), (None, "quality", "au1"))
        self.assertIsNone(rs.parse_attr_key("a:g1:Type@au1"))


class NoSchemaPackage(unittest.TestCase):
    """A package without schema.json (every package before round 14) refuses a: keys; its other keys still save."""

    def test_refused_without_schema(self):
        tmp = Path(tempfile.mkdtemp(prefix="objset_noschema_"))
        try:
            ps = golden_ps(tmp)

            def build(dst):
                with contextlib.redirect_stdout(io.StringIO()):
                    pkg = pp.build(ps, tmp / "rv", owner="pujan")
                shutil.copytree(pkg, dst)
            srv = LiveServer(build=build, name="pred_ps_20261105_abc123")
            try:
                self.assertEqual(put(srv, "0001.png", {"a:g1:type@au1": dec("ML", "au1")})[0], 400)
                self.assertEqual(put(srv, "0001.png", {"g1": dec("keep:EV", "au1")})[0], 200)
                self.assertFalse((srv.pkg / "edits" / rs.ATTR_LOG).exists())
            finally:
                srv.close()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------- attribute-only control (C3)
def attr_only(file, pkg_id, who="au1"):
    return {"file": file, "pkg": pkg_id, "fmt": 1, "rx": {}, "edit": {"t": 5, "dec": {
        f"a:g1:type@{who}": {"v": "ML", "t": 5, "who": who, "server_t": 6},
        f"a:g1:outline_wrong@{who}": {"v": True, "t": 5, "who": who, "server_t": 6},
        f"i:a:quality@{who}": {"v": "good", "t": 5, "who": who, "server_t": 6}}}}


class AttributeOnlyControl(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="objset_ctl_"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_is_edited_and_reviewed_json(self):
        ps = golden_ps(self.tmp)
        with contextlib.redirect_stdout(io.StringIO()):
            pkg = pp.build(ps, self.tmp / "rv", owner="pujan")
        (pkg / "edits").mkdir(exist_ok=True)
        # one real decision so the comparison covers an edited image too
        real = {"file": "0002.png", "pkg": pkg.name, "fmt": 1, "rx": {},
                "edit": {"t": 3, "dec": {"g6": {"v": "keep:MV", "t": 3, "who": "au2", "server_t": 4}}}}
        (pkg / "edits" / "0002.png.json").write_text(json.dumps(real), encoding="utf-8")
        before = pp.reviewed_polygons(pkg, write=False)
        self.assertFalse(pp.is_edited(attr_only("0001.png", pkg.name)["edit"]))
        (pkg / "edits" / "0001.png.json").write_text(json.dumps(attr_only("0001.png", pkg.name)), encoding="utf-8")
        mixed = json.loads(json.dumps(real))
        mixed["edit"]["dec"].update(attr_only("0002.png", pkg.name, "au3")["edit"]["dec"])
        (pkg / "edits" / "0002.png.json").write_text(json.dumps(mixed), encoding="utf-8")
        (pkg / "edits" / rs.ATTR_LOG).write_text('{"t": 1, "who": "au1", "key": "a:g1:type@au1", "v": "ML"}\n',
                                                 encoding="utf-8")
        self.assertEqual(pp.reviewed_polygons(pkg, write=False), before)
        # sensitivity: without the exclusion the attribute-only record would count as edited
        self.assertTrue(any(not k.startswith("i:") for k in attr_only("0001.png", pkg.name)["edit"]["dec"]))

    def test_apply_output(self):
        import test_apply_entry as tae
        from annotation import apply_review_edits as ar
        w = tae.World(source=True)
        try:
            def run(tag):
                out = w.tmp / f"out_{tag}"
                with contextlib.ExitStack() as st:
                    tae._patches(st)
                    with contextlib.redirect_stdout(io.StringIO()):
                        ar.make_version(w.base, w.pkg, out_root=out, who=None, people_path=w.people,
                                        stamp=tae.STAMP, log=lambda *a: None)
                lv = out / f"lv_{tae.STAMP}"
                files = {n: w.norm((lv / n).read_text(encoding="utf-8")).replace(str(out).replace("\\", "/"), "<OUT>")
                         for n in ("labels.coco.json", "changes.jsonl", "conflicts.csv")}
                vj = json.loads((lv / "version.json").read_text(encoding="utf-8"))
                return files, vj
            recs = sorted(w.edits.glob("*.json"))
            free = json.loads(recs[-1].read_text(encoding="utf-8"))["file"]
            recs[-1].unlink()                                    # an image with no record, for the third case
            f0, v0 = run("plain")
            # a: keys added to every existing record (records that change labels already)
            for f in sorted(w.edits.glob("*.json")):
                rec = json.loads(f.read_text(encoding="utf-8"))
                rec["edit"]["dec"].update(attr_only(rec["file"], rec.get("pkg"))["edit"]["dec"])
                f.write_text(json.dumps(rec), encoding="utf-8")
            f1, v1 = run("mixed")
            self.assertEqual(f1, f0)
            volatile = ("edits_dir", "package_path", "created", "tool")   # wall clock / paths, not the review's content
            strip = lambda v: {k: x for k, x in v.items() if k not in volatile}
            self.assertEqual(strip(v1), strip(v0))
            # an attribute-only record for an image without one: labels / changes / conflicts unchanged; version.json
            # counts it as a record without changes (as an i:note-only record), nothing else
            (w.edits / f"{free}.json").write_text(json.dumps(attr_only(free, "synthetic_pkg")), encoding="utf-8")
            f2, v2 = run("attronly")
            self.assertEqual(f2, f0)
            diff = {k for k in set(v0) | set(v2) if v0.get(k) != v2.get(k)}
            self.assertEqual(diff - set(volatile), {"n_records", "n_records_without_changes"})
            self.assertEqual(v2["n_records"], v0["n_records"] + 1)
        finally:
            shutil.rmtree(w.tmp, ignore_errors=True)


# ---------------------------------------------------------------- object sets (C4)
class ObjectSet(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="objset_"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_build_shape_shuffle_blind_files(self):
        pkg = build_set(self.tmp, mode="target+contained")
        self.assertRegex(pkg.name, r"^pred_obj_lamellarity_[0-9a-f]{4}$")
        d = load_data(pkg)
        self.assertEqual(d["kind"], "objects")
        self.assertEqual(d["schema"]["id"], "ls_lamellarity_v1")
        self.assertEqual(len(d["items"]), 10)
        for it in d["items"]:
            self.assertRegex(it["id"], r"^[0-9a-f]{5}$")
            self.assertEqual(set(it) - {"also"}, {"id", "image", "obj", "crop"})
            x0, y0, x1, y1 = it["crop"]
            self.assertTrue(0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1)
        items = OS.read_items(pkg)
        self.assertNotEqual([(i["image"], i["line"]) for i in items["items"]], ALL10)   # shuffled
        self.assertEqual(sorted((i["image"], i["line"]) for i in items["items"]), sorted(ALL10))
        # the object key both ways: item -> (image, line) -> the gt label key, and back
        gt = {(Path(im["file"]).stem, g["id"]): g for im in d["images"] for g in im["gt"]}
        meta = pp._read_json(pkg / pp.META)["images"]
        for it in items["items"]:
            lab = {m["id"]: m["line"] for m in meta[it["image"]]["labels"]}
            self.assertEqual(lab[int(it["obj"][1:])], it["line"])
            self.assertIn((it["image"], int(it["obj"][1:])), gt)
        # 0001 line 1 lies inside line 0 (the golden nested pair): drawn with it in target+contained
        it0 = next(i for i in items["items"] if (i["image"], i["line"]) == ("0001", 0))
        it1 = next(i for i in items["items"] if (i["image"], i["line"]) == ("0001", 1))
        self.assertIn(it1["obj"], it0["also"])
        rec = json.loads((pkg / pp.RECORD).read_text(encoding="utf-8"))
        from registry import store
        self.assertEqual(store.check_record(rec, store.schemas()), [])
        self.assertEqual((rec["kind"], rec["label_schema"], rec["n_items"], rec["blind_cols"]),
                         ("objects", "ls_lamellarity_v1", 10, ["gap_nm", "secret"]))
        self.assertEqual(json.loads((pkg / OS.SCHEMA_FILE).read_text(encoding="utf-8")), S.load("ls_lamellarity_v1"))
        # blind: no sentinel anywhere in the package but items.json
        hits = [p.name for p in pkg.rglob("*") if p.is_file() and p.name != OS.ITEMS
                and (SENT_NUM.encode() in p.read_bytes() or SENT_TXT.encode() in p.read_bytes())]
        self.assertEqual(hits, [])
        self.assertIn(SENT_NUM, (pkg / OS.ITEMS).read_text(encoding="utf-8"))

    def test_refusals(self):
        ps = golden_ps(self.tmp)
        cases = [([("0001", 0, 1, 2), ("0001", 0, 1, 2)], (), "same object"),
                 ([("0009", 0, 1, 2)], (), "not in the prediction set"),
                 ([("0001", 99, 1, 2)], (), "not a polygon"),
                 ([("0001", 0, 1, 2)], ("line",), "object's key"),
                 ([("0001", 0, 1, "S1")], ("secret",), "shows anyway"),       # S1 = a sample label
                 ([("0001", 0, 1, "golden_0001")], ("secret",), "shows anyway"),
                 ([("0001", 0, 1, 2)], ("nope",), "not in the items CSV")]
        for n, (rows, blind, why) in enumerate(cases):
            src = items_csv(self.tmp / f"i{n}.csv", rows)
            with self.assertRaises(OS.ObjectSetError) as cm:
                OS.plan(ps, src, "ls_lamellarity_v1", line_col="line", blind_cols=blind)
            self.assertIn(why, str(cm.exception))
        with self.assertRaises(OS.ObjectSetError):
            OS.plan(ps, items_csv(self.tmp / "ok.csv", [("0001", 0, 1, 2)]), "ls_legacy_v1", line_col="line")
        self.assertEqual(list((self.tmp / GOLDEN.name).rglob("adopted")), [])

    def test_cli_dry_run_writes_nothing(self):
        ps = golden_ps(self.tmp)
        src = items_csv(self.tmp / "items.csv", [("0001", 0, 1, 2)])
        before = sorted(p.relative_to(self.tmp) for p in self.tmp.rglob("*"))
        with contextlib.redirect_stdout(io.StringIO()) as out:
            rc = OS.main(["build", "--ps", str(ps), "--items", str(src), "--schema", "ls_lamellarity_v1",
                          "--line-col", "line", "--review-root", str(self.tmp / "rv")])
        self.assertEqual(rc, 0)
        self.assertIn("DRY RUN", out.getvalue())
        self.assertEqual(sorted(p.relative_to(self.tmp) for p in self.tmp.rglob("*")), before)
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(OS.main(["build", "--ps", str(ps), "--items", str(src), "--schema", "ls_lamellarity_v1",
                                      "--line-col", "line", "--register"]), 2)        # --register needs --do + --registry

    def test_round_trip_two_reviewers_and_coco(self):
        """10 items on ls_lamellarity_v1, scripted decisions from two reviewers through the real server:
        export -> import -> export identical; COCO carries the attributes; progress counts done items."""
        pkg = build_set(self.tmp)
        srv = MultiLive(pkg)
        try:
            items = load_data(srv.pkg)["items"]
            file_of = {Path(i["file"]).stem: i["file"] for i in load_data(srv.pkg)["images"]}
            opts = {"type": ["single", "ML", "MV_vesicular", "ML+MV", "unsure"], "membranes": ["1", "2", "3", "4+"],
                    "inner": ["0", "1", "2", "3+"]}
            t = 1000
            for n, it in enumerate(items):
                for who, shift in (("au1", 0), ("au2", n % 2)):
                    if who == "au2" and n == 9:
                        continue                                     # one item only au1 decided: not done
                    t += 1
                    d = {f"a:{it['obj']}:{a}@{who}": dec(o[(n + shift) % len(o)], who, t) for a, o in opts.items()}
                    d[f"a:{it['obj']}:outline_wrong@{who}"] = dec(n % 3 == 0, who, t)
                    if n % 4 == 0:
                        d[f"a:{it['obj']}:note@{who}"] = dec(f'note {n}, "{who}" – ünïcode', who, t)
                    code, body = put(srv, file_of[it["image"]], d, t=t)
                    self.assertEqual(code, 200, body)
            # one change of mind: history 2 on that key
            it = items[0]
            put(srv, file_of[it["image"]], {f"a:{it['obj']}:type@au1": dec("unsure", "au1", t + 50)}, t=t + 50)
            tab = OS.table(srv.pkg)
            text = OS.to_csv(tab)
            back = OS.import_csv(text, OS.read_schema(srv.pkg))
            self.assertEqual(OS.to_csv(back), text)
            out = self.tmp / "export.csv"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(OS.main(["export", "--pkg", str(srv.pkg), "--out", str(out), "--coco",
                                          str(self.tmp / "coco.json")]), 0)
                self.assertEqual(OS.main(["import", "--pkg", str(srv.pkg), "--csv", str(out)]), 0)
            self.assertEqual(out.read_text(encoding="utf-8"), text)
            cols = tab["columns"]
            self.assertEqual(cols[:4], ["item", "image", "obj", "line"])
            self.assertEqual(cols[4:6], ["type@au1", "type@au2"])
            self.assertEqual(cols[-7:], ["reviewers", "last_t", "history_n", "schema_id", "schema_version", "gap_nm",
                                         "secret"])
            r0 = next(r for r in tab["rows"] if r["item"] == items[0]["id"])
            self.assertEqual((r0["type@au1"], r0["history_n"]), ("unsure", 2 * 5 + 1))   # 4-5 attrs x 2 + the change
            r9 = next(r for r in tab["rows"] if r["item"] == items[9]["id"])
            self.assertIsNone(r9["agree_type"])
            self.assertEqual(r9["reviewers"], "au1")
            r1 = next(r for r in tab["rows"] if r["item"] == items[1]["id"])
            self.assertIs(r1["agree_type"], False)
            r2 = next(r for r in tab["rows"] if r["item"] == items[2]["id"])
            self.assertIs(r2["agree_type"], True)
            self.assertEqual(r0["gap_nm"], SENT_NUM)                                        # blind cols joined back
            coco = json.loads((self.tmp / "coco.json").read_text(encoding="utf-8"))
            self.assertEqual(len(coco["annotations"]), 10)
            a0 = next(a for a in coco["annotations"] if a["item"] == items[0]["id"])
            self.assertEqual(a0["attributes"]["type@au1"], "unsure")
            self.assertEqual(a0["attributes_schema"], {"id": "ls_lamellarity_v1", "version": 1})
            self.assertNotIn(SENT_NUM, json.dumps(coco))
            prog = om.progress(srv.pkg)
            self.assertEqual((prog["n_items"], prog["n_done"], prog["by_reviewer"]), (10, 9, {"au1": 10, "au2": 9}))
            # a tampered value is refused on import
            bad = text.replace(",ML,", ",MV,", 1)
            if bad != text:
                with self.assertRaises(OS.ObjectSetError):
                    OS.import_csv(bad, OS.read_schema(srv.pkg))
        finally:
            srv.close()


class BlindHttp(unittest.TestCase):
    """Every HTTP response of review_server and multi_review_server for an object set: no sentinel."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="objset_http_"))
        self.pkg = build_set(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _check(self, bodies):
        for name, body in bodies.items():
            self.assertNotIn(SENT_NUM.encode(), body, name)
            self.assertNotIn(SENT_TXT.encode(), body, name)

    def test_review_server(self):
        srv = LiveServer(build=lambda dst: shutil.copytree(self.pkg, dst), name=self.pkg.name)
        try:
            d = srv.data
            im = next(i["file"] for i in d["images"] if Path(i["file"]).stem == d["items"][0]["image"])
            # standalone review_server: no trusted identity, so attribute decisions are refused (QC 9)
            self.assertEqual(put(srv, im, {f"a:{d['items'][0]['obj']}:type@au1": dec("ML", "au1")})[0], 403)
            bodies = {}
            for path in ("/", "/data.js", "/api/ping", "/api/meta", "/api/edits", "/api/groups", "/api/people",
                         "/api/claims", "/items.json", "/schema.json", "/edits/_attr_log.jsonl",
                         *(f"/{im['src']}" for im in d["images"])):
                c = __import__("http.client").client.HTTPConnection("127.0.0.1", srv.port, timeout=10)
                c.request("GET", path, headers={"X-Review-Key": KEY})
                r = c.getresponse()
                bodies[path] = r.read()
                if path in ("/items.json", "/schema.json", "/edits/_attr_log.jsonl"):
                    self.assertEqual(r.status, 404, path)
                c.close()
            self._check(bodies)
        finally:
            srv.close()

    def test_multi_review_server(self):
        from annotation import multi_review_server as mrs
        root = self.tmp / "review"
        key = (self.pkg / ".review_key").read_text(encoding="utf-8").strip()
        env = {"CRYOEV_SERVICES_CONFIG": str(self.tmp / "no_services.json"), "CJ_REVIEW_EXCLUDE": "",
               "CJ_REVIEW_PREFIXES": ""}
        with mock.patch.dict(os.environ, env):
            server = mrs.make_server(root, port=0, people=[])
        th = threading.Thread(target=server.serve_forever, daemon=True)
        th.start()
        try:
            import http.client
            port = server.server_address[1]
            pid = self.pkg.name
            d = load_data(self.pkg)
            bodies = {}

            def get(path):
                c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
                c.request("GET", f"/{pid}{path}", headers={"X-Review-Key": key})
                r = c.getresponse()
                b = r.read()
                c.close()
                return r.status, b
            im0 = next(i["file"] for i in d["images"] if Path(i["file"]).stem == d["items"][0]["image"])
            c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
            body = json.dumps({"t": 1, "dec": {f"a:{d['items'][0]['obj']}:type@au1": dec("ML", "au1")}}).encode()
            c.request("PUT", f"/{pid}/api/edits?f={im0}", body=body,
                      headers={"X-Review-Key": key, "Content-Type": "application/json"})
            self.assertEqual(c.getresponse().status, 403)                # no portal identity: refused (QC 9)
            c.close()
            c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
            c.request("PUT", f"/{pid}/api/edits?f={im0}", body=body,
                      headers={"X-Review-Key": key, "X-CJ-User": "au1", "Content-Type": "application/json"})
            self.assertEqual(c.getresponse().status, 200)                # the portal's identity, a set reviewer
            c.close()
            for path in ("/", "/data.js", "/api/ping", "/api/meta", "/api/edits", "/api/groups", "/items.json",
                         "/schema.json", *(f"/{im['src']}" for im in d["images"])):
                st, b = get(path)
                bodies[path] = b
                if path in ("/items.json", "/schema.json"):
                    self.assertEqual(st, 404, path)
            self.assertEqual(json.loads(bodies["/api/ping"])["caps"], ["i", "t", "m", "g", "a"])
            self.assertIn(b'"kind":"objects"', bodies["/data.js"])
            self._check(bodies)
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
