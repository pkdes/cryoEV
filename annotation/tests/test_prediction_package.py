"""
Tests for annotation/prediction_package.py and annotation/multi_review_server.py against the CJ golden fixtures
(tests/fixtures/cj_golden, BUILD_CONTRACTS.md sections 3 and 6). Packages are built in temp folders; the fixtures are
only read (checked byte-identical at the end).

  python -m unittest discover -s annotation/tests -v

- build: labels (conf >= 0.50) vs candidates (0.25-0.50); stub COCO; sidecar; registry record; refusals.
- reviewed.json: control (no edits: every image unreviewed, objects = the prediction files at conf >= 0.50, parsed
  here independently); the golden review (0002: G deleted, F reclassed MV, reviewed) done through the multi server's
  HTTP API as the page does it (cookie from ?k=, PUT of the whole record with pending decisions) == reviewed.json.
- multi server: two packages with their own keys and cookies; unknown id 404; traversal 400/404; non-prefix folder
  not served; a package added after start; index / data.js / images / claims for both; the CLI.
- spot check: deterministic, spread over both sample labels, favours images with many candidates.
"""
import hashlib
import http.client
import json
import math
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from urllib.parse import quote

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))
from annotation import multi_review_server as mrs  # noqa: E402
from annotation import prediction_package as pp  # noqa: E402
sys.path.insert(0, str(REPO / "registry"))
from schema_check import check_record, load_schemas  # noqa: E402

GOLDEN = REPO / "tests" / "fixtures" / "cj_golden"
PS_ID = "ps_20261105_abc123"
PS = GOLDEN / "predict" / PS_ID
MODEL = "v5_evnonev_y8m_20261029"
PKG = f"pred_{PS_ID}"
PS_B = "ps_20261105_def456"
PKG_B = f"pred_{PS_B}"
W = H = 400
T0 = 1_762_400_000_000


def tree_digest(root: Path) -> dict:
    return {str(p.relative_to(root)): hashlib.md5(p.read_bytes()).hexdigest() for p in sorted(root.rglob("*"))
            if p.is_file()}


FIXTURE_BEFORE = tree_digest(GOLDEN)


def tearDownModule():
    assert tree_digest(GOLDEN) == FIXTURE_BEFORE, "the golden fixtures changed"


def raw_predictions(image: str) -> list:
    """The prediction file as (cls, conf, [[x, y] px]) — read here, not through the code under test."""
    out = []
    for line in (PS / "p" / MODEL / "predictions" / f"{image}.txt").read_text().splitlines():
        v = line.split()
        xy = [float(x) for x in v[2:]]
        out.append((int(v[0]), float(v[1]), [[xy[i] * W, xy[i + 1] * H] for i in range(0, len(xy), 2)]))
    return out


def centroid(poly):
    return (sum(p[0] for p in poly) / len(poly), sum(p[1] for p in poly) / len(poly))


def max_dist(a, b):
    return max(math.dist(p, q) for p, q in zip(a, b)) if len(a) == len(b) else math.inf


def copy_ps(dst_root: Path, ps_id: str = PS_ID, sample_labels=None, state=None) -> Path:
    """A temp copy of the golden prediction set, optionally under another id / with other sample labels / status."""
    dst = dst_root / "predict" / ps_id
    shutil.copytree(PS, dst)
    req = json.loads((dst / "request.json").read_text(encoding="utf-8"))
    req["ps"] = ps_id
    (dst / "request.json").write_text(json.dumps(req), encoding="utf-8")
    if sample_labels:
        rows = (dst / "images.csv").read_text(encoding="utf-8").splitlines()
        rows = [rows[0]] + [",".join(r.split(",")[:3] + [sample_labels.get(r.split(",")[0], r.split(",")[3])]
                                     + r.split(",")[4:]) for r in rows[1:]]
        (dst / "images.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    if state:
        st = json.loads((dst / "status.json").read_text(encoding="utf-8"))
        st["state"] = state
        (dst / "status.json").write_text(json.dumps(st), encoding="utf-8")
    return dst


class Tmp(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="predpkg_"))
        self.root = self.tmp / "review"
        self.root.mkdir()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def build(self, ps=PS, **kw):
        return pp.build(ps, self.root, **kw)


class BuildTests(Tmp):
    def test_labels_vs_candidates(self):
        pkg = self.build()
        self.assertEqual(pkg, self.root / PKG)
        data, meta = pp.read_datajs(pkg), json.loads((pkg / pp.META).read_text(encoding="utf-8"))
        by = {im["file"]: im for im in data["images"]}
        self.assertEqual(list(by), ["0001.png", "0002.png", "0003.png"])
        # 0001: A .90, B .80, C .95, E .70 (non-EV) are labels; D .30 is the one candidate
        m1 = meta["images"]["0001"]
        self.assertEqual([(m["cls"], m["conf"]) for m in m1["labels"]], [(0, .90), (0, .80), (0, .95), (1, .70)])
        self.assertEqual([(c["k"], c["cls"], c["conf"]) for c in m1["candidates"]], [(0, 0, .30)])
        self.assertEqual([g["c"] for g in by["0001.png"]["gt"]], ["EV", "EV", "EV", "non-EV"])
        self.assertEqual([(p["s"], p["c"], p["u"]) for p in by["0001.png"]["pr"]], [(0.3, "EV", 1)])
        self.assertEqual((by["0001.png"]["fg"], by["0001.png"]["fp"]), (0, 1))
        d = by["0001.png"]["pr"][0]["p"][0]
        self.assertLess(math.dist(centroid([[d[i] * W, d[i + 1] * H] for i in range(0, len(d), 2)]), (120, 300)), 0.5)
        # 0002: F G H I J, all >= 0.5; 0003: K L N
        self.assertEqual(len(by["0002.png"]["gt"]), 5)
        self.assertEqual(by["0002.png"]["pr"], [])
        self.assertEqual([m["conf"] for m in meta["images"]["0003"]["labels"]], [.92, .90, .91])
        # every label sits where its prediction is
        for img in ("0001", "0002", "0003"):
            want = [centroid(p) for c, s, p in raw_predictions(img) if s >= 0.5]
            got = [centroid([[r[i] * W, r[i + 1] * H] for i in range(0, len(r), 2)]) for g in by[f"{img}.png"]["gt"]
                   for r in g["p"]]
            self.assertEqual(len(got), len(want))
            self.assertTrue(all(math.dist(a, b) < 0.01 for a, b in zip(got, want)), img)
        ids = [g["id"] for im in data["images"] for g in im["gt"]]
        self.assertEqual(len(set(ids)), 12)

    def test_package_files_and_metadata(self):
        pkg = self.build()
        data = pp.read_datajs(pkg)
        self.assertEqual((data["id"], data["kind"], data["prediction_set"], data["model"], data["owner"]),
                         (PKG, "prediction", PS_ID, MODEL, "pujan"))
        self.assertEqual(data["categories"], ["EV", "non-EV", "MV", "inner"])
        self.assertEqual([im["split"] for im in data["images"]], ["S1", "S1", "S2"])
        self.assertEqual((pkg / "index.html").read_bytes(), (REPO / "annotation" / "review_tool" / "index.html").read_bytes())
        for im in data["images"]:
            self.assertTrue((pkg / im["src"]).is_file())
            self.assertEqual((im["w"], im["h"]), (W, H))
        coco = json.loads((pkg / "labels.coco.json").read_text(encoding="utf-8"))
        self.assertEqual(coco["categories"], [{"id": 0, "name": "EV"}, {"id": 1, "name": "non-EV"}])
        self.assertEqual(len(coco["annotations"]), 12)
        self.assertEqual(sorted({a["category_id"] for a in coco["annotations"]}), [0, 1])
        key = (pkg / ".review_key").read_text(encoding="utf-8")
        self.assertGreaterEqual(len(key), 20)
        rec = json.loads((pkg / pp.RECORD).read_text(encoding="utf-8"))
        self.assertEqual((rec["kind"], rec["owner"], rec["prediction_set"], rec["id"]), ("prediction", "pujan", PS_ID, PKG))
        # registry-ready except base_kind: the schema's enum (pull | export) has no value for a prediction set yet
        errs = check_record(rec, load_schemas())
        self.assertEqual([e for e in errs if not e.startswith("$.base_kind")], [])
        # rebuild without edits: same data, same key; with edits: refused
        self.build()
        self.assertEqual((pkg / ".review_key").read_text(encoding="utf-8"), key)
        self.assertEqual(pp.read_datajs(pkg)["images"], data["images"])
        (pkg / "edits").mkdir(exist_ok=True)
        (pkg / "edits" / "0001.png.json").write_text("{}", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            self.build()

    def test_owner_override_and_refusals(self):
        self.assertEqual(pp.read_datajs(self.build(owner="au1"))["owner"], "au1")
        with self.assertRaises(ValueError):
            self.build(owner="Bad Owner")
        with self.assertRaises(ValueError):  # not done yet
            pp.build(copy_ps(self.tmp / "a", state="running"), self.root)
        miss = copy_ps(self.tmp / "b")
        (miss / "p" / MODEL / "predictions" / "0002.txt").unlink()
        with self.assertRaises(FileNotFoundError):
            pp.build(miss, self.root)

    def test_sample_label_markup_is_not_put_in_split(self):
        ps = copy_ps(self.tmp / "c", sample_labels={"0001": "<img src=x onerror=alert(1)>"})
        pkg = pp.build(ps, self.root)
        split = pp.read_datajs(pkg)["images"][0]["split"]
        self.assertFalse(set(split) & set("<>\"'&=()"), split)
        meta = json.loads((pkg / pp.META).read_text(encoding="utf-8"))
        self.assertEqual(meta["images"]["0001"]["sample_label"], "<img src=x onerror=alert(1)>")


class ReviewedControlTests(Tmp):
    def test_no_edits_every_image_unreviewed_objects_are_predictions(self):
        """Known answer: no edits -> reviewed false everywhere, objects = the prediction files at conf >= 0.5."""
        pkg = self.build()
        out = pp.reviewed_polygons(pkg)
        self.assertEqual(json.loads((pkg / "reviewed.json").read_text(encoding="utf-8")), out)
        self.assertEqual({k: v for k, v in out.items() if k != "images"},
                         {"schema": "reviewed/1", "package": PKG, "prediction_set": PS_ID, "model": MODEL})
        self.assertEqual(sorted(out["images"]), ["0001", "0002", "0003"])
        for img, v in out["images"].items():
            self.assertEqual((v["reviewed"], v["reviewed_by"]), (False, []))
            want = [(c, s, p) for c, s, p in raw_predictions(img) if s >= 0.5]
            self.assertEqual(len(v["objects"]), len(want), img)
            for o, (c, s, p) in zip(v["objects"], want):
                self.assertEqual((o["cls"], o["conf"], o["source"], o["reviewer_class"]), (c, s, "model", None))
                self.assertLess(max_dist(o["poly"], p), 0.5)
        golden = json.loads((GOLDEN / "reviewed.json").read_text(encoding="utf-8"))
        for img in ("0001", "0003"):  # unreviewed in the golden scenario too
            assert_same_image(self, out["images"][img], golden["images"][img])


def assert_same_image(tc, got, want):
    """Same reviewed flag / reviewers, and the same objects (cls, conf, source, reviewer_class), outlines within
    0.5 px, in any order."""
    tc.assertEqual((got["reviewed"], got["reviewed_by"]), (want["reviewed"], want["reviewed_by"]))
    tc.assertEqual(len(got["objects"]), len(want["objects"]))
    left = list(got["objects"])
    for w in want["objects"]:
        hit = next((o for o in left if max_dist(o["poly"], w["poly"]) <= 0.5), None)
        tc.assertIsNotNone(hit, f"no object at {centroid(w['poly'])}")
        tc.assertEqual({k: hit[k] for k in ("cls", "source", "reviewer_class")},
                       {k: w[k] for k in ("cls", "source", "reviewer_class")})
        tc.assertAlmostEqual(hit["conf"], w["conf"], places=6)
        left.remove(hit)


class ServerCase(Tmp):
    """A multi server (in this process, on a free port) over a review root holding the golden package and a second
    one built from a copy of the set under another id."""

    def setUp(self):
        super().setUp()
        self.pkg = self.build()
        self.pkg_b = pp.build(copy_ps(self.tmp / "b", PS_B), self.root)
        self.key = (self.pkg / ".review_key").read_text(encoding="utf-8")
        self.key_b = (self.pkg_b / ".review_key").read_text(encoding="utf-8")
        self.assertNotEqual(self.key, self.key_b)
        self.srv = mrs.make_server(self.root, "127.0.0.1", 0, people=[])
        self.port = self.srv.server_address[1]
        self.thread = threading.Thread(target=self.srv.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.srv.shutdown()
        self.srv.server_close()
        super().tearDown()

    def req(self, method, path, body=None, key=None, cookie=None, headers=None):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        h = dict(headers or {})
        if key:
            h["X-Review-Key"] = key
        if cookie:
            h["Cookie"] = cookie
        data = json.dumps(body).encode() if body is not None else None
        c.request(method, path, body=data, headers=h)
        r = c.getresponse()
        raw = r.read()
        c.close()
        return r.status, raw, r

    def login(self, pkg_id, key):
        """Open the link once, as a phone does: the cookie the server sets for this package."""
        st, _, r = self.req("GET", f"/{pkg_id}/?k={key}")
        self.assertEqual(st, 200)
        cookie = r.getheader("Set-Cookie").split(";")[0]
        self.assertEqual(cookie.split("=")[0], f"review_key_{self.port}_{pkg_id}")
        return cookie

    def label_id(self, pkg, file, cx, cy):
        im = next(i for i in pp.read_datajs(pkg)["images"] if i["file"] == file)
        for g in im["gt"]:
            r = g["p"][0]
            if math.dist(centroid([[r[i] * W, r[i + 1] * H] for i in range(0, len(r), 2)]), (cx, cy)) < 1:
                return g["id"]
        raise AssertionError(f"no label at {cx},{cy}")

    def put_decisions(self, cookie, pkg_id, file, steps):
        """The page's way: each change edits the local record (new decisions pending), the whole record is PUT, and
        the server's merged reply replaces the local copy."""
        rec, t = {"dec": {}, "reviewed": False, "modified": False, "t": 0}, T0
        for step in steps:
            t += 1000
            rec = json.loads(json.dumps(rec))
            if step == "reviewed":
                rec["reviewed"], rec["reviewedT"] = True, t
            else:
                key, v, p = step
                rec["dec"][key] = {"v": v, "t": t, "who": "pujan", "pending": True, **({"p": p} if p else {})}
                rec["modified"] = True
            rec["t"] = t
            st, raw, _ = self.req("PUT", f"/{pkg_id}/api/edits?f={quote(file)}", rec, cookie=cookie)
            self.assertEqual(st, 200, raw)
            rec = json.loads(raw)["edit"]
        return rec


class ReviewThroughHttpTests(ServerCase):
    def test_golden_review_of_0002(self):
        """0002: delete G, reclass F as MV, mark reviewed -> reviewed.json as the golden one."""
        cookie = self.login(PKG, self.key)
        g, f = self.label_id(self.pkg, "0002.png", 300, 100), self.label_id(self.pkg, "0002.png", 100, 100)
        rec = self.put_decisions(cookie, PKG, "0002.png", [(f"g{g}", "del", None), (f"g{f}", "keep:MV", None), "reviewed"])
        self.assertTrue(rec["reviewed"])
        self.assertTrue(all("server_t" in d and "pending" not in d for d in rec["dec"].values()))
        st, raw, _ = self.req("GET", f"/{PKG}/api/edits", cookie=cookie)
        self.assertEqual((st, json.loads(raw)["0002.png"]), (200, rec))
        got = pp.reviewed_polygons(self.pkg)
        golden = json.loads((GOLDEN / "reviewed.json").read_text(encoding="utf-8"))
        self.assertEqual({k: v for k, v in got.items() if k != "images"},
                         {k: v for k, v in golden.items() if k != "images"})
        self.assertEqual(sorted(got["images"]), sorted(golden["images"]))
        for img in golden["images"]:
            assert_same_image(self, got["images"][img], golden["images"][img])
        # package B is untouched by A's review
        self.assertFalse(any(v["reviewed"] for v in pp.reviewed_polygons(self.pkg_b)["images"].values()))

    def test_edits_on_an_unreviewed_image_do_not_count(self):
        cookie = self.login(PKG, self.key)
        g = self.label_id(self.pkg, "0002.png", 300, 100)
        self.put_decisions(cookie, PKG, "0002.png", [(f"g{g}", "del", None)])
        v = pp.reviewed_polygons(self.pkg)["images"]["0002"]
        self.assertEqual((v["reviewed"], len(v["objects"])), (False, 5))

    def test_candidate_accepted_drawn_and_reshaped_objects(self):
        cookie = self.login(PKG, self.key)
        a = self.label_id(self.pkg, "0001.png", 120, 120)
        a16 = [2 * math.pi * i / 16 for i in range(16)]
        moved = [[v for t in a16 for v in (round((120 + 44 * math.cos(t)) / W, 5), round((120 + 44 * math.sin(t)) / H, 5))]]
        drawn = [[0.1, 0.9, 0.15, 0.9, 0.15, 0.95, 0.1, 0.95]]
        self.put_decisions(cookie, PKG, "0001.png", [("p0", "add:EV", None), ("nk1", "new:inner", drawn),
                                                     (f"g{a}", "keep:EV", moved), "reviewed"])
        v = pp.reviewed_polygons(self.pkg)["images"]["0001"]
        self.assertEqual((v["reviewed"], v["reviewed_by"], len(v["objects"])), (True, ["pujan"], 6))
        by_c = {tuple(round(x) for x in centroid(o["poly"])): o for o in v["objects"]}
        d = by_c[(120, 300)]   # D, the 0.30 candidate, accepted
        self.assertEqual((d["cls"], d["conf"], d["source"], d["reviewer_class"]), (0, 0.30, "model", "EV"))
        n = next(o for o in v["objects"] if o["reviewer_class"] == "inner")
        self.assertEqual((n["cls"], n["conf"], n["source"]), (0, None, "human"))
        self.assertLess(max_dist(n["poly"], [[40, 360], [60, 360], [60, 380], [40, 380]]), 0.01)
        big = next(o for o in v["objects"] if len(o["poly"]) == 16)  # A, reshaped: human outline, model conf
        self.assertEqual((big["cls"], big["conf"], big["source"]), (0, 0.90, "human"))
        e = next(o for o in v["objects"] if o["cls"] == 1)
        self.assertEqual((e["conf"], e["source"], e["reviewer_class"]), (0.70, "model", "nonEV"))


class MultiServerTests(ServerCase):
    def test_each_package_needs_its_own_key(self):
        for pkg_id, key, other in ((PKG, self.key, self.key_b), (PKG_B, self.key_b, self.key)):
            for path in ("/", "/data.js", "/images/0001.jpg", "/api/edits", "/api/ping"):
                self.assertEqual(self.req("GET", f"/{pkg_id}{path}")[0], 403, path)
                self.assertEqual(self.req("GET", f"/{pkg_id}{path}", key=other)[0], 403, path)
                self.assertEqual(self.req("GET", f"/{pkg_id}{path}?k={other}")[0], 403, path)
                self.assertEqual(self.req("GET", f"/{pkg_id}{path}", key=key)[0], 200, path)
            st, raw, _ = self.req("PUT", f"/{pkg_id}/api/edits?f=0001.png", {"dec": {}, "t": 1}, key=other)
            self.assertEqual(st, 403)

    def test_cookies_are_per_package(self):
        ca, cb = self.login(PKG, self.key), self.login(PKG_B, self.key_b)
        self.assertEqual(self.req("GET", f"/{PKG}/data.js", cookie=ca)[0], 200)
        self.assertEqual(self.req("GET", f"/{PKG_B}/data.js", cookie=ca)[0], 403)
        self.assertEqual(self.req("GET", f"/{PKG}/data.js", cookie=cb)[0], 403)
        both = f"{ca}; {cb}"  # one browser, both links opened: both stay signed in
        self.assertEqual(self.req("GET", f"/{PKG}/data.js", cookie=both)[0], 200)
        self.assertEqual(self.req("GET", f"/{PKG_B}/data.js", cookie=both)[0], 200)

    def test_serves_index_data_images_for_both(self):
        for pkg_id, key, pkg in ((PKG, self.key, self.pkg), (PKG_B, self.key_b, self.pkg_b)):
            st, raw, r = self.req("GET", f"/{pkg_id}/", key=key)
            self.assertEqual((st, r.getheader("Content-Type")), (200, "text/html; charset=utf-8"))
            self.assertEqual(raw, (pkg / "index.html").read_bytes())
            st, raw, _ = self.req("GET", f"/{pkg_id}/data.js", key=key)
            self.assertEqual((st, raw), (200, (pkg / "data.js").read_bytes()))
            self.assertIn(f'"id":"{pkg_id}"'.encode(), raw)
            st, raw, r = self.req("GET", f"/{pkg_id}/images/0002.jpg", key=key)
            self.assertEqual((st, r.getheader("Content-Type"), raw), (200, "image/jpeg", (pkg / "images" / "0002.jpg").read_bytes()))
            st, raw, _ = self.req("POST", f"/{pkg_id}/api/claim?c=0001.png|p0&who=pujan", key=key)
            self.assertEqual((st, json.loads(raw)["claims"]), (200, {"0001.png|p0": "pujan"}))
            st, raw, _ = self.req("GET", f"/{pkg_id}/api/people", key=key)
            self.assertEqual(st, 200)

    def test_redirect_unknown_and_traversal(self):
        st, _, r = self.req("GET", f"/{PKG}?k={self.key}")
        self.assertEqual((st, r.getheader("Location")), (301, f"/{PKG}/?k={self.key}"))
        self.assertEqual(self.req("GET", "/", key=self.key)[0], 404)
        self.assertEqual(self.req("GET", "/pred_ps_20990101_ffffff/", key=self.key)[0], 404)
        self.assertEqual(self.req("GET", "/pred_ps_20990101_ffffff/data.js", key=self.key)[0], 404)
        # a package folder without the prefix is not served (the real review's packages stay with their own server)
        shutil.copytree(self.pkg, self.root / "prelipo_like_review")
        self.assertEqual(self.req("GET", "/prelipo_like_review/data.js", key=self.key)[0], 404)
        secret = self.key.encode()
        for path in ("/../review/" + PKG + "/data.js", "/..", f"/{PKG}/../{PKG_B}/data.js", f"/{PKG}/images/../.review_key",
                     f"/{PKG}/images/..\\.review_key", f"/{PKG}/images/0001.jpg:x", "/%2e%2e/x", f"/{PKG}/%2e%2e/.review_key",
                     f"/{PKG}/.review_key", f"/{PKG}/edits/0001.png.json", f"/{PKG}/prediction_package.json",
                     f"/{PKG}//data.js", "/.%2e/x", "/C:/Windows/win.ini", f"/{PKG}/images/C:x"):
            st, raw, _ = self.req("GET", path, key=self.key)
            self.assertIn(st, (400, 404), path)
            self.assertNotIn(secret, raw, path)
        for path in ("/..", f"/{PKG}/../{PKG_B}/data.js", f"/{PKG}/images/..\\.review_key", "/%2e%2e/x"):
            self.assertEqual(self.req("GET", path, key=self.key)[0], 400, path)  # refused before any package sees it
        self.assertEqual(self.req("PUT", f"/{PKG}/../{PKG_B}/api/edits?f=0001.png", {"dec": {}, "t": 1}, key=self.key)[0], 400)

    def test_package_added_after_start(self):
        late = pp.build(copy_ps(self.tmp / "late", "ps_20261106_0a0b0c"), self.root)
        key = (late / ".review_key").read_text(encoding="utf-8")
        self.assertEqual(self.req("GET", f"/{late.name}/data.js", key=key)[0], 200)
        self.assertEqual(self.req("GET", f"/{late.name}/data.js", key=self.key)[0], 403)
        (late / ".review_key").write_text("rotated-key-123", encoding="utf-8")  # a new key is picked up
        self.assertEqual(self.req("GET", f"/{late.name}/data.js", key=key)[0], 403)
        self.assertEqual(self.req("GET", f"/{late.name}/data.js", key="rotated-key-123")[0], 200)
        shutil.rmtree(late)
        self.assertEqual(self.req("GET", f"/{late.name}/data.js", key="rotated-key-123")[0], 404)


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class CliTests(Tmp):
    def test_cli_build_reviewed_spotcheck_and_server(self):
        env_py = [sys.executable, "-m"]
        r = subprocess.run(env_py + ["annotation.prediction_package", "build", "--ps", str(PS), "--review-root",
                                     str(self.root)], cwd=REPO, capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 0, r.stderr)
        pkg = self.root / PKG
        key = (pkg / ".review_key").read_text(encoding="utf-8")
        self.assertNotIn(key, r.stdout)
        r = subprocess.run(env_py + ["annotation.prediction_package", "reviewed", "--pkg", str(pkg)], cwd=REPO,
                           capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue((pkg / "reviewed.json").is_file())
        r = subprocess.run(env_py + ["annotation.prediction_package", "spotcheck", "--pkg", str(pkg), "--n", "2"],
                           cwd=REPO, capture_output=True, text=True, timeout=120)
        self.assertEqual((r.returncode, json.loads(r.stdout.strip().splitlines()[-1])), (0, ["0001", "0003"]))
        port = free_port()
        log = open(self.tmp / "server.log", "wb")
        proc = subprocess.Popen(env_py + ["annotation.multi_review_server", "--root", str(self.root), "--port", str(port)],
                                cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
        try:
            for _ in range(200):
                try:
                    c = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
                    c.request("GET", f"/{PKG}/api/ping?k={key}")
                    st = c.getresponse().status
                    c.close()
                    break
                except OSError:
                    time.sleep(0.05)
            else:
                self.fail("multi_review_server did not start")
            self.assertEqual(st, 200)
        finally:
            proc.terminate()
            proc.wait(10)
            log.close()
        out = (self.tmp / "server.log").read_text(encoding="utf-8", errors="replace")
        self.assertIn(f"/{PKG}/", out)
        self.assertNotIn(key, out)  # the startup lines and the request log never show a key


class SpotCheckTests(Tmp):
    def test_deterministic_and_spread_over_samples(self):
        pkg = self.build()
        a, b = pp.spot_check(pkg, 2), pp.spot_check(pkg, 2)
        self.assertEqual(a, b)
        self.assertEqual(pp.spot_check(PS, 2, write=False), a)  # from the prediction set: the same answer
        meta = json.loads((pkg / pp.META).read_text(encoding="utf-8"))
        self.assertEqual({meta["images"][i]["sample_label"] for i in a}, {"S1", "S2"})
        # S1's 0001 has the 0.30 candidate: weight 2 vs 0002's 1; over seeds it is picked more often
        firsts = [pp.spot_check(pkg, 2, seed=s, write=False) for s in range(60)]
        self.assertGreater(sum("0001" in f for f in firsts), sum("0002" in f for f in firsts))
        self.assertEqual(sorted(pp.spot_check(pkg, 99)), ["0001", "0002", "0003"])  # n capped at the image count
        sc = json.loads((pkg / "spotcheck.json").read_text(encoding="utf-8"))
        self.assertEqual((sc["schema"], sc["package"], sc["n"]), ("spotcheck/1", PKG, 3))
        self.assertTrue({"image", "sample_label", "n_candidates", "n_labels"} <= set(sc["images"][0]))

    def test_favours_images_with_many_candidates(self):
        """A synthetic package sidecar: in S1, image a has 10 candidates and 9 images have none; S2 has 10 images."""
        pkg = self.root / "pred_ps_20261105_000000"
        pkg.mkdir()
        imgs = {"a": {"sample_label": "S1", "candidates": [{}] * 10, "labels": []}}
        imgs.update({f"s1_{k}": {"sample_label": "S1", "candidates": [], "labels": []} for k in range(9)})
        imgs.update({f"s2_{k}": {"sample_label": "S2", "candidates": [], "labels": [{}]} for k in range(10)})
        (pkg / pp.META).write_text(json.dumps({"package": pkg.name, "prediction_set": "ps_20261105_000000",
                                               "images": imgs}), encoding="utf-8")
        picks = [pp.spot_check(pkg, 2, seed=s, write=False) for s in range(100)]
        self.assertTrue(all(sorted(imgs[i]["sample_label"] for i in p) == ["S1", "S2"] for p in picks))
        n_a = sum("a" in p for p in picks)
        self.assertGreater(n_a, 35)  # weight 11 of 20 in S1: expected ~55 of 100; a uniform pick would give ~10
        self.assertEqual(pp.spot_check(pkg, 4, seed=7, write=False), pp.spot_check(pkg, 4, seed=7, write=False))


if __name__ == "__main__":
    unittest.main()
