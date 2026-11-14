"""
Tests for annotation/prediction_package.py and annotation/multi_review_server.py against the CryoEV golden fixtures
(tests/fixtures/cj_golden, BUILD_CONTRACTS.md sections 3 and 6). Packages are built in temp folders; the fixtures are
only read (checked byte-identical at the end).

  python -m unittest discover -s annotation/tests -v

- build: labels (conf >= 0.50) vs candidates (0.25-0.50); stub COCO; sidecar; registry record; images.csv checks;
  built in a temp folder and renamed (nothing left behind on failure); refusals.
- reviewed.json: control (no edits: every image unreviewed and unedited, objects = the prediction files at
  conf >= 0.50, parsed here independently); the golden review (0002: G deleted, F reclassed MV, reviewed) done through
  the multi server's HTTP API as the page does it (cookie from ?k=, PUT of the whole record with pending decisions)
  == reviewed.json; card decisions without Reviewed count (edited true, reviewed false); the reviewer_class map.
- multi server: two packages with their own keys and cookies; unknown id and wrong key both 403 with one body;
  aliases (case) refused; nothing made for an unauthenticated request; incomplete package not served; traversal
  400/404; non-prefix folder not served; a package added after start; index / data.js / images / claims; the CLI.
- spot check: representative (proportional, >= 1 per sample, random samples when n < samples) and prioritised.
"""
import hashlib
import http.client
import json
import math
import shutil
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
from registry.schema_check import check_record, load_schemas  # noqa: E402
sys.path.append(str(REPO / "tests"))
from cryoev_testkit import free_port  # noqa: E402  (shared helpers)

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
        self.assertEqual(sorted(data["categories"]), sorted(["EV", "Low conf EV", "MV", "inner", "Incomplete",
                                                             "Overlapping", "VLDL", "non-EV"]))  # no "(new class)"
        self.assertEqual(data["cand_conf"], 0.25)
        self.assertEqual([p.name for p in self.root.iterdir() if p.name != pp.META_LOCK_DIR], [PKG])  # no .building_ / .old_ folder left
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
        self.assertEqual(check_record(rec, load_schemas()), [])  # base_kind prediction_set: main 07f8288
        self.assertEqual(rec["path"], str(pkg.resolve()))
        # rebuild without edits: same data, same key; with edits: refused
        self.build()
        self.assertEqual((pkg / ".review_key").read_text(encoding="utf-8"), key)
        self.assertEqual(pp.read_datajs(pkg)["images"], data["images"])
        (pkg / "edits").mkdir(exist_ok=True)
        (pkg / "edits" / "0001.png.json").write_text("{}", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            self.build()

    def test_rebuild_keeps_reviewers(self):
        """Round 13: the Reviews hub writes `reviewers` into review_package.json; a rebuild keeps them (portal ids
        only, de-duplicated), still a valid registry record, and a first build has none."""
        pkg = self.build()
        rec = json.loads((pkg / pp.RECORD).read_text(encoding="utf-8"))
        self.assertNotIn("reviewers", rec)
        rec["reviewers"] = ["alyssia", "au1", "alyssia", "Bad Id", 5]
        (pkg / pp.RECORD).write_text(json.dumps(rec), encoding="utf-8")
        self.build()
        again = json.loads((pkg / pp.RECORD).read_text(encoding="utf-8"))
        self.assertEqual(again["reviewers"], ["alyssia", "au1"])
        self.assertEqual(check_record(again, load_schemas()), [])
        self.assertEqual([p.name for p in self.root.iterdir() if p.name != pp.META_LOCK_DIR], [PKG])
        (pkg / pp.RECORD).write_text("{not json", encoding="utf-8")       # unreadable: rebuilt without reviewers
        self.build()
        self.assertNotIn("reviewers", json.loads((pkg / pp.RECORD).read_text(encoding="utf-8")))

    def test_rebuild_waits_for_the_reviewers_lock(self):
        """The rebuild reads reviewers and swaps the folder under <root>/.meta_locks/<pkg>.lock (the Reviews hub's
        reviewers write takes the same OS lock): a held lock refuses cleanly after lock_wait, the old package and
        its reviewers untouched; reviewers written while the lock was held survive the next rebuild."""
        from data_utils.oslock import acquire_lock
        pkg = self.build()
        rec = json.loads((pkg / pp.RECORD).read_text(encoding="utf-8"))
        before = tree_digest(pkg)
        (self.root / pp.META_LOCK_DIR).mkdir(exist_ok=True)
        held = acquire_lock(self.root / pp.META_LOCK_DIR / f"{PKG}.lock")
        self.assertIsNotNone(held)
        try:
            rec["reviewers"] = ["alyssia"]                         # the hub's write, done under the lock
            (pkg / pp.RECORD).write_text(json.dumps(rec), encoding="utf-8")
            before = tree_digest(pkg)
            t0 = time.monotonic()
            with self.assertRaises(TimeoutError):
                self.build(lock_wait=1.0)
            self.assertGreaterEqual(time.monotonic() - t0, 0.9)    # it waited, then refused
            self.assertLess(time.monotonic() - t0, 30)
        finally:
            held.close()
        self.assertEqual(tree_digest(pkg), before)                 # the old package untouched
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), [pp.META_LOCK_DIR, PKG])   # no temp left
        released = threading.Event()
        held = acquire_lock(self.root / pp.META_LOCK_DIR / f"{PKG}.lock")

        def release():
            time.sleep(1.0)
            held.close()
            released.set()
        threading.Thread(target=release, daemon=True).start()
        self.build(lock_wait=10.0)                                 # waits for the holder, then rebuilds
        self.assertTrue(released.is_set())
        self.assertEqual(json.loads((pkg / pp.RECORD).read_text(encoding="utf-8"))["reviewers"], ["alyssia"])
        lk = acquire_lock(self.root / pp.META_LOCK_DIR / f"{PKG}.lock")   # the rebuild let go of it
        self.assertIsNotNone(lk)
        lk.close()

    def test_owner_override_and_refusals(self):
        self.assertEqual(pp.read_datajs(self.build(owner="au1"))["owner"], "au1")
        with self.assertRaises(ValueError):
            self.build(owner="Bad Owner")
        with self.assertRaises(ValueError):  # not done yet
            pp.build(copy_ps(self.tmp / "a", state="running"), self.root)
        miss = copy_ps(self.tmp / "b", "ps_20261105_bbbbbb")
        (miss / "p" / MODEL / "predictions" / "0002.txt").unlink()
        with self.assertRaises(FileNotFoundError):
            pp.build(miss, self.root)
        self.assertEqual([p.name for p in self.root.iterdir() if p.name != pp.META_LOCK_DIR], [PKG])  # a failed build leaves nothing behind
        for i, (image, file) in enumerate((("0001", "images/0001.tif"), ("0001", "../0001.png"), ("0001", "images/0002.png"),
                                           ("x001", "images/x001.png"), ("0001", "C:/images/0001.png"))):
            bad = copy_ps(self.tmp / f"csv{i}")
            rows = (bad / "images.csv").read_text(encoding="utf-8").splitlines()
            rows[1] = ",".join([image, file] + rows[1].split(",")[2:])
            (bad / "images.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError, msg=file):
                pp.build(bad, self.root)

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
                         {"schema": "reviewed/1", "package": PKG, "prediction_set": PS_ID, "model": MODEL,
                          "unknown_labels": 0})
        self.assertEqual(sorted(out["images"]), ["0001", "0002", "0003"])
        for img, v in out["images"].items():
            self.assertEqual((v["reviewed"], v["edited"], v["reviewed_by"]), (False, False, []))
            want = [(c, s, p) for c, s, p in raw_predictions(img) if s >= 0.5]
            self.assertEqual(len(v["objects"]), len(want), img)
            for o, (c, s, p) in zip(v["objects"], want):
                self.assertEqual((o["cls"], o["conf"], o["source"], o["reviewer_label"], o["reviewer_class"]),
                                 (c, s, "model", None, None))
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
        self.assertEqual({k: v for k, v in got.items() if k not in ("images", "unknown_labels")},
                         {k: v for k, v in golden.items() if k != "images"})
        self.assertEqual(got["unknown_labels"], 0)
        self.assertEqual(sorted(got["images"]), sorted(golden["images"]))
        for img in golden["images"]:
            assert_same_image(self, got["images"][img], golden["images"][img])
        self.assertEqual({i: v["edited"] for i, v in got["images"].items()}, {"0001": False, "0002": True, "0003": False})
        self.assertEqual(sorted(o["reviewer_label"] for o in got["images"]["0002"]["objects"]),
                         ["EV", "MV", "non-EV", "non-EV"])
        # package B is untouched by A's review
        self.assertFalse(any(v["reviewed"] for v in pp.reviewed_polygons(self.pkg_b)["images"].values()))

    def test_card_decisions_without_reviewed_count(self):
        """The phone cards never press Reviewed: their decisions still show (edited true, reviewed false)."""
        cookie = self.login(PKG, self.key)
        g, f = self.label_id(self.pkg, "0002.png", 300, 100), self.label_id(self.pkg, "0002.png", 100, 100)
        self.put_decisions(cookie, PKG, "0002.png", [(f"g{g}", "del", None), (f"g{f}", "keep:MV", None)])
        self.put_decisions(cookie, PKG, "0001.png", [("p0", "add:EV", None)])  # the 0.30 candidate, accepted on a card
        out = pp.reviewed_polygons(self.pkg)
        v = out["images"]["0002"]
        self.assertEqual((v["reviewed"], v["edited"], v["reviewed_by"]), (False, True, ["pujan"]))
        golden = json.loads((GOLDEN / "reviewed.json").read_text(encoding="utf-8"))["images"]["0002"]
        assert_same_image(self, dict(v, reviewed=True), golden)  # the same objects as the golden review
        v1 = out["images"]["0001"]
        self.assertEqual((v1["reviewed"], v1["edited"], len(v1["objects"])), (False, True, 5))
        self.assertIn((0, 0.30, "model", "EV"), [(o["cls"], o["conf"], o["source"], o["reviewer_class"])
                                                 for o in v1["objects"]])
        self.assertEqual((out["images"]["0003"]["edited"], out["images"]["0003"]["reviewed"]), (False, False))

    def test_reviewed_without_decisions(self):
        cookie = self.login(PKG, self.key)
        self.put_decisions(cookie, PKG, "0003.png", ["reviewed"])
        v = pp.reviewed_polygons(self.pkg)["images"]["0003"]
        self.assertEqual((v["reviewed"], v["edited"], v["reviewed_by"]), (True, False, []))  # authors of decisions
        self.assertEqual([(o["cls"], o["reviewer_label"], o["reviewer_class"]) for o in v["objects"]],
                         [(0, "EV", "EV")] * 3)

    def test_reviewer_class_map_and_unknown_labels(self):
        cookie = self.login(PKG, self.key)
        lid = {n: self.label_id(self.pkg, "0002.png", x, y) for n, x, y in
               (("F", 100, 100), ("G", 300, 100), ("H", 100, 300), ("I", 290, 290), ("J", 290, 290))}
        lid["J"] = next(g["id"] for g in pp.read_datajs(self.pkg)["images"][1]["gt"]
                        if g["id"] != lid["I"] and abs(g["p"][0][0] * W - 318) < 1)
        drawn = [[0.1, 0.9, 0.15, 0.9, 0.15, 0.95, 0.1, 0.95]]
        self.put_decisions(cookie, PKG, "0002.png", [
            (f"g{lid['F']}", "keep:Low conf EV", None), (f"g{lid['G']}", "keep:VLDL", None),
            (f"g{lid['H']}", "keep:Incomplete", None), (f"g{lid['I']}", "keep:Weird", None),
            (f"g{lid['J']}", "keep:Overlapping", None), ("nz1", "new:Other thing", drawn)])
        out = pp.reviewed_polygons(self.pkg)
        got = sorted((o["reviewer_label"], o["reviewer_class"], o["cls"]) for o in out["images"]["0002"]["objects"])
        self.assertEqual(got, sorted([("Low conf EV", "EV", 0), ("VLDL", "nonEV", 1), ("Incomplete", "EV", 0),
                                      ("Weird", None, 1),         # unknown: the model object's class (I is non-EV)
                                      ("Overlapping", "EV", 0), ("Other thing", None, 0)]))  # drawn, unknown: 0
        self.assertEqual(out["unknown_labels"], 2)

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
        self.assertEqual(self.req("GET", f"/{PKG}")[0], 403)  # no redirect without the key
        self.assertEqual(self.req("GET", "/", key=self.key)[0], 404)
        # an unknown id answers exactly as a wrong key does: ids can't be enumerated
        unknown = [self.req(m, "/pred_ps_20990101_ffffff/data.js", key=self.key)[:2] for m in ("GET", "PUT")]
        wrong = [self.req(m, f"/{PKG}/data.js", key="not-the-key")[:2] for m in ("GET", "PUT")]
        self.assertEqual(unknown, wrong)
        self.assertEqual([s for s, _ in unknown], [403, 403])
        self.assertEqual(self.req("GET", "/pred_ps_20990101_ffffff/", key=self.key)[0], 403)
        # a package folder without the prefix is not served (the real review's packages stay with their own server)
        shutil.copytree(self.pkg, self.root / "prelipo_like_review")
        self.assertEqual(self.req("GET", "/prelipo_like_review/data.js", key=self.key)[0], 403)
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
        self.assertEqual(self.req("GET", f"/{late.name}/data.js", key="rotated-key-123")[0], 403)

    def test_aliases_and_nothing_made_without_the_key(self):
        cache = self.srv.RequestHandlerClass.packages._cache
        self.assertEqual(cache, {})
        for key in (None, "wrong", self.key_b):
            self.assertEqual(self.req("GET", f"/{PKG}/api/edits", key=key)[0], 403)
            self.assertEqual(self.req("PUT", f"/{PKG}/api/edits?f=0001.png", {"dec": {}, "t": 1}, key=key)[0], 403)
        self.assertEqual(cache, {})
        self.assertFalse((self.pkg / "edits").exists())  # make_handler's mkdir only runs for a keyed request
        # another spelling of the folder (case on Windows; 8.3 names likewise) is not a second package
        for alias in ("pred_" + PS_ID.upper(), PKG.replace("abc123", "ABC123")):  # the prefix still matches
            st = self.req("GET", f"/{alias}/data.js", key=self.key)[0]
            self.assertEqual(st, 403, alias)
        self.assertEqual(cache, {})
        self.assertEqual(self.req("GET", f"/{PKG}/api/edits", key=self.key)[0], 200)
        self.assertEqual(list(cache), [self.pkg.resolve()])
        self.assertTrue((self.pkg / "edits").is_dir())

    def test_incomplete_package_not_served(self):
        part = self.root / "pred_ps_20261107_aaaaaa"
        shutil.copytree(self.pkg, part)
        key = (part / ".review_key").read_text(encoding="utf-8")
        (part / "review_package.json").unlink()  # the builder writes it last
        self.assertEqual(self.req("GET", f"/{part.name}/data.js", key=key)[0], 403)
        (part / "review_package.json").write_text("{}", encoding="utf-8")
        self.assertEqual(self.req("GET", f"/{part.name}/data.js", key=key)[0], 200)
        (part / ".review_key").unlink()  # no key file: not served (the server never makes one)
        self.assertEqual(self.req("GET", f"/{part.name}/data.js", key=key)[0], 403)
        self.assertEqual(self.req("GET", f"/{part.name}/data.js", key="")[0], 403)
        self.assertFalse((part / ".review_key").exists())


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
        self.assertEqual((r.returncode, json.loads(r.stdout.strip().splitlines()[-1])),
                         (0, pp.spot_check(pkg, 2, write=False)))
        self.assertEqual(json.loads((pkg / "spotcheck.json").read_text(encoding="utf-8"))["mode"], "representative")
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
    def synthetic(self, sizes, cands=None):
        """A package sidecar only (spot_check reads nothing else): sizes {sample: n images}; cands {image: n}."""
        pkg = self.root / "pred_ps_20261105_000000"
        pkg.mkdir(exist_ok=True)
        imgs = {f"{s}_{k:02d}": {"sample_label": s, "candidates": [{}] * (cands or {}).get(f"{s}_{k:02d}", 0),
                                 "labels": []} for s, n in sizes.items() for k in range(n)}
        (pkg / pp.META).write_text(json.dumps({"package": pkg.name, "prediction_set": "ps_20261105_000000",
                                               "images": imgs}), encoding="utf-8")
        return pkg, imgs

    def test_golden_deterministic_spread_and_labelled(self):
        pkg = self.build()
        for mode in pp.SPOT_MODES:
            a = pp.spot_check(pkg, 2, mode=mode)
            self.assertEqual(a, pp.spot_check(pkg, 2, mode=mode))
            self.assertEqual(pp.spot_check(PS, 2, write=False, mode=mode), a)  # from the prediction set: the same
            meta = json.loads((pkg / pp.META).read_text(encoding="utf-8"))
            self.assertEqual({meta["images"][i]["sample_label"] for i in a}, {"S1", "S2"})
            sc = json.loads((pkg / "spotcheck.json").read_text(encoding="utf-8"))
            self.assertEqual((sc["schema"], sc["package"], sc["n"], sc["mode"], sc["rule"]),
                             ("spotcheck/1", PKG, 2, mode, pp.SPOT_MODES[mode]))
            self.assertEqual(sorted(pp.spot_check(pkg, 99, mode=mode, write=False)), ["0001", "0002", "0003"])
        self.assertEqual(pp.spot_check.__defaults__[-1], "representative")
        with self.assertRaises(ValueError):
            pp.spot_check(pkg, 2, mode="best")

    def test_representative_is_proportional_and_uniform(self):
        pkg, imgs = self.synthetic({"A": 30, "B": 10, "C": 2}, cands={"A_00": 50})
        sample = lambda i: imgs[i]["sample_label"]
        for seed in range(20):
            got = pp.spot_check(pkg, 8, seed=seed, write=False)
            # exact 5.71 / 1.90 / 0.38; C raised to 1, then the largest remainders: A 5, B 2, C 1
            self.assertEqual(sorted(map(sample, got)), ["A"] * 5 + ["B"] * 2 + ["C"])
            self.assertEqual(len(set(got)), 8)
        self.assertEqual(sorted(map(sample, pp.spot_check(pkg, 3, write=False))), ["A", "B", "C"])
        # uniform within a sample: A_00's 50 candidates don't matter here (1 of 30 -> ~6 of 200 draws of 6)
        n_a00 = sum("A_00" in pp.spot_check(pkg, 8, seed=s, write=False) for s in range(200))
        self.assertLess(n_a00, 80)
        # n < samples: samples chosen at random, not the first ones alphabetically
        pkg, imgs = self.synthetic({s: 3 for s in "ABCDE"})
        chosen = [tuple(sorted(map(sample, pp.spot_check(pkg, 2, seed=s, write=False)))) for s in range(40)]
        self.assertTrue(all(len(set(c)) == 2 for c in chosen))
        self.assertGreater(len(set(chosen)), 4)
        self.assertGreater(sum("E" in c for c in chosen), 5)

    def test_prioritised_favours_images_with_many_candidates(self):
        """In S1, image S1_00 has 10 candidates and 9 images have none; S2 has 10 images."""
        pkg, imgs = self.synthetic({"S1": 10, "S2": 10}, cands={"S1_00": 10})
        picks = [pp.spot_check(pkg, 2, seed=s, write=False, mode="prioritised") for s in range(100)]
        self.assertTrue(all(sorted(imgs[i]["sample_label"] for i in p) == ["S1", "S2"] for p in picks))
        self.assertGreater(sum("S1_00" in p for p in picks), 35)  # weight 11 of 20: ~55; uniform would give ~10
        rep = sum("S1_00" in pp.spot_check(pkg, 2, seed=s, write=False) for s in range(100))
        self.assertLess(rep, 25)  # representative: uniform, ~10


if __name__ == "__main__":
    unittest.main()
