"""
Tests for annotation/review_server.py: the HTTP API against a live server on a free local port, plus
merge_edits / effective_labels / valid_edit on the fixture and (read-only) on the real September edits.

  python -m unittest discover -s annotation/tests -v
"""
import http.client
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from urllib.parse import quote

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
SERVER = REPO / "annotation" / "review_server.py"
sys.path.insert(0, str(REPO / "annotation"))
import review_server as rs  # noqa: E402
sys.path.insert(0, str(HERE))
import make_fixture  # noqa: E402

KEY = "test-key-123"
REAL_REL = Path("annotation review") / "cryoai_v4_review"
BOX_CRYOAI = Path.home() / "Box" / "Pujan Desai - Wang-Carney Shared" / "1. Project-Related" / "Cryo-EM" / "CryoAI"


def real_package():
    """The September review package (data.js + 33 saved edits), only ever read: $CRYOEV_REVIEW_PACKAGE, else the
    CryoAI/ folder next to the repo (also from a worktree), else the Box copy."""
    cands = [Path(os.environ["CRYOEV_REVIEW_PACKAGE"])] if os.environ.get("CRYOEV_REVIEW_PACKAGE") else []
    cands += [p / "CryoAI" / REAL_REL for p in REPO.parents] + [BOX_CRYOAI / REAL_REL]
    return next((p for p in cands if (p / "data.js").is_file() and (p / "edits").is_dir()), None)


def load_data(pkg: Path) -> dict:
    return json.loads((pkg / "data.js").read_text(encoding="utf-8").split("=", 1)[1].strip().rstrip(";"))


def free_port() -> int:
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    return port


class LiveServer:
    """review_server.py in a subprocess, serving a temp package: the fixture, or whatever build(folder) writes."""

    def __init__(self, build=make_fixture.build):
        self.dir = Path(tempfile.mkdtemp(prefix="review_pkg_"))
        self.pkg = self.dir / "pkg"
        build(self.pkg)
        self.data = load_data(self.pkg)
        self.port = free_port()
        self.proc = subprocess.Popen([sys.executable, str(SERVER), "--package", str(self.pkg), "--port", str(self.port),
                                      "--key", KEY], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for _ in range(100):
            try:
                if self.request("GET", "/api/ping")[0] == 200:
                    break
            except OSError:
                time.sleep(0.05)
        else:
            raise RuntimeError("review_server.py did not start")
        self.url = f"http://127.0.0.1:{self.port}/?k={KEY}"

    def request(self, method, path, body=None, key=KEY, headers=None):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        h = dict(headers or {})
        if key is not None:
            h["X-Review-Key"] = key
        data = json.dumps(body).encode() if body is not None and not isinstance(body, bytes) else body
        c.request(method, path, body=data, headers=h)
        r = c.getresponse()
        raw = r.read()
        c.close()
        try:
            return r.status, json.loads(raw), r
        except ValueError:
            return r.status, raw, r

    def put(self, file, edit, **kw):
        return self.request("PUT", "/api/edits?f=" + quote(file), edit, **kw)

    def close(self):
        self.proc.terminate()
        self.proc.wait(10)
        self.proc.stdout.close()
        shutil.rmtree(self.dir, ignore_errors=True)


class ServerAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = LiveServer()

    @classmethod
    def tearDownClass(cls):
        cls.srv.close()

    def saved(self, file):
        return json.loads((self.srv.pkg / "edits" / f"{file}.json").read_text(encoding="utf-8"))["edit"]

    def test_ping(self):
        code, body, _ = self.srv.request("GET", "/api/ping")
        self.assertEqual((code, body), (200, {"ok": True}))

    def test_access_key_required(self):
        edits_dir = self.srv.pkg / "edits"
        before = {f.name: f.read_bytes() for f in edits_dir.glob("*.json")}
        for method, path, body in [("GET", "/api/ping", None), ("GET", "/", None), ("GET", "/data.js", None),
                                   ("GET", "/api/edits", None), ("GET", "/api/claims", None),
                                   ("PUT", "/api/edits?f=fixC_old.png", {"t": 1, "dec": {}}),
                                   ("POST", "/api/claim?c=x&who=a", b"")]:
            for key in (None, "wrong-key"):
                with self.subTest(method=method, path=path, key=key):
                    self.assertEqual(self.srv.request(method, path, body, key=key)[0], 403)
        self.assertEqual({f.name: f.read_bytes() for f in edits_dir.glob("*.json")}, before)  # the PUT wrote nothing
        # the ?k= link sets a cookie that works on its own afterwards
        code, _, r = self.srv.request("GET", f"/?k={KEY}", key=None)
        self.assertEqual(code, 200)
        cookie = r.getheader("Set-Cookie").split(";")[0]
        self.assertEqual(self.srv.request("GET", "/api/ping", key=None, headers={"Cookie": cookie})[0], 200)

    def test_put_edit(self):
        f = "fixB_old.png"
        edit = {"t": 1000, "modified": True, "reviewed": False,
                "dec": {"g202": {"v": "keep:EV", "t": 1000, "who": "ann"}, "p1": {"v": "add:VLDL", "t": 1000, "who": "ann"}}}
        code, body, _ = self.srv.put(f, edit)
        self.assertEqual(code, 200)
        self.assertEqual(body["edit"]["dec"]["g202"]["v"], "keep:EV")
        self.assertEqual(self.saved(f)["dec"], body["edit"]["dec"])
        self.assertIn(f, self.srv.request("GET", "/api/edits")[1])
        labels = rs.effective_labels(next(i for i in self.srv.data["images"] if i["file"] == f), self.saved(f))
        self.assertEqual(sorted((g["id"] or 0, g["c"]) for g in labels), [(0, "VLDL"), (201, "EV"), (202, "EV"), (203, "MV")])

    def test_merge_two_edits_by_timestamp(self):
        f = "fixA_new.png"
        ring = make_fixture.ring(100, 100, 20)
        newer = {"t": 2000, "reviewed": True, "reviewedT": 2000, "modified": True, "dec": {
            "g103": {"v": "keep:EV", "t": 2000, "who": "bob"},
            "g104": {"v": "keep:Low conf EV", "t": 2000, "who": "bob", "p": [ring]}}}
        older = {"t": 1500, "reviewed": False, "reviewedT": 1500, "modified": True, "dec": {
            "g103": {"v": "keep:MV", "t": 1500, "who": "ann"},   # older: must lose to bob's
            "g105": {"v": "del", "t": 1500, "who": "ann"}}}      # only in ann's: must be kept
        self.assertEqual(self.srv.put(f, newer)[0], 200)
        code, body, _ = self.srv.put(f, older)
        self.assertEqual(code, 200)
        m = self.saved(f)
        self.assertEqual(m, body["edit"])
        self.assertEqual(m["dec"]["g103"]["v"], "keep:EV")
        self.assertEqual(m["dec"]["g103"]["who"], "bob")
        self.assertEqual(m["dec"]["g104"]["p"], [ring])
        self.assertEqual(m["dec"]["g105"]["v"], "del")
        self.assertTrue(m["reviewed"])
        self.assertEqual((m["t"], m["reviewedT"]), (2000, 2000))
        # a still newer decision on the same key replaces it
        self.srv.put(f, {"t": 2500, "dec": {"g103": {"v": "keep:non-EV", "t": 2500, "who": "ann"}}})
        self.assertEqual(self.saved(f)["dec"]["g103"]["v"], "keep:non-EV")
        labels = {g["id"]: g for g in rs.effective_labels(next(i for i in self.srv.data["images"] if i["file"] == f), self.saved(f))}
        self.assertNotIn(105, labels)
        self.assertEqual(labels[103]["c"], "non-EV")
        self.assertEqual(labels[104]["p"], [ring])

    def test_new_and_redraw_decisions(self):
        """The decision types added for the Editor (drawn shapes "n<id>" -> "new:<cls>", redraws as keep + p) round-trip
        through the server and apply without the tool."""
        f = "fixC_old.png"
        drawn, redrawn = make_fixture.ring(50, 50, 10, 8), make_fixture.ring(260, 150, 25, 12)
        self.srv.put(f, {"t": 3000, "modified": True, "dec": {
            "nabc1": {"v": "new:VLDL", "t": 3000, "who": "ann", "p": [drawn]},
            "nabc2": {"v": "new:EV", "t": 3000, "who": "ann", "p": [drawn]},
            "g302": {"v": "keep:MV", "t": 3000, "who": "ann", "p": [redrawn]}}})
        self.srv.put(f, {"t": 3100, "dec": {"nabc2": {"v": "del", "t": 3100, "who": "bob"}}})
        labels = rs.effective_labels(next(i for i in self.srv.data["images"] if i["file"] == f), self.saved(f))
        by = {g.get("src") or g["id"]: g for g in labels}
        self.assertEqual(set(by), {301, 302, "nabc1"})
        self.assertEqual((by["nabc1"]["c"], by["nabc1"]["p"], by["nabc1"]["id"]), ("VLDL", [drawn], None))
        self.assertEqual((by[302]["c"], by[302]["p"]), ("MV", [redrawn]))

    def test_bad_edits_rejected(self):
        ok = {"t": 1, "dec": {}}
        self.assertEqual(self.srv.put("not_in_package.png", ok)[0], 400)
        for bad in [[1, 2], {"t": "x"}, {"t": 1, "dec": {"q1": {"v": "del", "t": 1}}},
                    {"t": 1, "dec": {"n1": {"v": "new:EV", "t": 1, "p": [[0.1, 0.2]]}}},
                    {"t": 1, "dec": {"g1": {"v": 5, "t": 1}}}, {"t": 1, "gt": "x"}]:
            with self.subTest(bad=bad):
                self.assertEqual(self.srv.put("fixC_old.png", bad)[0], 400)
        self.assertEqual(self.srv.request("PUT", "/api/edits?f=fixC_old.png", b"{not json")[0], 400)

    def test_claims(self):
        def claim(case, who):
            return self.srv.request("POST", f"/api/claim?c={quote(case)}&who={who}", b"")

        code, body, _ = claim("fixA_new.png|p3", "ann")
        self.assertEqual(code, 200)
        self.assertEqual(body["claims"].get("fixA_new.png|p3"), "ann")
        self.assertEqual(claim("fixA_new.png|p3", "bob")[1]["claims"]["fixA_new.png|p3"], "ann")  # taken
        claim("fixB_old.png|editor", "ann")  # one claim per reviewer: ann's card is released
        active = self.srv.request("GET", "/api/claims")[1]["claims"]
        self.assertEqual(active.get("fixB_old.png|editor"), "ann")
        self.assertNotIn("fixA_new.png|p3", active)
        self.assertEqual(claim("fixA_new.png|p3", "bob")[1]["claims"]["fixA_new.png|p3"], "bob")


class MergeAndApply(unittest.TestCase):
    """No server: the record functions on hand-made and real records."""

    def test_old_bare_string_records_migrate(self):
        old = {"t": 50, "modified": True, "dec": {"g1": "keep:MV", "p0": "add:EV"}}
        m = rs.merge_edits(None, old)
        self.assertEqual(m["dec"]["g1"], {"v": "keep:MV", "t": 50, "who": ""})
        m2 = rs.merge_edits(old, {"t": 60, "dec": {"g1": {"v": "del", "t": 60, "who": "x"}}})
        self.assertEqual((m2["dec"]["g1"]["v"], m2["dec"]["p0"]["v"]), ("del", "add:EV"))

    def test_polygon_list_newest_wins_and_decisions_apply_on_top(self):
        im = {"gt": [{"id": 1, "c": "EV", "p": [[0, 0, 1, 0, 1, 1]]}, {"id": 2, "c": "EV", "p": [[0, 0, .5, 0, .5, .5]]}],
              "pr": [{"s": .9, "p": [[.2, .2, .3, .2, .3, .3]]}]}
        listed = [{"id": 1, "c": "MV", "p": [[0, 0, .9, 0, .9, .9]]}]  # an Editor save that dropped label 2
        a = {"t": 10, "gt": listed, "gtT": 10, "modified": True, "dec": {}}
        b = {"t": 20, "dec": {"p0": {"v": "add:inner", "t": 20, "who": "b"},
                              "g1": {"v": "keep:EV", "t": 5, "who": "b", "p": [[0, 0, .1, 0, .1, .1]]}}}
        m = rs.merge_edits(a, b)
        labels = rs.effective_labels(im, m)
        self.assertEqual([(g["id"], g["c"]) for g in labels], [(1, "EV"), (None, "inner")])
        self.assertEqual(labels[0]["p"], listed[0]["p"])  # g1's polygon (t 5) is older than the list (gtT 10)
        self.assertEqual(list(m["dec"]), ["p0", "g1"])  # key order follows mergeEdit() in the page

    def test_real_september_edits_load(self):
        pkg = real_package()
        if not pkg:
            self.skipTest("real cryoai_v4_review package not found")
        data = {im["file"]: im for im in load_data(pkg)["images"]}
        files = sorted((pkg / "edits").glob("*.json"))
        self.assertGreaterEqual(len(files), 33)
        changed = 0
        for f in files:
            rec = json.loads(f.read_text(encoding="utf-8"))
            with self.subTest(file=f.name):
                self.assertTrue(rs.valid_edit(rec["edit"]))
                m = rs.merge_edits(rs.merge_edits(None, rec["edit"]), rec["edit"])  # as the server stores a re-sent copy
                self.assertEqual(rs.merge_edits(m, rec["edit"]), m)  # merging the same record again changes nothing
                labels = rs.effective_labels(data[rec["file"]], rec["edit"])
                self.assertEqual(rs.effective_labels(data[rec["file"]], m), labels)
                base = {(g["id"], g["c"], json.dumps(g["p"])) for g in data[rec["file"]]["gt"]}
                changed += sum((g["id"], g["c"], json.dumps(g["p"])) not in base for g in labels) \
                    + len(base - {(g["id"], g["c"], json.dumps(g["p"])) for g in labels})
        self.assertGreater(changed, 0, "control: the September edits must change some labels")


if __name__ == "__main__":
    unittest.main()
