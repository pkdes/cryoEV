"""
Reviewer identity on prediction review packages (BUILD_CONTRACTS.md section 8): annotation/multi_review_server.py
trusts the portal's X-CJ-User only on a request that brings the package's key in X-Review-Key; then the page gets
window.REVIEW_IDENTITY (no name picker) and every saved decision / claim carries that id, whatever the body says.

Controls with known answers: the same server without the header, and the classic review_server.py (the real review
and the sandbox) with the header, serve index.html byte for byte as on disk and save the page's own who, as before.

Temp folders only (packages built from the CJ golden prediction set).

  python -m unittest discover -s annotation/tests -v
"""
import http.client
import json
import shutil
import sys
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest import mock
from urllib.parse import quote

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
from annotation import multi_review_server as mrs  # noqa: E402
from annotation import prediction_package as pp  # noqa: E402
from annotation import review_server as rs  # noqa: E402
import make_fixture  # noqa: E402

GOLDEN_PS = REPO / "tests" / "fixtures" / "cj_golden" / "predict" / "ps_20261105_abc123"
PKG = "pred_ps_20261105_abc123"
T0 = 1_762_400_000_000
IDENT = b"<script>window.REVIEW_IDENTITY = "  # the injected script (the tool's own code names the global too)


def serve(srv):
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def request(port, method, path, body=None, headers=None):
    """-> (status, body bytes). headers: a dict, or a list of pairs (to send one header twice)."""
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        data = body if body is None or isinstance(body, bytes) else json.dumps(body).encode()
        c.putrequest(method, path, skip_accept_encoding=True)
        for k, v in (headers.items() if isinstance(headers, dict) else headers or []):
            c.putheader(k, v)
        if data is not None:
            c.putheader("Content-Length", str(len(data)))
        c.endheaders(data)
        r = c.getresponse()
        return r.status, r.read()
    finally:
        c.close()


def pending(v, t, who):
    return {"v": v, "t": t, "who": who, "pending": True}


def old_tool_page() -> bytes:
    """review_tool/index.html as packages built before F2 hold it: the current page without the LOCKED_WHO lines
    (each F2 line names LOCKED_WHO; its comment lines start "// Portal sign-in" and the next ones follow them)."""
    lines, out, skip = mrs.TOOL_PAGE.read_bytes().splitlines(keepends=True), [], False
    for ln in lines:
        s = ln.strip()
        if s.startswith(b"// Portal sign-in"):
            skip = True
            continue
        if skip and s.startswith(b"//"):
            continue
        skip = False
        if b"LOCKED_WHO" in ln:
            if b"const KEY" in ln:  # the KEY line existed before: keep it without the ":u:" part
                out.append(ln.replace(b' + (LOCKED_WHO ? ":u:" + LOCKED_WHO : "")', b""))
            continue
        out.append(ln)
    return b"".join(out)


class MultiIdentity(unittest.TestCase):
    """The multi-package server on a temp review root holding one package built from the golden set."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="review_identity_"))
        cls.root = cls.tmp / "review"
        cls.root.mkdir()
        cls.pkg = pp.build(GOLDEN_PS, cls.root, owner="u-owner")
        cls.key = mrs.package_key(cls.pkg)
        cls.index = (cls.pkg / "index.html").read_bytes()
        cls.data = pp.read_datajs(cls.pkg)
        cls.srv = serve(mrs.make_server(cls.root, "127.0.0.1", 0, people=[]))
        cls.port = cls.srv.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self):
        for f in (self.pkg / "edits").glob("*.json"):
            f.unlink()

    def portal(self, user, **extra):
        """The headers the portal's pred proxy sends: the package key and the signed-in user."""
        return {"X-Review-Key": self.key, "X-CJ-User": user, **extra}

    def saved(self, file):
        return json.loads((self.pkg / "edits" / f"{file}.json").read_text(encoding="utf-8"))["edit"]

    def image(self, n=0):
        im = self.data["images"][n]
        return im["file"], [f"g{g['id']}" for g in im["gt"]]

    # ---- the page ----
    def test_page_carries_the_identity(self):
        st, body = request(self.port, "GET", f"/{PKG}/", headers=self.portal("u-alice"))
        self.assertEqual(st, 200)
        script = mrs.identity_script("u-alice")
        self.assertIn(b'window.REVIEW_IDENTITY = {"who": "u-alice", "locked": true};', script)
        self.assertEqual(body.count(IDENT), 1)
        self.assertLess(body.index(IDENT), body.index(b"data.js"))           # set before data.js and the tool run
        self.assertEqual(body.replace(script, b"", 1), mrs.TOOL_PAGE.read_bytes())  # the current tool, nothing else
        self.assertEqual((self.pkg / "index.html").read_bytes(), self.index)  # the file on disk is unchanged
        st, body2 = request(self.port, "GET", f"/{PKG}/index.html", headers=self.portal("u-alice"))
        self.assertEqual((st, body2), (200, body))
        st, head = request(self.port, "HEAD", f"/{PKG}/", headers=self.portal("u-alice"))
        self.assertEqual((st, head), (200, b""))

    def test_old_package_copy_gets_the_current_tool(self):
        """A package built before LOCKED_WHO keeps its old index.html: the portal gets the repo's tool (the old page
        would ask for a name and skip its own cards); a direct ?k= link still gets the package's copy."""
        old = old_tool_page()
        self.assertNotIn(b"LOCKED_WHO", old)
        try:
            (self.pkg / "index.html").write_bytes(old)
            st, body = request(self.port, "GET", f"/{PKG}/", headers=self.portal("u-alice"))
            self.assertEqual(st, 200)
            self.assertEqual(body.replace(mrs.identity_script("u-alice"), b"", 1), mrs.TOOL_PAGE.read_bytes())
            self.assertIn(b"LOCKED_WHO", body)
            st, body = request(self.port, "GET", f"/{PKG}/?k={self.key}")
            self.assertEqual((st, body), (200, old))
            with mock.patch.object(mrs, "TOOL_PAGE", self.tmp / "no-such-tool.html"):   # no repo tool: the copy
                st, body = request(self.port, "GET", f"/{PKG}/", headers=self.portal("u-alice"))
            self.assertEqual((st, body), (200, mrs.inject_identity(old, "u-alice")))
        finally:
            (self.pkg / "index.html").write_bytes(self.index)

    def test_identity_only_from_loopback(self):
        """A request from any other address (here: the server told the client is 10.1.2.3) is served as before, with
        the header ignored: the page is the package's copy and the body's who is saved."""
        srv = mrs.make_server(self.root, "127.0.0.1", 0, people=[])
        srv.finish_request = lambda sock, addr: srv.RequestHandlerClass(sock, ("10.1.2.3", addr[1]), srv)
        serve(srv)
        try:
            port = srv.server_address[1]
            st, body = request(port, "GET", f"/{PKG}/", headers=self.portal("u-alice"))
            self.assertEqual((st, body), (200, self.index))
            file, (g1, *_) = self.image()
            st, body = request(port, "PUT", f"/{PKG}/api/edits?f={quote(file)}",
                               {"t": T0 + 1, "dec": {g1: pending("keep:MV", T0 + 1, "pujan")}}, self.portal("u-alice"))
            self.assertEqual(st, 200, body)
            self.assertEqual(self.saved(file)["dec"][g1]["who"], "pujan")
            st, body = request(port, "GET", f"/{PKG}/", headers={"X-CJ-User": "u-alice"})   # and still a key
            self.assertEqual(st, 403)
        finally:
            srv.shutdown()
            srv.server_close()
        self.assertEqual(mrs.LOCAL_CLIENTS, {"127.0.0.1", "::1", "::ffff:127.0.0.1"})

    def test_control_without_identity_is_byte_identical(self):
        """Known answer: no X-CJ-User (or a key from ?k= / the cookie, which the header can't ride on) -> the file."""
        st, body = request(self.port, "GET", f"/{PKG}/", headers={"X-Review-Key": self.key})
        self.assertEqual((st, body), (200, self.index))
        st, body = request(self.port, "GET", f"/{PKG}/?k={self.key}", headers={"X-CJ-User": "mallory"})
        self.assertEqual((st, body), (200, self.index))
        cookie = f"{mrs.cookie_name(self.port, PKG)}={self.key}"
        st, body = request(self.port, "GET", f"/{PKG}/", headers={"Cookie": cookie, "X-CJ-User": "mallory"})
        self.assertEqual((st, body), (200, self.index))
        # ?k= outranks the header key, even when both are the right key: not the portal's shape, header ignored
        st, body = request(self.port, "GET", f"/{PKG}/?k={self.key}", headers=self.portal("mallory"))
        self.assertEqual((st, body), (200, self.index))

    def test_no_key_still_refused(self):
        for h in ({"X-CJ-User": "u-alice"}, {"X-CJ-User": "u-alice", "X-Review-Key": "wrong"},
                  {"X-CJ-User": "u-alice", "X-Review-Key": ""}):
            for method, path, body in (("GET", f"/{PKG}/", None), ("GET", f"/{PKG}/api/edits", None),
                                       ("PUT", f"/{PKG}/api/edits?f=0001.png", {"t": 1}),
                                       ("POST", f"/{PKG}/api/claim?c=x&who=u-alice", b"")):
                st, _ = request(self.port, method, path, body, headers=h)
                self.assertEqual(st, 403, (h, method, path))
        self.assertEqual(list((self.pkg / "edits").glob("*.json")), [])

    def test_bad_identity_refused(self):
        for h in (self.portal("<b>x</b>"), self.portal("x" * 41), self.portal(" "),
                  [("X-Review-Key", self.key), ("X-CJ-User", "u-alice"), ("X-CJ-User", "mallory")]):
            st, body = request(self.port, "GET", f"/{PKG}/", headers=h)
            self.assertEqual(st, 400, h)
            self.assertNotIn(IDENT, body)

    # ---- saves ----
    def test_forged_who_in_body_loses(self):
        file, (g1, g2, *_) = self.image()
        edit = {"t": T0 + 3, "modified": True, "reviewed": True, "reviewedT": T0 + 3, "dec": {
            g1: pending("keep:MV", T0 + 1, "mallory"),                                   # the page's new decision
            g2: {"v": "del", "t": T0 + 2, "who": "mallory", "server_t": T0 + 10 ** 9},  # forged, never stamped
            "p0": "add:EV"}}                                                             # oldest format: bare string
        st, body = request(self.port, "PUT", f"/{PKG}/api/edits?f={quote(file)}", edit, self.portal("u-alice"))
        self.assertEqual(st, 200, body)
        dec = self.saved(file)["dec"]
        self.assertEqual({k: d["who"] for k, d in dec.items()}, {g1: "u-alice", g2: "u-alice", "p0": "u-alice"})
        self.assertEqual((dec[g1]["v"], dec[g2]["v"], dec["p0"]["v"]), ("keep:MV", "del", "add:EV"))
        self.assertIn("server_t", dec[g1])
        self.assertNotIn("pending", dec[g1])
        out = pp.reviewed_polygons(self.pkg, write=False)
        self.assertEqual(out["images"][Path(file).stem]["reviewed_by"], ["u-alice"])  # contract section 3 format

    def test_echo_of_another_reviewer_keeps_their_name(self):
        """B's saved decision, echoed back in A's record (the page sends the whole record), stays B's."""
        file, (g1, g2, *_) = self.image()
        st, body = request(self.port, "PUT", f"/{PKG}/api/edits?f={quote(file)}",
                           {"t": T0 + 1, "dec": {g1: pending("keep:MV", T0 + 1, "x")}}, self.portal("u-bob"))
        self.assertEqual(st, 200, body)
        rec = json.loads(body)["edit"]                                  # what A's page now holds for this image
        rec["dec"][g2] = pending("del", T0 + 2, "u-bob")                 # A's own change, claiming to be B
        rec["t"] = T0 + 2
        st, body = request(self.port, "PUT", f"/{PKG}/api/edits?f={quote(file)}", rec, self.portal("u-alice"))
        self.assertEqual(st, 200, body)
        dec = self.saved(file)["dec"]
        self.assertEqual((dec[g1]["who"], dec[g2]["who"]), ("u-bob", "u-alice"))
        out = pp.reviewed_polygons(self.pkg, write=False)
        self.assertEqual(out["images"][Path(file).stem]["reviewed_by"], ["u-alice", "u-bob"])

    def test_control_without_identity_saves_the_pages_who(self):
        """Known answer: no identity header -> the page's who is saved, exactly as before."""
        file, (g1, *_) = self.image()
        st, body = request(self.port, "PUT", f"/{PKG}/api/edits?f={quote(file)}",
                           {"t": T0 + 1, "dec": {g1: pending("keep:MV", T0 + 1, "pujan")}},
                           {"X-Review-Key": self.key})
        self.assertEqual(st, 200, body)
        self.assertEqual(self.saved(file)["dec"][g1]["who"], "pujan")

    def test_bad_bodies_still_rejected(self):
        file, (g1, *_) = self.image()
        url = f"/{PKG}/api/edits?f={quote(file)}"
        for body, want in ((b"not json", 400), (b'{"t": NaN}', 400), (b"[1]", 400),
                           (json.dumps({"t": 1, "dec": {"zz": pending("del", 1, "x")}}).encode(), 400)):
            self.assertEqual(request(self.port, "PUT", url, body, self.portal("u-alice"))[0], want, body)
        self.assertEqual(request(self.port, "PUT", f"/{PKG}/api/edits?f=nope.png", {"t": 1},
                                 self.portal("u-alice"))[0], 400)
        self.assertEqual(list((self.pkg / "edits").glob("*.json")), [])

    def test_claim_uses_the_identity(self):
        st, body = request(self.port, "POST", f"/{PKG}/api/claim?c=0001.png%7Cg1&who=mallory", b"",
                           self.portal("u-alice"))
        self.assertEqual(st, 200, body)
        self.assertEqual(json.loads(body)["claims"].get("0001.png|g1"), "u-alice")
        st, body = request(self.port, "POST", f"/{PKG}/api/claim?c=0002.png%7Cg2", b"", self.portal("u-carol"))
        self.assertEqual(json.loads(body)["claims"].get("0002.png|g2"), "u-carol")   # no ?who= at all
        st, body = request(self.port, "POST", f"/{PKG}/api/claim?c=0003.png%7Cg3&who=pujan", b"",
                           {"X-Review-Key": self.key})                                # control: the page's name
        self.assertEqual(json.loads(body)["claims"].get("0003.png|g3"), "pujan")


class ClassicControl(unittest.TestCase):
    """review_server.py (the real review on :8766; the sandbox is the same handler with sandbox on) is untouched: with
    the key AND an X-CJ-User header, it serves index.html exactly as on disk and saves the page's own who."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="review_identity_classic_"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_classic(self, name, sandbox):
        pkg = self.tmp / name
        make_fixture.build(pkg)
        data = json.loads((pkg / "data.js").read_text(encoding="utf-8").split("=", 1)[1].strip().rstrip(";"))
        key = "classic-key"
        handler = rs.make_handler(pkg, key, {im["file"] for im in data["images"]}, data["id"], [], sandbox)
        srv = serve(ThreadingHTTPServer(("127.0.0.1", 0), handler))
        try:
            port, h = srv.server_address[1], {"X-Review-Key": key, "X-CJ-User": "u-alice"}
            st, body = request(port, "GET", "/", headers=h)
            self.assertEqual((st, body), (200, (pkg / "index.html").read_bytes()), name)
            self.assertEqual((pkg / "index.html").read_bytes(), make_fixture.TEMPLATE.read_bytes())
            self.assertNotIn(IDENT, body)
            st, js = request(port, "GET", "/data.js", headers=h)
            self.assertNotIn(b"REVIEW_IDENTITY", js)  # data.js: unchanged too
            file = data["images"][0]["file"]
            gid = f"g{data['images'][0]['gt'][0]['id']}"
            st, body = request(port, "PUT", f"/api/edits?f={quote(file)}",
                               {"t": T0 + 1, "dec": {gid: pending("del", T0 + 1, "pujan")}}, h)
            self.assertEqual(st, 200, body)
            saved = json.loads((pkg / "edits" / f"{file}.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["edit"]["dec"][gid]["who"], "pujan")
            st, body = request(port, "POST", "/api/claim?c=x&who=pujan", b"", h)
            self.assertEqual(json.loads(body)["claims"], {"x": "pujan"})
        finally:
            srv.shutdown()
            srv.server_close()

    def test_real_review_unchanged(self):
        self.run_classic("pkg", False)

    def test_sandbox_unchanged(self):
        self.run_classic("SANDBOX_pkg", True)


if __name__ == "__main__":
    unittest.main()
