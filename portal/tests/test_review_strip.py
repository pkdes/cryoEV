"""
Round 13 (B3): the review switcher strip, the /review/ pin and the audit owner's refusal (portal/app.py).

- render_strip: escaped, the current package marked (not a link), items link to their go_url only when it is a
  same-site path, at most STRIP_PER_GROUP per group, "D of T cards", the "All reviews" links.
- Through the portal (temp roster, temp review root with lab_ fixture packages, a dummy multi server, a temp services
  config; review_index.list_reviews / switcher_groups replaced by fakes, B1 fills the real ones):
  the strip on /review/<pkg>/ above the frame; on /review/ only with CJ_REVIEW_STRIP_CORE on; any error or a slow
  list = no strip and the page still answers (fast, and the event loop keeps serving others); list_reviews gets the
  person (show_all only for admins).
- A blind audit's owner gets 403 JSON on PUT api/edits / POST api/group (any spelling); reads go on; reviewers and
  revise packages are untouched; nothing reaches the multi server.
- registry_review_package: the services config's port-8766 review_server.py package wins over a newer lab_ revise job
  in the registry; the registry only when the config can't be read; check_review_package warns when they differ.

Never the real services config, registry or review root.

  python -m unittest portal.tests.test_review_strip -v
"""
import json
import os
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cjtest as T  # noqa: E402
from test_shell_pred import MultiDummy  # noqa: E402
from portal import app as APP  # noqa: E402
from portal import auth as A  # noqa: E402
from portal import review_index as RI  # noqa: E402
from annotation import multi_review_server as MRS  # noqa: E402
sys.path.insert(0, str(T.REPO / "annotation" / "tests"))
import make_fixture  # noqa: E402

AUDIT = "lab_compare_audit_20261113"      # a blind audit: owner u-adv, reviewer u-basic
REVISE = "lab_compare_revise_20261113"    # purpose revise: owner u-adv
CORE = "lab_compare_core_20261113"        # the portal's core package (/review/)
SVC = "lab_compare_single_20261113"       # the port-8766 single server's package in the services config


def row(pid, kind="label_review", title=None, **kw):
    r = {"id": pid, "kind": kind, "title": title or pid, "made_from": "", "owner": "u-adv", "reviewers": [],
         "role": "yours", "status": "in_progress", "cards": 10, "decided": 4, "n_images": 4, "last": None,
         "open_url": f"/review/{pid}/", "go_url": f"/reviews/go/{pid}", "mode": "cards", "needs_you": False,
         "can_assign": True, "blind": kind == "audit", "via": None, "facets": {}}
    r.update(kw)
    return r


def groups_of(rows, current_id, per_group=8):
    need = [r for r in rows if r.get("needs_you")]
    out = [("Needs you", need[:per_group])] if need else []
    for kind in RI.KINDS:
        rs = [r for r in rows if r["kind"] == kind and not r.get("needs_you")]
        if rs:
            out.append((APP.STRIP_KIND[kind], rs[:per_group]))
    return out


def build_pkg(root: Path, name: str, owner="u-adv", reviewers=("u-basic",), **extra):
    pkg = root / name
    make_fixture.build(pkg)
    (pkg / "review_package.json").write_text(json.dumps({"kind": "compare", "owner": owner,
                                                         "reviewers": list(reviewers), "n_images": 4, **extra}))
    (pkg / ".review_key").write_text(f"key-{name}", encoding="utf-8")
    return pkg


def services(path: Path, *svcs):
    path.write_text(json.dumps({"services": list(svcs)}), encoding="utf-8")
    return path


def review_svc(pkg, port, enabled=True, name="review"):
    return {"name": name, "enabled": enabled, "port": port, "cwd": "{repo}",
            "argv": ["{python}", "annotation/review_server.py", "--package", str(pkg), "--port", str(port)]}


class RenderStrip(unittest.TestCase):
    def test_current_marked_and_items_link_to_go(self):
        rows = [row("lab_a", title="A <b>", needs_you=True), row("lab_b", kind="audit", title="Audit B"),
                row("pred_ps_20261113_aaaaaa", kind="prediction", title="P", status="applied")]
        h = APP.render_strip(rows, "lab_a", groups_of(rows, "lab_a"))
        self.assertIn('href="/reviews/"', h)
        self.assertIn("All reviews", h)
        self.assertIn('<span class="cj-rs-kind" data-kind="label_review">Label set</span>', h)
        self.assertIn("A &lt;b&gt;", h)
        self.assertNotIn("<b>A", h)
        self.assertNotIn('href="/reviews/go/lab_a"', h)               # the current one is not a link
        self.assertIn('aria-current="page"', h)
        self.assertIn('href="/reviews/go/lab_b"', h)
        self.assertIn('href="/reviews/go/pred_ps_20261113_aaaaaa"', h)
        self.assertIn(">blind<", h)
        self.assertIn(">Applied<", h)
        self.assertIn("<b>4</b>", h)
        self.assertIn("<b>10</b>", h)
        self.assertIn("&#160;of&#160;", h)
        self.assertIn("&#160;cards</span>", h)
        self.assertLess(h.index("Needs you"), h.index("Audit B"))      # "Needs you" first

    def test_at_most_per_group_and_more_link(self):
        rows = [row(f"lab_n{i:02d}", needs_you=True) for i in range(11)]
        h = APP.render_strip(rows, "lab_n00", [("Needs you", rows)])   # even if handed more
        self.assertEqual(h.count('class="cj-rs-item'), APP.STRIP_PER_GROUP)
        self.assertIn("3 more in All reviews", h)

    def test_unsafe_go_urls_dropped(self):
        rows = [row("lab_x", go_url="javascript:alert(1)"), row("lab_y", go_url="//evil.example/x"),
                row("lab_z", go_url="/reviews/go/lab_z\\x"), row("lab_ok")]
        h = APP.render_strip(rows, None, [("Label set", rows)])
        self.assertNotIn("javascript:", h)
        self.assertNotIn("evil.example", h)
        self.assertNotIn("lab_z", h)
        self.assertIn('href="/reviews/go/lab_ok"', h)

    def test_no_current_row(self):
        h = APP.render_strip([], "lab_gone", [])
        self.assertIn("Switch review", h)
        self.assertIn("No other reviews yet.", h)
        self.assertNotIn("cj-rs-prog", h)                               # no counts without the row
        h = APP.render_strip([row("lab_a", cards=None, decided=None)], "lab_a", [])
        self.assertNotIn("cj-rs-prog", h)

    def test_flag_values(self):
        for v, on in (("on", True), ("1", True), ("TRUE", True), ("yes", True), ("", False), ("off", False),
                      ("0", False), ("no", False), ("nope", False)):
            self.assertEqual(APP.strip_core_on({APP.STRIP_CORE_ENV: v}), on, v)
        self.assertFalse(APP.strip_core_on({}))


class StripThroughThePortal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._env = {k: os.environ.pop(k) for k in list(os.environ) if k.startswith(("CRYOEV_", "CJ_"))}
        cls.up = MultiDummy()
        cls.e = T.Env()
        cls.root = cls.e.tmp / "review"
        cls.root.mkdir()
        build_pkg(cls.root, AUDIT)
        build_pkg(cls.root, REVISE, purpose="revise")
        build_pkg(cls.root, CORE)
        cls.svc = services(cls.e.tmp / "services.json", review_svc(cls.root / SVC, 8766))
        os.environ.update({"CRYOEV_SERVICES_CONFIG": str(cls.svc), MRS.PREFIX_ENV: "pred_,lab_"})
        cls.keys = {"admin": cls.e.key, "basic": A.add_key(cls.e.roster, "u-basic", "basic"),
                    "adv": A.add_key(cls.e.roster, "u-adv", "advanced")}
        cls.p = T.Portal(cls.e.config(pred_upstream=cls.up.url, review_root=cls.root,
                                      review_package=cls.root / CORE))
        cls.tok = {w: T.login(cls.p.port, k) for w, k in cls.keys.items()}

    @classmethod
    def tearDownClass(cls):
        cls.p.stop()
        cls.up.stop()
        cls.e.cleanup()
        for k in ("CRYOEV_SERVICES_CONFIG", MRS.PREFIX_ENV, APP.STRIP_CORE_ENV):
            os.environ.pop(k, None)
        os.environ.update(cls._env)

    def setUp(self):
        self.calls = []
        self.rows = [row(AUDIT, kind="audit", title="Audit fixture", needs_you=True),
                     row(REVISE, title="Revise fixture"), row(CORE, kind="audit", title="Core audit", open_url="/review/")]

        def fake_list(cfg, user, can_see, *, show_all=False, access_mode=False):
            self.calls.append((user.get("id"), show_all, access_mode))
            return list(self.rows)
        for name, fn in (("list_reviews", fake_list), ("switcher_groups", groups_of)):
            p = mock.patch.object(RI, name, fn)
            p.start()
            self.addCleanup(p.stop)

    def req(self, who, method, path, body=None, headers=None):
        h = {**T.cookie(self.tok[who]), **(headers or {})}
        if method in APP.UNSAFE:
            h.setdefault("Origin", self.p.origin)
        st, hd, b = T.request(self.p.port, method, path, h, body)
        return st, hd, b.decode("utf-8", "replace")

    def test_strip_on_a_package_above_the_frame(self):
        st, _, body = self.req("adv", "GET", f"/review/{REVISE}/")
        self.assertEqual(st, 200)
        self.assertIn('id="cj-rstrip"', body)
        self.assertLess(body.index('id="cj-rstrip"'), body.index('id="cj-review-frame"'))
        self.assertIn("Revise fixture", body)
        self.assertIn(f'href="/reviews/go/{AUDIT}"', body)
        self.assertNotIn(f'href="/reviews/go/{REVISE}"', body)          # current
        self.assertEqual(self.calls[-1], ("u-adv", False, False))       # not admin: own rows only

    def test_admin_lists_everyone(self):
        self.req("admin", "GET", f"/review/{REVISE}/")
        self.assertEqual(self.calls[-1], (T.USER, True, False))

    def test_core_page_needs_the_flag(self):
        os.environ.pop(APP.STRIP_CORE_ENV, None)
        st, _, body = self.req("admin", "GET", "/review/")
        self.assertEqual(st, 200)
        self.assertNotIn("cj-rstrip", body)
        self.assertIn('id="cj-review-frame"', body)
        self.assertEqual(self.calls, [])                                # not even built
        for v in ("on", "1", "true", "yes"):
            os.environ[APP.STRIP_CORE_ENV] = v
            try:
                st, _, body = self.req("admin", "GET", "/review/")
            finally:
                os.environ.pop(APP.STRIP_CORE_ENV, None)
            self.assertEqual(st, 200)
            self.assertIn('id="cj-rstrip"', body, v)
            self.assertIn("Core audit", body)                           # the core package is the current one
            self.assertNotIn(f'href="/reviews/go/{CORE}"', body)
        os.environ[APP.STRIP_CORE_ENV] = "off"
        try:
            self.assertNotIn("cj-rstrip", self.req("admin", "GET", "/review/")[2])
        finally:
            os.environ.pop(APP.STRIP_CORE_ENV, None)

    def test_error_means_no_strip(self):
        def boom(*a, **k):
            raise RuntimeError("secret detail C:/x")
        with mock.patch.object(RI, "list_reviews", boom), self.assertLogs("cj", "WARNING") as logs:
            st, _, body = self.req("adv", "GET", f"/review/{REVISE}/")
        self.assertEqual(st, 200)
        self.assertNotIn("cj-rstrip", body)
        self.assertIn('id="cj-review-frame"', body)
        self.assertNotIn("secret detail", " ".join(logs.output))       # the type only
        with mock.patch.object(RI, "switcher_groups", boom):
            st, _, body = self.req("adv", "GET", f"/review/{REVISE}/")
        self.assertEqual((st, "cj-rstrip" in body), (200, False))

    def test_contract_stub_means_no_strip(self):
        """Before B1's bodies land, the contract raises NotImplementedError: the shell still works."""
        def stub(*a, **k):
            raise NotImplementedError("B1 fills this in")
        with mock.patch.object(RI, "list_reviews", stub):
            st, _, body = self.req("adv", "GET", f"/review/{REVISE}/")
        self.assertEqual((st, "cj-rstrip" in body), (200, False))

    def test_slow_list_means_no_strip_and_the_loop_keeps_serving(self):
        release = threading.Event()

        def slow(*a, **k):
            release.wait(3)
            return list(self.rows)
        try:
            with mock.patch.object(RI, "list_reviews", slow):
                got = {}

                def shell():
                    t0 = time.monotonic()
                    got["shell"] = self.req("adv", "GET", f"/review/{REVISE}/")
                    got["shell_s"] = time.monotonic() - t0
                th = threading.Thread(target=shell)
                th.start()
                time.sleep(0.05)
                t0 = time.monotonic()
                st, _, _ = self.req("adv", "GET", "/healthz")           # answered while the strip is still building
                hz = time.monotonic() - t0
                th.join(5)
            self.assertEqual(st, 200)
            self.assertLess(hz, 0.25)
            st, _, body = got["shell"]
            self.assertEqual(st, 200)
            self.assertNotIn("cj-rstrip", body)
            self.assertIn('id="cj-review-frame"', body)
            self.assertLess(got["shell_s"], APP.STRIP_BUDGET_S + 1.0)    # the budget, not the 3 s list
        finally:
            release.set()

    def test_unknown_id_is_404_without_building(self):
        self.assertEqual(self.req("basic", "GET", f"/review/{REVISE}/")[0], 200)     # a listed reviewer: opens
        n = len(self.calls)
        self.assertEqual(self.req("basic", "GET", "/review/lab_compare_nope_20261113/")[0], 404)
        self.assertEqual(len(self.calls), n)

    # ---- the audit owner can't decide cards
    def test_audit_owner_cannot_decide(self):
        n = len(self.up.seen)
        for method, path in (("PUT", "api/edits?f=a.png"), ("POST", "api/group?who=x"), ("PUT", "api/edits/"),
                             ("PUT", "api/%65dits?f=a.png"), ("PUT", "API/Edits"), ("POST", "api/%2567roup")):
            with self.subTest(method=method, path=path):
                st, hd, body = self.req("adv", method, f"/review/{AUDIT}/app/{path}", body=b"{}",
                                        headers={"Content-Type": "application/json"})
                self.assertEqual(st, 403)
                self.assertIn("application/json", T.hget(hd, "content-type"))
                self.assertIn("can't decide its cards", json.loads(body)["error"])
        self.assertEqual(len(self.up.seen), n)                           # nothing reached the multi server

    def test_audit_owner_reads_and_others_write(self):
        n = len(self.up.seen)
        self.assertEqual(self.req("adv", "GET", f"/review/{AUDIT}/app/api/edits")[0], 200)     # owner reads
        self.assertEqual(self.req("adv", "POST", f"/review/{AUDIT}/app/api/claim", b"{}")[0], 200)
        self.assertEqual(self.req("basic", "PUT", f"/review/{AUDIT}/app/api/edits?f=a.png", b"{}")[0], 200)  # reviewer
        self.assertEqual(self.req("adv", "PUT", f"/review/{REVISE}/app/api/edits?f=a.png", b"{}")[0], 200)  # revise
        self.assertEqual(self.req("adv", "POST", f"/review/{REVISE}/app/api/group", b"{}")[0], 200)
        self.assertEqual(len(self.up.seen), n + 5)

    def test_audit_owner_rule(self):
        u = {"id": "u-adv", "role": "admin"}
        self.assertTrue(APP.audit_owner(u, {"kind": "compare", "owner": "u-adv"}))
        self.assertTrue(APP.audit_owner(u, {"kind": "compare", "owner": "u-adv", "purpose": "audit"}))
        self.assertTrue(APP.audit_owner(u, {"kind": "compare", "owner": "u-adv", "reviewers": ["u-adv"]}))
        self.assertFalse(APP.audit_owner(u, {"kind": "compare", "owner": "u-adv", "purpose": "revise"}))
        self.assertFalse(APP.audit_owner(u, {"kind": "compare", "owner": "someone"}))     # an admin, not the owner
        self.assertFalse(APP.audit_owner(u, {"kind": "prediction", "owner": "u-adv"}))
        self.assertFalse(APP.audit_owner(u, {"kind": "compare", "owner": None}))
        self.assertFalse(APP.audit_owner(None, {"kind": "compare", "owner": "u-adv"}))


class ReviewPackagePin(unittest.TestCase):
    """registry_review_package: the services config's port-8766 package first; the registry only when unreadable."""

    def setUp(self):
        self.env = T.Env()
        self.addCleanup(self.env.cleanup)
        self.audit = self.env.tmp / "review" / "lab_compare_test_audit_20261111"
        self.newer = self.env.tmp / "review" / "lab_compare_newer_20261113"
        for p, k in ((self.audit, "audit-key"), (self.newer, "newer-key")):
            p.mkdir(parents=True)
            (p / ".review_key").write_text(k, encoding="utf-8")
        jobs = [{"id": "job_20261111_audit", "type": "revise", "port": 8766, "status": "in_progress", "package": "audit"},
                {"id": "job_20261113_newer", "type": "revise", "port": 8766, "status": "in_progress", "package": "newer"}]
        pkgs = [{"id": "audit", "path": str(self.audit)}, {"id": "newer", "path": str(self.newer)}]
        T.TK.write_registry(self.env.registry, {ent: [{"schema": f"{ent}/1", **r} for r in rows]
                                                for ent, rows in (("job", jobs), ("review_package", pkgs))}, samples=False)
        self.cfg = self.env.tmp / "services.json"
        services(self.cfg, review_svc(self.audit, 8766),
                 review_svc(self.env.tmp / "review" / "SANDBOX_x", 8765, name="sandbox"),
                 {"name": "review-multi", "port": 8767, "argv": ["{python}", "-m", "annotation.multi_review_server",
                                                                 "--root", str(self.env.tmp / "review"), "--port", "8767"]})
        self.envd = {"CRYOEV_SERVICES_CONFIG": str(self.cfg)}

    def pick(self, environ=None):
        return APP.registry_review_package(self.env.registry, environ=self.envd if environ is None else environ)

    def test_configured_package_beats_a_newer_lab_revise_job(self):
        self.assertEqual(self.pick(), self.audit)
        self.assertEqual(APP.read_review_key(self.pick()), "audit-key")
        # control: without a readable config the registry's newest job wins (the hijack this guards against)
        self.assertEqual(self.pick({"CRYOEV_SERVICES_CONFIG": str(self.env.tmp / "missing.json")}), self.newer)

    def test_unreadable_configs_fall_back_to_the_registry(self):
        for text in ("{broken", "[]", "{}", '{"services": []}', '{"services": [{"argv": "not a list"}]}',
                     '{"services": [{"argv": ["review_server.py", "--port", "8766"]}]}'):    # 8766, no package
            with self.subTest(text=text):
                self.cfg.write_text(text, encoding="utf-8")
                self.assertEqual(self.pick(), self.newer)

    def test_readable_without_8766_is_none(self):
        services(self.cfg, review_svc(self.env.tmp / "review" / "SANDBOX_x", 8765, name="sandbox"))
        self.assertIsNone(self.pick())

    def test_enabled_wins_and_ambiguity_is_none(self):
        services(self.cfg, review_svc(self.newer, 8766, enabled=False), review_svc(self.audit, 8766))
        self.assertEqual(self.pick(), self.audit)
        services(self.cfg, review_svc(self.newer, 8766, enabled=False))         # only a disabled one: still it
        self.assertEqual(self.pick(), self.newer)
        services(self.cfg, review_svc(self.newer, 8766), review_svc(self.audit, 8766, name="review2"))
        with self.assertLogs("cj", "WARNING"):
            self.assertIsNone(self.pick())

    def test_forms_of_argv(self):
        for argv in (["{python}", "annotation/review_server.py", f"--package={self.audit}", "--port=8766"],
                     ["cmd", "/c", f'python annotation\\review_server.py --package "{self.audit}" --port 8766'],
                     ["{python}", "-m", "annotation.review_server", "--port", "8766", "--package", str(self.audit)]):
            with self.subTest(argv=argv):
                self.cfg.write_text(json.dumps({"services": [{"name": "review", "argv": argv}]}), encoding="utf-8")
                self.assertEqual(self.pick(), self.audit)

    def test_repo_placeholder_and_utf16(self):
        raw = json.dumps({"services": [{"name": "r", "argv": ["{python}", "annotation/review_server.py",
                                                               "--package", "{repo}/x/pkg", "--port", "8766"]}]})
        self.cfg.write_bytes(raw.encode("utf-16"))
        self.assertEqual(self.pick(), Path(str(APP.REPO) + "/x/pkg"))

    def test_startup_warning(self):
        self.assertIsNone(APP.check_review_package(self.audit, self.envd))
        self.assertIsNone(APP.check_review_package(Path(str(self.audit).upper()), self.envd)
                          if os.name == "nt" else None)
        with self.assertLogs("cj", "WARNING") as logs:
            msg = APP.check_review_package(self.newer, self.envd)       # an explicit --review-package elsewhere
        self.assertIn("NOT the port 8766", msg)
        self.assertIn(self.audit.name.lower(), " ".join(logs.output))
        with self.assertLogs("cj", "WARNING"):
            self.assertIn("can't be read", APP.check_review_package(self.audit, {"CRYOEV_SERVICES_CONFIG":
                                                                                str(self.env.tmp / "missing.json")}))

    def test_explicit_review_package_still_wins(self):
        args = APP.argparse.Namespace(registry=self.env.registry, review_package=self.newer, roster=self.env.roster,
                                      state_dir=self.env.state, review_upstream="http://127.0.0.1:1", origin=None,
                                      auth="key")
        with mock.patch.dict(os.environ, self.envd), self.assertLogs("cj", "WARNING"):
            cfg = APP.build_config(args)
        self.assertEqual((cfg.review_package, cfg.review_key), (self.newer, "newer-key"))
        args.review_package = None
        with mock.patch.dict(os.environ, self.envd):
            cfg = APP.build_config(args)
        self.assertEqual((cfg.review_package, cfg.review_key), (self.audit, "audit-key"))


if __name__ == "__main__":
    unittest.main()
