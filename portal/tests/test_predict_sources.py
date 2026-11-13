"""
Predict's session list (portal/predict_sources.py, predict.sources(), Admin's Predict sessions panel): discovery of
model-ready tile folders in temp roots (compatible ones, incompatible ones with the reason, the -N suffix, the
default label, the roots setting, the cache, read only), the hand list CJ_PREDICT_SOURCES giving exactly what the
old sources() gave (control: the old function, copied verbatim), the merge, the overrides (persist, audit, fail
closed on a bad file, concurrent writers in threads and processes, with a control that shows the race without the
lock), hidden sources absent for basic users and still shown to the admin, escaping, and the panel in a browser
(Playwright, Edge channel: 390 and 1280 px, light and dark). Screenshots go to $CJ_SHOT_DIR when set (else a temp
folder, removed afterwards).

Temp folders only (cjtest.Env plus a temp processed-data root); never the real ones.

  python -m unittest discover -s portal/tests -v
"""
import contextlib
import csv
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from urllib.parse import urlencode

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cjtest as T  # noqa: E402
from jobs import predict_set as PS  # noqa: E402
from portal import audit as PA  # noqa: E402
from portal import auth as A  # noqa: E402
from portal import predict_sources as PSRC  # noqa: E402
from portal.modules import predict as P  # noqa: E402

try:
    from playwright.sync_api import sync_playwright
except ImportError:  # pragma: no cover
    sync_playwright = None

XSS = '<img src=x onerror=alert(1)>'
TILES = "3d. Tiles no scalebar - 14.7 A-px PNG"
FORM = "application/x-www-form-urlencoded"
TOKEN_RE = re.compile(r'name="t" value="([0-9a-f]{40})"')
ENV_KEYS = ("CJ_PREDICT_SOURCES", "CJ_PREDICT_SCAN_ROOTS", "CJ_PREDICT_ROOT", "CJ_REVIEW_ROOT", "CRYOEV_LAPTOP_ROOT",
            "CRYOEV_CRYOAI_ROOT", "CRYOEV_PROCESSING_ROOT", "CRYOEV_REGISTRY")
SHAPE = {"key", "label", "dir", "manifest", "session", "sample_column", "problem"}


def png(bits=8, size=16) -> bytes:
    from PIL import Image
    im = Image.new("L", (size, size), 120) if bits == 8 else Image.new("I;16", (size, size), 1000)
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


def tile_dir(root: Path, session: str, sub=TILES) -> Path:
    d = root / session / "3. Processed images" / sub
    d.mkdir(parents=True, exist_ok=True)
    return d


def manifest(d: Path, rows, head=("output_name", "grid", "sample_label", "nm_per_px")):
    with open(d / "MANIFEST.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(head)
        w.writerows(rows)


def fkey(d: Path) -> str:
    """The key of a discovered 3d folder: root-relative (<session folder>/<3d folder>)."""
    return PSRC.found_key(d.parent.parent.name, d.name)


def junction(link: Path, target: Path) -> bool:
    """mklink /J link target (Windows); False when it can't be made (then the test skips)."""
    if os.name != "nt":
        try:
            os.symlink(target, link, target_is_directory=True)
            return True
        except OSError:
            return False
    r = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], capture_output=True)
    return r.returncode == 0 and link.exists()


def make_root(root: Path) -> dict:
    """A processed-data root like the lab's, every case of discovery in it. -> {name: tile dir}."""
    out = {}
    d = out["a"] = tile_dir(root, "20261027")                            # compatible: 3 tiles, 2 samples
    for i in range(3):
        (d / f"g{i % 2 + 1}_t{i}.png").write_bytes(png())
    manifest(d, [[f"g{i % 2 + 1}_t{i}.png", i % 2 + 1, f"Sample {i % 2 + 1}", "1.4724"] for i in range(3)])
    d = out["b"] = tile_dir(root, "20260827-2")                          # compatible, the -2 suffix
    for i in range(2):
        (d / f"g5_t{i}.png").write_bytes(png())
    manifest(d, [[f"g5_t{i}.png", 5, "PLL", "1.47288"] for i in range(2)])
    d = out["noman"] = tile_dir(root, "20261012")                        # no MANIFEST.csv
    (d / "t.png").write_bytes(png())
    d = out["nocol"] = tile_dir(root, "20261013")                        # no nm_per_px column
    (d / "t.png").write_bytes(png())
    manifest(d, [["t.png", 1, "S"]], head=("output_name", "grid", "sample_label"))
    d = out["nonm"] = tile_dir(root, "20261014")                         # nm column, no readable value
    (d / "t.png").write_bytes(png())
    manifest(d, [["t.png", 1, "S", ""]])
    d = out["noimg"] = tile_dir(root, "20261015")                        # manifest, no images
    manifest(d, [["t.png", 1, "S", "1.47"]])
    d = out["deep"] = tile_dir(root, "20261016")                         # 16-bit PNG tiles
    (d / "t.png").write_bytes(png(16))
    manifest(d, [["t.png", 1, "S", "1.47"]])
    (root / "20260727" / "3. Processed images" / "3a. Individual tiles").mkdir(parents=True)   # no 3d folder
    (root / "20260804").mkdir()                                                                # nothing at all
    (root / "notes").mkdir()                                             # not a session folder: ignored
    (root / "20261399").mkdir()                                          # not a real date: ignored
    (root / "readme.txt").write_text("x", encoding="utf-8")
    return out


def snapshot(root: Path) -> dict:
    """Every file and folder under root: (size, mtime_ns, md5) / mtime_ns."""
    snap = {}
    for p in sorted(root.rglob("*")):
        st = p.stat()
        snap[str(p)] = (st.st_size, st.st_mtime_ns, hashlib.md5(p.read_bytes()).hexdigest()) if p.is_file() \
            else st.st_mtime_ns
    snap[str(root)] = root.stat().st_mtime_ns
    return snap


def old_sources(cfg) -> list:
    """predict.sources() before discovery (main at b8fdc7f), copied verbatim: the back-compat control."""
    import hashlib as hashlib_
    raw = P._setting(cfg, "CJ_PREDICT_SOURCES")
    if not raw:
        return []
    if isinstance(raw, (list, tuple)):
        items = list(raw)
    else:
        text = str(raw).strip()
        if not text.startswith("["):
            text = Path(text).read_text(encoding="utf-8-sig")
        items = json.loads(text)
    out = []
    for it in items if isinstance(items, list) else []:
        if not isinstance(it, dict) or not it.get("dir"):
            continue
        d = Path(str(it["dir"]))
        man = Path(str(it["manifest"])) if it.get("manifest") else (d / "MANIFEST.csv")
        session = str(it.get("session") or "")
        if not session:
            m = re.search(r"(?<!\d)(\d{8})(?!\d)", d.name) or re.search(r"(?<!\d)(\d{8})(?!\d)", str(d))
            session = m.group(1) if m else ""
        problem = None
        if not d.is_dir():
            problem = "folder not found"
        elif not PS.SESSION_RE.match(session):
            problem = 'no session date: add "session": "YYYYMMDD" to this source'
        out.append({"key": hashlib_.sha1(os.path.normcase(str(d)).encode("utf-8")).hexdigest()[:10],
                    "label": str(it.get("label") or d.name), "dir": d,
                    "manifest": man if man.is_file() else None, "session": session,
                    "sample_column": it.get("sample_column"), "problem": problem})
    return out


class EnvCase(unittest.TestCase):
    """Saves and restores the settings; a temp processed-data root and state dir per test."""

    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in ENV_KEYS}
        for k in ENV_KEYS:
            os.environ.pop(k, None)
        self.tmp = Path(tempfile.mkdtemp(prefix="cj_psrc_"))
        self.root = self.tmp / "processed"
        self.root.mkdir()
        self.dirs = make_root(self.root)
        self.state = self.tmp / "state"
        os.environ["CRYOEV_PROCESSING_ROOT"] = str(self.tmp / "no_processing")   # the fallback finds nothing
        os.environ["CJ_PREDICT_SCAN_ROOTS"] = str(self.root)
        PSRC.clear_cache()

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        PSRC.clear_cache()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def cfg(self, **kw):
        return types.SimpleNamespace(state_dir=self.state, **kw)


class Discovery(EnvCase):
    def test_compatible_found_and_incompatible_reported_with_reason(self):
        m = PSRC.merged(self.cfg())
        by = {fkey(d): n for n, d in self.dirs.items()}
        rows = {by[r["key"]]: r for r in m["sources"]}
        self.assertEqual(set(rows), set(self.dirs))
        self.assertEqual({n for n, r in rows.items() if r["compatible"]}, {"a", "b"})
        self.assertEqual(rows["noman"]["reason"], "no MANIFEST.csv")
        self.assertEqual(rows["nocol"]["reason"], "MANIFEST.csv has no nm_per_px column")
        self.assertEqual(rows["nonm"]["reason"], "MANIFEST.csv gives no readable nm/px for these images")
        self.assertEqual(rows["noimg"]["reason"], "no PNG or JPEG images")
        self.assertEqual(rows["deep"]["reason"], "tiles are 16-bit PNG, not 8-bit")
        self.assertEqual((rows["a"]["tiles"], rows["a"]["samples"], rows["a"]["nm_min"]), (3, 2, 1.4724))
        self.assertEqual((rows["b"]["tiles"], rows["b"]["samples"]), (2, 1))
        self.assertTrue(all(r["origin"] == "found" and not r["hidden"] for r in rows.values()))
        self.assertEqual(sorted(s["folder"] for s in m["skipped"]), ["20260727", "20260804"])
        self.assertEqual(m["roots"], [{"root": self.root, "problem": None}])
        # newest first; non-session folders ignored
        self.assertEqual([r["session"] for r in m["sources"]],
                         ["20261027", "20261016", "20261015", "20261014", "20261013", "20261012", "20260827-2"])

    def test_suffix_and_default_label(self):
        rows = {r["key"]: r for r in PSRC.merged(self.cfg())["sources"]}
        b, a = rows[fkey(self.dirs["b"])], rows[fkey(self.dirs["a"])]
        self.assertEqual(b["session"], "20260827-2")                        # the registry's session id
        self.assertEqual(b["default_label"], "Session 27 Aug 2026 -2")
        self.assertEqual(a["label"], "Session 27 Oct 2026")
        self.assertEqual(PSRC.session_of("20260827-2"), "20260827-2")
        self.assertEqual(PSRC.session_of("20261105 EV prep"), "20261105")
        self.assertEqual(PSRC.default_label("20260827-2"), "Session 27 Aug 2026 -2")
        self.assertIsNone(PSRC.session_of("20261399"))
        self.assertIsNone(PSRC.session_of("notes"))
        self.assertEqual(PSRC.default_label("20261105"), "Session 5 Nov 2026")
        it = next(s for s in P.sources(self.cfg()) if s["key"] == b["key"])
        self.assertEqual(it["session"], "20260827-2")
        self.assertTrue(PS.SESSION_RE.match(it["session"]))

    def test_two_tile_folders_in_one_session_get_distinct_labels(self):
        d2 = tile_dir(self.root, "20261027", "3d. Tiles no scalebar - 7 A-px PNG")
        (d2 / "x.png").write_bytes(png())
        manifest(d2, [["x.png", 1, "S", "0.7"]])
        labels = [r["label"] for r in PSRC.merged(self.cfg(), fresh=True)["sources"] if r["session"] == "20261027"]
        self.assertEqual(sorted(labels), ["Session 27 Oct 2026 (14.7 A-px PNG)", "Session 27 Oct 2026 (7 A-px PNG)"])

    def test_predict_shape_and_problems(self):
        items = P.sources(self.cfg())
        self.assertTrue(all(set(it) == SHAPE for it in items))
        by = {it["key"]: it for it in items}
        a = by[fkey(self.dirs["a"])]
        self.assertIsNone(a["problem"])
        self.assertEqual(a["manifest"], self.dirs["a"] / "MANIFEST.csv")
        self.assertEqual(a["dir"], self.dirs["a"])
        self.assertEqual(by[fkey(self.dirs["noman"])]["problem"], "no MANIFEST.csv")
        self.assertIsNone(by[fkey(self.dirs["noman"])]["manifest"])
        # predict.py reads a discovered source as it reads a listed one
        t = P.tiles_of(a)
        self.assertEqual(t["n"], 3)
        self.assertEqual(sorted(g["label"] for g in t["groups"]), ["Sample 1", "Sample 2"])

    def test_roots_setting(self):
        other = self.tmp / "other"
        d = tile_dir(other, "20261101")
        (d / "t.png").write_bytes(png())
        manifest(d, [["t.png", 1, "S", "1.47"]])
        os.environ["CJ_PREDICT_SCAN_ROOTS"] = os.pathsep.join([str(self.root), str(other), str(self.tmp / "gone")])
        m = PSRC.merged(self.cfg())
        self.assertIn("20261101", [r["session"] for r in m["sources"]])
        self.assertEqual([r["problem"] for r in m["roots"]], [None, None, "folder not found"])
        # a Config attribute wins over the environment
        self.assertEqual(PSRC.scan_roots(self.cfg(cj_predict_scan_roots=str(other))), [other])
        self.assertEqual(PSRC.scan_roots(self.cfg(cj_predict_scan_roots=[str(other)])), [other])
        os.environ["CJ_PREDICT_SCAN_ROOTS"] = "off"
        self.assertEqual(PSRC.scan_roots(self.cfg()), [])
        self.assertEqual(P.sources(self.cfg()), [])
        # unset: the processing root data_utils/config.py knows
        os.environ.pop("CJ_PREDICT_SCAN_ROOTS")
        os.environ["CRYOEV_PROCESSING_ROOT"] = str(other)
        self.assertEqual(PSRC.scan_roots(self.cfg()), [other])
        os.environ["CRYOEV_PROCESSING_ROOT"] = str(self.tmp / "no_processing")
        self.assertEqual(PSRC.scan_roots(self.cfg()), [])

    def test_cache_ttl_and_root_mtime(self):
        calls = []
        real = PSRC.scan

        def counting(roots):
            calls.append(1)
            return real(roots)
        PSRC.scan = counting
        try:
            PSRC.discover(self.cfg())
            PSRC.discover(self.cfg())
            self.assertEqual(len(calls), 1)                      # cached
            new = tile_dir(self.root, "20261102")                # a new session folder: the root's mtime moves
            (new / "t.png").write_bytes(png())
            manifest(new, [["t.png", 1, "S", "1.47"]])
            os.utime(self.root, ns=(time.time_ns(), time.time_ns() + 10 ** 9))
            self.assertIn("20261102", [s["session"] for s in PSRC.discover(self.cfg())["found"]])
            self.assertEqual(len(calls), 2)
            ttl = PSRC.CACHE_TTL_S
            PSRC.CACHE_TTL_S = 0
            try:
                PSRC.discover(self.cfg())
                self.assertEqual(len(calls), 3)                  # expired
            finally:
                PSRC.CACHE_TTL_S = ttl
            PSRC.discover(self.cfg(), fresh=True)
            self.assertEqual(len(calls), 4)
        finally:
            PSRC.scan = real

    def test_scan_root_is_never_written(self):
        before = snapshot(self.root)
        PSRC.merged(self.cfg(), fresh=True)
        P.sources(self.cfg())
        for it in P.sources(self.cfg()):
            if not it["problem"]:
                P.tiles_of(it)
        PSRC.set_override(self.state, fkey(self.dirs["a"]), hidden=True, label="x", actor="t")
        P.sources(self.cfg())
        self.assertEqual(snapshot(self.root), before)
        self.assertTrue((self.state / "predict_sources.json").is_file())


class Links(EnvCase):
    def test_junctions_out_of_the_root_are_not_followed(self):
        outside = self.tmp / "outside"
        d = tile_dir(outside, "20261103")
        (d / "t.png").write_bytes(png())
        manifest(d, [["t.png", 1, "S", "1.47"]])
        if not junction(self.root / "20261103", outside / "20261103"):
            self.skipTest("can't make a junction here")
        proc = self.root / "20261104" / "3. Processed images"
        proc.mkdir(parents=True)
        self.assertTrue(junction(proc / "3d. Tiles no scalebar - linked", d))
        inner = self.root / "20261105"                                     # a link that stays inside the root
        self.assertTrue(junction(inner, self.root / "20261027"))
        m = PSRC.merged(self.cfg(), fresh=True)
        sessions = [r["session"] for r in m["sources"]]
        self.assertNotIn("20261103", sessions)
        self.assertNotIn("20261104", sessions)
        reasons = {s["folder"]: s["reason"] for s in m["skipped"]}
        self.assertEqual(reasons["20261103"], "links outside the scanned folder")
        self.assertEqual(reasons["20261104"], "links outside the scanned folder")
        # inside the root: allowed, but the same real folder is listed once
        self.assertEqual(sum(1 for r in m["sources"] if r["tiles"] == 3 and r["samples"] == 2), 1)
        self.assertIn("20261027", sessions)                                 # the folder itself, not the link
        self.assertNotIn("20261105", sessions)
        self.assertFalse(any(str(r["dir"]).startswith(str(outside)) for r in m["sources"]))

    def test_keys_survive_root_spelling(self):
        k1 = {r["session"]: r["key"] for r in PSRC.merged(self.cfg(), fresh=True)["sources"]}
        self.assertEqual(k1["20261027"], PSRC.found_key("20261027", TILES))
        PSRC.set_override(self.state, k1["20261027"], hidden=True, label="Kept", actor="t")
        spellings = [str(self.root) + os.sep, self.root.as_posix()]
        if os.name == "nt":
            spellings.append(str(self.root).upper())
        link = self.tmp / "via_link"
        if junction(link, self.root):
            spellings.append(str(link))
        for sp in spellings:
            with self.subTest(root=sp):
                os.environ["CJ_PREDICT_SCAN_ROOTS"] = sp
                rows = {r["session"]: r for r in PSRC.merged(self.cfg(), fresh=True)["sources"]}
                self.assertEqual({k: r["key"] for k, r in rows.items()}, k1)
                self.assertEqual((rows["20261027"]["label"], rows["20261027"]["hidden"]), ("Kept", True))
                it = next(i for i in P.sources(self.cfg()) if i["key"] == k1["20261027"])
                self.assertEqual(it["problem"], PSRC.HIDDEN)

    def test_hand_list_dedups_by_real_path(self):
        spell = str(self.dirs["a"]).upper() if os.name == "nt" else str(self.dirs["a"])
        os.environ["CJ_PREDICT_SOURCES"] = json.dumps([{"label": "Hand A", "dir": spell, "session": "20261027"},
                                                       {"label": "Hand A again", "dir": str(self.dirs["a"])}])
        items = P.sources(self.cfg())
        a = [it for it in items if it["label"].startswith("Hand A")]
        self.assertEqual(len(a), 1)
        self.assertEqual(a[0]["key"], fkey(self.dirs["a"]))                 # a found folder keeps its found key
        self.assertEqual(len(items), len(self.dirs))
        # scan off: the hand list's own (absolute-path) key, as before
        os.environ["CJ_PREDICT_SCAN_ROOTS"] = "off"
        self.assertEqual(P.sources(self.cfg())[0]["key"], PSRC.source_key(Path(spell)))

    def test_orphaned_overrides_are_listed(self):
        PSRC.set_override(self.state, "0123456789", label="Old run", hidden=True, actor="t")
        PSRC.set_override(self.state, fkey(self.dirs["a"]), label="Here", actor="t")
        m = PSRC.merged(self.cfg())
        self.assertEqual([o["key"] for o in m["orphans"]], ["0123456789"])
        self.assertEqual((m["orphans"][0]["label"], m["orphans"][0]["hidden"]), ("Old run", True))


class BackCompat(EnvCase):
    """Control: the hand list alone gives exactly what the old sources() gave."""

    def hand(self):
        return [{"label": "EV prep A " + XSS, "dir": str(self.dirs["a"])},
                {"label": "Plain", "dir": str(self.dirs["noman"]), "session": "20261028"},
                {"dir": str(self.tmp / "nowhere")},
                {"label": "No date", "dir": str(self.tmp)},
                {"label": "Own manifest", "dir": str(self.dirs["b"]), "manifest": str(self.dirs["a"] / "MANIFEST.csv"),
                 "sample_column": "grid"},
                "not a dict", {"label": "no dir"}]

    def test_hand_list_alone_same_output_as_before(self):
        os.environ["CJ_PREDICT_SCAN_ROOTS"] = "off"
        os.environ["CJ_PREDICT_SOURCES"] = json.dumps(self.hand())
        self.assertEqual(P.sources(self.cfg()), old_sources(self.cfg()))
        self.assertEqual(P.sources(None), old_sources(None))                 # predict tests call it with None
        self.assertEqual(len(old_sources(None)), 5)
        f = self.tmp / "sources.json"
        f.write_text("﻿" + json.dumps(self.hand()), encoding="utf-8")
        os.environ["CJ_PREDICT_SOURCES"] = str(f)
        self.assertEqual(P.sources(self.cfg()), old_sources(self.cfg()))
        cfg = self.cfg(cj_predict_sources=self.hand())
        self.assertEqual(P.sources(cfg), old_sources(cfg))
        os.environ.pop("CJ_PREDICT_SOURCES")
        self.assertEqual(P.sources(self.cfg()), old_sources(self.cfg()))
        self.assertEqual(P.sources(self.cfg()), [])
        # scan setting unset and the processing root not found (as the existing predict tests run): same
        os.environ.pop("CJ_PREDICT_SCAN_ROOTS")
        os.environ["CJ_PREDICT_SOURCES"] = json.dumps(self.hand())
        self.assertEqual(P.sources(self.cfg()), old_sources(self.cfg()))

    def test_bad_hand_list_still_raises(self):
        os.environ["CJ_PREDICT_SOURCES"] = "[not json"
        with self.assertRaises(ValueError):
            P.sources(self.cfg())
        srcs, err = P.sources_safe(self.cfg())
        self.assertEqual(srcs, [])
        self.assertTrue(err.startswith("Session list unreadable: "), err)

    def test_source_text_never_names_a_hidden_source(self):
        k = fkey(self.dirs["a"])
        PSRC.set_override(self.state, k, label="Secret name", hidden=True, actor="t")
        srcs = P.sources(self.cfg())
        req = {"source": {"kind": "session", "session": "20261027", "dir": str(self.dirs["a"].resolve())}}
        self.assertEqual(P.source_text(req, srcs), "Session 20261027")
        self.assertEqual(P.source_text(req, srcs, admin=True), "Session 20261027 (Secret name)")
        PSRC.set_override(self.state, k, hidden=False, actor="t")
        self.assertEqual(P.source_text(req, P.sources(self.cfg())), "Session 20261027 (Secret name)")

    def test_merge_dedup_hand_list_wins(self):
        os.environ["CJ_PREDICT_SOURCES"] = json.dumps([{"label": "Hand B", "dir": str(self.dirs["b"]),
                                                        "session": "20260827"}])
        items = P.sources(self.cfg())
        keys = [it["key"] for it in items]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(items[0]["label"], "Hand B")                        # the hand list first, its fields
        self.assertEqual(keys.count(fkey(self.dirs["b"])), 1)
        self.assertEqual(len(items), len(self.dirs))
        row = next(r for r in PSRC.merged(self.cfg())["sources"] if r["key"] == keys[0])
        self.assertEqual((row["origin"], row["tiles"], row["session"]), ("both", 2, "20260827-2"))
        self.assertEqual(items[0]["session"], "20260827-2")                 # the folder id wins over the hand list
        # a different spelling of the same folder is the same key (normcase), so still one row
        if os.name == "nt":
            os.environ["CJ_PREDICT_SOURCES"] = json.dumps([{"label": "Hand B", "dir": str(self.dirs["b"]).upper()}])
            self.assertEqual(len(P.sources(self.cfg())), len(self.dirs))


class Overrides(EnvCase):
    def test_hide_show_label_persist_and_apply_to_both(self):
        ka, kb = fkey(self.dirs["a"]), fkey(self.dirs["b"])
        os.environ["CJ_PREDICT_SOURCES"] = json.dumps([{"label": "Hand B", "dir": str(self.dirs["b"])}])
        PSRC.set_override(self.state, ka, hidden=True, actor="tester")
        PSRC.set_override(self.state, kb, label="  Liposomes   run 2 ", actor="tester")
        doc = json.loads((self.state / "predict_sources.json").read_text(encoding="utf-8"))
        self.assertEqual(doc["schema"], PSRC.SCHEMA)
        self.assertTrue(doc["sources"][ka]["hidden"])
        self.assertEqual(doc["sources"][kb]["label"], "Liposomes run 2")
        self.assertEqual(doc["sources"][kb]["by"], "tester")
        by = {it["key"]: it for it in P.sources(self.cfg())}
        self.assertEqual(by[ka]["problem"], PSRC.HIDDEN)
        self.assertEqual(by[kb]["label"], "Liposomes run 2")                 # a listed source too
        old, new = PSRC.set_override(self.state, ka, hidden=False, actor="tester")
        self.assertTrue(old["hidden"])
        self.assertFalse(new["hidden"])
        self.assertIsNone({it["key"]: it for it in P.sources(self.cfg())}[ka]["problem"])
        PSRC.set_override(self.state, kb, label="", actor="tester")          # back to the default (the hand label)
        self.assertEqual({it["key"]: it for it in P.sources(self.cfg())}[kb]["label"], "Hand B")

    def test_bad_input_changes_nothing(self):
        k = fkey(self.dirs["a"])
        for bad in ("x" * (PSRC.LABEL_MAX + 1), "a\x00b", "line\nbreak", "abc\u202edcba", "x\u2066y\u2069",
                    "rtl\u200f"):
            with self.assertRaises(ValueError):
                PSRC.set_override(self.state, k, label=bad)
        for key in ("", "../../x", "ABCDEF1234", "abc", None):
            with self.assertRaises(ValueError):
                PSRC.set_override(self.state, key, hidden=True)
        self.assertFalse((self.state / "predict_sources.json").exists())

    def test_unreadable_file_fails_closed_and_is_never_overwritten(self):
        self.state.mkdir(parents=True)
        f = self.state / "predict_sources.json"
        for text in ("{broken", json.dumps({"sources": []}), "[]",
                     json.dumps({"schema": PSRC.SCHEMA, "sources": {fkey(self.dirs["a"]): True}}),
                     json.dumps({"schema": PSRC.SCHEMA, "sources": {"0123456789": {"hidden": True},
                                                                    fkey(self.dirs["b"]): "hidden"}})):
            f.write_text(text, encoding="utf-8")
            with self.assertRaises(ValueError):
                P.sources(self.cfg())
            srcs, err = P.sources_safe(self.cfg())
            self.assertEqual(srcs, [])                                       # never a hidden session by accident
            self.assertTrue(err)
            with self.assertRaises(ValueError):
                PSRC.set_override(self.state, fkey(self.dirs["a"]), hidden=True)
            self.assertEqual(f.read_text(encoding="utf-8"), text)
            self.assertTrue(PSRC.merged(self.cfg())["overrides_error"])
        self.assertEqual([p.name for p in self.state.iterdir() if p.name.endswith(".tmp")], [])

    def test_write_failure_leaves_no_temp_and_changes_nothing(self):
        k = fkey(self.dirs["a"])
        PSRC.set_override(self.state, k, label="Before")
        real = PSRC._os_replace

        def deny(src, dst):
            raise PermissionError("held")
        PSRC._os_replace = deny
        try:
            with self.assertRaises(OSError):
                PSRC.set_override(self.state, k, label="After")
        finally:
            PSRC._os_replace = real
        self.assertEqual(PSRC.load_overrides(self.state)[k]["label"], "Before")
        self.assertEqual([p.name for p in self.state.iterdir() if p.name.endswith(".tmp")], [])


WRITER = r"""
import sys
sys.path.insert(0, sys.argv[1])
from portal import predict_sources as PSRC
state, tag = sys.argv[2], sys.argv[3]
for i in range(10):
    PSRC.set_override(state, f"{tag}{i:08x}", hidden=bool(i % 2), label=f"proc {tag} {i}", actor="p" + tag)
"""


class Concurrency(EnvCase):
    def slow_reads(self, delay=0.05):
        real = PSRC.load_overrides

        def slow(state_dir):
            out = real(state_dir)
            time.sleep(delay)                                # widen the read-modify-write window
            return out
        return real, slow

    def threads(self, n=12, tag="c"):
        errs = []

        def one(i):
            try:
                PSRC.set_override(self.state, f"{tag}{i:09x}"[:10], label=f"t{i}", actor="t")
            except Exception as e:                           # noqa: BLE001
                errs.append(e)
        ts = [threading.Thread(target=one, args=(i,)) for i in range(n)]
        for t in ts:
            t.start()
        for t in ts:
            t.join(60)
        return errs

    def test_threads_and_processes_never_lose_or_corrupt(self):
        repo = str(T.REPO)
        procs = [subprocess.Popen([sys.executable, "-c", WRITER, repo, str(self.state), tag],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE) for tag in ("a1", "b2")]
        real, slow = self.slow_reads()
        PSRC.load_overrides = slow
        try:
            errs = self.threads()
        finally:
            PSRC.load_overrides = real
        for p in procs:
            out, err = p.communicate(timeout=120)
            self.assertEqual(p.returncode, 0, err.decode(errors="replace")[-800:])
        self.assertEqual(errs, [])
        table = PSRC.load_overrides(self.state)                 # valid JSON, every write kept
        self.assertEqual(len(table), 12 + 20)
        self.assertEqual(sum(1 for k in table if k.startswith("a1")), 10)
        self.assertEqual(table["b200000003"]["label"], "proc b2 3")
        self.assertEqual([p.name for p in self.state.iterdir() if p.name.endswith(".tmp")], [])

    def test_control_without_the_lock_the_race_is_real(self):
        real_lock = PSRC.overrides_lock

        @contextlib.contextmanager
        def no_lock(state_dir, wait=0):
            path = PSRC.overrides_path(state_dir)
            path.parent.mkdir(parents=True, exist_ok=True)
            yield path
        real, slow = self.slow_reads(0.2)
        PSRC.overrides_lock, PSRC.load_overrides = no_lock, slow
        try:
            self.threads(8, "d")
        finally:
            PSRC.overrides_lock, PSRC.load_overrides = real_lock, real
        try:
            n = len(PSRC.load_overrides(self.state))
        except ValueError:
            n = -1
        self.assertLess(n, 8, "without the lock, concurrent writers should lose updates (else this test proves "
                              "nothing)")


class PortalCase(unittest.TestCase):
    """A portal on a temp Env (admin tester, advanced adv, basic bas) and a temp processed-data root."""

    @classmethod
    def setUpClass(cls):
        cls._saved = {k: os.environ.get(k) for k in ENV_KEYS}
        cls.env = T.Env()
        tmp = cls.env.tmp
        cls.root = tmp / "processed"
        cls.root.mkdir()
        cls.dirs = make_root(cls.root)
        lap = tmp / "lap"
        (tmp / "cai").mkdir()
        for k in ENV_KEYS:
            os.environ.pop(k, None)
        os.environ.update(CJ_PREDICT_SCAN_ROOTS=str(cls.root), CJ_PREDICT_ROOT=str(lap / "predict"),
                          CJ_REVIEW_ROOT=str(lap / "review"), CRYOEV_LAPTOP_ROOT=str(lap),
                          CRYOEV_CRYOAI_ROOT=str(tmp / "cai"), CRYOEV_PROCESSING_ROOT=str(tmp / "no_processing"))
        lap.mkdir()
        PSRC.clear_cache()
        cls.keys = {"admin": cls.env.key, "advanced": A.add_key(cls.env.roster, "adv", "advanced"),
                    "basic": A.add_key(cls.env.roster, "bas", "basic")}
        cls.p = T.Portal(cls.env.config())
        cls.tok = {r: T.login(cls.p.port, k) for r, k in cls.keys.items()}
        cls.ka, cls.kb = fkey(cls.dirs["a"]), fkey(cls.dirs["b"])
        cls.knoman = fkey(cls.dirs["noman"])

    @classmethod
    def tearDownClass(cls):
        cls.p.stop()
        for k, v in cls._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        PSRC.clear_cache()
        cls.env.cleanup()

    def setUp(self):
        for f in (self.env.state / "predict_sources.json", PA.audit_path(self.env.state)):
            if f.exists():
                f.unlink()
        PSRC.clear_cache()

    def get(self, path, who="admin"):
        st, h, body = T.request(self.p.port, "GET", path, T.cookie(self.tok[who]))
        return st, h, body.decode("utf-8", "replace")

    def form_token(self, who="admin") -> str:
        st, _, body = self.get("/admin/", who)                              # the same token on every Admin page
        self.assertEqual(st, 200)
        return TOKEN_RE.search(body).group(1)

    def post(self, path, fields, who="admin", origin=True, token=True):
        f = dict(fields)
        if token is True:
            f["t"] = self.form_token(who)
        elif token:
            f["t"] = token
        h = {**T.cookie(self.tok[who]), "Content-Type": FORM}
        if origin:
            h["Origin"] = self.p.origin
        st, hd, body = T.request(self.p.port, "POST", path, h, urlencode(f).encode())
        return st, hd, body.decode("utf-8", "replace")

    def audit_lines(self) -> list:
        return T.module_lines(PA.audit_path(self.env.state))   # Admin's own lines (not the Gate's, round 9)

    def follow(self, h):
        loc = T.hget(h, "location")
        self.assertTrue(loc.startswith("/admin/predict?f="), loc)
        st, _, body = self.get(loc)
        self.assertEqual(st, 200)
        return body


class PortalPanel(PortalCase):
    def test_admin_only(self):
        for who in ("basic", "advanced"):
            for path in ("/admin/predict", "/admin/api/predict"):
                st, _, _ = self.get(path, who)
                self.assertEqual(st, 403, (who, path))
            st, _, _ = T.request(self.p.port, "POST", "/admin/predict/show",
                                 {**T.cookie(self.tok[who]), "Content-Type": FORM, "Origin": self.p.origin},
                                 urlencode({"key": self.ka, "hidden": "yes"}).encode())
            self.assertEqual(st, 403)
        self.assertFalse((self.env.state / "predict_sources.json").exists())

    def test_panel_lists_found_and_incompatible_with_reasons(self):
        st, _, body = self.get("/admin/predict")
        self.assertEqual(st, 200)
        self.assertIn("<h1>Predict sessions</h1>", body)
        self.assertIn('aria-current="page">Predict sessions</a>', body)
        for word in ("Session 27 Oct 2026", "Session 27 Aug 2026 -2", "20260827-2", "no MANIFEST.csv",
                     "MANIFEST.csv has no nm_per_px column", "tiles are 16-bit PNG, not 8-bit",
                     "Session folders without model-ready tiles (2)", "20260804, 20260727"):
            self.assertTrue(html_esc(word) in body, word)
        self.assertEqual(body.count('action="/admin/predict/show"'), 2)     # only the usable ones can be toggled
        st, _, raw = self.get("/admin/api/predict")
        doc = json.loads(raw)
        self.assertEqual(doc["schema"], "cj_admin_predict/1")
        self.assertEqual(sorted(s["session"] for s in doc["sources"] if s["compatible"]), ["20260827-2", "20261027"])

    def test_hide_show_audited_and_hidden_absent_for_basic(self):
        st, h, _ = self.post("/admin/predict/show", {"key": self.ka, "hidden": "yes"})
        self.assertEqual(st, 303)
        body = self.follow(h)
        self.assertIn("Session 20261027 (Session 27 Oct 2026) is now hidden from Predict.", body)
        line = self.audit_lines()[-1]
        self.assertEqual((line["actor"], line["action"], line["target"]), ("tester", "predict_source_hidden", self.ka))
        self.assertEqual(line["detail"], {"session": "20261027", "label": "Session 27 Oct 2026"})
        self.assertTrue(PSRC.load_overrides(self.env.state)[self.ka]["hidden"])
        for who in ("basic", "advanced"):
            st, _, page = self.get("/predict/new/", who)
            self.assertEqual(st, 200)
            self.assertNotIn("Session 27 Oct 2026", page)
            self.assertNotIn(self.ka, page)
            self.assertIn("Session 27 Aug 2026 -2", page)
            self.assertNotIn("no MANIFEST.csv", page)                       # incompatible: never offered
            self.assertEqual(self.get(f"/predict/new/session/{self.ka}/", who)[0], 404)
            self.assertEqual(self.get(f"/predict/new/session/{self.knoman}/", who)[0], 404)
        st, _, page = self.get("/predict/new/", "admin")
        self.assertIn("Session 27 Oct 2026", page)                            # the admin still sees it
        self.assertIn(html_esc(PSRC.HIDDEN), page)
        self.assertEqual(self.get(f"/predict/new/session/{self.kb}/", "basic")[0], 200)
        # the same again: nothing changes, nothing logged
        n = len(self.audit_lines())
        st, h, _ = self.post("/admin/predict/show", {"key": self.ka, "hidden": "yes"})
        self.assertIn("already hidden", self.follow(h))
        self.assertEqual(len(self.audit_lines()), n)
        st, h, _ = self.post("/admin/predict/show", {"key": self.ka, "hidden": "no"})
        self.assertIn("is now shown in Predict", self.follow(h))
        self.assertEqual(self.audit_lines()[-1]["action"], "predict_source_shown")
        st, _, page = self.get("/predict/new/", "basic")
        self.assertIn("Session 27 Oct 2026", page)
        self.assertEqual(self.get(f"/predict/new/session/{self.ka}/", "basic")[0], 200)

    def test_rename_persists_audited_and_resets(self):
        st, h, _ = self.post("/admin/predict/label", {"key": self.kb, "label": "Liposomes, run 2"})
        self.assertEqual(st, 303)
        self.assertIn("is now called “Liposomes, run 2”", self.follow(h))
        line = self.audit_lines()[-1]
        self.assertEqual((line["action"], line["target"]), ("predict_source_renamed", self.kb))
        self.assertEqual(line["detail"], {"session": "20260827-2", "from": "Session 27 Aug 2026 -2",
                                          "to": "Liposomes, run 2"})
        self.assertIn("Liposomes, run 2", self.get("/predict/new/", "basic")[2])
        PSRC.clear_cache()                                                   # persists (the file, not memory)
        self.assertEqual(PSRC.load_overrides(self.env.state)[self.kb]["label"], "Liposomes, run 2")
        st, h, _ = self.post("/admin/predict/label", {"key": self.kb, "label": ""})
        self.assertIn("back to its default name, “Session 27 Aug 2026 -2”", self.follow(h))
        self.assertIn("Session 27 Aug 2026 -2", self.get("/predict/new/", "basic")[2])
        self.assertEqual(self.audit_lines()[-1]["detail"]["to"], "Session 27 Aug 2026 -2")

    def test_refusals_change_nothing(self):
        cases = [({"key": "0123456789", "hidden": "yes"}, "show", 404, "not in the list any more"),
                 ({"key": self.ka, "hidden": "maybe"}, "show", 400, "could not be read"),
                 ({"key": self.ka, "label": "x" * 81}, "label", 400, "at most 80"),
                 ({"key": "../x", "label": "y"}, "label", 404, "not in the list any more")]
        for fields, kind, status, words in cases:
            st, _, body = self.post(f"/admin/predict/{kind}", fields)
            self.assertEqual(st, status, (fields, body[:300]))
            self.assertIn(words, body)
        st, _, _ = self.post("/admin/predict/show", {"key": self.ka, "hidden": "yes"}, origin=False)
        self.assertEqual(st, 403)                                            # the Gate's Origin check
        st, _, body = self.post("/admin/predict/show", {"key": self.ka, "hidden": "yes"}, token="0" * 40)
        self.assertEqual(st, 403)
        self.assertIn("expired", body)
        self.assertIn('aria-current="page">Predict sessions</a>', body)    # the error stays on this tab
        self.assertIn('href="/admin/predict">Back to Predict sessions</a>', body)
        self.assertFalse((self.env.state / "predict_sources.json").exists())
        self.assertEqual(self.audit_lines(), [])

    def test_unreadable_overrides_locks_the_panel(self):
        f = self.env.state / "predict_sources.json"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("{broken", encoding="utf-8")
        try:
            st, _, body = self.get("/admin/predict")
            self.assertEqual(st, 200)
            self.assertIn("can&#x27;t be read", body)
            self.assertNotIn('action="/admin/predict/show"', body)
            st, _, page = self.get("/predict/new/", "basic")
            self.assertNotIn("Session 27", page)                             # fail closed
            st, _, body = self.post("/admin/predict/show", {"key": self.kb, "hidden": "yes"})
            self.assertEqual(st, 500)
            self.assertEqual(f.read_text(encoding="utf-8"), "{broken")
        finally:
            f.unlink()

    def test_orphans_on_the_panel(self):
        PSRC.set_override(self.env.state, "0123456789", label="Gone " + XSS, hidden=True, actor="tester")
        st, _, body = self.get("/admin/predict")
        self.assertIn("Settings for sessions no longer found (1)", body)
        self.assertIn("0123456789", body)
        self.assertNotIn(XSS, body)
        self.assertIn(html_esc("Gone " + XSS), body)
        doc = json.loads(self.get("/admin/api/predict")[2])
        self.assertEqual([o["key"] for o in doc["orphans"]], ["0123456789"])

    def test_escaping(self):
        st, h, _ = self.post("/admin/predict/label", {"key": self.ka, "label": XSS + ' & "q"'})
        self.assertEqual(st, 303)
        pages = [self.follow(h), self.get("/admin/predict")[2], self.get("/admin/audit")[2],
                 self.get("/predict/new/", "basic")[2], self.get("/predict/new/", "admin")[2]]
        for page in pages:
            self.assertNotIn(XSS, page)
            self.assertNotIn("<img", page)
        self.assertIn(html_esc(XSS), pages[1])
        self.assertIn(html_esc(XSS), pages[3])
        doc = json.loads(self.get("/admin/api/predict")[2])
        self.assertIn(XSS + ' & "q"', [s["label"] for s in doc["sources"]])  # JSON keeps the raw text


def html_esc(s):
    import html
    return html.escape(s)


class RealRoot(unittest.TestCase):
    """Read only: discovery on the processed-data root data_utils/config.py finds on this computer (skipped when
    there is none). Nothing under it may change."""

    def test_real_root_read_only(self):
        from data_utils import config
        saved = os.environ.pop("CJ_PREDICT_SCAN_ROOTS", None)
        try:
            root = config.processing_root()
            if root is None:
                self.skipTest("no processed-data root on this computer")
            def mtimes(extra=()):
                return {str(p): p.stat().st_mtime_ns for p in [root, *root.iterdir(), *extra]}
            before = mtimes()
            res = PSRC.scan([root])
            dirs = [s["dir"] for s in res["found"]]
            before_dirs = mtimes(dirs)
            PSRC.scan([root])
            self.assertEqual(mtimes(), before)
            self.assertEqual(mtimes(dirs), before_dirs)
            for s in res["found"]:
                self.assertTrue(str(s["dir"]).startswith(str(root)))
                self.assertTrue(s["compatible"] or s["reason"])
            if os.environ.get("CJ_SHOW_REAL_ROOT"):
                for s in res["found"]:
                    print(f"\n  {s['folder']}: {'compatible' if s['compatible'] else 'NOT: ' + s['reason']}, "
                          f"{s['tiles']} tiles, {s['samples']} samples", end="")
                for s in res["skipped"]:
                    print(f"\n  {s['folder']}: {s['reason']}", end="")
        finally:
            if saved is not None:
                os.environ["CJ_PREDICT_SCAN_ROOTS"] = saved


def launch(p):
    for kw in ({"channel": "msedge"}, {"channel": "chrome"}, {}):
        try:
            return p.chromium.launch(headless=True, **kw)
        except Exception:
            continue
    return None


class Browser(PortalCase):
    @classmethod
    def setUpClass(cls):
        if sync_playwright is None:
            raise unittest.SkipTest("playwright not installed")
        super().setUpClass()
        cls.own_shots = not os.environ.get("CJ_SHOT_DIR")
        cls.shots = Path(os.environ.get("CJ_SHOT_DIR") or tempfile.mkdtemp(prefix="cj_shots_"))
        cls.shots.mkdir(parents=True, exist_ok=True)
        cls.env.write_live(T.live_doc())                                    # the bar's status dots read it
        cls.pw = sync_playwright().start()
        cls.browser = launch(cls.pw)
        if cls.browser is None:
            cls.tearDownClass()
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

    def session(self, width, scheme):
        ctx = self.browser.new_context(viewport={"width": width, "height": 900}, color_scheme=scheme)
        origin = self.p.origin
        rec = {"requests": [], "foreign": [], "errors": [], "console": []}
        ctx.route("**/*", lambda r: (r.continue_() if r.request.url.startswith(origin + "/")
                                     else (rec["foreign"].append(r.request.url), r.abort())))
        page = ctx.new_page()
        page.on("request", lambda r: rec["requests"].append(r.url))
        page.on("pageerror", lambda e: rec["errors"].append(str(e)))
        page.on("console", lambda m: m.type == "error" and rec["console"].append(m.text))
        page.goto(f"{origin}/login?k={self.env.key}")
        return ctx, page, rec

    def check(self, page, rec, scheme):
        sw, iw = page.evaluate("[document.documentElement.scrollWidth, window.innerWidth]")
        self.assertLessEqual(sw, iw, "horizontal scroll")
        bg = page.evaluate("getComputedStyle(document.body).backgroundColor")
        self.assertEqual(bg, "rgb(15, 21, 27)" if scheme == "dark" else "rgb(243, 245, 247)")
        self.assertEqual(rec["foreign"], [])
        self.assertEqual(rec["errors"], [])
        self.assertEqual(rec["console"], [])

    def test_panel_sizes_schemes_and_hide_flow(self):
        for scheme in ("light", "dark"):
            for width in (390, 1280):
                with self.subTest(width=width, scheme=scheme):
                    PSRC.set_override(self.env.state, self.kb, label="Liposomes " + XSS, actor="tester")
                    ctx, page, rec = self.session(width, scheme)
                    try:
                        page.goto(self.p.origin + "/admin/predict")
                        page.wait_for_selector(".cj-admin-ps")
                        self.assertEqual(page.locator(".cj-admin-ps tbody tr").count(), len(self.dirs))
                        self.assertEqual(page.locator("img").count(), 0)
                        self.check(page, rec, scheme)
                        page.screenshot(path=str(self.shots / f"admin_predict_{width}_{scheme}.png"),
                                        full_page=True)
                        page.click('button[aria-label="Hide session 20261027"]')
                        page.wait_for_url("**/admin/predict?f=*")
                        self.assertIn("is now hidden from Predict", page.text_content(".cj-admin-note"))
                        self.assertTrue(page.is_visible('button[aria-label="Show session 20261027"]'))
                        self.check(page, rec, scheme)
                        page.screenshot(path=str(self.shots / f"admin_predict_hidden_{width}_{scheme}.png"),
                                        full_page=True)
                        page.click('button[aria-label="Show session 20261027"]')
                        page.wait_for_url("**/admin/predict?f=*")
                        self.assertIn("is now shown in Predict", page.text_content(".cj-admin-note"))
                    finally:
                        ctx.close()
                        f = self.env.state / "predict_sources.json"
                        if f.exists():
                            f.unlink()


if __name__ == "__main__":
    unittest.main()
