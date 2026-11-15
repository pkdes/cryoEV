"""
A hand-made object set package (round 14, contract C4) for the review page's object card, and a tiny stand-in for
the review server's API, so the page can be tested before (and independently of) annotation/object_set.py and the
attribute support in annotation/review_server.py.

    build(folder, *, blind=True, outline_mode="target", n_items=6, sentinel=None) -> data dict
        index.html (the repo page), data.js (D.kind "objects", D.schema = ls_lamellarity_v1 as in contract C2,
        D.items with opaque ids, shuffled), images/ (two synthetic PNGs with concentric and nested rings)
    FakeServer(folder, caps=("i", "t", "a"), who=None).start() -> .url, .records, .puts, .close()
        GET  /api/ping   {"ok": true, "caps": [...]}
        GET  /api/edits  every stored record (the server copy)
        PUT  /api/edits?f=<file>   stores the record: each pending decision gets server_t, and an a: key's @<rid>
                         is rewritten to the server's identity (who=...), as contract C3 says the real server does
        GET  /api/claims, /api/people, /api/groups   empty answers
        anything else    the package's files

    build_tree(folder, *, blind=True, big=False, instructions=None) -> data dict          (round 15, contract K3)
        a whole-tree set: D.mode "tree", D.schema = TREE_SCHEMA (ls_lamellarity_tree_v1 as in K2: applies_to, help),
        items with nodes [{obj, n, parent_n}] (root n 1, the rest in seeded order; no role / unit / call), and with
        big=True one item of 30 nodes (the chunked-save test)
    copy_live(folder, src) -> data dict        a BYTE copy of a live set's data.js + schema.json (never written back)
    FakeServer(..., attr_cap=64, closed=False, fail=False): attr_cap = the real server's ATTR_PUT_MAX (400 over it),
        closed = 409 "set closed" on any a: PUT (#80g), fail = 503 on every PUT (an offline server)

Used by annotation/tests/test_object_card_browser.py and the round-14 / 15 screenshots. Nothing here touches a real
package or server.
"""
from __future__ import annotations

import json
import math
import random
import shutil
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
PAGE = HERE.parent / "review_tool" / "index.html"

SCHEMA = {
    "schema": "label_schema/1", "id": "ls_lamellarity_v1", "name": "lamellarity", "version": 1, "maker": "pujan",
    "description": "Lamellarity of nested EVs (fixture copy of contract C2).",
    "classes": [{"name": "EV", "color": "#4ade80", "key": "1"}],
    "object_attrs": [
        {"id": "type", "label": "Type", "kind": "choice", "required": True,
         "options": [{"v": "single", "label": "Single", "key": "1"}, {"v": "ML", "label": "ML", "key": "2"},
                     {"v": "MV_vesicular", "label": "MV multivesicular", "key": "3"},
                     {"v": "ML+MV", "label": "ML+MV", "key": "4"}, {"v": "unsure", "label": "Unsure", "key": "0"}]},
        {"id": "membranes", "label": "Concentric membranes", "kind": "ordinal", "required": True,
         "options": [{"v": "1", "key": "a"}, {"v": "2", "key": "s"}, {"v": "3", "key": "d"}, {"v": "4+", "key": "f"}]},
        {"id": "inner", "label": "Inner vesicles", "kind": "ordinal", "required": True,
         "options": [{"v": "0", "key": "z"}, {"v": "1", "key": "x"}, {"v": "2", "key": "c"}, {"v": "3+", "key": "v"}]},
        {"id": "outline_wrong", "label": "Outlines wrong", "kind": "bool", "key": "o"},
        {"id": "note", "label": "Note", "kind": "text", "max": 300, "key": "n"}],
    "image_attrs": [], "tags": [], "supersedes": None, "class_mapping": None}


HELP_TYPE = ("Count the outer EV's own level only: concentric membranes = the outer membrane plus its own layers; "
             "each inner vesicle counts as 1.")
_TYPE_OPTS = [{"v": "single", "label": "Single", "key": "1"}, {"v": "ML", "label": "ML", "key": "2"},
              {"v": "MV_vesicular", "label": "MV", "key": "3"}, {"v": "ML+MV", "label": "ML+MV", "key": "4"},
              {"v": "unsure", "label": "Unsure", "key": "0"}]
TREE_SCHEMA = {
    "schema": "label_schema/1", "id": "ls_lamellarity_tree_v1", "name": "lamellarity_tree", "version": 1,
    "maker": "pujan", "description": "Whole-tree lamellarity (fixture copy of contract K2).",
    "classes": [{"name": "EV", "color": "#4ade80", "key": "1"}],
    "object_attrs": [
        {"id": "type", "label": "Type", "kind": "choice", "required": True, "applies_to": "root", "help": HELP_TYPE,
         "options": _TYPE_OPTS},
        {"id": "relation", "label": "What is it?", "kind": "choice", "required": True, "applies_to": "child",
         "help": "A layer hugs its parent all round; an inner vesicle sits free inside it.",
         "options": [{"v": "layer", "label": "Layer of parent", "key": "l"},
                     {"v": "inner_vesicle", "label": "Inner vesicle", "key": "v"},
                     {"v": "not_membrane", "label": "Not a membrane", "key": "x"},
                     {"v": "duplicate", "label": "Duplicate outline", "key": "d"}]},
        {"id": "inner_type", "label": "If inner vesicle: its type", "kind": "choice", "applies_to": "child",
         "options": [{k: v for k, v in o.items() if k != "key"} for o in _TYPE_OPTS]},
        {"id": "note", "label": "Note", "kind": "text", "max": 300, "key": "n"}],
    "image_attrs": [], "tags": [], "supersedes": None, "class_mapping": {"EV": "EV"}}


def ring(cx, cy, r, w, h, n=28):
    """A circle as a flat normalized ring."""
    out = []
    for k in range(n):
        a = 2 * math.pi * k / n
        out += [round((cx + r * math.cos(a)) / w, 5), round((cy + r * math.sin(a)) / h, 5)]
    return out


def _png(path: Path, w, h, circles):
    from PIL import Image, ImageDraw, ImageFilter
    rnd = random.Random(7)
    im = Image.new("L", (w, h), 118)
    px = im.load()
    for y in range(h):
        for x in range(w):
            px[x, y] = max(0, min(255, 118 + rnd.randint(-22, 22)))
    d = ImageDraw.Draw(im)
    for cx, cy, r in circles:
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=48, width=3)
    im.filter(ImageFilter.GaussianBlur(0.8)).convert("RGB").save(path)


def build(folder, *, blind=True, outline_mode="target", n_items=6, sentinel=None, title="Lamellarity check 1",
          instructions=None) -> dict:
    folder = Path(folder)
    (folder / "images").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(PAGE, folder / "index.html")
    w, h = 640, 480
    # image A: a multilamellar EV (3 concentric rings) and a multivesicular one (2 inner vesicles); image B: singles
    a_circ = [(200, 220, 110), (200, 220, 86), (200, 220, 64), (470, 200, 90), (440, 180, 28), (500, 225, 24)]
    b_circ = [(160, 160, 70), (420, 300, 95), (420, 300, 72), (560, 110, 40)]
    _png(folder / "images" / "imgA.png", w, h, a_circ)
    _png(folder / "images" / "imgB.png", w, h, b_circ)
    img_a = {"file": "imgA.png", "split": "S1", "w": w, "h": h, "src": "images/imgA.png",
             "gt": [{"id": i + 1, "c": "EV", "p": [ring(cx, cy, r, w, h)]} for i, (cx, cy, r) in enumerate(a_circ)],
             "pr": [{"s": 0.41, "c": "EV", "p": [ring(330, 400, 30, w, h)]}]}
    img_b = {"file": "imgB.png", "split": "S2", "w": w, "h": h, "src": "images/imgB.png",
             "gt": [{"id": i + 1, "c": "EV", "p": [ring(cx, cy, r, w, h)]} for i, (cx, cy, r) in enumerate(b_circ)],
             "pr": []}

    def crop(cx, cy, r):
        m = r * 1.35
        return [round(max(0, cx - m) / w, 5), round(max(0, cy - m) / h, 5), round(min(w, cx + m) / w, 5),
                round(min(h, cy + m) / h, 5)]
    pool = [("imgA", "g1", a_circ[0]), ("imgA", "g4", a_circ[3]), ("imgB", "g1", b_circ[0]),
            ("imgB", "g2", b_circ[1]), ("imgB", "g4", b_circ[3]), ("imgA", "p0", (330, 400, 30))]
    rnd = random.Random(14)
    ids = set()
    items = []
    for stem, obj, (cx, cy, r) in pool[:n_items]:
        iid = f"{rnd.getrandbits(16):04x}"
        while iid in ids:
            iid = f"{rnd.getrandbits(16):04x}"
        ids.add(iid)
        items.append({"id": iid, "image": stem, "obj": obj, "crop": crop(cx, cy, r)})
    rnd.shuffle(items)
    data = {"id": folder.name, "title": title, "kind": "objects", "source": "prediction_set:ps_20261114_abcdef",
            "run": "fixture", "categories": ["EV"], "schema": SCHEMA, "items": items, "outline_mode": outline_mode,
            "blind": bool(blind), "images": [img_a, img_b]}   # a flag only (QC 4): no column names in data.js
    if sentinel:                                       # a test may plant a value that must never reach the page
        data["note_for_tests"] = sentinel
    if instructions is not None:                       # r15 hotfix: D.instructions (object_set.py instructions)
        data["instructions"] = instructions
    (folder / "data.js").write_text("window.REVIEW_DATA = " + json.dumps(data) + ";\n", encoding="utf-8")
    (folder / "review_package.json").write_text(json.dumps({
        "kind": "objects", "schema": SCHEMA["id"], "base": "ps_20261114_abcdef", "base_kind": "prediction_set",
        "n_items": len(items), "blind_cols": ["d_outer_nm", "gap_median_nm"] if blind else [], "outline_mode": outline_mode, "reviewers": ["rev1"],
        "owner": "admin1", "built_by": "portal", "title": title}), encoding="utf-8")
    return data


def build_tree(folder, *, blind=True, big=False, title="Tree check 1", instructions=None, seed=15,
               closed=False) -> dict:
    """A whole-tree object set (round 15, K3): see the module docstring."""
    folder = Path(folder)
    (folder / "images").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(PAGE, folder / "index.html")
    w, h = 640, 480
    rnd = random.Random(seed)
    # {stem: [(cx, cy, r, parent index or None)]}: concentric layers share a centre (their badges must not collide)
    trees = {
        "treeA": [(230, 240, 150, None), (230, 240, 128, 0), (190, 210, 34, 1), (275, 275, 28, 1), (300, 190, 18, 1),
                  (520, 110, 60, None), (520, 110, 44, 5)],
        "treeB": [(320, 240, 170, None), (250, 200, 48, 0), (250, 200, 36, 1), (390, 290, 40, 0)],
    }
    if big:                                            # 1 root + 29 inner vesicles: a 30-node card
        trees["treeC"] = [(320, 240, 225, None)] + [(160 + 46 * (k % 7), 120 + 52 * (k // 7), 15, 0) for k in range(29)]
    images, items, ids = [], [], set()
    for stem, circ in trees.items():
        _png(folder / "images" / f"{stem}.png", w, h, [(cx, cy, r) for cx, cy, r, _ in circ])
        gt = [{"id": i + 1, "c": "EV", "p": [ring(cx, cy, r, w, h, n=40)]} for i, (cx, cy, r, _) in enumerate(circ)]
        images.append({"file": f"{stem}.png", "split": "S1", "w": w, "h": h, "src": f"images/{stem}.png", "gt": gt,
                       "pr": []})
        for ri in [i for i, c in enumerate(circ) if c[3] is None]:
            members = [ri]
            for i, c in enumerate(circ):              # every descendant of this root (parents come first in the list)
                if c[3] is not None and c[3] in members:
                    members.append(i)
            others = members[1:]
            rnd.shuffle(others)                        # neutral numbering: root n 1, the rest seeded
            num = {ri: 1, **{i: k + 2 for k, i in enumerate(others)}}
            nodes = sorted(({"obj": f"g{i + 1}", "n": num[i],
                             "parent_n": num[circ[i][3]] if circ[i][3] is not None else None} for i in members),
                           key=lambda x: x["n"])
            cx, cy, r, _ = circ[ri]
            m = r * 1.15
            iid = f"{rnd.getrandbits(20):05x}"
            while iid in ids:
                iid = f"{rnd.getrandbits(20):05x}"
            ids.add(iid)
            items.append({"id": iid, "image": stem, "obj": f"g{ri + 1}", "nodes": nodes,
                          "crop": [round(max(0, cx - m) / w, 5), round(max(0, cy - m) / h, 5),
                                   round(min(w, cx + m) / w, 5), round(min(h, cy + m) / h, 5)]})
    rnd.shuffle(items)
    if big:                                            # the 30-node item first, so a test opens on it
        items.sort(key=lambda it: -len(it["nodes"]))
    data = {"id": folder.name, "title": title, "kind": "objects", "mode": "tree",
            "source": "prediction_set:ps_20261114_abcdef", "run": "fixture", "categories": ["EV"],
            "schema": TREE_SCHEMA, "items": items, "outline_mode": "target", "blind": bool(blind), "images": images}
    if instructions is not None:
        data["instructions"] = instructions
    if closed:
        data["closed"] = True                     # as object_set.close_set writes it (r15 integration, QC F)
    (folder / "data.js").write_text("window.REVIEW_DATA = " + json.dumps(data) + ";\n", encoding="utf-8")
    (folder / "review_package.json").write_text(json.dumps({
        "kind": "objects", "mode": "tree", "schema": TREE_SCHEMA["id"], "base": "ps_20261114_abcdef",
        "base_kind": "prediction_set", "n_items": len(items), "blind_cols": [], "reviewers": ["rev1"],
        "owner": "admin1", "built_by": "portal", "title": title}), encoding="utf-8")
    return data


def copy_live(folder, src) -> dict:
    """A byte copy of a live set's data.js + schema.json (and the repo page) into folder; the source is only read."""
    folder, src = Path(folder), Path(src)
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(PAGE, folder / "index.html")
    for name in ("data.js", "schema.json"):
        (folder / name).write_bytes((src / name).read_bytes())
    t = (folder / "data.js").read_text(encoding="utf-8")
    return json.loads(t.split("=", 1)[1].strip().rstrip(";"))


class FakeServer:
    """The review server's API, just enough for the object card (see the module docstring)."""

    def __init__(self, folder, caps=("i", "t", "a"), who=None, attr_cap=64, closed=False, fail=False, lock409=0):
        self.folder, self.caps, self.who = Path(folder), list(caps), who
        self.attr_cap, self.closed, self.fail = attr_cap, closed, fail
        self.lock409 = lock409                    # the next N PUTs: multi_review_server's "open in another server" 409
        self.records, self.puts, self.t = {}, [], 1_700_000_000_000
        self._lock = threading.Lock()
        outer = self

        class H(SimpleHTTPRequestHandler):
            def __init__(self, *a, **kw):
                super().__init__(*a, directory=str(outer.folder), **kw)

            def log_message(self, *a):
                pass

            def _json(self, obj, status=200):
                body = json.dumps(obj).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                u = urlparse(self.path)
                if u.path == "/api/ping":
                    return self._json({"ok": True, "caps": outer.caps})
                if u.path == "/api/edits":
                    with outer._lock:
                        return self._json(json.loads(json.dumps(outer.records)))
                if u.path == "/api/claims":
                    return self._json({"claims": {}})
                if u.path in ("/api/people", "/api/groups", "/api/meta"):
                    return self._json({}, 404)
                return super().do_GET()

            def do_PUT(self):
                u = urlparse(self.path)
                if u.path != "/api/edits":
                    return self._json({"error": "no"}, 404)
                f = (parse_qs(u.query).get("f") or [""])[0]
                rec = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                if outer.fail:
                    return self._json({"error": "down"}, 503)
                if outer.lock409 > 0:
                    outer.lock409 -= 1
                    return self._json({"error": "This package is open in another review server: only one may save "
                                                "to it at a time."}, 409)
                pend = [k for k, d in (rec.get("dec") or {}).items() if (k.startswith("a:") or k.startswith("i:a:"))
                        and isinstance(d, dict) and d.get("pending") is True]
                if pend and outer.closed:
                    return self._json({"error": "set closed"}, 409)
                if outer.attr_cap is not None and len(pend) > outer.attr_cap:
                    with outer._lock:
                        outer.puts.append((f, json.loads(json.dumps(rec))))
                    return self._json({"error": f"at most {outer.attr_cap} attribute decisions per save"}, 400)
                with outer._lock:
                    outer.puts.append((f, json.loads(json.dumps(rec))))
                    have = outer.records.setdefault(f, {"dec": {}})
                    for k, d in (rec.get("dec") or {}).items():
                        if outer.who and k.startswith("a:") and "@" in k:
                            k = k.rsplit("@", 1)[0] + "@" + outer.who
                            d = {**d, "who": outer.who}           # the real server stamps who too (stamp_identity)
                        old = have["dec"].get(k)
                        if d.get("pending") or not old:
                            outer.t += 1
                            have["dec"][k] = {kk: vv for kk, vv in d.items() if kk != "pending"} | {"server_t": outer.t}
                    for kk in ("reviewed", "reviewedT", "modified", "t", "gt", "gtT"):
                        if kk in rec:
                            have[kk] = rec[kk]
                    out = json.loads(json.dumps(have))
                return self._json({"ok": True, "edit": out})

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.httpd.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}/"

    def start(self):
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        return self

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()
