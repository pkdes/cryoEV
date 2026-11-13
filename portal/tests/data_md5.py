"""
Byte-identity check for the Data module (round 11, refactor #2: data.py -> the portal/modules/data/ package).

Renders every Data route (HTML pages, JSON API, thumbnails, the Lab graph script, not-found paths and the Session
details POST) through the real app and Gate, as admin, advanced and basic, against one fixture world built from the
data tests' own fixture builders (test_data_module: samples + fixture rows, images on disk, lineage rows and lists;
test_data_graph: the Lab graph rows and the user's request.json; test_data_lineage_graph: the chain rows), and writes
one "<status> <md5>" per (role, method, path), the md5 of status + content-type + cache-control + location + body.

The world is rebuilt at a FIXED path (default <temp>/cj_data_md5_world) with every file's mtime pinned, because admin
pages show folder paths and thumbnail URLs carry a version made from path + mtime. Never the real registry or roots.

  python portal/tests/data_md5.py snap OUT.json [--world DIR] [--control]
  python portal/tests/data_md5.py diff A.json B.json          # exit 0 only when the keys and every md5 are equal

--control flips one byte of one response body (admin, GET /data/) before hashing: `diff` against a normal snapshot
must then report exactly that one route as a mismatch (the known answer for the comparison itself).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from urllib.parse import quote

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

PINNED_MTIME = 1_762_000_000          # 2025-11-01, any fixed value
ROLES = ("admin", "advanced", "basic")
CONTROL_KEY = "admin GET /data/"


def build_world(world: Path):
    """The fixture world at `world` (wiped first); -> (cfg, keys by role)."""
    import cjtest as T
    import test_data_graph as TG
    import test_data_lineage_graph as TLG
    import test_data_module as TM
    from portal import app as APP
    from portal import auth as A

    if world.exists():
        shutil.rmtree(world)
    lap, proc, outside, reg = (world / "CryoAI-laptop", world / "Cryo-image-processing", world / "outside",
                               world / "registry")
    for d in (lap, proc, outside, reg / "live"):
        d.mkdir(parents=True)
    os.environ["CRYOEV_LAPTOP_ROOT"], os.environ["CRYOEV_PROCESSING_ROOT"] = str(lap), str(proc)
    os.environ.pop("CJ_UPLOAD_SESSIONS", None)
    TM.build_data(lap, proc, outside)
    TM.write_registry(reg, outside)
    TM.add_lineage_records(reg)
    extra = TG.fixture_rows()
    for ent, recs in TLG.chain_rows().items():
        extra.setdefault(ent, []).extend(recs)
    for ent, recs in extra.items():
        with open(reg / f"{ent}.jsonl", "a", encoding="utf-8") as fh:
            fh.write("".join(json.dumps(r) + "\n" for r in recs))
    TG.write_disk(lap)
    for dirpath, _dirs, files in os.walk(world):
        for f in files:
            os.utime(Path(dirpath) / f, (PINNED_MTIME, PINNED_MTIME))
    roster = world / "roster.json"
    keys = {"admin": A.add_key(roster, T.USER, "admin"), "advanced": A.add_key(roster, "adv", "advanced"),
            "basic": A.add_key(roster, "bas", "basic")}
    cfg = APP.Config(registry=reg, roster=roster, state_dir=world / "state", review_upstream="http://127.0.0.1:9",
                     review_key=T.UPKEY, tokens=world / "agent_tokens.json")
    TG.clear_caches()
    return cfg, keys


def paths_to_visit(get, records) -> list:
    """Every Data GET path for the fixture world, enumerated from the admin's own JSON (so it covers every row)."""
    from portal.modules import data as D
    out = ["/data", "/data/", "/data/data.js", "/data/lineage/graph/", "/data/lineage/graph",
           "/data/api/overview", "/data/api/lineage/graph", "/data/list/nope", "/data/api/list/nope",
           "/data/session/nope", "/data/api/nope/x", "/data/lineage/session/nope/", "/data/api/lineage/model/nope",
           "/data/thumb/session/20261104/999.jpg", "/data/thumb/nope/x/0.jpg", "/data/session/20261104?page=abc"]
    for q in ("", "2602", "EV", "ps_", "sample", "zz_nothing", "<b>"):
        out += [f"/data/search?q={quote(q)}", f"/data/api/search?q={quote(q)}"]
    for kind in D.LIST_KINDS:
        out += [f"/data/list/{kind}", f"/data/api/list/{kind}"]
    focus = set()
    for kind in D.LIST_KINDS:
        rows = json.loads(get(f"/data/api/list/{kind}"))["rows"]
        dkind = next(k for k, v in D.DETAIL_LIST.items() if v == kind)
        for row in rows:
            eid = quote(str(row["id"]), safe="")
            base = [f"/data/{dkind}/{eid}", f"/data/api/{dkind}/{eid}", f"/data/lineage/{dkind}/{eid}/",
                    f"/data/lineage/{dkind}/{eid}", f"/data/api/lineage/{dkind}/{eid}"]
            for page in ("1", "2"):
                base += [f"/data/{dkind}/{eid}?page={page}", f"/data/api/{dkind}/{eid}?page={page}"]
            out += base
            d = json.loads(get(f"/data/api/{dkind}/{eid}?page=1"))
            out += [t["url"] for t in d.get("thumbs") or [] if isinstance(t, dict) and t.get("url")]
            for n in range(3):
                out.append(f"/data/thumb/{dkind}/{eid}/{n}.jpg")
            key = D.graph_focus_key(records, dkind, str(row["id"]))
            if key:
                focus.add(key)
    for key in sorted(focus) + ["session|20261104", "grid|2602", "model|nope", "session|?"]:
        out += [f"/data/api/lineage/graph?focus={quote(key)}",
                f"/data/api/lineage/graph?focus={quote(key)}&scope=item",
                f"/data/lineage/graph/?focus={quote(key)}"]
    seen, uniq = set(), []
    for p in out:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def snap(world: Path, control: bool) -> dict:
    import cjtest as T
    from portal.modules import data as D

    cfg, keys = build_world(world)
    portal = T.Portal(cfg)
    try:
        tok = {r: T.login(portal.port, k) for r, k in keys.items()}
        records = D.with_disk_packages(D.read_registry(cfg.registry)[0])

        def fetch(method, path, role, body=None, headers=None):
            h = dict(T.cookie(tok[role]))
            h.update(headers or {})
            return T.request(portal.port, method, path, h, body)

        def get_admin(path):
            st, _h, b = fetch("GET", path, "admin")
            assert st == 200, (path, st)
            return b

        paths = paths_to_visit(get_admin, records)
        out = {}
        for role in ROLES:
            for p in paths:
                st, h, b = fetch("GET", p, role)
                out[f"{role} GET {p}"] = digest(st, h, b, control and f"{role} GET {p}" == CONTROL_KEY)
        for role in ROLES:                      # last: the one POST route (switched off here, so nothing is written)
            p = "/data/session/20261104/details"
            st, h, b = fetch("POST", p, role, b"grid_1=x", {"Content-Type": "application/x-www-form-urlencoded",
                                                            "Origin": portal.origin})
            out[f"{role} POST {p}"] = digest(st, h, b, False)
        return out
    finally:
        portal.stop()


def digest(status, headers, body: bytes, flip: bool) -> str:
    import cjtest as T
    if flip:
        body = body[:-1] + bytes([body[-1] ^ 1])
    head = "|".join([str(status)] + [T.hget(headers, k) or "" for k in ("content-type", "cache-control", "location",
                                                                          "x-content-type-options")])
    return f"{status} " + hashlib.md5(head.encode() + b"\n" + body).hexdigest()


def diff(a: dict, b: dict) -> list:
    out = [f"only in A: {k}" for k in sorted(set(a) - set(b))] + [f"only in B: {k}" for k in sorted(set(b) - set(a))]
    out += [f"md5 differs: {k}" for k in sorted(set(a) & set(b)) if a[k] != b[k]]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("snap")
    s.add_argument("out", type=Path)
    s.add_argument("--world", type=Path, default=Path(tempfile.gettempdir()) / "cj_data_md5_world")
    s.add_argument("--control", action="store_true")
    d = sub.add_parser("diff")
    d.add_argument("a", type=Path)
    d.add_argument("b", type=Path)
    a = ap.parse_args(argv)
    if a.cmd == "snap":
        got = snap(a.world, a.control)
        a.out.write_text(json.dumps(got, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        by_role = {r: sum(1 for k in got if k.startswith(r + " ")) for r in ROLES}
        st = {r: {} for r in ROLES}
        for k, v in got.items():
            c = st[k.split(" ", 1)[0]]
            c[v.split()[0]] = c.get(v.split()[0], 0) + 1
        print(f"{len(got)} responses ({by_role}), {len(set(got.values()))} distinct md5s -> {a.out}")
        print("status codes by role:", {r: dict(sorted(c.items())) for r, c in st.items()})
        return 0
    da, db = (json.loads(p.read_text(encoding="utf-8")) for p in (a.a, a.b))
    probs = diff(da, db)
    for p in probs:
        print(p)
    print(f"{len(da)} vs {len(db)} responses: {'ALL EQUAL' if not probs else f'{len(probs)} differences'}")
    return 0 if not probs else 1


if __name__ == "__main__":
    sys.exit(main())
