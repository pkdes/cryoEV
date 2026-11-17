"""
Guest access (round 17): anyone with a verified email at a configured domain may sign in through the Cloudflare door as
a limited GUEST who can open only review sets explicitly opened to guests. Off unless configured.

    python -m portal.guests open  --package <review root>/<pkg id>     # review_package.json "open_to": "guests"
    python -m portal.guests close --package <review root>/<pkg id>     # removes it (guests can no longer open it)
    python -m portal.guests list  --root <review root>                 # the packages open to guests

- Config: `guest_email_domains` (a list of bare domains) in the PRIVATE access config
  (~/.cryoev/cloudflare_access.json, portal/access.py load_config). Missing or [] = no guests. Never in the repo.
- Who: access mode only, verified Cf-Access-Jwt-Assertion JWT only (portal/app.py Gate.access_user). An email that
  matches no active roster person and whose domain EXACTLY equals a configured one (after auth.access_email: ASCII,
  lower-case; sub-domains and look-alikes are refused) gets a roster entry under roster_lock, once:
  {"id": "guest-<first 8 hex of sha256(email)>", "role": "guest", "email", "added", "auto": true}. Later visits match
  it by email as any person. Any existing entry with that id or that email (revoked included) blocks creation: a
  revoked guest stays out (revoke with `python -m portal.auth revoke --id guest-...` or Admin; never delete the entry,
  or the next visit makes it again).
- What: role "guest" is not in auth.ROLES, so no module lists it (no nav), no agent token, no key link. The Gate lets a
  guest reach only GUEST_PATHS (home = the open sets, the shell's own files) and /review/<pkg id>/... (pred_mount),
  where reviews.may_open lets a guest open only a package with open_to == "guests". Opening one adds the guest's id to
  the package's `reviewers` (reviews.join_guest, under the package meta lock), so the review server's reviewer check
  for object-set answers accepts them (it re-reads reviewers per request).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import secrets
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from portal import auth as A  # noqa: E402

CONFIG_KEY = "guest_email_domains"
OPEN_KEY, OPEN_VALUE = "open_to", "guests"
ID_PREFIX = "guest-"
GUEST_ID_RE = re.compile(r"guest-[0-9a-f]{8}\Z")
MAX_DOMAINS = 20
DOMAIN_RE = re.compile(r"(?=.{1,253}\Z)[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?(\.[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?)+")


def is_guest(user) -> bool:
    return isinstance(user, dict) and user.get("role") == A.GUEST_ROLE


def parse_domains(value) -> tuple:
    """The configured guest domains as a tuple of bare lower-case host names; () for None. Raises ValueError for
    anything else than a list of bare domains (no scheme, '@', wildcard, port or path)."""
    if value is None:
        return ()
    if not isinstance(value, list) or len(value) > MAX_DOMAINS:
        raise ValueError(f"`{CONFIG_KEY}` must be a list of at most {MAX_DOMAINS} domains")
    out = []
    for d in value:
        if not isinstance(d, str) or not d.isascii():
            raise ValueError(f"`{CONFIG_KEY}`: every entry must be a plain ASCII domain")
        d = d.strip().lower()
        if not DOMAIN_RE.fullmatch(d):
            raise ValueError(f"`{CONFIG_KEY}`: {d[:80]!r} is not a bare domain (e.g. example.edu)")
        out.append(d)
    return tuple(dict.fromkeys(out))


def domain_ok(email, domains) -> bool:
    """True when `email` (auth.access_email: ASCII, lower-cased, one plain address) is at EXACTLY one of `domains`."""
    e = A.access_email(email)
    if not e or not domains or e.count("@") != 1:
        return False
    return e.rsplit("@", 1)[1] in set(domains)


def guest_id(email: str) -> str:
    return ID_PREFIX + hashlib.sha256(A.access_email(email).encode("utf-8")).hexdigest()[:8]


def ensure_guest(roster: Path, email: str, domains) -> dict | None:
    """The active guest entry for `email`, made once under roster_lock when the domain matches; None when it may not
    sign in: wrong domain, an existing entry with this id or email that is revoked or not this guest (no
    re-creation), or the roster can't be read. Raises OSError when the roster lock can't be had."""
    if not domain_ok(email, domains):
        return None
    e = A.access_email(email)
    gid = guest_id(e)
    with A.roster_lock(roster):
        try:
            doc = json.loads(Path(roster).read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None                                 # no roster at all: nobody, guests included
        except (OSError, ValueError):
            return None
        if not isinstance(doc, dict) or not isinstance(doc.get("people"), list):
            return None
        hits = [p for p in doc["people"] if isinstance(p, dict)
                and (p.get("id") == gid or str(p.get("email") or "").strip().lower() == e)]
        if hits:
            p = hits[-1]
            ok = (len(hits) == 1 and p.get("id") == gid and p.get("role") == A.GUEST_ROLE and not p.get("revoked")
                  and str(p.get("email") or "").strip().lower() == e)
            return p if ok else None
        entry = {"id": gid, "role": A.GUEST_ROLE, "email": e, "added": datetime.now().isoformat(timespec="seconds"),
                 "auto": True}
        doc["people"].append(entry)
        A.write_json_atomic(roster, doc)
        return entry


def is_open(meta) -> bool:
    return isinstance(meta, dict) and meta.get(OPEN_KEY) == OPEN_VALUE


def set_open(pkg: Path, open_: bool, lock_wait: float = 2.0) -> bool:
    """Set (or remove) open_to = "guests" in <pkg>/review_package.json under the package meta lock
    (<root>/.meta_locks/<pkg>.lock, the Reviews hub's), re-read, temp file + os.replace. -> True when it changed.
    Raises ValueError (unreadable) or OSError (busy)."""
    from data_utils.oslock import acquire_lock
    from portal.modules import reviews as REV
    pkg = Path(pkg).resolve()
    locks = pkg.parent / REV.LOCK_DIR
    locks.mkdir(exist_ok=True)
    lk = acquire_lock(locks / f"{pkg.name}.lock", wait=lock_wait)
    if lk is None:
        raise OSError("the package settings are being changed; try again")
    try:
        f = pkg / "review_package.json"
        meta = json.loads(f.read_text(encoding="utf-8"))
        if not isinstance(meta, dict):
            raise ValueError(f"{f} is not a JSON object")
        if is_open(meta) == bool(open_):
            return False
        if open_:
            meta[OPEN_KEY] = OPEN_VALUE
        else:
            meta.pop(OPEN_KEY, None)
        tmp = pkg / f".review_package.json.{secrets.token_hex(4)}.tmp"
        try:
            tmp.write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
            REV._replace(tmp, f)
        except OSError:
            tmp.unlink(missing_ok=True)
            raise
        return True
    finally:
        lk.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("open", "close"):
        s = sub.add_parser(name, help=f"{name} a review package to guests")
        s.add_argument("--package", type=Path, required=True, help="the package folder (<review root>/<pkg id>)")
    ls = sub.add_parser("list", help="packages open to guests")
    ls.add_argument("--root", type=Path, required=True, help="the review root")
    a = ap.parse_args(argv)
    if a.cmd == "list":
        for d in sorted(p for p in Path(a.root).iterdir() if p.is_dir()):
            try:
                meta = json.loads((d / "review_package.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if is_open(meta):
                revs = [r for r in meta.get("reviewers") or [] if isinstance(r, str)]
                print(f"{d.name}\t{sum(1 for r in revs if GUEST_ID_RE.match(r))} guest(s) joined")
        return 0
    try:
        changed = set_open(a.package, a.cmd == "open")
    except (OSError, ValueError) as e:
        print(f"[--] {e}")
        return 1
    print(f"[OK] {a.package.name}: {'open to guests' if a.cmd == 'open' else 'closed to guests'}"
          + ("" if changed else " (already)"))
    if a.cmd == "open":
        print("[NOTE] once a guest opens it, only ids in its `reviewers` list can save answers (an empty list "
              "otherwise means anyone on the review roster): add every lab reviewer first.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
