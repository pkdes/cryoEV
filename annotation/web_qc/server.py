#!/usr/bin/env python3
"""Browser QC tool for model detections -- PC and phone.

Reads images + CACHED prediction files (no model load), serves one page that
works with mouse, touch and stylus, and keeps one review JSON per image.

    python -m annotation.web_qc.server \
        --images-dir <tiles> --predictions-dir <model>/predictions \
        --review-dir <out>/qc_review [--sample 14 --seed 1]

Prints a URL + QR code. Phone on the same Wi-Fi: scan it. Away from the
network: install Tailscale on PC and phone; the Tailscale URL is printed too.
Every request needs the session token in that URL, so other devices on the
network cannot read or write.

Export + metrics (no server needed):

    python -m annotation.web_qc.export --images-dir ... --predictions-dir ... --review-dir ...
"""

from __future__ import annotations

import argparse
import io
import json
import random
import secrets
import socket
import subprocess
import time
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image

# Module level, not inside make_app: with postponed annotations FastAPI
# resolves `request: Request` from module globals, else it becomes a query param.
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response

HERE = Path(__file__).resolve().parent
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
MAX_EDGE = 1440


# ---------------------------------------------------------------- data layer

def load_classes(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    names = [c["name"] for c in cfg["classes"]]
    for role, name in cfg["role_to_class"].items():
        assert name in names, f"role_to_class: unknown class {name!r}"
    return cfg


def read_predictions(txt: Path, w: int, h: int):
    """(polygons in pixels, confidences). Same parsing as train_yolo's loaders."""
    from training.train_yolo import load_prediction_polygons
    return load_prediction_polygons(txt, w, h)


def seed_objects(txt: Path, w: int, h: int, cfg: dict, role_conf: float) -> list:
    """Model objects with an initial class from classify_roles().

    Roles are assigned among objects at/above role_conf -- the same population
    analysis.add_layer_roles uses -- so the tool shows the calls the report
    counted. Objects below it start as plain EV.
    """
    from analysis.add_layer_roles import roles_for

    polys, confs = read_predictions(txt, w, h)
    kept = [i for i, c in enumerate(confs) if c >= role_conf]
    masks = []
    for i in kept:
        m = np.zeros((h, w), np.uint8)
        cv2.fillPoly(m, [np.round(polys[i]).astype(np.int32)], 1)
        masks.append(m.astype(bool))
    role = {}
    if masks:
        r, _ = roles_for(masks)
        role = {kept[k]: v for k, v in r.items()}

    objs = []
    for i, (p, c) in enumerate(zip(polys, confs)):
        rl = role.get(i)
        cls = cfg["role_to_class"].get(rl, cfg["role_to_class"]["standalone"])
        objs.append({
            "id": f"m{i}",
            "poly": [[round(float(x) / w, 6), round(float(y) / h, 6)] for x, y in p],
            "conf": round(float(c), 4),
            "model_class": cls,
            "role": rl or "below-role-conf",
            "cls": cls,
            "status": "pending",
            "source": "model",
        })
    return objs


class Store:
    def __init__(self, images, predictions_dir, review_dir, cfg, role_conf):
        self.images = images
        self.by_name = {p.name: p for p in images}
        self.pred_dir = predictions_dir
        self.review_dir = review_dir
        self.cfg = cfg
        self.role_conf = role_conf
        review_dir.mkdir(parents=True, exist_ok=True)

    def review_path(self, name):
        return self.review_dir / (Path(name).stem + ".json")

    def size(self, name):
        with Image.open(self.by_name[name]) as im:
            return im.size

    def get(self, name):
        rp = self.review_path(name)
        if rp.exists():
            return json.loads(rp.read_text(encoding="utf-8"))
        w, h = self.size(name)
        return {
            "image": name, "width": w, "height": h, "version": 0, "done": False,
            "updated": None,
            "objects": seed_objects(self.pred_dir / (Path(name).stem + ".txt"),
                                    w, h, self.cfg, self.role_conf),
        }

    def put(self, name, doc):
        cur = self.get(name)
        if int(doc.get("version", -1)) != int(cur["version"]):
            return None, cur
        doc["version"] = int(cur["version"]) + 1
        doc["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        doc["image"], doc["width"], doc["height"] = name, cur["width"], cur["height"]
        tmp = self.review_path(name).with_suffix(".tmp")
        tmp.write_text(json.dumps(doc, indent=1), encoding="utf-8")
        tmp.replace(self.review_path(name))
        return doc, None

    def summary(self):
        out = []
        for p in self.images:
            rp = self.review_path(p.name)
            done, n, total = False, None, None
            if rp.exists():
                d = json.loads(rp.read_text(encoding="utf-8"))
                done = bool(d.get("done"))
                n = sum(o["status"] != "pending" for o in d["objects"])
                total = len(d["objects"])
            out.append({"name": p.name, "done": done, "reviewed": rp.exists(),
                        "n_decided": n, "n_objects": total})
        return out


# ---------------------------------------------------------------- web layer

def make_app(store: Store, token: str):
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @app.middleware("http")
    async def check_token(request: Request, call_next):
        if request.url.path == "/favicon.ico":
            return Response(status_code=204)
        given = request.query_params.get("t") or request.headers.get("x-token")
        if not given or not secrets.compare_digest(given, token):
            return JSONResponse({"detail": "missing or wrong token"}, status_code=401)
        return await call_next(request)

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (HERE / "static" / "index.html").read_text(encoding="utf-8")

    @app.get("/api/config")
    def config():
        return {"classes": store.cfg["classes"], "images": store.summary(),
                "role_conf": store.role_conf}

    def _name(name):
        if name not in store.by_name:
            raise HTTPException(404, "unknown image")
        return name

    @lru_cache(maxsize=64)
    def jpeg(name, mtime):
        with Image.open(store.by_name[name]) as im:
            im = im.convert("L") if im.mode not in ("L", "RGB") else im
            if max(im.size) > MAX_EDGE:
                im.thumbnail((MAX_EDGE, MAX_EDGE))
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=90)
            return buf.getvalue()

    @app.get("/api/image/{name}")
    def image(name: str):
        name = _name(name)
        data = jpeg(name, store.by_name[name].stat().st_mtime)
        return Response(data, media_type="image/jpeg",
                        headers={"Cache-Control": "private, max-age=3600"})

    @app.get("/api/ann/{name}")
    def get_ann(name: str):
        return store.get(_name(name))

    @app.put("/api/ann/{name}")
    async def put_ann(name: str, request: Request):
        doc = await request.json()
        saved, current = store.put(_name(name), doc)
        if saved is None:
            return JSONResponse({"detail": "stale", "current": current}, status_code=409)
        return saved

    return app


# ---------------------------------------------------------------- launch

def lan_ips():
    ips = set()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ips.add(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    ips.update(tailscale_hosts())
    return sorted(ips)


def tailscale_hosts():
    """This PC's Tailscale 100.x address and MagicDNS short name, if signed in.

    Falls back to the default Windows install path: a shell opened before the
    install doesn't have tailscale on PATH yet.
    """
    for exe in ("tailscale", r"C:\Program Files\Tailscale\tailscale.exe"):
        try:
            r = subprocess.run([exe, "status", "--json"], capture_output=True, text=True, timeout=3)
        except (OSError, subprocess.SubprocessError):
            continue
        try:
            me = json.loads(r.stdout)["Self"]
        except (ValueError, KeyError, TypeError):
            return []
        hosts = [ip for ip in me.get("TailscaleIPs") or [] if ip.startswith("100.")]
        if me.get("DNSName"):
            hosts.append(me["DNSName"].split(".")[0])
        return hosts
    return []


def collect_images(images_dir: Path, sample: int | None, seed: int, names_file: Path | None):
    imgs = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if names_file:
        wanted = [l.strip() for l in names_file.read_text().splitlines() if l.strip()]
        by = {p.name: p for p in imgs}
        missing = [w for w in wanted if w not in by]
        if missing:
            raise SystemExit(f"{len(missing)} listed image(s) not found, e.g. {missing[0]}")
        imgs = [by[w] for w in wanted]
    elif sample:
        imgs = sorted(random.Random(seed).sample(imgs, min(sample, len(imgs))))
    return imgs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images-dir", required=True, type=Path)
    ap.add_argument("--predictions-dir", required=True, type=Path)
    ap.add_argument("--review-dir", required=True, type=Path)
    ap.add_argument("--classes", type=Path, default=HERE / "classes.yaml")
    ap.add_argument("--role-conf", type=float, default=0.5,
                    help="confidence floor for multilayer role assignment (as in the analysis)")
    ap.add_argument("--images-list", type=Path, default=None,
                    help="text file of image names to review, in order")
    ap.add_argument("--sample", type=int, default=None, help="review a random subset of N images")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--token", default=None,
                    help="fixed token (default: kept in <review-dir>/token.txt)")
    ap.add_argument("--open-browser", action="store_true",
                    help="open the tool in this PC's default browser once the server is up")
    args = ap.parse_args()

    images = collect_images(args.images_dir, args.sample, args.seed, args.images_list)
    if not images:
        raise SystemExit(f"no images in {args.images_dir}")
    store = Store(images, args.predictions_dir, args.review_dir,
                  load_classes(args.classes), args.role_conf)
    # Remember the subset so export/metrics use the same images.
    (args.review_dir / "images_list.txt").write_text(
        "\n".join(p.name for p in images) + "\n", encoding="utf-8")

    # Same token every launch, so a bookmark on another PC keeps working.
    # Delete token.txt to rotate it.
    token_file = args.review_dir / "token.txt"
    if args.token:
        token = args.token
    elif token_file.exists() and token_file.read_text(encoding="utf-8").strip():
        token = token_file.read_text(encoding="utf-8").strip()
    else:
        token = secrets.token_urlsafe(9)
        token_file.write_text(token + "\n", encoding="utf-8")
    urls = [f"http://localhost:{args.port}/?t={token}"]
    urls += [f"http://{ip}:{args.port}/?t={token}" for ip in lan_ips()]
    (args.review_dir / "qc_urls.txt").write_text("\n".join(urls) + "\n", encoding="utf-8")
    print(f"\n{len(images)} image(s); reviews -> {args.review_dir}\n")
    for u in urls:
        print("  " + u)
    try:
        import qrcode
        qr = qrcode.QRCode(border=1)
        # Home Wi-Fi address works on a phone without Tailscale; else the last one.
        lan = [u for u in urls if u.split("//")[1].startswith(("192.168.", "10.", "172."))]
        qr.add_data((lan or urls)[-1])
        qr.make_image().save(args.review_dir / "phone_qr.png")  # before print: it can fail on a piped console
        qr.print_ascii(invert=True)
        print(f"  (QR also saved to {args.review_dir / 'phone_qr.png'})")
    except Exception as exc:  # QR is a convenience only
        print(f"  (no QR: {exc})")
    print("\nPhone off-network: install Tailscale on both; its 100.x URL appears above.")
    print("Keep this window open while reviewing; closing it stops the tool.\n")

    if args.open_browser:
        import threading
        import webbrowser
        threading.Timer(1.5, webbrowser.open, [urls[0]]).start()

    import uvicorn
    uvicorn.run(make_app(store, token), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
