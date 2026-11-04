"""The worker's web server. It has no pages, no photos, no accounts and no history: it only answers model jobs
from computers that were paired with a one-time code, and it forgets every image when the job is over."""
from __future__ import annotations

import hashlib
import io
import logging
import math
import os
import socket
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel, Field

from .. import config, netguard
from ..vision import registry
from ..vision.engine import Engine
from ..vision.registry import ModelError
from . import PROTOCOL
from .store import PairError, WorkerStore

MAX_BODY = 200 * 1024 * 1024         # one image upload
MAX_BLOBS = 6                        # images kept in memory at once
MAX_BLOB_BYTES = 400 * 1024 * 1024
BLOB_SECONDS = 15 * 60
QUEUE_WAIT = 60                      # how long a job waits for the graphics card before it is told "busy"
BAD_AUTH_PER_MINUTE = 30
log = logging.getLogger("veil.worker")


class Blobs:
    """Uploaded images, in memory only, by SHA-256. Old ones are dropped; nothing is ever written to disk."""

    def __init__(self):
        self._d: dict[str, tuple[float, bytes]] = {}
        self._lock = threading.Lock()

    def _tidy(self) -> None:
        now = time.monotonic()
        for k in [k for k, (t, _) in self._d.items() if now - t > BLOB_SECONDS]:
            del self._d[k]
        while self._d and (len(self._d) > MAX_BLOBS or sum(len(b) for _, b in self._d.values()) > MAX_BLOB_BYTES):
            del self._d[min(self._d, key=lambda k: self._d[k][0])]

    def put(self, sha: str, data: bytes) -> None:
        with self._lock:
            self._d[sha] = (time.monotonic(), data)
            self._tidy()

    def get(self, sha: str) -> bytes | None:
        with self._lock:
            self._tidy()
            hit = self._d.get(sha)
            if hit:
                self._d[sha] = (time.monotonic(), hit[1])
            return hit[1] if hit else None

    def clear(self) -> None:
        with self._lock:
            self._d.clear()


def decode(data: bytes, mode: str | None = None) -> Image.Image:
    try:
        im = Image.open(io.BytesIO(data))
        if im.format != "PNG":
            raise ValueError("not a PNG")
        if im.width * im.height > config.MAX_PIXELS:
            raise ValueError("too large")
        im.load()
    except Exception:
        raise HTTPException(400, "That is not a valid image.") from None
    return im.convert(mode) if mode else im


def png(img: Image.Image) -> Response:
    buf = io.BytesIO()
    img.save(buf, "PNG", compress_level=1)
    return Response(buf.getvalue(), media_type="image/png")


# ---------------------------------------------------------------- request shapes (every number is bounded)
class PairReq(BaseModel):
    code: str = Field(max_length=40)
    name: str = Field(default="Computer", max_length=60)


class SegmentReq(BaseModel):
    image: str = Field(pattern=r"^[0-9a-f]{64}$")
    points: list[list[float]] = Field(default_factory=list, max_length=24)
    box: list[float] | None = Field(default=None, min_length=4, max_length=4)


class ImageReq(BaseModel):
    image: str = Field(pattern=r"^[0-9a-f]{64}$")


class InpaintReq(ImageReq):
    mask: str = Field(pattern=r"^[0-9a-f]{64}$")
    grow: int | None = Field(default=None, ge=0, le=200)


class FillReq(InpaintReq):
    prompt: str = Field(max_length=1000)
    negative: str = Field(default="", max_length=1000)
    strength: float = Field(default=0.99, ge=0.0, le=1.0)
    seed: int | None = Field(default=None, ge=0, le=2**32 - 1)
    context: float = Field(default=0.4, ge=0.0, le=2.0)
    steps: int = Field(default=30, ge=1, le=60)
    feather_px: float = Field(default=3.0, ge=0.0, le=20.0)


class FacesReq(ImageReq):
    min_score: float = Field(default=0.6, ge=0.05, le=1.0)


class LandmarksReq(ImageReq):
    box: list[float] = Field(min_length=4, max_length=4)


def _finite(values: list[float]) -> bool:
    return all(math.isfinite(v) for v in values)


def create_app(data_dir: Path, engine: Engine | None = None, store: WorkerStore | None = None, name: str | None = None) -> FastAPI:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    store = store or WorkerStore(data_dir)
    engine = engine or Engine(config.default_models_dir())
    name = (name or os.environ.get("VEIL_WORKER_NAME") or socket.gethostname() or "worker")[:40]
    blobs = Blobs()
    slot_lock = threading.Semaphore(1)
    busy = {"now": False}
    bad: dict[str, list[float]] = {}
    app = FastAPI(title="Veil worker", docs_url=None, redoc_url=None, openapi_url=None)

    @app.middleware("http")
    async def guard(request: Request, call_next):
        peer = request.client.host if request.client else None
        if netguard.classify(peer) == "other":         # the listener never accepts these either; this is the second lock
            return JSONResponse({"detail": "This worker only accepts connections over Tailscale."}, status_code=403)
        if request.headers.get("origin"):               # a web page is never a client of this worker
            return JSONResponse({"detail": "Browsers cannot use this worker."}, status_code=403)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    def _peer(request: Request) -> str:
        return request.client.host if request.client else "?"

    def _count_bad(request: Request) -> None:
        now = time.monotonic()
        lst = [t for t in bad.get(_peer(request), []) if now - t < 60]
        lst.append(now)
        bad[_peer(request)] = lst

    def authed(request: Request) -> str:
        now = time.monotonic()
        if len([t for t in bad.get(_peer(request), []) if now - t < 60]) >= BAD_AUTH_PER_MINUTE:
            raise HTTPException(429, "Too many wrong tokens. Wait a minute.")
        header = request.headers.get("authorization", "")
        cid = store.check(header[7:].strip()) if header[:7].lower() == "bearer " else None
        if cid is None:
            _count_bad(request)
            raise HTTPException(401, "This computer is not paired with the worker.")
        request.state.client_id = cid
        return cid

    @contextmanager
    def slot():
        if not slot_lock.acquire(timeout=QUEUE_WAIT):
            raise HTTPException(503, "The worker is busy with another job.")
        busy["now"] = True
        try:
            yield
        finally:
            busy["now"] = False
            slot_lock.release()

    def run(fn):
        """Runs one job. Model problems are told in plain words; anything else is a generic 500 (never the image)."""
        with slot():
            try:
                return fn()
            except ModelError as e:
                raise HTTPException(400, str(e)) from None
            except HTTPException:
                raise
            except Exception as e:                       # noqa: BLE001
                log.error("job failed: %s", type(e).__name__)
                raise HTTPException(500, "The worker could not finish this job.") from None

    def need(*shas: str) -> list[bytes]:
        data = [blobs.get(s) for s in shas]
        missing = sorted({s for s, d in zip(shas, data) if d is None})
        if missing:
            raise HTTPException(409, {"need": missing})
        return data

    # ------------------------------------------------------------ pairing and status
    @app.post("/v1/pair")
    def pair(req: PairReq, request: Request):
        try:
            cid, token = store.redeem(req.code, req.name)
        except PairError as e:
            raise HTTPException(400, str(e)) from None
        return {"client_id": cid, "token": token, "worker_id": store.worker_id(), "name": name, "protocol": PROTOCOL}

    @app.get("/v1/health", dependencies=[Depends(authed)])
    def health():
        has = engine.has
        sel = any(has(n) for n, m in registry.MODELS.items() if m.get("group") == "selection")
        mat = any(has(n) for n, m in registry.MODELS.items() if m.get("group") == "matte")
        libs = engine.libraries_ok()
        try:
            mp_ok = engine.faces_ready()
        except Exception:                                  # noqa: BLE001
            mp_ok = False
        gpu, vram = "", 0.0
        try:
            import torch
            if torch.cuda.is_available():
                gpu = torch.cuda.get_device_name(0)
                vram = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
        except Exception:                                  # noqa: BLE001
            pass
        caps = {"segment": libs and sel, "matte": libs and mat, "inpaint": libs and has("lama"),
                "faces": libs and has("yunet"), "landmarks": libs and mp_ok, "generative": libs and engine.sdxl_ready()}
        return {"protocol": PROTOCOL, "worker_id": store.worker_id(), "name": name, "device": engine.device(), "gpu": gpu, "vram_gb": vram,
                "capabilities": caps, "models": [n for n in registry.MODELS if has(n)], "busy": busy["now"]}

    @app.post("/v1/unpair")
    def unpair(request: Request, cid: str = Depends(authed)):
        store.revoke(cid)
        return {"ok": True}

    # ------------------------------------------------------------ images (memory only)
    @app.put("/v1/images/{sha}", dependencies=[Depends(authed)])
    async def put_image(sha: str, request: Request):
        if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
            raise HTTPException(400, "Bad image id.")
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > MAX_BODY:
            raise HTTPException(413, "That image is too large.")
        chunks, total = [], 0
        async for chunk in request.stream():
            total += len(chunk)
            if total > MAX_BODY:
                raise HTTPException(413, "That image is too large.")
            chunks.append(chunk)
        data = b"".join(chunks)
        if hashlib.sha256(data).hexdigest() != sha:
            raise HTTPException(400, "The image did not arrive intact.")
        decode(data)                                       # PNG, sensible size, readable: checked before it is kept
        blobs.put(sha, data)
        return {"ok": True}

    # ------------------------------------------------------------ jobs (each takes images by id, never a face from anywhere else)
    @app.post("/v1/segment", dependencies=[Depends(authed)])
    def segment(req: SegmentReq):
        for p in req.points:
            if len(p) != 3 or not _finite(p) or not all(0 <= v <= 1 for v in p[:2]) or p[2] not in (0, 1):
                raise HTTPException(400, "Each point must be [x, y, 1 or 0] with x and y between 0 and 1.")
        b = req.box
        if b is not None and (not _finite(b) or not all(0 <= v <= 1 for v in b) or b[2] <= b[0] or b[3] <= b[1]):
            raise HTTPException(400, "The box must be [x0, y0, x1, y1] between 0 and 1.")
        (data,) = need(req.image)
        img = decode(data, "RGB")
        pts = [(float(p[0]), float(p[1]), int(p[2])) for p in req.points]
        return run(lambda: png(engine.segment(req.image, img, pts, tuple(b) if b else None)))

    @app.post("/v1/matte", dependencies=[Depends(authed)])
    def matte(req: ImageReq):
        (data,) = need(req.image)
        img = decode(data, "RGB")
        return run(lambda: png(engine.matte(img)))

    @app.post("/v1/inpaint", dependencies=[Depends(authed)])
    def inpaint(req: InpaintReq):
        d_img, d_mask = need(req.image, req.mask)
        img, mask = decode(d_img, "RGB"), decode(d_mask, "L")
        return run(lambda: png(engine.inpaint(img, mask, req.grow)))

    @app.post("/v1/generative_fill", dependencies=[Depends(authed)])
    def generative_fill(req: FillReq):
        d_img, d_mask = need(req.image, req.mask)
        img, mask = decode(d_img, "RGB"), decode(d_mask, "L")
        return run(lambda: png(engine.generative_fill(img, mask, req.prompt, req.negative, req.grow, strength=req.strength, seed=req.seed,
                                                       context=req.context, steps=req.steps, feather_px=req.feather_px)))

    @app.post("/v1/faces", dependencies=[Depends(authed)])
    def faces(req: FacesReq):
        (data,) = need(req.image)
        img = decode(data, "RGB")
        found = run(lambda: engine.detect_faces(img, req.min_score))
        return {"faces": [{"box": [float(v) for v in f["box"]], "score": float(f["score"])} for f in found[:50]]}

    @app.post("/v1/landmarks", dependencies=[Depends(authed)])
    def landmarks(req: LandmarksReq):
        if not _finite(req.box) or req.box[2] <= req.box[0] or req.box[3] <= req.box[1]:
            raise HTTPException(400, "Bad face box.")
        (data,) = need(req.image)
        img = decode(data, "RGB")
        pts = run(lambda: engine.landmarks(img, tuple(req.box)))
        return {"points": None if pts is None else [[float(x), float(y)] for x, y in pts]}

    app.state.store = store
    app.state.blobs = blobs
    app.state.engine = engine
    return app
