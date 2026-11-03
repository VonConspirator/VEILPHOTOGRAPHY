"""Veil web server. Phase 1: login, adult-use notice, photo upload, history (undo/redo), export."""
from __future__ import annotations

from pathlib import Path
from urllib.parse import urlsplit

import io

from fastapi import Depends, FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

from . import config
from .auth import AuthStore
from .history import PhotoError, PhotoStore
from .tools import ops

WEB_DIR = Path(__file__).parent / "web"
CSP = ("default-src 'self'; img-src 'self' blob: data:; style-src 'self'; script-src 'self'; "
       "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
PREVIEW_MAX_SIDE = 1400

COOKIE = "veil_session"
SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}


class Credentials(BaseModel):
    username: str
    password: str


class EditRequest(BaseModel):
    op: str
    params: dict = Field(default_factory=dict)
    mask: str | None = None  # the selection, as a PNG data URL (white = selected)


class GotoRequest(BaseModel):
    cursor: int


class ExportRequest(BaseModel):
    format: str = "jpeg"
    quality: int = 92


def create_app(data_dir: Path | None = None) -> FastAPI:
    data_dir = Path(data_dir) if data_dir else config.default_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    auth = AuthStore(data_dir)
    photos = PhotoStore(data_dir)
    hosts = config.allowed_hosts()
    app = FastAPI(title="Veil", docs_url=None, redoc_url=None, openapi_url=None)

    # ------------------------------------------------------------ network guard
    @app.middleware("http")
    async def guard(request: Request, call_next):
        host = (request.headers.get("host") or "").lower()
        bare = host.rsplit(":", 1)[0] if not host.endswith("]") else host
        if bare not in hosts and host not in hosts:
            return JSONResponse({"detail": "Unknown host."}, status_code=421)
        if request.method not in SAFE_METHODS:
            origin = request.headers.get("origin")
            if origin and urlsplit(origin).netloc.lower() != host:
                return JSONResponse({"detail": "Cross-site request refused."}, status_code=403)
            if not origin and request.headers.get("sec-fetch-site", "same-origin") == "cross-site":
                return JSONResponse({"detail": "Cross-site request refused."}, status_code=403)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = CSP
        return response

    # ------------------------------------------------------------ dependencies
    def require_user(request: Request) -> None:
        if not auth.session_ok(request.cookies.get(COOKIE)):
            raise HTTPException(401, "Please log in.")

    def require_ready(request: Request) -> None:
        require_user(request)
        if not auth.adult_acknowledged():
            raise HTTPException(403, "Please confirm the adult-use notice first.")

    def photo_call(fn, *args):
        try:
            return fn(*args)
        except PhotoError as e:
            raise HTTPException(404 if "not found" in str(e).lower() else 400, str(e)) from None

    # ------------------------------------------------------------ account
    @app.get("/api/status")
    def status(request: Request):
        return {
            "setup_needed": not auth.has_user(),
            "logged_in": auth.session_ok(request.cookies.get(COOKIE)),
            "adult_notice_acknowledged": auth.adult_acknowledged(),
        }

    def _start_session(response: Response, token: str) -> None:
        response.set_cookie(COOKIE, token, httponly=True, samesite="strict", max_age=config.SESSION_SECONDS, path="/")

    @app.post("/api/auth/setup")
    def setup(creds: Credentials, response: Response, request: Request):
        try:
            auth.create_owner(creds.username, creds.password)
        except ValueError as e:
            raise HTTPException(400, str(e)) from None
        except PermissionError as e:
            raise HTTPException(409, str(e)) from None
        token = auth.login(creds.username, creds.password, request.client.host if request.client else "?")
        _start_session(response, token)
        return {"ok": True}

    @app.post("/api/auth/login")
    def login(creds: Credentials, response: Response, request: Request):
        try:
            token = auth.login(creds.username, creds.password, request.client.host if request.client else "?")
        except PermissionError as e:
            raise HTTPException(429, str(e)) from None
        if not token:
            raise HTTPException(401, "Wrong username or password.")
        _start_session(response, token)
        return {"ok": True}

    @app.post("/api/auth/logout")
    def logout(request: Request, response: Response):
        auth.logout(request.cookies.get(COOKIE))
        response.delete_cookie(COOKIE, path="/")
        return {"ok": True}

    @app.post("/api/notice/ack", dependencies=[Depends(require_user)])
    def ack_notice():
        auth.acknowledge_adult_notice()
        return {"ok": True}

    # ------------------------------------------------------------ photos
    @app.post("/api/photos", dependencies=[Depends(require_ready)])
    async def upload(file: UploadFile = File(...)):
        data = await file.read(config.MAX_UPLOAD_BYTES + 1)
        return photo_call(photos.add_photo, data, file.filename or "photo")

    @app.get("/api/photos", dependencies=[Depends(require_ready)])
    def list_photos():
        return photos.list()

    @app.get("/api/photos/{photo_id}", dependencies=[Depends(require_ready)])
    def get_photo(photo_id: str):
        return photo_call(photos.get, photo_id)

    @app.get("/api/photos/{photo_id}/image", dependencies=[Depends(require_ready)])
    def get_image(photo_id: str, version: str = "current", size: int = 0):
        if version not in ("current", "original") and not version.isdigit():
            raise HTTPException(400, "version must be current, original or a number.")
        path = photo_call(photos.version_path, photo_id, version)
        if size:  # a smaller copy for thumbnails
            return Response(_small_jpeg(Image.open(path), max(64, min(2048, size)), 80), media_type="image/jpeg")
        return Response(path.read_bytes(), media_type="image/png")

    def _run_edit(photo_id: str, req: EditRequest, source: Image.Image):
        mask = photo_call(ops.decode_mask, req.mask)
        return photo_call(ops.apply_op, source, req.op, req.params, mask)

    @app.post("/api/photos/{photo_id}/edit", dependencies=[Depends(require_ready)])
    def edit(photo_id: str, req: EditRequest):
        current = photo_call(photos.open_current, photo_id)
        result = _run_edit(photo_id, req, current)
        label = ops.describe(req.op, req.params, bool(req.mask))
        return photo_call(photos.add_version, photo_id, result, label, {"op": req.op, **req.params})

    @app.post("/api/photos/{photo_id}/preview", dependencies=[Depends(require_ready)])
    def preview(photo_id: str, req: EditRequest):
        """The same edit on a smaller copy, not saved. Lets the sliders feel instant."""
        current = photo_call(photos.open_current, photo_id)
        small = current.copy()
        small.thumbnail((PREVIEW_MAX_SIDE, PREVIEW_MAX_SIDE), Image.Resampling.LANCZOS)
        return Response(_small_jpeg(_run_edit(photo_id, req, small), PREVIEW_MAX_SIDE, 85), media_type="image/jpeg")

    @app.post("/api/photos/{photo_id}/goto", dependencies=[Depends(require_ready)])
    def goto(photo_id: str, req: GotoRequest):
        return photo_call(photos.goto, photo_id, req.cursor)

    @app.post("/api/photos/{photo_id}/undo", dependencies=[Depends(require_ready)])
    def undo(photo_id: str):
        return photo_call(photos.undo, photo_id)

    @app.post("/api/photos/{photo_id}/redo", dependencies=[Depends(require_ready)])
    def redo(photo_id: str):
        return photo_call(photos.redo, photo_id)

    @app.post("/api/photos/{photo_id}/export", dependencies=[Depends(require_ready)])
    def export(photo_id: str, req: ExportRequest):
        body, mime = photo_call(photos.export, photo_id, req.format, req.quality)
        name = Path(photos.get(photo_id)["name"]).stem or "photo"
        ext = "jpg" if req.format.lower() == "jpeg" else req.format.lower()
        return Response(body, media_type=mime,
                        headers={"Content-Disposition": f'attachment; filename="{name}-veil.{ext}"'})

    @app.delete("/api/photos/{photo_id}", dependencies=[Depends(require_ready)])
    def delete(photo_id: str):
        photo_call(photos.delete, photo_id)
        return {"ok": True}

    app.state.photos = photos
    app.state.auth = auth
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")  # the editor page (after the API routes)
    return app


def _small_jpeg(img: Image.Image, max_side: int, quality: int) -> bytes:
    img = img.copy()
    img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    if img.mode == "RGBA":
        flat = Image.new("RGB", img.size, (255, 255, 255))
        flat.paste(img, mask=img.getchannel("A"))
        img = flat
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def app_factory() -> FastAPI:
    return create_app()
