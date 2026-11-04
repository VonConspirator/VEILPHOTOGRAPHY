"""Veil web server: login, photos, edits, history, export, and phone pairing over Tailscale."""
from __future__ import annotations

from pathlib import Path
from urllib.parse import urlsplit

import io

from fastapi import Depends, FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

from . import config, netguard
from .auth import AuthStore
from .history import PhotoError, PhotoStore
from .phone import DEVICE_SECONDS, PairingError, PhoneStore
from .tailscale import Tailscale
from .tools import ops

WEB_DIR = Path(__file__).parent / "web"
CSP = ("default-src 'self'; img-src 'self' blob: data:; style-src 'self'; script-src 'self'; "
       "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
PREVIEW_MAX_SIDE = 1400

COOKIE = "veil_session"          # the owner at the computer
DEVICE_COOKIE = "veil_device"    # a paired phone
SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}


class Credentials(BaseModel):
    username: str
    password: str


class EditRequest(BaseModel):
    op: str
    params: dict = Field(default_factory=dict)
    mask: str | None = None  # the selection, as a PNG data URL (white = selected)


class EnableRequest(BaseModel):
    enabled: bool


class RedeemRequest(BaseModel):
    token: str
    name: str = "Phone"


class GotoRequest(BaseModel):
    cursor: int


class ExportRequest(BaseModel):
    format: str = "jpeg"
    quality: int = 92


def create_app(data_dir: Path | None = None, tailscale: Tailscale | None = None) -> FastAPI:
    data_dir = Path(data_dir) if data_dir else config.default_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    auth = AuthStore(data_dir)
    photos = PhotoStore(data_dir)
    phones = PhoneStore(data_dir)
    tailscale = tailscale or Tailscale()
    hosts = config.allowed_hosts()
    app = FastAPI(title="Veil", docs_url=None, redoc_url=None, openapi_url=None)

    # ------------------------------------------------------------ network guard
    @app.middleware("http")
    async def guard(request: Request, call_next):
        scope = netguard.classify(request.client.host if request.client else None)
        request.state.scope = scope
        if scope == "other":  # home Wi-Fi, the open internet: never
            return JSONResponse({"detail": "Veil only accepts connections from this computer and your Tailscale network."}, status_code=403)
        if scope == "tailnet":
            if not phones.enabled():
                return JSONResponse({"detail": "Phone access is switched off on the computer."}, status_code=403)
            if netguard.is_local_only_path(request.url.path):
                return JSONResponse({"detail": "That can only be done at the computer."}, status_code=403)
        host = (request.headers.get("host") or "").lower()
        bare = host.rsplit(":", 1)[0] if not host.endswith("]") else host
        allowed = hosts if scope == "local" else tailscale.info().names()
        if bare not in allowed and host not in allowed:
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
    def signed_in(request: Request) -> bool:
        """The owner's login at the computer, or an approved paired phone. Never the other way round."""
        if request.state.scope == "local":
            return auth.session_ok(request.cookies.get(COOKIE))
        return phones.device_ok(request.cookies.get(DEVICE_COOKIE))

    def require_user(request: Request) -> None:
        if not signed_in(request):
            raise HTTPException(401, "Please log in." if request.state.scope == "local" else "This phone is not paired.")

    def require_owner(request: Request) -> None:
        """Managing phones: the owner, logged in, at this computer."""
        if request.state.scope != "local" or not auth.session_ok(request.cookies.get(COOKIE)):
            raise HTTPException(403, "Only the owner at the computer can do that.")

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
        if request.state.scope != "local":  # a phone: never offer account setup or a password form
            return {"setup_needed": False, "logged_in": signed_in(request), "scope": "remote",
                    "device_state": phones.state_for(request.cookies.get(DEVICE_COOKIE)),
                    "adult_notice_acknowledged": auth.adult_acknowledged()}
        return {
            "setup_needed": not auth.has_user(),
            "logged_in": auth.session_ok(request.cookies.get(COOKIE)),
            "adult_notice_acknowledged": auth.adult_acknowledged(),
            "scope": "local",
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

    # ------------------------------------------------------------ phones: the owner, at the computer
    def pairing_call(fn, *args):
        try:
            return fn(*args)
        except PairingError as e:
            raise HTTPException(400, str(e)) from None

    @app.get("/api/phone", dependencies=[Depends(require_owner)])
    def phone_status():
        info = tailscale.info()
        listening = config.LISTENING
        return {
            "enabled": phones.enabled(),
            "tailscale": {"running": info.running, "address": info.ipv4, "name": info.dns_name or info.host_name,
                          "problem": info.problem,
                          "listening": True if listening is None else bool(info.ipv4 and info.ipv4 in listening)},
            "devices": phones.list(),
        }

    @app.post("/api/phone/enable", dependencies=[Depends(require_owner)])
    def phone_enable(req: EnableRequest):
        phones.set_enabled(req.enabled)
        return {"enabled": phones.enabled()}

    @app.post("/api/phone/pair", dependencies=[Depends(require_owner)])
    def phone_pair():
        """A fresh one-time QR code. The token sits after the # so it never appears in any server log."""
        info = tailscale.info()
        if not info.running:
            raise HTTPException(409, info.problem or "Tailscale is not running.")
        token, seconds = pairing_call(phones.new_token)
        url = f"http://{info.ipv4}:{config.PORT}/pair.html#{token}"
        import segno
        qr = segno.make(url, error="m").svg_data_uri(scale=7, border=3, dark="#000000", light="#ffffff")
        return {"url": url, "qr": qr, "expires_in": seconds}

    @app.post("/api/phone/devices/{dev_id}/approve", dependencies=[Depends(require_owner)])
    def phone_approve(dev_id: str):
        pairing_call(phones.approve, dev_id)
        return phones.list()

    @app.post("/api/phone/devices/{dev_id}/deny", dependencies=[Depends(require_owner)])
    def phone_deny(dev_id: str):
        pairing_call(phones.deny, dev_id)
        return phones.list()

    @app.delete("/api/phone/devices/{dev_id}", dependencies=[Depends(require_owner)])
    def phone_remove(dev_id: str):
        pairing_call(phones.forget, dev_id)
        return phones.list()

    # ------------------------------------------------------------ phones: the phone itself
    def require_phone(request: Request) -> None:
        if request.state.scope != "tailnet":
            raise HTTPException(403, "Pairing is done from the phone.")

    @app.post("/api/pair/redeem", dependencies=[Depends(require_phone)])
    def pair_redeem(req: RedeemRequest, request: Request, response: Response):
        secret, device = pairing_call(phones.redeem, req.token, req.name, request.client.host)
        response.set_cookie(DEVICE_COOKIE, secret, httponly=True, samesite="strict", max_age=DEVICE_SECONDS, path="/")
        return {"state": device["status"]}

    @app.get("/api/pair/state", dependencies=[Depends(require_phone)])
    def pair_state(request: Request):
        return {"state": phones.state_for(request.cookies.get(DEVICE_COOKIE))}

    @app.post("/api/pair/forget", dependencies=[Depends(require_phone)])
    def pair_forget(request: Request, response: Response):
        """A phone un-pairs itself."""
        device = phones.device_for(request.cookies.get(DEVICE_COOKIE))
        if device:
            phones.forget(device["id"])
        response.delete_cookie(DEVICE_COOKIE, path="/")
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
