"""AI routes: the command box, model settings, and the model manager."""
from __future__ import annotations

import threading

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from ..history import PhotoError
from . import catalog, parser
from .providers import AIError, build
from .pull import Puller
from .settings import AISettings


class CommandRequest(BaseModel):
    photo_id: str
    text: str
    look_at_photo: bool = False          # let the model see the photo
    confirm_cloud_photo: bool = False    # say yes to sending it off this computer, for this one request


class ProviderRequest(BaseModel):
    id: str
    name: str = ""
    kind: str
    base_url: str
    model: str
    vision: bool = False
    api_key: str | None = None           # empty = keep the saved key


class ActiveRequest(BaseModel):
    id: str


class PullRequest(BaseModel):
    name: str = Field(max_length=80)


PHOTO_MAX_SIDE = 1024


def register(app: FastAPI, *, settings: AISettings, photos, small_jpeg, require_ready, require_owner) -> None:
    busy = threading.Lock()
    puller = Puller()

    def ai_call(fn, *args):
        try:
            return fn(*args)
        except AIError as e:
            raise HTTPException(400, str(e)) from None
        except PhotoError as e:
            raise HTTPException(404 if "not found" in str(e).lower() else 400, str(e)) from None

    @app.post("/api/ai/command", dependencies=[Depends(require_ready)])
    def command(req: CommandRequest, request: Request):
        provider = build(settings.active())
        image = None
        if req.look_at_photo:
            if not provider.vision:
                raise HTTPException(400, "The selected model cannot look at photos.")
            if not provider.local:
                # A photo leaves this computer only when asked for, for this request, at the computer itself.
                if request.state.scope != "local":
                    raise HTTPException(403, "Photos can only be sent to an online model from the computer itself.")
                if not req.confirm_cloud_photo:
                    raise HTTPException(400, "Sending the photo to an online model needs your confirmation each time.")
            current = ai_call(photos.open_current, req.photo_id)
            image = small_jpeg(current, PHOTO_MAX_SIDE, 85)   # a fresh JPEG: no EXIF, no location
        else:
            ai_call(photos.get, req.photo_id)   # still check the photo exists
        if not busy.acquire(blocking=False):
            raise HTTPException(429, "Veil is already working on a request.")
        try:
            plan = ai_call(parser.plan_from_text, provider, req.text, image)
        finally:
            busy.release()
        plan["asked"] = {"provider": provider.name, "model": provider.model, "local": provider.local,
                         "photo_sent": image is not None}
        return plan

    # ---- owner only: which model, and the model manager
    @app.get("/api/ai", dependencies=[Depends(require_ready)])
    def ai_status():
        pub = settings.public()
        act = next(p for p in pub["providers"] if p["id"] == pub["active"])
        return {"active": act}   # name, model, local, vision: enough for the command box; no keys

    @app.get("/api/ai/settings", dependencies=[Depends(require_owner)])
    def ai_settings():
        return {**settings.public(), "tiers": catalog.TIERS}

    @app.put("/api/ai/providers", dependencies=[Depends(require_owner)])
    def save_provider(req: ProviderRequest):
        ai_call(settings.save_provider, req.model_dump())
        return settings.public()

    @app.post("/api/ai/active", dependencies=[Depends(require_owner)])
    def set_active(req: ActiveRequest):
        ai_call(settings.set_active, req.id)
        return settings.public()

    @app.delete("/api/ai/providers/{provider_id}", dependencies=[Depends(require_owner)])
    def remove_provider(provider_id: str):
        ai_call(settings.remove, provider_id)
        return settings.public()

    @app.get("/api/ai/models", dependencies=[Depends(require_owner)])
    def models():
        """What the active provider has installed, next to the suggested tiers."""
        provider = build(settings.active())
        try:
            installed = provider.list_models()
            reachable = True
        except AIError:
            installed, reachable = [], False
        return {"provider": provider.name, "reachable": reachable, "installed": installed,
                "tiers": [{**t, "installed": t["name"] in installed} for t in catalog.TIERS],
                "can_download": provider.kind == "ollama" and provider.local, "download": puller.snapshot()}

    @app.post("/api/ai/models/pull", dependencies=[Depends(require_owner)])
    def pull(req: PullRequest):
        provider = build(settings.active())
        if provider.kind != "ollama" or not provider.local:
            raise HTTPException(400, "Models can only be downloaded into Ollama on this computer.")
        ai_call(puller.start, provider.base_url, req.name)
        return puller.snapshot()

    @app.post("/api/ai/test", dependencies=[Depends(require_owner)])
    def test_connection():
        """A tiny text-only request. Never sends a photo."""
        provider = build(settings.active())
        ai_call(parser.plan_from_text, provider, "make it a little brighter")
        return {"ok": True, "provider": provider.name, "model": provider.model}
