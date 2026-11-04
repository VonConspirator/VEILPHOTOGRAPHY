"""Watermark routes. Everyone signed in can list the presets and export with one; only the owner at the computer can
create, change or delete them, upload a logo or set the default."""
from __future__ import annotations

from fastapi import Depends, FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .history import PhotoError
from .watermark import MAX_LOGO_BYTES, WatermarkError, WatermarkStore, clean, render

PREVIEW_SIDE = 1400


class PresetBody(BaseModel):
    fields: dict = Field(default_factory=dict)


class DefaultBody(BaseModel):
    on: bool = False
    preset: str | None = None


class PreviewBody(BaseModel):
    preset: str | None = None          # a saved preset
    draft: dict | None = None          # or one being edited (owner only); `id` lets a draft use that preset's logo


def register(app: FastAPI, *, store: WatermarkStore, photos, small_jpeg, require_ready, require_owner) -> None:
    def call(fn, *args):
        try:
            return fn(*args)
        except WatermarkError as e:
            raise HTTPException(404 if "does not exist" in str(e) else 400, str(e)) from None
        except PhotoError as e:
            raise HTTPException(404 if "not found" in str(e).lower() else 400, str(e)) from None

    @app.get("/api/watermarks", dependencies=[Depends(require_ready)])
    def overview():
        return store.overview()

    @app.post("/api/watermarks", dependencies=[Depends(require_ready), Depends(require_owner)])
    def create(body: PresetBody):
        return call(store.create, body.fields)

    @app.put("/api/watermarks/default", dependencies=[Depends(require_ready), Depends(require_owner)])
    def set_default(body: DefaultBody):
        return call(store.set_default, body.on, body.preset)

    @app.put("/api/watermarks/{pid}", dependencies=[Depends(require_ready), Depends(require_owner)])
    def update(pid: str, body: PresetBody):
        return call(store.update, pid, body.fields)

    @app.delete("/api/watermarks/{pid}", dependencies=[Depends(require_ready), Depends(require_owner)])
    def delete(pid: str):
        call(store.delete, pid)
        return {"ok": True}

    @app.post("/api/watermarks/{pid}/logo", dependencies=[Depends(require_ready), Depends(require_owner)])
    async def upload_logo(pid: str, file: UploadFile = File(...)):
        data = await file.read(MAX_LOGO_BYTES + 1)
        return call(store.set_logo, pid, data)

    @app.get("/api/watermarks/{pid}/logo", dependencies=[Depends(require_ready), Depends(require_owner)])
    def get_logo(pid: str):
        p = call(store.get, pid)
        path = store.logo_path(pid)
        if not p.get("logo") or not path.is_file():
            raise HTTPException(404, "No logo uploaded.")
        return FileResponse(path, media_type="image/png")

    @app.post("/api/photos/{photo_id}/watermark-preview", dependencies=[Depends(require_ready)])
    def preview(photo_id: str, body: PreviewBody, request: Request):
        """What the export would look like (a smaller copy, not saved)."""
        if body.draft is not None:
            require_owner(request)
            draft = dict(body.draft)
            pid = str(draft.pop("id", "") or "")
            preset = call(clean, draft)
            logo = store.logo_path(pid) if pid else None
        elif body.preset:
            preset, pid = call(store.get, body.preset), body.preset
            logo = store.logo_path(pid)
        else:
            raise HTTPException(400, "Choose a watermark.")
        current = call(photos.open_current, photo_id)
        small = current.copy()
        small.thumbnail((PREVIEW_SIDE, PREVIEW_SIDE))
        out = call(render, small, preset, logo)
        return Response(small_jpeg(out, PREVIEW_SIDE, 88), media_type="image/jpeg")


def export_overlay(store: WatermarkStore, pid: str):
    """The function the exporter draws with, for a saved preset (raises a clear error if it is gone)."""
    try:
        preset = store.get(pid)
        logo = store.logo_path(pid)
    except WatermarkError as e:
        raise HTTPException(400, str(e)) from None
    return lambda img: render(img, preset, logo)
