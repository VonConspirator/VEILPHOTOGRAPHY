"""Editing-model routes: what is installed, optional downloads, and AI selection (click / box -> outline)."""
from __future__ import annotations

import io

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from ..history import PhotoError
from . import registry
from .engine import Engine
from .registry import Downloader, ModelError

MAX_POINTS = 24
OUT_MAX_SIDE = 1024


class SegmentRequest(BaseModel):
    points: list[list[float]] = Field(default_factory=list)   # [x, y, 1 or 0], fractions 0..1
    box: list[float] | None = None                             # [x0, y0, x1, y1], fractions 0..1


def register(app: FastAPI, *, engine: Engine, photos, require_ready, require_owner) -> None:
    downloader = Downloader(engine.root)

    def call(fn, *args):
        try:
            return fn(*args)
        except ModelError as e:
            raise HTTPException(400, str(e)) from None
        except PhotoError as e:
            raise HTTPException(404 if "not found" in str(e).lower() else 400, str(e)) from None

    @app.get("/api/vision", dependencies=[Depends(require_ready)])
    def vision_status():
        return {**engine.status(), "download": downloader.snapshot()}

    @app.post("/api/vision/models/{name}/download", dependencies=[Depends(require_owner)])
    def download(name: str):
        if name not in registry.MODELS:
            raise HTTPException(404, "Unknown model.")
        call(downloader.start, name)
        return downloader.snapshot()

    @app.post("/api/photos/{photo_id}/faces", dependencies=[Depends(require_ready)])
    def find_faces(photo_id: str):
        """Finds the faces so the owner can tick which ones an edit applies to. Boxes are fractions 0..1 of the photo."""
        img = call(photos.open_current, photo_id)
        found = call(engine.detect_faces, img)
        w, h = img.size
        return {"faces": [{"box": [round(d["box"][0] / w, 5), round(d["box"][1] / h, 5), round(d["box"][2] / w, 5), round(d["box"][3] / h, 5)],
                           "score": round(d["score"], 3)} for d in found[:20]],
                "mapped": engine.has("face-landmarker")}

    @app.post("/api/photos/{photo_id}/segment", dependencies=[Depends(require_ready)])
    def segment(photo_id: str, req: SegmentRequest):
        if len(req.points) > MAX_POINTS:
            raise HTTPException(400, "Too many points.")
        pts = []
        for p in req.points:
            if len(p) != 3 or not all(0 <= v <= 1 for v in p[:2]) or p[2] not in (0, 1):
                raise HTTPException(400, "Each point must be [x, y, 1 or 0] with x and y between 0 and 1.")
            pts.append((float(p[0]), float(p[1]), int(p[2])))
        box = None
        if req.box is not None:
            b = req.box
            if len(b) != 4 or not all(0 <= v <= 1 for v in b) or b[2] <= b[0] or b[3] <= b[1]:
                raise HTTPException(400, "The box must be [x0, y0, x1, y1] between 0 and 1.")
            box = tuple(float(v) for v in b)
        if not pts and box is None:
            raise HTTPException(400, "Click on the thing you want to select, or draw a box around it.")
        meta = call(photos.get, photo_id)
        img = call(photos.open_current, photo_id)
        key = f"{photo_id}:{meta['versions'][meta['cursor']]['n']}"
        mask = call(engine.segment, key, img, pts, box)
        mask.thumbnail((OUT_MAX_SIDE, OUT_MAX_SIDE))
        buf = io.BytesIO(); mask.save(buf, "PNG")
        return Response(buf.getvalue(), media_type="image/png")
