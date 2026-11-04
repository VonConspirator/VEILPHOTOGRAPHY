"""Photos and their edit history (undo/redo). The original upload is never modified.

Each photo is a folder: original.<ext> (as uploaded), v0000.png (the untouched working copy),
v0001.png, ... (one per edit) and meta.json. `cursor` says which version is showing.
Editing after an undo discards the versions that were undone (the usual redo behaviour).
"""
from __future__ import annotations

import io
import json
import re
import secrets
import shutil
import threading
import time
import warnings
from pathlib import Path

from PIL import Image, ImageOps

from . import config

ID_RE = re.compile(r"^[0-9a-f]{16}$")
_lock = threading.RLock()
Image.MAX_IMAGE_PIXELS = config.MAX_PIXELS


class PhotoError(Exception):
    """A problem the user can be told about."""


def _normalize(img: Image.Image) -> Image.Image:
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGB", "RGBA"):
        return img
    if img.mode in ("LA", "PA") or "transparency" in img.info:
        return img.convert("RGBA")
    return img.convert("RGB")


class PhotoStore:
    def __init__(self, data_dir: Path):
        self.root = data_dir / "photos"

    # ------------------------------------------------------------ paths
    def _dir(self, photo_id: str) -> Path:
        if not ID_RE.match(photo_id or ""):
            raise PhotoError("Photo not found.")
        d = self.root / photo_id
        if not d.is_dir():
            raise PhotoError("Photo not found.")
        return d

    def _meta(self, photo_id: str) -> dict:
        return json.loads((self._dir(photo_id) / "meta.json").read_text())

    def _write_meta(self, photo_id: str, meta: dict) -> None:
        d = self.root / photo_id
        tmp = d / "meta.tmp"
        tmp.write_text(json.dumps(meta))
        tmp.replace(d / "meta.json")

    @staticmethod
    def _save_png(img: Image.Image, path: Path, icc: bytes | None) -> None:
        kwargs = {"icc_profile": icc} if icc else {}
        img.save(path, format="PNG", compress_level=3, **kwargs)

    # ------------------------------------------------------------ create / read
    def add_photo(self, data: bytes, filename: str) -> dict:
        if len(data) > config.MAX_UPLOAD_BYTES:
            raise PhotoError("That file is too large.")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")  # a decompression-bomb warning becomes an error
                probe = Image.open(io.BytesIO(data))
                fmt = probe.format
                probe.verify()
                img = Image.open(io.BytesIO(data))
                if fmt not in config.ALLOWED_UPLOAD_FORMATS:
                    raise PhotoError("Supported formats: JPEG, PNG, WebP, TIFF.")
                icc = img.info.get("icc_profile")
                img = _normalize(img)
                img.load()
        except PhotoError:
            raise
        except Exception:
            raise PhotoError("That file could not be read as an image (or it is too large).") from None

        photo_id = secrets.token_hex(8)
        with _lock:
            d = self.root / photo_id
            d.mkdir(parents=True)
            ext = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp", "TIFF": "tif"}[fmt]
            (d / f"original.{ext}").write_bytes(data)
            self._save_png(img, d / "v0000.png", icc)
            meta = {
                "id": photo_id,
                "name": Path(filename or "photo").name[:120],
                "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "width": img.width,
                "height": img.height,
                "has_icc": bool(icc),
                "versions": [{"n": 0, "label": "Original", "params": {}}],
                "cursor": 0,
            }
            self._write_meta(photo_id, meta)
        return meta

    def get(self, photo_id: str) -> dict:
        return self._meta(photo_id)

    def list(self) -> list[dict]:
        out = []
        if self.root.is_dir():
            for d in sorted(self.root.iterdir()):
                if ID_RE.match(d.name) and (d / "meta.json").is_file():
                    out.append(self._meta(d.name))
        return sorted(out, key=lambda m: m["created"], reverse=True)

    def version_path(self, photo_id: str, which: str | int = "current") -> Path:
        meta = self._meta(photo_id)
        if which == "current":
            n = meta["versions"][meta["cursor"]]["n"]
        elif which == "original":
            n = 0
        else:
            n = int(which)
            if n not in {v["n"] for v in meta["versions"]}:
                raise PhotoError("Version not found.")
        return self._dir(photo_id) / f"v{n:04d}.png"

    def open_current(self, photo_id: str) -> Image.Image:
        img = Image.open(self.version_path(photo_id))
        img.load()
        return img

    # ------------------------------------------------------------ edit history
    def add_version(self, photo_id: str, img: Image.Image, label: str, params: dict | None = None) -> dict:
        """Called by every edit tool: saves the result as the next version and makes it current."""
        with _lock:
            meta = self._meta(photo_id)
            d = self._dir(photo_id)
            for v in meta["versions"][meta["cursor"] + 1:]:  # drop the undone (redo) versions
                (d / f"v{v['n']:04d}.png").unlink(missing_ok=True)
            meta["versions"] = meta["versions"][: meta["cursor"] + 1]
            n = max(v["n"] for v in meta["versions"]) + 1
            icc = None
            if meta.get("has_icc"):
                with Image.open(d / "v0000.png") as first:
                    icc = first.info.get("icc_profile")
            self._save_png(_normalize(img), d / f"v{n:04d}.png", icc)
            meta["versions"].append({"n": n, "label": label[:80], "params": params or {}})
            meta["cursor"] = len(meta["versions"]) - 1
            meta["width"], meta["height"] = img.size
            self._write_meta(photo_id, meta)
            return meta

    def undo(self, photo_id: str) -> dict:
        with _lock:
            meta = self._meta(photo_id)
            if meta["cursor"] > 0:
                meta["cursor"] -= 1
                self._refresh_size(photo_id, meta)
                self._write_meta(photo_id, meta)
            return meta

    def redo(self, photo_id: str) -> dict:
        with _lock:
            meta = self._meta(photo_id)
            if meta["cursor"] < len(meta["versions"]) - 1:
                meta["cursor"] += 1
                self._refresh_size(photo_id, meta)
                self._write_meta(photo_id, meta)
            return meta

    def _refresh_size(self, photo_id: str, meta: dict) -> None:
        n = meta["versions"][meta["cursor"]]["n"]  # use the new cursor, which is not on disk yet
        with Image.open(self._dir(photo_id) / f"v{n:04d}.png") as im:  # crops change the size
            meta["width"], meta["height"] = im.size

    def goto(self, photo_id: str, cursor: int) -> dict:
        with _lock:
            meta = self._meta(photo_id)
            if not 0 <= cursor < len(meta["versions"]):
                raise PhotoError("That step does not exist.")
            meta["cursor"] = cursor
            self._refresh_size(photo_id, meta)
            self._write_meta(photo_id, meta)
            return meta

    def delete(self, photo_id: str) -> None:
        with _lock:
            shutil.rmtree(self._dir(photo_id))

    # ------------------------------------------------------------ export
    def export(self, photo_id: str, fmt: str = "jpeg", quality: int = 92, overlay=None) -> tuple[bytes, str]:
        """Exports the current version with all metadata removed (EXIF, GPS, XMP, text chunks).
        Only the pixels and the colour profile are kept, so colours still look right.
        `overlay`, if given, draws on the exported copy only (the watermark); the saved versions are never touched."""
        fmt = (fmt or "jpeg").lower()
        if fmt not in ("jpeg", "png", "webp"):
            raise PhotoError("Export format must be jpeg, png or webp.")
        quality = max(1, min(100, int(quality)))
        with Image.open(self.version_path(photo_id)) as src:
            icc = src.info.get("icc_profile")
            img = src.convert("RGBA" if src.mode == "RGBA" else "RGB")
        if overlay is not None:
            img = overlay(img)
        clean = Image.new(img.mode, img.size)
        clean.paste(img)  # a fresh image carries no metadata
        if fmt == "jpeg" and clean.mode == "RGBA":
            flat = Image.new("RGB", clean.size, (255, 255, 255))
            flat.paste(clean, mask=clean.getchannel("A"))
            clean = flat
        buf = io.BytesIO()
        kwargs: dict = {"icc_profile": icc} if icc else {}
        if fmt == "jpeg":
            clean.save(buf, "JPEG", quality=quality, optimize=True, **kwargs)
        elif fmt == "webp":
            clean.save(buf, "WEBP", quality=quality, **kwargs)
        else:
            clean.save(buf, "PNG", **kwargs)
        return buf.getvalue(), {"jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}[fmt]
