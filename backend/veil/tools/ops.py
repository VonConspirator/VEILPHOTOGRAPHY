"""Basic edits: adjustments (optionally only inside a selection), crop, rotate, flip, straighten."""
from __future__ import annotations

import base64
import binascii
import io
import math

from PIL import Image, ImageEnhance

from ..history import PhotoError

ADJUST_KEYS = ("brightness", "contrast", "saturation", "warmth", "sharpness")
MAX_MASK_BYTES = 12 * 1024 * 1024


def _num(params: dict, key: str, lo: float, hi: float, default: float = 0.0) -> float:
    try:
        v = float(params.get(key, default))
    except (TypeError, ValueError):
        raise PhotoError(f"{key} must be a number.") from None
    if not math.isfinite(v) or not lo <= v <= hi:
        raise PhotoError(f"{key} must be between {lo:g} and {hi:g}.")
    return v


def decode_mask(data_url: str | None) -> Image.Image | None:
    """A selection sent by the page: a PNG as a data URL. White = selected."""
    if not data_url:
        return None
    try:
        raw = data_url.split(",", 1)[1] if data_url.startswith("data:") else data_url
        if len(raw) > MAX_MASK_BYTES * 4 // 3:
            raise PhotoError("The selection is too large.")
        img = Image.open(io.BytesIO(base64.b64decode(raw, validate=True)))
        return img.convert("L")
    except PhotoError:
        raise
    except (binascii.Error, ValueError, OSError, IndexError):
        raise PhotoError("The selection could not be read.") from None


def _split_alpha(img: Image.Image) -> tuple[Image.Image, Image.Image | None]:
    if img.mode == "RGBA":
        return img.convert("RGB"), img.getchannel("A")
    return img.convert("RGB"), None


def adjust(img: Image.Image, params: dict) -> Image.Image:
    """Each slider runs from -100 to 100; 0 changes nothing."""
    p = {k: _num(params, k, -100, 100) for k in ADJUST_KEYS}
    rgb, alpha = _split_alpha(img)
    if p["brightness"]:
        rgb = ImageEnhance.Brightness(rgb).enhance(1 + p["brightness"] / 100)
    if p["contrast"]:
        rgb = ImageEnhance.Contrast(rgb).enhance(1 + p["contrast"] / 100)
    if p["saturation"]:
        rgb = ImageEnhance.Color(rgb).enhance(1 + p["saturation"] / 100)
    if p["warmth"]:
        r, g, b = rgb.split()
        k = p["warmth"] / 250
        r = r.point(lambda v: min(255, int(v * (1 + k))))
        b = b.point(lambda v: min(255, int(v * (1 - k))))
        rgb = Image.merge("RGB", (r, g, b))
    if p["sharpness"]:
        rgb = ImageEnhance.Sharpness(rgb).enhance(max(0.0, 1 + p["sharpness"] / 50))
    if alpha is not None:
        rgb.putalpha(alpha)
    return rgb


def crop(img: Image.Image, params: dict) -> Image.Image:
    w, h = img.size
    x0 = round(_num(params, "x0", 0, w)); y0 = round(_num(params, "y0", 0, h))
    x1 = round(_num(params, "x1", 0, w)); y1 = round(_num(params, "y1", 0, h))
    x0, x1 = sorted((x0, x1)); y0, y1 = sorted((y0, y1))
    if x1 - x0 < 8 or y1 - y0 < 8:
        raise PhotoError("The crop area is too small.")
    return img.crop((x0, y0, x1, y1))


def rotate90(img: Image.Image, params: dict) -> Image.Image:
    turns = int(_num(params, "turns", -3, 3, 1)) % 4  # +1 = a quarter turn clockwise
    if turns == 0:
        raise PhotoError("Nothing to rotate.")
    return img.transpose({1: Image.Transpose.ROTATE_270, 2: Image.Transpose.ROTATE_180, 3: Image.Transpose.ROTATE_90}[turns])


def flip(img: Image.Image, params: dict) -> Image.Image:
    axis = params.get("axis")
    if axis == "horizontal":
        return img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if axis == "vertical":
        return img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    raise PhotoError("axis must be horizontal or vertical.")


def _inscribed_size(w: float, h: float, angle: float) -> tuple[float, float]:
    """The largest upright rectangle that fits inside a w x h rectangle rotated by `angle` radians."""
    long_is_w = w >= h
    long_side, short_side = (w, h) if long_is_w else (h, w)
    sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
    if short_side <= 2 * sin_a * cos_a * long_side or abs(sin_a - cos_a) < 1e-10:
        x = 0.5 * short_side
        return (x / sin_a, x / cos_a) if long_is_w else (x / cos_a, x / sin_a)
    cos_2a = cos_a * cos_a - sin_a * sin_a
    return (w * cos_a - h * sin_a) / cos_2a, (h * cos_a - w * sin_a) / cos_2a


def straighten(img: Image.Image, params: dict) -> Image.Image:
    """Level the horizon by a small angle (clockwise degrees), cropping away the empty corners."""
    deg = _num(params, "degrees", -45, 45)
    if abs(deg) < 0.05:
        raise PhotoError("Nothing to straighten.")
    w, h = img.size
    rotated = img.rotate(-deg, resample=Image.Resampling.BICUBIC, expand=True)
    cw, ch = _inscribed_size(w, h, math.radians(deg))
    cw, ch = max(8, min(rotated.width, int(cw))), max(8, min(rotated.height, int(ch)))
    left, top = (rotated.width - cw) // 2, (rotated.height - ch) // 2
    return rotated.crop((left, top, left + cw, top + ch))


GEOMETRY = {"crop": crop, "rotate90": rotate90, "flip": flip, "straighten": straighten}


def apply_op(img: Image.Image, op: str, params: dict, mask: Image.Image | None = None, engine=None) -> Image.Image:
    from ..vision import ops as vision_ops   # the AI edits (phase 5)
    if op in vision_ops.AI_OPS:
        return vision_ops.run(engine, op, img, params, mask)
    if op == "adjust":
        edited = adjust(img, params)
        if mask is not None:  # only the selected area changes
            m = mask.resize(img.size, Image.Resampling.BILINEAR)
            edited = Image.composite(edited, img.convert(edited.mode), m)
        return edited
    if op in GEOMETRY:
        if mask is not None:
            raise PhotoError("Crop, rotate and flip apply to the whole photo, not a selection.")
        return GEOMETRY[op](img, params)
    raise PhotoError("Unknown edit.")


def describe(op: str, params: dict, selected: bool) -> str:
    from ..vision import ops as vision_ops
    if op in vision_ops.AI_OPS:
        return vision_ops.describe(op, params)
    if op == "adjust":
        parts = [f"{k} {int(float(params[k])):+d}" for k in ADJUST_KEYS if k in params and float(params[k])]
        return "Adjust " + ", ".join(parts) + (" (selection)" if selected else "")
    return {"crop": "Crop", "rotate90": "Rotate", "flip": "Flip", "straighten": "Straighten"}.get(op, "Edit")
