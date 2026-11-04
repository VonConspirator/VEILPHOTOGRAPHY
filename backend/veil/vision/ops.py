"""The phase 5 edits: remove an object or tattoo, remove / blur / replace the background.
Each takes the current photo, the parameters and (where it makes sense) a selection."""
from __future__ import annotations

import re

import numpy as np
from PIL import Image, ImageFilter

from ..history import PhotoError
from .engine import Engine, binarize
from .registry import ModelError

AI_OPS = ("remove_object", "remove_tattoo", "remove_background", "blur_background", "replace_background")
NEEDS_AREA = ("remove_object", "remove_tattoo")
COLORS = {"white": "#ffffff", "black": "#000000", "grey": "#808080", "gray": "#808080", "red": "#c0392b", "blue": "#2e5eaa",
          "green": "#2e8b57", "pink": "#e58fb0", "beige": "#d8c3a5", "cream": "#f5efe0", "yellow": "#e6c84a", "purple": "#7d4fa3"}
HEX_RE = re.compile(r"^#[0-9a-fA-F]{6}$")
TATTOO_PROMPT = ("clean bare skin with natural skin texture and pores, seamless, photograph", "tattoo, ink, drawing, text, scar, deformed, blurry")
OBJECT_PROMPT = ("natural background continuing seamlessly, photograph", "object, text, watermark, deformed, blurry")


def parse_color(value) -> str | None:
    v = str(value or "").strip().lower()
    if HEX_RE.match(v):
        return v
    return COLORS.get(v)


def _num(params: dict, key: str, lo: float, hi: float, default: float) -> float:
    try:
        v = float(params.get(key, default))
    except (TypeError, ValueError):
        raise PhotoError(f"{key} must be a number.") from None
    if not lo <= v <= hi or v != v:
        raise PhotoError(f"{key} must be between {lo:g} and {hi:g}.")
    return v


def _rgb(img: Image.Image) -> Image.Image:
    return img.convert("RGB")


def _foreground(engine: Engine, img: Image.Image, mask: Image.Image | None) -> Image.Image:
    """The person: the owner's selection if there is one, otherwise found automatically."""
    if mask is not None:
        return mask.resize(img.size, Image.Resampling.BILINEAR).filter(ImageFilter.GaussianBlur(1.0))
    return engine.matte(img)


def _alpha_keep(src: Image.Image, out: Image.Image) -> Image.Image:
    if src.mode == "RGBA":
        out = out.convert("RGB"); out.putalpha(src.getchannel("A"))
    return out


def run(engine: Engine, op: str, img: Image.Image, params: dict, mask: Image.Image | None) -> Image.Image:
    if engine is None:
        raise PhotoError("The editing models are not set up.")
    try:
        return _run(engine, op, img, params, mask)
    except ModelError as e:
        raise PhotoError(str(e)) from None


def _run(engine: Engine, op: str, img: Image.Image, params: dict, mask: Image.Image | None) -> Image.Image:
    if op in NEEDS_AREA:
        if mask is None:
            raise PhotoError("Mark the area first (brush, box, or click it with the Point tool).")
        if np.count_nonzero(binarize(mask, img.size)) == 0:
            raise PhotoError("Mark the area first (brush, box, or click it with the Point tool).")
        generative = str(params.get("engine", "lama")) == "sdxl"
        grow = int(_num(params, "grow", 0, 200, 0)) or None
        if op == "remove_tattoo" and grow is None:
            grow = max(6, round(0.006 * max(img.size)))      # tattoos have soft edges: take a little extra
        if generative:
            pos, neg = TATTOO_PROMPT if op == "remove_tattoo" else OBJECT_PROMPT
            out = engine.generative_fill(img, mask, str(params.get("prompt") or pos), neg, grow)
        else:
            out = engine.inpaint(img, mask, grow)
        return _alpha_keep(img, out)

    if op == "remove_background":
        matte = _foreground(engine, img, mask)
        out = _rgb(img); out.putalpha(matte)
        if img.mode == "RGBA":
            a = np.minimum(np.asarray(img.getchannel("A")), np.asarray(matte))
            out.putalpha(Image.fromarray(a, "L"))
        return out

    if op == "blur_background":
        amount = _num(params, "amount", 0, 100, 50)
        matte = _foreground(engine, img, mask)
        radius = max(1.0, amount / 100 * 0.03 * max(img.size))
        base = _rgb(img)
        out = Image.composite(base, base.filter(ImageFilter.GaussianBlur(radius)), matte)
        return _alpha_keep(img, out)

    if op == "replace_background":
        matte = _foreground(engine, img, mask)
        base = _rgb(img)
        scene = str(params.get("scene") or "").strip()
        if scene:                                           # a described scene needs the optional generative model
            if not engine.sdxl_ready():
                raise PhotoError("Describing a new background needs the optional Stable Diffusion XL Inpainting download. "
                                 "Use a plain colour instead, or download it under Model, then Editing models.")
            inverse = Image.fromarray(255 - np.asarray(matte), "L")
            out = engine.generative_fill(base, inverse, f"{scene[:200]}, photograph, natural light, behind a person",
                                         "person, people, face, text, watermark, deformed", grow=6)
        else:
            color = parse_color(params.get("color")) or "#ffffff"
            out = Image.composite(base, Image.new("RGB", base.size, color), matte)
        return _alpha_keep(img, out)

    raise PhotoError("Unknown edit.")


def describe(op: str, params: dict) -> str:
    return {"remove_object": "Remove object", "remove_tattoo": "Remove tattoo", "remove_background": "Remove background",
            "blur_background": f"Blur background {int(float(params.get('amount', 50)))}%",
            "replace_background": "Replace background"}[op]
