"""Watermarks: saved presets (text or logo) and the drawing of them.

A watermark is only ever added at export. The working copy and every version in the history stay clean.
Presets are made on the computer; a paired phone can only list them and pick one. Text uses the bundled Inter font
(SIL Open Font License 1.1); a logo is the owner's own picture, kept as a PNG in the data folder.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import re
import secrets
import threading
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .history import PhotoError

FONT_PATH = Path(__file__).parent / "assets" / "fonts" / "Inter-Variable.ttf"
FONT_SHA256 = "29160a80ff49ddcab2c97711247e08b1fab27a484a329ce8b813d820dc559031"   # google/fonts ofl/inter @ 0b58fb37
POSITIONS = ("top-left", "top-center", "top-right", "middle-left", "center", "middle-right",
             "bottom-left", "bottom-center", "bottom-right", "tile")
KINDS = ("text", "logo")
WEIGHTS = {"regular": 400, "bold": 700}
MAX_LOGO_BYTES = 8 * 1024 * 1024
MAX_LOGO_SIDE = 2000
MAX_PRESETS = 50
ID_RE = re.compile(r"^[0-9a-f]{12}$")
HEX_RE = re.compile(r"^#[0-9a-fA-F]{6}$")
DEFAULTS = {"kind": "text", "text": "", "weight": "regular", "color": "#ffffff", "opacity": 60, "size": 25, "position": "bottom-right",
            "margin": 3, "angle": 0, "spacing": 60, "outline": True}
_lock = threading.RLock()
_font_ok = False


class WatermarkError(PhotoError):
    """Something the owner can be told about."""


def font_ready() -> bool:
    return FONT_PATH.is_file()


def _check_font() -> None:
    """The font file is checked against its pinned checksum the first time it is used."""
    global _font_ok
    if _font_ok:
        return
    if not font_ready():
        raise WatermarkError("Veil's watermark font is missing. Reinstall Veil.")
    if hashlib.sha256(FONT_PATH.read_bytes()).hexdigest() != FONT_SHA256:
        raise WatermarkError("Veil's watermark font file does not match its checksum, so it was not used. Reinstall Veil.")
    _font_ok = True


def _font(px: int, weight: str) -> ImageFont.FreeTypeFont:
    _check_font()
    f = ImageFont.truetype(str(FONT_PATH), max(6, int(px)))
    f.set_variation_by_axes([min(32, max(14, px * 0.25)), WEIGHTS[weight]])       # optical size, weight
    return f


def _unsupported(text: str, weight: str) -> list[str]:
    """Characters Inter cannot draw (they would show as empty boxes)."""
    f = _font(48, weight)
    box = bytes(f.getmask("￿"))
    bad = []
    for ch in dict.fromkeys(text):
        if ch.isspace():
            continue
        if bytes(f.getmask(ch)) == box and ch not in bad:
            bad.append(ch)
    return bad


# ---------------------------------------------------------------- checking a preset
def clean(fields: dict, base: dict | None = None) -> dict:
    """A preset from untrusted input: only known keys, every value checked and clamped."""
    if not isinstance(fields, dict):
        raise WatermarkError("That is not a watermark.")
    out = dict(DEFAULTS)
    if base:
        out.update({k: base[k] for k in DEFAULTS if k in base})
        out["name"] = base.get("name", "")
        if base.get("logo"):
            out["logo"] = True
    f = {k: v for k, v in fields.items() if k in DEFAULTS or k == "name"}
    out.update(f)
    name = " ".join(str(out.get("name", "")).split())[:60]
    if not name:
        raise WatermarkError("Give the watermark a name.")
    out["name"] = name
    if out["kind"] not in KINDS:
        raise WatermarkError("A watermark is either text or a logo.")
    if out["position"] not in POSITIONS:
        raise WatermarkError("Choose where the watermark goes.")
    if out["weight"] not in WEIGHTS:
        raise WatermarkError("Weight must be regular or bold.")
    if not isinstance(out["color"], str) or not HEX_RE.match(out["color"]):
        raise WatermarkError("The colour must look like #ffffff.")
    out["color"] = out["color"].lower()
    for key, lo, hi in (("opacity", 5, 100), ("size", 2, 80), ("margin", 0, 25), ("angle", -90, 90), ("spacing", 0, 400)):
        try:
            v = float(out[key])
        except (TypeError, ValueError):
            raise WatermarkError(f"{key} must be a number.") from None
        if not math.isfinite(v):
            raise WatermarkError(f"{key} must be a number.")
        out[key] = int(round(min(hi, max(lo, v))))
    out["outline"] = bool(out["outline"])
    text = "".join(c for c in str(out["text"]) if c == "\n" or c.isprintable()).strip()
    out["text"] = "\n".join(ln.strip() for ln in text.split("\n"))[:200]
    if out["kind"] == "text":
        if not out["text"]:
            raise WatermarkError("Type the words for the watermark.")
        bad = _unsupported(out["text"], out["weight"])
        if bad:
            raise WatermarkError("The watermark font cannot draw: " + " ".join(bad) + ". Use letters from Latin, Greek or Cyrillic alphabets, digits and common symbols, or use a logo.")
    return out


def _public(p: dict) -> dict:
    return {"id": p["id"], "name": p["name"], "kind": p["kind"], "text": p["text"], "weight": p["weight"], "color": p["color"],
            "opacity": p["opacity"], "size": p["size"], "position": p["position"], "margin": p["margin"], "angle": p["angle"],
            "spacing": p["spacing"], "outline": p["outline"], "has_logo": bool(p.get("logo"))}


# ---------------------------------------------------------------- the saved presets
class WatermarkStore:
    def __init__(self, data_dir: Path):
        self.dir = Path(data_dir) / "watermarks"
        self.file = self.dir / "presets.json"
        self.dir.mkdir(parents=True, exist_ok=True)

    def _read(self) -> dict:
        try:
            data = json.loads(self.file.read_text())
            if isinstance(data, dict) and isinstance(data.get("presets"), list):
                return data
        except (OSError, ValueError):
            pass
        return {"presets": [], "default": {"on": False, "preset": None}}

    def _write(self, data: dict) -> None:
        tmp = self.file.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1))
        tmp.replace(self.file)

    def overview(self) -> dict:
        with _lock:
            d = self._read()
            ids = {p["id"] for p in d["presets"]}
            df = d.get("default") or {}
            return {"presets": [_public(p) for p in d["presets"]],
                    "default": {"on": bool(df.get("on")) and df.get("preset") in ids, "preset": df.get("preset") if df.get("preset") in ids else None},
                    "font_ok": font_ready()}

    def get(self, pid: str) -> dict:
        with _lock:
            for p in self._read()["presets"]:
                if p["id"] == pid:
                    return p
        raise WatermarkError("That watermark does not exist.")

    def create(self, fields: dict) -> dict:
        p = clean(fields)
        if p["kind"] == "logo":
            p["text"] = ""
        with _lock:
            d = self._read()
            if len(d["presets"]) >= MAX_PRESETS:
                raise WatermarkError("That is the most watermarks Veil keeps. Delete one first.")
            p["id"] = secrets.token_hex(6)
            p["logo"] = False
            d["presets"].append(p)
            self._write(d)
            return _public(p)

    def update(self, pid: str, fields: dict) -> dict:
        with _lock:
            d = self._read()
            for i, old in enumerate(d["presets"]):
                if old["id"] == pid:
                    p = clean(fields, old)
                    p["id"], p["logo"] = pid, bool(old.get("logo"))
                    if p["kind"] == "logo":
                        p["text"] = ""
                    d["presets"][i] = p
                    self._write(d)
                    return _public(p)
        raise WatermarkError("That watermark does not exist.")

    def delete(self, pid: str) -> None:
        with _lock:
            d = self._read()
            keep = [p for p in d["presets"] if p["id"] != pid]
            if len(keep) == len(d["presets"]):
                raise WatermarkError("That watermark does not exist.")
            d["presets"] = keep
            if (d.get("default") or {}).get("preset") == pid:
                d["default"] = {"on": False, "preset": None}
            self._write(d)
            self.logo_path(pid).unlink(missing_ok=True)

    def set_default(self, on: bool, pid: str | None) -> dict:
        with _lock:
            d = self._read()
            if pid is not None and pid not in {p["id"] for p in d["presets"]}:
                raise WatermarkError("That watermark does not exist.")
            d["default"] = {"on": bool(on) and pid is not None, "preset": pid}
            self._write(d)
        return self.overview()["default"]

    # ------------------------------------------------------------ logos
    def logo_path(self, pid: str) -> Path:
        if not ID_RE.match(pid):
            raise WatermarkError("That watermark does not exist.")
        return self.dir / f"{pid}.png"

    def set_logo(self, pid: str, data: bytes) -> dict:
        """Takes the owner's picture, checks it really is one, and keeps a clean PNG (no hidden details) of it."""
        if len(data) > MAX_LOGO_BYTES:
            raise WatermarkError("That logo file is too large (8 MB at most).")
        try:
            with Image.open(io.BytesIO(data)) as im:
                if im.format not in ("PNG", "JPEG", "WEBP"):
                    raise WatermarkError("Use a PNG, JPEG or WebP logo. (Export an SVG logo as a PNG first.)")
                im.load()
                logo = im.convert("RGBA")
        except WatermarkError:
            raise
        except Exception:
            raise WatermarkError("Veil could not read that file as a picture.") from None
        logo.thumbnail((MAX_LOGO_SIDE, MAX_LOGO_SIDE), Image.Resampling.LANCZOS)
        clean_png = Image.new("RGBA", logo.size)
        clean_png.paste(logo)
        with _lock:
            d = self._read()
            for i, p in enumerate(d["presets"]):
                if p["id"] == pid:
                    buf = io.BytesIO()
                    clean_png.save(buf, "PNG")
                    self.logo_path(pid).write_bytes(buf.getvalue())
                    p["logo"] = True
                    d["presets"][i] = p
                    self._write(d)
                    return _public(p)
        raise WatermarkError("That watermark does not exist.")


# ---------------------------------------------------------------- drawing
def _text_item(p: dict, target_w: int) -> Image.Image:
    """The words as a transparent picture about `target_w` wide, in the preset's colour (opacity not applied yet)."""
    def draw(px: int) -> Image.Image:
        f = _font(px, p["weight"])
        probe = ImageDraw.Draw(Image.new("L", (1, 1)))
        stroke = max(1, round(px / 14)) if p["outline"] else 0
        l, t, r, b = (math.floor(v) if i < 2 else math.ceil(v) for i, v in enumerate(
            probe.multiline_textbbox((0, 0), p["text"], font=f, align="center", stroke_width=stroke)))
        pad = stroke + 2
        im = Image.new("RGBA", (r - l + 2 * pad, b - t + 2 * pad), (0, 0, 0, 0))
        rgb = tuple(int(p["color"][i:i + 2], 16) for i in (1, 3, 5))
        lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
        edge = (0, 0, 0, 170) if lum > 128 else (255, 255, 255, 170)
        ImageDraw.Draw(im).multiline_text((pad - l, pad - t), p["text"], font=f, fill=rgb + (255,), align="center",
                                          stroke_width=stroke, stroke_fill=edge)
        return im
    first = draw(100)
    px = max(6, round(100 * target_w / max(1, first.width)))
    return draw(min(px, 2000))


def _logo_item(path: Path, target_w: int, max_h: int) -> Image.Image:
    with Image.open(path) as im:
        logo = im.convert("RGBA")
    w = max(1, target_w)
    h = max(1, round(logo.height * w / logo.width))
    if h > max_h:
        w, h = max(1, round(w * max_h / h)), max_h
    return logo.resize((w, h), Image.Resampling.LANCZOS)


def _with_opacity(item: Image.Image, opacity: int) -> Image.Image:
    a = item.getchannel("A").point(lambda v: round(v * opacity / 100))
    item = item.copy()
    item.putalpha(a)
    return item


def render(img: Image.Image, preset: dict, logo_path: Path | None = None) -> Image.Image:
    """The picture with the watermark drawn on it. Returns a new image of the same size and mode."""
    W, H = img.size
    base = img.convert("RGBA")
    short = min(W, H)
    target_w = max(4, round(preset["size"] / 100 * W))
    if preset["kind"] == "logo":
        if logo_path is None or not Path(logo_path).is_file():
            raise WatermarkError("This watermark has no logo yet. Upload one first.")
        item = _logo_item(Path(logo_path), target_w, round(0.9 * H))
    else:
        item = _text_item(preset, target_w)
    item = _with_opacity(item, preset["opacity"])
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    if preset["position"] == "tile":
        gap_x = round(item.width * preset["spacing"] / 100) + 1
        gap_y = round(item.height * (0.5 + preset["spacing"] / 100)) + 1
        side = math.ceil(math.hypot(W, H)) + 2 * item.width
        big = Image.new("RGBA", (side, side), (0, 0, 0, 0))
        row = 0
        for y in range(-item.height, side, item.height + gap_y):
            shift = (item.width + gap_x) // 2 if row % 2 else 0
            for x in range(-item.width - shift, side, item.width + gap_x):
                if 0 <= x <= side - item.width and 0 <= y <= side - item.height:
                    big.alpha_composite(item, (x, y))
            row += 1
        if preset["angle"]:
            big = big.rotate(preset["angle"], resample=Image.Resampling.BICUBIC)
        cx, cy = side // 2, side // 2
        layer = big.crop((cx - W // 2, cy - H // 2, cx - W // 2 + W, cy - H // 2 + H))
    else:
        if preset["angle"]:
            item = item.rotate(preset["angle"], resample=Image.Resampling.BICUBIC, expand=True)
        m = round(preset["margin"] / 100 * short)
        v, h = ("center", "center") if preset["position"] == "center" else preset["position"].replace("middle", "center").split("-")
        x = {"left": m, "center": (W - item.width) // 2, "right": W - item.width - m}[h]
        y = {"top": m, "center": (H - item.height) // 2, "bottom": H - item.height - m}[v]
        if item.width > W or item.height > H:                      # never wider than the photo
            item = item.crop((0, 0, min(W, item.width), min(H, item.height)))
        layer.alpha_composite(item, (min(max(0, x), W - item.width), min(max(0, y), H - item.height)))
    out = Image.alpha_composite(base, layer)
    return out if img.mode == "RGBA" else out.convert("RGB")
