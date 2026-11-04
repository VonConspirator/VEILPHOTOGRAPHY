"""The only things a typed command can turn into. The model proposes; this list decides.
Anything not listed here is refused, whatever the model says."""
from __future__ import annotations

import math

ADJUST_KEYS = ("brightness", "contrast", "saturation", "warmth", "sharpness")
ASPECTS = ("free", "orig", "1:1", "4:5", "5:4", "3:2", "2:3", "16:9")

# region: "none" = whole photo, "optional" = whole photo unless the user marks an area,
#         "required" = Veil must ask where (brush or box) before it can apply,
#         "faces" = Veil finds the faces and the owner ticks which ones (a face is only ever edited from this photo).
# available: False = understood, but not something the command bar runs (the watermark is added when the photo is exported).
ACTIONS: dict[str, dict] = {
    "adjust":    {"label": "Adjust", "region": "optional", "available": True,
                  "about": "Change brightness, contrast, saturation, warmth or sharpness (each -100..100)."},
    "rotate90":  {"label": "Rotate", "region": "none", "available": True,
                  "about": "Quarter turns. turns: 1 = clockwise, -1 = counter-clockwise, 2 = upside down."},
    "flip":      {"label": "Flip", "region": "none", "available": True,
                  "about": "axis: horizontal or vertical."},
    "straighten": {"label": "Straighten", "region": "none", "available": True,
                   "about": "Level the horizon. degrees: -45..45, positive = clockwise."},
    "crop":      {"label": "Crop", "region": "none", "available": True,
                  "about": "Open the crop tool. aspect: one of " + ", ".join(ASPECTS) + "."},
    # --- faces (phase 6). region "faces": Veil finds the faces and the owner ticks which ones. Never a face from another photo.
    "blur_face":        {"label": "Blur a face", "region": "faces", "available": True,
                         "about": "Anonymise a face by blurring it. amount: 0..100 (default 80)."},
    "pixelate_face":    {"label": "Pixelate a face", "region": "faces", "available": True,
                         "about": "Anonymise a face by pixelating it. amount: 0..100 (default 70)."},
    "generate_face":    {"label": "Replace a face with a generated one", "region": "faces", "available": True,
                         "about": "Anonymise a face by painting a newly generated, made-up face (never one from another photo). No other settings."},
    "skin_smooth":      {"label": "Smooth skin", "region": "faces", "available": True,
                         "about": "Soften skin on the face, keeping some natural texture. amount: 0..100 (default 40)."},
    "teeth_whiten":     {"label": "Whiten teeth", "region": "faces", "available": True,
                         "about": "Whiten visible teeth. amount: 0..100 (default 60)."},
    "eye_color":        {"label": "Change eye colour", "region": "faces", "available": True,
                         "about": "Recolour the irises. color: blue, green, brown, hazel, grey, amber, violet or #rrggbb."},
    "eyelash_length":   {"label": "Lengthen eyelashes", "region": "faces", "available": True,
                         "about": "Longer, fuller upper lashes. amount: 0..100 (default 60)."},
    "makeup_add":       {"label": "Add makeup", "region": "faces", "available": True,
                         "about": "Add makeup. part: lips, blush, eyeshadow, eyeliner or face (all of them). color: a name or #rrggbb (optional). amount: 0..100."},
    "makeup_remove":    {"label": "Remove makeup", "region": "faces", "available": True,
                         "about": "Remove makeup, leaving a natural face. part: lips, eyes, cheeks or all (default). Needs the optional generative model."},
    "hair_color":       {"label": "Change hair colour", "region": "faces", "available": True,
                         "about": "Recolour the hair. color: for example black, brown, blonde, red, auburn, grey, pink, blue, or #rrggbb."},
    "hairstyle":        {"label": "Change the hairstyle", "region": "faces", "available": True,
                         "about": "Restyle the hair. style: a few plain words such as curly, wavy, straight with bangs. color: optional. "
                                  "Works best for texture (curly, wavy); changing the length is unreliable. Needs the optional generative model."},
    "facial_hair":      {"label": "Add or remove facial hair", "region": "faces", "available": True,
                         "about": "mode: add or remove. style: stubble, moustache, goatee or beard. Needs the optional generative model."},
    "remove_jewelry":   {"label": "Remove jewelry", "region": "required", "available": True,
                         "about": "Remove earrings, a necklace, rings or similar from the photo."},
    "remove_tattoo":    {"label": "Remove a tattoo", "region": "required", "available": True,
                         "about": "Remove a tattoo from the skin."},
    "remove_object":    {"label": "Remove an object", "region": "required", "available": True,
                         "about": "Remove something and fill in behind it."},
    "remove_background": {"label": "Remove the background", "region": "none", "available": True,
                          "about": "Cut out the person."},
    "replace_background": {"label": "Replace the background", "region": "none", "available": True,
                           "about": "Put a different background behind the person. color: a colour name or #rrggbb, OR scene: a short description (scenes need an optional model)."},
    "blur_background":  {"label": "Blur the background", "region": "none", "available": True,
                         "about": "Soften the background. amount: 0..100."},
    "watermark":        {"label": "Add a watermark", "region": "none", "available": False,
                         "about": "Add a text or logo watermark. text: the words. (Veil adds watermarks when the photo is exported.)"},
}


def _clamp(v, lo: float, hi: float) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return max(lo, min(hi, f))


def clean_params(action: str, raw: object) -> dict:
    """Keep only the parameters this action knows, forced into their allowed ranges."""
    raw = raw if isinstance(raw, dict) else {}
    out: dict = {}
    if action == "adjust":
        for k in ADJUST_KEYS:
            v = _clamp(raw.get(k), -100, 100)
            if v:
                out[k] = round(v)
    elif action == "rotate90":
        v = _clamp(raw.get("turns", 1), -3, 3)
        out["turns"] = round(v) if v else 1
    elif action == "flip":
        out["axis"] = "vertical" if str(raw.get("axis", "")).lower().startswith("v") else "horizontal"
    elif action == "straighten":
        v = _clamp(raw.get("degrees"), -45, 45)
        if v is not None:
            out["degrees"] = round(v, 1)
    elif action == "crop":
        a = str(raw.get("aspect", "free")).strip()
        out["aspect"] = a if a in ASPECTS else "free"
    elif action == "blur_background":
        v = _clamp(raw.get("amount", 50), 0, 100)
        out["amount"] = round(v if v is not None else 50)
    elif action == "replace_background":
        from ..vision.ops import parse_color
        color = parse_color(raw.get("color"))
        scene = str(raw.get("scene", "") or "")[:200].strip()
        if color:
            out["color"] = color
        elif scene:
            out["scene"] = scene
        else:
            out["color"] = "#ffffff"
    elif action in ("blur_face", "pixelate_face", "skin_smooth", "teeth_whiten", "eyelash_length"):
        v = _clamp(raw.get("amount"), 0, 100)
        if v is not None:
            out["amount"] = round(v)
    elif action == "eye_color":
        from ..vision.looks import EYE_COLORS
        from ..vision.ops import parse_color
        c = str(raw.get("color", "")).strip().lower()
        out["color"] = c if c in EYE_COLORS else (parse_color(c) or "blue")
    elif action == "makeup_add":
        from ..vision.looks import MAKEUP_PARTS, LIP_COLORS
        from ..vision.ops import parse_color
        part = str(raw.get("part", "lips")).strip().lower()
        out["part"] = part if part in MAKEUP_PARTS else "lips"
        c = str(raw.get("color", "")).strip().lower()
        col = c if c in LIP_COLORS else parse_color(c)
        if col:
            out["color"] = col
        v = _clamp(raw.get("amount"), 0, 100)
        if v is not None:
            out["amount"] = round(v)
    elif action == "makeup_remove":
        part = str(raw.get("part", "all")).strip().lower()
        out["part"] = part if part in ("lips", "eyes", "cheeks", "all") else "all"
    elif action == "hair_color":
        from ..vision.looks import HAIR_COLORS
        from ..vision.ops import parse_color
        c = str(raw.get("color", "")).strip().lower()
        out["color"] = c if c in HAIR_COLORS else (parse_color(c) or "")
    elif action == "hairstyle":
        import re
        st = re.sub(r"[^A-Za-z \-]", "", str(raw.get("style", "")))[:40].strip()
        out["style"] = st
        c = str(raw.get("color", "")).strip().lower()
        from ..vision.looks import HAIR_COLORS
        if c in HAIR_COLORS:
            out["color"] = c
    elif action == "facial_hair":
        from ..vision.looks import FACIAL_HAIR_STYLES
        out["mode"] = "remove" if str(raw.get("mode", "")).lower().startswith("r") else "add"
        st = str(raw.get("style", "stubble")).strip().lower()
        out["style"] = st if st in FACIAL_HAIR_STYLES else "stubble"
    elif action == "watermark":
        out["text"] = str(raw.get("text", ""))[:100]
    return out


def catalog_for_prompt() -> str:
    return "\n".join(f"- {name}: {a['about']}" for name, a in ACTIONS.items())
