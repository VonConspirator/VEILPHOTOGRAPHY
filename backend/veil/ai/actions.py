"""The only things a typed command can turn into. The model proposes; this list decides.
Anything not listed here is refused, whatever the model says."""
from __future__ import annotations

import math

ADJUST_KEYS = ("brightness", "contrast", "saturation", "warmth", "sharpness")
ASPECTS = ("free", "orig", "1:1", "4:5", "5:4", "3:2", "2:3", "16:9")

# region: "none" = whole photo, "optional" = whole photo unless the user marks an area,
#         "required" = Veil must ask where (brush or box) before it can apply.
# available: False = understood, but the tool arrives in a later phase.
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
    "blur_face":        {"label": "Blur a face", "region": "required", "available": False,
                         "about": "Anonymise a face by blurring it."},
    "pixelate_face":    {"label": "Pixelate a face", "region": "required", "available": False,
                         "about": "Anonymise a face by pixelating it."},
    "generate_face":    {"label": "Replace a face with a generated one", "region": "required", "available": False,
                         "about": "Anonymise a face with a newly generated face (never one from another photo)."},
    "face_edit":        {"label": "Edit the face or look", "region": "optional", "available": False,
                         "about": "Skin, eyes, hair colour and similar look edits. what: short description."},
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
                         "about": "Add a text or logo watermark. text: the words."},
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
    elif action == "face_edit":
        out["what"] = str(raw.get("what", ""))[:200]
    elif action == "watermark":
        out["text"] = str(raw.get("text", ""))[:100]
    return out


def catalog_for_prompt() -> str:
    return "\n".join(f"- {name}: {a['about']}" for name, a in ACTIONS.items())
