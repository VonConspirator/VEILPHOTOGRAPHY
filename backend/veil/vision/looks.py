"""The phase 6 edits: hide a face (blur, pixelate, generated face) and the look edits (skin, eyes, teeth, makeup, hair,
facial hair, eyelashes, jewelry).

Rules that hold for every edit in this file:
  * The faces to change are named by the page (boxes inside THIS photo). Nothing runs on "all faces" by itself.
  * A face is never taken from another picture. No edit accepts an image, a reference or a source of any kind, and the
    generative model is only ever given this photo, a mask and a fixed sentence, never a face to copy.
  * Pixels outside the edited area are left byte-for-byte as they were.
"""
from __future__ import annotations

import re
import time

import cv2
import numpy as np
from PIL import Image

from ..history import PhotoError
from . import faces as F
from .engine import Engine, binarize
from .ops import FACE_OPS, parse_color  # noqa: F401
from .registry import ModelError

MAX_FACES = 20
# Words a request must never carry: there is no way to hand these edits a picture to copy a face from.
FORBIDDEN_KEYS = ("image", "images", "reference", "ref", "source", "src", "face_image", "swap", "target_face", "donor", "photo", "url", "file", "identity")
STYLE_RE = re.compile(r"^[A-Za-z][A-Za-z \-]{1,39}$")

EYE_COLORS = {"blue": "#4a86c5", "green": "#4f9a5b", "brown": "#6b4423", "hazel": "#8e6b3a", "grey": "#8a9aa5", "gray": "#8a9aa5",
              "amber": "#b8801f", "violet": "#8a63c4", "black": "#1a1a1a"}
HAIR_COLORS = {"black": "#1c1a19", "brown": "#4a3020", "dark brown": "#33201a", "light brown": "#7a5a3a", "chestnut": "#5e3a26",
               "blonde": "#d9b27a", "blond": "#d9b27a", "platinum": "#e8dcc0", "red": "#a8401f", "auburn": "#7a2f1c",
               "ginger": "#b5541f", "grey": "#9a9a9a", "gray": "#9a9a9a", "silver": "#c0c0c0", "white": "#ece8e0",
               "pink": "#e58fb0", "blue": "#2e5eaa", "green": "#2e8b57", "purple": "#7d4fa3", "teal": "#2a8a8a"}
LIP_COLORS = {"red": "#b3202a", "pink": "#e0709a", "nude": "#c48a7a", "coral": "#e5675a", "plum": "#7a2f4f", "berry": "#8a2a4f",
              "brown": "#8a4a3a", "orange": "#d9602a", "purple": "#6a2f7a"}
MAKEUP_DEFAULT = {"lips": "#b3202a", "blush": "#e58fb0", "eyeshadow": "#8a6a9a", "eyeliner": "#101010"}
MAKEUP_PARTS = ("lips", "blush", "eyeshadow", "eyeliner", "face")
FACIAL_HAIR_STYLES = ("stubble", "moustache", "mustache", "goatee", "beard")

GEN_NEEDS = ("generate_face", "makeup_remove", "hairstyle", "facial_hair")     # these need the SDXL download


# ---------------------------------------------------------------- inputs
def _num(params: dict, key: str, lo: float, hi: float, default: float) -> float:
    try:
        v = float(params.get(key, default))
    except (TypeError, ValueError):
        raise PhotoError(f"{key} must be a number.") from None
    if not lo <= v <= hi or v != v:
        raise PhotoError(f"{key} must be between {lo:g} and {hi:g}.")
    return v


def _color(params: dict, key: str, table: dict, default: str) -> str:
    raw = str(params.get(key) or "").strip().lower()
    if not raw:
        return default
    return table.get(raw) or parse_color(raw) or _bad_color(raw)


def _bad_color(raw: str):
    raise PhotoError(f"I don't know the colour '{raw[:30]}'. Use a name like blue or a code like #3b7dd8.")


def _style(params: dict, key: str = "style") -> str:
    s = re.sub(r"\s+", " ", str(params.get(key) or "").strip())
    if not STYLE_RE.match(s):
        raise PhotoError("Describe the style in a few plain words, for example: short wavy bob.")
    return s.lower()


def reject_foreign_images(params: dict) -> None:
    """The never-a-face-from-another-picture rule, enforced where requests enter."""
    for k in params:
        if str(k).lower() in FORBIDDEN_KEYS:
            raise PhotoError("Veil never uses a face or picture from somewhere else. Remove that setting.")
    for v in params.values():
        if isinstance(v, str) and (v.startswith("data:") or v.lower().startswith(("http://", "https://", "file:"))):
            raise PhotoError("Veil never uses a face or picture from somewhere else.")


def faces_from(engine: Engine, img: Image.Image, params: dict, *, need_landmarks: bool) -> list[F.Face]:
    raw = params.get("faces")
    if not isinstance(raw, list) or not raw:
        raise PhotoError("Choose which faces to change first: use Find faces in the Faces panel and tick the ones you mean.")
    if len(raw) > MAX_FACES:
        raise PhotoError("Too many faces at once.")
    w, h = img.size
    out = []
    for b in raw:
        if not (isinstance(b, (list, tuple)) and len(b) == 4):
            raise PhotoError("Each face must be a box [x0, y0, x1, y1].")
        try:
            x0, y0, x1, y1 = (float(v) for v in b)
        except (TypeError, ValueError):
            raise PhotoError("Each face must be a box [x0, y0, x1, y1].") from None
        if not all(0 <= v <= 1 for v in (x0, y0, x1, y1)) or x1 - x0 < 0.004 or y1 - y0 < 0.004:
            raise PhotoError("A face box is outside the photo or too small.")
        box = (x0 * w, y0 * h, x1 * w, y1 * h)
        pts = None
        if engine.has("face-landmarker") or need_landmarks:
            pts = engine.landmarks(img, box)
            if pts is None and need_landmarks:
                raise PhotoError("Veil could not map one of those faces closely enough for this edit "
                                 "(it may be too small, turned away or covered). Untick it and try again.")
        out.append(F.Face(box, pts))
    return out


# ---------------------------------------------------------------- pixel helpers
def _lab(arr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(arr.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)


def _rgb(lab: np.ndarray) -> np.ndarray:
    return np.clip(cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_LAB2RGB) * 255.0, 0, 255)


def _hex(c: str) -> tuple[int, int, int]:
    return int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)


def _target_lab(c: str) -> np.ndarray:
    return _lab(np.array([[_hex(c)]], dtype=np.uint8))[0, 0]


def _window(img: Image.Image, face: F.Face, margin: float) -> tuple[tuple[int, int], np.ndarray, F.Face]:
    """A square-ish window around the face (so heavy maths runs on the face, not the whole photo)."""
    w, h = img.size
    m = margin * max(face.w, face.h)
    x0, y0 = max(0, int(face.box[0] - m)), max(0, int(face.box[1] - m))
    x1, y1 = min(w, int(face.box[2] + m)), min(h, int(face.box[3] + m))
    arr = np.asarray(img.crop((x0, y0, x1, y1)).convert("RGB")).copy()
    off = np.array([x0, y0], dtype=np.float64)
    pts = face.pts - off if face.pts is not None else None
    sf = F.Face((face.box[0] - x0, face.box[1] - y0, face.box[2] - x0, face.box[3] - y0), pts)
    return (x0, y0), arr, sf


def _paste(out: Image.Image, off: tuple[int, int], arr: np.ndarray, new: np.ndarray, alpha: np.ndarray) -> None:
    a = alpha[..., None].astype(np.float32)
    mixed = np.rint(arr.astype(np.float32) * (1 - a) + new.astype(np.float32) * a)
    res = np.where(a > 0, mixed, arr).clip(0, 255).astype(np.uint8)     # alpha 0 keeps the original byte for byte
    out.paste(Image.fromarray(res), off)


def _restore_alpha(src: Image.Image, out: Image.Image) -> Image.Image:
    if src.mode == "RGBA":
        out = out.convert("RGB"); out.putalpha(src.getchannel("A"))
    return out


# ---------------------------------------------------------------- hide a face
def _face_alpha(face: F.Face, shape, grow: float, edge: float) -> np.ndarray:
    return F.soft(F.face_area(face, shape, grow), edge * face.size())


def blur_face(arr, face, p):
    amt = _num(p, "amount", 0, 100, 80) / 100
    fw = face.size()
    r = fw * (0.07 + 0.25 * amt)
    k = int(r * 3) | 1
    blurred = cv2.GaussianBlur(arr, (k, k), r, borderType=cv2.BORDER_REFLECT)
    return blurred, _face_alpha(face, arr.shape[:2], 0.05, 0.035)


def pixelate_face(arr, face, p):
    amt = _num(p, "amount", 0, 100, 70) / 100
    blocks = max(3, round(24 - 21 * amt))                    # blocks across the face: 24 (fine) ... 3 (coarse)
    bs = max(2, int(face.size() / blocks))
    h, w = arr.shape[:2]
    small = cv2.resize(arr, (max(1, w // bs), max(1, h // bs)), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST), _face_alpha(face, arr.shape[:2], 0.04, 0.01)


# ---------------------------------------------------------------- skin, teeth, eyes
def skin_smooth(arr, face, p):
    amt = _num(p, "amount", 0, 100, 40) / 100
    fw = face.size()
    s = min(1.0, 480.0 / max(fw, 1))
    h, w = arr.shape[:2]
    small = cv2.resize(arr, (max(8, round(w * s)), max(8, round(h * s))), interpolation=cv2.INTER_AREA) if s < 1 else arr
    sig_s = max(2.0, 0.025 * fw * s * (0.6 + amt))
    smooth = cv2.bilateralFilter(small, 0, 14 + 40 * amt, sig_s)
    smooth = cv2.resize(smooth, (w, h), interpolation=cv2.INTER_CUBIC) if s < 1 else smooth
    g = max(1.0, 0.004 * fw)
    mid = cv2.GaussianBlur(arr, (int(g * 4) | 1, int(g * 4) | 1), g)
    detail = arr.astype(np.float32) - mid.astype(np.float32)
    keep = 0.55 - 0.40 * amt                                  # fine pores kept: more at low amounts
    new = np.clip(smooth.astype(np.float32) + detail * keep, 0, 255)
    return new, F.soft(F.skin_mask(face, arr.shape[:2]), 0.02 * fw) * (0.35 + 0.65 * amt)


def teeth_whiten(arr, face, p):
    amt = _num(p, "amount", 0, 100, 60) / 100
    shape = arr.shape[:2]
    mouth = F._shrunk(F.mouth_opening(face, shape), max(1, 0.006 * face.size()))
    lab = _lab(arr)
    teeth = (mouth > 0) & (lab[..., 0] > 38) & (lab[..., 1] < 18)        # bright and not red: not tongue, gums or shadow
    if teeth.sum() < 0.0006 * face.size() ** 2:
        raise PhotoError("No teeth are showing on that face.")
    t = F.soft(teeth.astype(np.uint8), 0.9)
    lab2 = lab.copy()
    lab2[..., 2] *= (1 - 0.8 * amt)                                       # less yellow
    lab2[..., 1] *= (1 - 0.3 * amt)
    lab2[..., 0] = np.clip(lab2[..., 0] + 10 * amt * (1 - lab2[..., 0] / 100), 0, 100)
    return _rgb(lab2), t


def eye_color(arr, face, p):
    amt = _num(p, "amount", 0, 100, 85) / 100
    target = _target_lab(_color(p, "color", EYE_COLORS, EYE_COLORS["blue"]))
    shape = arr.shape[:2]
    lab = _lab(arr)
    alpha = np.zeros(shape, np.float32)
    out = lab.copy()
    for disc, r in F.iris_discs(face, shape):
        if disc.sum() == 0:
            continue
        sel = disc > 0
        L = lab[..., 0]
        mean_l = float(L[sel].mean())
        tex = np.clip(0.55 + 0.45 * L / max(mean_l, 8.0), 0.5, 1.5)
        dark = np.clip((L - 10) / 25.0, 0, 1)                             # the pupil stays dark
        glint = 1 - np.clip((L - 82) / 14.0, 0, 1)                        # reflections stay white
        wgt = dark * glint
        new_l = np.clip(L + (target[0] - mean_l) * 0.5 * wgt, 0, 100)
        out[..., 0] = np.where(sel, new_l, out[..., 0])
        out[..., 1] = np.where(sel, lab[..., 1] + (target[1] * tex * 0.65 - lab[..., 1]) * wgt, out[..., 1])
        out[..., 2] = np.where(sel, lab[..., 2] + (target[2] * tex * 0.65 - lab[..., 2]) * wgt, out[..., 2])
        alpha = np.maximum(alpha, F.soft(disc, max(0.6, r * 0.06)))
    if alpha.max() == 0:
        raise PhotoError("The eyes are closed or hidden on that face.")
    return _rgb(out), alpha * amt


# ---------------------------------------------------------------- makeup (adding it is plain colour work)
def eyelash_length(arr, face, p):
    """Draws extra upper lashes along the eyelid: thin, tapered, curled up and out. Plain drawing, no model needed."""
    amt = _num(p, "amount", 0, 100, 60) / 100
    pts = F.need_pts(face, "find the eyelashes")
    h, w = arr.shape[:2]
    ss = 3
    big = np.zeros((h * ss, w * ss), np.uint8)
    fw = face.size()
    rng = np.random.default_rng(3)
    for ring, eye in ((F.UPPER_R, F.EYE_R), (F.UPPER_L, F.EYE_L)):
        lid = pts[ring]                                   # outer corner -> inner corner
        centre = pts[eye].mean(axis=0)
        ew = float(np.hypot(*(lid[0] - lid[-1])))
        seg = np.hypot(*np.diff(lid, axis=0).T)
        cum = np.concatenate([[0], np.cumsum(seg)]) / seg.sum()
        side = lid[0] - centre
        outward = np.array([np.sign(side[0]) or 1.0, 0.0])        # toward the outer corner of this eye
        n = max(16, int(ew / (0.0065 * fw)))
        for t in np.linspace(0.03, 0.80, n):
            base = np.array([np.interp(t, cum, lid[:, 0]), np.interp(t, cum, lid[:, 1])])
            i = min(len(lid) - 2, max(0, int(np.searchsorted(cum, t) - 1)))
            tan = lid[i + 1] - lid[i]; tan = tan / (np.hypot(*tan) or 1)
            nrm = np.array([-tan[1], tan[0]])
            if np.dot(nrm, base - centre) < 0:
                nrm = -nrm                                       # point away from the eye, up and out
            taper = 0.55 + 0.45 * np.sin(np.pi * min(1.0, (1 - t) ** 0.8 + 0.08))
            length = ew * (0.06 + 0.20 * amt) * taper * (0.85 + 0.3 * rng.random())
            tip = base + nrm * length + outward * length * 0.40
            ctrl = base + nrm * length * 0.75
            u = np.linspace(0, 1, 9)[:, None]
            curve = (1 - u) ** 2 * base + 2 * (1 - u) * u * ctrl + u ** 2 * tip
            for k in range(len(curve) - 1):
                th = max(1, round(ss * max(0.7, 0.0042 * fw * (1 - 0.7 * k / 8))))
                a = np.round(curve[k] * ss).astype(np.int32); b = np.round(curve[k + 1] * ss).astype(np.int32)
                cv2.line(big, tuple(int(v) for v in a), tuple(int(v) for v in b), 255, th, cv2.LINE_AA)
    a = cv2.resize(big, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    a = cv2.GaussianBlur(a, (3, 3), 0.6)
    dark = np.empty_like(arr); dark[:] = (30, 22, 22)
    return dark, np.clip(a * (0.5 + 0.35 * amt), 0, 1)


def _tint(lab, target, w, light=0.0):
    out = lab.copy()
    out[..., 1] = lab[..., 1] + (target[1] - lab[..., 1]) * w
    out[..., 2] = lab[..., 2] + (target[2] - lab[..., 2]) * w
    out[..., 0] = lab[..., 0] + (target[0] - lab[..., 0]) * w * light
    return out


def makeup_add(arr, face, p):
    amt = _num(p, "amount", 0, 100, 70) / 100
    part = str(p.get("part") or "lips").lower()
    if part not in MAKEUP_PARTS:
        raise PhotoError("part must be one of: " + ", ".join(MAKEUP_PARTS) + ".")
    shape = arr.shape[:2]
    fw = face.size()
    lab = _lab(arr)
    out = lab.copy()
    alpha = np.zeros(shape, np.float32)
    parts = ("lips", "blush", "eyeshadow", "eyeliner") if part == "face" else (part,)
    for pt in parts:
        color = _color(p, "color", LIP_COLORS if pt == "lips" else {}, MAKEUP_DEFAULT[pt]) if part != "face" else MAKEUP_DEFAULT[pt]
        tgt = _target_lab(color)
        if pt == "lips":
            a = F.soft(F.lips_mask(face, shape), 0.008 * fw) * 0.88 * amt
            out = np.where(a[..., None] > 0, _tint(out, tgt, 1.0, 0.35), out); alpha = np.maximum(alpha, a)
        elif pt == "blush":
            a = np.zeros(shape, np.float32)
            yy, xx = np.mgrid[:shape[0], :shape[1]].astype(np.float32)
            for cx, cy, r in F.cheek_centres(face):
                a = np.maximum(a, np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * (0.55 * r) ** 2)))
            a = a * F.soft(F.skin_mask(face, shape), 0.02 * fw) * 0.42 * amt
            out = np.where(a[..., None] > 0, _tint(out, tgt, 1.0), out); alpha = np.maximum(alpha, a)
        elif pt == "eyeshadow":
            a = np.zeros(shape, np.float32)
            for band in F.lid_band(face, shape, 0.55):
                a = np.maximum(a, F.soft(band, 0.012 * fw))
            a = a * 0.6 * amt
            out = np.where(a[..., None] > 0, _tint(out, tgt, 1.0, 0.45), out); alpha = np.maximum(alpha, a)
        elif pt == "eyeliner":
            m = np.zeros(shape, np.uint8)
            pts = face.pts
            for ring in (F.UPPER_R, F.UPPER_L):
                line = np.round(pts[ring]).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(m, [line], False, 1, max(1, round(0.011 * fw * (0.6 + amt))), cv2.LINE_AA)
            a = F.soft(m, 0.5) * 0.9 * amt
            out = np.where(a[..., None] > 0, _tint(out, tgt, 1.0, 0.9), out); alpha = np.maximum(alpha, a)
    return _rgb(out), np.clip(alpha, 0, 1)


# ---------------------------------------------------------------- hair
def hair_area(engine: Engine, img: Image.Image, face: F.Face, user_mask: Image.Image | None) -> np.ndarray:
    """0/1 mask of the hair. Uses the owner's own selection if there is one; otherwise asks SAM with a box around
    the head, a point on the hair above the forehead and points on the face to keep out."""
    shape = (img.height, img.width)
    if user_mask is not None:
        m = binarize(user_mask, img.size)
        if m.sum() == 0:
            raise PhotoError("The selection is empty.")
        return m
    x0, y0, x1, y1 = face.box
    fw, fh = x1 - x0, y1 - y0
    hb = F.head_box(face, shape)
    cx = (x0 + x1) / 2
    W, H = img.size
    pos = [(cx, y0 - 0.22 * fh), (x0 + 0.20 * fw, y0 - 0.10 * fh), (x1 - 0.20 * fw, y0 - 0.10 * fh)]
    neg = [(cx, y0 + 0.55 * fh), (cx, y1 - 0.08 * fh), (x0 + 0.20 * fw, y0 + 0.65 * fh), (x1 - 0.20 * fw, y0 + 0.65 * fh)]
    pts = [(min(max(x / W, 0), 1), min(max(y / H, 0), 1), 1) for x, y in pos] + [(x / W, y / H, 0) for x, y in neg]
    box = (hb[0] / W, hb[1] / H, hb[2] / W, hb[3] / H)
    sam = binarize(engine.segment(f"hair:{time.monotonic_ns()}", img, pts, box), img.size)
    box_m = np.zeros(shape, np.uint8); box_m[hb[1]:hb[3], hb[0]:hb[2]] = 1
    person = binarize(engine.matte(img), img.size)                  # the backdrop is never hair
    m = sam & box_m & person
    if face.pts is not None:
        m = m & (1 - F._shrunk(F.face_area(face, shape), 0.01 * face.size()))
        m = _drop_skin(img, face, m, hb)
    m = cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    seeds = [(int(x), int(y)) for x, y in pos if 0 <= x < W and 0 <= y < H]
    m = _component_with(m, seeds, 0.0015 * (hb[2] - hb[0]) * (hb[3] - hb[1]))
    if m.sum() < 0.02 * fw * fh:
        raise PhotoError("Veil could not find the hair. Select it with the Point tool and run this again.")
    return m


def _drop_skin(img: Image.Image, face: F.Face, m: np.ndarray, hb) -> np.ndarray:
    """Ears, neck and forehead are skin, not hair: take out bright pixels whose colour matches this face's skin.
    (Dark pixels are never dropped, so dark brown hair stays.)"""
    shape = m.shape
    lab = _lab(np.asarray(img.convert("RGB").crop((hb[0], hb[1], hb[2], hb[3]))))
    sk = F.skin_mask(face, shape)[hb[1]:hb[3], hb[0]:hb[2]] > 0
    if sk.sum() < 200:
        return m
    samp = lab[sk]
    if len(samp) > 20000:
        samp = samp[np.random.default_rng(0).choice(len(samp), 20000, replace=False)]
    mu = samp.mean(axis=0)
    inv = np.linalg.inv(np.cov(samp.T) + np.eye(3) * 2.0)
    d = lab - mu
    d2 = np.einsum("...i,ij,...j->...", d, inv, d)
    colour = (d2 < 1.8 ** 2) & (lab[..., 0] > 0.8 * mu[0])       # looks exactly like this face's skin
    # Light strands can match the skin colour, and skin only ever sits close to the face. So: right at the face
    # (forehead, ears) the colour rule is strict; a little further out (ears, jaw, neck) it only applies below the
    # eyes; further still it only applies to smooth pixels, because skin is smooth and hair is full of fine lines.
    tex = _texture(lab[..., 0])
    skin_tex = float(np.median(tex[sk]))
    smooth = tex < max(2.2 * skin_tex, skin_tex + 1.5)
    fa = F.face_area(face, shape)
    fs = face.size()

    def zone(g):
        return F._grown(fa, g * fs)[hb[1]:hb[3], hb[0]:hb[2]] > 0
    eye_y = float(face.pts[[33, 263]][:, 1].mean())
    below = np.arange(hb[1], hb[3])[:, None] >= eye_y - 0.10 * face.height()
    colour &= zone(0.18) | (zone(0.30) & smooth)
    loose_ok = zone(0.06) | (zone(0.18) & below) | (zone(0.30) & below & smooth)
    # next to the face (ear, hairline, neck) anything much lighter than the typical hair is skin too
    ring = F._grown(F.face_area(face, shape), 0.28 * face.size())[hb[1]:hb[3], hb[0]:hb[2]] > 0
    far = (m[hb[1]:hb[3], hb[0]:hb[2]] > 0) & ~ring
    ref = float(np.median(lab[..., 0][far])) if far.sum() > 500 else None
    skinlike = colour
    if ref is not None:
        skinlike = skinlike | (ring & (lab[..., 0] > ref + 22) & (lab[..., 0] > 0.7 * mu[0]) & loose_ok)
    out = m.copy()
    sub = out[hb[1]:hb[3], hb[0]:hb[2]]
    sub[skinlike] = 0
    return out


def _texture(L: np.ndarray) -> np.ndarray:
    """How much fine detail there is around each pixel (high on hair strands, low on smooth skin)."""
    L = L.astype(np.float32)
    g = np.hypot(cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3))
    return cv2.blur(g, (7, 7))


def _component_with(m: np.ndarray, seeds: list[tuple[int, int]], min_area: float) -> np.ndarray:
    """Keep the connected piece(s) that touch the seed points (the hair above the forehead), and any other big piece."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    wanted = {int(lab[y, x]) for x, y in seeds if lab[y, x] > 0}
    # a seed may sit just off the mask: look at a small neighbourhood
    for x, y in seeds:
        patch = lab[max(0, y - 6):y + 7, max(0, x - 6):x + 7]
        wanted |= {int(v) for v in np.unique(patch) if v > 0}
    keep = np.zeros_like(m)
    for i in range(1, n):
        if i in wanted and stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[lab == i] = 1
    return keep


def _recolor_hair(arr, hair, target_rgb):
    lab = _lab(arr)
    sel = hair > 0
    L = lab[..., 0]
    tgt = _target_lab(target_rgb)
    mean_l = float(L[sel].mean()) if sel.any() else 50.0
    lo, hi = (float(np.percentile(L[sel], 2)), float(np.percentile(L[sel], 98))) if sel.any() else (0.0, 100.0)
    span = max(hi - lo, 8.0)
    # keep every strand's light and shade (its detail) but move the average toward the new colour
    # going lighter loses depth (shadows between strands flatten), so give the strands a bit more contrast then
    gain = 0.9 + min(0.45, max(0.0, (tgt[0] - mean_l) / 100))
    new_l = np.clip((L - mean_l) * gain + (mean_l + (tgt[0] - mean_l) * 0.85), 0, 100)
    wchroma = np.clip(0.55 + 0.45 * (L - lo) / span, 0.35, 1.2)
    out = lab.copy()
    out[..., 0] = new_l
    out[..., 1] = tgt[1] * wchroma * 0.85
    out[..., 2] = tgt[2] * wchroma * 0.85
    return _rgb(out)


def _smooth_region(img: Image.Image, m: np.ndarray, sigma_color: float = 60.0) -> Image.Image:
    """Wipes fine texture (stubble) inside the mask so the generative pass starts from clean skin."""
    ys, xs = np.nonzero(m)
    x0, x1, y0, y1 = max(0, xs.min() - 8), min(img.width, xs.max() + 9), max(0, ys.min() - 8), min(img.height, ys.max() + 9)
    arr = np.asarray(img.crop((x0, y0, x1, y1)).convert("RGB")).copy()
    f = min(1.0, 400.0 / max(arr.shape[:2]))
    small = cv2.resize(arr, (max(8, int(arr.shape[1] * f)), max(8, int(arr.shape[0] * f))), interpolation=cv2.INTER_AREA)
    for _ in range(2):
        small = cv2.bilateralFilter(small, 0, sigma_color, 9)
    sm = cv2.resize(small, (arr.shape[1], arr.shape[0]), interpolation=cv2.INTER_CUBIC)
    out = img.convert("RGB").copy()
    _paste(out, (x0, y0), arr, sm, F.soft(m[y0:y1, x0:x1], 3.0))
    return out


# ---------------------------------------------------------------- the dispatcher
def run(engine: Engine, op: str, img: Image.Image, params: dict, mask: Image.Image | None) -> Image.Image:
    if engine is None:
        raise PhotoError("The editing models are not set up.")
    reject_foreign_images(params)
    try:
        return _run(engine, op, img, params, mask)
    except ModelError as e:
        raise PhotoError(str(e)) from None


SIMPLE = {"blur_face": blur_face, "pixelate_face": pixelate_face, "skin_smooth": skin_smooth, "teeth_whiten": teeth_whiten,
          "eye_color": eye_color, "makeup_add": makeup_add, "eyelash_length": eyelash_length}
NEEDS_MAPPED = {"skin_smooth", "teeth_whiten", "eye_color", "makeup_add", "eyelash_length"}
PROMPT_FACE = ("photo of the face of an adult person, natural skin with pores, realistic eyes, sharp focus, same lighting and head angle as the "
               "surroundings, photograph")
NEG_FACE = "child, minor, teenager, cartoon, painting, deformed, extra eyes, blurry, text, watermark, mask, makeup"


def _run(engine: Engine, op: str, img: Image.Image, params: dict, mask: Image.Image | None) -> Image.Image:
    rgb = img.convert("RGB")
    if op == "remove_jewelry":
        if mask is None or binarize(mask, rgb.size).sum() == 0:
            raise PhotoError("Select the jewelry first (Point tool: click the earring, chain or ring).")
        grow = int(_num(params, "grow", 0, 200, 0)) or max(6, round(0.006 * max(rgb.size)))
        if str(params.get("engine", "lama")) == "sdxl":
            from .ops import TATTOO_PROMPT
            out = engine.generative_fill(rgb, mask, "bare skin with natural skin texture, seamless, photograph", TATTOO_PROMPT[1] + ", jewelry, metal", grow)
        else:
            out = engine.inpaint(rgb, mask, grow)
        return _restore_alpha(img, out)

    if op in SIMPLE:
        faces = faces_from(engine, rgb, params, need_landmarks=op in NEEDS_MAPPED)
        out = rgb.copy()
        for face in faces:
            margin = 0.5 if op in ("blur_face", "pixelate_face") else 0.35
            off, arr, sf = _window(rgb, face, margin)
            new, alpha = SIMPLE[op](arr, sf, params)
            _paste(out, off, arr, new, alpha)
        return _restore_alpha(img, out)

    if op == "hair_color":
        color = _color(params, "color", HAIR_COLORS, "")
        if not color:
            raise PhotoError("Which colour? For example: blonde, auburn, or #7a2f1c.")
        faces = faces_from(engine, rgb, params, need_landmarks=False)
        out = rgb.copy()
        for face in faces:
            hair = hair_area(engine, rgb, face, mask)
            ys, xs = np.nonzero(hair)
            x0, x1, y0, y1 = max(0, xs.min() - 4), min(rgb.width, xs.max() + 5), max(0, ys.min() - 4), min(rgb.height, ys.max() + 5)
            arr = np.asarray(out.crop((x0, y0, x1, y1))).copy()
            hm = hair[y0:y1, x0:x1]
            new = _recolor_hair(arr, hm, color)
            _paste(out, (x0, y0), arr, new, F.soft(hm, 1.2) * _num(params, "amount", 0, 100, 100) / 100)
            if mask is not None:
                break                                     # a hand-made selection is one area, not one per face
        return _restore_alpha(img, out)

    # ---- the generative edits (they need the optional Stable Diffusion XL download)
    if not engine.sdxl_ready():
        raise PhotoError("This edit needs the optional Stable Diffusion XL Inpainting download. Open Model, then Editing models.")
    shape = (rgb.height, rgb.width)
    out = rgb
    if op == "generate_face":
        faces = faces_from(engine, rgb, params, need_landmarks=False)
        seed0 = params.get("seed")
        seed0 = int(_num(params, "seed", 0, 2**31 - 1, 0)) if seed0 is not None else int(time.time_ns() % (2**31 - 1))
        for i, face in enumerate(faces):
            m = F.face_area(face, shape, grow=0.07)
            out = engine.generative_fill(out, Image.fromarray(m * 255, "L"), PROMPT_FACE, NEG_FACE, grow=0,
                                         strength=0.99, seed=seed0 + i, context=0.9, feather_px=max(2.0, 0.02 * face.size()))
        return _restore_alpha(img, out)

    if op == "makeup_remove":
        part = str(params.get("part") or "all").lower()
        if part not in ("lips", "eyes", "cheeks", "all"):
            raise PhotoError("part must be lips, eyes, cheeks or all.")
        faces = faces_from(engine, rgb, params, need_landmarks=True)
        for i, face in enumerate(faces):
            fw = face.size()
            m = np.zeros(shape, np.uint8)
            if part in ("lips", "all"):
                m |= F._grown(F.lips_mask(face, shape), 0.012 * fw)
            if part in ("eyes", "all"):
                for band in F.lid_band(face, shape, 0.6):
                    m |= band
            if part in ("cheeks", "all"):
                for cx, cy, r in F.cheek_centres(face):
                    cv2.circle(m, (round(cx), round(cy)), round(r), 1, -1)
            m &= F._grown(F.face_area(face, shape), 0.03)
            out = engine.generative_fill(out, Image.fromarray(m * 255, "L"),
                                         "natural bare face without any makeup, natural lips, natural skin, photograph",
                                         "lipstick, lip gloss, eyeshadow, eyeliner, mascara, blush, makeup, cosmetics, deformed",
                                         grow=0, strength=0.55, seed=11 + i, context=1.1, feather_px=max(2.0, 0.012 * fw))
        return _restore_alpha(img, out)

    if op == "facial_hair":
        mode = str(params.get("mode") or "add").lower()
        if mode not in ("add", "remove"):
            raise PhotoError("mode must be add or remove.")
        style = str(params.get("style") or "stubble").lower().strip()
        if style not in FACIAL_HAIR_STYLES:
            raise PhotoError("style must be one of: " + ", ".join(FACIAL_HAIR_STYLES) + ".")
        faces = faces_from(engine, rgb, params, need_landmarks=True)
        for i, face in enumerate(faces):
            m = F.beard_region(face, shape, "beard" if mode == "remove" else style)
            if mode == "add":
                prompt = {"stubble": "light stubble facial hair, five o'clock shadow", "beard": "a full well-groomed beard",
                          "goatee": "a goatee", "moustache": "a moustache", "mustache": "a moustache"}[style] + ", realistic hair texture, photograph"
                neg, strength = "smooth skin, cartoon, deformed, wig, blurry", 0.92
            else:
                out = _smooth_region(out, m)
                prompt, neg, strength = "clean shaven smooth bare skin, natural skin texture, photograph", "beard, stubble, moustache, facial hair, five o'clock shadow, deformed", 0.7
            out = engine.generative_fill(out, Image.fromarray(m * 255, "L"), prompt, neg, grow=0, strength=strength, seed=21 + i,
                                         context=0.9, feather_px=max(2.0, 0.012 * face.size()))
        return _restore_alpha(img, out)

    if op == "hairstyle":
        style = _style(params)
        color = str(params.get("color") or "").strip().lower()
        if color:
            _color(params, "color", HAIR_COLORS, "")             # checks it is a colour we know
        faces = faces_from(engine, rgb, params, need_landmarks=False)
        if mask is not None and len(faces) > 1:
            raise PhotoError("With a selection, choose just one face.")
        for i, face in enumerate(faces):
            if mask is not None:
                area = binarize(mask, rgb.size)           # paint where the new hair should be
            else:
                hair = hair_area(engine, rgb, face, None)
                area = F._grown(hair, 0.10 * face.size())
                hb = F.head_box(face, shape)
                top = np.zeros(shape, np.uint8)
                top[hb[1]:int(face.box[1] + 0.15 * face.h), hb[0]:hb[2]] = 1
                area = (area | top)
            if face.pts is not None:
                area &= 1 - F._shrunk(F.face_area(face, shape), 0.015 * face.size())
            else:
                area &= 1 - F.face_area(face, shape)
            short = any(w in style for w in ("short", "bob", "pixie", "crop", "buzz", "shaved"))
            long_ = any(w in style for w in ("long", "waist", "extra"))
            neg = ("wig, cartoon, deformed, blurry, hat, extra face" + (", short hair" if long_ else "")
                   + (", long hair, hair on the shoulders" if short else "") + ("" if "bald" in style else ", bald"))
            prompt = f"{color + ' ' if color else ''}{style} hairstyle, natural realistic hair, detailed strands, photograph"
            out = engine.generative_fill(out, Image.fromarray(area * 255, "L"), prompt,
                                         neg, grow=0, strength=0.99, seed=31 + i,
                                         context=0.6, feather_px=max(2.0, 0.015 * face.size()))
        return _restore_alpha(img, out)

    raise PhotoError("Unknown edit.")


def describe(op: str, params: dict) -> str:
    n = len(params.get("faces") or [])
    plural = f" ({n} faces)" if n > 1 else ""
    c = str(params.get("color") or "").strip()[:20]
    return {
        "blur_face": "Blur face" + plural, "pixelate_face": "Pixelate face" + plural,
        "generate_face": "Replace face with a generated one" + plural, "skin_smooth": "Smooth skin" + plural,
        "teeth_whiten": "Whiten teeth" + plural, "eye_color": f"Eye colour {c}".strip() + plural,
        "eyelash_length": "Longer eyelashes" + plural, "makeup_add": f"Add makeup ({params.get('part', 'lips')})" + plural,
        "makeup_remove": "Remove makeup" + plural, "hair_color": f"Hair colour {c}".strip(),
        "hairstyle": f"Hairstyle: {str(params.get('style', ''))[:30]}", "facial_hair": f"Facial hair: {params.get('mode', 'add')} {params.get('style', '')}".strip(),
        "remove_jewelry": "Remove jewelry",
    }[op]
