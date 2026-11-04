"""Where things are on a face. A Face is the box YuNet found plus, when MediaPipe could map it, 478 landmark points.
Everything here is plain geometry (no model calls) so it is cheap to test: the outline, eyes, irises, lips and so on
become masks, all in pixel coordinates of the photo. Nothing here ever reads another picture."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# MediaPipe face-mesh point numbers
OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
        172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
LIPS_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
EYE_R = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]       # the person's right eye (left in the picture)
EYE_L = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
UPPER_R = [33, 246, 161, 160, 159, 158, 157, 173, 133]       # upper eyelid edge, outer corner to inner corner
UPPER_L = [263, 466, 388, 387, 386, 385, 384, 398, 362]
BROW_R = [46, 53, 52, 65, 55, 70, 63, 105, 66, 107]
BROW_L = [276, 283, 282, 295, 285, 300, 293, 334, 296, 336]
IRIS_R = [468, 469, 470, 471, 472]       # centre first, then four points on the edge
IRIS_L = [473, 474, 475, 476, 477]
CHEEK_R, CHEEK_L = 205, 425
NOSE_TIP, NOSE_BASE, CHIN = 1, 2, 152
JAW_LOWER = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454]


@dataclass
class Face:
    box: tuple[float, float, float, float]            # x0, y0, x1, y1 in pixels
    pts: np.ndarray | None = None                     # (478, 2) in pixels, or None if the landmark model could not map this face

    @property
    def w(self) -> float:
        return self.box[2] - self.box[0]

    @property
    def h(self) -> float:
        return self.box[3] - self.box[1]

    @property
    def centre(self) -> tuple[float, float]:
        return ((self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2)

    def size(self) -> float:
        """The face width: ear to ear when mapped, otherwise the box width."""
        if self.pts is not None:
            return float(np.hypot(*(self.pts[454] - self.pts[234])))
        return self.w

    def height(self) -> float:
        if self.pts is not None:
            return float(np.hypot(*(self.pts[CHIN] - self.pts[10])))
        return self.h


def _poly(shape: tuple[int, int], pts: np.ndarray) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 1)
    return m


def _grown(m: np.ndarray, px: float) -> np.ndarray:
    px = int(round(px))
    if px <= 0:
        return m
    return cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1)))


def _shrunk(m: np.ndarray, px: float) -> np.ndarray:
    px = int(round(px))
    if px <= 0:
        return m
    return cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1)))


def soft(m: np.ndarray, sigma: float) -> np.ndarray:
    """A 0/1 mask as a soft 0..1 float edge."""
    f = m.astype(np.float32)
    if sigma <= 0.3:
        return f
    k = int(sigma * 4) | 1
    return np.clip(cv2.GaussianBlur(f, (k, k), sigma), 0, 1)


# ---------------------------------------------------------------- masks (full image shape)
def face_area(face: Face, shape: tuple[int, int], grow: float = 0.0) -> np.ndarray:
    """The whole face (forehead to chin, ear to ear) as a 0/1 mask. `grow` is a fraction of the face width."""
    if face.pts is not None:
        pts = face.pts[OVAL].copy()
        c = pts.mean(axis=0)
        fh = face.height()
        pts = c + (pts - c) * np.array([1.05, 1.06])
        pts[pts[:, 1] < c[1] - 0.3 * fh, 1] -= 0.10 * fh     # take in the forehead up to the hairline
        m = _poly(shape, pts)
    else:
        cx, cy = face.centre
        m = np.zeros(shape, np.uint8)
        cv2.ellipse(m, (round(cx), round(cy - 0.02 * face.h)), (round(0.60 * face.w), round(0.62 * face.h)), 0, 0, 360, 1, -1)
    return _grown(m, grow * face.size())


def need_pts(face: Face, what: str) -> np.ndarray:
    from ..history import PhotoError
    if face.pts is None:
        raise PhotoError(f"Veil could not map this face closely enough to {what}. Try a sharper photo or a face that is turned less.")
    return face.pts


def eye_polys(face: Face, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    p = need_pts(face, "find the eyes")
    return _poly(shape, p[EYE_R]), _poly(shape, p[EYE_L])


def iris_discs(face: Face, shape: tuple[int, int]) -> list[tuple[np.ndarray, float]]:
    """For each eye: the iris as a 0/1 disc clipped to the opening of the eye, and the iris radius in pixels."""
    p = need_pts(face, "find the irises")
    eyes = eye_polys(face, shape)
    out = []
    for ring, eye in zip((IRIS_R, IRIS_L), eyes):
        c = p[ring[0]]
        r = float(np.mean([np.hypot(*(p[i] - c)) for i in ring[1:]]))
        disc = np.zeros(shape, np.uint8)
        cv2.circle(disc, (round(float(c[0])), round(float(c[1]))), max(2, round(r * 1.04)), 1, -1)
        out.append((disc & _shrunk(eye, 1), r))
    return out


def lips_mask(face: Face, shape: tuple[int, int]) -> np.ndarray:
    """The lips themselves: outer outline minus the opening of the mouth."""
    p = need_pts(face, "find the lips")
    return _poly(shape, p[LIPS_OUTER]) & (1 - _poly(shape, p[LIPS_INNER]))


def mouth_opening(face: Face, shape: tuple[int, int]) -> np.ndarray:
    p = need_pts(face, "find the mouth")
    return _poly(shape, p[LIPS_INNER])


def skin_mask(face: Face, shape: tuple[int, int]) -> np.ndarray:
    """Face skin only: the face outline without eyes, brows, lips or the inside of the mouth."""
    p = need_pts(face, "find the skin")
    fw = face.size()
    m = face_area(face, shape) & (1 - _poly(shape, p[LIPS_OUTER]))
    for ring in (EYE_R, EYE_L):
        m &= 1 - _grown(_poly(shape, p[ring]), 0.035 * fw)
    for ring in (BROW_R, BROW_L):
        m &= 1 - _grown(_poly(shape, p[ring]), 0.02 * fw)
    return _shrunk(m, 0.012 * fw)


def cheek_centres(face: Face) -> list[tuple[float, float, float]]:
    p = need_pts(face, "find the cheeks")
    fw = face.size()
    return [(float(p[i][0]), float(p[i][1]), 0.17 * fw) for i in (CHEEK_R, CHEEK_L)]


def lid_band(face: Face, shape: tuple[int, int], up: float) -> list[np.ndarray]:
    """For each eye, the strip just above the upper eyelid (eyeshadow area), `up` x the eye width tall."""
    p = need_pts(face, "find the eyelids")
    out = []
    for ring_up, eye in ((UPPER_R, EYE_R), (UPPER_L, EYE_L)):
        lid = p[ring_up]
        ew = float(np.hypot(*(lid[0] - lid[-1])))
        poly = np.vstack([lid, (lid + np.array([0, -up * ew]))[::-1]])
        m = _poly(shape, poly) & (1 - _poly(shape, p[eye]))
        out.append(_grown(m, 0.04 * ew))
    return out


def lash_band(face: Face, shape: tuple[int, int]) -> list[np.ndarray]:
    """For each eye, a thin band around the upper lash line where eyelashes grow."""
    p = need_pts(face, "find the eyelashes")
    out = []
    for ring_up, eye in ((UPPER_R, EYE_R), (UPPER_L, EYE_L)):
        lid = p[ring_up]
        ew = float(np.hypot(*(lid[0] - lid[-1])))
        poly = np.vstack([lid + np.array([0, 0.05 * ew]), (lid + np.array([0, -0.30 * ew]))[::-1]])
        out.append(_grown(_poly(shape, poly), 0.02 * ew))
    return out


def beard_region(face: Face, shape: tuple[int, int], style: str) -> np.ndarray:
    """Where facial hair grows: the lower face (jaw, chin, cheeks) and the upper lip, never the lips or nostrils."""
    p = need_pts(face, "find the jaw")
    fw, fh = face.size(), face.height()
    mouth_y = float(p[13][1])
    nose_y = float(p[NOSE_BASE][1])
    m = np.zeros(shape, np.uint8)
    lip_top = float(p[0][1])
    if style in ("moustache", "mustache"):
        corners = (p[61], p[291])
        poly = np.array([[corners[0][0] - 0.05 * fw, nose_y + 0.08 * fh], [corners[1][0] + 0.05 * fw, nose_y + 0.08 * fh],
                         [corners[1][0] + 0.03 * fw, lip_top - 0.01 * fh], [corners[0][0] - 0.03 * fw, lip_top - 0.01 * fh]])
        m = _poly(shape, poly)
    elif style == "goatee":
        c = p[CHIN]
        cv2.ellipse(m, (round(float(c[0])), round(float(c[1] - 0.10 * fh))), (round(0.17 * fw), round(0.14 * fh)), 0, 0, 360, 1, -1)
        m |= _moustache_band(p, shape, nose_y, lip_top, fw, fh)
    else:                                           # stubble, beard: the whole lower face
        jaw = p[JAW_LOWER]
        poly = np.vstack([jaw, [jaw[-1][0] - 0.02 * fw, mouth_y - 0.04 * fh], [jaw[0][0] + 0.02 * fw, mouth_y - 0.04 * fh]])
        m = _poly(shape, poly) | _moustache_band(p, shape, nose_y, lip_top, fw, fh)
    m &= 1 - _grown(_poly(shape, p[LIPS_OUTER]), 0.012 * fw)
    return m & _grown(face_area(face, shape), 0.04)


def _moustache_band(p: np.ndarray, shape: tuple[int, int], nose_y: float, lip_top: float, fw: float, fh: float) -> np.ndarray:
    a, b = p[61], p[291]
    return _poly(shape, np.array([[a[0] - 0.03 * fw, nose_y + 0.08 * fh], [b[0] + 0.03 * fw, nose_y + 0.08 * fh],
                                  [b[0] + 0.02 * fw, lip_top], [a[0] - 0.02 * fw, lip_top]]))


def head_box(face: Face, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    """A box around the head including hair: wider and taller than the face."""
    x0, y0, x1, y1 = face.box
    w, h = x1 - x0, y1 - y0
    return (max(0, int(x0 - 0.7 * w)), max(0, int(y0 - 0.9 * h)), min(shape[1], int(x1 + 0.7 * w)), min(shape[0], int(y1 + 1.0 * h)))
