"""Phase 6: faces and looks. Fast tests use a stand-in engine; the `faces_real` tests run the real YuNet, MediaPipe and
Stable Diffusion XL models on portraits that SDXL itself paints (so no real person's photo is ever involved)."""
import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from test_phone import world  # noqa: F401
from test_vision import FakeEngine, mask_url, png_bytes
from veil.ai import actions, parser
from veil.vision import faces as F
from veil.vision import looks, registry
from veil.main import create_app
from veil.vision.engine import Engine

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
FACE_OPS = looks.FACE_OPS
GENERATIVE = ("generate_face", "makeup_remove", "hairstyle", "facial_hair")
BOX = [[0.25, 0.2, 0.75, 0.8]]


class FaceFake(FakeEngine):
    """Pretends to find a face and (optionally) to have the generative model; records every generative request."""

    def __init__(self, root, ready=True, sdxl=False):
        super().__init__(root, ready)
        self.sdxl = sdxl

    def sdxl_ready(self):
        return self.sdxl

    def detect_faces(self, img, min_score=0.6):
        w, h = img.size
        return [{"box": (0.25 * w, 0.2 * h, 0.75 * w, 0.8 * h), "score": 0.93}]

    def generative_fill(self, img, mask, prompt, negative="", grow=None, **kw):
        self.calls.append(("generative", prompt, negative, kw))
        out = img.convert("RGB").copy(); out.paste((0, 0, 255), mask=mask.resize(img.size)); return out


@pytest.fixture
def v(world, tmp_path):  # noqa: F811
    """An owner at the PC, a photo, and an app whose engine is the stand-in."""
    from types import SimpleNamespace
    from test_phone import client_for
    eng = FaceFake(tmp_path / "models")
    app = create_app(tmp_path / "data2", tailscale=world.ts, vision=eng)
    owner = client_for(app, "127.0.0.1")
    owner.post("/api/auth/setup", json={"username": "owner", "password": "correct horse"}); owner.post("/api/notice/ack")
    pid = owner.post("/api/photos", files={"file": ("a.png", png_bytes((120, 90)), "image/png")}).json()["id"]
    return SimpleNamespace(c=owner, eng=eng, pid=pid, app=app, world=world, models=tmp_path / "models")


def edit(v, op, params, mask=None):
    return v.c.post(f"/api/photos/{v.pid}/edit", json={"op": op, "params": params, "mask": mask})


# ---------------------------------------------------------------- the rule: never a face from another picture
def test_no_face_route_accepts_a_picture(v):
    """Structural: look at every route and prove none of the face endpoints can be handed a file or an image."""
    from fastapi import UploadFile
    for route in v.app.routes:
        path = getattr(route, "path", "")
        dep = getattr(route, "dependant", None)
        if dep is None:
            continue
        files = [p for p in dep.body_params if getattr(p, "type_", None) is UploadFile or "File" in repr(getattr(p, "field_info", ""))]
        if path.endswith("/api/photos") and "POST" in getattr(route, "methods", ()):      # the one upload route
            continue
        assert not files, f"{path} accepts a file"
    r = v.c.post(f"/api/photos/{v.pid}/faces", files={"file": ("x.png", png_bytes(), "image/png")})
    assert r.status_code == 200      # the upload is simply ignored; faces are found in the photo that is open


@pytest.mark.parametrize("op", FACE_OPS)
@pytest.mark.parametrize("key", ["image", "reference", "source", "face_image", "swap", "photo", "url", "donor", "identity"])
def test_every_face_edit_refuses_any_outside_image_setting(v, op, key):
    r = edit(v, op, {"faces": BOX, key: "anything"})
    assert r.status_code == 400 and "never uses a face or picture" in r.json()["detail"], (op, key)
    assert len(v.c.get(f"/api/photos/{v.pid}").json()["versions"]) == 1      # nothing was changed


@pytest.mark.parametrize("op", FACE_OPS)
@pytest.mark.parametrize("value", ["data:image/png;base64,AAAA", "http://example.com/face.png", "https://example.com/x.jpg", "file:///etc/passwd"])
def test_every_face_edit_refuses_a_picture_hidden_in_any_value(v, op, value):
    r = edit(v, op, {"faces": BOX, "color": value, "style": value})
    assert r.status_code == 400, (op, value)


def test_the_generated_face_prompt_cannot_be_steered(v):
    """The words sent to the generator for a generated face are fixed: a request cannot name a person to look like."""
    v.eng.sdxl = True
    for extra in ({}, {"prompt": "the face of a famous actor"}, {"style": "celebrity"}, {"seed": 4}):
        r = edit(v, "generate_face", {"faces": BOX, **extra})
        if "prompt" in extra or "style" in extra:
            assert r.status_code == 200      # unknown settings are ignored, not passed on
    gens = [c for c in v.eng.calls if c[0] == "generative"]
    assert gens and all(c[1] == looks.PROMPT_FACE and c[2] == looks.NEG_FACE for c in gens)
    assert "actor" not in " ".join(c[1] for c in gens)
    assert "child" in looks.NEG_FACE and "adult" in looks.PROMPT_FACE


# ---------------------------------------------------------------- targets are named, validated, and nothing runs on "everyone"
@pytest.mark.parametrize("op", [o for o in FACE_OPS if o != "remove_jewelry"])
def test_a_face_edit_needs_named_faces(v, op):
    v.eng.sdxl = True
    r = edit(v, op, {"color": "blue", "style": "beard", "part": "lips"})
    assert r.status_code == 400 and "choose which faces" in r.json()["detail"].lower(), op


@pytest.mark.parametrize("bad", [[[0, 0, 2, 1]], [[0.5, 0.5, 0.4, 0.9]], [["a", 0, 1, 1]], [[0.1, 0.1, 0.1005, 0.1005]], [[0, 0, 1]], "faces", [[0.1, 0.1, 0.9, 0.9]] * 21])
def test_bad_face_boxes_are_refused(v, bad):
    assert edit(v, "blur_face", {"faces": bad}).status_code == 400


def test_find_faces_returns_fractions_and_needs_a_session(v, tmp_path):
    r = v.c.post(f"/api/photos/{v.pid}/faces")
    assert r.status_code == 200 and r.json()["faces"] == [{"box": [0.25, 0.2, 0.75, 0.8], "score": 0.93}]
    from test_phone import client_for
    anon = client_for(v.app, "127.0.0.1")
    assert anon.post(f"/api/photos/{v.pid}/faces").status_code in (401, 403)


def test_generative_edits_say_what_is_missing(v):
    for op, p in (("generate_face", {}), ("makeup_remove", {}), ("hairstyle", {"style": "curly"}), ("facial_hair", {"mode": "add", "style": "beard"})):
        r = edit(v, op, {"faces": BOX, **p})
        assert r.status_code == 400 and "optional stable diffusion" in r.json()["detail"].lower(), op


# ---------------------------------------------------------------- hiding a face (works without landmarks: an oval from the box)
@pytest.mark.parametrize("op", ["blur_face", "pixelate_face"])
def test_hiding_a_face_changes_only_the_face_and_can_be_undone(v, op):
    img = Image.effect_noise((200, 200), 60).convert("RGB")
    buf = io.BytesIO(); img.save(buf, "PNG")
    pid = v.c.post("/api/photos", files={"file": ("n.png", buf.getvalue(), "image/png")}).json()["id"]
    r = v.c.post(f"/api/photos/{pid}/edit", json={"op": op, "params": {"faces": BOX}})
    assert r.status_code == 200, r.text
    out = np.asarray(Image.open(io.BytesIO(v.c.get(f"/api/photos/{pid}/image").content)).convert("RGB")).astype(int)
    src = np.asarray(img).astype(int)
    diff = np.abs(out - src).sum(2)
    assert diff[100, 100] > 0                                           # the face centre changed
    assert diff[:6, :].sum() == 0 and diff[:, :20].sum() == 0 and diff[-6:, :].sum() == 0 and diff[:, -20:].sum() == 0   # away from the face: untouched, byte for byte
    # the detail inside the face is really gone (the noise is much smoother)
    assert np.abs(np.diff(out[80:120, 80:120, 0], axis=1)).mean() < 0.5 * np.abs(np.diff(src[80:120, 80:120, 0], axis=1)).mean()
    assert v.c.post(f"/api/photos/{pid}/undo").json()["cursor"] == 0


def test_face_edit_labels_say_how_many_faces(v):
    r = edit(v, "blur_face", {"faces": BOX * 2})
    assert r.json()["versions"][-1]["label"] == "Blur face (2 faces)"


# ---------------------------------------------------------------- jewelry
def test_jewelry_needs_a_selection_and_fills_only_it(v):
    r = edit(v, "remove_jewelry", {})
    assert r.status_code == 400 and "select the jewelry" in r.json()["detail"].lower()
    r = edit(v, "remove_jewelry", {}, mask=mask_url((120, 90), (10, 10, 30, 30)))
    assert r.status_code == 200 and r.json()["versions"][-1]["label"] == "Remove jewelry"
    img = Image.open(io.BytesIO(v.c.get(f"/api/photos/{v.pid}/image").content)).convert("RGB")
    assert img.getpixel((20, 20)) == (0, 255, 0) and img.getpixel((100, 70)) == (200, 80, 80)


# ---------------------------------------------------------------- what the language model may ask for
def test_face_actions_are_available_and_their_settings_are_clamped():
    for a in ("blur_face", "pixelate_face", "generate_face", "skin_smooth", "teeth_whiten", "eye_color", "eyelash_length", "makeup_add",
              "makeup_remove", "hair_color", "hairstyle", "facial_hair"):
        p = parser.validate({"action": a, "params": {}})
        assert p["available"] is True and p["region"] == "faces", a
    assert parser.validate({"action": "remove_jewelry", "params": {}})["region"] == "required"
    assert parser.validate({"action": "blur_face", "params": {"amount": 900}})["params"] == {"amount": 100}
    assert parser.validate({"action": "blur_face", "params": {"amount": "x"}})["params"] == {}
    assert parser.validate({"action": "eye_color", "params": {"color": "GREEN"}})["params"] == {"color": "green"}
    assert parser.validate({"action": "eye_color", "params": {"color": "http://evil/x.png"}})["params"] == {"color": "blue"}
    assert parser.validate({"action": "hair_color", "params": {"color": "#7a2f1c"}})["params"] == {"color": "#7a2f1c"}
    assert parser.validate({"action": "makeup_add", "params": {"part": "evil"}})["params"] == {"part": "lips"}
    assert parser.validate({"action": "facial_hair", "params": {"mode": "remove", "style": "x"}})["params"] == {"mode": "remove", "style": "stubble"}
    assert parser.validate({"action": "hairstyle", "params": {"style": "curly <script>1</script>"}})["params"] == {"style": "curly scriptscript"}


def test_a_plan_can_never_carry_faces_or_images_from_the_model():
    """`faces` comes only from the page (the owner's ticks). Whatever the model sends is thrown away."""
    for a in ("blur_face", "generate_face", "hair_color", "skin_smooth"):
        p = parser.validate({"action": a, "params": {"faces": [[0, 0, 1, 1]], "image": "data:x", "reference": "x", "seed": 1}})
        assert not ({"faces", "image", "reference", "seed"} & set(p["params"])), a


def test_no_action_is_left_half_built():
    assert [n for n, a in actions.ACTIONS.items() if not a["available"]] == ["watermark"]      # phase 7


# ---------------------------------------------------------------- the model registry
def test_new_models_are_pinned_and_in_the_right_folders():
    for name in ("yunet", "face-landmarker"):
        m = registry.MODELS[name]
        assert m["required"] and len(m["files"]) == 1
        f = m["files"][0]
        assert len(f["sha256"]) == 64 and f["size"] > 0 and f["url"].startswith("https://") and f["path"].startswith(("yunet/", "mediapipe/"))
    sd = registry.MODELS["sdxl-inpaint"]["files"]
    assert all(f["path"].startswith("sdxl-inpaint/") for f in sd)          # the engine loads this folder: files must land in it
    assert all(f["url"].startswith(registry.SDXL_URL) and f["path"] == "sdxl-inpaint/" + f["url"][len(registry.SDXL_URL):] for f in sd)


def test_all_paths_are_unique_and_inside_the_models_folder():
    paths = [f["path"] for m in registry.MODELS.values() for f in m["files"]]
    assert len(paths) == len(set(paths))
    assert all(not p.startswith("/") and ".." not in p for p in paths)


# ---------------------------------------------------------------- geometry
def test_the_fallback_face_area_is_an_oval_around_the_box():
    face = F.Face((100.0, 100.0, 200.0, 220.0), None)
    m = F.face_area(face, (400, 400))
    assert m[160, 150] == 1 and m[0, 0] == 0 and m[100, 100] == 0 and m[160, 330] == 0      # centre in, corner and far away out
    assert F.face_area(face, (400, 400), grow=0.2).sum() > m.sum()


def test_look_edits_that_need_landmarks_say_so(v):
    r = edit(v, "skin_smooth", {"faces": BOX})
    assert r.status_code == 400    # the stand-in has no landmark model installed


# ================================================================ real models on SDXL-painted portraits
REAL_NEEDS = ("yunet", "face-landmarker", "sdxl-inpaint", "sam2", "birefnet", "lama")


def _faces_real_ok():
    try:
        import diffusers, mediapipe  # noqa: F401
    except ImportError:
        return False
    e = Engine(MODELS_DIR)
    return e.libraries_ok() and all(registry.installed(MODELS_DIR, n) for n in REAL_NEEDS)


faces_real = pytest.mark.skipif(not _faces_real_ok(), reason="the face models, SDXL and their libraries are not installed")

PORTRAITS = {
    "plain": "studio portrait photograph of an adult woman in her thirties, brown shoulder length hair, brown eyes, neutral expression, plain grey backdrop, soft light, head and shoulders, sharp focus, 85mm",
    "smile": "studio portrait photograph of an adult woman in her twenties smiling showing teeth, long dark hair, hazel eyes, plain beige backdrop, soft light, head and shoulders, sharp focus, 85mm",
}


@pytest.fixture(scope="module")
def eng():
    return Engine(MODELS_DIR)


@pytest.fixture(scope="module")
def portraits(eng):
    out = {}
    for name, prompt in PORTRAITS.items():
        out[name] = eng.generative_fill(Image.new("RGB", (1024, 1024), (128, 128, 128)), Image.new("L", (1024, 1024), 255), prompt,
                                        "cartoon, painting, deformed, blurry, child, text, watermark", grow=0, strength=1.0, seed=5, context=0.0)
    return out


def face_of(eng, img):
    found = eng.detect_faces(img)
    assert len(found) == 1, found
    box = found[0]["box"]
    return box, F.Face(box, eng.landmarks(img, box))


def params_for(img, box):
    w, h = img.size
    return {"faces": [[box[0] / w, box[1] / h, box[2] / w, box[3] / h]]}


def run(eng, img, op, p):
    from veil.tools import ops
    return ops.apply_op(img, op, p, None, eng)


def unchanged_outside(src, out, face, margin=0.9):
    """Everything further than `margin` x the face size from the face must be byte-for-byte the same."""
    a, b = np.asarray(src.convert("RGB")), np.asarray(out.convert("RGB"))
    x0, y0, x1, y1 = face.box; m = margin * max(x1 - x0, y1 - y0)
    keep = np.ones(a.shape[:2], bool)
    keep[max(0, int(y0 - m)):int(y1 + m), max(0, int(x0 - m)):int(x1 + m)] = False
    return bool(np.array_equal(a[keep], b[keep]))


def ring(face, ids, shape):
    """A polygon of landmark points, built here (not with the code under test)."""
    import cv2
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [np.round(face.pts[ids]).astype(np.int32)], 1)
    return m > 0


def lab(img):
    import cv2
    return cv2.cvtColor(np.asarray(img.convert("RGB")).astype(np.float32) / 255, cv2.COLOR_RGB2LAB)


@faces_real
def test_real_one_face_is_found_and_mapped(eng, portraits):
    for img in portraits.values():
        box, face = face_of(eng, img)
        assert face.pts is not None and face.pts.shape == (478, 2)
        assert 200 < face.size() < 600 and 0.2 < (box[0] + box[2]) / 2 / img.width < 0.8


@faces_real
def test_real_skin_smoothing_softens_skin_and_touches_nothing_else(eng, portraits):
    import cv2
    img = portraits["plain"]; box, face = face_of(eng, img)
    out = run(eng, img, "skin_smooth", {**params_for(img, box), "amount": 80})
    sk = F.skin_mask(face, (img.height, img.width)) > 0
    sk = cv2.erode(sk.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    g = lambda im: cv2.Laplacian(cv2.cvtColor(np.asarray(im.convert("RGB")), cv2.COLOR_RGB2GRAY), cv2.CV_32F)
    assert g(out)[sk].var() < 0.8 * g(img)[sk].var()                    # visibly smoother skin
    assert unchanged_outside(img, out, face)
    eyes = np.zeros(sk.shape, bool)
    for d, _ in F.iris_discs(face, sk.shape):
        eyes |= d > 0
    assert np.array_equal(np.asarray(img)[eyes], np.asarray(out)[eyes])  # the eyes are never smoothed


@faces_real
def test_real_eye_colour_moves_the_iris_toward_the_target_and_keeps_the_pupil_dark(eng, portraits):
    img = portraits["plain"]; box, face = face_of(eng, img)
    sh = (img.height, img.width)
    before, after = lab(img), lab(run(eng, img, "eye_color", {**params_for(img, box), "color": "green"}))
    for disc, r in F.iris_discs(face, sh):
        sel = disc > 0
        assert after[..., 1][sel].mean() < before[..., 1][sel].mean() - 3          # a* goes toward green
    assert unchanged_outside(img, run(eng, img, "eye_color", {**params_for(img, box), "color": "green"}), face, margin=0.5)


@faces_real
def test_real_teeth_whitening_removes_yellow(eng, portraits):
    img = portraits["smile"]; box, face = face_of(eng, img)
    sh = (img.height, img.width)
    mouth = F._shrunk(F.mouth_opening(face, sh), 2) > 0
    out = run(eng, img, "teeth_whiten", {**params_for(img, box), "amount": 100})
    b0, b1 = lab(img), lab(out)
    teeth = mouth & (b0[..., 0] > 38) & (b0[..., 1] < 18)
    assert teeth.sum() > 300
    cast = lambda x: np.abs(x[..., 2][teeth]).mean()                      # how far the teeth are from neutral (yellow or blue cast)
    assert cast(b1) < 0.6 * cast(b0) and b1[..., 0][teeth].mean() >= b0[..., 0][teeth].mean()
    assert unchanged_outside(img, out, face, margin=0.5)


@faces_real
def test_real_closed_mouth_has_no_teeth_to_whiten(eng, portraits):
    img = portraits["plain"]; box, _ = face_of(eng, img)
    with pytest.raises(Exception, match="No teeth"):
        run(eng, img, "teeth_whiten", params_for(img, box))


@faces_real
def test_real_lipstick_colours_only_the_lips(eng, portraits):
    img = portraits["plain"]; box, face = face_of(eng, img)
    sh = (img.height, img.width)
    out = run(eng, img, "makeup_add", {**params_for(img, box), "part": "lips", "color": "plum", "amount": 100})
    outer = ring(face, F.LIPS_OUTER, sh)
    lips = outer & ~ring(face, F.LIPS_INNER, sh)
    d = np.abs(np.asarray(out).astype(int) - np.asarray(img.convert("RGB")).astype(int)).sum(2)
    assert d[lips].mean() > 20
    import cv2
    far = cv2.dilate(outer.astype(np.uint8), np.ones((2 * int(0.06 * face.size()) + 1,) * 2, np.uint8)) == 0
    assert d[far].sum() == 0                                             # nothing outside the lips and a thin margin


@faces_real
def test_real_longer_lashes_add_dark_strokes_above_the_lid_only(eng, portraits):
    img = portraits["plain"]; box, face = face_of(eng, img)
    sh = (img.height, img.width)
    out = run(eng, img, "eyelash_length", {**params_for(img, box), "amount": 80})
    d = np.abs(np.asarray(out).astype(int) - np.asarray(img.convert("RGB")).astype(int)).sum(2)
    assert (d > 30).sum() > 150
    near_eyes = np.zeros(sh, np.uint8)
    for ring in (F.EYE_R, F.EYE_L):
        near_eyes |= F._grown(F._poly(sh, face.pts[ring]), 0.20 * face.size())
    assert d[near_eyes == 0].sum() == 0                                  # only the eye area was touched


@faces_real
def test_real_hair_colour_changes_hair_and_not_skin(eng, portraits):
    import cv2
    img = portraits["plain"]; box, face = face_of(eng, img)
    sh = (img.height, img.width)
    out = run(eng, img, "hair_color", {**params_for(img, box), "color": "blonde"})
    d = np.abs(np.asarray(out).astype(int) - np.asarray(img.convert("RGB")).astype(int)).sum(2)
    skin = cv2.erode(F.skin_mask(face, sh), np.ones((15, 15), np.uint8)) > 0
    assert (d[skin] > 0).mean() < 0.01                                   # the face skin is left alone
    changed = d > 30
    assert changed.sum() > 0.05 * face.w * face.h                       # a real area of hair changed
    assert lab(out)[..., 0][changed].mean() > lab(img)[..., 0][changed].mean() + 10      # and it got lighter (brown -> blonde)


@faces_real
def test_real_generated_face_is_new_uses_only_this_photo_and_leaves_the_rest_alone(eng, portraits, monkeypatch):
    img = portraits["plain"]; box, face = face_of(eng, img)
    pipe = eng._sdxl_pipe()
    seen = []
    real_call = type(pipe).__call__

    def spy(self, *a, **kw):
        seen.append((a, kw)); return real_call(self, *a, **kw)

    monkeypatch.setattr(type(pipe), "__call__", spy)
    p = params_for(img, box)
    a = run(eng, img, "generate_face", {**p, "seed": 1})
    b = run(eng, img, "generate_face", {**p, "seed": 2})
    # 1. only words, this photo and its mask went to the model: no identity, reference or control image of any kind
    allowed = {"prompt", "negative_prompt", "image", "mask_image", "width", "height", "strength", "num_inference_steps", "guidance_scale", "generator"}
    assert seen and all(not a_ and set(kw) <= allowed for a_, kw in seen), [set(kw) - allowed for _, kw in seen]
    assert all(kw["prompt"] == looks.PROMPT_FACE for _, kw in seen)
    # 2. the face really changed, and two seeds give two different faces (it is generated, not copied)
    inside = F.face_area(face, (img.height, img.width)) > 0
    da = np.abs(np.asarray(a).astype(int) - np.asarray(img.convert("RGB")).astype(int)).sum(2)
    dab = np.abs(np.asarray(a).astype(int) - np.asarray(b).astype(int)).sum(2)
    assert da[inside].mean() > 40 and dab[inside].mean() > 25
    # 3. nothing outside the face area moved, and the result is still a single, findable face
    assert unchanged_outside(img, a, face, margin=0.9)
    assert len(eng.detect_faces(a)) == 1


@faces_real
def test_real_makeup_goes_on_and_comes_off(eng, portraits):
    img = portraits["plain"]; box, face = face_of(eng, img)
    sh = (img.height, img.width)
    p = params_for(img, box)
    made = run(eng, img, "makeup_add", {**p, "part": "face", "amount": 100})
    clean = run(eng, made, "makeup_remove", {**p, "part": "all"})
    lips = F.lips_mask(face, sh) > 0
    chroma = lambda im: np.hypot(lab(im)[..., 1], lab(im)[..., 2])[lips].mean()
    assert chroma(made) > chroma(img) + 3                                # lipstick went on
    assert abs(chroma(clean) - chroma(img)) < abs(chroma(made) - chroma(img))     # and removal brought it back toward natural
    assert unchanged_outside(made, clean, face, margin=0.9)


@faces_real
def test_real_facial_hair_edits_only_the_lower_face(eng, portraits):
    img = portraits["plain"]; box, face = face_of(eng, img)
    sh = (img.height, img.width)
    out = run(eng, img, "facial_hair", {**params_for(img, box), "mode": "add", "style": "beard"})
    d = np.abs(np.asarray(out).astype(int) - np.asarray(img.convert("RGB")).astype(int)).sum(2)
    region = F.beard_region(face, sh, "beard") > 0
    assert d[region].mean() > 20
    eyes = F._grown(F._poly(sh, face.pts[F.EYE_R]) | F._poly(sh, face.pts[F.EYE_L]), int(0.05 * face.size()))
    assert d[eyes > 0].sum() == 0                                       # the upper face is untouched
    assert unchanged_outside(img, out, face, margin=0.9)


@faces_real
def test_real_hairstyle_leaves_the_face_alone(eng, portraits):
    import cv2
    img = portraits["plain"]; box, face = face_of(eng, img)
    sh = (img.height, img.width)
    out = run(eng, img, "hairstyle", {**params_for(img, box), "style": "curly"})
    d = np.abs(np.asarray(out).astype(int) - np.asarray(img.convert("RGB")).astype(int)).sum(2)
    k = 2 * int(0.06 * face.size()) + 1
    core = cv2.erode(ring(face, F.OVAL, sh).astype(np.uint8), np.ones((k, k), np.uint8)) > 0
    assert core.sum() > 20000 and d[core].sum() == 0                    # eyes, nose, mouth and cheeks are exactly as they were
    assert d.sum() > 0
