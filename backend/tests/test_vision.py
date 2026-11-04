"""Phase 5: selection, object/tattoo removal, background tools, and the model files.
Fast tests use a stand-in engine; the last section runs the real models when they are installed."""
import hashlib
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from conftest import make_jpeg
from test_phone import world  # noqa: F401
from veil.main import create_app
from veil.vision import registry
from veil.vision.engine import Engine
from veil.vision.registry import ModelError


def png_bytes(size=(120, 90), color=(200, 80, 80)):
    b = io.BytesIO(); Image.new("RGB", size, color).save(b, "PNG"); return b.getvalue()


def mask_url(size, box):
    import base64
    m = Image.new("L", size, 0); m.paste(255, box)
    b = io.BytesIO(); m.save(b, "PNG")
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


class FakeEngine(Engine):
    """Does no AI: remembers what it was asked and returns simple, checkable results."""

    def __init__(self, root, ready=True):
        super().__init__(root)
        self.calls = []
        self.ready = ready

    def status(self):
        s = super().status(); s["ready"] = self.ready; return s

    def segment(self, key, img, points, box):
        self.calls.append(("segment", key, points, box))
        m = Image.new("L", img.size, 0)
        w, h = img.size
        x, y = (points[0][0], points[0][1]) if points else (box[0], box[1])
        m.paste(255, (int(x * w) - 10, int(y * h) - 10, int(x * w) + 10, int(y * h) + 10))
        return m

    def inpaint(self, img, mask, grow=None):
        self.calls.append(("inpaint", grow))
        out = img.convert("RGB").copy(); out.paste((0, 255, 0), mask=mask.resize(img.size)); return out

    def matte(self, img):
        self.calls.append(("matte",))
        m = Image.new("L", img.size, 0); m.paste(255, (img.width // 4, 0, 3 * img.width // 4, img.height)); return m

    def sdxl_ready(self):
        return False


@pytest.fixture
def v(world, tmp_path):  # noqa: F811
    """An owner at the PC, a photo, and an app whose AI engine is the stand-in."""
    from types import SimpleNamespace
    eng = FakeEngine(tmp_path / "models")
    app = create_app(tmp_path / "data2", tailscale=world.ts, vision=eng)
    from test_phone import client_for
    owner = client_for(app, "127.0.0.1")
    owner.post("/api/auth/setup", json={"username": "owner", "password": "correct horse"}); owner.post("/api/notice/ack")
    pid = owner.post("/api/photos", files={"file": ("a.png", png_bytes((120, 90)), "image/png")}).json()["id"]
    return SimpleNamespace(c=owner, eng=eng, pid=pid, app=app, world=world, models=tmp_path / "models")


# ---------------------------------------------------------------- edits: what they do and refuse
def test_removing_things_needs_an_area(v):
    for op in ("remove_object", "remove_tattoo"):
        r = v.c.post(f"/api/photos/{v.pid}/edit", json={"op": op, "params": {}})
        assert r.status_code == 400 and "mark the area" in r.json()["detail"].lower(), op
        empty = mask_url((120, 90), (0, 0, 0, 0))
        r = v.c.post(f"/api/photos/{v.pid}/edit", json={"op": op, "params": {}, "mask": empty})
        assert r.status_code == 400, op
    assert not [c for c in v.eng.calls if c[0] == "inpaint"]  # the engine was never asked
    assert len(v.c.get(f"/api/photos/{v.pid}").json()["versions"]) == 1


def test_removal_fills_only_the_marked_area_and_can_be_undone(v):
    r = v.c.post(f"/api/photos/{v.pid}/edit", json={"op": "remove_object", "params": {}, "mask": mask_url((120, 90), (10, 10, 30, 30))})
    assert r.status_code == 200 and r.json()["versions"][-1]["label"] == "Remove object"
    img = Image.open(io.BytesIO(v.c.get(f"/api/photos/{v.pid}/image").content)).convert("RGB")
    assert img.getpixel((20, 20)) == (0, 255, 0) and img.getpixel((100, 70)) == (200, 80, 80)
    assert v.c.post(f"/api/photos/{v.pid}/undo").json()["cursor"] == 0


def test_tattoo_removal_takes_a_little_extra_around_the_edge(v):
    v.c.post(f"/api/photos/{v.pid}/edit", json={"op": "remove_tattoo", "params": {}, "mask": mask_url((120, 90), (10, 10, 30, 30))})
    v.c.post(f"/api/photos/{v.pid}/edit", json={"op": "remove_object", "params": {}, "mask": mask_url((120, 90), (10, 10, 30, 30))})
    grows = [c[1] for c in v.eng.calls if c[0] == "inpaint"]
    assert grows[0] is not None and grows[0] >= 6 and grows[1] is None


def test_remove_background_makes_it_transparent_around_the_person(v):
    r = v.c.post(f"/api/photos/{v.pid}/edit", json={"op": "remove_background", "params": {}})
    assert r.status_code == 200
    img = Image.open(io.BytesIO(v.c.get(f"/api/photos/{v.pid}/image").content))
    assert img.mode == "RGBA" and img.getpixel((5, 45))[3] == 0 and img.getpixel((60, 45))[3] == 255


def test_blur_background_keeps_the_person_exactly(v):
    # a photo with a sharp pattern, so blurring is visible
    arr = (np.indices((90, 120)).sum(axis=0) % 2 * 255).astype(np.uint8)
    b = io.BytesIO(); Image.fromarray(arr).convert("RGB").save(b, "PNG")
    pid = v.c.post("/api/photos", files={"file": ("p.png", b.getvalue(), "image/png")}).json()["id"]
    assert v.c.post(f"/api/photos/{pid}/edit", json={"op": "blur_background", "params": {"amount": 60}}).status_code == 200
    out = np.asarray(Image.open(io.BytesIO(v.c.get(f"/api/photos/{pid}/image").content)).convert("L")).astype(int)
    assert np.array_equal(out[:, 40:80], arr[:, 40:80].astype(int))          # person: untouched
    assert out[:, :20].std() < 40 and arr[:, :20].std() > 100                 # background: blurred


def test_replace_background_with_a_colour_and_with_a_scene(v):
    assert v.c.post(f"/api/photos/{v.pid}/edit", json={"op": "replace_background", "params": {"color": "#112233"}}).status_code == 200
    img = Image.open(io.BytesIO(v.c.get(f"/api/photos/{v.pid}/image").content)).convert("RGB")
    assert img.getpixel((5, 45)) == (0x11, 0x22, 0x33) and img.getpixel((60, 45)) == (200, 80, 80)
    r = v.c.post(f"/api/photos/{v.pid}/edit", json={"op": "replace_background", "params": {"scene": "a beach at sunset"}})
    assert r.status_code == 400 and "optional" in r.json()["detail"].lower()


def test_edits_say_so_when_the_models_are_not_installed(world, tmp_path):  # noqa: F811
    app = create_app(tmp_path / "d3", tailscale=world.ts, vision=Engine(tmp_path / "empty-models"))
    from test_phone import client_for
    c = client_for(app, "127.0.0.1")
    c.post("/api/auth/setup", json={"username": "o", "password": "correct horse"}); c.post("/api/notice/ack")
    pid = c.post("/api/photos", files={"file": ("a.png", png_bytes(), "image/png")}).json()["id"]
    for op, extra in [("remove_background", {}), ("remove_object", {"mask": mask_url((120, 90), (1, 1, 9, 9))})]:
        r = c.post(f"/api/photos/{pid}/edit", json={"op": op, "params": {}, **extra})
        assert r.status_code == 400 and ("not downloaded" in r.json()["detail"].lower() or "not installed" in r.json()["detail"].lower()), r.text
    assert c.post(f"/api/photos/{pid}/segment", json={"points": [[0.5, 0.5, 1]]}).status_code == 400


def test_ai_edits_have_no_live_preview(v):
    r = v.c.post(f"/api/photos/{v.pid}/preview", json={"op": "remove_background", "params": {}})
    assert r.status_code == 400
    assert not v.eng.calls


@pytest.mark.parametrize("op,params", [("blur_background", {"amount": 500}), ("blur_background", {"amount": "x"}),
                                       ("remove_object", {"grow": 9999})])
def test_out_of_range_numbers_are_refused(v, op, params):
    r = v.c.post(f"/api/photos/{v.pid}/edit", json={"op": op, "params": params, "mask": mask_url((120, 90), (1, 1, 9, 9))})
    assert r.status_code == 400


# ---------------------------------------------------------------- AI selection
def test_a_click_becomes_an_outline(v):
    r = v.c.post(f"/api/photos/{v.pid}/segment", json={"points": [[0.5, 0.5, 1]]})
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    m = Image.open(io.BytesIO(r.content))
    assert m.mode == "L" and m.getpixel((60, 45)) == 255 and m.getpixel((5, 5)) == 0
    name, key, pts, box = v.eng.calls[-1]
    assert key.startswith(v.pid) and pts == [(0.5, 0.5, 1)] and box is None
    assert v.c.post(f"/api/photos/{v.pid}/segment", json={"box": [0.2, 0.2, 0.6, 0.8]}).status_code == 200


@pytest.mark.parametrize("body", [{}, {"points": [[1.5, 0.5, 1]]}, {"points": [[0.5, 0.5, 2]]}, {"points": [[0.5, 0.5]]},
                                  {"points": [[0.1, 0.1, 1]] * 25}, {"box": [0.6, 0.2, 0.2, 0.8]}, {"box": [0, 0, 1]},
                                  {"box": [-0.1, 0, 1, 1]}])
def test_bad_selection_requests_are_refused(v, body):
    n = len(v.eng.calls)
    assert v.c.post(f"/api/photos/{v.pid}/segment", json=body).status_code in (400, 422)
    assert len(v.eng.calls) == n


def test_selection_needs_a_login_and_a_real_photo(v):
    assert v.c.post("/api/photos/0123456789abcdef/segment", json={"points": [[0.5, 0.5, 1]]}).status_code == 404
    from test_phone import client_for
    anon = client_for(v.app, "127.0.0.1")
    assert anon.post(f"/api/photos/{v.pid}/segment", json={"points": [[0.5, 0.5, 1]]}).status_code == 401


# ---------------------------------------------------------------- the command parser knows the new tools
def test_new_tools_are_available_actions_and_background_colours_are_cleaned():
    from veil.ai import parser
    for a in ("remove_tattoo", "remove_object", "remove_background", "blur_background", "replace_background"):
        assert parser.validate({"action": a, "params": {}})["available"] is True
    assert parser.validate({"action": "watermark", "params": {"text": "x"}})["available"] is False      # the watermark is added at export, not by the command bar
    p = parser.validate({"action": "replace_background", "params": {"color": "Beige"}})["params"]
    assert p == {"color": "#d8c3a5"}
    assert parser.validate({"action": "replace_background", "params": {"color": "javascript:alert(1)"}})["params"] == {"color": "#ffffff"}
    assert parser.validate({"action": "replace_background", "params": {"scene": "a beach"}})["params"] == {"scene": "a beach"}
    assert parser.validate({"action": "blur_background", "params": {"amount": 9000}})["params"] == {"amount": 100}
    assert parser.validate({"action": "remove_tattoo"})["region"] == "required"


# ---------------------------------------------------------------- model files and downloads
def test_status_lists_models_and_phones_cannot_download(v):
    s = v.c.get("/api/vision").json()
    names = {m["name"]: m for m in s["models"]}
    assert set(names) == {"sam2", "sam2-tiny", "sam2-large", "lama", "birefnet", "birefnet-lite", "yunet", "face-landmarker", "sdxl-inpaint"}
    assert names["sdxl-inpaint"]["required"] is False and "RAIL" in names["sdxl-inpaint"]["license"]
    assert names["sam2"]["license"] == "Apache-2.0" and names["birefnet"]["license"] == "MIT"
    w = v.world; w.enable(); w.pair()
    assert w.phone.get("/api/vision").status_code in (200, 401, 403)
    assert w.phone.post("/api/vision/models/sam2/download").status_code == 403
    assert v.c.post("/api/vision/models/nope/download").status_code == 404


def tiny_model(tmp_path, content=b"hello veil", name="t.bin"):
    return {"path": name, "size": len(content), "sha256": hashlib.sha256(content).hexdigest()}


def test_a_changed_file_is_never_accepted(tmp_path):
    f = tiny_model(tmp_path)
    (tmp_path / "t.bin").write_bytes(b"hello veil")
    assert registry.file_ok(tmp_path, f, deep=True)
    (tmp_path / "t.bin").write_bytes(b"hello vell")          # same size, one letter different
    assert not registry.file_ok(tmp_path, f, deep=True) and registry.file_ok(tmp_path, f, deep=False)
    (tmp_path / "t.bin").write_bytes(b"short")
    assert not registry.file_ok(tmp_path, f, deep=False)
    assert not registry.file_ok(tmp_path, tiny_model(tmp_path, name="missing.bin"))


def test_engine_verifies_the_checksum_before_using_a_model(tmp_path, monkeypatch):
    f = registry.MODELS["lama"]["files"][0]
    (tmp_path / "lama").mkdir()
    (tmp_path / "lama" / "big-lama.pt").write_bytes(b"\0" * f["size"])       # right size, wrong content
    eng = Engine(tmp_path)
    assert eng.has("lama") is True                                           # the quick check passes...
    with pytest.raises(ModelError, match="checksum"):                        # ...the real one does not
        eng.inpaint(Image.new("RGB", (64, 64)), Image.new("L", (64, 64), 255))


def test_git_sha1_files_are_checked_too(tmp_path):
    data = b'{"a": 1}\n'
    f = {"path": "c.json", "size": len(data), "git_sha1": hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()}
    (tmp_path / "c.json").write_bytes(data)
    assert registry.file_ok(tmp_path, f)
    (tmp_path / "c.json").write_bytes(b'{"a": 2}\n')
    assert not registry.file_ok(tmp_path, f)


class FileServer:
    def __init__(self, files):
        class H(BaseHTTPRequestHandler):
            def log_message(self, *a): pass
            def do_GET(self):
                body = files.get(self.path)
                if body is None:
                    self.send_response(404); self.send_header("content-length", "0"); self.end_headers(); return
                self.send_response(200); self.send_header("content-length", str(len(body))); self.end_headers(); self.wfile.write(body)
        self.s = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.s.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.s.server_address[1]}"


def wait(d):
    import time
    for _ in range(100):
        if not d.snapshot()["running"]:
            return d.snapshot()
        time.sleep(0.05)
    raise AssertionError("download did not finish")


@pytest.fixture
def one_file_model(monkeypatch):
    good = b"model-bytes" * 1000
    srv = FileServer({"/good.bin": good, "/bad.bin": good[:-1] + b"X"})
    def make(path):
        return {"label": "T", "use": "", "license": "MIT", "required": False, "tier": "Optional",
                "files": [{"path": path, "size": len(good), "sha256": hashlib.sha256(good).hexdigest(), "url": srv.url + "/" + path}]}
    monkeypatch.setitem(registry.MODELS, "good", make("good.bin"))
    monkeypatch.setitem(registry.MODELS, "bad", make("bad.bin"))
    yield good
    srv.s.shutdown()


def test_a_download_is_checked_and_kept_only_if_it_matches(tmp_path, one_file_model):
    d = registry.Downloader(tmp_path)
    d.start("good"); s = wait(d)
    assert s["error"] == "" and (tmp_path / "good.bin").read_bytes() == one_file_model and registry.installed(tmp_path, "good", deep=True)
    d2 = registry.Downloader(tmp_path)
    d2.start("bad"); s = wait(d2)
    assert "checksum" in s["error"] and not (tmp_path / "bad.bin").exists() and not (tmp_path / "bad.bin.part").exists()


def test_only_known_models_can_be_downloaded(tmp_path):
    with pytest.raises(ModelError):
        registry.Downloader(tmp_path).start("../../etc/passwd")


def test_every_registry_url_is_pinned_and_secure():
    for name, m in registry.MODELS.items():
        for f in m["files"]:
            assert f["url"].startswith("https://") and f["size"] > 0 and ("sha256" in f or "git_sha1" in f), (name, f["path"])
            assert ".." not in f["path"] and not f["path"].startswith("/")
    assert "main" not in registry.SDXL_URL and registry.SDXL_REV in registry.SDXL_URL   # a fixed revision, not a moving branch
    assert sum(f["size"] for f in registry.MODELS["sdxl-inpaint"]["files"]) == 6941218469


# ---------------------------------------------------------------- the real models (skipped when they are not installed)
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
real = pytest.mark.skipif(not (Engine(MODELS_DIR).libraries_ok() and all(registry.installed(MODELS_DIR, n) for n in ("sam2", "lama", "birefnet"))),
                          reason="editing models are not installed")


def scene():
    """A smooth backdrop with a bright red disc (the 'object') on it."""
    w, h = 640, 480
    yy, xx = np.mgrid[0:h, 0:w]
    bg = np.stack([60 + 100 * xx / w, 90 + 80 * yy / h, 140 + 0 * xx], axis=-1).astype(np.uint8)
    disc = (xx - 320) ** 2 + (yy - 240) ** 2 < 70 ** 2
    img = bg.copy(); img[disc] = (210, 30, 40)
    return Image.fromarray(img), Image.fromarray(bg), disc


@real
def test_real_sam_outlines_the_object_from_one_click():
    img, _, disc = scene()
    eng = Engine(MODELS_DIR)
    m = np.asarray(eng.segment("k", img, [(0.5, 0.5, 1)], None)) > 127
    iou = (m & disc).sum() / (m | disc).sum()
    assert iou > 0.85, iou
    box = np.asarray(eng.segment("k", img, [], (0.38, 0.3, 0.62, 0.7))) > 127
    assert (box & disc).sum() / (box | disc).sum() > 0.85
    eng.release()


@real
def test_real_lama_removes_the_object_and_leaves_the_rest_alone():
    img, bg, disc = scene()
    eng = Engine(MODELS_DIR)
    mask = Image.fromarray(disc.astype(np.uint8) * 255)
    out = np.asarray(eng.inpaint(img, mask)).astype(int)
    ref = np.asarray(bg).astype(int)
    inside = np.abs(out - ref)[disc].mean()
    assert inside < 12, f"filled area is {inside:.1f} levels away from the true backdrop"
    assert np.abs(out[:, :200] - np.asarray(img).astype(int)[:, :200]).max() == 0   # far away: byte-identical
    eng.release()


@real
def test_real_birefnet_finds_the_subject():
    img, _, disc = scene()
    eng = Engine(MODELS_DIR)
    m = np.asarray(eng.matte(img)).astype(float) / 255
    assert m[disc].mean() > 0.7 and m[~disc].mean() < 0.3, (m[disc].mean(), m[~disc].mean())
    eng.release()


@real
def test_real_lama_on_a_large_hole_changes_nothing_outside_it():
    """A big hole means a big crop, which is shrunk for the model and scaled back; only the hole may change."""
    w, h = 3200, 2400
    yy, xx = np.mgrid[0:h, 0:w]
    bg = np.stack([60 + 100 * xx / w, 90 + 80 * yy / h, 140 + 30 * np.sin(xx / 90.0)], axis=-1).astype(np.uint8)
    hole = (xx - 1600) ** 2 + (yy - 1200) ** 2 < 800 ** 2
    img = bg.copy(); img[hole] = (210, 30, 40)
    eng = Engine(MODELS_DIR)
    out = np.asarray(eng.inpaint(Image.fromarray(img), Image.fromarray(hole.astype(np.uint8) * 255))).astype(int)
    ring = ~hole & ((xx - 1600) ** 2 + (yy - 1200) ** 2 > 860 ** 2)             # outside the hole and its grown edge and feathering
    assert np.abs(out - img.astype(int))[ring].max() == 0, "pixels outside the hole were altered"
    assert np.abs(out - bg.astype(int))[hole].mean() < 25                         # and the hole is filled with plausible backdrop
    eng.release()


# ---------------------------------------------------------------- phase 8: model tiers (SAM tiny/base+/large, BiRefNet lite/full)
def _sparse(root, names):
    for n in names:
        for f in registry.MODELS[n]["files"]:
            p = root / f["path"]
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "wb") as fh:
                fh.truncate(f["size"])


def test_engine_uses_the_best_installed_variant(tmp_path):
    from veil.vision.engine import Engine, ModelError
    e = Engine(tmp_path)
    with pytest.raises(ModelError, match="Segment Anything is not downloaded"):
        e.pick("selection")
    with pytest.raises(ModelError, match="BiRefNet is not downloaded"):
        e.pick("matte")
    _sparse(tmp_path, ["sam2-tiny", "birefnet-lite"])
    assert (e.pick("selection"), e.pick("matte")) == ("sam2-tiny", "birefnet-lite")
    _sparse(tmp_path, ["sam2"])
    assert e.pick("selection") == "sam2"
    _sparse(tmp_path, ["sam2-large", "birefnet"])
    assert (e.pick("selection"), e.pick("matte")) == ("sam2-large", "birefnet")


@pytest.mark.parametrize("have,ready", [
    (["sam2-tiny", "birefnet-lite", "lama", "yunet", "face-landmarker"], True),             # the Minimum set
    (["sam2", "birefnet", "lama", "yunet", "face-landmarker"], True),                       # Recommended
    (["sam2-large", "birefnet", "lama", "yunet", "face-landmarker"], True),                 # High-end
    (["sam2-tiny", "birefnet", "lama", "yunet", "face-landmarker"], True),                  # mixed is fine
    (["sam2-tiny", "lama", "yunet", "face-landmarker"], False),                             # no person finder
    (["birefnet-lite", "lama", "yunet", "face-landmarker"], False),                         # no outline model
    (["sam2-tiny", "birefnet-lite", "yunet", "face-landmarker"], False),                    # no LaMa
])
def test_ready_means_one_of_each_choice_plus_the_single_models(tmp_path, have, ready):
    from veil.vision.engine import Engine
    _sparse(tmp_path, have)
    e = Engine(tmp_path)
    if not e.libraries_ok():
        pytest.skip("torch/opencv are not installed")
    assert e.status()["ready"] is ready


def test_every_sam_variant_names_a_real_config_and_lite_uses_the_small_backbone():
    for n in ("sam2", "sam2-tiny", "sam2-large"):
        assert registry.MODELS[n]["sam_config"].startswith("configs/sam2.1/sam2.1_hiera_")
    assert registry.MODELS["birefnet"]["backbone"] == 6 and registry.MODELS["birefnet-lite"]["backbone"] == 3
    src = (Path(__file__).resolve().parents[1] / "veil/vision/vendor/birefnet.py").read_text()
    assert "][self.bb_index]" in src and "bb_index = 6" in src                  # the full model is still the default
