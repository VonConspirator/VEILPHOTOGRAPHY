"""The GPU worker: pairing, the worker's locks, the client's fallback, and what travels where. Models are faked here;
the real-model comparison is in test_worker_real.py."""
import hashlib
import io
import json
import os
import stat
import threading

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from conftest import make_jpeg
from veil import remote
from veil.main import create_app
from veil.remote import AddressError, Dispatcher, WorkerClient, WorkerConfig, WorkerUnavailable, parse_address, png_bytes, resolve_private
from veil.vision.registry import ModelError
from veil.worker import server as wserver
from veil.worker import store as wstore
from veil.worker.server import create_app as worker_app
from veil.worker.store import PairError, WorkerStore

TAIL = ("100.64.0.9", 40000)
ADDR = "100.101.102.103"


class WorkerEngine:
    """Stands in for the real Engine on the worker. Remembers what it was asked to do."""

    def __init__(self, caps=("segment", "matte", "inpaint", "faces", "landmarks", "generative")):
        self.caps = set(caps)
        self.calls = []
        self.boom = None

    def has(self, n):
        return {"sam2": "segment", "birefnet": "matte", "lama": "inpaint", "yunet": "faces"}.get(n) in self.caps

    def libraries_ok(self): return True
    def faces_ready(self): return "landmarks" in self.caps
    def sdxl_ready(self): return "generative" in self.caps
    def device(self): return "cuda"

    def _go(self, name, *a):
        self.calls.append((name, a))
        if self.boom:
            raise self.boom
        return None

    def segment(self, key, img, points, box):
        self._go("segment", key, img.size, points, box)
        return Image.new("L", img.size, 200)

    def matte(self, img):
        self._go("matte", img.size)
        return Image.new("L", img.size, 120)

    def inpaint(self, img, mask, grow=None):
        self._go("inpaint", img.size, mask.mode, grow)
        return Image.new("RGB", img.size, (1, 2, 3))

    def generative_fill(self, img, mask, prompt, negative="", grow=None, **kw):
        self._go("generative_fill", prompt, negative, grow, kw)
        return Image.new("RGB", img.size, (9, 8, 7))

    def detect_faces(self, img, min_score=0.6):
        self._go("detect_faces", min_score)
        return [{"box": (1.0, 2.0, 30.0, 40.0), "score": 0.9}]

    def landmarks(self, img, box):
        self._go("landmarks", box)
        return np.arange(478 * 2, dtype=float).reshape(478, 2)


class LocalEngine(WorkerEngine):
    """The engine on the computer running Veil."""
    root = None

    def status(self):
        return {"ready": False, "device": "cpu", "libraries": True, "models": []}

    def release(self): pass


class Bridge(httpx.BaseTransport):
    """Carries the client's requests straight into the worker's app, as if over Tailscale."""

    def __init__(self, tc):
        self.tc, self.calls = tc, []

    def handle_request(self, request):
        self.calls.append((request.method, request.url.path))
        r = self.tc.request(request.method, request.url.path, content=request.content, headers=dict(request.headers))
        return httpx.Response(r.status_code, headers={"content-type": r.headers.get("content-type", "")}, content=r.content)


class Dead(httpx.BaseTransport):
    def __init__(self): self.calls = 0
    def handle_request(self, request):
        self.calls += 1
        raise httpx.ConnectError("no route")


def img(size=(64, 48), color=(10, 20, 30)):
    return Image.new("RGB", size, color)


@pytest.fixture
def wdir(tmp_path):
    return tmp_path / "worker-data"


@pytest.fixture
def weng():
    return WorkerEngine()


@pytest.fixture
def wapp(wdir, weng):
    return worker_app(wdir, engine=weng, name="gpu-box")


@pytest.fixture
def wtc(wapp):
    return TestClient(wapp, base_url="http://100.101.102.103:5196", client=TAIL)


def pair_token(wtc, wapp, name="laptop"):
    code, _ = wapp.state.store.new_code()
    r = wtc.post("/v1/pair", json={"code": code, "name": name})
    assert r.status_code == 200, r.text
    return r.json()["token"]


@pytest.fixture
def token(wtc, wapp):
    return pair_token(wtc, wapp)


def auth(token):
    return {"Authorization": "Bearer " + token}


def upload(wtc, token, data):
    sha = hashlib.sha256(data).hexdigest()
    assert wtc.put(f"/v1/images/{sha}", content=data, headers=auth(token)).status_code == 200
    return sha


# ================================================================ pairing store
def test_a_code_works_once(tmp_path):
    s = WorkerStore(tmp_path)
    code, _ = s.new_code()
    cid, tok = s.redeem(code, "laptop")
    assert s.check(tok) == cid
    with pytest.raises(PairError):
        s.redeem(code, "again")


def test_code_is_typed_forgivingly(tmp_path):
    s = WorkerStore(tmp_path)
    code, _ = s.new_code()
    s.redeem(wstore.pretty_code(code).lower() + " ", "x")


def test_a_code_expires(tmp_path, monkeypatch):
    s = WorkerStore(tmp_path)
    code, _ = s.new_code()
    real = wstore.time.time
    monkeypatch.setattr(wstore.time, "time", lambda: real() + wstore.CODE_SECONDS + 1)
    with pytest.raises(PairError):
        s.redeem(code, "late")


def test_five_wrong_guesses_burn_every_pending_code(tmp_path):
    s = WorkerStore(tmp_path)
    good, _ = s.new_code()
    for i in range(wstore.MAX_BAD_CODES):
        with pytest.raises(PairError):
            s.redeem(f"WRONG{i}WRONG", "guesser")
    s._tries.clear()                       # (the per-minute limit is tested separately)
    with pytest.raises(PairError):
        s.redeem(good, "owner")            # the real code died with the guessing


def test_pairing_attempts_are_rate_limited(tmp_path):
    s = WorkerStore(tmp_path)
    s.new_code()
    seen = []
    for _ in range(wstore.PAIR_MAX_PER_WINDOW + 2):
        try:
            s.redeem("NOPE", "x")
        except PairError as e:
            seen.append(str(e))
    assert any("Too many tries" in m for m in seen)


def test_only_five_computers_can_pair(tmp_path):
    s = WorkerStore(tmp_path)
    for i in range(wstore.MAX_CLIENTS):
        s.redeem(s.new_code()[0], f"c{i}")
    with pytest.raises(PairError, match="already has 5"):
        s.redeem(s.new_code()[0], "sixth")


def test_revoked_token_stops_working(tmp_path):
    s = WorkerStore(tmp_path)
    cid, tok = s.redeem(s.new_code()[0], "x")
    assert s.revoke(cid) and s.check(tok) is None and not s.revoke(cid)


def test_the_file_keeps_only_hashes_and_is_private(tmp_path):
    s = WorkerStore(tmp_path)
    code, _ = s.new_code()
    _, tok = s.redeem(code, "x")
    text = s.path.read_text()
    assert tok not in text and code not in text
    assert stat.S_IMODE(os.stat(s.path).st_mode) == 0o600


def test_garbage_tokens_are_refused(tmp_path):
    s = WorkerStore(tmp_path)
    assert s.check("") is None and s.check("x" * 500) is None and s.check("vwk_nope") is None


# ================================================================ the worker's locks
def test_other_addresses_are_refused(wapp):
    for peer in ("192.168.1.20", "8.8.8.8", "10.0.0.5"):
        tc = TestClient(wapp, client=(peer, 1))
        assert tc.get("/v1/health").status_code == 403
        assert tc.post("/v1/pair", json={"code": "x"}).status_code == 403


def test_a_web_page_is_never_a_client(wtc, token):
    r = wtc.get("/v1/health", headers={**auth(token), "Origin": "http://evil.example"})
    assert r.status_code == 403


def test_no_token_or_wrong_token(wtc, token):
    assert wtc.get("/v1/health").status_code == 401
    assert wtc.get("/v1/health", headers=auth("vwk_wrong")).status_code == 401
    assert wtc.get("/v1/health", headers={"Authorization": token}).status_code == 401      # not "Bearer ..."
    assert wtc.get("/v1/health", headers=auth(token)).status_code == 200


def test_too_many_wrong_tokens_are_slowed(wtc):
    codes = [wtc.get("/v1/health", headers=auth("vwk_x")).status_code for _ in range(wserver.BAD_AUTH_PER_MINUTE + 3)]
    assert codes[0] == 401 and codes[-1] == 429


def test_every_job_route_needs_a_token(wapp, wtc):
    paths = {r.path for r in wapp.routes if r.path.startswith("/v1/")}
    assert paths == {"/v1/pair", "/v1/health", "/v1/unpair", "/v1/images/{sha}", "/v1/segment", "/v1/matte", "/v1/inpaint",
                     "/v1/generative_fill", "/v1/faces", "/v1/landmarks"}
    for p in paths - {"/v1/pair"}:
        for m in ("get", "post", "put"):
            r = getattr(wtc, m)(p.replace("{sha}", "a" * 64))
            assert r.status_code in (401, 405), (m, p, r.status_code)


def test_health_reports_what_the_worker_can_do(wtc, token):
    h = wtc.get("/v1/health", headers=auth(token)).json()
    assert h["name"] == "gpu-box" and h["protocol"] == 1 and h["device"] == "cuda"
    assert all(h["capabilities"].values())
    assert "lama" in h["models"]


def test_health_on_a_worker_missing_models(wdir):
    app = worker_app(wdir, engine=WorkerEngine(caps=("matte",)))
    tc = TestClient(app, client=TAIL)
    tok = pair_token(tc, app)
    caps = tc.get("/v1/health", headers=auth(tok)).json()["capabilities"]
    assert caps["matte"] and not caps["segment"] and not caps["generative"] and not caps["inpaint"]


def test_unpair_removes_the_token(wtc, wapp, token):
    assert wtc.post("/v1/unpair", headers=auth(token)).status_code == 200
    assert wtc.get("/v1/health", headers=auth(token)).status_code == 401


# ================================================================ images and jobs
def test_a_job_without_its_image_asks_for_it(wtc, token):
    sha = hashlib.sha256(b"x").hexdigest()
    r = wtc.post("/v1/matte", json={"image": sha}, headers=auth(token))
    assert r.status_code == 409 and r.json()["detail"] == {"need": [sha]}


def test_upload_then_job(wtc, token, weng):
    data = png_bytes(img())
    sha = upload(wtc, token, data)
    r = wtc.post("/v1/matte", json={"image": sha}, headers=auth(token))
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    out = Image.open(io.BytesIO(r.content))
    assert out.size == (64, 48) and out.getpixel((1, 1)) == 120
    assert weng.calls == [("matte", ((64, 48),))]


def test_an_image_that_arrives_damaged_is_refused(wtc, token):
    data = png_bytes(img())
    r = wtc.put(f"/v1/images/{hashlib.sha256(data).hexdigest()}", content=data[:-5] + b"xxxxx", headers=auth(token))
    assert r.status_code == 400


def test_only_png_is_accepted(wtc, token):
    buf = io.BytesIO(); img().save(buf, "JPEG")
    data = buf.getvalue()
    assert wtc.put(f"/v1/images/{hashlib.sha256(data).hexdigest()}", content=data, headers=auth(token)).status_code == 400
    junk = b"not an image at all"
    assert wtc.put(f"/v1/images/{hashlib.sha256(junk).hexdigest()}", content=junk, headers=auth(token)).status_code == 400


def test_huge_pictures_are_refused(wtc, token, monkeypatch):
    monkeypatch.setattr(wserver.config, "MAX_PIXELS", 1000)
    data = png_bytes(img((100, 100)))
    assert wtc.put(f"/v1/images/{hashlib.sha256(data).hexdigest()}", content=data, headers=auth(token)).status_code == 400


def test_oversized_uploads_are_refused(wtc, token, monkeypatch):
    monkeypatch.setattr(wserver, "MAX_BODY", 100)
    data = png_bytes(img((50, 50), (1, 1, 1)))
    assert len(data) > 100
    assert wtc.put(f"/v1/images/{hashlib.sha256(data).hexdigest()}", content=data, headers=auth(token)).status_code == 413


def test_bad_image_ids(wtc, token):
    assert wtc.put("/v1/images/not-a-hash", content=b"x", headers=auth(token)).status_code == 400
    assert wtc.post("/v1/matte", json={"image": "../../etc/passwd"}, headers=auth(token)).status_code == 422


def test_segment_validates_points_and_box(wtc, token, weng):
    sha = upload(wtc, token, png_bytes(img()))
    h = auth(token)
    for bad in ({"points": [[2, 0.5, 1]]}, {"points": [[0.5, 0.5, 7]]}, {"points": [[0.5, 0.5]]}, {"box": [0.5, 0.5, 0.2, 0.9]}, {"box": [0, 0, 1, 2]}):
        assert wtc.post("/v1/segment", json={"image": sha, **bad}, headers=h).status_code in (400, 422), bad
    assert wtc.post("/v1/segment", json={"image": sha, "points": [[0.5, 0.5, 1]]}, headers=h).status_code == 200
    name, (key, size, pts, box) = weng.calls[-1]
    assert key == sha and pts == [(0.5, 0.5, 1)] and box is None
    assert wtc.post("/v1/segment", json={"image": sha, "points": [[0.5, 0.5, 1]] * 25}, headers=h).status_code == 422


def test_inpaint_and_fill_get_their_values(wtc, token, weng):
    si, sm = upload(wtc, token, png_bytes(img())), upload(wtc, token, png_bytes(Image.new("L", (64, 48), 255)))
    h = auth(token)
    r = wtc.post("/v1/inpaint", json={"image": si, "mask": sm, "grow": 7}, headers=h)
    assert r.status_code == 200 and weng.calls[-1] == ("inpaint", ((64, 48), "L", 7))
    r = wtc.post("/v1/generative_fill", json={"image": si, "mask": sm, "prompt": "skin", "steps": 12, "seed": 5}, headers=h)
    assert r.status_code == 200
    name, (prompt, neg, grow, kw) = weng.calls[-1]
    assert (prompt, kw["steps"], kw["seed"]) == ("skin", 12, 5)
    for bad in ({"steps": 500}, {"strength": 2}, {"prompt": "x" * 1001}, {"seed": -1}, {"feather_px": 99}):
        assert wtc.post("/v1/generative_fill", json={"image": si, "mask": sm, "prompt": "p", **bad}, headers=h).status_code == 422, bad


def test_faces_and_landmarks(wtc, token):
    sha = upload(wtc, token, png_bytes(img()))
    h = auth(token)
    assert wtc.post("/v1/faces", json={"image": sha}, headers=h).json() == {"faces": [{"box": [1.0, 2.0, 30.0, 40.0], "score": 0.9}]}
    pts = wtc.post("/v1/landmarks", json={"image": sha, "box": [1, 2, 30, 40]}, headers=h).json()["points"]
    assert len(pts) == 478 and pts[1] == [2.0, 3.0]
    assert wtc.post("/v1/landmarks", json={"image": sha, "box": [30, 2, 1, 40]}, headers=h).status_code == 400


def test_model_problems_are_told_plainly(wtc, token, weng):
    sha = upload(wtc, token, png_bytes(img()))
    weng.boom = ModelError("Mark the area first.")
    r = wtc.post("/v1/matte", json={"image": sha}, headers=auth(token))
    assert r.status_code == 400 and r.json()["detail"] == "Mark the area first."


def test_a_crash_never_leaks_details(wtc, token, weng):
    sha = upload(wtc, token, png_bytes(img()))
    weng.boom = RuntimeError("secret path /home/x/photo.png")
    r = wtc.post("/v1/matte", json={"image": sha}, headers=auth(token))
    assert r.status_code == 500 and "secret" not in r.text and "/home" not in r.text


def test_a_busy_worker_says_so(wtc, wapp, token, monkeypatch):
    sha = upload(wtc, token, png_bytes(img()))
    monkeypatch.setattr(wserver, "QUEUE_WAIT", 0.05)
    started, release = threading.Event(), threading.Event()
    real = wapp.state.engine.matte

    def slow(i):
        started.set(); release.wait(5)
        return real(i)
    wapp.state.engine.matte = slow
    t = threading.Thread(target=lambda: wtc.post("/v1/matte", json={"image": sha}, headers=auth(token)))
    t.start()
    assert started.wait(5)
    try:
        r = wtc.post("/v1/matte", json={"image": sha}, headers=auth(token))
        assert r.status_code == 503 and "busy" in r.json()["detail"]
    finally:
        release.set(); t.join()


def test_images_live_in_memory_only_and_are_dropped(wtc, wapp, wdir, token, monkeypatch):
    first = upload(wtc, token, png_bytes(img(color=(1, 1, 1))))
    for i in range(wserver.MAX_BLOBS + 2):
        upload(wtc, token, png_bytes(img(color=(2 + i, 3, 4))))
    assert wapp.state.blobs.get(first) is None                        # the oldest were forgotten
    assert sorted(p.name for p in wdir.iterdir()) == ["worker-server.json"]       # no photo ever touched the disk
    monkeypatch.setattr(wserver, "BLOB_SECONDS", -1)
    assert wapp.state.blobs.get(hashlib.sha256(png_bytes(img(color=(2 + wserver.MAX_BLOBS + 1, 3, 4)))).hexdigest()) is None


def test_no_job_takes_a_second_picture_to_copy_a_face_from(wapp):
    """The face rule: a picture is only ever the photo being edited, or its mask. Nothing accepts another picture to take a face from."""
    for model in (wserver.SegmentReq, wserver.ImageReq, wserver.InpaintReq, wserver.FillReq, wserver.FacesReq, wserver.LandmarksReq):
        imgs = {f for f in model.model_fields if f in ("image", "mask") or "img" in f or "face" in f or "source" in f or "reference" in f or "style" in f}
        assert imgs <= {"image", "mask"}, (model.__name__, imgs)
    assert set(wserver.FillReq.model_fields) == {"image", "mask", "grow", "prompt", "negative", "strength", "seed", "context", "steps", "feather_px"}


# ================================================================ addresses
@pytest.mark.parametrize("text,want", [("100.101.102.103", (ADDR, 5196)), ("100.101.102.103:6000", (ADDR, 6000)), ("My-GPU.tail1.ts.net", ("my-gpu.tail1.ts.net", 5196)),
                                       ("[fd7a:115c:a1e0::1]:6001", ("fd7a:115c:a1e0::1", 6001)), ("fd7a:115c:a1e0::1", ("fd7a:115c:a1e0::1", 5196))])
def test_parse_address(text, want):
    assert parse_address(text) == want


@pytest.mark.parametrize("text", ["", "http://100.1.1.1", "100.1.1.1/x", "user@100.1.1.1", "a b", "100.1.1.1:99999", "100.1.1.1:abc", "host?x=1", "ho#st"])
def test_bad_addresses(text):
    with pytest.raises(AddressError):
        parse_address(text)


@pytest.mark.parametrize("ip", ["192.168.1.5", "10.0.0.2", "8.8.8.8", "172.16.0.1", "169.254.1.1", "100.63.255.255", "100.128.0.1"])
def test_only_tailscale_addresses_are_allowed(ip):
    with pytest.raises(AddressError, match="not a Tailscale address"):
        resolve_private(ip, 5196)


@pytest.mark.parametrize("ip", ["100.64.0.1", "100.101.102.103", "100.127.255.254", "127.0.0.1", "fd7a:115c:a1e0::5"])
def test_tailscale_addresses_pass(ip):
    assert resolve_private(ip, 5196) == ip


def test_a_name_that_points_at_the_public_internet_is_refused(monkeypatch):
    monkeypatch.setattr(remote.socket, "getaddrinfo", lambda *a, **k: [(2, 1, 6, "", ("93.184.216.34", 5196))])
    with pytest.raises(AddressError, match="not a Tailscale address"):
        resolve_private("evil.example.com", 5196)
    monkeypatch.setattr(remote.socket, "getaddrinfo", lambda *a, **k: [(2, 1, 6, "", ("100.70.0.1", 5196)), (2, 1, 6, "", ("93.184.216.34", 5196))])
    with pytest.raises(AddressError):                    # one bad answer among good ones is still refused
        resolve_private("mixed.example.com", 5196)


def test_an_unknown_name_is_explained(monkeypatch):
    def nope(*a, **k): raise OSError("no")
    monkeypatch.setattr(remote.socket, "getaddrinfo", nope)
    with pytest.raises(AddressError, match="Could not find"):
        resolve_private("nowhere.ts.net", 5196)


# ================================================================ the client and the fallback
@pytest.fixture
def cfg(tmp_path):
    return WorkerConfig(tmp_path / "client-data")


def paired_client(cfg, wtc, wapp):
    bridge = Bridge(wtc)
    client = WorkerClient(cfg, transport=bridge)
    code, _ = wapp.state.store.new_code()
    client.pair(ADDR, code, "my-pc")
    return client, bridge


@pytest.fixture
def setup(cfg, wtc, wapp):
    client, bridge = paired_client(cfg, wtc, wapp)
    local = LocalEngine(caps=())
    return Dispatcher(local, client), client, bridge, local


def test_pairing_saves_a_private_config(setup, cfg):
    d = cfg.load()
    assert d["enabled"] and d["host"] == ADDR and d["token"].startswith("vwk_") and d["name"] == "gpu-box"
    assert stat.S_IMODE(os.stat(cfg.path).st_mode) == 0o600


def test_pairing_with_a_wrong_code_explains(cfg, wtc, wapp):
    client = WorkerClient(cfg, transport=Bridge(wtc))
    with pytest.raises(AddressError, match="not right"):
        client.pair(ADDR, "WRONGCODE1", "pc")
    assert not cfg.load()


def test_pairing_never_leaves_tailscale(cfg):
    client = WorkerClient(cfg, transport=Dead())
    with pytest.raises(AddressError, match="not a Tailscale address"):
        client.pair("192.168.1.9", "ABCDE-FGHJK", "pc")


def test_a_job_goes_to_the_worker_and_is_noted(setup, weng):
    disp, client, bridge, local = setup
    out = disp.matte(img())
    assert out.getpixel((0, 0)) == 120 and not local.calls
    assert disp.last["where"] == "worker" and disp.last["worker"] == "gpu-box" and not disp.last["fallback"]
    assert [c[0] for c in weng.calls] == ["matte"]


def test_the_image_is_uploaded_only_when_the_worker_lacks_it(setup, weng):
    disp, client, bridge, local = setup
    pic = img()
    disp.matte(pic)
    puts1 = [c for c in bridge.calls if c[0] == "PUT"]
    disp.matte(pic)
    assert len([c for c in bridge.calls if c[0] == "PUT"]) == len(puts1) == 1


def test_each_kind_of_job_reaches_the_worker(setup, weng):
    disp, *_ = setup
    pic, mask = img(), Image.new("L", (64, 48), 255)
    assert disp.segment("k", pic, [(0.5, 0.5, 1)], None).size == (64, 48)
    assert disp.inpaint(pic, mask, 5).getpixel((0, 0)) == (1, 2, 3)
    assert disp.generative_fill(pic, mask, "hello", "bad", None, steps=9, seed=3).getpixel((0, 0)) == (9, 8, 7)
    assert disp.detect_faces(pic)[0]["box"] == (1.0, 2.0, 30.0, 40.0)
    assert disp.landmarks(pic, (1, 2, 30, 40)).shape == (478, 2)
    assert [c[0] for c in weng.calls] == ["segment", "inpaint", "generative_fill", "detect_faces", "landmarks"]


def test_clicks_on_one_photo_upload_it_once(setup, weng):
    disp, client, bridge, local = setup
    pic = img()
    for x in (0.2, 0.4, 0.6):
        disp.segment("photo1:3", pic, [(x, 0.5, 1)], None)
    assert len([c for c in bridge.calls if c[0] == "PUT"]) == 1


def test_switched_off_means_nothing_leaves_this_computer(cfg, wtc, wapp):
    client, bridge = paired_client(cfg, wtc, wapp)
    client.set_enabled(False)
    local = LocalEngine()
    disp = Dispatcher(local, client)
    before = len(bridge.calls)
    disp.matte(img()); disp.detect_faces(img()); disp.segment("k", img(), [(0.5, 0.5, 1)], None)
    assert len(bridge.calls) == before and len(local.calls) == 3
    assert not disp.last                                   # nothing to report


def test_an_unreachable_worker_falls_back_here_and_says_so(cfg):
    cfg.save({"enabled": True, "host": ADDR, "port": 5196, "token": "vwk_x", "worker_id": "w", "name": "gpu-box"})
    dead = Dead()
    local = LocalEngine()
    disp = Dispatcher(local, WorkerClient(cfg, transport=dead))
    out = disp.matte(img())
    assert out.getpixel((0, 0)) == 120 and local.calls                        # done here
    assert disp.last["fallback"] and "could not be reached" in disp.last["reason"] and "this computer" in disp.last["reason"]


def test_a_dead_worker_is_only_waited_for_once_in_a_while(cfg):
    cfg.save({"enabled": True, "host": ADDR, "port": 5196, "token": "vwk_x", "worker_id": "w", "name": "n"})
    dead = Dead()
    disp = Dispatcher(LocalEngine(), WorkerClient(cfg, transport=dead))
    for _ in range(5):
        disp.matte(img())
    assert dead.calls == 1 and disp.seq == 5               # every job still reported its own notice


def test_a_worker_without_the_tool_runs_it_here(cfg, wdir):
    app = worker_app(wdir, engine=WorkerEngine(caps=("matte",)))
    tc = TestClient(app, client=TAIL)
    client, bridge = paired_client(cfg, tc, app)
    local = LocalEngine()
    disp = Dispatcher(local, client)
    disp.inpaint(img(), Image.new("L", (64, 48), 255))
    assert local.calls and disp.last["fallback"] and "does not have" in disp.last["reason"]
    assert not [c for c in bridge.calls if c[1] == "/v1/inpaint"]
    disp.matte(img())
    assert disp.last["where"] == "worker"


def test_a_real_request_problem_is_not_hidden_by_fallback(setup, weng):
    disp, client, bridge, local = setup
    weng.boom = ModelError("Click on the thing you want to select, or draw a box around it.")
    with pytest.raises(ModelError, match="Click on the thing"):
        disp.matte(img())
    assert not local.calls


def test_nothing_is_sent_when_there_is_nothing_to_select(setup):
    disp, client, bridge, local = setup
    before = len(bridge.calls)
    disp.segment("k", img(), [], None)
    assert len(bridge.calls) == before and local.calls[0][0] == "segment"


def test_a_revoked_computer_falls_back_and_is_told_to_pair_again(setup, wapp):
    disp, client, bridge, local = setup
    for c in wapp.state.store.clients():
        wapp.state.store.revoke(c["id"])
    disp.matte(img())
    assert local.calls and "no longer trusts" in disp.last["reason"]


def test_a_different_machine_answering_is_not_trusted(setup, wdir, cfg):
    disp, client, bridge, local = setup
    cfg.save({**cfg.load(), "worker_id": "someone-else"})
    client._health = None
    disp.matte(img())
    assert local.calls and "not the worker that was paired" in disp.last["reason"]


def test_when_the_fallback_cannot_work_either_both_reasons_are_given(cfg):
    cfg.save({"enabled": True, "host": ADDR, "port": 5196, "token": "vwk_x", "worker_id": "w", "name": "n"})
    local = LocalEngine()
    local.boom = ModelError("Segment Anything is not downloaded yet.")
    disp = Dispatcher(local, WorkerClient(cfg, transport=Dead()))
    with pytest.raises(ModelError) as e:
        disp.matte(img())
    assert "could not be reached" in str(e.value) and "not downloaded" in str(e.value)


def test_a_redirect_is_never_followed(cfg):
    class Redirect(httpx.BaseTransport):
        def handle_request(self, request):
            return httpx.Response(302, headers={"location": "http://169.254.169.254/"})
    cfg.save({"enabled": True, "host": ADDR, "port": 5196, "token": "vwk_x", "worker_id": "w", "name": "n"})
    local = LocalEngine()
    disp = Dispatcher(local, WorkerClient(cfg, transport=Redirect()))
    disp.matte(img())
    assert local.calls and disp.last["fallback"]


def test_the_client_ignores_proxy_settings(cfg, monkeypatch):
    seen = {}
    real = httpx.Client

    def spy(*a, **k):
        seen.update(k)
        return real(*a, **k)
    monkeypatch.setattr(remote.httpx, "Client", spy)
    cfg.save({"enabled": True, "host": ADDR, "port": 5196, "token": "vwk_x", "worker_id": "w", "name": "n"})
    with pytest.raises(WorkerUnavailable):
        WorkerClient(cfg, transport=Dead()).health()
    assert seen["trust_env"] is False and seen["follow_redirects"] is False


def test_forgetting_tells_the_worker_and_clears_the_token(setup, wtc, wapp, cfg):
    disp, client, bridge, local = setup
    client.forget()
    assert not cfg.path.exists() and wapp.state.store.clients() == []


def test_status_counts_a_worker_as_ready(setup):
    disp, client, bridge, local = setup
    st = disp.status()
    assert st["ready"] is True and st["worker"]["reachable"] and st["worker"]["capabilities"]["segment"]
    client.set_enabled(False)
    st = disp.status()
    assert st["ready"] is False and not st["worker"]["reachable"]


# ================================================================ inside Veil
@pytest.fixture
def veil(tmp_path, wtc, wapp):
    bridge = Bridge(wtc)
    local = LocalEngine(caps=())
    app = create_app(tmp_path / "data", vision=local, worker_transport=bridge)
    tc = TestClient(app, base_url="http://localhost", client=("127.0.0.1", 50000))
    assert tc.post("/api/auth/setup", json={"username": "owner", "password": "correct horse"}).status_code == 200
    assert tc.post("/api/notice/ack").status_code == 200
    return tc, app, bridge, local


def test_owner_pairs_and_a_photo_is_selected_on_the_worker(veil, wapp, weng):
    tc, app, bridge, local = veil
    code, _ = wapp.state.store.new_code()
    r = tc.post("/api/worker/pair", json={"address": ADDR, "code": code})
    assert r.status_code == 200 and r.json()["name"] == "gpu-box"
    pid = tc.post("/api/photos", files={"file": ("a.jpg", make_jpeg((80, 60)), "image/jpeg")}).json()["id"]
    r = tc.post(f"/api/photos/{pid}/segment", json={"points": [[0.5, 0.5, 1]]})
    assert r.status_code == 200 and not local.calls
    assert [c[0] for c in weng.calls] == ["segment"]
    note = tc.get("/api/worker").json()["last"]
    assert note["where"] == "worker" and note["worker"] == "gpu-box"


def test_faces_are_found_on_the_worker(veil, wapp, weng):
    tc, app, bridge, local = veil
    tc.post("/api/worker/pair", json={"address": ADDR, "code": wapp.state.store.new_code()[0]})
    pid = tc.post("/api/photos", files={"file": ("a.jpg", make_jpeg((80, 60)), "image/jpeg")}).json()["id"]
    r = tc.post(f"/api/photos/{pid}/faces").json()
    assert len(r["faces"]) == 1 and [c[0] for c in weng.calls] == ["detect_faces"]


def test_the_token_never_reaches_a_page(veil, wapp):
    tc, app, bridge, local = veil
    tc.post("/api/worker/pair", json={"address": ADDR, "code": wapp.state.store.new_code()[0]})
    token = json.loads((app.state.data_dir / "worker.json").read_text())["token"]
    assert token.startswith("vwk_")
    for path in ("/api/worker", "/api/worker/address", "/api/vision", "/api/status"):
        assert token not in tc.get(path).text
    assert tc.get("/worker.json").status_code == 404 or token not in tc.get("/worker.json").text


def test_pairing_over_the_public_internet_is_refused_in_veil(veil):
    tc, *_ = veil
    r = tc.post("/api/worker/pair", json={"address": "8.8.8.8", "code": "ABCDE-FGHJK"})
    assert r.status_code == 400 and "Tailscale" in r.json()["detail"]


def test_only_the_owner_at_the_computer_manages_the_worker(veil, wapp):
    tc, app, bridge, local = veil
    tc.post("/api/worker/pair", json={"address": ADDR, "code": wapp.state.store.new_code()[0]})
    anon = TestClient(app, base_url="http://localhost", client=("127.0.0.1", 50001))
    for method, path, body in (("post", "/api/worker/pair", {"address": ADDR, "code": "x"}), ("post", "/api/worker/enable", {"enabled": False}),
                               ("post", "/api/worker/test", None), ("delete", "/api/worker", None), ("get", "/api/worker/address", None)):
        r = getattr(anon, method)(path, **({"json": body} if body is not None else {}))
        assert r.status_code in (401, 403), (path, r.status_code)
    assert tc.get("/api/worker").json()["enabled"]


def test_a_paired_phone_can_read_the_note_but_never_manage_the_worker(tmp_path, wtc, wapp):
    from veil.tailscale import TailscaleInfo

    class TS:
        def info(self): return TailscaleInfo(running=True, ipv4="100.64.0.20", dns_name="pc.tail1.ts.net", host_name="pc")
    app = create_app(tmp_path / "data", tailscale=TS(), vision=LocalEngine(caps=()), worker_transport=Bridge(wtc))
    owner = TestClient(app, base_url="http://localhost", client=("127.0.0.1", 50000))
    owner.post("/api/auth/setup", json={"username": "owner", "password": "correct horse"}); owner.post("/api/notice/ack")
    owner.post("/api/worker/pair", json={"address": ADDR, "code": wapp.state.store.new_code()[0]})
    owner.post("/api/phone/enable", json={"enabled": True})
    phone = TestClient(app, base_url="http://100.64.0.20:5195", client=("100.99.88.77", 40000))
    url = owner.post("/api/phone/pair").json()["url"].split("#", 1)[1]
    assert phone.post("/api/pair/redeem", json={"token": url, "name": "iPhone"}).status_code == 200
    dev = owner.get("/api/phone").json()["devices"][0]
    owner.post(f"/api/phone/devices/{dev['id']}/approve")
    r = phone.get("/api/worker")
    assert r.status_code == 200 and r.json()["enabled"] and "token" not in r.text and "vwk_" not in r.text
    for method, path, body in (("post", "/api/worker/pair", {"address": ADDR, "code": "x"}), ("post", "/api/worker/enable", {"enabled": False}),
                               ("post", "/api/worker/test", {}), ("delete", "/api/worker", None), ("get", "/api/worker/address", None)):
        r = getattr(phone, method)(path, **({"json": body} if body is not None else {}))
        assert r.status_code == 403, (path, r.status_code)
    assert owner.get("/api/worker").json()["enabled"]


def test_enable_test_and_forget(veil, wapp):
    tc, app, bridge, local = veil
    tc.post("/api/worker/pair", json={"address": ADDR, "code": wapp.state.store.new_code()[0]})
    assert tc.post("/api/worker/test").json()["ok"]
    assert tc.post("/api/worker/enable", json={"enabled": False}).status_code == 200
    assert tc.get("/api/worker").json()["enabled"] is False
    assert tc.delete("/api/worker").status_code == 200
    s = tc.get("/api/worker").json()
    assert not s["configured"] and wapp.state.store.clients() == []
    assert tc.post("/api/worker/enable", json={"enabled": True}).status_code == 400


def test_the_worker_data_file_is_not_served_as_a_web_page(veil):
    tc, *_ = veil
    for p in ("/worker.json", "/data/worker.json", "/../data/worker.json"):
        assert tc.get(p).status_code in (404, 401, 403) or "vwk_" not in tc.get(p).text


def test_the_page_may_offer_faces_and_generative_fill_when_the_worker_can_do_them(setup, cfg):
    disp, client, bridge, local = setup
    st = disp.status()
    assert st["faces_available"] and st["generative_available"]
    client.set_enabled(False)
    st = disp.status()
    assert not st["faces_available"] and not st["generative_available"]       # this computer has neither


def test_a_worker_without_face_models_does_not_enable_the_face_tools(cfg, wdir):
    app = worker_app(wdir, engine=WorkerEngine(caps=("segment", "matte", "inpaint", "faces")))      # no landmarks, no generative
    tc = TestClient(app, client=TAIL)
    client, _ = paired_client(cfg, tc, app)
    st = Dispatcher(LocalEngine(caps=()), client).status()
    assert st["ready"] and not st["faces_available"] and not st["generative_available"]
