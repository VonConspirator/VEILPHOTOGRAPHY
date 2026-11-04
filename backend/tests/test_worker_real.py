"""The worker with the real models over a real socket: what comes back through the worker is exactly what the engine makes.
Skipped when the models or libraries are not installed."""
import socket
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import uvicorn
from PIL import Image

from test_faces import PORTRAITS, REAL_NEEDS
from test_vision import MODELS_DIR, scene
from veil.remote import Dispatcher, WorkerClient, WorkerConfig
from veil.vision import registry
from veil.vision.engine import Engine
from veil.worker.server import create_app

try:
    import diffusers, mediapipe  # noqa: F401
    LIBS = True
except ImportError:
    LIBS = False
HAVE = LIBS and Engine(MODELS_DIR).libraries_ok() and all(registry.installed(MODELS_DIR, n) for n in REAL_NEEDS)
real = pytest.mark.skipif(not HAVE, reason="the models and libraries are not installed")


class LocalOnly:
    root = MODELS_DIR

    def __init__(self, eng): self.eng = eng
    def __getattr__(self, n): return getattr(self.eng, n)


@pytest.fixture(scope="module")
def rig(tmp_path_factory):
    eng = Engine(MODELS_DIR)
    wdir, cdir = tmp_path_factory.mktemp("worker"), tmp_path_factory.mktemp("client")
    app = create_app(wdir, engine=eng, name="real-gpu")
    sock = socket.socket(); sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); sock.bind(("127.0.0.1", 0))
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning"))
    t = threading.Thread(target=lambda: server.run(sockets=[sock]), daemon=True)
    t.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.1)
    assert server.started
    client = WorkerClient(WorkerConfig(cdir))
    client.pair(f"127.0.0.1:{sock.getsockname()[1]}", app.state.store.new_code()[0], "test-pc")
    disp = Dispatcher(LocalOnly(eng), client)
    yield eng, disp, client
    server.should_exit = True
    t.join(10)
    eng.release()


@real
def test_selection_matte_and_fill_match_the_engine_exactly(rig):
    eng, disp, client = rig
    img, _, disc = scene()
    m_direct = np.asarray(eng.segment("k", img, [(0.5, 0.5, 1)], None))
    m_worker = np.asarray(disp.segment("k", img, [(0.5, 0.5, 1)], None))
    assert disp.last["where"] == "worker" and np.array_equal(m_direct, m_worker)
    assert np.array_equal(np.asarray(eng.matte(img)), np.asarray(disp.matte(img)))
    hole = Image.new("L", img.size, 0); hole.paste(255, (280, 200, 360, 280))
    a, b = np.asarray(eng.inpaint(img, hole)), np.asarray(disp.inpaint(img, hole))
    assert disp.last["where"] == "worker"
    assert np.abs(a.astype(int) - b.astype(int)).max() <= 2          # LaMa itself differs by one level between two identical runs
    assert (np.asarray(m_worker) > 127)[disc].mean() > 0.9


@real
def test_faces_and_landmarks_match(rig):
    eng, disp, client = rig
    p = eng.generative_fill(Image.new("RGB", (1024, 1024), (128, 128, 128)), Image.new("L", (1024, 1024), 255), PORTRAITS["plain"],
                            "cartoon, painting, deformed, blurry, child, text, watermark", grow=0, strength=1.0, seed=5, context=0.0)
    direct, via = eng.detect_faces(p), disp.detect_faces(p)
    assert disp.last["where"] == "worker" and len(direct) == 1 == len(via)
    assert np.allclose(direct[0]["box"], via[0]["box"]) and abs(direct[0]["score"] - via[0]["score"]) < 1e-9
    lm_direct, lm_via = eng.landmarks(p, direct[0]["box"]), disp.landmarks(p, direct[0]["box"])
    assert lm_via.shape == (478, 2) and np.allclose(lm_direct, lm_via)


@real
def test_generative_fill_runs_on_the_worker(rig):
    eng, disp, client = rig
    base = Image.new("RGB", (512, 512), (120, 130, 140))
    area = Image.new("L", (512, 512), 0); area.paste(255, (150, 150, 360, 360))
    out = disp.generative_fill(base, area, "a red apple on a table, photograph", "", None, steps=8, seed=3)
    assert disp.last["where"] == "worker" and disp.last["job"] == "generative fill"
    a = np.asarray(out).astype(int)
    assert np.abs(a[170:340, 170:340] - np.asarray(base).astype(int)[170:340, 170:340]).mean() > 5          # the area was repainted
    assert np.array_equal(a[:100, :100], np.asarray(base)[:100, :100])                                      # far from the area: untouched
