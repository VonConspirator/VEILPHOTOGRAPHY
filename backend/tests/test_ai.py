"""The AI layer. A real HTTP server plays the language model, so the tests see exactly what Veil sends."""
import base64
import io
import json
import stat
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import httpx
import pytest
from PIL import Image

from conftest import make_jpeg
from test_phone import world  # noqa: F401  (fixture: owner at the PC, plus a phone that can be paired)
from veil.ai import actions, parser
from veil.ai.providers import AIError

PLAN = {"action": "adjust", "params": {"brightness": 15}, "reply": "Brightening it a little."}


class Fake:
    """Pretends to be Ollama, an OpenAI-style service and Anthropic, and records every request."""

    def __init__(self):
        self.requests: list[SimpleNamespace] = []
        self.reply_text = json.dumps(PLAN)
        self.status = 200
        self.thinking_only = False
        self.models = ["qwen3.5:9b", "llama3.2:latest"]
        fake = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _go(self):
                n = int(self.headers.get("content-length") or 0)
                body = json.loads(self.rfile.read(n) or b"null")
                fake.requests.append(SimpleNamespace(method=self.command, path=self.path, headers=dict(self.headers), body=body))
                if fake.status != 200:
                    self.send_response(fake.status); self.send_header("content-length", "2"); self.end_headers(); self.wfile.write(b"{}"); return
                p = self.path
                if p == "/api/chat":
                    if fake.thinking_only:  # what real Ollama 0.32 does with some qwen3 requests
                        out = {"message": {"role": "assistant", "content": "", "thinking": fake.reply_text}}
                    else:
                        out = {"message": {"role": "assistant", "content": fake.reply_text}}
                elif p == "/api/tags":
                    out = {"models": [{"name": m} for m in fake.models]}
                elif p == "/api/pull":
                    out = {"status": "success"}
                elif p.endswith("/chat/completions"):
                    out = {"choices": [{"message": {"content": fake.reply_text}}]}
                elif p.endswith("/models"):
                    out = {"data": [{"id": "m1"}]}
                elif p == "/v1/messages":
                    out = {"content": [{"type": "text", "text": fake.reply_text}]}
                else:
                    self.send_response(404); self.send_header("content-length", "2"); self.end_headers(); self.wfile.write(b"{}"); return
                data = json.dumps(out).encode()
                self.send_response(200); self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(data))); self.end_headers(); self.wfile.write(data)

            do_GET = do_POST = _go

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @property
    def local_url(self):
        return f"http://127.0.0.1:{self.port}"

    @property
    def cloud_url(self):  # a name that is NOT this computer; the fixture below routes it to the fake
        return "https://cloud.example.test"


@pytest.fixture
def fake(monkeypatch):
    f = Fake()
    real_post, real_get = httpx.post, httpx.get

    def reroute(url):
        return url.replace("https://cloud.example.test", f.local_url)

    monkeypatch.setattr(httpx, "post", lambda url, **kw: real_post(reroute(url), **kw))
    monkeypatch.setattr(httpx, "get", lambda url, **kw: real_get(reroute(url), **kw))
    yield f
    f.server.shutdown()


@pytest.fixture
def ai(world, fake):  # noqa: F811
    """The owner's client with the active model pointed at the fake Ollama, and one uploaded photo."""
    c = world.owner
    r = c.put("/api/ai/providers", json={"id": "local-ollama", "name": "Local", "kind": "ollama",
                                         "base_url": fake.local_url, "model": "qwen3-vl:8b", "vision": True})
    assert r.status_code == 200, r.text
    pid = c.post("/api/photos", files={"file": ("a.jpg", make_jpeg(size=(1600, 1200)), "image/jpeg")}).json()["id"]
    return SimpleNamespace(c=c, world=world, fake=fake, pid=pid)


def use_cloud(ai, kind="openai", key="sk-secret-123", vision=True):
    path = {"openai": "/v1", "anthropic": ""}[kind]
    r = ai.c.put("/api/ai/providers", json={"id": "cloud", "name": "Cloud", "kind": kind, "base_url": ai.fake.cloud_url + path,
                                            "model": "big-model", "vision": vision, "api_key": key})
    assert r.status_code == 200, r.text
    assert ai.c.post("/api/ai/active", json={"id": "cloud"}).status_code == 200


def command(ai, text="make it a bit brighter", client=None, **kw):
    return (client or ai.c).post("/api/ai/command", json={"photo_id": ai.pid, "text": text, **kw})


# ---------------------------------------------------------------- the parser never goes outside the action list
@pytest.mark.parametrize("model_says,expected_action", [
    ({"action": "delete_all_photos"}, "none"),
    ({"action": "os.system('rm -rf /')", "params": {}}, "none"),
    ({"action": "ADJUST", "params": {"brightness": 15}}, "adjust"),
    ({"action": "adjust", "params": {}}, "none"),
    ({"action": "straighten", "params": {}}, "none"),
    ({"action": None}, "none"),
    ({}, "none"),
])
def test_unknown_or_empty_actions_become_none(model_says, expected_action):
    assert parser.validate(model_says)["action"] == expected_action


def test_parameters_are_clamped_and_unknown_ones_dropped():
    plan = parser.validate({"action": "adjust", "params": {"brightness": 900, "contrast": "-300", "evil": "x", "warmth": float("nan")}})
    assert plan["params"] == {"brightness": 100, "contrast": -100}
    assert parser.validate({"action": "rotate90", "params": {"turns": 99}})["params"] == {"turns": 3}
    assert parser.validate({"action": "crop", "params": {"aspect": "7:13"}})["params"] == {"aspect": "free"}
    assert parser.validate({"action": "flip", "params": {"axis": "../../etc"}})["params"] == {"axis": "horizontal"}


def test_later_phase_tools_are_understood_but_marked_unavailable():
    plan = parser.validate({"action": "blur_face", "params": {}})
    assert plan["available"] is False and plan["region"] == "required"
    assert all(not actions.ACTIONS[n]["available"] for n in ("blur_face", "pixelate_face", "generate_face", "face_edit", "watermark"))
    assert parser.validate({"action": "adjust", "params": {"brightness": 5}})["available"] is True


@pytest.mark.parametrize("raw", [
    json.dumps(PLAN),
    "```json\n" + json.dumps(PLAN) + "\n```",
    "<think>hmm, brighter</think>" + json.dumps(PLAN),
    "Sure! Here you go: " + json.dumps(PLAN) + " Hope that helps.",
])
def test_messy_model_output_is_still_understood(raw):
    assert parser._extract_json(raw)["action"] == "adjust"


@pytest.mark.parametrize("raw", ["", "I cannot help with that.", "[1,2,3]", "{not json"])
def test_unreadable_model_output_is_a_clean_error(raw):
    with pytest.raises(AIError):
        parser._extract_json(raw)


# ---------------------------------------------------------------- the command endpoint, wire level
def test_a_command_reaches_the_model_as_text_only_by_default(ai):
    r = command(ai)
    assert r.status_code == 200, r.text
    plan = r.json()
    assert plan["action"] == "adjust" and plan["params"] == {"brightness": 15} and plan["available"]
    assert plan["asked"] == {"provider": "Local", "model": "qwen3-vl:8b", "local": True, "photo_sent": False}
    sent = ai.fake.requests[-1]
    assert sent.path == "/api/chat" and sent.body["model"] == "qwen3-vl:8b" and sent.body["stream"] is False
    assert all("images" not in m for m in sent.body["messages"])
    assert sent.body["messages"][-1]["content"] == "make it a bit brighter"


def test_the_command_does_not_change_the_photo(ai):
    before = ai.c.get(f"/api/photos/{ai.pid}").json()
    command(ai)
    assert ai.c.get(f"/api/photos/{ai.pid}").json() == before  # a plan is only a plan


def test_a_local_model_can_look_at_the_photo_and_gets_no_metadata(ai):
    r = command(ai, look_at_photo=True)
    assert r.status_code == 200 and r.json()["asked"]["photo_sent"] is True
    img_b64 = ai.fake.requests[-1].body["messages"][-1]["images"][0]
    raw = base64.b64decode(img_b64)
    img = Image.open(io.BytesIO(raw))
    assert img.format == "JPEG" and max(img.size) <= 1024  # shrunk
    assert len(img.getexif()) == 0 and b"TestCam" not in raw and b"GPS" not in raw  # no camera or location details


def test_a_model_that_cannot_see_is_never_sent_a_photo(ai):
    ai.c.put("/api/ai/providers", json={"id": "local-ollama", "kind": "ollama", "base_url": ai.fake.local_url,
                                        "model": "llama3.2:latest", "vision": False})
    n = len(ai.fake.requests)
    assert command(ai, look_at_photo=True).status_code == 400
    assert len(ai.fake.requests) == n


def test_a_command_needs_a_login_and_the_adult_notice(world, fake, tmp_path):
    from veil.main import create_app
    from fastapi.testclient import TestClient
    anon = TestClient(create_app(tmp_path / "d2"), base_url="http://localhost", client=("127.0.0.1", 1))
    assert anon.post("/api/ai/command", json={"photo_id": "0" * 16, "text": "x"}).status_code == 401
    anon.post("/api/auth/setup", json={"username": "owner", "password": "correct horse"})
    assert anon.post("/api/ai/command", json={"photo_id": "0" * 16, "text": "x"}).status_code == 403  # notice not confirmed


@pytest.mark.parametrize("text,code", [("", 400), ("   ", 400), ("x" * 501, 400)])
def test_empty_and_huge_commands_are_refused_before_asking_the_model(ai, text, code):
    n = len(ai.fake.requests)
    assert command(ai, text=text).status_code == code
    assert len(ai.fake.requests) == n


def test_a_command_for_a_missing_photo(ai):
    r = ai.c.post("/api/ai/command", json={"photo_id": "0123456789abcdef", "text": "brighter"})
    assert r.status_code == 404


def test_an_answer_that_ollama_puts_in_the_thinking_field_is_still_read(ai):
    ai.fake.thinking_only = True
    assert command(ai).json()["action"] == "adjust"
    assert "format" not in ai.fake.requests[-1].body  # the option that triggers it is not used


def test_a_model_that_answers_nonsense_gives_a_plain_error(ai):
    ai.fake.reply_text = "I am a teapot."
    r = command(ai)
    assert r.status_code == 400 and "reword" in r.json()["detail"].lower()


def test_a_prompt_injected_reply_still_cannot_do_anything_unlisted(ai):
    ai.fake.reply_text = json.dumps({"action": "export_to_internet", "params": {"url": "http://evil.test"}, "reply": "done"})
    plan = command(ai).json()
    assert plan["action"] == "none" and "params" not in plan


# ---------------------------------------------------------------- the cloud rule: text only unless you say so, each time
def test_cloud_models_get_text_only_by_default(ai):
    use_cloud(ai)
    r = command(ai)
    assert r.status_code == 200 and r.json()["asked"]["local"] is False and r.json()["asked"]["photo_sent"] is False
    body = ai.fake.requests[-1].body
    assert "image_url" not in json.dumps(body)


def test_sending_a_photo_to_the_cloud_needs_confirmation_every_time(ai):
    use_cloud(ai)
    n = len(ai.fake.requests)
    r = command(ai, look_at_photo=True)  # asked for the photo but did not confirm
    assert r.status_code == 400 and "confirm" in r.json()["detail"].lower()
    assert len(ai.fake.requests) == n  # nothing left the computer
    ok = command(ai, look_at_photo=True, confirm_cloud_photo=True)
    assert ok.status_code == 200 and ok.json()["asked"]["photo_sent"] is True
    assert "image_url" in json.dumps(ai.fake.requests[-1].body)
    # the permission was for that one request: asking again without it is refused again
    assert command(ai, look_at_photo=True).status_code == 400
    assert not (ai.world.data / "ai.json").read_text().count("confirm")  # and it was never saved


def test_confirming_without_asking_for_the_photo_sends_no_photo(ai):
    use_cloud(ai)
    r = command(ai, confirm_cloud_photo=True)
    assert r.json()["asked"]["photo_sent"] is False
    assert "image_url" not in json.dumps(ai.fake.requests[-1].body)


def test_a_phone_can_never_send_a_photo_to_the_cloud(ai):
    w = ai.world
    use_cloud(ai)
    w.enable(); w.pair()
    n = len(ai.fake.requests)
    r = command(ai, client=w.phone, look_at_photo=True, confirm_cloud_photo=True)
    assert r.status_code == 403
    assert len(ai.fake.requests) == n
    assert command(ai, client=w.phone).status_code == 200  # plain text commands from the phone are fine


def test_anthropic_and_openai_wire_formats(ai):
    use_cloud(ai, "openai")
    assert command(ai, look_at_photo=True, confirm_cloud_photo=True).status_code == 200
    req = ai.fake.requests[-1]
    assert req.path == "/v1/chat/completions" and req.headers["Authorization"] == "Bearer sk-secret-123"
    assert req.body["messages"][-1]["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    use_cloud(ai, "anthropic", key="sk-ant-xyz")
    assert command(ai, look_at_photo=True, confirm_cloud_photo=True).status_code == 200
    req = ai.fake.requests[-1]
    assert req.path == "/v1/messages" and req.headers["x-api-key"] == "sk-ant-xyz" and req.headers["anthropic-version"]
    assert req.body["system"] and req.body["messages"][0]["content"][0]["source"]["media_type"] == "image/jpeg"


# ---------------------------------------------------------------- settings and keys
def test_api_keys_are_never_sent_back_and_the_file_is_private(ai):
    use_cloud(ai, key="sk-very-secret")
    for url in ["/api/ai/settings", "/api/ai"]:
        assert "sk-very-secret" not in ai.c.get(url).text
    s = ai.c.get("/api/ai/settings").json()
    assert next(p for p in s["providers"] if p["id"] == "cloud")["has_key"] is True
    f = ai.world.data / "ai.json"
    assert "sk-very-secret" in f.read_text() and stat.S_IMODE(f.stat().st_mode) == 0o600


def test_leaving_the_key_blank_keeps_the_saved_key(ai):
    use_cloud(ai, key="sk-keep-me")
    r = ai.c.put("/api/ai/providers", json={"id": "cloud", "kind": "openai", "base_url": ai.fake.cloud_url + "/v1", "model": "other", "api_key": ""})
    assert r.status_code == 200
    command(ai)
    assert ai.fake.requests[-1].headers["Authorization"] == "Bearer sk-keep-me"


def test_an_api_key_will_not_be_sent_over_plain_http_to_another_machine(ai):
    r = ai.c.put("/api/ai/providers", json={"id": "bad", "kind": "openai", "base_url": "http://example.test/v1", "model": "m", "api_key": "sk-x"})
    assert r.status_code == 400 and "https" in r.json()["detail"]
    assert not any(p["id"] == "bad" for p in ai.c.get("/api/ai/settings").json()["providers"])


@pytest.mark.parametrize("bad", [
    {"id": "Bad Id!", "kind": "openai", "base_url": "https://x.test", "model": "m"},
    {"id": "ok", "kind": "skynet", "base_url": "https://x.test", "model": "m"},
    {"id": "ok", "kind": "openai", "base_url": "file:///etc/passwd", "model": "m"},
    {"id": "ok", "kind": "openai", "base_url": "https://x.test", "model": ""},
])
def test_bad_provider_settings_are_refused(ai, bad):
    assert ai.c.put("/api/ai/providers", json=bad).status_code == 400


def test_active_provider_switching_and_removal(ai):
    use_cloud(ai)
    assert ai.c.get("/api/ai").json()["active"]["id"] == "cloud"
    assert ai.c.delete("/api/ai/providers/cloud").status_code == 200
    assert ai.c.get("/api/ai").json()["active"]["id"] == "local-ollama"  # falls back
    assert ai.c.delete("/api/ai/providers/local-ollama").status_code == 400  # the last one stays
    assert ai.c.post("/api/ai/active", json={"id": "nope"}).status_code == 400


def test_only_the_owner_at_the_computer_can_change_ai_settings(ai):
    w = ai.world
    w.enable(); w.pair()
    body = {"id": "x", "kind": "openai", "base_url": "https://x.test", "model": "m"}
    for call in [lambda c: c.get("/api/ai/settings"), lambda c: c.put("/api/ai/providers", json=body),
                 lambda c: c.post("/api/ai/active", json={"id": "x"}), lambda c: c.delete("/api/ai/providers/local-ollama"),
                 lambda c: c.get("/api/ai/models"), lambda c: c.post("/api/ai/models/pull", json={"name": "qwen3.5:2b"}),
                 lambda c: c.post("/api/ai/test")]:
        assert call(w.phone).status_code == 403
    assert w.phone.get("/api/ai").status_code == 200  # the phone may see which model is in use (no keys)


# ---------------------------------------------------------------- provider failures
@pytest.mark.parametrize("status,word", [(401, "api key"), (404, "model"), (429, "rate"), (500, "error")])
def test_provider_errors_become_plain_messages(ai, status, word):
    ai.fake.status = status
    r = command(ai)
    assert r.status_code == 400 and word in r.json()["detail"].lower()


def test_an_unreachable_model_says_so(ai):
    ai.c.put("/api/ai/providers", json={"id": "local-ollama", "kind": "ollama", "base_url": "http://127.0.0.1:9", "model": "m"})
    r = command(ai)
    assert r.status_code == 400 and "could not reach" in r.json()["detail"].lower()


# ---------------------------------------------------------------- model manager
def test_model_list_shows_installed_models_against_the_tiers(ai):
    m = ai.c.get("/api/ai/models").json()
    assert m["reachable"] and "qwen3.5:9b" in m["installed"] and m["can_download"] is True
    tiers = {t["tier"]: t for t in m["tiers"]}
    assert tiers["Recommended"]["name"] == "qwen3.5:9b" and tiers["Recommended"]["installed"] is True
    assert tiers["Minimum"]["installed"] is False
    assert all(t["license"] == "Apache-2.0" for t in m["tiers"])


def test_downloads_are_limited_to_the_listed_models_and_to_local_ollama(ai):
    n = len(ai.fake.requests)
    assert ai.c.post("/api/ai/models/pull", json={"name": "some/other-model:latest"}).status_code == 400
    assert ai.c.post("/api/ai/models/pull", json={"name": "llama3.2"}).status_code == 400
    assert len(ai.fake.requests) == n  # nothing was requested
    use_cloud(ai)
    assert ai.c.post("/api/ai/models/pull", json={"name": "qwen3.5:2b"}).status_code == 400
    assert ai.c.get("/api/ai/models").json()["can_download"] is False


def test_a_listed_model_downloads_through_ollama_with_progress(ai):
    import time
    r = ai.c.post("/api/ai/models/pull", json={"name": "qwen3.5:2b"})
    assert r.status_code == 200 and r.json()["name"] == "qwen3.5:2b"
    for _ in range(50):
        d = ai.c.get("/api/ai/models").json()["download"]
        if not d["running"]:
            break
        time.sleep(0.1)
    assert d["running"] is False and d["error"] == ""
    pulls = [q for q in ai.fake.requests if q.path == "/api/pull"]
    assert len(pulls) == 1 and pulls[0].body["model"] == "qwen3.5:2b"


def test_a_fresh_install_defaults_to_qwen35_9b(ready):
    a = ready.get("/api/ai").json()["active"]
    assert a["model"] == "qwen3.5:9b" and a["local"] is True and a["vision"] is True
