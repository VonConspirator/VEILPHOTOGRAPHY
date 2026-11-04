"""Phone pairing and the network rules. Every refusal is proven by a request that is actually refused."""
import json
import re
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from conftest import make_jpeg
from veil import netguard
from veil.main import create_app
from veil.tailscale import TailscaleInfo

TS_IP = "100.64.0.20"
TS_NAME = "veil-pc.tail1234.ts.net"


class FakeTailscale:
    def __init__(self, running=True):
        self.running = running

    def info(self):
        if not self.running:
            return TailscaleInfo(problem="Tailscale is not connected. Open Tailscale and sign in.")
        return TailscaleInfo(running=True, ipv4=TS_IP, dns_name=TS_NAME, host_name="veil-pc")


def client_for(app, peer, host="localhost"):
    return TestClient(app, base_url=f"http://{host}", client=(peer, 40000))


@pytest.fixture
def world(tmp_path):
    ts = FakeTailscale()
    app = create_app(tmp_path / "data", tailscale=ts)
    w = SimpleNamespace(app=app, ts=ts, data=tmp_path / "data")
    w.owner = client_for(app, "127.0.0.1")
    assert w.owner.post("/api/auth/setup", json={"username": "owner", "password": "correct horse"}).status_code == 200
    assert w.owner.post("/api/notice/ack").status_code == 200
    w.new_phone = lambda peer="100.99.88.77": client_for(app, peer, host=f"{TS_IP}:5195")
    w.phone = w.new_phone()

    def enable(on=True):
        assert w.owner.post("/api/phone/enable", json={"enabled": on}).status_code == 200

    def token():
        r = w.owner.post("/api/phone/pair")
        assert r.status_code == 200, r.text
        return r.json()["url"].split("#", 1)[1]

    def pair(phone=None, approve=True, name="Test phone"):
        phone = phone or w.phone
        r = phone.post("/api/pair/redeem", json={"token": token(), "name": name})
        assert r.status_code == 200, r.text
        dev = w.owner.get("/api/phone").json()["devices"][0]
        if approve:
            assert w.owner.post(f"/api/phone/devices/{dev['id']}/approve").status_code == 200
        return dev["id"]

    w.enable, w.token, w.pair = enable, token, pair
    return w


# ---------------------------------------------------------------- who may connect at all
@pytest.mark.parametrize("peer", ["192.168.1.50", "10.0.0.5", "172.16.4.4", "8.8.8.8", "100.63.255.255", "100.128.0.1",
                                  "2001:db8::1", "fe80::1", "not-an-address"])
def test_home_wifi_and_the_internet_are_always_refused(world, peer):
    world.enable()
    stranger = client_for(world.app, peer, host=f"{TS_IP}:5195")
    for url in ["/api/status", "/", "/pair.html", "/api/photos"]:
        assert stranger.get(url).status_code == 403, (peer, url)
    assert stranger.post("/api/pair/redeem", json={"token": world.token()}).status_code == 403


def test_a_stolen_phone_cookie_does_not_work_from_the_wrong_network(world):
    world.enable(); world.pair()
    cookie = world.phone.cookies.get("veil_device")
    assert cookie
    thief = client_for(world.app, "203.0.113.9", host=f"{TS_IP}:5195")
    thief.cookies.set("veil_device", cookie)
    assert thief.get("/api/photos").status_code == 403


def test_tailscale_peers_are_refused_while_phone_access_is_off(world):
    for url in ["/api/status", "/", "/pair.html"]:
        assert world.phone.get(url).status_code == 403
    assert world.phone.post("/api/pair/redeem", json={"token": "x"}).status_code == 403


def test_forged_headers_cannot_make_a_phone_look_local(world):
    world.enable()
    sneaky = {"x-forwarded-for": "127.0.0.1", "x-real-ip": "127.0.0.1", "forwarded": "for=127.0.0.1"}
    assert world.phone.get("/api/phone", headers=sneaky).status_code == 403
    assert world.phone.post("/api/phone/enable", json={"enabled": False}, headers=sneaky).status_code == 403
    assert world.phone.get("/api/status", headers=sneaky).json()["scope"] == "remote"
    lan = client_for(world.app, "192.168.1.50")
    assert lan.get("/api/status", headers=sneaky).status_code == 403
    assert world.phone.get("/api/status", headers={"host": "localhost"}).status_code == 421  # a rebinding page can't pick the Host
    assert world.phone.get("/api/status", headers={"host": "evil.example.com"}).status_code == 421
    assert world.phone.get("/api/status", headers={"host": TS_NAME}).status_code == 200  # the real Tailscale name is fine


@pytest.mark.parametrize("peer,expected", [
    ("127.0.0.1", "local"), ("::1", "local"), ("::ffff:127.0.0.1", "local"),
    ("100.64.0.1", "tailnet"), ("100.127.255.254", "tailnet"), ("fd7a:115c:a1e0::1", "tailnet"), ("::ffff:100.100.1.1", "tailnet"),
    ("100.63.0.1", "other"), ("100.128.0.1", "other"), ("192.168.0.2", "other"), ("::2", "other"), ("", "other"), (None, "other"),
    ("999.1.1.1", "other"),
])
def test_classify(peer, expected):
    assert netguard.classify(peer) == expected


# ---------------------------------------------------------------- the pairing flow
def test_full_pairing_flow(world):
    world.enable()
    r = world.owner.post("/api/phone/pair"); body = r.json()
    assert re.fullmatch(rf"http://{re.escape(TS_IP)}:5195/pair\.html#[\w-]{{40,}}", body["url"])
    assert body["qr"].startswith("data:image/svg+xml") and body["expires_in"] == 300
    token = body["url"].split("#", 1)[1]

    st = world.phone.get("/api/status").json()          # a phone that has not paired yet
    assert st == {"setup_needed": False, "logged_in": False, "scope": "remote", "device_state": "none", "adult_notice_acknowledged": True}
    assert world.phone.get("/api/photos").status_code == 401

    red = world.phone.post("/api/pair/redeem", json={"token": token, "name": "Ana's phone"})
    assert red.status_code == 200 and red.json() == {"state": "pending"}
    cookie = red.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=strict" in cookie
    assert world.phone.get("/api/pair/state").json() == {"state": "pending"}
    assert world.phone.get("/api/photos").status_code == 401        # waiting is not enough

    dev = world.owner.get("/api/phone").json()["devices"][0]
    assert dev["name"] == "Ana's phone" and dev["status"] == "pending"
    world.owner.post(f"/api/phone/devices/{dev['id']}/approve")
    assert world.phone.get("/api/pair/state").json() == {"state": "approved"}
    assert world.phone.get("/api/status").json()["logged_in"] is True

    up = world.phone.post("/api/photos", files={"file": ("p.jpg", make_jpeg(gps=False), "image/jpeg")})
    assert up.status_code == 200                                    # the full editor works from the phone
    pid = up.json()["id"]
    assert world.phone.post(f"/api/photos/{pid}/edit", json={"op": "adjust", "params": {"brightness": 10}}).status_code == 200
    assert [p["id"] for p in world.owner.get("/api/photos").json()] == [pid]   # same library as the computer


def test_a_paired_phone_cannot_manage_anything(world):
    world.enable(); dev = world.pair()
    for method, url, body in [
        ("get", "/api/phone", None), ("post", "/api/phone/pair", None), ("post", "/api/phone/enable", {"enabled": False}),
        ("post", f"/api/phone/devices/{dev}/approve", None), ("delete", f"/api/phone/devices/{dev}", None),
        ("post", "/api/auth/login", {"username": "owner", "password": "correct horse"}),
        ("post", "/api/auth/setup", {"username": "me", "password": "another pass 1"}), ("post", "/api/notice/ack", None),
    ]:
        r = getattr(world.phone, method)(url, **({"json": body} if body else {}))
        assert r.status_code == 403, (method, url, r.status_code)
        assert "computer" in r.json()["detail"].lower()
    assert world.owner.get("/api/phone").json()["devices"][0]["status"] == "approved"   # nothing changed


def test_the_computer_cannot_be_paired_as_a_phone(world):
    world.enable()
    assert world.owner.post("/api/pair/redeem", json={"token": world.token()}).status_code == 403


def test_owner_login_cookie_is_useless_on_a_phone_and_phone_cookie_on_the_computer(world):
    world.enable(); world.pair()
    leaked = world.phone.__class__(world.app, base_url=f"http://{TS_IP}:5195", client=("100.99.88.99", 1))
    leaked.cookies.set("veil_session", world.owner.cookies.get("veil_session"))
    assert leaked.get("/api/photos").status_code == 401           # a session cookie does not authenticate a phone
    local_visitor = client_for(world.app, "127.0.0.1")
    local_visitor.cookies.set("veil_device", world.phone.cookies.get("veil_device"))
    assert local_visitor.get("/api/photos").status_code == 401    # and a phone cookie does not log in at the computer


# ---------------------------------------------------------------- tokens
def test_a_qr_code_works_once(world):
    world.enable(); token = world.token()
    assert world.phone.post("/api/pair/redeem", json={"token": token}).status_code == 200
    second = world.new_phone("100.99.88.78")
    r = second.post("/api/pair/redeem", json={"token": token})
    assert r.status_code == 400 and "expired or was already used" in r.json()["detail"]
    assert second.get("/api/pair/state").json() == {"state": "none"}


def test_wrong_and_empty_tokens_are_refused(world):
    world.enable()
    for bad in ["", "x", "A" * 43, "../etc/passwd", "null"]:
        assert world.phone.post("/api/pair/redeem", json={"token": bad}).status_code == 400
    assert world.owner.get("/api/phone").json()["devices"] == []


def test_a_qr_code_expires_after_five_minutes(world):
    world.enable(); token = world.token()
    path = world.data / "phone.json"
    db = json.loads(path.read_text()); db["tokens"] = {k: 1.0 for k in db["tokens"]}; path.write_text(json.dumps(db))
    assert world.phone.post("/api/pair/redeem", json={"token": token}).status_code == 400


def test_guessing_is_rate_limited_even_for_a_real_token(world):
    world.enable(); token = world.token()
    for _ in range(8):
        assert world.phone.post("/api/pair/redeem", json={"token": "wrong"}).status_code == 400
    r = world.phone.post("/api/pair/redeem", json={"token": token})
    assert r.status_code == 400 and "Too many tries" in r.json()["detail"]
    assert world.owner.get("/api/phone").json()["devices"] == []


def test_only_hashes_are_stored(world):
    world.enable(); token = world.token()
    world.phone.post("/api/pair/redeem", json={"token": token})
    stored = (world.data / "phone.json").read_text()
    assert token not in stored and world.phone.cookies.get("veil_device") not in stored
    assert (world.data / "phone.json").stat().st_mode & 0o777 == 0o600


# ---------------------------------------------------------------- owner controls
def test_pending_phones_get_nothing_and_denied_phones_stay_out(world):
    world.enable(); dev = world.pair(approve=False)
    assert world.phone.get("/api/photos").status_code == 401
    world.owner.post(f"/api/phone/devices/{dev}/deny")
    assert world.phone.get("/api/pair/state").json() == {"state": "denied"}
    assert world.phone.get("/api/photos").status_code == 401
    assert world.owner.post(f"/api/phone/devices/{dev}/approve").status_code == 400   # a denied phone can't be approved later


def test_a_phone_waiting_too_long_is_dropped(world):
    world.enable(); world.pair(approve=False)
    path = world.data / "phone.json"
    db = json.loads(path.read_text())
    for d in db["devices"].values():
        d["created"] -= 3600
    path.write_text(json.dumps(db))
    assert world.phone.get("/api/pair/state").json() == {"state": "denied"}


def test_removing_a_phone_cuts_it_off_at_once(world):
    world.enable(); dev = world.pair()
    assert world.phone.get("/api/photos").status_code == 200
    assert world.owner.delete(f"/api/phone/devices/{dev}").status_code == 200
    assert world.phone.get("/api/photos").status_code == 401
    assert world.phone.get("/api/pair/state").json() == {"state": "none"}


def test_switching_phone_access_off_locks_every_phone_and_kills_open_qr_codes(world):
    world.enable(); world.pair(); open_token = world.token()
    world.enable(False)
    assert world.phone.get("/api/photos").status_code == 403
    world.enable(True)
    assert world.phone.post("/api/pair/redeem", json={"token": open_token}).status_code == 400   # the old QR code died
    assert world.phone.get("/api/photos").status_code == 200                                        # the paired phone is back


def test_a_phone_can_unpair_itself(world):
    world.enable(); world.pair()
    assert world.phone.post("/api/pair/forget").status_code == 200
    assert world.phone.get("/api/photos").status_code == 401
    assert world.owner.get("/api/phone").json()["devices"] == []


def test_only_the_logged_in_owner_manages_phones(world):
    stranger = client_for(world.app, "127.0.0.1")           # at the computer, but not logged in
    assert stranger.get("/api/phone").status_code == 403
    assert stranger.post("/api/phone/enable", json={"enabled": True}).status_code == 403
    assert stranger.post("/api/phone/pair").status_code == 403


def test_no_qr_code_while_phone_access_is_off_or_tailscale_is_down(world):
    assert world.owner.post("/api/phone/pair").status_code == 400            # access is off
    world.enable(); world.ts.running = False
    r = world.owner.post("/api/phone/pair")
    assert r.status_code == 409 and "Tailscale" in r.json()["detail"]


def test_limits_on_waiting_phones(world):
    world.enable()
    for i in range(5):
        p = world.new_phone(f"100.99.1.{i + 1}")
        assert p.post("/api/pair/redeem", json={"token": world.token(), "name": f"p{i}"}).status_code == 200
    assert world.owner.post("/api/phone/pair").status_code == 400            # six waiting is too many


def test_cross_site_post_to_a_phone_is_refused(world):
    world.enable()
    r = world.phone.post("/api/pair/redeem", json={"token": world.token()}, headers={"origin": "http://evil.example.com"})
    assert r.status_code == 403


def test_phone_status_shows_tailscale_problems_to_the_owner(world):
    world.ts.running = False
    t = world.owner.get("/api/phone").json()["tailscale"]
    assert t["running"] is False and "Tailscale" in t["problem"]
