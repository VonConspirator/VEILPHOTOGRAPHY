from conftest import make_jpeg


def test_first_run_setup_then_closed(client):
    assert client.get("/api/status").json()["setup_needed"] is True
    assert client.post("/api/auth/setup", json={"username": "a", "password": "short"}).status_code == 400
    r = client.post("/api/auth/setup", json={"username": "owner", "password": "correct horse"})
    assert r.status_code == 200
    cookie = r.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=strict" in cookie
    # a second setup must be refused (nobody can take over an existing install)
    assert client.post("/api/auth/setup", json={"username": "x", "password": "another pass 1"}).status_code == 409


def test_photo_routes_need_login(client):
    for method, url in [("get", "/api/photos"), ("post", "/api/notice/ack")]:
        assert getattr(client, method)(url).status_code == 401
    files = {"file": ("a.jpg", make_jpeg(), "image/jpeg")}
    assert client.post("/api/photos", files=files).status_code == 401


def test_adult_notice_required_before_photos(client):
    client.post("/api/auth/setup", json={"username": "owner", "password": "correct horse"})
    assert client.get("/api/photos").status_code == 403
    client.post("/api/notice/ack")
    assert client.get("/api/photos").status_code == 200


def test_wrong_password_and_lockout(client):
    client.post("/api/auth/setup", json={"username": "owner", "password": "correct horse"})
    client.post("/api/auth/logout")
    for _ in range(5):
        assert client.post("/api/auth/login", json={"username": "owner", "password": "nope nope"}).status_code == 401
    # locked out now, even with the right password
    assert client.post("/api/auth/login", json={"username": "owner", "password": "correct horse"}).status_code == 429


def test_logout_ends_session(ready):
    ready.post("/api/auth/logout")
    assert ready.get("/api/photos").status_code == 401


def test_unknown_host_refused(ready):
    assert ready.get("/api/status", headers={"host": "evil.example.com"}).status_code == 421


def test_cross_site_post_refused(ready):
    r = ready.post("/api/auth/logout", headers={"origin": "http://evil.example.com"})
    assert r.status_code == 403
    assert ready.get("/api/photos").status_code == 200  # still logged in: the forged request did nothing
