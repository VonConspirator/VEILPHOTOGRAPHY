import io

from PIL import Image

from conftest import make_jpeg


def upload(client, data, name="a.jpg"):
    return client.post("/api/photos", files={"file": (name, data, "application/octet-stream")})


def test_upload_keeps_original_bytes_and_applies_orientation(ready, tmp_path):
    data = make_jpeg(size=(60, 40), orientation=6)  # camera says: rotate 90 degrees
    r = upload(ready, data)
    assert r.status_code == 200
    meta = r.json()
    assert (meta["width"], meta["height"]) == (40, 60)  # rotated upright
    originals = list((tmp_path / "data" / "photos" / meta["id"]).glob("original.*"))
    assert len(originals) == 1 and originals[0].read_bytes() == data  # untouched
    img = Image.open(io.BytesIO(ready.get(f"/api/photos/{meta['id']}/image").content))
    assert img.size == (40, 60)


def test_rejects_non_images_and_odd_formats(ready):
    assert upload(ready, b"this is not an image").status_code == 400
    gif = io.BytesIO()
    Image.new("RGB", (4, 4)).save(gif, "GIF")
    assert upload(ready, gif.getvalue(), "a.gif").status_code == 400


def test_bad_ids_cannot_escape_the_photo_folder(ready, tmp_path):
    for bad in ["..", "../auth.json", "0123456789abcdeg", "x"]:
        assert ready.get(f"/api/photos/{bad}").status_code in (404, 400)
    assert ready.delete("/api/photos/..%2Fauth.json").status_code in (404, 400, 405)
    assert (tmp_path / "data" / "auth.json").exists()  # nothing outside the photo folder was touched
    for url in ["/..%2Fauth.json", "/../auth.json", "/%2e%2e/auth.json", "/..%2f..%2fauth.json"]:
        r = ready.get(url)
        assert b"scrypt" not in r.content and b'"hash"' not in r.content, url  # the page server never serves the account file


def test_undo_redo_and_redo_is_discarded_by_a_new_edit(ready, tmp_path):
    client = ready
    meta = upload(client, make_jpeg()).json()
    pid = meta["id"]
    store = client.app.state.photos
    cur = store.open_current(pid)
    store.add_version(pid, cur.point(lambda v: min(255, v + 30)), "Lighten", {"amount": 30})
    store.add_version(pid, cur.point(lambda v: max(0, v - 30)), "Darken", {"amount": -30})
    m = client.get(f"/api/photos/{pid}").json()
    assert [v["label"] for v in m["versions"]] == ["Original", "Lighten", "Darken"] and m["cursor"] == 2

    assert client.post(f"/api/photos/{pid}/undo").json()["cursor"] == 1
    assert client.post(f"/api/photos/{pid}/undo").json()["cursor"] == 0
    assert client.post(f"/api/photos/{pid}/undo").json()["cursor"] == 0  # can't go below the original
    assert client.post(f"/api/photos/{pid}/redo").json()["cursor"] == 1

    store.add_version(pid, cur.point(lambda v: v), "Crop", {})  # new edit after an undo
    m = client.get(f"/api/photos/{pid}").json()
    assert [v["label"] for v in m["versions"]] == ["Original", "Lighten", "Crop"]
    assert client.post(f"/api/photos/{pid}/redo").json()["cursor"] == 2  # nothing left to redo
    files = sorted(p.name for p in (tmp_path / "data" / "photos" / pid).glob("v*.png"))
    assert files == ["v0000.png", "v0001.png", "v0002.png"]  # the discarded Darken file is gone, its number is reused
    crop = Image.open(tmp_path / "data" / "photos" / pid / "v0002.png")
    assert list(crop.getdata()) == list(cur.getdata())  # v0002 holds the new edit, not the discarded Darken


def test_original_view_never_changes(ready):
    meta = upload(ready, make_jpeg()).json()
    pid = meta["id"]
    store = ready.app.state.photos
    before = ready.get(f"/api/photos/{pid}/image?version=original").content
    store.add_version(pid, Image.new("RGB", (10, 10), (1, 2, 3)), "Crop", {})
    assert ready.get(f"/api/photos/{pid}/image?version=original").content == before
    assert (ready.get(f"/api/photos/{pid}").json()["width"]) == 10


def test_export_strips_all_metadata_but_keeps_colour_profile(ready):
    src = make_jpeg()
    assert b"Exif" in src and b"TestCam" in src  # the test photo really has metadata
    pid = upload(ready, src).json()["id"]
    for fmt in ("jpeg", "png", "webp"):
        r = ready.post(f"/api/photos/{pid}/export", json={"format": fmt, "quality": 90})
        assert r.status_code == 200, fmt
        out = Image.open(io.BytesIO(r.content))
        assert len(out.getexif()) == 0, fmt
        assert b"TestCam" not in r.content and b"GPS" not in r.content, fmt
        assert out.info.get("icc_profile"), fmt
    assert ready.post(f"/api/photos/{pid}/export", json={"format": "gif"}).status_code == 400


def test_export_of_transparent_image_as_jpeg(ready):
    buf = io.BytesIO()
    Image.new("RGBA", (8, 8), (255, 0, 0, 0)).save(buf, "PNG")
    pid = upload(ready, buf.getvalue(), "t.png").json()["id"]
    r = ready.post(f"/api/photos/{pid}/export", json={"format": "jpeg"})
    assert r.status_code == 200 and Image.open(io.BytesIO(r.content)).mode == "RGB"


def test_delete(ready, tmp_path):
    pid = upload(ready, make_jpeg()).json()["id"]
    assert ready.delete(f"/api/photos/{pid}").status_code == 200
    assert ready.get(f"/api/photos/{pid}").status_code == 404
    assert not (tmp_path / "data" / "photos" / pid).exists()
