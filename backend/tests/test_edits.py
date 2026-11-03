import base64
import io

from PIL import Image

from conftest import make_jpeg


def upload(client, size=(60, 40)):
    r = client.post("/api/photos", files={"file": ("a.jpg", make_jpeg(size=size, gps=False), "image/jpeg")})
    return r.json()["id"]


def edit(client, pid, op, params=None, mask=None):
    return client.post(f"/api/photos/{pid}/edit", json={"op": op, "params": params or {}, "mask": mask})


def png_data_url(img):
    buf = io.BytesIO(); img.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def current(client, pid):
    return Image.open(io.BytesIO(client.get(f"/api/photos/{pid}/image").content)).convert("RGB")


def test_brightness_lightens_and_darkens(ready):
    pid = upload(ready)
    before = current(ready, pid).getpixel((50, 30))
    assert edit(ready, pid, "adjust", {"brightness": 40}).status_code == 200
    lighter = current(ready, pid).getpixel((50, 30))
    assert sum(lighter) > sum(before)
    assert edit(ready, pid, "adjust", {"brightness": -60}).status_code == 200
    assert sum(current(ready, pid).getpixel((50, 30))) < sum(lighter)


def test_warmth_shifts_red_and_blue_opposite_ways(ready):
    pid = upload(ready)
    r0, g0, b0 = current(ready, pid).getpixel((50, 30))
    edit(ready, pid, "adjust", {"warmth": 80})
    r1, g1, b1 = current(ready, pid).getpixel((50, 30))
    assert r1 >= r0 and b1 < b0 and g1 == g0


def test_adjust_only_changes_the_selected_area(ready):
    pid = upload(ready, size=(80, 40))
    mask = Image.new("L", (80, 40), 0)
    mask.paste(255, (0, 0, 40, 40))  # left half selected
    before = current(ready, pid)
    assert edit(ready, pid, "adjust", {"brightness": 50}, png_data_url(mask)).status_code == 200
    after = current(ready, pid)
    assert after.getpixel((60, 30)) == before.getpixel((60, 30))      # right half untouched
    assert sum(after.getpixel((10, 30))) > sum(before.getpixel((10, 30)))  # left half lighter
    assert ready.get(f"/api/photos/{pid}").json()["versions"][-1]["label"].endswith("(selection)")


def test_mask_with_a_different_size_is_stretched_to_fit(ready):
    pid = upload(ready, size=(80, 40))
    mask = Image.new("L", (8, 4), 255)  # whole photo, tiny mask
    before = current(ready, pid)
    assert edit(ready, pid, "adjust", {"brightness": 50}, png_data_url(mask)).status_code == 200
    assert sum(current(ready, pid).getpixel((70, 30))) > sum(before.getpixel((70, 30)))


def test_crop_rotate_flip_straighten_change_size_as_expected(ready):
    pid = upload(ready, size=(60, 40))
    assert edit(ready, pid, "crop", {"x0": 10, "y0": 5, "x1": 50, "y1": 35}).status_code == 200
    assert current(ready, pid).size == (40, 30)
    edit(ready, pid, "rotate90", {"turns": 1})
    assert current(ready, pid).size == (30, 40)
    edit(ready, pid, "flip", {"axis": "horizontal"})
    assert current(ready, pid).size == (30, 40)
    meta = ready.get(f"/api/photos/{pid}").json()
    assert (meta["width"], meta["height"]) == (30, 40)

    pid2 = upload(ready, size=(200, 100))
    assert edit(ready, pid2, "straighten", {"degrees": 5}).status_code == 200
    w, h = current(ready, pid2).size
    assert 0 < w < 200 and 0 < h < 100  # corners were cropped away


def test_rotate_90_clockwise_moves_top_left_to_top_right(ready):
    pid = upload(ready, size=(60, 40))  # blue block is top-left
    edit(ready, pid, "rotate90", {"turns": 1})
    img = current(ready, pid)
    assert img.getpixel((img.width - 3, 3))[2] > 150 > img.getpixel((3, 3))[2]  # blue now top-right


def test_undo_after_crop_restores_the_size(ready):
    pid = upload(ready, size=(60, 40))
    edit(ready, pid, "crop", {"x0": 0, "y0": 0, "x1": 30, "y1": 20})
    m = ready.post(f"/api/photos/{pid}/undo").json()
    assert (m["width"], m["height"]) == (60, 40)
    m = ready.post(f"/api/photos/{pid}/goto", json={"cursor": 1}).json()
    assert (m["width"], m["height"]) == (30, 20)
    assert ready.post(f"/api/photos/{pid}/goto", json={"cursor": 9}).status_code == 400


def test_bad_edits_are_refused_without_changing_anything(ready):
    pid = upload(ready)
    bad = [("adjust", {"brightness": 500}), ("adjust", {"brightness": "lots"}), ("adjust", {"brightness": float("nan")}),
           ("crop", {"x0": 0, "y0": 0, "x1": 2, "y1": 2}), ("flip", {"axis": "diagonal"}), ("wiggle", {}),
           ("rotate90", {"turns": 0}), ("straighten", {"degrees": 80})]
    for op, params in bad:
        r = ready.post(f"/api/photos/{pid}/edit", content=__import__("json").dumps({"op": op, "params": params}),
                       headers={"content-type": "application/json"})
        assert r.status_code in (400, 422), (op, params, r.status_code)
    assert len(ready.get(f"/api/photos/{pid}").json()["versions"]) == 1
    assert edit(ready, pid, "adjust", {"brightness": 5}, mask="not-an-image").status_code == 400
    assert edit(ready, pid, "crop", {"x0": 0, "y0": 0, "x1": 30, "y1": 30},
                mask=png_data_url(Image.new("L", (4, 4), 255))).status_code == 400  # geometry can't use a selection


def test_preview_is_small_and_saves_nothing(ready):
    pid = upload(ready, size=(3000, 2000))
    r = ready.post(f"/api/photos/{pid}/preview", json={"op": "adjust", "params": {"contrast": 30}})
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    assert max(Image.open(io.BytesIO(r.content)).size) <= 1400
    assert len(ready.get(f"/api/photos/{pid}").json()["versions"]) == 1


def test_thumbnail_size_and_page_is_served_with_security_headers(ready):
    pid = upload(ready, size=(600, 400))
    t = ready.get(f"/api/photos/{pid}/image?size=100")
    assert t.headers["content-type"] == "image/jpeg" and max(Image.open(io.BytesIO(t.content)).size) <= 100
    page = ready.get("/")
    assert page.status_code == 200 and b"Veil" in page.content
    csp = page.headers["content-security-policy"]
    assert "script-src 'self'" in csp and "frame-ancestors 'none'" in csp
