"""Phase 7: watermarks. Drawing, presets, the bundled font, and who may do what (owner at the computer vs a paired phone)."""
import hashlib
import io
import json

import numpy as np
import pytest
from PIL import Image, PngImagePlugin

from conftest import make_jpeg
from test_phone import world  # noqa: F401  (the owner + paired-phone fixture)
from veil import watermark as W


def base(**kw):
    return W.clean({"name": "Mine", "text": "© Studio Veil", **kw})


def photo(size=(800, 600), colour=(90, 110, 130)):
    return Image.new("RGB", size, colour)


def changed_box(a: Image.Image, b: Image.Image, thr=6):
    d = np.abs(np.asarray(a.convert("RGB")).astype(int) - np.asarray(b.convert("RGB")).astype(int)).sum(2) > thr
    ys, xs = np.nonzero(d)
    if not len(xs):
        return None
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def make_logo(path, size=(300, 150)):
    im = Image.new("RGBA", size, (0, 0, 0, 0))
    im.paste((255, 0, 100, 255), (20, 20, size[0] - 20, size[1] - 20))
    im.save(path)
    return path


# ---------------------------------------------------------------- the bundled font
def test_font_is_the_pinned_one_and_its_licence_ships_with_it():
    assert hashlib.sha256(W.FONT_PATH.read_bytes()).hexdigest() == W.FONT_SHA256
    lic = (W.FONT_PATH.parent / "Inter-OFL.txt").read_text()
    assert "SIL Open Font License, Version 1.1" in lic and "Inter Project Authors" in lic


def test_a_tampered_font_is_refused(monkeypatch):
    monkeypatch.setattr(W, "_font_ok", False)
    monkeypatch.setattr(W, "FONT_SHA256", "0" * 64)
    with pytest.raises(W.WatermarkError, match="checksum"):
        W.render(photo(), base(), None)


# ---------------------------------------------------------------- checking a preset
@pytest.mark.parametrize("fields,msg", [
    ({"name": ""}, "name"), ({"name": "   "}, "name"), ({"kind": "video"}, "text or a logo"), ({"position": "up"}, "where"),
    ({"weight": "heavy"}, "Weight"), ({"color": "red"}, "colour"), ({"color": "#12345"}, "colour"), ({"text": ""}, "words"),
    ({"text": "   \n  "}, "words"), ({"opacity": "lots"}, "number"), ({"size": float("nan")}, "number"), ({"angle": None}, "number"),
    ({"text": "漢字"}, "cannot draw"), ({"text": "hello 😀"}, "cannot draw"),
])
def test_bad_presets_are_refused_with_a_reason(fields, msg):
    with pytest.raises(W.WatermarkError, match=msg):
        base(**fields)


def test_numbers_are_clamped_and_unknown_fields_dropped():
    p = base(opacity=500, size=-3, margin=99, angle=400, spacing=-5, logo="../../etc/passwd", id="x", evil="y")
    assert (p["opacity"], p["size"], p["margin"], p["angle"], p["spacing"]) == (100, 2, 25, 90, 0)
    assert "evil" not in p and p.get("logo") in (None, False) and "id" not in p


def test_text_is_trimmed_and_control_characters_removed():
    p = base(text="  Hello\x00 \x07world\t \nsecond‮ line ")
    assert "\x00" not in p["text"] and "\x07" not in p["text"] and "‮" not in p["text"]
    assert len(base(text="a" * 900)["text"]) == 200


def test_accents_greek_and_cyrillic_are_fine():
    for t in ("Café Ñandú", "Σοφία", "Мария", "© ® ™ 2026 – studio"):
        assert base(text=t)["text"] == t


# ---------------------------------------------------------------- drawing
def test_text_goes_in_the_chosen_corner_and_nothing_else_changes():
    img = photo()
    out = W.render(img, base(position="bottom-right", size=30, margin=4), None)
    assert out.size == img.size and out.mode == "RGB"
    x0, y0, x1, y1 = changed_box(img, out)
    assert x0 > img.width * 0.5 and y0 > img.height * 0.8 and x1 <= img.width - 0.03 * 600 and y1 <= img.height - 0.03 * 600
    arr_a, arr_b = np.asarray(img), np.asarray(out)
    assert np.array_equal(arr_a[: int(img.height * 0.7)], arr_b[: int(img.height * 0.7)])      # everywhere else is untouched


@pytest.mark.parametrize("pos,xrange,yrange", [
    ("top-left", (0, .4), (0, .25)), ("top-center", (.3, .7), (0, .25)), ("top-right", (.6, 1), (0, .25)),
    ("middle-left", (0, .4), (.35, .65)), ("center", (.3, .7), (.35, .65)), ("middle-right", (.6, 1), (.35, .65)),
    ("bottom-left", (0, .4), (.75, 1)), ("bottom-center", (.3, .7), (.75, 1)), ("bottom-right", (.6, 1), (.75, 1)),
])
def test_all_nine_places(pos, xrange, yrange):
    img = photo()
    x0, y0, x1, y1 = changed_box(img, W.render(img, base(position=pos, size=22), None))
    assert xrange[0] * img.width <= x0 and x1 <= xrange[1] * img.width, (pos, x0, x1)
    assert yrange[0] * img.height <= y0 and y1 <= yrange[1] * img.height, (pos, y0, y1)


def test_size_is_a_share_of_the_photo_width():
    img = photo((1000, 700))
    for size in (15, 30, 50):
        x0, _, x1, _ = changed_box(img, W.render(img, base(position="center", size=size, outline=False), None))
        assert abs((x1 - x0) / 1000 - size / 100) < 0.04, size


def test_opacity_makes_it_fainter():
    img = photo()
    strong = changed_box(img, W.render(img, base(opacity=100), None))
    d = lambda o: np.abs(np.asarray(W.render(img, base(opacity=o), None)).astype(int) - np.asarray(img).astype(int)).sum()
    assert d(100) > d(60) > d(20) > 0
    assert strong is not None


def test_tile_covers_the_whole_photo_and_can_be_rotated():
    img = photo((900, 700))
    for angle in (0, -30):
        out = W.render(img, base(position="tile", size=18, opacity=40, angle=angle), None)
        a, b = np.asarray(img).astype(int), np.asarray(out).astype(int)
        diff = np.abs(a - b).sum(2) > 6
        for qy in (0, 1):
            for qx in (0, 1):
                quad = diff[qy * 350:(qy + 1) * 350, qx * 450:(qx + 1) * 450]
                assert quad.mean() > 0.01, (angle, qx, qy)


def test_the_input_picture_is_not_modified():
    img = photo()
    before = img.tobytes()
    W.render(img, base(), None)
    assert img.tobytes() == before


def test_a_transparent_picture_keeps_its_transparency():
    img = Image.new("RGBA", (400, 300), (0, 0, 0, 0))
    out = W.render(img, base(position="center", size=40), None)
    assert out.mode == "RGBA" and out.getpixel((0, 0))[3] == 0 and np.asarray(out)[..., 3].max() > 0


def test_a_logo_is_drawn_in_its_own_colours(tmp_path):
    logo = make_logo(tmp_path / "l.png")
    img = photo((1000, 800), (20, 20, 20))
    p = W.clean({"name": "L", "kind": "logo", "size": 20, "position": "top-left", "opacity": 100, "margin": 5})
    out = W.render(img, p, logo)
    x0, y0, x1, y1 = changed_box(img, out)
    assert 0.04 * 800 - 2 <= x0 and y0 >= 0.04 * 800 - 2 and x1 < 0.4 * 1000 and abs((x1 - x0) / 1000 - 0.2) < 0.05
    r, g, b = out.getpixel(((x0 + x1) // 2, (y0 + y1) // 2))
    assert r > 200 and g < 60 and 60 < b < 140                                  # the logo's pink, not recoloured


def test_a_logo_preset_without_a_logo_says_so():
    with pytest.raises(W.WatermarkError, match="logo"):
        W.render(photo(), W.clean({"name": "L", "kind": "logo"}), None)


def test_a_huge_watermark_never_overflows_the_photo():
    img = photo((200, 120))
    out = W.render(img, base(text="A very very long watermark text indeed", size=80, position="bottom-right", angle=45), None)
    assert out.size == img.size


# ---------------------------------------------------------------- saved presets and logos
@pytest.fixture
def store(tmp_path):
    return W.WatermarkStore(tmp_path)


def test_presets_are_saved_changed_and_deleted(store, tmp_path):
    a = store.create({"name": "One", "text": "x"})
    b = store.create({"name": "Two", "text": "y", "opacity": 30})
    assert [p["name"] for p in store.overview()["presets"]] == ["One", "Two"]
    assert store.update(a["id"], {"opacity": 20})["opacity"] == 20 and store.get(a["id"])["text"] == "x"
    store.set_default(True, a["id"])
    assert store.overview()["default"] == {"on": True, "preset": a["id"]}
    store.delete(a["id"])
    assert store.overview()["default"] == {"on": False, "preset": None}
    assert [p["id"] for p in store.overview()["presets"]] == [b["id"]]
    with pytest.raises(W.WatermarkError):
        store.delete(a["id"])
    assert W.WatermarkStore(tmp_path).overview()["presets"][0]["id"] == b["id"]            # survives a restart


def test_a_broken_presets_file_does_not_stop_veil(store):
    store.file.write_text("{ not json")
    assert store.overview()["presets"] == []
    assert store.create({"name": "n", "text": "t"})["id"]


def test_default_must_be_a_real_preset(store):
    with pytest.raises(W.WatermarkError):
        store.set_default(True, "deadbeefdead")


def test_there_is_a_limit(store):
    for i in range(W.MAX_PRESETS):
        store.create({"name": f"p{i}", "text": "t"})
    with pytest.raises(W.WatermarkError, match="most"):
        store.create({"name": "one more", "text": "t"})


def test_a_logo_is_checked_and_stored_clean(store, tmp_path):
    p = store.create({"name": "L", "kind": "logo"})
    img = Image.new("RGBA", (3000, 1000), (255, 0, 0, 255))
    meta = PngImagePlugin.PngInfo(); meta.add_text("Author", "secret name"); meta.add_text("GPS", "40.1,-70.2")
    buf = io.BytesIO(); img.save(buf, "PNG", pnginfo=meta)
    out = store.set_logo(p["id"], buf.getvalue())
    assert out["has_logo"] is True
    with Image.open(store.logo_path(p["id"])) as saved:
        assert saved.format == "PNG" and max(saved.size) == W.MAX_LOGO_SIDE and saved.info == {}


def test_a_jpeg_logo_loses_its_exif(store):
    p = store.create({"name": "L", "kind": "logo"})
    store.set_logo(p["id"], make_jpeg())
    with Image.open(store.logo_path(p["id"])) as saved:
        assert not saved.getexif() and "exif" not in saved.info


@pytest.mark.parametrize("data", [b"", b"not a picture", b"<svg xmlns='http://www.w3.org/2000/svg'><script>alert(1)</script></svg>",
                                  b"GIF89a" + b"\0" * 40, b"\x89PNG\r\n\x1a\n" + b"junk" * 10])
def test_things_that_are_not_png_jpeg_or_webp_are_refused(store, data):
    p = store.create({"name": "L", "kind": "logo"})
    with pytest.raises(W.WatermarkError):
        store.set_logo(p["id"], data)
    assert not store.logo_path(p["id"]).exists() and store.get(p["id"]).get("logo") in (False, None)


def test_a_gif_is_refused_even_when_valid(store):
    p = store.create({"name": "L", "kind": "logo"})
    buf = io.BytesIO(); Image.new("P", (8, 8)).save(buf, "GIF")
    with pytest.raises(W.WatermarkError, match="PNG, JPEG or WebP"):
        store.set_logo(p["id"], buf.getvalue())


def test_oversize_logo_files_are_refused(store):
    p = store.create({"name": "L", "kind": "logo"})
    with pytest.raises(W.WatermarkError, match="too large"):
        store.set_logo(p["id"], b"0" * (W.MAX_LOGO_BYTES + 1))


@pytest.mark.parametrize("bad", ["../x", "..", "a/b", "ZZZZZZZZZZZZ", "", "deadbeef"])
def test_a_preset_id_can_never_be_a_path(store, bad):
    with pytest.raises(W.WatermarkError):
        store.logo_path(bad)


def test_deleting_a_preset_removes_its_logo_file(store):
    p = store.create({"name": "L", "kind": "logo"})
    store.set_logo(p["id"], make_jpeg())
    path = store.logo_path(p["id"]); assert path.exists()
    store.delete(p["id"])
    assert not path.exists()


# ---------------------------------------------------------------- the web routes
def upload(client, name="a.jpg"):
    r = client.post("/api/photos", files={"file": (name, make_jpeg((400, 300)), "application/octet-stream")})
    assert r.status_code == 200, r.text
    return r.json()["id"]


def decode(r):
    return Image.open(io.BytesIO(r.content))


def version_bytes(world, pid):  # noqa: F811
    d = world.data / "photos" / pid
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.glob("v*.png"))}, json.loads((d / "meta.json").read_text())


def test_export_adds_the_watermark_to_the_file_only(world):  # noqa: F811
    o = world.owner
    pid = upload(o)
    preset = o.post("/api/watermarks", json={"fields": {"name": "Corner", "text": "© me", "size": 30, "position": "center", "opacity": 100}}).json()
    before = version_bytes(world, pid)
    plain = o.post(f"/api/photos/{pid}/export", json={"format": "png"})
    marked = o.post(f"/api/photos/{pid}/export", json={"format": "png", "watermark": preset["id"]})
    assert plain.status_code == marked.status_code == 200
    a, b = decode(plain), decode(marked)
    assert a.size == b.size == (400, 300) and changed_box(a, b) is not None
    assert version_bytes(world, pid) == before                                           # the saved versions are untouched
    assert o.get(f"/api/photos/{pid}").json()["versions"] == o.get(f"/api/photos/{pid}").json()["versions"]
    assert len(o.get(f"/api/photos/{pid}").json()["versions"]) == 1                        # no history entry was added
    shown = decode(o.get(f"/api/photos/{pid}/image?version=current"))
    assert changed_box(shown.convert("RGB"), a.convert("RGB")) is None                     # what you see while editing is still clean


def test_the_exported_file_still_has_no_hidden_details(world):  # noqa: F811
    o = world.owner
    pid = upload(o)
    preset = o.post("/api/watermarks", json={"fields": {"name": "n", "text": "t"}}).json()
    for fmt in ("jpeg", "png", "webp"):
        img = decode(o.post(f"/api/photos/{pid}/export", json={"format": fmt, "watermark": preset["id"]}))
        assert not img.getexif() and not any(k.lower() in ("exif", "xmp", "gps", "make") for k in img.info), fmt


def test_export_without_a_watermark_is_exactly_as_before(world):  # noqa: F811
    o = world.owner
    pid = upload(o)
    a = o.post(f"/api/photos/{pid}/export", json={"format": "png"}).content
    b = o.post(f"/api/photos/{pid}/export", json={"format": "png", "watermark": None}).content
    assert a == b


def test_export_with_a_missing_preset_is_refused_not_silently_unmarked(world):  # noqa: F811
    o = world.owner
    pid = upload(o)
    for bad in ("deadbeefdead", "../../etc/passwd", "x" * 5000):
        assert o.post(f"/api/photos/{pid}/export", json={"format": "png", "watermark": bad}).status_code == 400


def test_a_logo_watermark_exports(world, tmp_path):  # noqa: F811
    o = world.owner
    pid = upload(o)
    p = o.post("/api/watermarks", json={"fields": {"name": "L", "kind": "logo", "size": 30, "position": "center", "opacity": 100}}).json()
    assert o.post(f"/api/photos/{pid}/export", json={"format": "png", "watermark": p["id"]}).status_code == 400          # no logo yet
    r = o.post(f"/api/watermarks/{p['id']}/logo", files={"file": ("l.png", make_logo(tmp_path / "l.png").read_bytes(), "image/png")})
    assert r.status_code == 200 and r.json()["has_logo"] is True
    assert o.get(f"/api/watermarks/{p['id']}/logo").headers["content-type"] == "image/png"
    out = decode(o.post(f"/api/photos/{pid}/export", json={"format": "png", "watermark": p["id"]}))
    assert out.getpixel((200, 150))[:3] == (255, 0, 100)


def test_preview_shows_the_watermark_and_changes_nothing(world):  # noqa: F811
    o = world.owner
    pid = upload(o)
    p = o.post("/api/watermarks", json={"fields": {"name": "n", "text": "PROOF", "position": "center", "size": 40, "opacity": 100}}).json()
    before = version_bytes(world, pid)
    saved = decode(o.post(f"/api/photos/{pid}/watermark-preview", json={"preset": p["id"]}))
    clean_img = decode(o.get(f"/api/photos/{pid}/image?version=current")).convert("RGB")
    assert changed_box(clean_img, saved) is not None
    draft = decode(o.post(f"/api/photos/{pid}/watermark-preview", json={"draft": {"name": "d", "text": "DRAFT", "position": "top-left"}}))
    assert changed_box(clean_img, draft) is not None
    assert o.post(f"/api/photos/{pid}/watermark-preview", json={}).status_code == 400
    assert o.post(f"/api/photos/{pid}/watermark-preview", json={"draft": {"name": "d", "text": ""}}).status_code == 400
    assert version_bytes(world, pid) == before


def test_not_signed_in_gets_nothing(world):  # noqa: F811
    from fastapi.testclient import TestClient
    from veil.main import create_app  # noqa: F401
    anon = TestClient(world.app, base_url="http://localhost", client=("127.0.0.1", 50001))
    assert anon.get("/api/watermarks").status_code == 401
    assert anon.post("/api/watermarks", json={"fields": {"name": "n", "text": "t"}}).status_code == 401


# ---------------------------------------------------------------- the phone: pick and switch, never create
def test_a_phone_can_list_preview_and_export_with_a_saved_preset(world):  # noqa: F811
    world.enable(); world.pair()
    o, ph = world.owner, world.phone
    pid = upload(o)
    p = o.post("/api/watermarks", json={"fields": {"name": "Phone one", "text": "© me", "position": "center", "size": 40, "opacity": 100}}).json()
    o.put("/api/watermarks/default", json={"on": True, "preset": p["id"]})
    r = ph.get("/api/watermarks")
    assert r.status_code == 200 and [x["id"] for x in r.json()["presets"]] == [p["id"]] and r.json()["default"] == {"on": True, "preset": p["id"]}
    assert decode(ph.post(f"/api/photos/{pid}/watermark-preview", json={"preset": p["id"]})).size[0] <= 1400
    plain = decode(ph.post(f"/api/photos/{pid}/export", json={"format": "png"}))
    marked = decode(ph.post(f"/api/photos/{pid}/export", json={"format": "png", "watermark": p["id"]}))
    assert changed_box(plain.convert("RGB"), marked.convert("RGB")) is not None
    assert "logo" not in json.dumps(r.json()).replace("has_logo", "")                    # no file paths are given to the phone


def test_a_phone_cannot_create_change_delete_or_upload(world, tmp_path):  # noqa: F811
    world.enable(); world.pair()
    o, ph = world.owner, world.phone
    pid = upload(o)
    p = o.post("/api/watermarks", json={"fields": {"name": "Mine", "text": "t"}}).json()
    logo = ("l.png", make_logo(tmp_path / "l.png").read_bytes(), "image/png")
    attempts = [
        ph.post("/api/watermarks", json={"fields": {"name": "Sneaky", "text": "t"}}),
        ph.put(f"/api/watermarks/{p['id']}", json={"fields": {"text": "changed"}}),
        ph.delete(f"/api/watermarks/{p['id']}"),
        ph.post(f"/api/watermarks/{p['id']}/logo", files={"file": logo}),
        ph.get(f"/api/watermarks/{p['id']}/logo"),
        ph.put("/api/watermarks/default", json={"on": True, "preset": p["id"]}),
        ph.post(f"/api/photos/{pid}/watermark-preview", json={"draft": {"name": "d", "text": "x"}}),
    ]
    assert [a.status_code for a in attempts] == [403] * 7, [a.text for a in attempts]
    assert o.get("/api/watermarks").json()["presets"][0]["text"] == "t" and len(o.get("/api/watermarks").json()["presets"]) == 1
    assert o.get("/api/watermarks").json()["default"]["on"] is False


def test_an_unpaired_or_unapproved_phone_gets_nothing(world):  # noqa: F811
    world.enable()
    assert world.phone.get("/api/watermarks").status_code == 401
    world.pair(approve=False)
    assert world.phone.get("/api/watermarks").status_code in (401, 403)
    assert world.phone.post("/api/watermarks", json={"fields": {"name": "n", "text": "t"}}).status_code in (401, 403)


def test_forged_headers_do_not_make_a_phone_the_owner(world):  # noqa: F811
    world.enable(); world.pair()
    sneaky = {"x-forwarded-for": "127.0.0.1", "x-real-ip": "127.0.0.1", "host": "localhost"}
    r = world.phone.post("/api/watermarks", json={"fields": {"name": "n", "text": "t"}}, headers=sneaky)
    assert r.status_code in (403, 421)
    assert world.owner.get("/api/watermarks").json()["presets"] == []


def test_deleting_the_default_preset_turns_the_default_off(world):  # noqa: F811
    o = world.owner
    p = o.post("/api/watermarks", json={"fields": {"name": "n", "text": "t"}}).json()
    assert o.put("/api/watermarks/default", json={"on": True, "preset": p["id"]}).json() == {"on": True, "preset": p["id"]}
    o.delete(f"/api/watermarks/{p['id']}")
    assert o.get("/api/watermarks").json()["default"] == {"on": False, "preset": None}
    assert o.put("/api/watermarks/default", json={"on": True, "preset": "deadbeefdead"}).status_code == 404


def test_bad_bodies_are_400_not_500(world):  # noqa: F811
    o = world.owner
    assert o.post("/api/watermarks", json={"fields": {"name": "n", "text": "t", "opacity": "x"}}).status_code == 400
    assert o.post("/api/watermarks", json={"fields": []}).status_code == 422
    assert o.put("/api/watermarks/deadbeefdead", json={"fields": {"name": "n", "text": "t"}}).status_code == 404
    p = o.post("/api/watermarks", json={"fields": {"name": "n", "kind": "logo"}}).json()
    assert o.post(f"/api/watermarks/{p['id']}/logo", files={"file": ("x.png", b"nope", "image/png")}).status_code == 400


# ---------------------------------------------------------------- the logo upload can never feed a face edit
def test_the_watermark_code_cannot_reach_the_face_or_ai_code():
    """A logo is only ever drawn on an exported copy. Nothing in the watermark modules imports a face, vision or AI module,
    and nothing in those reads the watermark folder, so an uploaded picture can never become a face."""
    import pathlib, re
    root = pathlib.Path(W.__file__).parent
    for name in ("watermark.py", "watermark_routes.py"):
        src = (root / name).read_text()
        assert not re.search(r"^\s*(from|import)\s+.*(vision|faces|looks|\.ai\b|engine|torch|diffusers|mediapipe)", src, re.M), name
    links = ("WatermarkStore", "watermark_routes", "from .watermark", "from ..watermark", "veil.watermark", "watermarks/", "presets.json", "logo_path")
    for sub in ("vision", "ai"):
        for p in (root / sub).glob("*.py"):
            src = p.read_text()
            assert not [x for x in links if x in src], p.name


def test_a_face_edit_refuses_a_watermark_logo_reference(world):  # noqa: F811
    o = world.owner
    pid = upload(o)
    p = o.post("/api/watermarks", json={"fields": {"name": "L", "kind": "logo"}}).json()
    for key in ("logo", "image", "reference", "face_image"):
        r = o.post(f"/api/photos/{pid}/edit", json={"op": "generate_face", "params": {key: p["id"]}})
        assert r.status_code == 400, (key, r.text)
