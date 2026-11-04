import io

import pytest
from fastapi.testclient import TestClient
from PIL import ExifTags, Image, ImageCms

from veil.main import create_app


@pytest.fixture
def client(tmp_path):
    return TestClient(create_app(tmp_path / "data"), base_url="http://localhost", client=("127.0.0.1", 50000))


@pytest.fixture
def ready(client):
    """A client that is logged in and has confirmed the adult-use notice."""
    assert client.post("/api/auth/setup", json={"username": "owner", "password": "correct horse"}).status_code == 200
    assert client.post("/api/notice/ack").status_code == 200
    return client


def make_jpeg(size=(60, 40), orientation=None, gps=True, icc=True) -> bytes:
    img = Image.new("RGB", size, (200, 80, 80))
    img.paste((20, 20, 220), (0, 0, size[0] // 2, size[1] // 2))  # blue top-left block
    exif = Image.Exif()
    exif[ExifTags.Base.Make] = "TestCam"
    if orientation:
        exif[ExifTags.Base.Orientation] = orientation
    if gps:
        ifd = exif.get_ifd(ExifTags.IFD.GPSInfo)
        ifd[ExifTags.GPS.GPSLatitudeRef] = "N"
        ifd[ExifTags.GPS.GPSLatitude] = (40.0, 26.0, 46.0)
    kwargs = {}
    if icc:
        kwargs["icc_profile"] = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    buf = io.BytesIO()
    img.save(buf, "JPEG", exif=exif, **kwargs)
    return buf.getvalue()
