import io
import os

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from common import CLASSES, ROOT

EXAMPLE = os.path.join(ROOT, "web", "assets", "example.jpg")


@pytest.fixture(scope="module")
def client():
    from server.main import app
    return TestClient(app)


def _post(client, path, data=None, name="a.jpg", content=None):
    content = content if content is not None else open(EXAMPLE, "rb").read()
    return client.post(path, files={"file": (name, content, "image/jpeg")}, data=data or {})


def test_health(client):
    j = client.get("/api/health").json()
    assert j["status"] == "ok" and j["classes"] == CLASSES


def test_predict_contract(client):
    j = _post(client, "/api/predict").json()
    assert j["classes"] == CLASSES
    assert abs(sum(j["probs"]) - 1.0) < 1e-2
    assert CLASSES[j["top"]] == "pizza"
    assert j["latency_ms"] > 0 and j["device"] and "EfficientNet-B0" in j["model"]
    assert len(j["neighbors"]) == 5
    for n in j["neighbors"]:
        assert os.path.exists(os.path.join(ROOT, "web", n["url"].lstrip("/")))   # URLs resolve to shipped thumbnails


def test_explain_returns_a_jpeg_data_url(client):
    j = _post(client, "/api/explain", data={"cls": "pizza"}).json()
    assert j["cam"].startswith("data:image/jpeg;base64,")


def test_rejects_non_images_and_unknown_classes(client):
    assert _post(client, "/api/predict", content=b"not an image").status_code == 400
    assert _post(client, "/api/explain", data={"cls": "cat"}).status_code == 400


def test_exif_rotated_upload_gives_the_same_answer(client):
    """A phone-style upload (rotated pixels + orientation tag) must classify like the upright photo."""
    img = Image.open(EXAMPLE).convert("RGB")
    exif = Image.Exif()
    exif[274] = 6
    buf = io.BytesIO()
    img.rotate(90, expand=True).save(buf, "JPEG", quality=95, exif=exif.tobytes())
    a = _post(client, "/api/predict").json()
    b = _post(client, "/api/predict", content=buf.getvalue()).json()
    assert a["top"] == b["top"]


def test_synthetic_counts_match_the_filter_report(client):
    j = client.get("/api/synthetic").json()
    assert (j["generated"], j["accepted"]) == (200, 139)
    assert all(len(v["images"]) == 6 for v in j["classes"].values())


@pytest.mark.parametrize("path", ["/", "/classifier", "/synthetic", "/css/style.css", "/css/lens.css", "/assets/example.jpg"])
def test_pages_and_static_files(client, path):
    assert client.get(path).status_code == 200
