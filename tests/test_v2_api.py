import pytest
import io
from flask import Flask
from api.v2_api import v2_blueprint

@pytest.fixture
def app():
    app = Flask(__name__)
    app.register_blueprint(v2_blueprint)
    app.config["TESTING"] = True
    return app

@pytest.fixture
def client(app):
    return app.test_client()

def test_v2_detect_missing_image(client):
    response = client.post("/v2/detect", data={"query": "car"})
    assert response.status_code == 400
    assert response.json["error"] == "No image uploaded"

def test_v2_detect_missing_query(client):
    data = {
        "image": (io.BytesIO(b"fake image data"), "test.jpg")
    }
    response = client.post("/v2/detect", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    assert "query or reference image" in response.json["error"]

def test_v2_detect_invalid_extension(client):
    data = {
        "image": (io.BytesIO(b"fake data"), "test.txt"),
        "query": "car"
    }
    response = client.post("/v2/detect", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    assert "Unsupported format" in response.json["error"]
