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

def test_v2_detect_video_missing_video(client):
    response = client.post("/v2/detect-video", data={"query": "car"})
    assert response.status_code == 400
    assert response.json["error"] == "No video uploaded"

def test_v2_detect_video_missing_query(client):
    data = {
        "video": (io.BytesIO(b"fake data"), "test.mp4")
    }
    response = client.post("/v2/detect-video", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    assert "query or reference image" in response.json["error"]

def test_v2_detect_video_invalid_extension(client):
    data = {
        "video": (io.BytesIO(b"fake data"), "test.txt"),
        "query": "car"
    }
    response = client.post("/v2/detect-video", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    assert "Unsupported format" in response.json["error"]

def test_v2_detect_video_success(client):
    data = {
        "video": (io.BytesIO(b"fake data"), "test.mp4"),
        "query": "car"
    }
    response = client.post("/v2/detect-video", data=data, content_type="multipart/form-data")
    assert response.status_code == 200
    assert response.json["status"] == "processing"
    assert "job_id" in response.json
