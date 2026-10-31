import pytest
import io
import time
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

def test_v2_detect_video_smoke(client):
    with open("synthetic_test.mp4", "rb") as f:
        video_data = f.read()
        
    data = {
        "video": (io.BytesIO(video_data), "synthetic_test.mp4"),
        "query": "car"
    }
    
    response = client.post("/v2/detect-video", data=data, content_type="multipart/form-data")
    assert response.status_code == 200
    
    res_json = response.json
    assert res_json["status"] == "processing"
    job_id = res_json["job_id"]
    
    # Poll progress
    for _ in range(30):
        time.sleep(1)
        prog_res = client.get(f"/v2/video-progress/{job_id}")
        assert prog_res.status_code == 200
        prog_json = prog_res.json
        print("Progress:", prog_json)
        
        if prog_json["status"] in ["completed", "failed"]:
            assert prog_json["status"] == "completed"
            
            result = prog_json["result"]
            assert result["status"] == "SUCCESS"
            assert result["media_type"] == "VIDEO"
            assert result["query"] == "car"
            assert "total_frames" in result
            assert "annotated_video_url" in result
            break
    else:
        pytest.fail("Timeout waiting for video processing to complete")
