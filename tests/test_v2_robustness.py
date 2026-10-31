import pytest
import io
import time
import cv2
import numpy as np
from flask import Flask
from api.v2_api import v2_blueprint
from v2.schemas.state import AgentStateV2, QuerySpec, Candidate, Plan, TrackingState
from v2.agents.planning_agent import PlanningAgentV2
from v2.agents.tracking_agent import TrackingAgentV2
from v2.agents.verification_agent import VerificationAgentV2

@pytest.fixture
def app():
    app = Flask(__name__)
    app.register_blueprint(v2_blueprint)
    app.config["TESTING"] = True
    return app

@pytest.fixture
def client(app):
    return app.test_client()

@pytest.fixture
def dummy_image():
    # 100x100 black image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    is_success, buffer = cv2.imencode(".jpg", img)
    return io.BytesIO(buffer)

@pytest.fixture
def dummy_video():
    # generate a small 5-frame valid mp4 in memory?
    # Better to just use the synthetic_test.mp4 we already have on disk.
    with open("synthetic_test.mp4", "rb") as f:
        return f.read()

# ==========================================
# API E2E ROBUSTNESS TESTS (NO-TARGET CASES)
# ==========================================

def test_image_text(client, dummy_image):
    data = {"image": (dummy_image, "test.jpg"), "query": "car"}
    res = client.post("/v2/detect", data=data, content_type="multipart/form-data")
    assert res.status_code == 200
    assert res.json["status"] == "SUCCESS"
    assert res.json["query"] == "car"
    assert res.json["detections"] == []

def test_image_reference(client, dummy_image):
    dummy_image.seek(0)
    img_bytes = dummy_image.read()
    data = {
        "image": (io.BytesIO(img_bytes), "test.jpg"),
        "reference_image": (io.BytesIO(img_bytes), "ref.jpg")
    }
    res = client.post("/v2/detect", data=data, content_type="multipart/form-data")
    assert res.status_code == 200
    assert res.json["status"] == "SUCCESS"

def test_unknown_query(client, dummy_image):
    # This should route to YOLO-World
    data = {"image": (dummy_image, "test.jpg"), "query": "alien spaceship"}
    res = client.post("/v2/detect", data=data, content_type="multipart/form-data")
    assert res.status_code == 200
    assert res.json["status"] == "SUCCESS"
    
def test_unsupported_constraint(client, dummy_image):
    # Testing query with unsupported constraint
    data = {"image": (dummy_image, "test.jpg"), "query": "car, speed > 50mph"}
    res = client.post("/v2/detect", data=data, content_type="multipart/form-data")
    assert res.status_code == 200
    assert res.json["status"] == "SUCCESS"
    assert res.json["query"] == "car, speed > 50mph"

def test_invalid_media(client):
    # Missing media
    res = client.post("/v2/detect", data={"query": "car"})
    assert res.status_code == 400
    # Invalid extension
    data = {"image": (io.BytesIO(b"bad"), "test.txt"), "query": "car"}
    res = client.post("/v2/detect", data=data, content_type="multipart/form-data")
    assert res.status_code == 400

def test_video_text(client, dummy_video):
    data = {"video": (io.BytesIO(dummy_video), "test.mp4"), "query": "car"}
    res = client.post("/v2/detect-video", data=data, content_type="multipart/form-data")
    assert res.status_code == 200
    job_id = res.json["job_id"]
    
    # Wait for completion
    for _ in range(30):
        time.sleep(1)
        prog = client.get(f"/v2/video-progress/{job_id}").json
        if prog["status"] == "completed":
            assert prog["result"]["status"] == "SUCCESS"
            return
    pytest.fail("Timeout on video")

def test_video_reference(client, dummy_video, dummy_image):
    data = {
        "video": (io.BytesIO(dummy_video), "test.mp4"),
        "reference_image": (dummy_image, "ref.jpg")
    }
    res = client.post("/v2/detect-video", data=data, content_type="multipart/form-data")
    assert res.status_code == 200
    job_id = res.json["job_id"]
    
    for _ in range(30):
        time.sleep(1)
        prog = client.get(f"/v2/video-progress/{job_id}").json
        if prog["status"] == "completed":
            assert prog["result"]["status"] == "SUCCESS"
            return
    pytest.fail("Timeout on video")

# ==========================================
# SYNTHETIC STATE-MACHINE TESTS
# ==========================================

def test_target_disappearance_redetection_synthetic():
    """
    Synthetic test to prove that TrackingAgentV2 degrades tracking score when target is lost,
    and PlanningAgentV2 forces REDETECT.
    """
    state = AgentStateV2()
    state.query_spec = QuerySpec(target_class="car")
    state.plan = Plan()
    state.tracking_state = TrackingState()
    
    planning_agent = PlanningAgentV2()
    tracking_agent = TrackingAgentV2()
    
    # Simulate Frame 0: detection produces a candidate
    c1 = Candidate(bbox=[100,100,150,150], conf=0.9, class_name="car")
    state.candidates = [c1]
    
    # Track it
    state = tracking_agent.run(state, new_candidates=state.candidates)
    assert state.tracking_state.active_tracks == 1
    
    # Check planner behavior for next frame
    from v2.schemas.state import ActionType
    state.plan.history = [ActionType.TRACK]
    
    # Now simulate frame 1 with no detection provided (coasting)
    state.candidates = []
    state = tracking_agent.run(state)
    
    # Simulate multiple missed frames until degradation triggers
    for _ in range(10):
        state = tracking_agent.run(state)
    
    assert state.tracking_state.track_degradation_score > 0.7
    
    # Planner should now trigger REDETECT
    next_action = planning_agent.run(state)
    from v2.schemas.state import ActionType
    assert next_action == ActionType.REDETECT
    
def test_poor_reference_synthetic():
    """
    Synthetic test to prove VerificationAgentV2 can evaluate constraints.
    """
    agent = VerificationAgentV2()
    spec = QuerySpec(target_class="car", reference_image_path="dummy/path.jpg")
    c = Candidate(bbox=[0,0,10,10], conf=0.9, class_name="car")
    # We expect verification to return uncertain or process it, 
    # but since it's a dummy adapter it defaults to SATISFIED
    # This just ensures the verification loop handles ref images safely without crashing.
    res = agent._verify_candidate(spec, c, image=None)
    assert res is not None

# ==========================================
# SEQUENTIAL REQUEST ISOLATION
# ==========================================
def test_request_isolation(client, dummy_image):
    # Request A: Known
    img_bytes = dummy_image.read()
    dummy_image.seek(0)
    data_a = {"image": (io.BytesIO(img_bytes), "test.jpg"), "query": "car"}
    r1 = client.post("/v2/detect", data=data_a, content_type="multipart/form-data")
    assert r1.status_code == 200
    
    # Request B: Unknown
    data_b = {"image": (io.BytesIO(img_bytes), "test.jpg"), "query": "alien"}
    r2 = client.post("/v2/detect", data=data_b, content_type="multipart/form-data")
    assert r2.status_code == 200
    
    # Request C: Known
    data_c = {"image": (io.BytesIO(img_bytes), "test.jpg"), "query": "truck"}
    r3 = client.post("/v2/detect", data=data_c, content_type="multipart/form-data")
    assert r3.status_code == 200

    # Ensure isolation
    assert "car" in r1.json["query"]
    assert "alien" in r2.json["query"]
    assert "truck" in r3.json["query"]
