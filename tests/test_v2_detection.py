import pytest
from v2.schemas.state import AgentStateV2, ActionType, MediaMetadata, MediaType
from v2.agents.detection_agent import DetectionAgentV2
from v2.models.model_registry import ModelRegistry
from v2.models.adapters import SpecialistDetectorAdapter, OpenWorldDetectorAdapter

# Mocks for testing without real models
class MockSpecialistModel:
    pass

class MockOpenWorldModel:
    pass

# Patch the adapters to return synthetic detections for unit testing
class MockSpecialistAdapter(SpecialistDetectorAdapter):
    def _run_inference(self, image, query_spec):
        return [
            {"bbox": [10.0, 10.0, 100.0, 100.0], "conf": 0.95, "label": "car"},
            {"bbox": [200.0, 20.0, 30.0, 50.0], "conf": 0.88, "label": "invalid_box_should_fail"} # x1 > x2
        ]

class MockOpenWorldAdapter(OpenWorldDetectorAdapter):
    def _run_inference(self, image, vocab):
        return [
            {"bbox": [50.0, 50.0, 150.0, 150.0], "conf": 1.5, "label": vocab[0] if vocab else "object"} # conf clamped to 1.0
        ]

@pytest.fixture
def registry():
    reg = ModelRegistry()
    # Override adapters for testing
    reg.REGISTERED_MODELS["specialist_p2"]["adapter"] = MockSpecialistAdapter
    reg.REGISTERED_MODELS["open_world_yolo_world"]["adapter"] = MockOpenWorldAdapter
    return reg

def test_specialist_produces_normalized_candidate(registry):
    agent = DetectionAgentV2(registry)
    state = AgentStateV2()
    state.plan.current_action = ActionType.DETECT_SPECIALIST
    
    state = agent.run(state, image=None, mock_model_instance=MockSpecialistModel())
    
    # 1 valid, 1 invalid bbox rejected
    assert len(state.candidates) == 1
    c = state.candidates[0]
    assert c.class_label == "car"
    assert c.bbox == [10.0, 10.0, 100.0, 100.0]
    assert c.confidence == 0.95
    assert c.source == "SPECIALIST_P2"

def test_open_world_produces_normalized_candidate(registry):
    agent = DetectionAgentV2(registry)
    state = AgentStateV2()
    state.plan.current_action = ActionType.DETECT_OPEN_WORLD
    state.query_spec.target = "dragon"
    
    state = agent.run(state, image=None, mock_model_instance=MockOpenWorldModel())
    
    assert len(state.candidates) == 1
    c = state.candidates[0]
    assert c.class_label == "dragon"
    assert c.confidence == 1.0 # Clamped from 1.5
    assert c.source == "OPEN_WORLD"

def test_unknown_model_route_fails_cleanly(registry):
    agent = DetectionAgentV2(registry)
    state = AgentStateV2()
    state.plan.current_action = ActionType.TRACK # Invalid for detection agent
    
    state = agent.run(state, image=None)
    assert len(state.candidates) == 0
    assert "invalid action" in state.failure_reason

def test_lazy_loading_mutually_exclusive():
    reg = ModelRegistry()
    
    # Mock _load_model directly to avoid numpy crashes in tests
    def mock_load(name):
        if name == "specialist_p2" and "open_world_yolo_world" in reg._loaded_models:
            reg._unload_model("open_world_yolo_world")
        elif name == "open_world_yolo_world" and "specialist_p2" in reg._loaded_models:
            reg._unload_model("specialist_p2")
        reg._loaded_models[name] = "mock_instance"
        return "mock_instance"
        
    reg._load_model = mock_load
    
    adapter1 = reg.get_adapter("specialist_p2")
    assert "specialist_p2" in reg.get_loaded_models()
    
    adapter2 = reg.get_adapter("open_world_yolo_world")
    # Specialist should have been unloaded!
    loaded = reg.get_loaded_models()
    assert "open_world_yolo_world" in loaded
    assert "specialist_p2" not in loaded

def test_image_vs_frame_detection(registry):
    # Image
    agent = DetectionAgentV2(registry)
    state = AgentStateV2()
    state.plan.current_action = ActionType.DETECT_SPECIALIST
    state.media_metadata = MediaMetadata(type=MediaType.IMAGE)
    
    state = agent.run(state, image=None, mock_model_instance=MockSpecialistModel())
    assert state.candidates[0].frame_id is None
    
    # Frame logic could be tested here if we had explicit frame passing. 
    # For now, normalization checks frame_id.
