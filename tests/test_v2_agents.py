import pytest
from v2.schemas.state import AgentStateV2, QueryConstraint, Candidate, VerificationResult, ConstraintResult, ActionType, MediaType
from v2.agents.query_agent import QueryAgentV2
from v2.agents.planning_agent import PlanningAgentV2
from v2.agents.verification_agent import VerificationAgentV2

# ================================
# QUERY AGENT TESTS
# ================================
def test_query_agent_simple():
    agent = QueryAgentV2()
    spec = agent.run(raw_query="find a person")
    assert spec.target == "person"
    assert len(spec.constraints) == 0

def test_query_agent_multi_constraint():
    agent = QueryAgentV2()
    spec = agent.run(raw_query="find a white car near a tree")
    assert spec.target == "car"
    assert any(c.constraint_type == "attribute" and c.value == "white" for c in spec.constraints)
    assert any(c.constraint_type == "relation" and c.value == "near" and c.metadata.get("object") == "a tree" for c in spec.constraints)

def test_query_agent_reference_image():
    agent = QueryAgentV2()
    spec = agent.run(raw_query="", reference_image_path="/some/image.jpg")
    assert spec.target == "reference object"
    assert spec.reference_image_path == "/some/image.jpg"

def test_query_agent_unsupported_preservation():
    agent = QueryAgentV2()
    # "wearing a blue shirt" should create an attribute constraint for "blue" 
    # and maybe clothing preservation depending on parser logic.
    spec = agent.run(raw_query="person wearing blue shirt")
    assert spec.target == "person"
    assert any(c.value == "blue" for c in spec.constraints)
    assert any(c.value == "wearing blue shirt" for c in spec.constraints)

# ================================
# PLANNING AGENT TESTS
# ================================
def test_planner_specialist_routing():
    planner = PlanningAgentV2()
    state = AgentStateV2()
    state.query_spec.target = "car"
    action = planner.run(state)
    assert action == ActionType.DETECT_SPECIALIST

def test_planner_open_world_routing():
    planner = PlanningAgentV2()
    state = AgentStateV2()
    state.query_spec.target = "dragon" # Unknown
    action = planner.run(state)
    assert action == ActionType.DETECT_OPEN_WORLD

def test_planner_reference_image_routing():
    planner = PlanningAgentV2()
    state = AgentStateV2()
    state.query_spec.target = "reference object" # Treated as unknown by the simple set
    action = planner.run(state)
    assert action == ActionType.DETECT_OPEN_WORLD

def test_planner_zero_candidate():
    planner = PlanningAgentV2()
    state = AgentStateV2()
    state.plan.history = [ActionType.DETECT_SPECIALIST]
    state.query_spec.target = "person" # Small object triggers SAHI
    state.candidates = []
    action = planner.run(state)
    assert action == ActionType.TERMINATE

def test_planner_retry_budget_exhaustion():
    planner = PlanningAgentV2()
    state = AgentStateV2()
    state.plan.history = [ActionType.DETECT_SPECIALIST]
    state.query_spec.target = "person"
    state.candidates = []
    state.tool_usage.retry_count = 2 # Exhausted
    action = planner.run(state)
    assert action == ActionType.TERMINATE

def test_planner_verification_failure_retry():
    planner = PlanningAgentV2()
    state = AgentStateV2()
    state.plan.history = [ActionType.VERIFY_CANDIDATES]
    # No targets verified
    state.verification_results = [
        VerificationResult(candidate_id="c1", satisfies_query=False, constraint_results=[])
    ]
    action = planner.run(state)
    assert action == ActionType.REDETECT
    assert state.tool_usage.retry_count == 1

def test_planner_video_tracking():
    from v2.schemas.state import ConstraintStatus
    planner = PlanningAgentV2()
    state = AgentStateV2()
    state.plan.history = [ActionType.VERIFY_CANDIDATES]
    state.media_metadata.type = MediaType.VIDEO
    state.verification_results = [
        VerificationResult(candidate_id="c1", status=ConstraintStatus.SATISFIED, satisfies_query=True, constraint_results=[])
    ]
    action = planner.run(state)
    assert action == ActionType.TRACK

def test_planner_degraded_tracking():
    planner = PlanningAgentV2()
    state = AgentStateV2()
    state.plan.history = [ActionType.TRACK]
    state.tracking_state.track_degradation_score = 0.8
    action = planner.run(state)
    assert action == ActionType.REDETECT

def test_planner_terminate_success():
    from v2.schemas.state import ConstraintStatus
    planner = PlanningAgentV2()
    state = AgentStateV2()
    state.plan.history = [ActionType.VERIFY_CANDIDATES]
    state.media_metadata.type = MediaType.IMAGE
    state.verification_results = [
        VerificationResult(candidate_id="c1", status=ConstraintStatus.SATISFIED, satisfies_query=True, constraint_results=[])
    ]
    action = planner.run(state)
    assert action == ActionType.TERMINATE

# ================================
# VERIFICATION AGENT TESTS
# ================================
def test_verification_unsupported_constraint():
    from PIL import Image
    from v2.schemas.state import ConstraintStatus
    agent = VerificationAgentV2()
    agent_query = QueryAgentV2()
    spec = agent_query.run(raw_query="find a person near a car")
    
    candidate = Candidate(id="c1", bbox=[10,10,100,100], class_label="person", confidence=0.9, source="model")
    img = Image.new("RGB", (200, 200))
    
    results = agent.run(spec, [candidate], image=img)
    assert len(results) == 1
    assert results[0].satisfies_query is False
    assert results[0].rejection_reason is not None
    # Check that "near" relation is explicitly unsupported in the mock
    relation_res = next((cr for cr in results[0].constraint_results if cr.constraint_id == "near"), None)
    assert relation_res is not None
    assert relation_res.status == ConstraintStatus.UNSUPPORTED
