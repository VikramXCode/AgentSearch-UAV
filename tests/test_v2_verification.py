import pytest
from v2.schemas.state import AgentStateV2, Candidate, QuerySpec, QueryConstraint, MediaMetadata, ActionType, ConstraintStatus, VerificationResult, ConstraintResult
from v2.agents.verification_agent import VerificationAgentV2
from v2.agents.planning_agent import PlanningAgentV2
from PIL import Image

def get_dummy_image():
    return Image.new("RGB", (2000, 2000))

def test_target_constraint_passes():
    verifier = VerificationAgentV2()
    # Mocking target as an attribute constraint for now
    query = QuerySpec(constraints=[QueryConstraint(constraint_type="attribute", value="car")])
    candidate = Candidate(id="1", bbox=[0,0,10,10], confidence=0.9, class_label="car")
    
    results = verifier.run(query, [candidate], image=get_dummy_image())
    assert len(results) == 1
    assert results[0].status == ConstraintStatus.SATISFIED
    assert results[0].satisfies_query is True
    assert results[0].ambiguity == 0.0

def test_attribute_constraint_passes():
    verifier = VerificationAgentV2()
    query = QuerySpec(constraints=[QueryConstraint(constraint_type="attribute", value="red")])
    candidate = Candidate(id="1", bbox=[0,0,10,10], confidence=0.9, class_label="car")
    
    results = verifier.run(query, [candidate], image=get_dummy_image())
    assert results[0].status == ConstraintStatus.SATISFIED

def test_unsupported_constraint_returns_explicit_uncertainty():
    verifier = VerificationAgentV2()
    query = QuerySpec(constraints=[QueryConstraint(constraint_type="relation", value="chasing")])
    candidate = Candidate(id="1", bbox=[0,0,10,10], confidence=0.9, class_label="car")
    
    results = verifier.run(query, [candidate], image=get_dummy_image())
    assert results[0].status == ConstraintStatus.UNSUPPORTED
    assert results[0].satisfies_query is False
    assert results[0].ambiguity == 1.0

def test_reference_image_verification_mock(tmp_path):
    # Create a dummy reference image file so Image.open doesn't crash
    dummy_path = str(tmp_path / "dummy_ref.jpg")
    img = Image.new("RGB", (10, 10))
    img.save(dummy_path)
    
    verifier = VerificationAgentV2()
    query = QuerySpec(reference_image_path=dummy_path)
    candidate = Candidate(id="1", bbox=[0,0,10,10], confidence=0.9, class_label="car")
    
    results = verifier.run(query, [candidate], image=get_dummy_image())
    assert results[0].status == ConstraintStatus.SATISFIED
    assert results[0].constraint_results[0].constraint_id == "reference_image_similarity"

def test_spatial_constraint_synthetic_boxes():
    verifier = VerificationAgentV2()
    query = QuerySpec(constraints=[QueryConstraint(constraint_type="spatial", value="left")])
    media_meta = MediaMetadata(resolution=[1920, 1080]) # w, h
    
    c_left = Candidate(id="1", bbox=[10, 10, 100, 100], confidence=0.9, class_label="car") # cx=55
    c_right = Candidate(id="2", bbox=[1500, 10, 1600, 100], confidence=0.9, class_label="car") # cx=1550
    
    results = verifier.run(query, [c_left, c_right], media_meta, image=get_dummy_image())
    
    assert results[0].status == ConstraintStatus.SATISFIED
    assert results[1].status == ConstraintStatus.VIOLATED
    assert "Violated constraints: left" in results[1].rejection_reason
    
def test_multiple_constraints_all_pass():
    verifier = VerificationAgentV2()
    query = QuerySpec(constraints=[
        QueryConstraint(constraint_type="attribute", value="blue"),
        QueryConstraint(constraint_type="spatial", value="top")
    ])
    media_meta = MediaMetadata(resolution=[1920, 1080])
    c = Candidate(id="1", bbox=[10, 10, 100, 100], confidence=0.9, class_label="car") # cy=55 (top)
    
    results = verifier.run(query, [c], media_meta, image=get_dummy_image())
    assert results[0].status == ConstraintStatus.SATISFIED
    assert len(results[0].constraint_results) == 2

def test_one_failed_constraint_rejects():
    verifier = VerificationAgentV2()
    query = QuerySpec(constraints=[
        QueryConstraint(constraint_type="attribute", value="blue"), # passes in mock
        QueryConstraint(constraint_type="spatial", value="bottom") # fails for this box
    ])
    media_meta = MediaMetadata(resolution=[1920, 1080])
    c = Candidate(id="1", bbox=[10, 10, 100, 100], confidence=0.9, class_label="car")
    
    results = verifier.run(query, [c], media_meta, image=get_dummy_image())
    assert results[0].status == ConstraintStatus.VIOLATED

def test_ambiguous_constraint_does_not_become_false_positive():
    verifier = VerificationAgentV2()
    # Attribute passes, but relation is unsupported
    query = QuerySpec(constraints=[
        QueryConstraint(constraint_type="attribute", value="blue"), 
        QueryConstraint(constraint_type="relation", value="following")
    ])
    c = Candidate(id="1", bbox=[10, 10, 100, 100], confidence=0.9, class_label="car")
    
    results = verifier.run(query, [c], image=get_dummy_image())
    assert results[0].status == ConstraintStatus.UNSUPPORTED
    
def test_uncertain_aggregation():
    # If the mock is updated to return UNCERTAIN, test it.
    from v2.models.semantic_adapter import MockSemanticAdapter
    verifier = VerificationAgentV2(semantic_adapter=MockSemanticAdapter(text_score=0.22)) # 0.20 <= 0.22 < 0.25 (UNCERTAIN)
    query = QuerySpec(constraints=[QueryConstraint(constraint_type="attribute", value="blue")])
    c = Candidate(id="1", bbox=[10, 10, 100, 100], confidence=0.9, class_label="car")
    
    results = verifier.run(query, [c], image=get_dummy_image())
    assert results[0].status == ConstraintStatus.UNCERTAIN

def test_planner_reactions():
    planner = PlanningAgentV2()
    from v2.schemas.state import VerificationResult, ConstraintStatus
    
    # 1. No candidates -> REDETECT via ENHANCE_SAHI if rules match, or TERMINATE
    state = AgentStateV2()
    state.plan.history = [ActionType.DETECT_SPECIALIST]
    state.query_spec.target = "person"
    assert planner.run(state) == ActionType.TERMINATE
    
    # 2. Candidates rejected -> VIOLATED
    state = AgentStateV2()
    state.plan.history = [ActionType.VERIFY_CANDIDATES]
    state.verification_results = [
        VerificationResult(candidate_id="1", status=ConstraintStatus.VIOLATED)
    ]
    # First time -> REDETECT
    assert planner.run(state) == ActionType.REDETECT
    
    # 3. Candidates ambiguous -> UNCERTAIN
    state = AgentStateV2()
    state.plan.history = [ActionType.VERIFY_CANDIDATES]
    state.verification_results = [
        VerificationResult(candidate_id="1", status=ConstraintStatus.UNCERTAIN)
    ]
    assert planner.run(state) == ActionType.REDETECT
    
    # 4. Candidates unsupported -> UNSUPPORTED
    state = AgentStateV2()
    state.plan.history = [ActionType.VERIFY_CANDIDATES]
    state.verification_results = [
        VerificationResult(candidate_id="1", status=ConstraintStatus.UNSUPPORTED)
    ]
    assert planner.run(state) == ActionType.TERMINATE
    
    # 5. Candidates accepted -> SATISFIED (TERMINATE for image)
    state = AgentStateV2()
    state.plan.history = [ActionType.VERIFY_CANDIDATES]
    state.media_metadata.type = "IMAGE"
    state.verification_results = [
        VerificationResult(candidate_id="1", status=ConstraintStatus.SATISFIED)
    ]
    assert planner.run(state) == ActionType.TERMINATE
    
def test_retry_budget():
    from v2.schemas.state import VerificationResult
    planner = PlanningAgentV2()
    state = AgentStateV2()
    state.plan.history = [ActionType.VERIFY_CANDIDATES]
    state.verification_results = [
        VerificationResult(candidate_id="1", satisfies_query=False, ambiguity=0.0)
    ]
    
    # Budget starts at 0 -> REDETECT
    action = planner.run(state)
    assert action == ActionType.REDETECT
    assert state.tool_usage.retry_count == 1
    
    # Second time with same state -> retry budget still allows one more
    action = planner.run(state)
    assert action == ActionType.REDETECT
    assert state.tool_usage.retry_count == 2
    
    # Third time -> exhausted budget
    action = planner.run(state)
    assert action == ActionType.TERMINATE
