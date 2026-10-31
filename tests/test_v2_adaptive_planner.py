import pytest
from v2.schemas.state import AgentStateV2, ActionType, MediaType, Candidate, VerificationResult, ConstraintResult, ConstraintStatus, QuerySpec, MediaMetadata
from v2.agents.planning_agent import PlanningAgentV2

@pytest.fixture
def planner():
    return PlanningAgentV2()

def test_initial_known_target(planner):
    state = AgentStateV2()
    state.query_spec.target = "car"
    action = planner.run(state)
    assert action == ActionType.DETECT_SPECIALIST

def test_initial_unknown_target(planner):
    state = AgentStateV2()
    state.query_spec.target = "unknown_drone"
    action = planner.run(state)
    assert action == ActionType.DETECT_OPEN_WORLD

def test_zero_candidates_small_image_no_sahi(planner):
    state = AgentStateV2()
    state.plan.history.append(ActionType.DETECT_SPECIALIST)
    state.media_metadata.resolution = [1000, 1000]
    action = planner.run(state)
    assert action == ActionType.TERMINATE

def test_zero_candidates_large_image_sahi(planner):
    state = AgentStateV2()
    state.plan.history.append(ActionType.DETECT_SPECIALIST)
    state.media_metadata.resolution = [2500, 2500]
    action = planner.run(state)
    assert action == ActionType.TERMINATE

def test_sahi_already_used(planner):
    state = AgentStateV2()
    state.plan.history.append(ActionType.DETECT_SPECIALIST)
    state.media_metadata.resolution = [2500, 2500]
    state.tool_usage.sahi_used = True
    action = planner.run(state)
    assert action == ActionType.TERMINATE

def test_tiny_weak_candidate_sr(planner):
    state = AgentStateV2()
    state.plan.history.append(ActionType.DETECT_SPECIALIST)
    # Area = 10x10 = 100 < 1024, Conf = 0.3 < 0.4
    c = Candidate(id="1", bbox=[0, 0, 10, 10], confidence=0.3)
    state.candidates.append(c)
    action = planner.run(state)
    assert action == ActionType.VERIFY_CANDIDATES

def test_sr_already_used(planner):
    state = AgentStateV2()
    state.plan.history.append(ActionType.DETECT_SPECIALIST)
    c = Candidate(id="1", bbox=[0, 0, 10, 10], confidence=0.3)
    state.candidates.append(c)
    state.tool_usage.sr_used = True
    action = planner.run(state)
    assert action == ActionType.VERIFY_CANDIDATES

def test_satisfied_image(planner):
    state = AgentStateV2()
    state.plan.history.append(ActionType.VERIFY_CANDIDATES)
    v = VerificationResult(candidate_id="1", status=ConstraintStatus.SATISFIED)
    state.verification_results.append(v)
    state.media_metadata.type = MediaType.IMAGE
    action = planner.run(state)
    assert action == ActionType.TERMINATE

def test_satisfied_video(planner):
    state = AgentStateV2()
    state.plan.history.append(ActionType.VERIFY_CANDIDATES)
    v = VerificationResult(candidate_id="1", status=ConstraintStatus.SATISFIED)
    state.verification_results.append(v)
    state.media_metadata.type = MediaType.VIDEO
    action = planner.run(state)
    assert action == ActionType.TRACK

def test_violated_with_retry(planner):
    state = AgentStateV2()
    state.plan.history.append(ActionType.VERIFY_CANDIDATES)
    v = VerificationResult(candidate_id="1", status=ConstraintStatus.VIOLATED)
    state.verification_results.append(v)
    state.tool_usage.retry_count = 0
    action = planner.run(state)
    assert action == ActionType.REDETECT
    assert state.tool_usage.retry_count == 1

def test_uncertain_with_retry(planner):
    state = AgentStateV2()
    state.plan.history.append(ActionType.VERIFY_CANDIDATES)
    v = VerificationResult(candidate_id="1", status=ConstraintStatus.UNCERTAIN)
    state.verification_results.append(v)
    state.tool_usage.retry_count = 0
    action = planner.run(state)
    assert action == ActionType.REDETECT
    assert state.tool_usage.retry_count == 1

def test_unsupported_no_retry(planner):
    state = AgentStateV2()
    state.plan.history.append(ActionType.VERIFY_CANDIDATES)
    v = VerificationResult(candidate_id="1", status=ConstraintStatus.UNSUPPORTED)
    state.verification_results.append(v)
    state.tool_usage.retry_count = 0
    action = planner.run(state)
    assert action == ActionType.TERMINATE

def test_tracking_degradation(planner):
    state = AgentStateV2()
    state.plan.history.append(ActionType.TRACK)
    state.tracking_state.track_degradation_score = 0.8
    action = planner.run(state)
    assert action == ActionType.REDETECT
