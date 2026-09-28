import pytest
from v2.schemas.state import AgentStateV2, Candidate
from v2.agents.tracking_agent import TrackingAgentV2

def test_first_track_creation():
    tracker = TrackingAgentV2()
    state = AgentStateV2()
    
    candidates = [
        Candidate(id="c1", bbox=[10, 10, 100, 100], class_label="car", confidence=0.9)
    ]
    
    state = tracker.run(state, new_candidates=candidates)
    
    assert state.tracking_state.active_tracks == 1
    assert len(state.candidates) == 1
    assert state.candidates[0].track_id == 1

def test_repeated_detection_preserves_track_id():
    tracker = TrackingAgentV2()
    state = AgentStateV2()
    
    # Frame 1
    c1 = Candidate(id="c1", bbox=[10, 10, 100, 100], class_label="car", confidence=0.9)
    state = tracker.run(state, new_candidates=[c1])
    
    # Frame 2 (Slightly moved)
    c2 = Candidate(id="c2", bbox=[12, 12, 102, 102], class_label="car", confidence=0.85)
    state = tracker.run(state, new_candidates=[c2])
    
    assert state.tracking_state.active_tracks == 1
    assert state.candidates[0].track_id == 1 # Still track 1

def test_new_object_receives_new_track_id():
    tracker = TrackingAgentV2()
    state = AgentStateV2()
    
    # Frame 1
    c1 = Candidate(id="c1", bbox=[10, 10, 100, 100], class_label="car", confidence=0.9)
    state = tracker.run(state, new_candidates=[c1])
    
    # Frame 2 (Completely different location)
    c2 = Candidate(id="c2", bbox=[500, 500, 600, 600], class_label="car", confidence=0.85)
    state = tracker.run(state, new_candidates=[c2])
    
    # Because of low IoU (0.0), it should be a new track
    assert state.tracking_state.active_tracks == 2
    assert {c.track_id for c in state.candidates} == {1, 2}

def test_track_degradation_increases():
    tracker = TrackingAgentV2()
    state = AgentStateV2()
    
    # Frame 1
    c1 = Candidate(id="c1", bbox=[10, 10, 100, 100], class_label="car", confidence=0.9)
    state = tracker.run(state, new_candidates=[c1])
    assert state.tracking_state.track_degradation_score == 0.0
    
    # Frame 2, 3, 4 (No new candidates, tracker coasts)
    state = tracker.run(state, new_candidates=None)
    assert state.tracking_state.track_degradation_score == 0.2  # 1/5 missed
    
    state = tracker.run(state, new_candidates=None)
    assert state.tracking_state.track_degradation_score == 0.4  # 2/5 missed

def test_redetect_reassociates_old_track():
    tracker = TrackingAgentV2()
    state = AgentStateV2()
    
    # Initial detect
    c1 = Candidate(id="c1", bbox=[10, 10, 100, 100], class_label="car", confidence=0.9)
    state = tracker.run(state, new_candidates=[c1])
    
    # Coast for 3 frames (degraded)
    for _ in range(3):
        state = tracker.run(state, new_candidates=None)
        
    assert state.tracking_state.track_degradation_score == 0.6
    assert state.candidates[0].track_id == 1
    
    # REDETECT provides new candidate near the coasted box
    c_new = Candidate(id="c_new", bbox=[15, 15, 105, 105], class_label="car", confidence=0.95)
    state = tracker.run(state, new_candidates=[c_new])
    
    # Re-associated
    assert state.candidates[0].track_id == 1
    assert state.tracking_state.track_degradation_score == 0.0 # reset

def test_lost_track_surfaced_for_redetect():
    tracker = TrackingAgentV2()
    state = AgentStateV2()
    
    # Initial detect
    c1 = Candidate(id="c1", bbox=[10, 10, 100, 100], class_label="car", confidence=0.9)
    state = tracker.run(state, new_candidates=[c1])
    
    # Coast beyond max_missed_frames (5)
    for _ in range(6):
        state = tracker.run(state, new_candidates=None)
        
    # Track should be dead, active_tracks = 0, degradation = 1.0 (very high)
    assert state.tracking_state.active_tracks == 0
    assert state.tracking_state.track_degradation_score == 1.0

def test_multiple_simultaneous_tracks():
    tracker = TrackingAgentV2()
    state = AgentStateV2()
    
    # Frame 1: two cars
    c1 = Candidate(id="c1", bbox=[10, 10, 100, 100], class_label="car", confidence=0.9)
    c2 = Candidate(id="c2", bbox=[200, 200, 300, 300], class_label="car", confidence=0.9)
    state = tracker.run(state, new_candidates=[c1, c2])
    
    assert state.tracking_state.active_tracks == 2
    assert {c.track_id for c in state.candidates} == {1, 2}
    
    # Frame 2: both cars move slightly
    c1_moved = Candidate(id="c3", bbox=[12, 12, 102, 102], class_label="car", confidence=0.9)
    c2_moved = Candidate(id="c4", bbox=[205, 205, 305, 305], class_label="car", confidence=0.9)
    state = tracker.run(state, new_candidates=[c1_moved, c2_moved])
    
    assert state.tracking_state.active_tracks == 2
    track_ids = {c.track_id for c in state.candidates}
    assert track_ids == {1, 2}
