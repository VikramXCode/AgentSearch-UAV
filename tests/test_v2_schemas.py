import pytest
from v2.schemas.state import (
    QuerySpec,
    QueryConstraint,
    MediaMetadata,
    MediaType,
    Candidate,
    VerificationResult,
    ConstraintResult,
)

def test_simple_object_query():
    query = QuerySpec(
        target="person",
        raw_query="find a person"
    )
    assert query.target == "person"
    assert len(query.constraints) == 0

def test_attribute_query():
    constraint = QueryConstraint(constraint_type="attribute", value="white")
    query = QuerySpec(
        target="car",
        constraints=[constraint],
        raw_query="find a white car"
    )
    assert query.target == "car"
    assert query.constraints[0].value == "white"

def test_relationship_query():
    constraint = QueryConstraint(
        constraint_type="relation",
        value="near",
        metadata={"subject": "person", "object": "car"}
    )
    query = QuerySpec(
        target="person",
        constraints=[constraint],
        raw_query="find a person near a car"
    )
    assert query.target == "person"
    assert query.constraints[0].metadata["object"] == "car"

def test_multi_constraint_query():
    c1 = QueryConstraint(constraint_type="attribute", value="orange shirt")
    c2 = QueryConstraint(constraint_type="relation", value="near", metadata={"object": "auto-rickshaw"})
    query = QuerySpec(
        target="person",
        constraints=[c1, c2],
        raw_query="find a person wearing an orange shirt near an auto-rickshaw"
    )
    assert query.target == "person"
    assert len(query.constraints) == 2
    assert query.constraints[0].value == "orange shirt"

def test_reference_image_query():
    query = QuerySpec(
        target="object",
        reference_image_path="/path/to/image.jpg",
        raw_query="find the object shown in this image"
    )
    assert query.reference_image_path == "/path/to/image.jpg"

def test_image_metadata():
    meta = MediaMetadata(
        type=MediaType.IMAGE,
        path="/images/test1.jpg",
        resolution=[1920, 1080]
    )
    assert meta.type == MediaType.IMAGE
    assert meta.resolution == [1920, 1080]

def test_video_metadata():
    meta = MediaMetadata(
        type=MediaType.VIDEO,
        path="/videos/drone_flight.mp4",
        resolution=[3840, 2160],
        total_frames=1500,
        fps=30.0
    )
    assert meta.type == MediaType.VIDEO
    assert meta.total_frames == 1500
    assert meta.fps == 30.0

def test_candidate_with_frame_id():
    candidate = Candidate(
        id="c1",
        bbox=[100.0, 150.0, 200.0, 250.0],
        class_label="car",
        confidence=0.85,
        source="YOLOv8s-P2",
        frame_id=45
    )
    assert candidate.frame_id == 45
    assert candidate.track_id is None

def test_candidate_with_track_id():
    candidate = Candidate(
        id="c2",
        bbox=[120.0, 160.0, 210.0, 260.0],
        class_label="car",
        confidence=0.88,
        source="VideoTracker",
        frame_id=46,
        track_id=12
    )
    assert candidate.track_id == 12

def test_generic_verification_results():
    res1 = ConstraintResult(
        constraint_id="color_red",
        satisfied=False,
        confidence=0.9,
        evidence="HSV check failed"
    )
    res2 = ConstraintResult(
        constraint_id="size_large",
        satisfied=True,
        confidence=0.85,
        evidence="Bounding box ratio matches 'large'"
    )
    verification = VerificationResult(
        candidate_id="c1",
        satisfies_query=False,
        constraint_results=[res1, res2],
        overall_confidence=0.4,
        rejection_reason="Failed color constraint"
    )
    assert len(verification.constraint_results) == 2
    assert verification.satisfies_query is False
    assert verification.rejection_reason == "Failed color constraint"
