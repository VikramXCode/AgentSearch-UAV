from typing import Dict, List, Optional, Any
from enum import Enum
from pydantic import BaseModel, Field, field_validator

# ==========================================
# ENUMS
# ==========================================

class ActionType(str, Enum):
    DETECT_SPECIALIST = "DETECT_SPECIALIST"
    DETECT_OPEN_WORLD = "DETECT_OPEN_WORLD"
    VERIFY_CANDIDATES = "VERIFY_CANDIDATES"
    ENHANCE_SAHI = "ENHANCE_SAHI"
    ENHANCE_SR = "ENHANCE_SR"
    TRACK = "TRACK"
    REDETECT = "REDETECT"
    TERMINATE = "TERMINATE"

class MediaType(str, Enum):
    IMAGE = "IMAGE"
    VIDEO = "VIDEO"

class MissionStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"

# ==========================================
# QUERY SCHEMAS
# ==========================================

class QueryConstraint(BaseModel):
    constraint_type: str  # e.g. "attribute", "spatial", "relation", "context"
    value: str            # e.g. "red", "bottom left", "moving fast"
    metadata: Dict[str, Any] = Field(default_factory=dict)

class QuerySpec(BaseModel):
    target: str = ""
    constraints: List[QueryConstraint] = Field(default_factory=list)
    parts: List[str] = Field(default_factory=list)
    reference_image_path: Optional[str] = None
    ambiguity_score: float = Field(0.0, ge=0.0, le=1.0)
    raw_query: str = ""

# ==========================================
# MEDIA SCHEMAS
# ==========================================

class MediaMetadata(BaseModel):
    type: MediaType = MediaType.IMAGE
    path: str = ""
    resolution: Optional[List[int]] = None
    total_frames: Optional[int] = None
    fps: Optional[float] = None

# ==========================================
# PLAN & ACTION SCHEMAS
# ==========================================

class Plan(BaseModel):
    current_action: ActionType = ActionType.TERMINATE
    history: List[ActionType] = Field(default_factory=list)
    decision_rationale: str = ""

# ==========================================
# DETECTOR ROUTING SCHEMAS
# ==========================================

class DetectorRouting(BaseModel):
    selected_detector: str = ""
    reason: str = ""

# ==========================================
# CANDIDATE & VERIFICATION SCHEMAS
# ==========================================

class Candidate(BaseModel):
    id: str = ""
    bbox: List[float] = Field(default_factory=list, min_length=4, max_length=4)
    class_label: str = ""
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    source: str = ""
    frame_id: Optional[int] = None
    track_id: Optional[int] = None
    crop_ref: Optional[str] = None

    @field_validator('bbox')
    def validate_bbox(cls, v):
        if len(v) == 4:
            x1, y1, x2, y2 = v
            if x1 > x2 or y1 > y2:
                raise ValueError("bbox coordinates must be [x1, y1, x2, y2] where x2 >= x1 and y2 >= y1")
        return v

class ConstraintStatus(str, Enum):
    SATISFIED = "SATISFIED"
    VIOLATED = "VIOLATED"
    UNCERTAIN = "UNCERTAIN"
    UNSUPPORTED = "UNSUPPORTED"

class ConstraintResult(BaseModel):
    constraint_id: str = ""
    status: ConstraintStatus = ConstraintStatus.UNCERTAIN
    satisfied: bool = False
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    evidence: str = ""

class VerificationResult(BaseModel):
    candidate_id: str
    status: ConstraintStatus = ConstraintStatus.UNCERTAIN
    satisfies_query: bool = False
    constraint_results: List[ConstraintResult] = Field(default_factory=list)
    overall_confidence: float = Field(0.0, ge=0.0, le=1.0)
    ambiguity: float = Field(0.0, ge=0.0, le=1.0)
    rejection_reason: Optional[str] = None

# ==========================================
# TRACKING SCHEMAS
# ==========================================

class TrackingState(BaseModel):
    active_tracks: int = 0
    frames_processed: int = 0
    track_degradation_score: float = Field(0.0, ge=0.0, le=1.0)

class ToolUsage(BaseModel):
    sahi_used: bool = False
    sr_used: bool = False
    retry_count: int = 0

# ==========================================
# MAIN AGENT STATE
# ==========================================

class AgentStateV2(BaseModel):
    mission_id: str = ""
    query_spec: QuerySpec = Field(default_factory=QuerySpec)
    media_metadata: MediaMetadata = Field(default_factory=MediaMetadata)
    
    plan: Plan = Field(default_factory=Plan)
    detector_routing: DetectorRouting = Field(default_factory=DetectorRouting)
    tool_usage: ToolUsage = Field(default_factory=ToolUsage)
    
    candidates: List[Candidate] = Field(default_factory=list)
    verification_results: List[VerificationResult] = Field(default_factory=list)
    verified_targets: List[Candidate] = Field(default_factory=list)
    
    tracking_state: TrackingState = Field(default_factory=TrackingState)
    
    status: MissionStatus = MissionStatus.PENDING
    failure_reason: Optional[str] = None
