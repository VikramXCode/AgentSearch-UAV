from typing import Dict, List, Optional
from pydantic import BaseModel, Field


# -----------------------------
# Query Agent
# -----------------------------
class QueryState(BaseModel):
    raw_query: str = ""
    target: str = ""
    original_target: str = ""
    attributes: Dict[str, str] = Field(default_factory=dict)
    quantity: str = "all"
    quantity_value: Optional[int] = None
    canonical_query: str = ""
    search_mode: str = "text"


# -----------------------------
# Knowledge Agent
# -----------------------------
class KnowledgeState(BaseModel):
    previous_searches: List[Dict] = Field(default_factory=list)
    known_locations: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


# -----------------------------
# Strategy Agent
# -----------------------------
class StrategyState(BaseModel):

    detector: str = ""

    enable_super_resolution: bool = False

    enable_sahi: bool = False

    enable_clip_verification: bool = False

    enable_tracking: bool = False

    confidence_threshold: float = 0.30
    clip_threshold: float = 0.10
    target_similarity_threshold: float = 0.60
    attribute_similarity_threshold: float = 0.12

    execution_priority: List[str] = Field(default_factory=list)

    reasoning: List[str] = Field(default_factory=list)

# -----------------------------
# Mission Agent
# -----------------------------
class MissionState(BaseModel):

    mission_id: str = ""

    steps: List[str] = Field(default_factory=list)

    current_step: int = 0

    status: str = "Pending"

# -----------------------------
# Detection Agent
# -----------------------------
class DetectionState(BaseModel):
    objects_found: List[Dict] = Field(default_factory=list)
    image_path: Optional[str] = None
    processed_image_path: Optional[str] = None
    output_image_path: Optional[str] = None
    detection_time: float = 0.0


# -----------------------------
# Verification Agent
# -----------------------------
class VerificationState(BaseModel):
    verified_objects: List[Dict] = Field(default_factory=list)
    confidence_score: float = 0.0
    color_filter_time: float = 0.0
    ai_verification_time: float = 0.0
    total_verification_time: float = 0.0


# -----------------------------
# Explanation Agent
# -----------------------------
class ExplanationState(BaseModel):
    summary: str = ""
    reasoning: List[str] = Field(default_factory=list)


# ==========================================================
# COMPLETE SHARED STATE
# ==========================================================
class AgentState(BaseModel):

    query: QueryState = QueryState()

    knowledge: KnowledgeState = KnowledgeState()

    strategy: StrategyState = StrategyState()

    mission: MissionState = MissionState()

    detection: DetectionState = DetectionState()

    verification: VerificationState = VerificationState()

    explanation: ExplanationState = ExplanationState()