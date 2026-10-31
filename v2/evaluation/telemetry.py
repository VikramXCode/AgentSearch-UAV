import json
import time
import os
from typing import Dict, Any, Optional
from datetime import datetime

class TelemetryLogger:
    """
    Lightweight JSONL telemetry logger for V2 adaptive planning and evaluation.
    """
    def __init__(self, log_path: str = "v2/evaluation/telemetry.jsonl"):
        self.log_path = log_path
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(self.log_path)), exist_ok=True)
        
    def log_step(
        self,
        mission_id: str,
        step_index: int,
        query_target: str,
        selected_detector: str,
        action: str,
        rationale: str,
        candidate_count: int,
        median_candidate_area: float,
        max_confidence: float,
        sahi_used: bool,
        sr_used: bool,
        verification_status: str,
        step_runtime_ms: float
    ):
        record = {
            "mission_id": mission_id,
            "step_index": step_index,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "query_target": query_target,
            "selected_detector": selected_detector,
            "action": action,
            "rationale": rationale,
            "candidate_count": candidate_count,
            "median_candidate_area": median_candidate_area,
            "max_confidence": max_confidence,
            "sahi_used": sahi_used,
            "sr_used": sr_used,
            "verification_status": verification_status,
            "step_runtime_ms": step_runtime_ms
        }
        
        try:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            # Safe for unit tests
            print(f"Warning: Failed to write telemetry: {e}")
