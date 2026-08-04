from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_YOLO_WORLD_WEIGHTS = PROJECT_ROOT / "weights" / "yolov8s-world.pt"
YOLO_WORLD_WEIGHTS_ENV = "AGENTSEARCH_YOLO_WORLD_WEIGHTS"


def resolve_yolo_world_weights(explicit_path: str | os.PathLike[str] | None = None) -> str:
    candidate = explicit_path or os.environ.get(YOLO_WORLD_WEIGHTS_ENV) or DEFAULT_YOLO_WORLD_WEIGHTS
    candidate_path = Path(candidate)

    if not candidate_path.is_absolute():
        candidate_path = (PROJECT_ROOT / candidate_path).resolve()

    return str(candidate_path)