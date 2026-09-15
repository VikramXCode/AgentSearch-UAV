from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_YOLO_WORLD_WEIGHTS = PROJECT_ROOT / "weights" / "best.pt"
BASELINE_YOLO_WORLD_WEIGHTS = PROJECT_ROOT / "weights" / "yolov8s-world.pt"
YOLO_WORLD_WEIGHTS_ENV = "AGENTSEARCH_YOLO_WORLD_WEIGHTS"

VISDRONE_CLASS_NAMES = [
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
]


def resolve_yolo_world_weights(explicit_path: str | os.PathLike[str] | None = None) -> str:
    candidate = explicit_path or os.environ.get(YOLO_WORLD_WEIGHTS_ENV) or DEFAULT_YOLO_WORLD_WEIGHTS
    candidate_path = Path(candidate)

    if not candidate_path.is_absolute():
        candidate_path = (PROJECT_ROOT / candidate_path).resolve()

    return str(candidate_path)


def is_visdrone_checkpoint(model_or_names_or_path) -> bool:
    """Check if the provided model, names dict/list, or path represents a VisDrone checkpoint."""
    if model_or_names_or_path is None:
        return False

    # Check if a model object with .names attribute
    names = getattr(model_or_names_or_path, "names", None)
    if names is not None:
        if isinstance(names, dict):
            name_set = {str(v).lower() for v in names.values()}
        elif isinstance(names, (list, tuple)):
            name_set = {str(v).lower() for v in names}
        else:
            name_set = set()
        if "awning-tricycle" in name_set or "pedestrian" in name_set and len(name_set) <= 15:
            return True

    # Check if a dict of names directly
    if isinstance(model_or_names_or_path, dict):
        name_set = {str(v).lower() for v in model_or_names_or_path.values()}
        if "awning-tricycle" in name_set or ("pedestrian" in name_set and "motor" in name_set):
            return True

    # Check if string or Path
    path_str = str(model_or_names_or_path).lower()
    if "best.pt" in path_str or "visdrone" in path_str:
        return True

    return False