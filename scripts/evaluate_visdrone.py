from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


CLASS_NAMES = [
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned YOLO-World checkpoint on VisDrone.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "configs" / "visdrone.yaml")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default=0)
    parser.add_argument("--project", type=Path, default=PROJECT_ROOT / "experiments")
    parser.add_argument("--name", default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"Checkpoint not found: {args.checkpoint}")
        return 1

    try:
        from ultralytics import YOLO
    except ImportError:
        print("Ultralytics is missing. Install requirements before evaluation.")
        return 1

    model = YOLO(str(args.checkpoint))
    metrics = model.val(
        data=str(args.data),
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        split="val",
        project=str(args.project),
        name=args.name or f"eval_{args.checkpoint.stem}",
        exist_ok=True,
        verbose=False,
    )

    report = build_report(args.checkpoint, args.data, metrics)
    output_path = args.json_out or (args.project / f"{args.checkpoint.stem}_visdrone_eval.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report["metrics"], indent=2))
    print(f"Saved evaluation JSON to: {output_path}")
    return 0


def build_report(checkpoint: Path, data_path: Path, metrics) -> dict:
    results_dict = getattr(metrics, "results_dict", {}) or {}
    scalar_metrics = {
        "precision": pick_metric(results_dict, ["metrics/precision(B)", "metrics/precision", "precision"]),
        "recall": pick_metric(results_dict, ["metrics/recall(B)", "metrics/recall", "recall"]),
        "map50": pick_metric(results_dict, ["metrics/mAP50(B)", "metrics/mAP50", "map50"]),
        "map50_95": pick_metric(results_dict, ["metrics/mAP50-95(B)", "metrics/mAP50-95", "map"]),
    }

    per_class = extract_per_class_metrics(metrics)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "checkpoint": str(checkpoint),
        "dataset": str(data_path),
        "model_name": getattr(metrics, "model_name", checkpoint.stem),
        "metrics": scalar_metrics,
        "per_class": per_class,
        "raw_results_dict": to_builtin(results_dict),
    }


def pick_metric(results_dict: dict, keys: list[str]):
    for key in keys:
        if key in results_dict:
            return results_dict[key]
    return None


def extract_per_class_metrics(metrics) -> list[dict]:
    box_metrics = getattr(metrics, "box", None)
    if box_metrics is None:
        return []

    class_ids_raw = getattr(box_metrics, "ap_class_index", None)
    class_ids = list(class_ids_raw) if class_ids_raw is not None else []
    all_ap = getattr(box_metrics, "all_ap", None)
    map50_95_values = getattr(box_metrics, "maps", None)
    map50_values = getattr(box_metrics, "ap50", None)

    per_class: list[dict] = []

    if all_ap is not None and class_ids:
        for row_index, class_id in enumerate(class_ids):
            row = all_ap[row_index]
            row_values = list(row) if hasattr(row, "__iter__") else [row]
            per_class.append(
                {
                    "class_id": int(class_id),
                    "class_name": CLASS_NAMES[int(class_id)] if int(class_id) < len(CLASS_NAMES) else str(class_id),
                    "map50": float(row_values[0]) if row_values else None,
                    "map50_95": float(sum(row_values) / len(row_values)) if row_values else None,
                }
            )
        return per_class

    if map50_95_values is None:
        return []

    map50_95_values = list(map50_95_values)
    if map50_values is not None:
        map50_values = list(map50_values)

    for class_id, map50_95 in enumerate(map50_95_values):
        per_class.append(
            {
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else str(class_id),
                "map50": float(map50_values[class_id]) if map50_values is not None and class_id < len(map50_values) else None,
                "map50_95": float(map50_95),
            }
        )

    return per_class


def to_builtin(value):
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            pass
    return value


if __name__ == "__main__":
    raise SystemExit(main())