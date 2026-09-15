from __future__ import annotations

import argparse
import json
from pathlib import Path


METRIC_KEYS = ["precision", "recall", "map50", "map50_95"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare baseline VisDrone metrics with a fine-tuned evaluation.")
    parser.add_argument("--baseline", type=Path, default=Path("experiments/baseline_visdrone.json"))
    parser.add_argument("--fine-tuned", type=Path, required=True)
    args = parser.parse_args()

    baseline = load_json(args.baseline)
    fine_tuned = load_json(args.fine_tuned)

    print("Metric             Baseline    Fine-tuned    Difference")
    for key in METRIC_KEYS:
        baseline_value = baseline.get("metrics", {}).get(key)
        fine_value = fine_tuned.get("metrics", {}).get(key)
        difference = format_difference(baseline_value, fine_value)
        print(f"{key:<18} {format_value(baseline_value):<10} {format_value(fine_value):<11} {difference}")

    baseline_classes = per_class_map(baseline)
    fine_classes = per_class_map(fine_tuned)

    if baseline_classes and fine_classes:
        print("\nPer-class mAP50-95 comparison:")
        for class_name in sorted(set(baseline_classes) | set(fine_classes)):
            baseline_value = baseline_classes.get(class_name, {}).get("map50_95")
            fine_value = fine_classes.get(class_name, {}).get("map50_95")
            difference = format_difference(baseline_value, fine_value)
            print(f"{class_name:<18} {format_value(baseline_value):<10} {format_value(fine_value):<11} {difference}")

    return 0


def per_class_map(report: dict) -> dict:
    if report.get("per_class"):
        return {item["class_name"]: item for item in report.get("per_class", [])}

    legacy_map50 = report.get("per_class_map50", {})
    return {
        class_name: {
            "class_name": class_name,
            "map50": value,
            "map50_95": None,
        }
        for class_name, value in legacy_map50.items()
    }


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def format_value(value) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int, float)):
        return f"{value:.4f}"
    return str(value)


def format_difference(baseline, fine_tuned) -> str:
    if baseline is None or fine_tuned is None:
        return "n/a"
    return f"{float(fine_tuned) - float(baseline):+.4f}"


if __name__ == "__main__":
    raise SystemExit(main())