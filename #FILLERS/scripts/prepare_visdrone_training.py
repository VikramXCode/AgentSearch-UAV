from __future__ import annotations

import json
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "datasets" / "VisDrone2019"
DERIVED_ROOT = PROJECT_ROOT / "datasets" / "VisDrone2019-YOLO"
EPSILON = 1e-5

CLASS_NAMES = {
    0: "pedestrian",
    1: "people",
    2: "bicycle",
    3: "car",
    4: "van",
    5: "truck",
    6: "tricycle",
    7: "awning-tricycle",
    8: "bus",
    9: "motor",
}


@dataclass(frozen=True)
class SplitPaths:
    source_images: Path
    source_labels: Path
    derived_images: Path
    derived_labels: Path


def iter_label_rows(label_file: Path) -> list[str]:
    return [line for line in label_file.read_text(encoding="utf-8").splitlines() if line.strip()]


def symlink_or_copy_image(source: Path, target: Path) -> None:
    if target.exists() or target.is_symlink():
        return
    try:
        os.symlink(source, target)
    except OSError:
        shutil.copy2(source, target)


def copy_validated_labels(source: Path, target: Path) -> tuple[int, list[dict[str, object]]]:
    removed: list[dict[str, object]] = []
    valid_lines: list[str] = []

    for line_number, raw in enumerate(iter_label_rows(source), start=1):
        parts = raw.split()
        if len(parts) != 5:
            removed.append({"line_number": line_number, "label": raw, "reason": "INVALID_ROW_FORMAT"})
            continue

        class_text, x_text, y_text, w_text, h_text = parts

        try:
            class_id = int(class_text)
        except ValueError:
            removed.append({"line_number": line_number, "label": raw, "reason": "INVALID_CLASS_ID"})
            continue

        if class_id < 0 or class_id > 9:
            removed.append({"line_number": line_number, "label": raw, "reason": "INVALID_CLASS_ID"})
            continue

        try:
            x_center = float(x_text)
            y_center = float(y_text)
            width = float(w_text)
            height = float(h_text)
        except ValueError:
            removed.append({"line_number": line_number, "label": raw, "reason": "NON_NUMERIC_VALUE"})
            continue

        values = [x_center, y_center, width, height]
        if any(math.isnan(value) for value in values):
            removed.append({"line_number": line_number, "label": raw, "reason": "NAN_VALUE"})
            continue
        if any(math.isinf(value) for value in values):
            removed.append({"line_number": line_number, "label": raw, "reason": "INFINITY_VALUE"})
            continue
        if width <= 0:
            removed.append({"line_number": line_number, "label": raw, "reason": "NON_POSITIVE_WIDTH"})
            continue
        if height <= 0:
            removed.append({"line_number": line_number, "label": raw, "reason": "NON_POSITIVE_HEIGHT"})
            continue

        x1 = x_center - width / 2.0
        y1 = y_center - height / 2.0
        x2 = x_center + width / 2.0
        y2 = y_center + height / 2.0
        coordinates = [x_center, y_center, width, height, x1, y1, x2, y2]
        if any(value < -EPSILON or value > 1.0 + EPSILON for value in coordinates):
            removed.append({"line_number": line_number, "label": raw, "reason": "OUT_OF_BOUNDS"})
            continue

        valid_lines.append(raw)

    target.write_text("\n".join(valid_lines) + ("\n" if valid_lines else ""), encoding="utf-8")
    return len(valid_lines), removed


def prepare_split(split_name: str) -> dict[str, object]:
    source_images = SOURCE_ROOT / f"VisDrone2019-DET-{split_name}" / "images"
    source_labels = SOURCE_ROOT / f"VisDrone2019-DET-{split_name}" / "labels"
    derived_images = DERIVED_ROOT / split_name / "images"
    derived_labels = DERIVED_ROOT / split_name / "labels"

    derived_images.mkdir(parents=True, exist_ok=True)
    derived_labels.mkdir(parents=True, exist_ok=True)

    image_count = 0
    label_count = 0
    instance_count = 0
    source_instance_count = 0
    removed_rows: list[dict[str, object]] = []

    image_files = sorted([path for path in source_images.iterdir() if path.is_file()])
    label_files = sorted([path for path in source_labels.iterdir() if path.suffix.lower() == ".txt"])

    for image_file in image_files:
        symlink_or_copy_image(image_file, derived_images / image_file.name)
        image_count += 1

    for label_file in label_files:
        target_label = derived_labels / label_file.name
        source_instance_count += len(iter_label_rows(label_file))
        valid_count, removed = copy_validated_labels(label_file, target_label)
        label_count += 1
        instance_count += valid_count
        for entry in removed:
            removed_rows.append({
                "split": split_name,
                "filename": label_file.name,
                "line_number": entry["line_number"],
                "removed_label": entry["label"],
                "reason": entry["reason"],
            })

    return {
        "source_images": image_count,
        "source_labels": label_count,
        "source_instances": source_instance_count,
        "derived_instances": instance_count,
        "removed_annotations": len(removed_rows),
        "removed_rows": removed_rows,
    }


def write_dataset_yaml() -> None:
    yaml_path = PROJECT_ROOT / "configs" / "visdrone_yolo_prepared.yaml"
    yaml_text = """path: datasets/VisDrone2019-YOLO
nc: 10

train: train/images
val: val/images

names:
  0: pedestrian
  1: people
  2: bicycle
  3: car
  4: van
  5: truck
  6: tricycle
  7: awning-tricycle
  8: bus
  9: motor
"""
    yaml_path.write_text(yaml_text, encoding="utf-8")


def main() -> int:
    if not SOURCE_ROOT.exists():
        raise SystemExit(f"Missing source dataset root: {SOURCE_ROOT}")

    split_summaries = {
        split_name: prepare_split(split_name)
        for split_name in ["train", "val"]
    }

    write_dataset_yaml()

    report_path = DERIVED_ROOT / "preparation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "source_train_images": split_summaries["train"]["source_images"],
        "source_train_labels": split_summaries["train"]["source_labels"],
        "source_train_instances": split_summaries["train"]["source_instances"],
        "derived_train_instances": split_summaries["train"]["derived_instances"],
        "source_val_images": split_summaries["val"]["source_images"],
        "source_val_labels": split_summaries["val"]["source_labels"],
        "source_val_instances": split_summaries["val"]["source_instances"],
        "derived_val_instances": split_summaries["val"]["derived_instances"],
        "removed_train_annotations": split_summaries["train"]["removed_annotations"],
        "removed_val_annotations": split_summaries["val"]["removed_annotations"],
        "removed_annotations": split_summaries["train"]["removed_rows"] + split_summaries["val"]["removed_rows"],
        "class_names": CLASS_NAMES,
        "epsilon": EPSILON,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Prepared derived dataset at: {DERIVED_ROOT}")
    print(f"Wrote report: {report_path}")
    print(f"Wrote dataset YAML: {PROJECT_ROOT / 'configs' / 'visdrone_yolo_prepared.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())