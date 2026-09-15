from __future__ import annotations

import math
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "datasets" / "VisDrone2019"
EPSILON = 1e-5
SPLITS = {
    "train": {
        "images": DATASET_ROOT / "VisDrone2019-DET-train" / "images",
        "labels": DATASET_ROOT / "VisDrone2019-DET-train" / "labels",
    },
    "val": {
        "images": DATASET_ROOT / "VisDrone2019-DET-val" / "images",
        "labels": DATASET_ROOT / "VisDrone2019-DET-val" / "labels",
    },
}

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
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Validate a VisDrone YOLO dataset.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DATASET_ROOT,
        help="Root directory containing train/ and val/ subdirectories.",
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    split_root_names = {
        "train": "train",
        "val": "val",
    }

    errors: list[str] = []
    class_counts: Counter[int] = Counter()
    total_objects = 0
    total_warnings = 0

    print(f"Dataset root: {dataset_root}")

    if not dataset_root.exists():
        print("FAIL: dataset root does not exist.")
        return 1

    for split_name, split_dir_name in split_root_names.items():
        print(f"\nChecking {split_name} split...")
        split_summary = validate_split(
            dataset_root / split_dir_name / "images",
            dataset_root / split_dir_name / "labels",
            split_name,
        )
        split_errors = split_summary["errors"]
        split_counts = split_summary["class_counts"]
        split_objects = split_summary["instances"]
        split_warnings = split_summary["warnings"]
        errors.extend(split_errors)
        class_counts.update(split_counts)
        total_objects += split_objects
        total_warnings += split_warnings

        print(f"  images: {split_summary['images']}")
        print(f"  labels: {split_summary['labels']}")
        print(f"  instances: {split_objects}")
        print(f"  warnings: {split_warnings}")
        print(f"  errors: {len(split_errors)}")

    print("\nClass distribution:")
    for class_id in range(10):
        print(f"  {class_id} {CLASS_NAMES[class_id]:<16} {class_counts.get(class_id, 0)}")

    if errors:
        print("\nFAIL")
        for error in errors:
            print(f"- {error}")
        return 1

    print("\nPASS")
    print(f"Validated objects: {total_objects}")
    print(f"Warnings: {total_warnings}")
    return 0


def validate_split(images_dir: Path, labels_dir: Path, split_name: str) -> dict[str, object]:
    errors: list[str] = []
    warnings: list[str] = []
    class_counts: Counter[int] = Counter()
    object_count = 0
    warning_count = 0

    if not images_dir.exists():
        errors.append(f"{split_name}: missing images directory: {images_dir}")
        return {
            "errors": errors,
            "warnings": warning_count,
            "class_counts": class_counts,
            "instances": object_count,
            "images": 0,
            "labels": 0,
        }
    if not labels_dir.exists():
        errors.append(f"{split_name}: missing labels directory: {labels_dir}")
        return {
            "errors": errors,
            "warnings": warning_count,
            "class_counts": class_counts,
            "instances": object_count,
            "images": 0,
            "labels": 0,
        }

    image_files = sorted([path for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES])
    label_files = sorted([path for path in labels_dir.iterdir() if path.suffix.lower() == ".txt"])
    image_count = len(image_files)
    label_count = len(label_files)

    if image_count != label_count:
        errors.append(f"{split_name}: image/label count mismatch ({image_count} images vs {label_count} labels)")

    image_stems = {path.stem for path in image_files}
    label_stems = {path.stem for path in label_files}

    missing_labels = sorted(image_stems - label_stems)
    missing_images = sorted(label_stems - image_stems)

    if missing_labels:
        errors.append(f"{split_name}: missing labels for {len(missing_labels)} images (example: {missing_labels[:5]})")
    if missing_images:
        errors.append(f"{split_name}: label files without images for {len(missing_images)} stems (example: {missing_images[:5]})")

    for label_file in label_files:
        file_errors, file_warnings, file_counts, file_objects = validate_label_file(label_file)
        errors.extend(f"{split_name}/{label_file.name}: {message}" for message in file_errors)
        warning_count += len(file_warnings)
        class_counts.update(file_counts)
        object_count += file_objects

    return {
        "errors": errors,
        "warnings": warning_count,
        "class_counts": class_counts,
        "instances": object_count,
        "images": image_count,
        "labels": label_count,
    }


def validate_label_file(label_file: Path) -> tuple[list[str], list[str], Counter[int], int]:
    errors: list[str] = []
    warnings: list[str] = []
    class_counts: Counter[int] = Counter()
    object_count = 0

    raw_text = label_file.read_text(encoding="utf-8").strip()
    if not raw_text:
        errors.append("empty label file")
        return errors, warnings, class_counts, object_count

    for line_number, line in enumerate(raw_text.splitlines(), start=1):
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"line {line_number}: expected 5 fields, got {len(parts)}")
            continue

        class_text, x_text, y_text, w_text, h_text = parts

        try:
            class_id = int(class_text)
        except ValueError:
            errors.append(f"line {line_number}: class id is not an integer: {class_text}")
            continue

        if class_id not in CLASS_NAMES:
            errors.append(f"line {line_number}: class id out of range 0-9: {class_id}")
            continue

        try:
            x_center = float(x_text)
            y_center = float(y_text)
            width = float(w_text)
            height = float(h_text)
        except ValueError:
            errors.append(f"line {line_number}: non-numeric box values")
            continue

        values = [x_center, y_center, width, height]
        if not all(math.isfinite(value) for value in values):
            errors.append(f"line {line_number}: non-finite box values")
            continue

        if width <= 0 or height <= 0:
            errors.append(f"line {line_number}: width/height must be > 0")
            continue

        x1 = x_center - width / 2.0
        y1 = y_center - height / 2.0
        x2 = x_center + width / 2.0
        y2 = y_center + height / 2.0

        coordinates = [x_center, y_center, width, height, x1, y1, x2, y2]
        if any(value < -EPSILON or value > 1.0 + EPSILON for value in coordinates):
            errors.append(f"line {line_number}: box extends outside normalized image bounds")
            continue

        if any(value < 0.0 or value > 1.0 for value in coordinates):
            warnings.append(f"line {line_number}: box touches normalized image bounds within epsilon")

        class_counts[class_id] += 1
        object_count += 1

    return errors, warnings, class_counts, object_count


if __name__ == "__main__":
    raise SystemExit(main())