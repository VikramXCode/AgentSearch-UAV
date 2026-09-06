from __future__ import annotations

import argparse
import random
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "datasets" / "VisDrone2019"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "dataset_validation"
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
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize a few VisDrone YOLO labels.")
    parser.add_argument("--split", choices=["train", "val", "both"], default="both")
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()

    try:
        import cv2  # type: ignore
    except ImportError:
        print("OpenCV is missing. Install requirements before running label visualization.")
        return 1

    splits = [args.split] if args.split != "both" else ["train", "val"]
    rng = random.Random(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for split in splits:
        images_dir = DATASET_ROOT / f"VisDrone2019-DET-{split}" / "images"
        labels_dir = DATASET_ROOT / f"VisDrone2019-DET-{split}" / "labels"
        image_files = sorted(path for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)

        if not image_files:
            print(f"No images found for split {split}")
            continue

        sample_size = min(args.count, len(image_files))
        selected = rng.sample(image_files, sample_size)

        for image_path in selected:
            label_path = labels_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                print(f"Skipping {image_path.name}: missing label file")
                continue

            output_path = args.output_dir / f"{split}_{image_path.stem}_labels.jpg"
            draw_labels(cv2, image_path, label_path, output_path)
            print(f"Saved {output_path}")

    return 0


def draw_labels(cv2, image_path: Path, label_path: Path, output_path: Path) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    height, width = image.shape[:2]
    raw_text = label_path.read_text(encoding="utf-8").strip()

    if raw_text:
        for line in raw_text.splitlines():
            class_id, x_center, y_center, box_width, box_height = line.split()
            class_index = int(class_id)
            x_center = float(x_center) * width
            y_center = float(y_center) * height
            box_width = float(box_width) * width
            box_height = float(box_height) * height

            x1 = max(0, int(round(x_center - box_width / 2.0)))
            y1 = max(0, int(round(y_center - box_height / 2.0)))
            x2 = min(width - 1, int(round(x_center + box_width / 2.0)))
            y2 = min(height - 1, int(round(y_center + box_height / 2.0)))

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{CLASS_NAMES[class_index]} ({class_index})"
            cv2.putText(
                image,
                label,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


if __name__ == "__main__":
    raise SystemExit(main())