from models.detector import DetectionEngine
from utils.visualizer import DetectionVisualizer


def main():

    detector = DetectionEngine()

    image_path = input("Image Path: ").strip()

    target = input("Target Object: ").strip()

    result = detector.detect(
        image_path=image_path,
        target=target,
    )

    print("\n==============================")
    print("MODEL INFORMATION")
    print("==============================")

    print(f"Model          : {result.model_name}")
    print(f"Inference Time : {result.inference_time:.3f} sec")
    print(f"Image Size     : {result.image_width} x {result.image_height}")

    print("\n==============================")
    print("DETECTIONS")
    print("==============================")

    if len(result.filtered_detections) == 0:
        print("No objects found.")
        return

    for i, detection in enumerate(result.filtered_detections, start=1):

        print(f"\nDetection {i}")
        print(f"Label      : {detection.label}")
        print(f"Confidence : {detection.confidence:.3f}")
        print(f"BBox       : {detection.bbox}")

    output_path = "outputs/detection_result.jpg"

    DetectionVisualizer.draw(
        image_path=image_path,
        detections=result.filtered_detections,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()