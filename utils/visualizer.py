import cv2
import os


class DetectionVisualizer:

    @staticmethod
    def draw(image_path: str, detections: list, output_path: str):

        image = cv2.imread(image_path)

        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        for detection in detections:

            label = detection.label

            confidence = detection.confidence

            x1, y1, x2, y2 = map(int, detection.bbox)

            text = f"{label} {confidence:.2f}"

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            cv2.putText(
                image,
                text,
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        cv2.imwrite(output_path, image)

        print(f"\nAnnotated image saved to:\n{output_path}")