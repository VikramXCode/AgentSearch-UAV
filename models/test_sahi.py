from models.sahi_engine import SAHIEngine


def main():

    detector = SAHIEngine()

    image = input("Image Path: ").strip()

    target = input("Target Object: ").strip()

    detections = detector.detect(image, target)

    print("\nDetected:", len(detections), "objects\n")

    for i, detection in enumerate(detections, start=1):
        print(f"Detection {i}")
        print(detection)
        print()


if __name__ == "__main__":
    main()