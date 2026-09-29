import os
import sys

# Ensure YOLO uses local clip
os.environ["YOLO_CLIP_PATH"] = os.path.abspath("weights/clip/ViT-B-32.pt")
os.environ["ULTRALYTICS_ASSETS_DIR"] = os.path.abspath("weights")

from ultralytics import YOLOWorld

def test_yolo_world():
    print(f"Python: {sys.executable}")
    weight_path = "weights/yolov8l-worldv2.pt"
    if not os.path.exists(weight_path):
        print(f"File not found: {weight_path}")
        return
        
    print("Loading YOLO-World...")
    try:
        model = YOLOWorld(weight_path)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
        
    print("Setting classes ['black SUV', 'damaged drone']...")
    try:
        model.set_classes(["black SUV", "damaged drone"])
        print("Classes set successfully!")
    except Exception as e:
        print(f"Failed to set classes: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_yolo_world()
