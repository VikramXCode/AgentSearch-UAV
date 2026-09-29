import os
import sys

# Ensure local clip is used
os.environ["YOLO_CLIP_PATH"] = os.path.abspath("weights/clip/ViT-B-32.pt")
os.environ["ULTRALYTICS_ASSETS_DIR"] = os.path.abspath("weights")

def test_environment():
    print("--- 2. VERIFY PYTHON ENVIRONMENT ---")
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    
    try:
        import clip
        print("CLIP package import status: SUCCESS")
    except ImportError:
        print("CLIP package import status: FAILED")
        
    import ultralytics
    print(f"YOLO/Ultralytics version: {ultralytics.__version__}")
    
    import torch
    print(f"CUDA availability: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU name: {torch.cuda.get_device_name(0)}")

if __name__ == "__main__":
    test_environment()
