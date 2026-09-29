import time
import os
from PIL import Image

def run_sr_smoke_test():
    print("--- ADAPTIVE SR SMOKE TEST ---")
    
    image_path = "scratch/dataset_validation_sample.jpg"
    try:
        image = Image.open(image_path)
    except Exception as e:
        print(f"Failed to load image: {e}")
        return
        
    # Create a small crop
    crop = image.crop((500, 500, 564, 564)) # 64x64
    print(f"Original crop size: {crop.size}")
    
    try:
        from models.super_resolution import SuperResolutionEngine
    except ImportError as e:
        print(f"Failed to import SuperResolutionEngine: {e}")
        return
        
    print("Loading SR Engine...")
    start_time = time.time()
    try:
        sr_engine = SuperResolutionEngine()
    except Exception as e:
        print(f"SR MODEL LOADING FAILED: {e}")
        return
    print(f"SR Engine loaded in {time.time() - start_time:.4f}s")
    
    print("Enhancing crop...")
    start_time = time.time()
    try:
        enhanced_crop = sr_engine.upscale_pil(crop)
    except Exception as e:
        print(f"SR INFERENCE FAILED: {e}")
        return
    elapsed = time.time() - start_time
    
    print(f"Enhanced crop size: {enhanced_crop.size}")
    print(f"SR inference time: {elapsed:.4f}s")
    
    print("Smoke test completed successfully.")

if __name__ == "__main__":
    run_sr_smoke_test()
