import time
from PIL import Image
from utils.adaptive_sahi import AdaptiveSAHI

def run_sahi_smoke_test():
    print("--- ADAPTIVE SAHI SMOKE TEST ---")
    
    image_path = "scratch/dataset_validation_sample.jpg"
    try:
        image = Image.open(image_path)
    except Exception as e:
        print(f"Failed to load image: {e}")
        return
        
    sahi = AdaptiveSAHI(default_slice_width=640, default_slice_height=640, default_overlap=0.20)
    
    start_time = time.time()
    patches, coords = sahi.generate_slices(image, strategy="standard")
    elapsed = time.time() - start_time
    
    print(f"Original image size: {image.size}")
    print(f"Generated {len(patches)} slices.")
    print(f"Slice generation time: {elapsed:.4f}s")
    
    if len(patches) > 0:
        print(f"First patch size: {patches[0].size}")
        print(f"First patch coords: {coords[0]}")
        
    print("Smoke test completed successfully.")

if __name__ == "__main__":
    run_sahi_smoke_test()
