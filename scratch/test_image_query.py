import argparse
import sys
from pathlib import Path
from workflows.graph import run_pipeline
from utils.model_paths import DEFAULT_YOLO_WORLD_WEIGHTS
from PIL import Image

def main():
    print(f"Default weights: {DEFAULT_YOLO_WORLD_WEIGHTS}")
    
    # We need a reference image. Let's just create a mock image.
    ref_img_path = "scratch/ref_image.jpg"
    img = Image.new('RGB', (100, 100), color = 'red')
    img.save(ref_img_path)
    test_img_path = "scratch/test_image.jpg"
    img.save(test_img_path)

    # Run text query
    print("\n--- Running Text Query ---")
    state, time = run_pipeline("Find cars", test_img_path)
    print(f"Status: {state.mission.status}")
    print(f"Objects found: {len(state.verification.verified_objects)}")

    print("\n--- Running Image Query ---")
    state, time = run_pipeline("", test_img_path, reference_image_path=ref_img_path)
    print(f"Status: {state.mission.status}")
    print(f"Search mode: {state.query.search_mode}")
    print(f"Objects found: {len(state.verification.verified_objects)}")

if __name__ == '__main__':
    main()
