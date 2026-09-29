import os
import argparse
from pathlib import Path
from PIL import Image

# Setup paths
IMG_PATH = "datasets/VisDrone2019/VisDrone2019-DET-val/images/0000001_02999_d_0000005.jpg"
LBL_PATH = "datasets/VisDrone2019-YOLO/val/labels/0000001_02999_d_0000005.txt"
POS_REF = "scratch/positive_ref.jpg"
NEG_REF = "scratch/negative_ref.jpg"

def main():
    if not os.path.exists(IMG_PATH):
        print(f"Image not found: {IMG_PATH}")
        return
    if not os.path.exists(LBL_PATH):
        print(f"Label not found: {LBL_PATH}")
        return

    # 1. Read image
    img = Image.open(IMG_PATH)
    w, h = img.size

    # 2. Read labels
    with open(LBL_PATH, 'r') as f:
        lines = f.readlines()

    # Find a car (class 3 in VisDrone YOLO typically, but let's check)
    # 0: pedestrian, 1: people, 2: bicycle, 3: car, 4: van, 5: truck, 6: tricycle, 7: awning-tricycle, 8: bus, 9: motor
    pos_box = None
    neg_box = None
    pos_cls = None
    neg_cls = None
    
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5: continue
        cls_id = int(parts[0])
        x_center, y_center, width, height = map(float, parts[1:5])
        
        # Denormalize
        x1 = (x_center - width/2) * w
        y1 = (y_center - height/2) * h
        x2 = (x_center + width/2) * w
        y2 = (y_center + height/2) * h
        
        # Get a decent sized crop for positive (e.g. car class 3)
        if cls_id == 3 and not pos_box and (x2-x1) > 20:
            pos_box = (x1, y1, x2, y2)
            pos_cls = cls_id
        
        # Get a decent sized crop for negative (e.g. pedestrian class 0)
        if cls_id == 0 and not neg_box and (x2-x1) > 10:
            neg_box = (x1, y1, x2, y2)
            neg_cls = cls_id
            
    if not pos_box:
        print("No positive box found")
        return
        
    print(f"Positive crop (class {pos_cls}) box: {pos_box}")
    img.crop(pos_box).save(POS_REF)
    
    if neg_box:
        print(f"Negative crop (class {neg_cls}) box: {neg_box}")
        img.crop(neg_box).save(NEG_REF)

    # 3. Run Pipeline with positive ref
    print("\n" + "="*50)
    print("RUNNING POSITIVE CONTROL")
    print("="*50)
    from workflows.graph import run_pipeline
    
    # We will search for all objects, but use the reference image.
    # The pipeline will extract features from reference image and score all candidates against it.
    state, time_taken = run_pipeline("", IMG_PATH, reference_image_path=POS_REF)
    print(f"Status: {state.mission.status}")
    print(f"Verified Objects: {len(state.verification.verified_objects)}")

    # 4. Run Pipeline with negative ref
    if neg_box:
        print("\n" + "="*50)
        print("RUNNING NEGATIVE CONTROL")
        print("="*50)
        state_neg, time_neg = run_pipeline("", IMG_PATH, reference_image_path=NEG_REF)
        print(f"Status: {state_neg.mission.status}")
        print(f"Verified Objects: {len(state_neg.verification.verified_objects)}")


if __name__ == '__main__':
    main()
