import torch
from ultralytics import YOLO

def analyze_transfer():
    print("--- WEIGHT TRANSFER ANALYSIS ---")
    base = YOLO("weights/yolov8s.pt").model.state_dict()
    p2 = YOLO("models/yolov8s-p2.yaml").model.state_dict()
    
    transferred_backbone = []
    transferred_neck = []
    transferred_head = []
    skipped_shape_mismatch = []
    skipped_not_in_base = []
    
    for k, v in p2.items():
        if k in base:
            if v.shape == base[k].shape:
                if k.startswith("model.0.") or k.startswith("model.1.") or k.startswith("model.2.") or \
                   k.startswith("model.3.") or k.startswith("model.4.") or k.startswith("model.5.") or \
                   k.startswith("model.6.") or k.startswith("model.7.") or k.startswith("model.8.") or \
                   k.startswith("model.9."):
                    transferred_backbone.append(k)
                elif k.startswith("model.28."): # Detect head in p2 model is 28
                    transferred_head.append(k)
                else:
                    transferred_neck.append(k)
            else:
                skipped_shape_mismatch.append(k)
        else:
            skipped_not_in_base.append(k)
            
    print(f"Backbone tensors transferred: {len(transferred_backbone)}")
    print(f"Neck tensors transferred: {len(transferred_neck)}")
    print(f"Detect head tensors transferred: {len(transferred_head)}")
    print(f"Tensors skipped due to shape mismatch: {len(skipped_shape_mismatch)}")
    print(f"Newly initialized tensors (not in base): {len(skipped_not_in_base)}")
    
    # Print head specific stuff
    print(f"\nDetect head tensors that transferred: {transferred_head}")
    print(f"Detect head tensors mismatched: {[k for k in skipped_shape_mismatch if '28.' in k]}")
    print(f"Detect head tensors not in base: {[k for k in skipped_not_in_base if '28.' in k]}")

def analyze_checkpointing():
    print("\n--- GRADIENT CHECKPOINTING ANALYSIS ---")
    try:
        from ultralytics.utils import DEFAULT_CFG
        if 'cos_lr' in DEFAULT_CFG.keys():
            pass
        # search ultralytics codebase for gradient_checkpointing
    except ImportError:
        pass

if __name__ == "__main__":
    analyze_transfer()
