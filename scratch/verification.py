import torch
from ultralytics import YOLO
import sys

def verify_checkpoint():
    print("--- 1. YOLOv8s CHECKPOINT ---")
    ckpt_path = "weights/yolov8s.pt"
    
    try:
        model_base = YOLO(ckpt_path)
        base_params = sum(p.numel() for p in model_base.model.parameters())
        print(f"Exact checkpoint path: {ckpt_path}")
        print(f"Model parameter count: {base_params}")
        
        # Check if it is actually YOLOv8s by looking at layers and tasks
        print(f"Is actually YOLOv8s: {'yolov8s' in model_base.ckpt.get('model', '').__class__.__name__.lower() or base_params == 11135987 or base_params == 11166560}")
        # Note: ultralytics might load the actual model structure in `model_base.model`.
        
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return

    print("\n--- 4. PRETRAINED TRANSFER CHECK ---")
    try:
        # Load the newly defined yaml
        model_p2 = YOLO("models/yolov8s-p2.yaml")
        
        # Load the base weights into the p2 model
        transferred = model_p2.load(ckpt_path)
        
        # ultralytics `.load()` prints/returns some dict/list. We can inspect what transferred.
        # Alternatively we can inspect the state_dict manually
        p2_state = model_p2.model.state_dict()
        base_state = model_base.model.state_dict()
        
        transferred_weights = 0
        transferred_params = 0
        newly_init_params = 0
        
        for k, v in p2_state.items():
            if k in base_state and v.shape == base_state[k].shape:
                transferred_weights += 1
                transferred_params += v.numel()
            else:
                newly_init_params += v.numel()
        
        total_p2_params = sum(p.numel() for p in model_p2.model.parameters())
        print(f"Total parameters in YOLOv8s-P2: {total_p2_params}")
        print(f"Number of weights (tensors) successfully transferable: {transferred_weights} out of {len(p2_state)}")
        print(f"Number of transferable parameters: {transferred_params}")
        print(f"Number of newly initialized parameters: {newly_init_params}")
        
    except Exception as e:
        print(f"Failed to check transfer: {e}")

if __name__ == "__main__":
    verify_checkpoint()
