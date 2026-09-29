import time
import torch
import gc
from v2.schemas.state import AgentStateV2, ActionType, MediaMetadata, MediaType
from v2.agents.detection_agent import DetectionAgentV2
from v2.models.model_registry import ModelRegistry

def print_memory(label):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        print(f"[{label}] CUDA Memory - Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")
    else:
        print(f"[{label}] CUDA not available")

def bypass_clip_download(url, root):
    import os
    print(f"[Mock] Bypassing CLIP download for {url}, using existing file.")
    return os.path.join(root, "ViT-B-32.pt")

def main():
    import unittest.mock
    # Patch clip download to prevent unauthorized downloads due to checksum mismatch
    import clip.clip
    clip.clip._download = bypass_clip_download
    
    print("=== PHASE 3B-1 SMOKE TEST ===")
    image_path = "scratch/dataset_validation_sample.jpg"
    
    registry = ModelRegistry()
    detection_agent = DetectionAgentV2(registry)
    
    print_memory("Initial State")
    
    # 1. SPECIALIST MODEL TEST
    print("\n--- 1. SPECIALIST MODEL TEST ---")
    state_e3 = AgentStateV2()
    state_e3.plan.current_action = ActionType.DETECT_SPECIALIST
    state_e3.media_metadata = MediaMetadata(type=MediaType.IMAGE, path=image_path)
    state_e3.query_spec.target = "car"
    
    t0 = time.time()
    state_e3 = detection_agent.run(state_e3, image=image_path)
    t1 = time.time()
    
    print(f"Loaded: {registry.get_loaded_models()}")
    print_memory("After E3 Load & Inference")
    print(f"Inference Time: {t1 - t0:.3f} s")
    print(f"Candidates generated: {len(state_e3.candidates)}")
    if state_e3.candidates:
        c = state_e3.candidates[0]
        print(f"Sample Candidate: {c.model_dump()}")
        print(f"BBox valid: {c.bbox[2] >= c.bbox[0] and c.bbox[3] >= c.bbox[1]}")
        
    # 2. OPEN-WORLD MODEL TEST & MUTUAL EXCLUSION
    print("\n--- 2. OPEN-WORLD MODEL TEST & MUTUAL EXCLUSION ---")
    state_ow = AgentStateV2()
    state_ow.plan.current_action = ActionType.DETECT_OPEN_WORLD
    state_ow.media_metadata = MediaMetadata(type=MediaType.IMAGE, path=image_path)
    state_ow.query_spec.target = "truck"
    
    t2 = time.time()
    try:
        state_ow = detection_agent.run(state_ow, image=image_path)
        t3 = time.time()
        print(f"Inference Time: {t3 - t2:.3f} s")
        print(f"Candidates generated: {len(state_ow.candidates)}")
        if state_ow.candidates:
            c = state_ow.candidates[0]
            print(f"Sample Candidate: {c.model_dump()}")
            print(f"BBox valid: {c.bbox[2] >= c.bbox[0] and c.bbox[3] >= c.bbox[1]}")
    except Exception as e:
        print(f"Open-World Inference Failed: {e}")
        
    print(f"Loaded: {registry.get_loaded_models()}")
    print_memory("After YOLO-World Load & Inference")

    print("\n--- 3. UNLOAD ALL ---")
    registry._unload_model("open_world_yolo_world")
    print(f"Loaded: {registry.get_loaded_models()}")
    print_memory("After Unload")
    
if __name__ == "__main__":
    main()
