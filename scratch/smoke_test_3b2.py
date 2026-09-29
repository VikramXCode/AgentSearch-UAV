import time
import torch
import cv2
import numpy as np
from v2.schemas.state import AgentStateV2, ActionType, MediaMetadata, MediaType
from v2.agents.detection_agent import DetectionAgentV2
from v2.agents.tracking_agent import TrackingAgentV2
from v2.models.model_registry import ModelRegistry

def print_memory(label):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        print(f"[{label}] CUDA Memory - Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")
    else:
        print(f"[{label}] CUDA not available")

def simulate_video_frames(image_path, num_frames=5):
    img = cv2.imread(image_path)
    frames = []
    h, w = img.shape[:2]
    # Create slight artificial panning to simulate video motion
    for i in range(num_frames):
        shift = i * 10
        M = np.float32([[1, 0, shift], [0, 1, 0]])
        frame = cv2.warpAffine(img, M, (w, h))
        frames.append(frame)
    return frames

def main():
    print("=== PHASE 3B-2 SMOKE TEST ===")
    image_path = "scratch/dataset_validation_sample.jpg"
    
    registry = ModelRegistry()
    detection_agent = DetectionAgentV2(registry)
    tracking_agent = TrackingAgentV2()
    
    print("\n==================================================")
    print("PART A: REAL YOLO-WORLD INFERENCE")
    print("==================================================")
    
    print("Skipping YOLO-World inference because CLIP dependency is still corrupted (checksum mismatch).")
    
    # We will clear YOLO-World and load E3 for the video test to avoid memory bloat
    # registry._unload_model("open_world_yolo_world")
    
    print("\n==================================================")
    print("PART B & C: REAL VIDEO TRACKING SMOKE TEST")
    print("==================================================")
    
    frames = simulate_video_frames(image_path, num_frames=10)
    
    state_video = AgentStateV2()
    state_video.media_metadata = MediaMetadata(type=MediaType.VIDEO, total_frames=10, fps=30.0)
    state_video.query_spec.target = "car"
    
    # FRAME 1: Initial Detection
    print("\n[FRAME 1] Initial Detection")
    state_video.plan.current_action = ActionType.DETECT_SPECIALIST
    t2 = time.time()
    state_video = detection_agent.run(state_video, image=frames[0])
    t3 = time.time()
    print(f"Detection Inference Time: {t3 - t2:.3f} s")
    
    state_video = tracking_agent.run(state_video, new_candidates=state_video.candidates)
    
    print(f"Active Tracks: {state_video.tracking_state.active_tracks}")
    print(f"Degradation Score: {state_video.tracking_state.track_degradation_score:.2f}")
    if state_video.candidates:
        print(f"Track ID of first candidate: {state_video.candidates[0].track_id}")
    
    # FRAME 2-4: Simulated coasting (detector is off, tracks just exist and degrade)
    print("\n[FRAME 2-4] Normal Coasting (Tracker only)")
    for i in range(1, 4):
        # We simulate missed detections to force degradation
        state_video = tracking_agent.run(state_video, new_candidates=None)
        
    print(f"Active Tracks: {state_video.tracking_state.active_tracks}")
    print(f"Degradation Score: {state_video.tracking_state.track_degradation_score:.2f}")
    
    # FRAME 5: REDETECT (Planner sees high degradation and requests redetect)
    print("\n[FRAME 5] REDETECT (High Degradation Triggered)")
    state_video.plan.current_action = ActionType.REDETECT
    t4 = time.time()
    state_video = detection_agent.run(state_video, image=frames[4])
    t5 = time.time()
    print(f"REDETECT Inference Time: {t5 - t4:.3f} s")
    
    # Feed the fresh candidates to the tracking agent to test REASSOCIATION
    state_video = tracking_agent.run(state_video, new_candidates=state_video.candidates)
    
    print(f"Active Tracks: {state_video.tracking_state.active_tracks}")
    print(f"Degradation Score: {state_video.tracking_state.track_degradation_score:.2f}")
    if state_video.candidates:
        print(f"Track ID of first candidate after reassociation: {state_video.candidates[0].track_id}")
        
    print_memory("End of Video Test")

if __name__ == "__main__":
    main()
