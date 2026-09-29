import os
import time
import glob
from PIL import Image
import torch

from v2.models.model_registry import ModelRegistry
from v2.schemas.state import AgentStateV2, ActionType, QuerySpec, MediaMetadata
from v2.agents.planning_agent import PlanningAgentV2
from v2.evaluation.telemetry import TelemetryLogger

def run_experiment():
    print("--- PHASE 7B: CONTROLLED REAL ENHANCEMENT VALIDATION ---")
    
    # 1. Dataset Sample
    val_images_dir = "datasets/VisDrone2019/VisDrone2019-DET-val/images"
    # Find up to 10 images deterministically
    all_images = sorted(glob.glob(os.path.join(val_images_dir, "*.jpg")))
    sample_images = all_images[:10]
    
    print("\nA. EXACT 10 IMAGE FILENAMES:")
    for img in sample_images:
        print(f" - {os.path.basename(img)}")
        
    if not sample_images:
        print("No images found! Stopping.")
        return
        
    # 2. Baseline E3 Inference
    registry = ModelRegistry()
    print("\nLoading E3 Specialist...")
    try:
        e3_adapter = registry.get_adapter("specialist_e3")
    except Exception as e:
        print(f"Failed to load E3 adapter: {e}")
        return
        
    baseline_results = []
    print("\nB. E3 BASELINE RESULTS:")
    for img_path in sample_images:
        img_name = os.path.basename(img_path)
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        
        query = QuerySpec(target="person")  # standard aerial class
        meta = MediaMetadata(resolution=[w, h])
        
        start_time = time.time()
        print(f"Detecting on {img_name}...")
        try:
            candidates = e3_adapter.detect(img, query, meta)
        except Exception as e:
            print(f"Exception during detect: {e}")
            break
        print(f"Detection done for {img_name}.")
        elapsed = time.time() - start_time
        
        count = len(candidates)
        if count > 0:
            confs = [c.confidence for c in candidates]
            max_conf = max(confs)
            confs.sort()
            med_conf = confs[len(confs)//2]
        else:
            max_conf = 0.0
            med_conf = 0.0
            
        print(f"{img_name} ({w}x{h}) - {count} cands | MaxConf: {max_conf:.3f} | MedConf: {med_conf:.3f} | Time: {elapsed:.3f}s")
        baseline_results.append({
            "name": img_name,
            "res": [w, h],
            "count": count,
            "max_conf": max_conf,
            "runtime": elapsed,
            "candidates": candidates
        })
        
    # 3. Real SAHI Detection
    print("\nC. REAL SAHI RESULTS (Integration Check):")
    # We will test SAHI on the first image only for brevity and to avoid long loops if it takes too much time.
    # Actually, instructions say "For each image record: number of slices, base detector candidate count, SAHI candidate count..."
    # We will run SAHI on all 10 images.
    
    from utils.adaptive_sahi import AdaptiveSAHI
    
    # We need a detect_fn for run_sliced_inference
    # We need to map PIL images to models.schemas.Detection for AdaptiveSAHI
    from models.schemas import Detection
    from models.enhanced_postprocessor import EnhancedPostProcessor
    
    def sahi_detect_fn(patches):
        # run E3 on a batch of patches
        # e3_adapter.model expects a list of PIL images or numpy arrays?
        # yolo model takes list of images
        results = e3_adapter.model(patches, verbose=False)
        batch_dets = []
        for r in results:
            dets = []
            for box in r.boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                label = r.names[cls_id]
                dets.append(Detection(
                    label=label,
                    confidence=conf,
                    bbox=[x1, y1, x2, y2]
                ))
            batch_dets.append(dets)
        return batch_dets
        
    sahi = AdaptiveSAHI(default_slice_width=640, default_slice_height=640, default_overlap=0.20)
    
    sahi_stats = []
    
    for i, img_path in enumerate(sample_images):
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        
        start_time = time.time()
        # run_sliced_inference expects `image`, `detect_fn`, `strategy`
        # However, `AdaptiveSAHI` does NOT contain NMS. We will use V1 EnhancedPostProcessor just for validation.
        global_detections = sahi.run_sliced_inference(img, sahi_detect_fn, strategy="standard")
        
        # Merge overlapping detections via NMS
        # Wait, EnhancedPostProcessor uses models.detection_config.DetectionConfig
        try:
            from models.detection_config import DEFAULT_CONFIG
            merged_detections = EnhancedPostProcessor.apply_nms(
                global_detections,
                config=DEFAULT_CONFIG,
                image_width=w,
                image_height=h
            )
            sahi_count = len(merged_detections)
        except Exception as e:
            print(f"Integration limitation during SAHI NMS merge: {e}")
            sahi_count = len(global_detections)  # fallback to unmerged
            
        elapsed = time.time() - start_time
        
        base_count = baseline_results[i]['count']
        base_time = baseline_results[i]['runtime']
        added = sahi_count - base_count
        
        slices_count = len(sahi.generate_slices(img, strategy="standard")[0])
        
        print(f"{os.path.basename(img_path)} | Slices: {slices_count} | Base: {base_count} -> SAHI: {sahi_count} (+{added}) | BaseTime: {base_time:.3f}s -> SAHITime: {elapsed:.3f}s")
        sahi_stats.append({
            "slices": slices_count,
            "base_count": base_count,
            "sahi_count": sahi_count,
            "sahi_time": elapsed
        })
        
    # 4. Real Learned SR Check
    print("\nF. LEARNED SR AVAILABILITY:")
    print("Checking models/super_resolution.py...")
    # As observed in Phase 7A, SuperResolutionEngine only wraps PIL Lanczos interpolation.
    # It does not load an ESRGAN or SwinIR checkpoint.
    print("Learned SR not validated; only interpolation fallback (Lanczos) is available.")

    # 5. Planner Routing Validation
    print("\nH. PLANNER ROUTING RESULTS:")
    planner = PlanningAgentV2()
    for res in baseline_results:
        state = AgentStateV2()
        state.query_spec.target = "person"
        state.media_metadata.resolution = res['res']
        state.candidates = res['candidates']
        
        # Mock initial detection step 
        state.plan.history.append(ActionType.DETECT_SPECIALIST)
        
        action = planner.run(state)
        print(f"Image: {res['name']}, Cands: {res['count']}, MaxConf: {res['max_conf']:.3f} -> Planner Action: {action}")
        
    # 6. Telemetry Validation
    print("\nI. TELEMETRY RESULT:")
    logger = TelemetryLogger("scratch/phase7b_telemetry.jsonl")
    try:
        logger.log_step(
            mission_id="m_val_7b",
            step_index=1,
            query_target="person",
            selected_detector="specialist_e3",
            action="VERIFY_CANDIDATES",
            rationale="Candidates exist. Proceeding to verification.",
            candidate_count=baseline_results[0]['count'],
            median_candidate_area=100.0,
            max_confidence=baseline_results[0]['max_conf'],
            sahi_used=False,
            sr_used=False,
            verification_status="SATISFIED",
            step_runtime_ms=baseline_results[0]['runtime'] * 1000
        )
        print("Successfully wrote telemetry to scratch/phase7b_telemetry.jsonl")
        with open("scratch/phase7b_telemetry.jsonl", "r") as f:
            print(f"Output: {f.read().strip()}")
    except Exception as e:
        print(f"Telemetry write failed: {e}")

if __name__ == "__main__":
    run_experiment()
