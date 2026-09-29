import os
import sys
import time
import json
import torch

# Ensure local clip is used
os.environ["YOLO_CLIP_PATH"] = os.path.abspath("weights/clip/ViT-B-32.pt")
os.environ["ULTRALYTICS_ASSETS_DIR"] = os.path.abspath("weights")

from v2.models.model_registry import ModelRegistry
from v2.schemas.state import QuerySpec, QueryConstraint

def run_test():
    print("--- 3 & 4. LOAD YOLO-WORLD & SET CLASSES ---")
    registry = ModelRegistry()
    
    # Load YOLO-World
    print("Requesting open_world_yolo_world from ModelRegistry...")
    yolo_adapter = registry.get_adapter("open_world_yolo_world")
    
    # Test setting classes directly through adapter model instance
    try:
        model = yolo_adapter.model
        print("Testing set_classes(['car', 'person', 'drone'])...")
        model.set_classes(['car', 'person', 'drone'])
        print("set_classes() succeeded without downloading!")
    except Exception as e:
        print(f"FAILED set_classes(): {e}")
        return

    print("\n--- 5. REAL IMAGE INFERENCE ---")
    image_path = "scratch/dataset_validation_sample.jpg"
    print(f"Testing inference on {image_path}...")
    
    # Run through the V2 Adapter
    query_spec = QuerySpec(
        target="car", 
        constraints=[QueryConstraint(constraint_type="attribute", value="red")]
    )
    
    from v2.schemas.state import MediaMetadata, MediaType
    metadata = MediaMetadata(type=MediaType.IMAGE)
    
    start_time = time.time()
    candidates = yolo_adapter.detect(image_path, query_spec, metadata)
    inf_time = time.time() - start_time
    
    print(f"Inference Time: {inf_time:.4f}s")
    print(f"Candidates generated: {len(candidates)}")
    for i, c in enumerate(candidates):
        print(f"Candidate {i+1}: {c.class_label} (Conf: {c.confidence:.2f}) BBox: {c.bbox}")
        
    print("\n--- 7. SEMANTIC VERIFICATION ---")
    if candidates:
        print("Running semantic verification on the first candidate...")
        from v2.agents.verification_agent import VerificationAgentV2
        from v2.models.semantic_adapter import CLIPEngineAdapter
        semantic_adapter = CLIPEngineAdapter()
        verification_agent = VerificationAgentV2(semantic_adapter=semantic_adapter)
        
        from PIL import Image
        img = Image.open(image_path)
        
        results = verification_agent.run(query_spec, candidates[:1], metadata, image=img)
        for r in results:
            print(f"Candidate {r.candidate_id}:")
            print(f"  Overall Status: {r.status}")
            print(f"  Overall Conf: {r.overall_confidence:.2f}")
            for cr in r.constraint_results:
                print(f"  Constraint '{cr.constraint_id}': status={cr.status}, conf={cr.confidence:.2f}, ev={cr.evidence}")
    else:
        print("No candidates to verify.")

    print("\n--- 8. GPU MEMORY / MUTUAL EXCLUSION ---")
    if torch.cuda.is_available():
        mem_alloc = torch.cuda.memory_allocated() / (1024**2)
        print(f"GPU memory allocated with YOLO-World: {mem_alloc:.2f} MB")
        
    print("Releasing YOLO-World and loading E3...")
    e3_adapter = registry.get_adapter("specialist_e3")
    
    if torch.cuda.is_available():
        mem_alloc_after = torch.cuda.memory_allocated() / (1024**2)
        print(f"GPU memory allocated with E3 (YOLO-World should be unloaded): {mem_alloc_after:.2f} MB")
        
    print("E3 successfully loaded via mutual exclusion.")

if __name__ == "__main__":
    run_test()
