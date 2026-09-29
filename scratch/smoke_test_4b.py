import os
from PIL import Image
from v2.schemas.state import QuerySpec, QueryConstraint, Candidate, MediaMetadata
from v2.agents.verification_agent import VerificationAgentV2
from v2.models.semantic_adapter import CLIPEngineAdapter

def run_smoke_test():
    print("=== PHASE 4B: REAL SEMANTIC VERIFICATION SMOKE TEST ===")
    
    # Check if CLIP is available
    try:
        import clip
    except ImportError:
        print("[FAIL] CLIP dependency missing or corrupted. Skipping real smoke test.")
        return
        
    print("1. Loading Real CLIP Engine Adapter...")
    adapter = CLIPEngineAdapter()
    verifier = VerificationAgentV2(semantic_adapter=adapter)
    
    # Load test image
    base_img_path = "scratch/dataset_validation_sample.jpg"
    ref_img_path = "scratch/positive_ref.jpg"
    
    if not os.path.exists(base_img_path) or not os.path.exists(ref_img_path):
        print(f"[FAIL] Missing test images.")
        return
        
    image = Image.open(base_img_path).convert("RGB")
    media_meta = MediaMetadata(resolution=list(image.size))
    
    # We define two synthetic candidates on the image
    # We don't need real YOLO detections, just bounding boxes to crop
    
    # Candidate 1: Let's assume it's a vehicle (we just guess a box)
    # The exact content doesn't matter, just that they are different
    c1 = Candidate(id="box_1", bbox=[100, 100, 300, 300], confidence=0.9, class_label="unknown")
    
    # Candidate 2: Different part of image (e.g. background or another object)
    c2 = Candidate(id="box_2", bbox=[800, 500, 1000, 700], confidence=0.9, class_label="unknown")
    
    print("\n--- TEST A & B: Semantic and Reference Image Similarity ---")
    query1 = QuerySpec(
        constraints=[QueryConstraint(constraint_type="attribute", value="a photo of a car")],
        reference_image_path=ref_img_path
    )
    
    results = verifier.run(query1, [c1, c2], media_metadata=media_meta, image=image)
    
    for res in results:
        print(f"\nCandidate: {res.candidate_id}")
        for c_res in res.constraint_results:
            print(f"  Constraint '{c_res.constraint_id}': Satisfied={c_res.satisfied}, Evidence='{c_res.evidence}'")
        print(f"  Overall Candidate Satisfied: {res.satisfies_query}")
        
    print("\n--- TEST C & D: Multi-constraint Verification ---")
    # Add a spatial constraint that one of them passes and the other fails
    # c1 cx=200, w=image.width. If left, cx < w/2
    query2 = QuerySpec(
        constraints=[
            QueryConstraint(constraint_type="attribute", value="a photo of a vehicle"),
            QueryConstraint(constraint_type="spatial", value="left")
        ]
    )
    
    results2 = verifier.run(query2, [c1, c2], media_metadata=media_meta, image=image)
    for res in results2:
        print(f"\nCandidate: {res.candidate_id}")
        for c_res in res.constraint_results:
            print(f"  Constraint '{c_res.constraint_id}': Satisfied={c_res.satisfied}, Evidence='{c_res.evidence}'")
        print(f"  Overall Candidate Satisfied: {res.satisfies_query}, Ambiguity: {res.ambiguity}")

if __name__ == "__main__":
    run_smoke_test()
