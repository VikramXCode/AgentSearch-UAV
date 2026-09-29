import json
import sys
from pathlib import Path
from PIL import Image

from v2.evaluation.visdrone_pairs import collect_visdrone_crops, generate_text_calibration_pairs
from v2.evaluation.calibration import CalibrationRecord, GroundTruthLabel, ThresholdEvaluator
from v2.models.semantic_adapter import CLIPEngineAdapter

def run_experiment():
    print("=== PHASE 5E: VISDRONE CATEGORY-SEMANTIC CALIBRATION ===")
    print(f"Interpreter: {sys.executable}")
    
    dataset_path = "datasets/VisDrone2019" # Contains VisDrone2019
    max_per_class = 20 # 20 per class means up to 200 crops
    
    try:
        crops = collect_visdrone_crops(dataset_path, max_per_class=max_per_class)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
        
    print(f"Collected {len(crops)} crops from VisDrone2019-DET-val")
    
    pairs = generate_text_calibration_pairs(crops)
    print(f"Generated {len(pairs)} calibration pairs")
    
    # Analyze composition
    pos_count = sum(1 for p in pairs if p["ground_truth"] == "MATCH")
    neg_count = sum(1 for p in pairs if p["ground_truth"] == "NON_MATCH")
    hard_neg_count = sum(1 for p in pairs if p["pair_type"] == "hard_negative")
    
    print(f"Positive pairs: {pos_count}")
    print(f"Negative pairs: {neg_count} (of which {hard_neg_count} are hard negatives)")
    
    # Load Semantic Adapter
    print("Loading CLIP via SemanticAdapter...")
    try:
        adapter = CLIPEngineAdapter()
    except Exception as e:
        print(f"Failed to load CLIP: {e}")
        return
        
    records = []
    
    # Process pairs
    # Keep track of loaded images to avoid re-reading
    image_cache = {}
    
    print("Scoring pairs...")
    for i, pair in enumerate(pairs):
        img_path = pair["crop"]["image_path"]
        if img_path not in image_cache:
            try:
                image_cache[img_path] = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"Failed to load {img_path}: {e}")
                continue
                
        img = image_cache[img_path]
        x1, y1, x2, y2 = pair["crop"]["bbox"]
        
        # Safe crop
        w, h = img.size
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue
            
        crop_img = img.crop((x1, y1, x2, y2))
        
        score = adapter.score_image_against_text(crop_img, pair["query"])
        
        gt = GroundTruthLabel.MATCH if pair["ground_truth"] == "MATCH" else GroundTruthLabel.NON_MATCH
        
        # Save record
        records.append(
            CalibrationRecord(
                query_type="text",
                query_text=pair["query"],
                reference_image_path=None,
                candidate_crop_info=f"{pair['crop']['image_id']}_{pair['crop']['class_name']}",
                ground_truth=gt,
                similarity_score=score
            )
        )
        
        if (i+1) % 50 == 0:
            print(f"  Processed {i+1}/{len(pairs)} pairs")
            
    print(f"Scoring complete. Generated {len(records)} valid records.")
    
    evaluator = ThresholdEvaluator(records)
    
    # Save raw records to scratch/
    out_records = [
        {
            "query": r.query_text,
            "crop_id": r.candidate_crop_info,
            "ground_truth": r.ground_truth.value,
            "score": r.similarity_score
        }
        for r in records
    ]
    with open("scratch/visdrone_calibration_scores_5e.json", "w") as f:
        json.dump(out_records, f, indent=2)
        
    # Get metrics
    dist = evaluator.get_score_distribution()
    ops = evaluator.find_operating_points()
    
    with open("scratch/visdrone_calibration_metrics_5e.json", "w") as f:
        json.dump({
            "distributions": dist,
            "operating_points": ops
        }, f, indent=2)
        
    print("\n--- RESULTS ---")
    print(f"Distributions (summary): MATCH avg={sum(dist['MATCH'])/len(dist['MATCH']):.3f} (n={len(dist['MATCH'])}), NON_MATCH avg={sum(dist['NON_MATCH'])/len(dist['NON_MATCH']):.3f} (n={len(dist['NON_MATCH'])})")
    
    for op_name, metrics in ops.items():
        print(f"Candidate ({op_name}): Threshold {metrics['threshold']:.3f} -> F1 {metrics['f1']:.3f} (Prec: {metrics['precision']:.3f}, Rec: {metrics['recall']:.3f})")

if __name__ == "__main__":
    run_experiment()
