import json
import os
import cv2
from pathlib import Path
from utils.search_utils import extract_query_components
from v2.agents.query_agent import QueryAgentV2
from v2.schemas.state import QuerySpec, AgentStateV2
from models.color_verifier import ColorVerifier
from v2.models.semantic_adapter import SemanticEmbeddingAdapter
import numpy as np

def search_indexed_dataset(query: str, cache_file: str = "scratch/raw_fused_detections_cache.json", images_dir: str = "datasets/VisDrone2019/VisDrone2019-DET-val/images"):
    print(f"Loading indexed detections from {cache_file}...")
    with open(cache_file, "r") as f:
        cache = json.load(f)
        
    query_agent = QueryAgentV2()
    query_spec = query_agent.run(raw_query=query)
    
    target_class = query_spec.target.lower() if query_spec.target else ""
    attributes = [c.value.lower() for c in query_spec.constraints if c.constraint_type == "attribute"]
    
    print(f"Parsed Query -> Target: '{target_class}', Attributes: {attributes}")
    
    color_verifier = ColorVerifier()
    semantic_adapter = SemanticEmbeddingAdapter()
    
    matching_images = set()
    total_matches = 0
    
    for img_name, detections in cache.items():
        img_path = os.path.join(images_dir, img_name)
        img_matches = []
        
        # We need to load image only if attributes are present and we need to verify
        img = None
        
        for det in detections:
            label = det.get("label", "").lower()
            
            # Exact-label matching avoided: substring match
            if target_class and target_class != "object" and target_class not in label and label not in target_class:
                # Also check synonyms
                from utils.search_utils import canonicalize_target
                if canonicalize_target(target_class) != canonicalize_target(label):
                    continue
                    
            if not attributes:
                img_matches.append(det)
                continue
                
            # If attributes exist, verify them
            if img is None:
                if not os.path.exists(img_path):
                    break
                img = cv2.imread(img_path)
                if img is None:
                    break
                    
            x1, y1, x2, y2 = map(int, det["bbox"])
            # Ensure within bounds
            h, w = img.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 - x1 < 5 or y2 - y1 < 5:
                continue
                
            crop = img[y1:y2, x1:x2]
            
            # Verify attributes
            match = True
            for attr in attributes:
                # Check if it's a color
                from utils.search_utils import COLOR_WORDS, COLOR_ALIASES
                is_color = attr in COLOR_WORDS or attr in COLOR_ALIASES
                if is_color:
                    color_passed, score, _ = color_verifier.verify_color(crop, attr)
                    if not color_passed:
                        # Fallback to CLIP
                        import torch
                        from PIL import Image
                        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                        pil_crop = Image.fromarray(crop_rgb)
                        prompt = f"a {attr} {label}"
                        emb = semantic_adapter.embed_image([pil_crop])
                        txt_emb = semantic_adapter.embed_text([prompt, f"a {label}"])
                        sims = torch.nn.functional.cosine_similarity(emb, txt_emb)
                        if sims[0][0] < sims[0][1]: # Not the attribute
                            match = False
                            break
            
            if match:
                img_matches.append(det)
                
        if img_matches:
            matching_images.add(img_name)
            total_matches += len(img_matches)
            
    print(f"\nFound {total_matches} valid objects across {len(matching_images)} images.")
    return list(matching_images)

if __name__ == "__main__":
    import sys
    query = sys.argv[1] if len(sys.argv) > 1 else "red car"
    results = search_indexed_dataset(query)
    print("Matching images:")
    for r in results:
        print(f" - {r}")
