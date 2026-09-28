import os
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict
import random
from PIL import Image

# VisDrone object categories
VISDRONE_CLASSES = {
    1: "pedestrian",
    2: "people",
    3: "bicycle",
    4: "car",
    5: "van",
    6: "truck",
    7: "tricycle",
    8: "awning-tricycle",
    9: "bus",
    10: "motor"
}

HARD_NEGATIVES_MAP = {
    "car": ["van", "truck", "bus"],
    "van": ["car", "truck", "bus"],
    "truck": ["car", "van", "bus"],
    "bus": ["car", "van", "truck"],
    "pedestrian": ["people"],
    "people": ["pedestrian"],
    "bicycle": ["motor", "tricycle"],
    "motor": ["bicycle", "tricycle"],
    "tricycle": ["awning-tricycle", "bicycle", "motor"],
    "awning-tricycle": ["tricycle", "motor"]
}

def parse_visdrone_annotation(file_path: str) -> List[Dict]:
    annotations = []
    if not os.path.exists(file_path):
        return annotations
    
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 6:
                try:
                    category_id = int(parts[5])
                    if category_id in VISDRONE_CLASSES:
                        x = int(parts[0])
                        y = int(parts[1])
                        w = int(parts[2])
                        h = int(parts[3])
                        
                        # Only take valid boxes
                        if w > 10 and h > 10:
                            annotations.append({
                                "category_id": category_id,
                                "class_name": VISDRONE_CLASSES[category_id],
                                "bbox": (x, y, x + w, y + h)
                            })
                except ValueError:
                    pass
    return annotations

def collect_visdrone_crops(dataset_path: str, max_per_class: int = 50, seed: int = 42) -> List[Dict]:
    """
    Collects a deterministic subset of crop metadata from VisDrone.
    """
    random.seed(seed)
    
    val_dir = Path(dataset_path) / "VisDrone2019-DET-val"
    images_dir = val_dir / "images"
    annotations_dir = val_dir / "annotations"
    
    if not images_dir.exists() or not annotations_dir.exists():
        raise FileNotFoundError(f"VisDrone validation set not found at {val_dir}")
        
    class_crops = defaultdict(list)
    image_files = sorted(list(images_dir.glob("*.jpg")))
    
    for img_path in image_files:
        ann_path = annotations_dir / f"{img_path.stem}.txt"
        if ann_path.exists():
            anns = parse_visdrone_annotation(str(ann_path))
            for ann in anns:
                if len(class_crops[ann["class_name"]]) < max_per_class:
                    crop_info = {
                        "image_path": str(img_path),
                        "image_id": img_path.name,
                        "class_name": ann["class_name"],
                        "bbox": ann["bbox"]
                    }
                    class_crops[ann["class_name"]].append(crop_info)
                    
        # Early exit if all classes are full
        all_full = True
        for c in VISDRONE_CLASSES.values():
            if len(class_crops[c]) < max_per_class:
                all_full = False
                break
        if all_full:
            break
            
    # Flatten
    all_crops = []
    for crops in class_crops.values():
        all_crops.extend(crops)
        
    # Sort for determinism
    all_crops.sort(key=lambda x: (x["image_id"], x["bbox"]))
    return all_crops

def generate_text_calibration_pairs(crops: List[Dict], seed: int = 42) -> List[Dict]:
    """
    Generates deterministic MATCH, NON_MATCH, and HARD_NEGATIVE pairs.
    """
    random.seed(seed)
    pairs = []
    
    class_to_crops = defaultdict(list)
    for crop in crops:
        class_to_crops[crop["class_name"]].append(crop)
        
    for crop in crops:
        true_class = crop["class_name"]
        
        # 1. MATCH pair
        pairs.append({
            "query": true_class,
            "crop": crop,
            "ground_truth": "MATCH",
            "pair_type": "positive"
        })
        
        # 2. HARD NEGATIVE pair (if available)
        hard_negs = HARD_NEGATIVES_MAP.get(true_class, [])
        if hard_negs:
            hard_class = random.choice(hard_negs)
            pairs.append({
                "query": hard_class,
                "crop": crop,
                "ground_truth": "NON_MATCH",
                "pair_type": "hard_negative"
            })
            
        # 3. TRIVIAL NEGATIVE pair
        all_classes = list(VISDRONE_CLASSES.values())
        trivial_candidates = [c for c in all_classes if c != true_class and c not in hard_negs]
        if trivial_candidates:
            trivial_class = random.choice(trivial_candidates)
            pairs.append({
                "query": trivial_class,
                "crop": crop,
                "ground_truth": "NON_MATCH",
                "pair_type": "trivial_negative"
            })
            
    return pairs
