import pytest
import os
import tempfile
from pathlib import Path

from v2.evaluation.visdrone_pairs import (
    parse_visdrone_annotation,
    VISDRONE_CLASSES,
    collect_visdrone_crops,
    generate_text_calibration_pairs
)

def test_parse_visdrone_annotation():
    with tempfile.NamedTemporaryFile("w", delete=False) as f:
        # x, y, w, h, score, class, trunc, occ
        f.write("100,100,50,50,1,4,0,0\n") # car
        f.write("200,200,10,10,1,1,0,0\n") # pedestrian (too small - w,h > 10 required? Wait, I wrote > 10, so 10 is not >10. Let's make it 20,20)
        f.write("200,200,20,20,1,1,0,0\n") # pedestrian
        f.write("0,0,50,50,1,0,0,0\n") # ignored (class 0)
        temp_name = f.name
        
    try:
        anns = parse_visdrone_annotation(temp_name)
        assert len(anns) == 2
        assert anns[0]["class_name"] == "car"
        assert anns[0]["bbox"] == (100, 100, 150, 150)
        assert anns[1]["class_name"] == "pedestrian"
    finally:
        os.unlink(temp_name)

def test_generate_text_calibration_pairs():
    dummy_crops = [
        {"image_path": "a.jpg", "image_id": "a", "class_name": "car", "bbox": (0,0,50,50)},
        {"image_path": "b.jpg", "image_id": "b", "class_name": "pedestrian", "bbox": (0,0,50,50)}
    ]
    
    pairs = generate_text_calibration_pairs(dummy_crops, seed=42)
    # 2 crops, each generates 1 POSITIVE, 1 HARD NEG, 1 TRIVIAL NEG
    # car -> positive(car), hard_neg(van/truck/bus), trivial_neg(...)
    # pedestrian -> positive(pedestrian), hard_neg(people), trivial_neg(...)
    assert len(pairs) == 6
    
    positives = [p for p in pairs if p["pair_type"] == "positive"]
    assert len(positives) == 2
    
    hard_negs = [p for p in pairs if p["pair_type"] == "hard_negative"]
    assert len(hard_negs) == 2
    assert hard_negs[0]["query"] in ["van", "truck", "bus"] # for car
    assert hard_negs[1]["query"] in ["people"] # for pedestrian
