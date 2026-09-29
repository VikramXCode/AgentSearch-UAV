import json
import random
from pathlib import Path

from v2.evaluation.calibration import CalibrationRecord, GroundTruthLabel, ThresholdEvaluator

def run_calibration_experiment():
    print("--- SYNTHETIC HARNESS DEMO (REAL INFERENCE UNAVAILABLE/LACKS DATA) ---")
    
    # Text-to-image dataset
    text_records = []
    # Generate some synthetic matching scores (mean 0.45, std 0.1)
    for i in range(10):
        score = max(0.0, min(1.0, random.gauss(0.45, 0.1)))
        text_records.append(CalibrationRecord("text", "synthetic_query", None, f"pos_{i}", GroundTruthLabel.MATCH, score))
    # Generate some synthetic non-matching scores (mean 0.20, std 0.1)
    for i in range(15):
        score = max(0.0, min(1.0, random.gauss(0.20, 0.1)))
        text_records.append(CalibrationRecord("text", "synthetic_query", None, f"neg_{i}", GroundTruthLabel.NON_MATCH, score))

    # Image-to-image dataset
    image_records = []
    # Matching (mean 0.8, std 0.1)
    for i in range(10):
        score = max(0.0, min(1.0, random.gauss(0.80, 0.1)))
        image_records.append(CalibrationRecord("image", None, "ref.jpg", f"pos_{i}", GroundTruthLabel.MATCH, score))
    # Non-matching (mean 0.4, std 0.15)
    for i in range(15):
        score = max(0.0, min(1.0, random.gauss(0.40, 0.15)))
        image_records.append(CalibrationRecord("image", None, "ref.jpg", f"neg_{i}", GroundTruthLabel.NON_MATCH, score))
            
    print(f"Generated {len(text_records)} Text-to-Image synthetic records.")
    print(f"Generated {len(image_records)} Image-to-Image synthetic records.")
    
    print("\n--- TEXT TO IMAGE EVALUATION ---")
    if len(text_records) >= 2:
        evaluator_text = ThresholdEvaluator(text_records)
        print("Distributions:", json.dumps(evaluator_text.get_score_distribution(), indent=2))
        ops = evaluator_text.find_operating_points()
        print("Operating Points:\n", json.dumps(ops, indent=2))

    print("\n--- IMAGE TO IMAGE EVALUATION ---")
    if len(image_records) >= 2:
        evaluator_image = ThresholdEvaluator(image_records)
        print("Distributions:", json.dumps(evaluator_image.get_score_distribution(), indent=2))
        ops = evaluator_image.find_operating_points()
        print("Operating Points:\n", json.dumps(ops, indent=2))

    print("\n--- CONCLUSION ---")
    print("The repository lacks a sufficiently large, trusted attribute/ReID calibration dataset.")
    print("The synthetic scores collected above demonstrate the harness works.")
    print("We STOP here rather than inventing fake semantic ground truth labels.")

if __name__ == "__main__":
    run_calibration_experiment()
