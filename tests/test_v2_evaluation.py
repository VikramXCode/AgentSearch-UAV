import pytest
from v2.evaluation.calibration import CalibrationRecord, GroundTruthLabel, ThresholdEvaluator

def test_metrics_calculation():
    records = [
        CalibrationRecord(query_type="text", query_text="car", reference_image_path=None, candidate_crop_info="1", ground_truth=GroundTruthLabel.MATCH, similarity_score=0.30),
        CalibrationRecord(query_type="text", query_text="car", reference_image_path=None, candidate_crop_info="2", ground_truth=GroundTruthLabel.MATCH, similarity_score=0.26),
        CalibrationRecord(query_type="text", query_text="car", reference_image_path=None, candidate_crop_info="3", ground_truth=GroundTruthLabel.NON_MATCH, similarity_score=0.15),
        CalibrationRecord(query_type="text", query_text="car", reference_image_path=None, candidate_crop_info="4", ground_truth=GroundTruthLabel.NON_MATCH, similarity_score=0.20),
    ]
    
    evaluator = ThresholdEvaluator(records)
    
    # At threshold 0.25, 0.30 and 0.26 are TP. 0.15 and 0.20 are TN.
    metrics = evaluator.calculate_metrics_at_threshold(0.25)
    assert metrics["TP"] == 2
    assert metrics["TN"] == 2
    assert metrics["FP"] == 0
    assert metrics["FN"] == 0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    
    # At threshold 0.18, TP=2, FP=1 (0.20), TN=1 (0.15)
    metrics_low = evaluator.calculate_metrics_at_threshold(0.18)
    assert metrics_low["TP"] == 2
    assert metrics_low["FP"] == 1
    assert metrics_low["TN"] == 1
    assert metrics_low["FN"] == 0
    assert metrics_low["precision"] == 2.0 / 3.0
    assert metrics_low["recall"] == 1.0
    
def test_sweep_thresholds():
    records = [
        CalibrationRecord(query_type="text", query_text="car", reference_image_path=None, candidate_crop_info="1", ground_truth=GroundTruthLabel.MATCH, similarity_score=0.8),
        CalibrationRecord(query_type="text", query_text="car", reference_image_path=None, candidate_crop_info="2", ground_truth=GroundTruthLabel.NON_MATCH, similarity_score=0.2),
    ]
    evaluator = ThresholdEvaluator(records)
    sweep = evaluator.sweep_thresholds(steps=5)
    assert len(sweep) == 5
    
def test_operating_points():
    records = [
        CalibrationRecord(query_type="text", query_text="car", reference_image_path=None, candidate_crop_info="1", ground_truth=GroundTruthLabel.MATCH, similarity_score=0.8),
        CalibrationRecord(query_type="text", query_text="car", reference_image_path=None, candidate_crop_info="2", ground_truth=GroundTruthLabel.NON_MATCH, similarity_score=0.2),
    ]
    evaluator = ThresholdEvaluator(records)
    points = evaluator.find_operating_points()
    assert "balanced_f1" in points
