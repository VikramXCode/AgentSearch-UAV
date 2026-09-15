import os
import tempfile
from pathlib import Path
import numpy as np
import pytest

from evaluation.comprehensive_evaluator import (
    compute_box_iou_matrix,
    compute_coco_101_point_ap,
    evaluate_predictions_comprehensive,
    save_comparison_csv,
    CLASS_NAMES,
)


def test_compute_box_iou_matrix():
    boxes1 = np.array([
        [0.0, 0.0, 10.0, 10.0],
        [10.0, 10.0, 20.0, 20.0],
    ], dtype=np.float32)

    boxes2 = np.array([
        [0.0, 0.0, 10.0, 10.0],   # Exact match with boxes1[0] -> IoU = 1.0
        [0.0, 0.0, 5.0, 10.0],    # Half overlap with boxes1[0] -> IoU = 50 / 100 = 0.5
        [100.0, 100.0, 110.0, 110.0],  # No overlap -> IoU = 0.0
    ], dtype=np.float32)

    iou_mat = compute_box_iou_matrix(boxes1, boxes2)
    assert iou_mat.shape == (2, 3)
    assert np.isclose(iou_mat[0, 0], 1.0)
    assert np.isclose(iou_mat[0, 1], 0.5)
    assert np.isclose(iou_mat[0, 2], 0.0)
    assert np.isclose(iou_mat[1, 0], 0.0)


def test_compute_coco_101_point_ap():
    recalls = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
    precisions = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    ap = compute_coco_101_point_ap(recalls, precisions)
    assert np.isclose(ap, 1.0)

    # Imperfect precision
    precisions_half = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    ap_half = compute_coco_101_point_ap(recalls, precisions_half)
    assert np.isclose(ap_half, 0.5)


def test_evaluate_predictions_comprehensive():
    ground_truths = {
        "img1.jpg": [
            {"class_id": 3, "bbox": [10.0, 10.0, 50.0, 50.0]},  # car
            {"class_id": 0, "bbox": [100.0, 100.0, 130.0, 180.0]},  # pedestrian
        ],
    }

    predictions = {
        "img1.jpg": [
            {"class_id": 3, "confidence": 0.95, "bbox": [10.0, 10.0, 50.0, 50.0]},  # True positive
            {"class_id": 0, "confidence": 0.90, "bbox": [100.0, 100.0, 130.0, 180.0]},  # True positive
            {"class_id": 3, "confidence": 0.40, "bbox": [200.0, 200.0, 250.0, 250.0]},  # False positive
        ],
    }

    results = evaluate_predictions_comprehensive(predictions, ground_truths)
    assert "precision" in results
    assert "recall" in results
    assert "map50" in results
    assert "map50_95" in results
    assert "f1" in results
    assert results["total_tp"] == 2
    assert results["total_fp"] == 1
    assert results["total_gt"] == 2
    assert np.isclose(results["recall"], 1.0)
    assert np.isclose(results["precision"], 2 / 3, atol=1e-3)


def test_save_comparison_csv():
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "test_comparison.csv"
        results_table = [
            {
                "System/Model Name": "Baseline YOLO-World (Zero-Shot)",
                "Precision": 0.3139,
                "Recall": 0.2523,
                "mAP@50": 0.0890,
                "mAP@50-95": 0.0584,
                "F1-Score": 0.2798,
                "extra_field_should_be_ignored": 1234,
            },
            {
                "System/Model Name": "Fine-Tuned YOLO-World",
                "Precision": 0.4377,
                "Recall": 0.5929,
                "mAP@50": 0.3398,
                "mAP@50-95": 0.2306,
                "F1-Score": 0.5036,
            }
        ]

        save_comparison_csv(results_table, csv_path)
        assert csv_path.exists()
        content = csv_path.read_text(encoding="utf-8").splitlines()
        assert content[0] == "System/Model Name,Precision,Recall,mAP@50,mAP@50-95,F1-Score"
        assert len(content) == 3
        assert "extra_field" not in content[0]
