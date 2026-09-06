import math

import pytest

from metrics.evaluator import (
    calculate_f1,
    calculate_precision,
    calculate_recall,
    calculate_iou,
    evaluate_pipeline,
)
from models.video_tracker import VideoTracker


def test_basic_detection_metrics():
    assert calculate_precision(10, 2) == pytest.approx(0.8333333333)
    assert calculate_recall(10, 3) == pytest.approx(0.7692307692)
    assert calculate_f1(10, 2, 3) == pytest.approx(0.8)


def test_iou_calculation():
    box_a = [0, 0, 10, 10]
    box_b = [5, 5, 15, 15]
    iou = calculate_iou(box_a, box_b)
    assert iou > 0.0
    assert iou < 1.0
    assert math.isfinite(iou)


def test_evaluate_pipeline_without_ground_truth():
    result = evaluate_pipeline(
        detections=[
            {"label": "car", "confidence": 0.9, "bbox": [0, 0, 10, 10]},
            {"label": "truck", "confidence": 0.8, "bbox": [20, 20, 30, 30]},
        ],
        clip_scores=[0.12, 0.15],
        processing_time=1.5,
        total_images_tested=1,
    )

    assert result["total_detected_objects"] == 2
    assert result["ground_truth_available"] is False
    assert result["avg_detection_confidence"] == pytest.approx(0.85)
    assert result["avg_clip_similarity"] == pytest.approx(0.135)
    assert result["fps"] == pytest.approx(0.6666666667)


def test_select_best_target_prefers_large_central_vehicle():
    frame_shape = (720, 1280)
    detections = [
        {"label": "car", "confidence": 0.82, "bbox": [10, 10, 60, 45]},
        {"label": "car", "confidence": 0.91, "bbox": [500, 310, 780, 530]},
        {"label": "car", "confidence": 0.89, "bbox": [1180, 10, 1270, 120]},
    ]

    best = VideoTracker._select_best_detection(frame_shape, detections, "car")
    assert best["bbox"] == pytest.approx([500, 310, 780, 530])


def test_filter_detections_removes_tiny_and_edge_boxes():
    frame_shape = (720, 1280)
    detections = [
        {"label": "car", "confidence": 0.73, "bbox": [10, 10, 30, 22]},
        {"label": "car", "confidence": 0.91, "bbox": [380, 200, 780, 500]},
        {"label": "car", "confidence": 0.87, "bbox": [1230, 30, 1275, 120]},
        {"label": "truck", "confidence": 0.88, "bbox": [610, 250, 980, 520]},
    ]

    kept = VideoTracker._filter_detections(frame_shape, detections, "car")
    assert len(kept) == 2
    assert kept[0]["bbox"] == pytest.approx([380, 200, 780, 500])
    assert kept[1]["bbox"] == pytest.approx([610, 250, 980, 520])


def test_select_best_detections_keeps_multiple_query_matches():
    frame_shape = (720, 1280)
    detections = [
        {"label": "car", "confidence": 0.82, "bbox": [40, 40, 170, 120]},
        {"label": "car", "confidence": 0.91, "bbox": [500, 310, 780, 530]},
        {"label": "car", "confidence": 0.89, "bbox": [900, 330, 1100, 540]},
        {"label": "truck", "confidence": 0.93, "bbox": [100, 100, 250, 200]},
    ]

    ranked = VideoTracker._select_best_detections(frame_shape, detections, "car", max_targets=None)
    assert len(ranked) == 4
    assert [item["bbox"] for item in ranked] == [
        pytest.approx([500, 310, 780, 530]),
        pytest.approx([900, 330, 1100, 540]),
        pytest.approx([100, 100, 250, 200]),
        pytest.approx([40, 40, 170, 120]),
    ]


def test_select_best_detections_keeps_all_valid_matches_for_many_cars():
    frame_shape = (720, 1280)
    detections = [
        {"label": "car", "confidence": 0.71, "bbox": [50, 50, 150, 140]},
        {"label": "car", "confidence": 0.74, "bbox": [220, 60, 330, 160]},
        {"label": "car", "confidence": 0.78, "bbox": [420, 70, 530, 180]},
        {"label": "car", "confidence": 0.82, "bbox": [610, 80, 720, 190]},
        {"label": "car", "confidence": 0.77, "bbox": [820, 90, 930, 200]},
    ]

    ranked = VideoTracker._select_best_detections(frame_shape, detections, "car")
    assert len(ranked) == 5
