from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from typing import Any, Iterable


DEFAULT_IOU_THRESHOLD = 0.50


def calculate_precision(tp: int, fp: int) -> float:
    """Precision = TP / (TP + FP)."""
    denom = tp + fp
    if denom == 0:
        return 0.0
    return tp / denom


def calculate_recall(tp: int, fn: int) -> float:
    """Recall = TP / (TP + FN)."""
    denom = tp + fn
    if denom == 0:
        return 0.0
    return tp / denom


def calculate_f1(tp: int, fp: int, fn: int) -> float:
    """F1 = 2 * precision * recall / (precision + recall)."""
    precision = calculate_precision(tp, fp)
    recall = calculate_recall(tp, fn)
    denom = precision + recall
    if denom == 0:
        return 0.0
    return 2 * precision * recall / denom


def calculate_iou(box_a: Iterable[float], box_b: Iterable[float]) -> float:
    """Compute IoU for two boxes in [x1, y1, x2, y2] format."""
    a = [float(v) for v in box_a]
    b = [float(v) for v in box_b]
    if len(a) != 4 or len(b) != 4:
        raise ValueError("Boxes must contain exactly four coordinates: [x1, y1, x2, y2]")

    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(0.0, (b[2] - b[0]) * (b[3] - b[1]))
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def _to_box_list(box: Any) -> list[float]:
    if isinstance(box, (list, tuple)):
        return [float(v) for v in box]
    if isinstance(box, dict):
        keys = ["x1", "y1", "x2", "y2"]
        return [float(box.get(k, 0.0)) for k in keys]
    raise TypeError(f"Unsupported box format: {type(box)!r}")


def _normalize_label_name(label: str) -> str:
    return str(label).strip().lower()


def _matching_pairs(predictions: list[dict], ground_truth: list[dict], iou_threshold: float = DEFAULT_IOU_THRESHOLD) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    matched_pred_ids: set[int] = set()
    matched_gt_ids: set[int] = set()
    matched_pairs: list[tuple[int, int]] = []

    for pred_idx, pred in enumerate(predictions):
        if pred_idx in matched_pred_ids:
            continue
        pred_label = _normalize_label_name(pred.get("label", ""))
        pred_box = _to_box_list(pred.get("bbox", [0, 0, 0, 0]))
        best_gt_idx = None
        best_iou = -1.0

        for gt_idx, gt in enumerate(ground_truth):
            if gt_idx in matched_gt_ids:
                continue
            gt_label = _normalize_label_name(gt.get("label", ""))
            if pred_label and gt_label and pred_label != gt_label:
                continue
            gt_box = _to_box_list(gt.get("bbox", [0, 0, 0, 0]))
            iou = calculate_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_gt_idx is not None and best_iou >= iou_threshold:
            matched_pairs.append((pred_idx, best_gt_idx))
            matched_pred_ids.add(pred_idx)
            matched_gt_ids.add(best_gt_idx)

    unmatched_preds = [idx for idx in range(len(predictions)) if idx not in matched_pred_ids]
    unmatched_gts = [idx for idx in range(len(ground_truth)) if idx not in matched_gt_ids]
    return matched_pairs, unmatched_preds, unmatched_gts


def calculate_map50(predictions: list[dict], ground_truth: list[dict], iou_threshold: float = DEFAULT_IOU_THRESHOLD) -> float:
    """Compute a simple detection mAP@50-style score for a single class group or all labels.
    This is an IoU-based detection accuracy estimate for the current project and is not a full COCO-style AP sweep.
    """
    if not predictions and not ground_truth:
        return 0.0
    if not ground_truth:
        return 0.0

    all_labels = sorted({str(item.get("label", "")).lower() for item in predictions + ground_truth if str(item.get("label", "")).strip()})
    if not all_labels:
        return 0.0

    per_class_ap = []
    for label in all_labels:
        pred_for_class = [p for p in predictions if _normalize_label_name(p.get("label", "")) == label]
        gt_for_class = [g for g in ground_truth if _normalize_label_name(g.get("label", "")) == label]

        if not gt_for_class:
            continue

        matched_pairs, unmatched_preds, unmatched_gts = _matching_pairs(pred_for_class, gt_for_class, iou_threshold)
        tp = len(matched_pairs)
        fp = len(unmatched_preds)
        fn = len(unmatched_gts)
        precision = calculate_precision(tp, fp)
        recall = calculate_recall(tp, fn)
        if precision + recall == 0:
            ap = 0.0
        else:
            ap = 2 * precision * recall / (precision + recall)
        per_class_ap.append(ap)

    if not per_class_ap:
        return 0.0
    return sum(per_class_ap) / len(per_class_ap)


def calculate_timing_metrics(processing_time: float | None = None, total_images_tested: int = 0, stage_times: dict[str, float] | None = None) -> dict[str, float | int]:
    """Return timing metrics and FPS. If no processing time is supplied, estimates from stage_times."""
    stage_times = stage_times or {}
    total_time = float(processing_time if processing_time is not None else sum(stage_times.values()))
    if total_images_tested <= 0:
        fps = 0.0
        avg_time_per_image = 0.0
    else:
        fps = total_images_tested / total_time if total_time > 0 else 0.0
        avg_time_per_image = total_time / total_images_tested if total_time > 0 else 0.0

    metrics = {
        "total_inference_time": total_time,
        "average_time_per_image": avg_time_per_image,
        "fps": fps,
        "total_images_tested": total_images_tested,
    }
    for name, value in stage_times.items():
        metrics[f"{name}_time"] = float(value)
    return metrics


def calculate_clip_metrics(clip_scores: Iterable[float]) -> dict[str, float]:
    values = [float(value) for value in clip_scores if value is not None and math.isfinite(float(value))]
    if not values:
        return {
            "avg_clip_similarity": 0.0,
            "min_clip_similarity": 0.0,
            "max_clip_similarity": 0.0,
        }
    return {
        "avg_clip_similarity": sum(values) / len(values),
        "min_clip_similarity": min(values),
        "max_clip_similarity": max(values),
    }


def _safe_average(values: Iterable[float]) -> float:
    xs = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not xs:
        return 0.0
    return sum(xs) / len(xs)


def evaluate_pipeline(
    detections: list[dict] | None = None,
    ground_truth: list[dict] | None = None,
    clip_scores: Iterable[float] | None = None,
    processing_time: float | None = None,
    total_images_tested: int = 0,
    stage_times: dict[str, float] | None = None,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> dict[str, Any]:
    """Evaluate the current pipeline using real detections and optionally matched ground truth annotations."""
    detections = detections or []
    ground_truth = ground_truth or []
    clip_scores = list(clip_scores or [])

    total_detected_objects = len(detections)
    total_ground_truth_objects = len(ground_truth)

    if ground_truth:
        matched_pairs, unmatched_preds, unmatched_gts = _matching_pairs(detections, ground_truth, iou_threshold)
        tp = len(matched_pairs)
        fp = len(unmatched_preds)
        fn = len(unmatched_gts)
        precision = calculate_precision(tp, fp)
        recall = calculate_recall(tp, fn)
        f1 = calculate_f1(tp, fp, fn)
        map50 = calculate_map50(detections, ground_truth, iou_threshold)
        ground_truth_available = True
    else:
        tp = fp = fn = 0
        precision = recall = f1 = map50 = 0.0
        ground_truth_available = False

    avg_confidence = _safe_average([float(item.get("confidence", 0.0)) for item in detections])
    clip_summary = calculate_clip_metrics(clip_scores)
    timing = calculate_timing_metrics(
        processing_time=processing_time,
        total_images_tested=total_images_tested,
        stage_times=stage_times,
    )

    return {
        "ground_truth_available": ground_truth_available,
        "iou_threshold": iou_threshold,
        "total_images_tested": total_images_tested,
        "total_ground_truth_objects": total_ground_truth_objects,
        "total_detected_objects": total_detected_objects,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "map50": map50,
        "avg_detection_confidence": avg_confidence,
        "avg_clip_similarity": clip_summary["avg_clip_similarity"],
        "min_clip_similarity": clip_summary["min_clip_similarity"],
        "max_clip_similarity": clip_summary["max_clip_similarity"],
        "total_inference_time": timing["total_inference_time"],
        "average_time_per_image": timing["average_time_per_image"],
        "fps": timing["fps"],
        **{key: value for key, value in timing.items() if key not in {"total_inference_time", "average_time_per_image", "fps", "total_images_tested"}},
    }


def load_ground_truth_annotations(annotation_path: str | Path) -> list[dict]:
    """Load YOLO-style or COCO-like annotation files when available.
    Supported formats:
      - YOLO .txt file: one object per line, format: class_id x_center y_center width height
      - JSON file: COCO-style dict with 'annotations' and 'images'
    """
    annotation_path = Path(annotation_path)
    if annotation_path.suffix.lower() == ".json":
        with annotation_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        annotations = payload.get("annotations", [])
        results = []
        for item in annotations:
            if isinstance(item, dict):
                bbox = item.get("bbox", [0, 0, 0, 0])
                category = item.get("category_id")
                label = item.get("label") or str(category)
                results.append({
                    "label": str(label),
                    "bbox": [
                        float(bbox[0]),
                        float(bbox[1]),
                        float(bbox[0] + bbox[2]),
                        float(bbox[1] + bbox[3]),
                    ],
                })
        return results

    if annotation_path.suffix.lower() == ".txt":
        entries: list[dict] = []
        text = annotation_path.read_text(encoding="utf-8").strip()
        if not text:
            return entries
        for line in text.splitlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, x_center, y_center, width, height = parts
            x = float(x_center) - float(width) / 2.0
            y = float(y_center) - float(height) / 2.0
            entries.append({
                "label": str(cls),
                "bbox": [x, y, x + float(width), y + float(height)],
            })
        return entries

    return []


def export_metrics_report(metrics: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "Metric": [
            "Precision",
            "Recall",
            "F1-Score",
            "mAP@50",
            "Average Detection Confidence",
            "Average CLIP Similarity",
            "Average Inference Time",
            "FPS",
            "Images Tested",
            "Ground Truth Objects",
            "Detected Objects",
            "TP",
            "FP",
            "FN",
        ],
        "Value": [
            metrics.get("precision", 0.0),
            metrics.get("recall", 0.0),
            metrics.get("f1_score", 0.0),
            metrics.get("map50", 0.0),
            metrics.get("avg_detection_confidence", 0.0),
            metrics.get("avg_clip_similarity", 0.0),
            metrics.get("average_time_per_image", 0.0),
            metrics.get("fps", 0.0),
            metrics.get("total_images_tested", 0),
            metrics.get("total_ground_truth_objects", 0),
            metrics.get("total_detected_objects", 0),
            metrics.get("tp", 0),
            metrics.get("fp", 0),
            metrics.get("fn", 0),
        ],
    }

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Metric", "Value"])
        for metric_name, value in zip(summary["Metric"], summary["Value"]):
            writer.writerow([metric_name, value])

    return path
