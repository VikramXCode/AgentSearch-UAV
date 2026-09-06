#!/usr/bin/env python3
"""
Comprehensive Evaluation Engine for AgentSearch-UAV.

Standardized benchmarking framework supporting baseline vs optimized pipeline
comparison across PR-curve metrics, operating points, scale stratification,
confusion matrices, and runtime latency.
"""

from __future__ import annotations

import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
from tqdm import tqdm

from models.schemas import Detection

# Resolve project root (supporting both direct and #FILLERS nested paths)
_current_dir = Path(__file__).resolve().parent
PROJECT_ROOT = _current_dir.parent.parent if _current_dir.parent.name == "#FILLERS" else _current_dir.parent

CLASS_NAMES = [
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
]

IOU_THRESHOLDS = np.linspace(0.50, 0.95, 10)


def compute_box_iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)

    x1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    y1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    x2 = min_x = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    y2 = min_y = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])

    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1[:, None] + area2[None, :] - inter_area

    iou = np.zeros_like(inter_area)
    valid = union > 0
    iou[valid] = inter_area[valid] / union[valid]
    return iou


def compute_coco_101_point_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    recall_levels = np.linspace(0.0, 1.0, 101)
    inds = np.searchsorted(mrec, recall_levels, side="left")
    inds = np.clip(inds, 0, len(mpre) - 1)
    return float(np.mean(mpre[inds]))


class ComprehensiveEvaluator:
    """Evaluates detection pipelines against VisDrone ground truth."""

    def __init__(
        self,
        val_dir: Optional[Union[str, Path]] = None,
        iou_threshold: float = 0.50,
    ):
        if val_dir:
            self.val_dir = Path(val_dir)
        else:
            candidates = [
                PROJECT_ROOT / "#FILLERS" / "datasets" / "VisDrone2019" / "VisDrone2019-DET-val",
                PROJECT_ROOT / "datasets" / "VisDrone2019" / "VisDrone2019-DET-val",
            ]
            self.val_dir = next((c for c in candidates if c.exists()), candidates[0])
        self.iou_threshold = iou_threshold
        self.ground_truths, self.image_metadata = self._load_ground_truth()

    def _load_ground_truth(self) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Dict[str, Any]]]:
        images_dir = self.val_dir / "images"
        annotations_dir = self.val_dir / "annotations"
        all_gt: Dict[str, List[Dict[str, Any]]] = {}
        image_meta: Dict[str, Dict[str, Any]] = {}

        image_paths = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))

        for img_path in image_paths:
            with Image.open(img_path) as img:
                w, h = img.size

            boxes: List[Dict[str, Any]] = []
            ann_file = annotations_dir / f"{img_path.stem}.txt"
            if ann_file.exists():
                for line in ann_file.read_text(encoding="utf-8").splitlines():
                    parts = [p.strip() for p in line.split(",") if p.strip()]
                    if len(parts) >= 6:
                        try:
                            left, top, width, height, score, category = map(float, parts[:6])
                            cat_int = int(category)
                            if 1 <= cat_int <= 10 and score > 0:
                                cls_id = cat_int - 1
                                x1 = max(0.0, left)
                                y1 = max(0.0, top)
                                x2 = min(float(w), left + width)
                                y2 = min(float(h), top + height)
                                bw = max(0.0, x2 - x1)
                                bh = max(0.0, y2 - y1)
                                area = bw * bh
                                size_cat = "small" if area < 32 * 32 else ("medium" if area <= 96 * 96 else "large")

                                boxes.append({
                                    "class_id": cls_id,
                                    "class_name": CLASS_NAMES[cls_id],
                                    "bbox": [x1, y1, x2, y2],
                                    "area": area,
                                    "size_cat": size_cat,
                                })
                        except (ValueError, IndexError):
                            continue

            all_gt[img_path.name] = boxes
            image_meta[img_path.name] = {"width": w, "height": h, "num_objects": len(boxes)}

        return all_gt, image_meta

    def evaluate_predictions(
        self,
        predictions: Dict[str, List[Dict[str, Any]]],
        system_name: str = "Optimized Pipeline",
        latencies: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate full prediction dictionary against loaded ground truth.
        """
        eval_images = [img for img in predictions.keys() if img in self.ground_truths]
        num_classes = len(CLASS_NAMES)
        BG_CLASS_ID = num_classes

        # 1. Prepare GT and Pred mappings
        gt_by_img_cls: Dict[Tuple[str, int], List[np.ndarray]] = {}
        total_gt = np.zeros(num_classes, dtype=int)

        for img in eval_images:
            for g in self.ground_truths[img]:
                c = g["class_id"]
                if 0 <= c < num_classes:
                    total_gt[c] += 1
                    gt_by_img_cls.setdefault((img, c), []).append(np.array(g["bbox"], dtype=np.float32))

        preds_by_cls: Dict[int, List[Dict[str, Any]]] = {c: [] for c in range(num_classes)}
        for img in eval_images:
            for p in predictions.get(img, []):
                c = p["class_id"] if "class_id" in p else CLASS_NAMES.index(p.get("label", "car"))
                if 0 <= c < num_classes:
                    preds_by_cls[c].append({
                        "img": img,
                        "score": float(p["confidence"]),
                        "bbox": np.array(p["bbox"], dtype=np.float32),
                    })

        for c in range(num_classes):
            preds_by_cls[c].sort(key=lambda x: x["score"], reverse=True)

        # 2. PR-Curve Computation across 10 IoU thresholds
        ap_matrix = np.zeros((num_classes, len(IOU_THRESHOLDS)), dtype=np.float64)
        class_metrics: List[Dict[str, Any]] = []

        total_tp = 0
        total_fp = 0

        for c in range(num_classes):
            c_name = CLASS_NAMES[c]
            c_preds = preds_by_cls[c]
            n_gt = int(total_gt[c])

            if n_gt == 0 or len(c_preds) == 0:
                class_metrics.append({
                    "class_id": c,
                    "class_name": c_name,
                    "ap50": 0.0,
                    "ap50_95": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "gt_count": n_gt,
                    "pred_count": len(c_preds),
                    "tp": 0,
                    "fp": len(c_preds),
                    "fn": n_gt,
                })
                continue

            c_boxes = np.array([p["bbox"] for p in c_preds], dtype=np.float32)
            c_imgs = [p["img"] for p in c_preds]

            img_to_indices: Dict[str, List[int]] = {}
            for idx, img in enumerate(c_imgs):
                img_to_indices.setdefault(img, []).append(idx)

            for t_idx, iou_thresh in enumerate(IOU_THRESHOLDS):
                tp = np.zeros(len(c_preds), dtype=np.float64)
                fp = np.zeros(len(c_preds), dtype=np.float64)

                for (img, cls_id), gt_boxes_list in gt_by_img_cls.items():
                    if cls_id != c or img not in img_to_indices:
                        continue

                    gt_boxes = np.array(gt_boxes_list, dtype=np.float32)
                    p_indices = img_to_indices[img]
                    p_boxes = c_boxes[p_indices]

                    iou_mat = compute_box_iou_matrix(p_boxes, gt_boxes)
                    matched_gt = set()

                    for local_i, global_i in enumerate(p_indices):
                        ious = iou_mat[local_i]
                        best_g = -1
                        best_iou = 0.0
                        for g_i, iou_val in enumerate(ious):
                            if iou_val > best_iou:
                                best_iou = iou_val
                                best_g = g_i

                        if best_iou >= iou_thresh and best_g not in matched_gt:
                            tp[global_i] = 1.0
                            matched_gt.add(best_g)
                        else:
                            fp[global_i] = 1.0

                for p_i, img in enumerate(c_imgs):
                    if (img, c) not in gt_by_img_cls:
                        fp[p_i] = 1.0

                tp_cumsum = np.cumsum(tp)
                fp_cumsum = np.cumsum(fp)
                recalls = tp_cumsum / max(1, n_gt)
                precisions = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, np.finfo(np.float64).eps)

                ap = compute_coco_101_point_ap(recalls, precisions)
                ap_matrix[c, t_idx] = ap

                if np.isclose(iou_thresh, self.iou_threshold):
                    c_tp = int(np.sum(tp))
                    c_fp = int(np.sum(fp))
                    c_fn = max(0, n_gt - c_tp)
                    total_tp += c_tp
                    total_fp += c_fp

                    p_val = float(c_tp / (c_tp + c_fp)) if (c_tp + c_fp) > 0 else 0.0
                    r_val = float(c_tp / n_gt) if n_gt > 0 else 0.0
                    f1_val = float(2 * p_val * r_val / (p_val + r_val)) if (p_val + r_val) > 0 else 0.0

                    class_metrics.append({
                        "class_id": c,
                        "class_name": c_name,
                        "ap50": round(ap, 4),
                        "ap50_95": 0.0,
                        "precision": round(p_val, 4),
                        "recall": round(r_val, 4),
                        "f1": round(f1_val, 4),
                        "gt_count": n_gt,
                        "pred_count": len(c_preds),
                        "tp": c_tp,
                        "fp": c_fp,
                        "fn": c_fn,
                    })

        for item in class_metrics:
            c = item["class_id"]
            item["ap50_95"] = round(float(np.mean(ap_matrix[c])), 4)

        # 3. Overall Macro Metrics
        overall_map50 = round(float(np.mean(ap_matrix[:, 0])), 4)
        overall_map50_95 = round(float(np.mean(ap_matrix)), 4)
        tot_gt_all = int(sum(total_gt))
        tot_fn_all = tot_gt_all - total_tp
        overall_precision = round(float(total_tp / (total_tp + total_fp)), 4) if (total_tp + total_fp) > 0 else 0.0
        overall_recall = round(float(total_tp / tot_gt_all), 4) if tot_gt_all > 0 else 0.0
        overall_f1 = round(float(2 * overall_precision * overall_recall / (overall_precision + overall_recall)), 4) if (overall_precision + overall_recall) > 0 else 0.0

        # 4. Scale-Stratified Performance
        size_scales = ["small", "medium", "large"]
        size_stats: Dict[str, Dict[str, Any]] = {
            s: {"gt": 0, "tp": 0, "fp": 0, "fn": 0} for s in size_scales
        }

        for img in eval_images:
            gts = self.ground_truths.get(img, [])
            preds = predictions.get(img, [])

            for s in size_scales:
                s_gts = [g for g in gts if g["size_cat"] == s]
                s_preds = [p for p in preds if (p.get("scale_category") == s or ((p["bbox"][2]-p["bbox"][0])*(p["bbox"][3]-p["bbox"][1]) < 32*32 if s=="small" else (((p["bbox"][2]-p["bbox"][0])*(p["bbox"][3]-p["bbox"][1]) <= 96*96) if s=="medium" else True)))]

                size_stats[s]["gt"] += len(s_gts)
                if not s_gts and not s_preds:
                    continue
                if s_gts and not s_preds:
                    size_stats[s]["fn"] += len(s_gts)
                    continue
                if not s_gts and s_preds:
                    size_stats[s]["fp"] += len(s_preds)
                    continue

                g_boxes = np.array([g["bbox"] for g in s_gts], dtype=np.float32)
                p_boxes = np.array([p["bbox"] for p in s_preds], dtype=np.float32)
                iou_mat = compute_box_iou_matrix(p_boxes, g_boxes)

                matched_g = set()
                p_sorted = np.argsort([-float(p["confidence"]) for p in s_preds])
                for p_i in p_sorted:
                    ious = iou_mat[p_i]
                    best_g_idx = -1
                    best_iou = 0.0
                    for g_i, iou_val in enumerate(ious):
                        if iou_val > best_iou and g_i not in matched_g:
                            best_iou = iou_val
                            best_g_idx = g_i
                    if best_iou >= self.iou_threshold and best_g_idx >= 0:
                        size_stats[s]["tp"] += 1
                        matched_g.add(best_g_idx)
                    else:
                        size_stats[s]["fp"] += 1
                size_stats[s]["fn"] += len(s_gts) - len(matched_g)

        for s in size_scales:
            tp = size_stats[s]["tp"]
            fp = size_stats[s]["fp"]
            fn = size_stats[s]["fn"]
            gt = size_stats[s]["gt"]
            p_s = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            r_s = float(tp / gt) if gt > 0 else 0.0
            f1_s = float(2 * p_s * r_s / (p_s + r_s)) if (p_s + r_s) > 0 else 0.0
            size_stats[s]["precision"] = round(p_s, 4)
            size_stats[s]["recall"] = round(r_s, 4)
            size_stats[s]["f1"] = round(f1_s, 4)

        # 5. Confusion Matrix (10x10 + Background)
        confusion_matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=int)
        for img in eval_images:
            gts = self.ground_truths.get(img, [])
            preds = predictions.get(img, [])
            if not gts and not preds:
                continue

            g_boxes = np.array([g["bbox"] for g in gts], dtype=np.float32) if gts else np.empty((0, 4), dtype=np.float32)
            p_boxes = np.array([p["bbox"] for p in preds], dtype=np.float32) if preds else np.empty((0, 4), dtype=np.float32)

            matched_g_set = set()
            matched_p_set = set()

            if len(g_boxes) > 0 and len(p_boxes) > 0:
                iou_mat = compute_box_iou_matrix(p_boxes, g_boxes)
                p_sorted = np.argsort([-float(p["confidence"]) for p in preds])
                for p_i in p_sorted:
                    ious = iou_mat[p_i]
                    best_g = -1
                    best_iou = 0.0
                    for g_i, iou_val in enumerate(ious):
                        if iou_val > best_iou and g_i not in matched_g_set:
                            best_iou = iou_val
                            best_g = g_i
                    if best_iou >= self.iou_threshold and best_g >= 0:
                        p_cls = preds[p_i].get("class_id", CLASS_NAMES.index(preds[p_i].get("label", "car")))
                        g_cls = gts[best_g]["class_id"]
                        confusion_matrix[g_cls, p_cls] += 1
                        matched_g_set.add(best_g)
                        matched_p_set.add(p_i)

            for p_i, p in enumerate(preds):
                if p_i not in matched_p_set:
                    p_cls = p.get("class_id", CLASS_NAMES.index(p.get("label", "car")))
                    confusion_matrix[BG_CLASS_ID, p_cls] += 1

            for g_i, g in enumerate(gts):
                if g_i not in matched_g_set:
                    confusion_matrix[g["class_id"], BG_CLASS_ID] += 1

        # Sort classes by AP@50 descending
        ranked_classes = sorted(class_metrics, key=lambda x: (x["ap50"], x["ap50_95"], x["f1"]), reverse=True)
        for r_idx, item in enumerate(ranked_classes, start=1):
            item["rank"] = r_idx

        return {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "system_name": system_name,
            "images_evaluated": len(eval_images),
            "overall": {
                "map50": overall_map50,
                "map50_95": overall_map50_95,
                "precision": overall_precision,
                "recall": overall_recall,
                "f1": overall_f1,
                "total_gt": tot_gt_all,
                "total_tp": total_tp,
                "total_fp": total_fp,
                "total_fn": tot_fn_all,
            },
            "class_metrics": ranked_classes,
            "size_stats": size_stats,
            "confusion_matrix": confusion_matrix.tolist(),
            "latencies": latencies or {},
        }
