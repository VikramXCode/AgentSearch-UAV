"""
Enhanced post-processing with soft-NMS, scale-aware NMS, and deduplication.
Replaces basic NMS with sophisticated multi-method approach for better accuracy.
"""

import torch
import numpy as np
from torchvision.ops import nms
from typing import List, Optional, Dict, Tuple

from models.schemas import Detection
from models.detection_config import DetectionConfig, DEFAULT_CONFIG
from utils.search_utils import normalize_label


class EnhancedPostProcessor:
    """Advanced post-processing with soft-NMS, scale-aware NMS, and deduplication."""
    
    @staticmethod
    def apply_nms(
        detections: List[Detection],
        config: Optional[DetectionConfig] = None,
        image_width: Optional[int] = None,
        image_height: Optional[int] = None,
    ) -> List[Detection]:
        """
        Apply advanced NMS with optional soft-NMS and scale-aware thresholds.
        
        Args:
            detections: List of Detection objects
            config: DetectionConfig for optimization parameters
            image_width: Image width for size-based filtering (optional)
            image_height: Image height for size-based filtering (optional)
        
        Returns:
            Filtered detections after NMS
        """
        
        if config is None:
            config = DEFAULT_CONFIG
        
        if len(detections) == 0:
            return detections
        
        # Group by label
        grouped = {}
        for detection in detections:
            label = normalize_label(detection.label)
            grouped.setdefault(label, []).append(detection)
        
        filtered = []
        
        # Process each class independently
        for label, class_detections in grouped.items():
            
            # Apply soft-NMS if enabled
            if config.use_soft_nms:
                class_detections = EnhancedPostProcessor._apply_soft_nms(
                    class_detections,
                    config,
                    image_width,
                    image_height,
                )
            else:
                # Use standard NMS with scale-aware thresholds
                class_detections = EnhancedPostProcessor._apply_standard_nms(
                    class_detections,
                    config,
                    image_width,
                    image_height,
                )
            
            filtered.extend(class_detections)
        
        # Apply deduplication if enabled
        if config.use_deduplication and len(filtered) > 1:
            filtered = EnhancedPostProcessor._apply_deduplication(
                filtered,
                config,
            )
        
        # Apply score filtering
        filtered = [d for d in filtered if d.confidence >= config.nms_score_threshold]
        
        # Sort by confidence
        return sorted(filtered, key=lambda d: d.confidence, reverse=True)
    
    @staticmethod
    def _apply_soft_nms(
        detections: List[Detection],
        config: DetectionConfig,
        image_width: Optional[int] = None,
        image_height: Optional[int] = None,
    ) -> List[Detection]:
        """Apply Soft-NMS (linear or gaussian decay) with accurate index tracking."""
        
        if len(detections) <= 1:
            return detections
        
        boxes = torch.tensor(
            [d.bbox for d in detections],
            dtype=torch.float32,
        )
        
        scores = torch.tensor(
            [d.confidence for d in detections],
            dtype=torch.float32,
        )
        
        # Apply standard NMS if configured
        if config.soft_nms_method == "standard":
            keep = nms(boxes, scores, config.soft_nms_iou_threshold)
            return [detections[i] for i in keep.tolist()]
        
        indices = torch.arange(len(detections))
        score_thresh = config.nms_score_threshold
        result_detections = []
        
        while len(scores) > 0:
            max_idx = scores.argmax().item()
            best_orig_idx = indices[max_idx].item()
            best_score = float(scores[max_idx].item())
            best_box = boxes[max_idx].unsqueeze(0)
            
            if best_score >= score_thresh:
                orig_det = detections[best_orig_idx]
                result_detections.append(
                    Detection(
                        label=orig_det.label,
                        confidence=best_score,
                        bbox=orig_det.bbox,
                    )
                )
            
            if len(scores) == 1:
                break
            
            other_mask = torch.ones(len(scores), dtype=torch.bool)
            other_mask[max_idx] = False
            
            other_boxes = boxes[other_mask]
            other_scores = scores[other_mask]
            other_indices = indices[other_mask]
            
            ious = EnhancedPostProcessor._calculate_iou(best_box, other_boxes).squeeze(0)
            
            if config.soft_nms_method == "linear":
                decay = torch.ones_like(ious)
                high_iou = ious > config.soft_nms_iou_threshold
                decay[high_iou] = torch.clamp(1.0 - ious[high_iou] * config.soft_nms_score_decay, min=0.0)
                other_scores = other_scores * decay
            else:  # gaussian
                decay = torch.exp(-(ious ** 2) / (config.soft_nms_sigma ** 2))
                other_scores = other_scores * decay
            
            # Keep active candidates above minimum score threshold
            keep_mask = other_scores >= (score_thresh * 0.5)
            boxes = other_boxes[keep_mask]
            scores = other_scores[keep_mask]
            indices = other_indices[keep_mask]
        
        return result_detections
    
    @staticmethod
    def _apply_standard_nms(
        detections: List[Detection],
        config: DetectionConfig,
        image_width: Optional[int] = None,
        image_height: Optional[int] = None,
    ) -> List[Detection]:
        """Apply standard NMS with optional scale-aware thresholds."""
        
        if len(detections) <= 1:
            return detections
        
        boxes = torch.tensor(
            [d.bbox for d in detections],
            dtype=torch.float32,
        )
        
        scores = torch.tensor(
            [d.confidence for d in detections],
            dtype=torch.float32,
        )
        
        # Determine NMS threshold
        if config.use_scale_aware_nms and image_width and image_height:
            # Use average object size to determine threshold
            areas = []
            for bbox in boxes:
                x1, y1, x2, y2 = bbox
                area = (x2 - x1) * (y2 - y1)
                areas.append(area)
            
            avg_area = torch.mean(torch.tensor(areas))
            image_area = image_width * image_height
            area_ratio = (avg_area / image_area).item()
            
            iou_threshold = config.get_nms_threshold(area_ratio)
        else:
            iou_threshold = config.nms_iou_threshold
        
        keep = nms(boxes, scores, iou_threshold)
        return [detections[i] for i in keep.tolist()]
    
    @staticmethod
    def _apply_deduplication(
        detections: List[Detection],
        config: DetectionConfig,
    ) -> List[Detection]:
        """Remove duplicate detections from SAHI slice overlaps."""
        
        if len(detections) <= 1:
            return detections
        
        boxes = torch.tensor(
            [d.bbox for d in detections],
            dtype=torch.float32,
        )
        
        # Find duplicate groups using high IoU threshold
        keep_indices = []
        removed = set()
        
        for i in range(len(detections)):
            if i in removed:
                continue
            
            keep_indices.append(i)
            
            # Find duplicates of this detection (only within the same class)
            for j in range(i + 1, len(detections)):
                if j in removed:
                    continue
                
                if detections[i].label == detections[j].label:
                    iou = EnhancedPostProcessor._calculate_single_iou(
                        boxes[i],
                        boxes[j]
                    ).item()
                    
                    if iou > config.dedup_iou_threshold:
                        # Mark as duplicate
                        removed.add(j)
                        
                        # Optionally merge confidences
                        if config.dedup_confidence_merge:
                            detections[i].confidence = max(
                                detections[i].confidence,
                                (detections[i].confidence + detections[j].confidence) / 2
                            )
        
        return [detections[i] for i in keep_indices]
    
    @staticmethod
    def _calculate_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """Calculate IoU between all pairs of boxes."""
        
        # boxes1: [N, 4], boxes2: [M, 4]
        # Returns: [N, M]
        
        x1_min, y1_min, x1_max, y1_max = boxes1[:, 0], boxes1[:, 1], boxes1[:, 2], boxes1[:, 3]
        x2_min, y2_min, x2_max, y2_max = boxes2[:, 0], boxes2[:, 1], boxes2[:, 2], boxes2[:, 3]
        
        # Expand dimensions for broadcasting
        x1_min = x1_min.unsqueeze(1)  # [N, 1]
        y1_min = y1_min.unsqueeze(1)
        x1_max = x1_max.unsqueeze(1)
        y1_max = y1_max.unsqueeze(1)
        
        x2_min = x2_min.unsqueeze(0)  # [1, M]
        y2_min = y2_min.unsqueeze(0)
        x2_max = x2_max.unsqueeze(0)
        y2_max = y2_max.unsqueeze(0)
        
        # Calculate intersection
        inter_xmin = torch.max(x1_min, x2_min)
        inter_ymin = torch.max(y1_min, y2_min)
        inter_xmax = torch.min(x1_max, x2_max)
        inter_ymax = torch.min(y1_max, y2_max)
        
        inter_w = (inter_xmax - inter_xmin).clamp(min=0)
        inter_h = (inter_ymax - inter_ymin).clamp(min=0)
        inter_area = inter_w * inter_h
        
        # Calculate union
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - inter_area
        
        # Calculate IoU
        iou = inter_area / union_area.clamp(min=1e-6)
        return iou
    
    @staticmethod
    def _calculate_single_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
        """Calculate IoU between two boxes."""
        
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # Intersection
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)
        
        inter_w = max(0, inter_xmax - inter_xmin)
        inter_h = max(0, inter_ymax - inter_ymin)
        inter_area = inter_w * inter_h
        
        # Union
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - inter_area
        
        # IoU
        iou = inter_area / max(union_area, 1e-6)
        return torch.as_tensor(iou, dtype=torch.float32)
    
    @staticmethod
    def apply_size_filtering(
        detections: List[Detection],
        config: DetectionConfig,
        image_width: int,
        image_height: int,
    ) -> List[Detection]:
        """Filter detections by size constraints."""
        
        if not config.use_size_filtering or len(detections) == 0:
            return detections
        
        image_area = image_width * image_height
        min_area = image_area * config.min_area_ratio
        max_area = image_area * config.max_area_ratio
        
        filtered = []
        
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            bbox_area = (x2 - x1) * (y2 - y1)
            
            if min_area <= bbox_area <= max_area:
                filtered.append(detection)
        
        return filtered
    
    @staticmethod
    def apply_aspect_ratio_filtering(
        detections: List[Detection],
        config: DetectionConfig,
    ) -> List[Detection]:
        """Filter detections by aspect ratio."""
        
        if not config.use_aspect_ratio_filtering or len(detections) == 0:
            return detections
        
        filtered = []
        
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            width = x2 - x1
            height = y2 - y1
            
            if height == 0 or width == 0:
                continue
            
            aspect_ratio = width / height if height > 0 else float('inf')
            
            if config.min_aspect_ratio <= aspect_ratio <= config.max_aspect_ratio:
                filtered.append(detection)
        
        return filtered
    
    @staticmethod
    def apply_edge_filtering(
        detections: List[Detection],
        config: DetectionConfig,
        image_width: int,
        image_height: int,
    ) -> List[Detection]:
        """Remove detections at image edges (likely partial/false positives)."""
        
        if not config.use_edge_filtering or len(detections) == 0:
            return detections
        
        margin_x = image_width * config.edge_margin_ratio
        margin_y = image_height * config.edge_margin_ratio
        
        filtered = []
        
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            
            # Check if detection is within margins
            if (x1 > margin_x and x2 < image_width - margin_x and
                y1 > margin_y and y2 < image_height - margin_y):
                filtered.append(detection)
        
        return filtered


# Keep legacy interface for backward compatibility
def apply_nms_legacy(detections, iou_threshold=0.35):
    """Legacy NMS function for backward compatibility."""
    config = DEFAULT_CONFIG
    config.nms_iou_threshold = iou_threshold
    config.use_soft_nms = False
    config.use_scale_aware_nms = False
    return EnhancedPostProcessor.apply_nms(detections, config)
