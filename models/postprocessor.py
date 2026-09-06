import torch
from torchvision.ops import nms

from utils.search_utils import normalize_label


class DetectionPostProcessor:

    @staticmethod
    def apply_nms(
        detections,
        iou_threshold=0.35,
    ):

        if len(detections) == 0:
            return detections

        grouped = {}

        for detection in detections:
            grouped.setdefault(normalize_label(detection.label), []).append(detection)

        filtered = []

        for _, group in grouped.items():
            boxes = torch.tensor(
                [d.bbox for d in group],
                dtype=torch.float32,
            )

            scores = torch.tensor(
                [d.confidence for d in group],
                dtype=torch.float32,
            )

            keep = nms(
                boxes,
                scores,
                iou_threshold,
            )

            for idx in keep.tolist():
                filtered.append(group[idx])

        return sorted(filtered, key=lambda detection: detection.confidence, reverse=True)