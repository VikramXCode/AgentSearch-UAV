import torch
from torchvision.ops import nms


class DetectionPostProcessor:

    @staticmethod
    def apply_nms(
        detections,
        iou_threshold=0.50,
    ):

        if len(detections) == 0:
            return detections

        boxes = torch.tensor(
            [d.bbox for d in detections],
            dtype=torch.float32,
        )

        scores = torch.tensor(
            [d.confidence for d in detections],
            dtype=torch.float32,
        )

        keep = nms(
            boxes,
            scores,
            iou_threshold,
        )

        filtered = []

        for idx in keep.tolist():
            filtered.append(detections[idx])

        return filtered