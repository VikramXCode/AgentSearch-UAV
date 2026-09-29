import sys, json
from pathlib import Path
from ultralytics import YOLO
sys.path.insert(0, str(Path.cwd()))
from evaluation.comprehensive_evaluator import ComprehensiveEvaluator, CLASS_NAMES

model = YOLO('runs/detect/experiments/model_search/E3_yolo11l_1536_aug/weights/best.pt')
images_dir = Path("datasets/VisDrone2019/VisDrone2019-DET-val/images")
images = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))[:10]

preds = {}
for img in images:
    res = model.predict(str(img), conf=0.001, iou=0.6, imgsz=1536, max_det=317, half=True, verbose=False)
    boxes = []
    if res and len(res[0].boxes) > 0:
        for box in res[0].boxes:
            cid = int(box.cls)
            if 0 <= cid < len(CLASS_NAMES):
                boxes.append({
                    "class_id": cid,
                    "label": CLASS_NAMES[cid],
                    "confidence": float(box.conf),
                    "bbox": box.xyxy[0].tolist(),
                    "source": "control"
                })
    preds[img.name] = boxes

evaluator = ComprehensiveEvaluator(iou_threshold=0.50)
evaluator.ground_truths = {k: v for k, v in evaluator.ground_truths.items() if k in preds}

# Manually compute mAP50 using the fixed logic
import numpy as np

def compute_coco_101_point_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    recall_levels = np.linspace(0.0, 1.0, 101)
    inds = np.searchsorted(mrec, recall_levels, side="left")
    inds = np.clip(inds, 0, len(mpre) - 1)
    return float(np.mean(mpre[inds]))

def compute_box_iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)
    x1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    y1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    x2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    y2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])
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

num_classes = len(CLASS_NAMES)
gt_by_img_cls = {}
total_gt = np.zeros(num_classes, dtype=int)
for img in evaluator.ground_truths:
    for g in evaluator.ground_truths[img]:
        c = g["class_id"]
        if 0 <= c < num_classes:
            total_gt[c] += 1
            gt_by_img_cls.setdefault((img, c), []).append(np.array(g["bbox"], dtype=np.float32))

preds_by_cls = {c: [] for c in range(num_classes)}
for img in preds:
    for p in preds[img]:
        c = p["class_id"]
        if 0 <= c < num_classes:
            preds_by_cls[c].append({
                "img": img,
                "score": float(p["confidence"]),
                "bbox": np.array(p["bbox"], dtype=np.float32),
            })
for c in range(num_classes):
    preds_by_cls[c].sort(key=lambda x: x["score"], reverse=True)

ap_matrix = np.zeros((num_classes, 1), dtype=np.float64)
IOU_THRESHOLDS = [0.50]
for c in range(num_classes):
    c_preds = preds_by_cls[c]
    n_gt = int(total_gt[c])
    if n_gt == 0 or len(c_preds) == 0:
        continue
    c_boxes = np.array([p["bbox"] for p in c_preds], dtype=np.float32)
    c_imgs = [p["img"] for p in c_preds]
    img_to_indices = {}
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
                    if iou_val > best_iou and g_i not in matched_gt:
                        best_iou = iou_val
                        best_g = g_i
                if best_iou >= iou_thresh and best_g != -1:
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

overall_map50 = float(np.mean(ap_matrix[:, 0]))
with open("scratch/fixed_ap.json", "w") as f: json.dump({"map50": overall_map50}, f)
