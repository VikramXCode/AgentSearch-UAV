import cv2
import time
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Callable, Tuple
from models.schemas import Detection

def compute_iou(box1: List[float], box2: List[float]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, (box1[2] - box1[0]) * (box1[3] - box1[1]))
    area2 = max(0.0, (box2[2] - box2[0]) * (box2[3] - box2[1]))
    union = area1 + area2 - inter_area

    if union <= 0.0:
        return 0.0
    return inter_area / union

class AdaptiveRegionProcessor:
    def __init__(self, config=None):
        self.config = config or {}
        self.sr_engine = None
        self.sahi_engine = None
        
        self.roi_padding = 0.5 # Add 50% context around objects
        self.min_roi_size = 256
        self.max_roi_size = 800
        
    def _get_sr_engine(self):
        if self.sr_engine is None:
            from models.super_resolution import SuperResolutionEngine
            self.sr_engine = SuperResolutionEngine()
        return self.sr_engine
        
    def analyze_difficulty(self, det: Detection, image_cv: np.ndarray) -> Dict[str, Any]:
        x1, y1, x2, y2 = map(int, det.bbox)
        h_img, w_img = image_cv.shape[:2]
        
        bw, bh = max(1, x2 - x1), max(1, y2 - y1)
        area = bw * bh
        rel_area = area / (w_img * h_img)
        
        # Crop with some context to measure blur
        cx1, cy1 = max(0, x1 - 10), max(0, y1 - 10)
        cx2, cy2 = min(w_img, x2 + 10), min(h_img, y2 + 10)
        crop = image_cv[cy1:cy2, cx1:cx2]
        
        if crop.size == 0:
            sharpness = 0
            contrast = 0
        else:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            contrast = gray.std()
            
        is_small = area < 1024  # 32x32
        is_low_conf = det.confidence < 0.35
        is_blurry = sharpness < 100.0 or contrast < 20.0
        
        score = 0
        if is_small: score += 2
        if is_low_conf: score += 1
        if is_blurry: score += 2
        
        branch = "normal"
        if is_small and is_blurry:
            branch = "sahi_sr"
        elif is_small:
            branch = "sahi"
        elif is_blurry or is_low_conf:
            branch = "sr"
            
        return {
            "score": score,
            "branch": branch,
            "area": area,
            "sharpness": sharpness,
            "contrast": contrast,
            "is_small": is_small,
            "is_blurry": is_blurry,
            "is_low_conf": is_low_conf
        }
        
    def process_regions(self, image: Image.Image, initial_dets: List[Detection], detect_fn) -> Tuple[List[Detection], List[Detection], List[Dict[str, Any]]]:
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        h_img, w_img = image_cv.shape[:2]
        
        filtered_initial_dets = list(initial_dets)
        difficult_rois = [] # (roi_bbox, branch, orig_dets, reason)
        
        for det in initial_dets:
            analysis = self.analyze_difficulty(det, image_cv)
            branch = analysis["branch"]
            if branch == "normal":
                continue
                
            # Define ROI around this difficult detection
            x1, y1, x2, y2 = det.bbox
            bw, bh = x2 - x1, y2 - y1
            pad_w = max(self.min_roi_size - bw, bw * self.roi_padding)
            pad_h = max(self.min_roi_size - bh, bh * self.roi_padding)
            
            rx1 = max(0, int(x1 - pad_w/2))
            ry1 = max(0, int(y1 - pad_h/2))
            rx2 = min(w_img, int(x2 + pad_w/2))
            ry2 = min(h_img, int(y2 + pad_h/2))
            
            # Enforce max ROI size to avoid doing SR on the whole image
            if rx2 - rx1 > self.max_roi_size:
                rx1 = max(0, int(x1 + bw/2 - self.max_roi_size/2))
                rx2 = min(w_img, rx1 + self.max_roi_size)
            if ry2 - ry1 > self.max_roi_size:
                ry1 = max(0, int(y1 + bh/2 - self.max_roi_size/2))
                ry2 = min(h_img, ry1 + self.max_roi_size)
                
            difficult_rois.append({
                "roi": [rx1, ry1, rx2, ry2],
                "branch": branch,
                "reason": analysis,
                "orig_det": det
            })
            
        # Group overlapping ROIs that share the same branch
        merged_rois = []
        used = [False] * len(difficult_rois)
        for i in range(len(difficult_rois)):
            if used[i]: continue
            cur = difficult_rois[i]
            used[i] = True
            
            cur_roi = list(cur["roi"])
            orig_dets = [cur["orig_det"]]
            
            for j in range(i + 1, len(difficult_rois)):
                if used[j] or difficult_rois[j]["branch"] != cur["branch"]: continue
                if compute_iou(cur_roi, difficult_rois[j]["roi"]) > 0.1:
                    cur_roi[0] = min(cur_roi[0], difficult_rois[j]["roi"][0])
                    cur_roi[1] = min(cur_roi[1], difficult_rois[j]["roi"][1])
                    cur_roi[2] = max(cur_roi[2], difficult_rois[j]["roi"][2])
                    cur_roi[3] = max(cur_roi[3], difficult_rois[j]["roi"][3])
                    orig_dets.append(difficult_rois[j]["orig_det"])
                    used[j] = True
                    
            merged_rois.append({
                "roi": cur_roi,
                "branch": cur["branch"],
                "reason": cur["reason"], # just keep first reason
                "orig_dets": orig_dets
            })
            
        # Execute processing
        final_processed_dets = []
        logs = []
        
        for m_roi in merged_rois:
            rx1, ry1, rx2, ry2 = m_roi["roi"]
            branch = m_roi["branch"]
            crop_pil = image.crop((rx1, ry1, rx2, ry2))
            
            t_start = time.perf_counter()
            if branch == "sr" or branch == "sahi_sr":
                # Apply SR
                sr_img = self._get_sr_engine().upscale_pil(crop_pil, scale=2)
                
                if branch == "sahi_sr":
                    # SAHI on SR
                    from utils.adaptive_sahi import AdaptiveSAHI
                    temp_sahi = AdaptiveSAHI(default_slice_width=320, default_slice_height=320, default_overlap=0.2)
                    roi_dets = temp_sahi.run_sliced_inference(sr_img, lambda patches: detect_fn(patches, conf=0.10, imgsz=640))
                    # Scale boxes back down since we upscaled by 2
                    for d in roi_dets:
                        d.bbox = [b / 2.0 for b in d.bbox]
                else:
                    # Normal YOLO on SR
                    roi_dets = detect_fn([sr_img], conf=0.10, imgsz=640)[0]
                    for d in roi_dets:
                        d.bbox = [b / 2.0 for b in d.bbox]
            elif branch == "sahi":
                from utils.adaptive_sahi import AdaptiveSAHI
                temp_sahi = AdaptiveSAHI(default_slice_width=320, default_slice_height=320, default_overlap=0.2)
                roi_dets = temp_sahi.run_sliced_inference(crop_pil, lambda patches: detect_fn(patches, conf=0.10, imgsz=640))
            else:
                roi_dets = []
                
            t_proc = time.perf_counter() - t_start
            
            # Map back to global coordinates and verify
            verified_roi_dets = []
            
            # Compare Initial vs Processed
            recovered = 0
            increased_conf = 0
            improved_loc = 0
            false_positive = 0
            duplicate = 0
            for new_det in roi_dets:
                # Map to global coordinates first!
                new_det.bbox = [
                    new_det.bbox[0] + rx1, 
                    new_det.bbox[1] + ry1, 
                    new_det.bbox[2] + rx1, 
                    new_det.bbox[3] + ry1
                ]
                
                # 1. Reject out of bounds or weird shapes
                bw = new_det.bbox[2] - new_det.bbox[0]
                bh = new_det.bbox[3] - new_det.bbox[1]
                if bw < 3 or bh < 3 or bw/bh > 10 or bh/bw > 10:
                    continue # Rejection Reason: Geometric Anomaly

                # 2. Compare against initial detections for duplication and false pos rejection
                best_iou = 0
                best_orig = None
                for orig_det in m_roi["orig_dets"]:
                    if new_det.class_id != orig_det.class_id: continue
                    iou = compute_iou(new_det.bbox, orig_det.bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_orig = orig_det
                
                if best_iou == 0:
                    # New target found. Reject if confidence is extremely low (likely SR hallucination)
                    if new_det.confidence < 0.30:
                        false_positive += 1
                        continue # Rejection Reason: Low Confidence Hallucination
                    else:
                        recovered += 1
                elif best_iou > 0.85:
                    # Near identical to original. We only keep it if confidence is significantly higher
                    if new_det.confidence <= best_orig.confidence + 0.05:
                        duplicate += 1
                        continue # Rejection Reason: Duplicate without significant improvement
                    else:
                        increased_conf += 1
                else:
                    if new_det.confidence > best_orig.confidence:
                        increased_conf += 1
                    improved_loc += 1 

                new_det.source = f"adaptive_{branch}"
                verified_roi_dets.append(new_det)
            
            final_processed_dets.extend(verified_roi_dets)
            
            # 3. Filter the initial detections: if a difficult object was NOT found by SAHI/SR, it's likely a false positive.
            for orig_det in m_roi["orig_dets"]:
                best_iou = 0
                for new_det in roi_dets:
                    if new_det.class_id == orig_det.class_id:
                        iou = compute_iou(new_det.bbox, orig_det.bbox)
                        if iou > best_iou:
                            best_iou = iou
                
                if best_iou == 0 and orig_det.confidence < 0.50:
                    if orig_det in filtered_initial_dets:
                        filtered_initial_dets.remove(orig_det)
            
            logs.append({
                "roi": [rx1, ry1, rx2, ry2],
                "branch": branch,
                "reason": m_roi["reason"],
                "processing_time": t_proc,
                "initial_dets": len(m_roi["orig_dets"]),
                "new_dets": len(verified_roi_dets),
                "comparison": {
                    "recovered_missed": recovered,
                    "increased_confidence": increased_conf,
                    "improved_localization": improved_loc,
                    "potential_false_positive": false_positive,
                    "duplicates": duplicate
                }
            })
            
        return final_processed_dets, filtered_initial_dets, logs
