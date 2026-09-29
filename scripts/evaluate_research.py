#!/usr/bin/env python3
import os
import sys
import json
import csv
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any

from tqdm import tqdm
from ultralytics import YOLO

_current = Path(__file__).resolve().parent
PROJECT_ROOT = _current.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.comprehensive_evaluator import ComprehensiveEvaluator, CLASS_NAMES
from utils.adaptive_sahi import AdaptiveSAHI
from models.clip_engine import CLIPEngine
from models.schemas import Detection
from models.detection_config import DetectionConfig
from models.enhanced_postprocessor import EnhancedPostProcessor
from models.super_resolution import SuperResolutionEngine

def main():
    parser = argparse.ArgumentParser(description="Evaluate Research Experiment")
    parser.add_argument("--exp-dir", type=str, required=True)
    parser.add_argument("--sahi", action="store_true")
    parser.add_argument("--sr", action="store_true")
    parser.add_argument("--clip", action="store_true")
    parser.add_argument("--adaptive", action="store_true")
    parser.add_argument("--open-vocab", action="store_true")
    parser.add_argument("--weights", type=str, help="Path to custom weights file (defaults to E3)")
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    val_dir = PROJECT_ROOT / "datasets" / "VisDrone2019" / "VisDrone2019-DET-val"
    if not val_dir.exists():
        val_dir = PROJECT_ROOT / "#FILLERS" / "datasets" / "VisDrone2019" / "VisDrone2019-DET-val"

    images_dir = val_dir / "images"
    all_images = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
    
    if args.open_vocab:
        weights_path = str(PROJECT_ROOT / "weights/yolov8s-world.pt")
        print(f"Loading OV detector from {weights_path}...")
        model = YOLO(weights_path)
        ov_classes = ["building", "tree", "river", "boat", "dog"]
        model.set_classes(ov_classes)
        CLASS_NAMES_LOCAL = ov_classes
    else:
        if args.weights:
            weights_path = args.weights
        else:
            weights_path = str(PROJECT_ROOT / "runs/detect/experiments/model_search/E3_yolo11l_1536_aug/weights/best.pt")
            if not Path(weights_path).exists():
                weights_path = str(PROJECT_ROOT / "weights/best.pt")
        print(f"Loading detector from {weights_path}...")
        model = YOLO(weights_path)
        CLASS_NAMES_LOCAL = CLASS_NAMES
    
    clip_engine = None
    if args.clip:
        print("Loading CLIP engine...")
        clip_engine = CLIPEngine()

    adaptive_sahi = None
    if args.sahi:
        adaptive_sahi = AdaptiveSAHI(
            min_density_for_sahi=1,
            min_small_ratio_for_sahi=0.0,
            default_slice_width=640,
            default_slice_height=640,
            default_overlap=0.20
        )
        
    predictions = {}
    latencies = []
    
    print(f"Evaluating {len(all_images)} images. SAHI={args.sahi}, SR={args.sr}, CLIP={args.clip}, Adaptive={args.adaptive}, OpenVocab={args.open_vocab}")
    
    for img_path in tqdm(all_images):
        img_name = img_path.name
        t0 = time.perf_counter()
        
        current_sahi = args.sahi
        current_sr = args.sr
        
        if args.adaptive:
            from workflows.state import AgentState, SearchQuery
            from agents.strategy_agent import StrategyAgent
            state = AgentState(query=SearchQuery(raw_query="car"))
            strategy_agent = StrategyAgent()
            state = strategy_agent.run(state)
            state = strategy_agent.refine_with_image(state, str(img_path))
            current_sahi = state.strategy.enable_sahi
            current_sr = state.strategy.enable_super_resolution
        
        # Base Prediction
        res = model.predict(str(img_path), conf=0.001, iou=0.6, max_det=902, imgsz=1536 if not args.open_vocab else 640, verbose=False)
        base_preds = []
        if res and len(res[0].boxes) > 0:
            for box in res[0].boxes:
                cid = int(box.cls)
                if 0 <= cid < len(CLASS_NAMES_LOCAL):
                    base_preds.append(Detection(
                        class_id=cid,
                        label=CLASS_NAMES_LOCAL[cid],
                        confidence=float(box.conf),
                        bbox=box.xyxy[0].tolist(),
                        source="full_image"
                    ))
                    
        # SAHI Integration
        if current_sahi and adaptive_sahi:
            from PIL import Image
            pil_image = Image.open(img_path).convert("RGB")
            
            def sahi_detect_fn(patches):
                res_patches = model.predict(source=patches, imgsz=640, conf=0.001, verbose=False)
                batch_dets = []
                for r in res_patches:
                    dets = []
                    if r.boxes is not None and len(r.boxes) > 0:
                        for b, s, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy()):
                            if int(c) < len(CLASS_NAMES_LOCAL):
                                dets.append(Detection(
                                    label=CLASS_NAMES_LOCAL[int(c)], 
                                    confidence=float(s), 
                                    bbox=[float(b[0]), float(b[1]), float(b[2]), float(b[3])],
                                    class_id=int(c),
                                    source="sahi"
                                ))
                    batch_dets.append(dets)
                return batch_dets
                
            sahi_dets = adaptive_sahi.run_sliced_inference(pil_image, detect_fn=sahi_detect_fn)
            for d in sahi_dets:
                if d.label in CLASS_NAMES_LOCAL:
                    cid = CLASS_NAMES_LOCAL.index(d.label)
                    base_preds.append(Detection(
                        class_id=cid,
                        label=CLASS_NAMES_LOCAL[cid],
                        confidence=d.confidence,
                        bbox=d.bbox,
                        source="sahi"
                    ))
        
        # Apply Selective SR
        if current_sr:
            from PIL import Image
            pil_image = Image.open(img_path).convert("RGB")
            sr_engine = SuperResolutionEngine()
            
            # Find small object regions (area < 32x32)
            small_boxes = [d for d in base_preds if (d.bbox[2]-d.bbox[0]) * (d.bbox[3]-d.bbox[1]) < 32 * 32]
            
            # Select up to 4 top-confidence small boxes to crop around
            small_boxes = sorted(small_boxes, key=lambda x: x.confidence, reverse=True)[:4]
            
            for d in small_boxes:
                cx = (d.bbox[0] + d.bbox[2]) / 2
                cy = (d.bbox[1] + d.bbox[3]) / 2
                crop_size = 320
                x1 = max(0, int(cx - crop_size / 2))
                y1 = max(0, int(cy - crop_size / 2))
                x2 = min(pil_image.width, int(cx + crop_size / 2))
                y2 = min(pil_image.height, int(cy + crop_size / 2))
                
                crop = pil_image.crop((x1, y1, x2, y2))
                upscaled = sr_engine.upscale_pil(crop, scale=2)
                
                sr_res = model.predict(source=upscaled, conf=0.001, iou=0.6, max_det=300, verbose=False)
                if sr_res and len(sr_res[0].boxes) > 0:
                    for sr_box in sr_res[0].boxes:
                        cid = int(sr_box.cls)
                        if 0 <= cid < len(CLASS_NAMES_LOCAL):
                            bx = sr_box.xyxy[0].tolist()
                            gx1 = (bx[0] / 2) + x1
                            gy1 = (bx[1] / 2) + y1
                            gx2 = (bx[2] / 2) + x1
                            gy2 = (bx[3] / 2) + y1
                            base_preds.append(Detection(
                                class_id=cid,
                                label=CLASS_NAMES_LOCAL[cid],
                                confidence=float(sr_box.conf),
                                bbox=[gx1, gy1, gx2, gy2],
                                source="sr"
                            ))
                            
        # Apply NMS if SAHI or SR added duplicate predictions
        if current_sahi or current_sr:
            # Use improved NMS with Soft-NMS and Deduplication
            config = DetectionConfig.uav_benchmark()
            config.nms_iou_threshold = 0.45
            config.use_soft_nms = True
            config.use_deduplication = True
            config.dedup_iou_threshold = 0.65
            
            from PIL import Image
            img_w, img_h = Image.open(img_path).size
            
            pre_nms_count = len(base_preds)
            base_preds = EnhancedPostProcessor.apply_nms(
                base_preds, config=config, image_width=img_w, image_height=img_h
            )
            post_nms_count = len(base_preds)
            if img_name == "0000006_00159_d_0000001.jpg" or True: # Just print for the first image
                print(f"[{img_name}] Pre-NMS: {pre_nms_count}, Post-NMS: {post_nms_count}, Suppressed: {pre_nms_count - post_nms_count}")
            
        # CLIP Integration
        if args.clip and clip_engine and len(base_preds) > 0:
            from PIL import Image
            pil_image = Image.open(img_path).convert("RGB")
            verified_preds = []
            for p in base_preds:
                try:
                    box = p.bbox
                    crop = pil_image.crop((box[0], box[1], box[2], box[3]))
                    similarity_dict = clip_engine.score_image_against_texts(crop, [p.label])
                    similarity = similarity_dict.get(p.label, 0.0)
                    if similarity > 0.15:
                        p.confidence = (p.confidence + similarity) / 2
                        verified_preds.append(p)
                except Exception:
                    verified_preds.append(p)
            base_preds = verified_preds

        latencies.append(time.perf_counter() - t0)
        predictions[img_name] = [
            {
                "class_id": p.class_id if hasattr(p, 'class_id') else p.get("class_id"),
                "label": p.label if hasattr(p, 'label') else p.get("label"),
                "confidence": p.confidence if hasattr(p, 'confidence') else p.get("confidence"),
                "bbox": p.bbox if hasattr(p, 'bbox') else p.get("bbox")
            } for p in base_preds
        ]
        
    chk_path = exp_dir / "predictions.json"
    with open(chk_path, "w") as f:
        json.dump(predictions, f)
        
    evaluator = ComprehensiveEvaluator(val_dir=val_dir, iou_threshold=0.50)
    metrics = evaluator.evaluate_predictions(predictions, system_name=exp_dir.name)
    
    avg_latency = sum(latencies)/len(latencies) if latencies else 0.0
    metrics['overall']['avg_latency'] = avg_latency
    
    met_path = exp_dir / "metrics.json"
    with open(met_path, "w") as f:
        json.dump(metrics, f, indent=2)
        
    print(f"\nEvaluation complete for {exp_dir.name}")
    print(f"mAP50: {metrics['overall']['map50']:.4f}")
    
if __name__ == "__main__":
    main()
