import os
import sys
import json
import csv
import time
import argparse
import tracemalloc
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
from tqdm import tqdm
import torch
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# Setup paths
_current = Path(__file__).resolve().parent
PROJECT_ROOT = _current.parent
sys.path.insert(0, str(PROJECT_ROOT))


from faster_coco_eval import COCO, COCOeval_faster
from PIL import Image

CLASS_NAMES = {
    0: "pedestrian", 1: "people", 2: "bicycle", 3: "car", 4: "van",
    5: "truck", 6: "tricycle", 7: "awning-tricycle", 8: "bus", 9: "motor"
}
CLASS_LIST = [CLASS_NAMES[i] for i in range(10)]

class COCOEvaluatorWrapper:
    def __init__(self, val_dir: Path):
        self.val_dir = val_dir
        self.images_dir = val_dir / "images"
        self.annotations_dir = val_dir / "annotations"
        self.gt_json_path = val_dir / "coco_gt.json"
        self._build_gt()

    def _build_gt(self):
        if self.gt_json_path.exists():
            return
        print("Building COCO Ground Truth JSON...")
        images = []
        annotations = []
        ann_id = 1
        
        for img_id, img_path in enumerate(sorted(list(self.images_dir.glob("*.jpg")) + list(self.images_dir.glob("*.png"))), start=1):
            w, h = Image.open(img_path).size
            images.append({"id": img_id, "file_name": img_path.name, "width": w, "height": h})
            
            ann_file = self.annotations_dir / f"{img_path.stem}.txt"
            if ann_file.exists():
                for line in ann_file.read_text(encoding="utf-8").splitlines():
                    parts = [p.strip() for p in line.split(",") if p.strip()]
                    if len(parts) >= 6:
                        left, top, width, height, score, category = map(float, parts[:6])
                        cat_int = int(category)
                        if 1 <= cat_int <= 10 and score > 0:
                            cls_id = cat_int - 1
                            annotations.append({
                                "id": ann_id,
                                "image_id": img_id,
                                "category_id": cls_id + 1,
                                "bbox": [left, top, width, height],
                                "area": width * height,
                                "iscrowd": 0
                            })
                            ann_id += 1

        gt_json = {
            "images": images,
            "annotations": annotations,
            "categories": [{"id": i+1, "name": name} for i, name in CLASS_NAMES.items()]
        }
        with open(self.gt_json_path, "w") as f:
            json.dump(gt_json, f)

    def evaluate_predictions(self, predictions_dict: dict, system_name: str = "System"):
        with open(self.gt_json_path, "r") as f:
            gt_data = json.load(f)
        img_name_to_id = {img["file_name"]: img["id"] for img in gt_data["images"]}
        
        pred_list = []
        for img_name, preds in predictions_dict.items():
            if img_name not in img_name_to_id:
                continue
            img_id = img_name_to_id[img_name]
            for p in preds:
                # Convert from [x1, y1, x2, y2] to [x, y, w, h]
                x1, y1, x2, y2 = p["bbox"]
                w = x2 - x1
                h = y2 - y1
                pred_list.append({
                    "image_id": img_id,
                    "category_id": p["class_id"] + 1,
                    "bbox": [x1, y1, w, h],
                    "score": p["confidence"]
                })
        
        pred_json_path = self.val_dir / f"temp_preds_{system_name.replace(' ', '_').replace('(', '').replace(')', '')}.json"
        with open(pred_json_path, "w") as f:
            json.dump(pred_list, f)
            
        coco_gt = COCO(str(self.gt_json_path))
        coco_dt = coco_gt.loadRes(str(pred_json_path))
        evaluator = COCOeval_faster(coco_gt, coco_dt, "bbox")
        img_ids = list(set(p["image_id"] for p in pred_list))
        if img_ids:
            evaluator.params.imgIds = img_ids
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        
        metrics = {
            "overall": {
                "map50": evaluator.stats[1],
                "map50_95": evaluator.stats[0],
                "precision": 0.0,
                "recall": evaluator.stats[8]
            },
            "class_metrics": []
        }
        
        # class_metrics
        for cid in range(10):
            metrics["class_metrics"].append({
                "class_name": CLASS_NAMES[cid],
                "ap50": 0.0 # not easily extracted from COCOeval_faster without deep inspection
            })
            
        return metrics


def load_checkpoint(checkpoint_path: Path) -> Dict[str, List[Dict[str, Any]]]:
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[Warning] Failed to load checkpoint {checkpoint_path}: {e}")
            return {}
    return {}

def save_checkpoint(checkpoint_path: Path, data: Dict[str, List[Dict[str, Any]]]):
    temp_path = checkpoint_path.with_suffix(".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    temp_path.replace(checkpoint_path)

def get_gpu_memory() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2) # MB
    return 0.0

def run_control_evaluation(eval_images: List[Path], weights_path: str, output_dir: Path) -> Tuple[Dict[str, List[Dict[str, Any]]], dict]:
    system_key = "control_e3"
    checkpoint_path = output_dir / f"checkpoint_{system_key}.json"
    predictions = load_checkpoint(checkpoint_path)
    
    print(f"\n[1/2] Running Control (E3 Standard Inference) on {len(eval_images)} images...")
    remaining_images = [p for p in eval_images if p.name not in predictions]
    
    stats = {"total_time": 0.0, "total_detections_pre_cap": 0, "max_gpu_mb": 0.0, "capped_count": 0}
    
    if remaining_images:
        model = YOLO(weights_path)
        if hasattr(model, "set_classes"):
            model.set_classes(CLASS_NAMES)
            
        save_counter = 0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            
        start_time = time.time()
        for img_path in tqdm(remaining_images, desc="Control (E3 Standard)"):
            img_name = img_path.name
            img_str = str(img_path)
            
            res = model.predict(img_str, conf=0.001, iou=0.6, imgsz=1536, max_det=300, half=True, device="0", verbose=False)
            
            preds = []
            if res and len(res[0].boxes) > 0:
                for box in res[0].boxes:
                    cid = int(box.cls)
                    if 0 <= cid < len(CLASS_NAMES):
                        preds.append({
                            "class_id": cid,
                            "label": CLASS_NAMES[cid],
                            "confidence": float(box.conf),
                            "bbox": box.xyxy[0].tolist(),
                            "source": "control"
                        })
            stats["total_detections_pre_cap"] += len(preds)
            predictions[img_name] = preds
            save_counter += 1
            if save_counter % 10 == 0:
                save_checkpoint(checkpoint_path, predictions)
                
        stats["total_time"] = time.time() - start_time
        stats["max_gpu_mb"] = get_gpu_memory()
        save_checkpoint(checkpoint_path, predictions)
        
    print(f"[1/2] Control completed: {len(predictions)} images processed.")
    return predictions, stats

def run_tiled_evaluation(eval_images: List[Path], weights_path: str, output_dir: Path, slice_size: int = 1280, overlap: float = 0.2) -> Tuple[Dict[str, List[Dict[str, Any]]], dict]:
    system_key = f"tiled_e4_{slice_size}_{int(overlap*100)}"
    checkpoint_path = output_dir / f"checkpoint_{system_key}.json"
    predictions = load_checkpoint(checkpoint_path)
    
    print(f"\n[2/2] Running Tiled (E4 SAHI) on {len(eval_images)} images (slice={slice_size}, overlap={overlap})...")
    remaining_images = [p for p in eval_images if p.name not in predictions]
    
    stats = {"total_time": 0.0, "total_detections_pre_cap": 0, "max_gpu_mb": 0.0, "capped_count": 0}
    
    if remaining_images:
        sahi_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=weights_path,
            confidence_threshold=0.001,
            device="cuda:0",
            image_size=slice_size
        )
        
        save_counter = 0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            
        start_time = time.time()
        for img_path in tqdm(remaining_images, desc=f"Tiled E4 ({slice_size}x{slice_size})"):
            img_name = img_path.name
            img_str = str(img_path)
            
            sahi_res = get_sliced_prediction(
                img_str,
                sahi_model,
                slice_height=slice_size,
                slice_width=slice_size,
                overlap_height_ratio=overlap,
                overlap_width_ratio=overlap,
                verbose=0
            )
            
            preds = []
            for obj in sahi_res.object_prediction_list:
                lbl = obj.category.name.lower()
                cid = CLASS_LIST.index(lbl) if lbl in CLASS_LIST else int(obj.category.id)
                if 0 <= cid < len(CLASS_NAMES):
                    preds.append({
                        "class_id": cid,
                        "label": CLASS_NAMES[cid],
                        "confidence": float(obj.score.value),
                        "bbox": [
                            float(obj.bbox.minx),
                            float(obj.bbox.miny),
                            float(obj.bbox.maxx),
                            float(obj.bbox.maxy)
                        ],
                        "source": "tiled"
                    })
            
            stats["total_detections_pre_cap"] += len(preds)
            if len(preds) > 300:
                stats["capped_count"] += 1
                preds.sort(key=lambda x: x["confidence"], reverse=True)
                preds = preds[:300]
                
            predictions[img_name] = preds
            save_counter += 1
            if save_counter % 5 == 0:
                save_checkpoint(checkpoint_path, predictions)
                
        stats["total_time"] = time.time() - start_time
        stats["max_gpu_mb"] = get_gpu_memory()
        save_checkpoint(checkpoint_path, predictions)
        
    print(f"[2/2] Tiled completed: {len(predictions)} images processed.")
    return predictions, stats

def main():
    parser = argparse.ArgumentParser(description="E4 Tiled Inference Benchmark")
    parser.add_argument("--test-subset", type=int, default=0, help="Run on N images only for sanity check")
    parser.add_argument("--slice-size", type=int, default=1280, help="SAHI tile size")
    parser.add_argument("--overlap", type=float, default=0.2, help="SAHI overlap ratio")
    args = parser.parse_args()
    
    weights_path = str(PROJECT_ROOT / "runs" / "detect" / "experiments" / "model_search" / "E3_yolo11l_1536_aug" / "weights" / "best.pt")
    
    candidates = [
        PROJECT_ROOT / "datasets" / "VisDrone2019" / "VisDrone2019-DET-val",
    ]
    val_dir = next((c for c in candidates if c.exists()), candidates[0])
    images_dir = val_dir / "images"
    
    output_dir = PROJECT_ROOT / "runs" / "detect" / "experiments" / "inference_search" / "E4_gpu_tiled"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_images = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
    if args.test_subset > 0:
        all_images = all_images[:args.test_subset]
        print(f"Running SANITY CHECK on {args.test_subset} images.")
    
    
    preds_tiled, stats_tiled = run_tiled_evaluation(all_images, weights_path, output_dir, slice_size=args.slice_size, overlap=args.overlap)
    
    print("
[1/2] Running Control (E3 Standard Inference) using YOLO.val()...")
    model = YOLO(weights_path)
    
    start_time = time.time()
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    
    val_results = model.val(data="configs/visdrone.yaml", imgsz=1536, split="val", device="0", verbose=False)
    
    stats_control = {
        "total_time": (val_results.speed['preprocess'] + val_results.speed['inference'] + val_results.speed['postprocess']) * len(all_images) / 1000.0,
        "total_detections_pre_cap": 0,
        "max_gpu_mb": get_gpu_memory(),
        "capped_count": 0
    }
    
    metrics_control = {
        "overall": {
            "map50": val_results.results_dict.get("metrics/mAP50(B)", 0.0),
            "map50_95": val_results.results_dict.get("metrics/mAP50-95(B)", 0.0),
            "precision": val_results.results_dict.get("metrics/precision(B)", 0.0),
            "recall": val_results.results_dict.get("metrics/recall(B)", 0.0)
        },
        "class_metrics": []
    }
    for cid in range(10):
        metrics_control["class_metrics"].append({
            "class_name": CLASS_NAMES[cid],
            "ap50": val_results.box.maps[cid] if hasattr(val_results.box, 'maps') and len(val_results.box.maps) > cid else 0.0
        })
    preds_control = {img.name: [] for img in all_images} # dummy empty predictions just for counting
    

    evaluator = COCOEvaluatorWrapper(val_dir=val_dir)
    metrics_tiled = evaluator.evaluate_predictions(preds_tiled, system_name="Tiled (E4)")
    
    
    summary = []
    runs_data = [
        ("Control (E3)", metrics_control, preds_control, stats_control, f"1536 full-image inference"),
        ("Tiled (E4)", metrics_tiled, preds_tiled, stats_tiled, f"slice={args.slice_size}, overlap={args.overlap}")
    ]
    
    for sys_name, metrics, preds, run_stats, config in runs_data:
        num_processed = len(preds)
        total_post_cap = sum(len(p) for p in preds.values())
        avg_det = total_post_cap / max(1, num_processed)
        avg_time = run_stats["total_time"] / max(1, num_processed) if run_stats["total_time"] > 0 else 0
        
        row = {
            "System": sys_name,
            "mAP@50": metrics["overall"]["map50"],
            "mAP@50-95": metrics["overall"]["map50_95"],
            "Precision": metrics["overall"]["precision"],
            "Recall": metrics["overall"]["recall"],
            "Total_Detections": total_post_cap,
            "Avg_Detections_Per_Img": round(avg_det, 2),
            "Images_Processed": num_processed,
            "Inference_Time_sec": round(run_stats["total_time"], 2),
            "Avg_Time_Per_Img": round(avg_time, 3),
            "Max_GPU_Memory_MB": round(run_stats["max_gpu_mb"], 2),
            "PreCap_Total_Detections": run_stats["total_detections_pre_cap"],
            "Capped_Images_Count": run_stats["capped_count"],
            "Configuration": config
        }
        for cls_metrics in metrics["class_metrics"]:
            row[f"AP50_{cls_metrics['class_name']}"] = cls_metrics["ap50"]
        summary.append(row)
        
    csv_path = output_dir / "E4_gpu_tiled_comparison.csv"
    if summary:
        fieldnames = list(summary[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary)
            
    json_path = output_dir / "E4_gpu_tiled.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "control": metrics_control, "tiled": metrics_tiled}, f, indent=2)
        
    print(f"\nSaved CSV comparison to {csv_path}")
    print(f"Saved JSON results to {json_path}")
    
    print("\n--- QUICK RESULTS ---")
    print(f"Control mAP50: {metrics_control['overall']['map50']:.4f}")
    print(f"Tiled   mAP50: {metrics_tiled['overall']['map50']:.4f}")
    diff = metrics_tiled['overall']['map50'] - metrics_control['overall']['map50']
    print(f"Difference:   {diff:+.4f}")

if __name__ == "__main__":
    main()
