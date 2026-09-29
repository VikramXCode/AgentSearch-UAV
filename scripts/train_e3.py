import torch
import time
import json
import os
from ultralytics import YOLO
from datetime import datetime, timezone
import subprocess

def get_git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    except Exception:
        return "N/A"

def main():
    print("=== EXPERIMENT E3: YOLO11-LARGE AUGMENTATION TUNING (1536) ===")
    
    # 1. Config
    model_id = "yolo11l.pt"  # Pretrained base (controlled experiment)
    dataset_yaml = "configs/visdrone.yaml"
    imgsz = 1536
    batch_size = 4
    device = 0
    epochs = 100
    patience = 30
    project_dir = "experiments/model_search"
    experiment_name = "E3_yolo11l_1536_aug"
    
    # Augmentation changes
    # degrees: 10.0 -> Random rotation in [-10, +10] degrees.
    # VisDrone images are captured from UAVs, which naturally experience
    # rotation (roll/yaw variations) relative to the ground.
    # This geometric augmentation forces rotation invariance without degrading contrast.
    aug_args = {
        "degrees": 10.0
    }
    
    # Paths
    os.makedirs(project_dir, exist_ok=True)
    json_record_path = os.path.join(project_dir, f"{experiment_name}.json")
    
    # 2. Setup Record
    record = {
        "experiment_id": experiment_name,
        "git_commit": get_git_commit(),
        "model": "YOLO11-Large",
        "parameters": 0,
        "pretrained_checkpoint": model_id,
        "dataset": dataset_yaml,
        "train_images": 6471,
        "val_images": 548,
        "imgsz": imgsz,
        "batch": batch_size,
        "epochs": epochs,
        "device": f"cuda:{device}",
        "optimizer": "auto",
        "learning_rate": "auto",
        "augmentation_configuration": aug_args,
        "best_epoch": None,
        "best_validation_mAP50": None,
        "best_validation_mAP50_95": None,
        "precision": None,
        "recall": None,
        "training_time_seconds": None,
        "inference_time": None,
        "checkpoint_path": None,
        "timestamp_start": datetime.now(timezone.utc).isoformat()
    }
    
    # 3. Load Model with Resume Logic
    last_ckpt = f"runs/detect/{project_dir}/{experiment_name}/weights/last.pt"
    if os.path.exists(last_ckpt):
        print(f"\nFound existing checkpoint at {last_ckpt}. Resuming training...")
        model = YOLO(last_ckpt)
        resume_flag = True
    else:
        print(f"\nLoading model: {model_id}...")
        model = YOLO(model_id)
        resume_flag = False
        
    record["parameters"] = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {record['parameters']:,}")
    
    # 4. Train
    print(f"\nStarting Training (100 epochs, patience={patience}, imgsz={imgsz}, batch={batch_size})...")
    start_time = time.time()
    
    try:
        if resume_flag:
            results = model.train(resume=True)
        else:
            results = model.train(
                data=dataset_yaml,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch_size,
                device=device,
                project=project_dir,
                name=experiment_name,
                exist_ok=True,
                amp=True,
                workers=8,
                save=True,
                patience=patience,
                **aug_args
            )
        
        # 5. Extract Initial Metrics
        end_time = time.time()
        record["training_time_seconds"] = end_time - start_time
        
        # 6. Final Authoritative Evaluation
        best_ckpt = f"runs/detect/{project_dir}/{experiment_name}/weights/best.pt"
        if os.path.exists(best_ckpt):
            record["checkpoint_path"] = best_ckpt
            
            print("\n--- Running Final Authoritative Evaluation on Best Checkpoint ---")
            best_model = YOLO(best_ckpt)
            
            eval_results = best_model.val(
                data=dataset_yaml,
                imgsz=imgsz,
                batch=batch_size,
                device=device,
                split="val",
                project=project_dir,
                name=f"{experiment_name}_authoritative_eval",
                exist_ok=True,
                verbose=True
            )
            
            erd = getattr(eval_results, "results_dict", {})
            record["precision"] = erd.get("metrics/precision(B)")
            record["recall"] = erd.get("metrics/recall(B)")
            record["best_validation_mAP50"] = erd.get("metrics/mAP50(B)")
            record["best_validation_mAP50_95"] = erd.get("metrics/mAP50-95(B)")
            
            if hasattr(eval_results, "speed"):
                record["inference_time"] = eval_results.speed
            
            print(f"\nFINAL EVALUATION RESULTS:")
            if record["precision"] is not None: print(f"Precision: {record['precision']:.4f}")
            if record["recall"] is not None:    print(f"Recall:    {record['recall']:.4f}")
            if record["best_validation_mAP50"] is not None: print(f"mAP@50:    {record['best_validation_mAP50']:.4f}")
            if record["best_validation_mAP50_95"] is not None: print(f"mAP@50-95: {record['best_validation_mAP50_95']:.4f}")
            
    except Exception as e:
        print(f"\nTRAINING FAILED: {repr(e)}")
        record["error"] = repr(e)
        
    # 7. Save Record
    record["timestamp_end"] = datetime.now(timezone.utc).isoformat()
    with open(json_record_path, "w") as f:
        json.dump(record, f, indent=4)
        
    print(f"\nExperiment record saved to {json_record_path}")

if __name__ == "__main__":
    main()
