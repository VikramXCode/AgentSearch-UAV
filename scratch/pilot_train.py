import torch
import time
import os
from ultralytics import YOLO

def main():
    print("--- YOLO11-Large Pilot Training (Phase 8) ---")
    
    model_id = "yolo11l.pt"
    dataset_yaml = "configs/visdrone.yaml"
    imgsz = 1280
    batch_size = 8
    device = 0
    epochs = 1
    project_dir = "experiments/model_search"
    experiment_name = "yolo11l_pilot"
    
    print(f"Model Identifier: {model_id}")
    print(f"Dataset YAML: {dataset_yaml}")
    print(f"Image Size: {imgsz}")
    print(f"Batch Size: {batch_size} (conservative for 40GB A100)")
    print(f"Device: cuda:{device} ({torch.cuda.get_device_name(device) if torch.cuda.is_available() else 'CPU'})")
    print(f"Target Checkpoint Dir: {project_dir}/{experiment_name}")
    
    checkpoint_path = f"runs/detect/{project_dir}/{experiment_name}/weights/last.pt"
    if os.path.exists(checkpoint_path):
        print(f"\nFound existing checkpoint at {checkpoint_path}. Resuming training...")
        model = YOLO(checkpoint_path)
        resume_flag = True
    else:
        print("\nLoading new model from scratch...")
        model = YOLO(model_id)
        resume_flag = False
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters: {total_params:,}")
    
    print("\nStarting 1-epoch pilot training...")
    start_time = time.time()
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    
    try:
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
            val=False,  
            resume=resume_flag,
            patience=10  
        )
        success = True
    except Exception as e:
        print(f"\nTRAINING FAILED: {repr(e)}")
        success = False
        
    end_time = time.time()
    duration = end_time - start_time
    
    if torch.cuda.is_available():
        peak_mem_alloc = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        peak_mem_res = torch.cuda.max_memory_reserved(device) / (1024 ** 3)
    else:
        peak_mem_alloc, peak_mem_res = 0.0, 0.0
    
    print("\n--- Pilot Summary ---")
    print(f"Training Successful: {success}")
    print(f"Training Duration: {duration:.2f} seconds")
    print(f"Peak GPU Memory (Allocated): {peak_mem_alloc:.2f} GB")
    print(f"Peak GPU Memory (Reserved): {peak_mem_res:.2f} GB")
    
    if success:
        print(f"Output Checkpoint: runs/detect/{project_dir}/{experiment_name}/weights/last.pt")

if __name__ == "__main__":
    main()
