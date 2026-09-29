import torch
from ultralytics import YOLO

def memory_smoke_test():
    print("--- 1536 MEMORY SMOKE TEST ---")
    model = YOLO("models/yolov8s-p2.yaml").load("weights/yolov8s.pt")
    
    def on_train_batch_end(trainer):
        print("\n--- BATCH ENDED (Loss and Backward successful) ---")
        if torch.cuda.is_available():
            print(f"Peak Memory Allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
            print(f"Peak Memory Reserved: {torch.cuda.max_memory_reserved() / 1e9:.2f} GB")
            total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"Total Device Memory: {total_mem:.2f} GB")
            print(f"Approximate Headroom: {total_mem - (torch.cuda.max_memory_reserved() / 1e9):.2f} GB")
        raise KeyboardInterrupt("Smoke test completed.")
        
    model.add_callback("on_train_batch_end", on_train_batch_end)

    try:
        model.train(
            data="configs/visdrone.yaml",
            epochs=1,
            batch=2,
            imgsz=1536,
            device='0' if torch.cuda.is_available() else 'cpu',
            project='scratch',
            name='test_memory',
            exist_ok=True,
            workers=0,
            amp=True
        )
    except KeyboardInterrupt:
        print("Smoke test successfully exited after 1 batch.")
    except Exception as e:
        print(f"CUDA OOM or other error: {e}")

if __name__ == "__main__":
    memory_smoke_test()
