import torch
from ultralytics import YOLO

def test_memory(batch_size):
    print(f"\n--- TESTING BATCH SIZE {batch_size} ---")
    model = YOLO("models/yolov8s-p2.yaml").load("weights/yolov8s.pt")
    
    def on_train_batch_end(trainer):
        print(f"\n--- BATCH {batch_size} ENDED (Loss and Backward successful) ---")
        if torch.cuda.is_available():
            print(f"Peak Memory Allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
            print(f"Peak Memory Reserved: {torch.cuda.max_memory_reserved() / 1e9:.2f} GB")
        raise KeyboardInterrupt("Smoke test completed.")
        
    model.add_callback("on_train_batch_end", on_train_batch_end)

    try:
        model.train(
            data="configs/visdrone.yaml",
            epochs=1,
            batch=batch_size,
            imgsz=1536,
            device='0' if torch.cuda.is_available() else 'cpu',
            project='scratch',
            name=f'test_memory_b{batch_size}',
            exist_ok=True,
            workers=0,
            amp=True
        )
        return True
    except KeyboardInterrupt:
        print(f"Batch {batch_size} smoke test successfully exited after 1 batch.")
        return True
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"CUDA OOM at batch size {batch_size}.")
            # Clear cache
            torch.cuda.empty_cache()
            return False
        else:
            print(f"RuntimeError: {e}")
            return False
    except Exception as e:
        print(f"Exception: {e}")
        return False

def main():
    if not test_memory(8):
        if not test_memory(4):
            test_memory(2)

if __name__ == "__main__":
    main()
