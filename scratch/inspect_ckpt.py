import torch

ckpt_path = "runs/detect/experiments/model_search/E2_yolo11l_1536/weights/last.pt"
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

print("--- Checkpoint Metadata ---")
if "epoch" in ckpt:
    print(f"Epoch: {ckpt['epoch']}")
else:
    print("Epoch not found in checkpoint.")

if "train_args" in ckpt:
    args = ckpt["train_args"]
    print(f"Model: {args.get('model')}")
    print(f"Image Size: {args.get('imgsz')}")
    print(f"Batch Size: {args.get('batch')}")
    print(f"Max Det: {args.get('max_det')}")
    print(f"Optimizer: {args.get('optimizer')}")
else:
    print("Train args not found.")
