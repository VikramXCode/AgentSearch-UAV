from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.model_paths import resolve_yolo_world_weights


def load_config(config_path: Path) -> dict:
    try:
        import yaml  # type: ignore
    except ImportError:
        print("PyYAML is missing. Using built-in training defaults instead of the config file.")
        return {}

    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Training config must be a mapping: {config_path}")
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare YOLO-World fine-tuning on VisDrone.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "train_visdrone.yaml")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved training configuration and exit.")
    parser.add_argument("--allow-cpu-debug", action="store_true", help="Allow a non-training debug run even when CUDA is unavailable.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--project", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--optimizer", default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--lr0", type=float, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    model_path = resolve_yolo_world_weights(args.model or config.get("model"))
    data_path = str((PROJECT_ROOT / (args.data or config.get("data", "configs/visdrone_yolo_prepared.yaml"))).resolve())

    train_kwargs = {
        "data": data_path,
        "epochs": args.epochs if args.epochs is not None else config.get("epochs", 50),
        "imgsz": args.imgsz if args.imgsz is not None else config.get("imgsz", 960),
        "batch": args.batch if args.batch is not None else config.get("batch", 8),
        "device": args.device if args.device is not None else config.get("device", 0),
        "workers": args.workers if args.workers is not None else config.get("workers", 8),
        "seed": args.seed if args.seed is not None else config.get("seed", 42),
        "project": args.project if args.project is not None else config.get("project", "runs/visdrone_yoloworld"),
        "name": args.name if args.name is not None else config.get("name", "visdrone_v1"),
        "optimizer": args.optimizer if args.optimizer is not None else config.get("optimizer", "auto"),
        "patience": args.patience if args.patience is not None else config.get("patience", 15),
        "lr0": args.lr0 if args.lr0 is not None else config.get("lr0", 0.001),
        "cos_lr": config.get("cos_lr", True),
        "rect": config.get("rect", True),
        "amp": config.get("amp", True),
        "cache": config.get("cache", False),
        "plots": config.get("plots", True),
        "exist_ok": config.get("exist_ok", True),
        "verbose": config.get("verbose", True),
        "close_mosaic": config.get("close_mosaic", 10),
    }

    if not args.allow_cpu_debug:
        try:
            import torch
        except ImportError:
            print("PyTorch is missing. TRAINING SHOULD NOT START.")
            return 1

        if not torch.cuda.is_available():
            print("CUDA is unavailable. TRAINING SHOULD NOT START.")
            print("Use --dry-run or --allow-cpu-debug only for local inspection, not real training.")
            return 1

    print(f"Resolved model: {model_path}")
    print(f"Resolved data: {data_path}")
    print("Resolved training configuration:")
    for key, value in train_kwargs.items():
        print(f"  {key}: {value}")

    if args.dry_run:
        print("Dry run complete. No training was started.")
        return 0

    try:
        from ultralytics import YOLO
    except ImportError:
        print("Ultralytics is missing. Install requirements before training.")
        return 1

    model = YOLO(model_path)

    if not hasattr(model, "set_classes"):
        print("Warning: loaded model does not expose set_classes(). Verify the checkpoint is YOLO-World.")

    if not args.allow_cpu_debug:
        try:
            import torch
        except ImportError:
            print("PyTorch is missing. Refusing to start training.")
            return 1

        if not torch.cuda.is_available():
            print("CUDA is unavailable. Refusing to start training.")
            return 1

    print("Starting YOLO-World fine-tuning on VisDrone...")
    results = model.train(**train_kwargs)
    save_dir = getattr(results, "save_dir", None) or getattr(getattr(model, "trainer", None), "save_dir", None)
    if save_dir is not None:
        print(f"Training artifacts saved to: {save_dir}")
        print(f"Best checkpoint: {Path(save_dir) / 'weights' / 'best.pt'}")
        print(f"Last checkpoint: {Path(save_dir) / 'weights' / 'last.pt'}")
    print("Training command finished.")
    print("Reminder: fine-tuning can reduce open-vocabulary behavior; run the regression checks before adopting the checkpoint.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())