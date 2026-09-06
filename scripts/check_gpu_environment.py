from __future__ import annotations

import importlib
import platform
import sys


REQUIRED_PACKAGES = ["torch", "torchvision", "ultralytics", "yaml", "PIL", "cv2", "sahi"]


def main() -> int:
    print(f"Python: {platform.python_version()} ({sys.executable})")

    torch = try_import("torch")
    ultralytics = try_import("ultralytics")

    if torch is not None:
        print(f"PyTorch: {torch.__version__}")
        print(f"PyTorch CUDA version: {getattr(torch.version, 'cuda', None)}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                print(f"CUDA device {index}: {torch.cuda.get_device_name(index)}")
        else:
            print("TRAINING SHOULD NOT START.")
    else:
        print("PyTorch: missing")
        print("TRAINING SHOULD NOT START.")

    if ultralytics is not None:
        print(f"Ultralytics: {ultralytics.__version__}")
    else:
        print("Ultralytics: missing")

    print("\nImport checks:")
    all_ok = True
    for package_name in REQUIRED_PACKAGES:
        module = try_import(package_name)
        ok = module is not None
        all_ok = all_ok and ok
        print(f"  {package_name}: {'OK' if ok else 'MISSING'}")

    if torch is None or not torch.cuda.is_available():
        return 1
    if not all_ok:
        return 1
    return 0


def try_import(module_name: str):
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None


if __name__ == "__main__":
    raise SystemExit(main())