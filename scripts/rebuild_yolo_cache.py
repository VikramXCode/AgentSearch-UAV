#!/usr/bin/env python3
"""
Rebuild the YOLO labels.cache for the VisDrone YOLO dataset.
This bypasses any stale/corrupt cache from a different machine.

Run: python3 scripts/rebuild_yolo_cache.py
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)

# ── delete ALL stale caches ────────────────────────────────────────────────
cache_files = list(Path("datasets/VisDrone2019-YOLO").rglob("*.cache"))
for c in cache_files:
    c.unlink()
    print(f"Deleted: {c}")
print(f"Deleted {len(cache_files)} cache files")
print()

# ── verify dataset integrity ───────────────────────────────────────────────
for split in ["train", "val"]:
    img_dir = Path(f"datasets/VisDrone2019-YOLO/{split}/images")
    lbl_dir = Path(f"datasets/VisDrone2019-YOLO/{split}/labels")

    imgs = {f.stem for f in img_dir.glob("*.jpg")}
    lbls = {f.stem for f in lbl_dir.glob("*.txt")}
    matched = imgs & lbls
    orphan_lbls = lbls - imgs
    orphan_imgs = imgs - lbls

    print(f"[{split}] images={len(imgs)}  labels={len(lbls)}  matched={len(matched)}")

    if orphan_lbls:
        print(f"  WARNING: {len(orphan_lbls)} label files have no matching image!")
        print("  Removing orphan labels...")
        for stem in orphan_lbls:
            (lbl_dir / f"{stem}.txt").unlink(missing_ok=True)
        print(f"  Removed {len(orphan_lbls)} orphan label files")

    if orphan_imgs:
        print(f"  NOTE: {len(orphan_imgs)} images have no label (will be treated as background)")

print()
print("Dataset clean. Now force-building fresh cache via ultralytics...")

# ── force build fresh cache ────────────────────────────────────────────────
from ultralytics.data.dataset import YOLODataset
from ultralytics.utils import LOGGER

for split in ["train", "val"]:
    img_path = str(Path(f"datasets/VisDrone2019-YOLO/{split}/images").resolve())
    print(f"Building cache for {split}...")
    try:
        ds = YOLODataset(
            img_path=img_path,
            data={
                "nc": 10,
                "names": {
                    0: "pedestrian", 1: "people", 2: "bicycle", 3: "car",
                    4: "van", 5: "truck", 6: "tricycle", 7: "awning-tricycle",
                    8: "bus", 9: "motor",
                },
                "channels": 3,
            },
            task="detect",
            augment=False,
        )
        print(f"  [{split}] Cache built successfully: {len(ds)} images")
    except Exception as e:
        print(f"  [{split}] Cache build failed: {e}")

print()
print("Done! You can now run training.")
