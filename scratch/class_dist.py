import os
from collections import Counter
from pathlib import Path

train_labels = Path("datasets/VisDrone2019-YOLO/train/labels")
val_labels = Path("datasets/VisDrone2019-YOLO/val/labels")

train_dist = Counter()
val_dist = Counter()

def count_labels(label_dir, dist):
    for f in label_dir.glob("*.txt"):
        with open(f, "r") as fh:
            for line in fh:
                parts = line.strip().split()
                if parts:
                    dist[int(parts[0])] += 1

count_labels(train_labels, train_dist)
count_labels(val_labels, val_dist)

class_names = {
    0: "pedestrian", 1: "people", 2: "bicycle", 3: "car", 4: "van",
    5: "truck", 6: "tricycle", 7: "awning-tricycle", 8: "bus", 9: "motor"
}

print("Train Distribution:")
for k, v in sorted(train_dist.items()):
    print(f"  {class_names[k]}: {v}")
    
print("Val Distribution:")
for k, v in sorted(val_dist.items()):
    print(f"  {class_names[k]}: {v}")
