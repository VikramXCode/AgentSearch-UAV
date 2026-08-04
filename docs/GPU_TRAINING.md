# GPU Training Guide

This repository is prepared for a future YOLO-World fine-tuning run on the VisDrone2019-DET dataset. The commands below are for the college NVIDIA GPU machine.

## 1. Clone or pull the repository

```bash
git pull
```

## 2. Create and activate an environment

```bash
python -m venv .venv
source .venv/bin/activate
```

## 3. Install dependencies

Install the general Python dependencies first:

```bash
pip install -r requirements.txt
```

Then install the GPU-compatible PyTorch wheels from the official PyTorch index that matches the college machine:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

If the college machine uses a different CUDA release, use the matching PyTorch wheel index instead.

## 4. Place the dataset

The dataset must live at:

```text
datasets/VisDrone2019/
```

with the existing train/val image and label folders already converted to YOLO format.

## 5. Verify the GPU

```bash
python scripts/check_gpu_environment.py
```

If the script prints `TRAINING SHOULD NOT START.`, stop and fix the environment first.

## 6. Validate the dataset

```bash
python scripts/validate_visdrone_dataset.py
```

You should see a `PASS` summary before training.

Current dataset state: the train/val image and label counts match, but the strict label validator reports many boxes that extend outside the normalized image bounds. That is common for edge-touching VisDrone annotations, but you should decide whether to keep the raw labels or generate a sanitized derived copy before long training runs.

## 7. Preview labels visually

```bash
python scripts/visualize_visdrone_labels.py --split both --count 6
```

Annotated verification images will be written to:

```text
outputs/dataset_validation/
```

## 8. Run training

The prepared training command is:

```bash
python scripts/train_visdrone.py
```

This uses the baseline YOLO-World checkpoint at `weights/yolov8s-world.pt` and writes outputs to a separate `runs/visdrone_yoloworld/` directory.

## 9. Find the trained checkpoint

After training, the expected files are:

```text
runs/visdrone_yoloworld/<run_name>/weights/best.pt
runs/visdrone_yoloworld/<run_name>/weights/last.pt
```

## 10. Evaluate the checkpoint

```bash
python scripts/evaluate_visdrone.py --checkpoint runs/visdrone_yoloworld/<run_name>/weights/best.pt
```

## 11. Compare against the baseline

```bash
python scripts/compare_visdrone_results.py --fine-tuned experiments/<run_name>_visdrone_eval.json
```

## 12. Run open-vocabulary regression checks

```bash
python -m models.inference
```

Use the known regression pair:

- query: `find a dog`
- image: `sample_images/world.png`

The current baseline result is approximately 3 dogs with average confidence around 0.914. Re-check that this capability still behaves sensibly after fine-tuning.

## 13. Integrate the checkpoint only if acceptable

Do not replace `weights/yolov8s-world.pt`.

Instead, point inference to the approved checkpoint with the environment variable:

```bash
export AGENTSEARCH_YOLO_WORLD_WEIGHTS=runs/visdrone_yoloworld/<run_name>/weights/best.pt
```

You can then run the normal CLI or API without changing source code.

## Open-vocabulary warning

Fine-tuning on VisDrone can improve the ten VisDrone categories while reducing text-query flexibility on objects outside those classes. Keep the baseline checkpoint and run regression checks against sample queries before promoting a fine-tuned model.