# EXP07: Open-Vocabulary Ground-Truth Evaluation

This experiment is currently **BLOCKED (REQUIRES_MANUAL_INTERVENTION)**.

## Goal
To quantitatively evaluate the open-vocabulary retrieval capabilities of YOLO-World models against actual out-of-vocabulary (OOV) ground truth targets, independent of the 548-image E3 VisDrone 10-class benchmark.

## Manual Intervention Required
Before this experiment can be run, you must manually create a dedicated OOV annotation subset. 
Do NOT invent or assume ground truth. 

### Annotation Format Requirements
1. Inspect the available VisDrone validation images.
2. Determine whether suitable OOV objects exist (e.g. "building", "tree", "river", "boat", "dog").
3. Create a manually verified annotation dataset. For each image/query pair record:
   - `image_id`
   - `query`
   - `target_present` (boolean)
   - `ground_truth_boxes` (if present)
4. You must include both **positive** (query object exists in image) and **negative** (query object does not exist in image) cases.

## Evaluation Requirements
The OOV evaluation script (to be written once the dataset is ready) must report:
- Precision
- Recall
- AP@0.50
- mAP@0.50 across evaluated OOV categories
- Query-level recall
- False positives on negative query cases

## Models to Evaluate
Compare the available OV detectors:
1. `YOLO-World-S` (`yolov8s-world.pt`)
2. `YOLO-World-L/v2` (if manually downloaded)

*Note: E3 (`E3_yolo11l_1536_aug/weights/best.pt`) MUST NOT be used here, as it is a frozen VisDrone specialist and has lost its open-vocabulary capacity.*

## Configurations to Evaluate
If CLIP verification is part of the final pipeline, evaluate:
1. OV detector alone
2. OV detector + CLIP
as separate configurations.

## Reporting
Do NOT claim that this EXP07 OOV mAP50 is directly comparable to E3's 62.10% mAP50. They measure different capabilities on different sets.
