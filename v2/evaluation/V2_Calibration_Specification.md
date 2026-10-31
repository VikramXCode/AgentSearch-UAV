# Phase 5D: V2 Verification Scope & Calibration Data Specification

## 1. Audit of Current Verification Capabilities
Based on an inspection of `v2/agents/verification_agent.py` and the V2 schemas:

### A. Text query → Candidate verification
- **Supported Inputs:** Any natural language text string encoded as an `attribute` constraint.
- **CLIP Scoring:** Uses `CLIPEngineAdapter` to score a candidate crop's visual embedding against the text query's semantic embedding (cosine similarity). 
- **Constraints Evaluated:** `attribute` constraints (e.g., "red car", "damaged roof").
- **Deterministic vs Semantic:** The aggregation of scores and threshold comparison is deterministic. The similarity scalar output itself is completely model-dependent (semantic).
- **Unsupported:** Complex spatially-aware semantics in text (e.g., "car *next to* a house"), as CLIP struggles with compositional relations.

### B. Reference image → Candidate verification
- **Supported Inputs:** Path to a local reference image.
- **Capabilities Verified:** This computes **visual similarity** (global appearance similarity) between the reference image and the candidate crop via CLIP embeddings.
- **Instance Recognition / ReID:** **NOT SUPPORTED.** The current implementation does *not* possess temporal awareness, part-based ReID matching, or track identity association. It merely asserts whether two crops share visual features in the CLIP latent space.

### C. Spatial Constraints
- **Supported Inputs:** "left", "right", "top", "bottom".
- **Deterministic Execution:** Resolves by comparing the candidate's bounding box center (cx, cy) strictly against the image's center point (w/2, h/2). Fully deterministic.

### D. Attribute Constraints
- **Implementation:** Passes the text string to the semantic backend. 
- **Limitations:** CLIP is a zero-shot global classifier. It can establish broad visual features (colors, generic subtypes) but struggles with fine-grained counting or specific states unless heavily represented in its training distribution. 

### E. Relation Constraints
- **Status:** **UNSUPPORTED**. Any constraint tagged as `relation` immediately evaluates to `ConstraintStatus.UNSUPPORTED`. 

---

## 2. Capability Definitions & Separation
To prevent capability conflation, V2 officially distinguishes the following definitions:
- **Semantic Similarity Verification:** Mapping natural language attributes to a crop's visual features (Implemented via CLIP).
- **Reference-Image Visual Similarity:** Measuring global appearance similarity between two image crops (Implemented via CLIP).
- **Instance Recognition / Temporal Identity (ReID) / Tracking Association:** Tracking the *exact same physical object* across disparate frames or cameras. **(NOT IMPLEMENTED).** We do not claim our visual similarity threshold guarantees physical identity.

---

## 3. Text Query Calibration Requirements
V2 text queries are intended to cover **attributes, subtypes, and object states** (e.g., "white truck", "police car", "damaged vehicle"). 

**Required Calibration Data:**
- **Positive Pair:** `(Crop A, "white truck")` where Crop A is explicitly labeled as a white truck.
- **Negative Pair:** `(Crop A, "blue truck")` (trivial negative) or `(Crop A, "white van")` (hard negative).
- **Hard-Negative Definition:** Crops that share the noun but not the adjective, or vice-versa, to ensure CLIP isn't just detecting the base class (e.g. relying on "truck" and ignoring "white").
- **Minimum Metadata:** Crop image, bounding box coordinates, source image ID, ground-truth attribute labels.
- **Leakage Risk:** If the calibration set shares images or sequences with the final evaluation/private set, thresholds will overfit. 

---

## 4. Reference-Image Calibration Requirements
V2 intends to support **Visual Similarity Search** (finding objects that look highly similar to the reference). It does **NOT** claim Exact-Instance Retrieval (ReID), which remains future work.

**Required Calibration Data:**
- **Positive Pair:** `(Crop A, Crop B)` where both crops belong to the same fine-grained visual sub-category (e.g., two different red sedans of the same model).
- **Negative Pair:** `(Crop A, Crop C)` where crops are visually distinct but perhaps share the base class (e.g., red sedan vs. blue sedan).
- **Hard-Negative Definition:** Same base class, similar color, but different sub-type (e.g., red sedan vs. red hatchback).

---

## 5. Dataset Requirements Table

| Capability | Required Pair Type | Positive Definition | Negative Definition | Hard-Negative | Required Annotation | External Data Required? |
|---|---|---|---|---|---|---|
| Text Attributes | Text $\rightarrow$ Crop | Crop matches attribute label | Crop lacks attribute | Same class, different attribute | Multi-label attributes | **YES** |
| Visual Similarity | Image $\rightarrow$ Crop | Crops share fine-grained visual appearance | Crops differ in appearance | Same class & color, different subtype | Fine-grained classification / pairs | **YES** |

*Note: The current repository (VisDrone-DET) provides bounding boxes and base classes, but completely lacks attribute tags and fine-grained similarity pairs.*

---

## 6. Calibration vs. Final Evaluation Split
To ensure scientific integrity:
1. **Development/Debug Data:** Trivial static images used strictly for integration smoke-testing (e.g. `scratch/test_image.jpg`).
2. **Threshold Calibration Data:** An empirical dataset used *exclusively* to plot score distributions and select candidate thresholds (Accept/Reject boundaries).
3. **Final/Private Evaluation Data:** The hidden benchmark. Must **never** be used to select thresholds, tune prompts, or tweak logic. 

---

## 7. Threshold Policy
V2 will adhere to the following empirical threshold process:
1. Process the isolated Calibration Dataset through the V2 verification agent.
2. Record the score distributions for `MATCH` and `NON_MATCH` populations.
3. Utilize `ThresholdEvaluator` to compute continuous precision/recall curves.
4. Report discrete candidate operating points (e.g., Balanced F1, High Precision, High Recall).
5. **No threshold will be set using the final evaluation/private scenario data.**

---

## 8. Current Evidence Matrix

| Capability | Implemented | Real Smoke-Tested | Empirically Calibrated | Final-Evaluated | Evidence Available |
|---|---|---|---|---|---|
| Base Object Detection (V1) | YES | YES | PARTIAL | YES | V1 Reports |
| V2 Text Verification (Semantic) | YES | YES | **NO** | NO | Synthetic Demo Only |
| V2 Image Verification (Visual) | YES | YES | **NO** | NO | Synthetic Demo Only |
| V2 Exact-Instance ReID | **NO** | N/A | N/A | N/A | None |
| V2 Spatial Verification | YES | NO (Mocked) | NO | NO | None |

---

## 9. Recommendations for Next Phase
- **External Data is Required:** The current repository cannot calibrate the V2 semantic logic. 
- **Exact Data Type Needed:** We require a dataset annotated with UAV-perspective vehicle/pedestrian **attributes** (colors, types) OR a Multi-Object Tracking (MOT) sequence that provides guaranteed instance pairings to approximate visual similarity ground truth. 
- **Next Step:** We recommend pausing architectural modifications and authorizing the acquisition/construction of a scoped calibration dataset (e.g., downloading a subset of VisDrone-MOT or an attribute dataset like UAV-Attribute) before proceeding to final production integration.

## 10. Safety Validation
- **V1 Safety:** Untouched.
- **Git Status:** Clean. No files were modified. 
- **Execution:** Stopped successfully.
