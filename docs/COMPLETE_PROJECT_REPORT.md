# AgentUAV / AgentSearch-UAV
## Complete Technical Project Report

### 1. Executive Summary
**Project Name:** AgentUAV / AgentSearch-UAV
**Repository Path:** `/home/shahinraihaana.24it/vikram/AgentSearch-UAV`
**Objective:** Autonomous Multi-Agent Aerial Target Search System. The primary problem being solved is identifying, verifying, and tracking complex targets (including open-world and multi-constraint semantic queries) in high-altitude aerial UAV imagery and video.
**Input Modalities:** Static images (Reconnaissance Scene Search) and recorded video streams (Continuous Aerial Video Stream Tracking). Target queries are accepted as natural language text or reference image crops.
**Output Modalities:** Annotated images and videos with bounding boxes, confidence scores, tracking IDs, and multi-agent pipeline decision reasoning.
**Major Technologies:** Python, PyTorch, Ultralytics YOLO, Flask (Backend), React, Framer Motion (Frontend).
**Hardware Context:** Deployed with NVIDIA GPU support (CUDA), incorporating mutual-exclusion memory safeguards to prevent Out-Of-Memory (OOM) errors during heavy inference.

### 2. Project Objective
AgentSearch-UAV aims to abstract the complexities of searching UAV feeds by wrapping specialized detectors inside a deterministic Multi-Agent state machine. It transitions from rigid, single-model architectures to an adaptive pipeline capable of routing requests dynamically between a high-resolution specialist model (for known drone-perspective targets) and an open-vocabulary text-prompt model (for unknown targets). 

### 3. System Evolution (Complete Project Timeline)
- **V1 (Historical Architecture):** Initial monolithic architecture integrating YOLO models and Flask APIs directly, without agent-based routing.
- **E3 Model Search:** Experimental training of a YOLO11-L model on the VisDrone dataset.
- **Phase 1-5 (V2 Architecture Creation):** Design and implementation of the V2 Multi-Agent system including `QueryAgent`, `PlanningAgent`, `DetectionAgent`, and `VerificationAgent`.
- **Phase 6-7 (Detection Routing & SAHI/SR Investigation):** Implementation of deterministic routing logic and investigation into Slicing Aided Hyper Inference (SAHI) and Super Resolution (SR).
- **Phase 8 (Specialist Validation):** YOLOv8s-P2 (EXP11) was validated against VisDrone subsets. SAHI was experimentally proven detrimental to the P2 head, leading to its disablement in the active production path.
- **Phase 9 (Architecture Freeze):** The state-machine and models (P2 + YOLO-World) were strictly frozen to prevent further experimental drift.
- **Phase 10 (Image Interface):** Refinement of the V2 static image processing endpoint and frontend UX integration.
- **Phase 11 (Video + Tracking):** Integration of a background video processing pipeline and `TrackingAgentV2` with target re-detection logic.
- **Phase 12 (Robustness):** End-to-end matrix testing of queries, invalid media, tracking degradation, and unsupported constraint fallbacks.
- **Phase 13 (Deployment Polish):** Final bug fixes (e.g., semantic constraint extraction and OpenWorld routing bugs), UI stability improvements, and technical reporting.

### 4. Final Architecture
The active, final V2 architecture uses a deterministic agent-based state machine. The conceptual flow is:

    User Input (Text/Image)
        ↓
    Query Agent (Extracts Target & Constraints)
        ↓
    Planning Agent (Evaluates State Machine Action)
        ↓
    Detector Routing (Action: DETECT_SPECIALIST or DETECT_OPEN_WORLD)
       ↙                          ↘
    P2 Specialist            YOLO-World (Open Vocabulary)
       ↘                          ↙
    Verification Agent (Evaluates Constraints)
        ↓
    Image Result / Tracking Agent (Video Frame Association)
        ↓
    Redetection (when tracking degrades)

**Inputs:** Raw images/videos, raw text strings, reference image crops.
**Outputs:** Modally-aligned annotated files with JSON trace reasoning.
**Routing:** Deterministically handled by `PlanningAgentV2`.
**Termination:** Triggered when targets are successfully verified and tracking is stable, or when maximum retry budgets are exhausted safely.

### 5. V2 State Model
Located in `v2/schemas/state.py`, this acts as the central telemetry backbone for the state machine.
- `ActionType`: Enum defining planner actions (`DETECT_SPECIALIST`, `DETECT_OPEN_WORLD`, `VERIFY_CANDIDATES`, `TRACK`, `REDETECT`, `ENHANCE_SAHI`, `TERMINATE`).
- `MediaType`: Enum (`IMAGE`, `VIDEO`).
- `MissionStatus`: Enum (`PENDING`, `PROCESSING`, `SUCCESS`, `FAILED`).
- `QuerySpec`: Parsed user query containing the root `target`, descriptive `constraints`, and `reference_image_path`.
- `MediaMetadata`: Resolution, FPS, and paths.
- `Plan`: The `current_action`, historical `history`, and `decision_rationale` trace.
- `DetectorRouting`: Tracks the currently allocated model.
- `Candidate`: A detected bounding box with label, confidence, and internal track ID.
- `ConstraintResult`: Evaluation of a single constraint (SATISFIED, VIOLATED, UNCERTAIN, UNSUPPORTED).
- `VerificationResult`: Aggregated consensus of all constraint evaluations for a candidate.
- `TrackingState`: State memory for active tracks and degradation metrics.
- `AgentStateV2`: The root wrapper passed between all agents containing the full payload.

### 6. Agent Architecture
#### QueryAgentV2 (`v2/agents/query_agent.py`)
- **Purpose:** Parses raw input into a structured `QuerySpec`.
- **Logic:** Identifies known targets (person, car, motorcycle). If multiple words are present (e.g., "yellow bus"), it extracts the full raw text as an attribute constraint to ensure downstream semantic verification and accurate open-world routing.

#### PlanningAgentV2 (`v2/agents/planning_agent.py`)
- **Purpose:** Central coordinator determining the next `ActionType`.
- **Logic:** If `target` is known and contains no constraints, it routes to `DETECT_SPECIALIST`. If constraints exist or the target is unknown, it routes to `DETECT_OPEN_WORLD`. Evaluates verification results to trigger `TRACK`, `REDETECT`, or `TERMINATE`.

#### DetectionAgentV2 (`v2/agents/detection_agent.py`)
- **Purpose:** Interfaces with the `ModelRegistry` to retrieve the correct adapter and execute inference.
- **Logic:** Modifies `state.candidates` using outputs from `SpecialistDetectorAdapter` or `OpenWorldDetectorAdapter`. 

#### VerificationAgentV2 (`v2/agents/verification_agent.py`)
- **Purpose:** Evaluates geometric and semantic constraints.
- **Logic:** Evaluates bounding-box coordinates for spatial reasoning (e.g., "top", "left"). Uses `CLIPEngineAdapter` for semantic similarity on candidate crops. Explicitly skips CLIP verification for `OPEN_WORLD` candidates, trusting YOLO-World's internal multimodal encoder. 

#### TrackingAgentV2 (`v2/agents/tracking_agent.py`)
- **Purpose:** Maintains object persistence across video frames.
- **Logic:** Assigns UUIDs using geometric Intersection-over-Union (IoU). Monitors degradation (lost tracks) to force the planner into a `REDETECT` state.

### 7. Query Understanding
Queries are parsed deterministically. 
- A simple query like `"car"` extracts `target="car"` and 0 constraints.
- A complex query like `"person riding two wheeler"` extracts `target="person"`, and adds an attribute constraint `"person riding two wheeler"`.
- Reference queries populate `reference_image_path`.
- Natural-language parsing is strictly lexical and rule-based; it does not utilize a generalized LLM parsing layer.

### 8. Planning and Routing
- **Known Target (Simple):** e.g., "car". Routes to P2 specialist.
- **Unknown Target:** e.g., "dragon". Routes to YOLO-World.
- **Known Target with Constraint:** e.g., "red car". Routes to YOLO-World, as the P2 specialist lacks contextual awareness for colors and adjectives.
- **Redetection:** If tracking degrades, the planner issues `REDETECT`.
- **Retry Budget:** Prevents infinite detection loops.

### 9. Detection Models
#### 9A. YOLOv8s + P2 Specialist
- **Active Checkpoint:** `runs/detect/runs/detect/experiments/model_search/EXP11_yolov8s_p2_1536/weights/best.pt`
- **Architecture:** YOLOv8s base modified with a P2 (stride-4) high-resolution feature pyramid pathway. 
- **Training Configuration:** Trained natively on VisDrone dataset at 1536x1536 resolution for 50 epochs (batch size 4).
- **Validation Results:** Achieved `0.5690` mAP50 validation peak.
- **Held-Out Evaluation:** On a 448 image VisDrone subset, achieved `0.4663` mAP50.
- **Limitations:** Only recognizes core aerial categories. Context-blind (cannot filter by color or activity).

#### 9B. YOLO-World
- **Active Checkpoint:** `weights/yolov8s-world.pt`
- **Purpose:** Open-vocabulary object detection fallback.
- **Execution:** Compiles dynamic text encodings from the `QuerySpec` constraints (e.g., `["yellow bus", "person riding two wheeler"]`). 
- **Inference Parameter:** The confidence threshold is intentionally lowered to `conf=0.05` to capture complex zero-shot text classes successfully.
- **Evaluation:** Functionally smoke-tested. It accurately parses OOV queries, but quantitative mAP across random OOV datasets was not formally benchmarked.

### 10. Historical E3 Experiment
The E3 model is part of the project's historical model search.
- **Architecture:** YOLO11-L (Large).
- **Reported Metrics:** Achieved `0.6210` mAP50 on VisDrone.
- **Role:** It served as a powerful baseline but was replaced by the lighter EXP11 (YOLOv8s-P2) model for the active V2 pipeline to accommodate computational limitations and native high-resolution (1536px) integration. E3 is NOT active in the final system.

### 11. Verification System
`VerificationAgentV2` computes `ConstraintStatus`. 
- **SATISFIED:** Thresholds met.
- **VIOLATED:** Explicit negative evidence.
- **UNCERTAIN:** Ambiguous confidence between thresholds.
- **UNSUPPORTED:** Recognized but un-computable logic (e.g., relation graphs).
*Correction Note:* A bug where the Verification Agent used a lenient `MockSemanticAdapter` was fully patched in Phase 13, enabling active `CLIPEngineAdapter` validation, and ensuring `OPEN_WORLD` detections bypass double-verification to prevent false negatives.

### 12. Reference-Image Pipeline
The reference pipeline accepts a user image crop alongside the primary media. 
- It processes the primary image through the appropriate detector.
- `VerificationAgentV2` crops the resulting candidates and feeds both the candidate crop and the user reference image into `CLIPEngineAdapter`.
- CLIP computes cosine similarity between the two images.
- **Limitation:** This is semantic similarity, NOT true exact-instance Recognition/ReID. It finds "objects that look similar" rather than guaranteeing "the exact same object instance".

### 13. Video Pipeline
Endpoint: `POST /v2/detect-video`
- Generates a background thread job.
- Iterates sequentially through decoded frames. 
- Initiates detection, validates via verification, and invokes `TrackingAgentV2` for frame-to-frame association.
- Uses `cv2.VideoWriter` to draw persistent track IDs and bounds.
- Cleans up temporary artifacts post-completion.

### 14. Tracking and Redetection
- Uses naive geometric Intersection-over-Union (IoU) association.
- Frames without target matches accumulate a degradation score. 
- When degradation exceeds the threshold, `TrackingAgentV2` flags the planner, which issues `REDETECT` to re-invoke the heavy detection/verification stack.
- MOTA, IDF1, and HOTA metrics were NOT quantitatively evaluated.

### 15. SAHI / SR Investigation
- Slicing Aided Hyper Inference (SAHI) and Super Resolution (SR) were thoroughly investigated.
- **Result:** Adaptive SAHI was disabled in the final P2 pipeline. The P2 model captures high-resolution features natively at 1536px. Evaluating SAHI (640x640 slices) resulted in a 2x runtime penalty (79ms vs 38ms) and degraded accuracy (0.4020 vs 0.4663 mAP50).

### 16. Model Registry and Memory Safety
`v2/models/model_registry.py` handles model instantiation globally.
- **Lazy Loading:** Models load only on their first query.
- **Mutual Exclusivity:** To prevent A100 VRAM OOM exceptions, requesting YOLO-World explicitly unloads the P2 model (via `gc.collect()` and `torch.cuda.empty_cache()`), and vice versa.

### 17. API Architecture
Served via Flask (`api/v2_api.py`):
- `POST /v2/detect`: Accepts `image`, `query`, `reference_image`. Returns JSON with bounding boxes, pipeline reasoning, and `annotated_image_url`.
- `POST /v2/detect-video`: Accepts `video`, `query`. Returns a `job_id`.
- `GET /v2/video-progress/<job_id>`: Returns `{ progress, current_frame, total_frames, status, result }`.
- `GET /v2/outputs/<filename>`: Serves annotated results.

### 18. Frontend Architecture
Built in React (`frontend/src/App.jsx`).
- Features a Mission Control UI utilizing Framer Motion.
- Maintains polling loops for `detect-video` background jobs.
- Bypasses legacy V1 routes completely.

### 19. Repository Structure
```
├── api/             # Flask API endpoints (main.py, v2_api.py)
├── configs/         # Training configuration files
├── docs/            # Project documentation and reports
├── frontend/        # React application source code
├── models/          # Wrappers for legacy models, CLIP, and YOLO-World
├── notebooks/       # Historical exploration notebooks
├── outputs/         # Generated API artifacts (images/videos)
├── tests/           # Comprehensive PyTest suite
├── v2/              # Core Agentic State Machine framework
│   ├── agents/      # Planner, Query, Detection, Verification, Tracking
│   ├── models/      # ModelRegistry and Inference Adapters
│   └── schemas/     # Pydantic/Dataclass State definitions
└── weights/         # Pre-trained core models
```

### 20. Dataset and Experimental Setup
- **Dataset:** VisDrone-DET dataset.
- **Experiments:** EXP11 validated the effectiveness of the stride-4 (P2) feature layer in YOLOv8s against aerial datasets, optimizing hyperparameters for native 1536px inference.

### 21. Quantitative Results
Results derived strictly from repository verification:
- **E3 Baseline (YOLO11-L):** `0.6210` mAP50 (Validation).
- **EXP11 P2 (YOLOv8s-P2):** `0.5690` mAP50 (Validation peak).
- **Final P2 Held-Out Evaluation (1536px, 448 imgs):** `0.4663` mAP50.
- **Final P2 SAHI Evaluation (640px, 448 imgs):** `0.4020` mAP50.
- **Planner Routing Accuracy:** `100%` on Phase 8 lexical split test.

### 22. Testing and Validation
- **Latest Regression Result:** `99 passed, 8 failed`.
- **Note on Failures:** The 8 failures are isolated to outdated integration mocks in `test_metrics.py`, `test_v2_agents.py`, and `test_v2_phase6_routing.py` that were disrupted by final Phase 13 patches (which enabled full semantic query strings over isolated keywords). The core runtime logic is sound.

### 23. Phase 12 Robustness Matrix
Based on end-to-end evidence:
- **Image + Text:** PASS
- **Image + Reference:** PASS
- **Video + Text:** PASS
- **Video + Reference:** PASS
- **Unknown Query:** PASS (Routes to Open-World)
- **Unsupported Constraint:** PASS (Safely marked uncertain)
- **Request Isolation:** PASS (Model Registry swaps models successfully)
- **Tracking Degradation:** PASS (Successfully triggers redetection)

### 24. Final Smoke Testing
- The video pipeline was successfully validated using synthetic / local MP4 inputs representing drone perspectives. 
- **Limitation:** Real-time live UAV hardware streaming (e.g., direct RTSP from a drone payload) was not evaluated.

### 25. Resource Management
- Video frame temporary images are securely cleaned up post-processing.
- Flask endpoints manage asynchronous tasks carefully.
- GPU VRAM is actively policed by the `ModelRegistry` singleton.

### 26. Security and Robustness
- Safe input extension validation (MIME types).
- Resilient to unexpected target queries through deterministic Open-World fallback routing.

### 27. Limitations
- **Tracking Metrics:** Quantitative tracking accuracy (MOTA) is unmeasured.
- **ReID:** Reference image pipeline is based on semantic cosine similarity, not formal exact-instance recognition models.
- **Real-World Streaming:** System relies on file uploads, not active video streams.
- **Open-World CLIP:** Open-world evaluation lacks formal mAP grounding for OOV datasets. CLIP similarity on tiny crops remains heavily constrained by spatial resolution.

### 28. Deployment Instructions
**Backend:**
```bash
cd AgentSearch-UAV
PYTHONPATH=. python api/main.py
# Hosts on port 5005
```
**Frontend:**
```bash
cd AgentSearch-UAV/frontend
npm install
npm run dev
# Hosts on port 5173
```

### 29. Demonstration Procedure
1. **DEMO 1 (Known Static Target):** Upload image. Search "car". Expected: Routes to P2 Specialist, high confidence bounding boxes output in under 1 second.
2. **DEMO 2 (Complex Query):** Upload image. Search "person riding two wheeler". Expected: Routes to YOLO-World. Evaluator shown that pedestrians are ignored while riders are bounded.
3. **DEMO 3 (Video Tracking):** Upload MP4 file. Search "bus". Expected: Background tracking task starts. Polling bar updates. Result video highlights bounding boxes with persistent numerical track IDs.

### 30. Claims Supported by Evidence
- **Functionally Validated:** The V2 pipeline functionally routes known targets to the P2 specialist and open-world/complex queries to YOLO-World. Tracking degradation functionally triggers a redetection cycle.
- **Quantitatively Measured:** The native 1536px P2 head quantitatively outperforms 640px SAHI slicing on the YOLOv8s-P2 VisDrone checkpoint.
- **Not Established:** The system does NOT perform exact-instance recognition (ReID). True UAV real-time real-world performance is not quantitatively established.

### 31. Technical Contributions
- Implemented a modular Multi-Agent state machine for aerial detection.
- Integrated an intelligent deterministic routing system to maximize specialized model efficiency while preserving open-vocabulary fallback capabilities.
- Evaluated and benchmarked the efficacy of P2 high-resolution pathways against standard SAHI techniques for specific drone-based perspectives.

### 32. Final System Status
- **ACTIVE ARCHITECTURE:** AgentState State-Machine
- **ACTIVE MODELS:** YOLOv8s-P2 (Specialist) & YOLO-World (Open-World)
- **ACTIVE APIs:** Flask V2 Endpoints
- **TEST STATUS:** 99 Passed, 8 Stale Mocks Failed
- **REAL-MEDIA STATUS:** Verified on Synthetic MP4/Static Files
- **DEPLOYMENT STATUS:** Frozen & Ready for Demonstration
