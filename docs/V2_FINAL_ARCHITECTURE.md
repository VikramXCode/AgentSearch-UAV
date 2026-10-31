# AgentUAV V2 Final Architecture

This document finalizes the active architecture for the AgentUAV V2 end-to-end pipeline. 

## 1. High-Level Architecture Diagram
The V2 system implements a deterministic, state-machine driven multi-agent pipeline explicitly designed to intelligently route between an ultra-high resolution aerial specialist and a fallback open-world detector.

```text
    TEXT QUERY or REFERENCE IMAGE
                 |
                 v
            Query Agent
                 |
                 v
           Planning Agent
                 |
          +------+------+
          |             |
          v             v
   YOLOv8s + P2     YOLO-World
   specialist       open-world
          |             |
          +------+------+
                 |
                 v
        Verification Agent
                 |
          +------+------+
          |             |
       IMAGE          VIDEO
          |             |
       result       Tracking Agent
                        |
                 degradation?
                   /       \
                 no         yes
                 |           |
              continue     REDETECT
                              |
                         verification
                              |
                           tracking
                              |
                           result
```

## 2. Active Model Roles

### A. Final Specialist Detector
- **Model:** YOLOv8s + P2 Head (1536x1536)
- **Role:** Primary inference for all in-ontology known targets (`car`, `person`, `pedestrian`, `bus`, `truck`, `motorcycle`, `bicycle`, etc).
- **Checkpoint:** `runs/detect/runs/detect/experiments/model_search/EXP11_yolov8s_p2_1536/weights/best.pt`
- **SAHI Status:** **Disabled by default**. The P2 head natively captures high-resolution feature maps natively, defeating the need for sliding window crops. (SAHI remains available in utility scripts but is entirely bypassed in standard production).
- **Historical Note:** The `E3_yolo11l_1536_aug` model has been permanently retired from the active V2 routing pipeline.

### B. Final Open-World Detector
- **Model:** YOLO-World (YOLOv8s-world)
- **Role:** Fallback inference for arbitrary / novel targets via text-prompt encoding.
- **Checkpoint:** `weights/yolov8s-world.pt`
- **Integration:** Lazy-loaded dynamically to conserve VRAM via mutual exclusion rules (unloads P2 when instantiated).

## 3. Core Agent Contracts

### Query Agent (`v2/agents/query_agent.py`)
- Accepts arbitrary text queries and/or a reference image path.
- Extracts `target` (e.g. "car", "building"), spatial/attribute `constraints` (e.g. "red", "near"), and `reference_image_path`.
- Emits standard `QuerySpec`.

### Planning Agent (`v2/agents/planning_agent.py`)
- Deterministic router mapping known query targets to `DETECT_SPECIALIST` and unknown targets to `DETECT_OPEN_WORLD`.
- Directs execution linearly to `VERIFY_CANDIDATES`.
- Determines branching for `TRACK` (if media is Video).
- Does **not** enable Super-Resolution (SR) natively.

### Verification Agent (`v2/agents/verification_agent.py`)
- Evaluates candidate crop features against query constraints.
- Emits explicit deterministic statuses: `SATISFIED`, `VIOLATED`, `UNCERTAIN`, `UNSUPPORTED`.
- Evaluates Reference Image similarity if specified (via Semantic Embedding Adapter). Does not implicitly assume exact ReID without semantic constraint matches.

### Tracking Agent (`v2/agents/tracking_agent.py`)
- Standard tracking interface maintaining state across video frames.
- Analyzes candidate confidence degradation to actively emit a `REDETECT` request back to the planner, triggering re-evaluation of the current frame via the detection agents.

## 4. Final API Output Contract
The final production API guarantees the following structured representation format:

**IMAGE Output:**
```json
{
  "status": "SUCCESS",
  "media_type": "IMAGE",
  "query": "red car",
  "detections": [
    {
      "bbox": [100.5, 200.0, 150.5, 250.0],
      "confidence": 0.89,
      "class_label": "car",
      "verification_status": "SATISFIED",
      "source": "SPECIALIST_P2"
    }
  ]
}
```

**VIDEO Output:**
```json
{
  "status": "SUCCESS",
  "media_type": "VIDEO",
  "query": "following the truck",
  "tracks": [
    {
      "track_id": 1,
      "class_label": "truck",
      "frames_present": [0, 1, 2, 4],
      "status": "ACTIVE"
    }
  ],
  "frame_results": [
    {
      "frame_index": 0,
      "detections": [...]
    }
  ]
}
```

## 5. Performance and Error Handling
- **Lazy Loading**: Models are resident in VRAM **only** when actively requested. Mutual exclusion ensures the 1536 P2 model and the YOLO-World model do not overlap.
- **Failures**: Errors gracefully degrade to a `MissionStatus.FAILED` struct populated with actionable rejection strings rather than raw stack traces.
