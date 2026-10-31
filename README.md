# AgentUAV / AgentSearch-UAV (V2 Architecture)

This repository contains the final V2 architecture of the AgentUAV system—a dynamically routed, multi-agent pipeline designed for rigorous UAV object detection, verification, and tracking across complex aerial streams.

## FINAL CAPABILITIES

The final architecture supports the following robust modes:
1. **Image + Text Query**: Target detection from a natural language query in a still image.
2. **Image + Reference Image**: Cross-verifying and filtering candidates against a visual reference crop.
3. **Video + Text Query**: Temporal tracking and detection natively processed on a `.mp4`/`.mov` array via textual specification.
4. **Video + Reference Image**: Temporal tracking bounded by visual reference matching.

## FINAL DETECTION ROUTING
Model initialization is actively governed via a memory-safe Mutual Exclusion Model Registry:
- **Known/in-ontology target** → Routed mathematically to `YOLOv8s + P2 specialist` (optimized for dense, small-object VisDrone aerial views).
- **Unknown/open-world target** → Dynamically routed to the open-world `YOLO-World` inference model.

*(Note: The previous E3 / YOLO11-L experiments were highly informative and remain preserved in the `historical_artifacts/` directory for record-keeping but are deactivated from the runtime path.)*

## VIDEO PIPELINE
The `TrackingAgentV2` continuously tracks objects per frame:
- **Primary Loop**: Detection → Verification → Tracking
- **Failure Handling**: Tracking degradation → Redetection trigger → Verification → Tracking resumes seamlessly.

*(No quantitative MOTA/HOTA tracking accuracy metrics, ReID, or exact-instance recognition capabilities are mathematically claimed. System tracks objects contextually via bounding dynamics.)*

## GETTING STARTED

The system supports a fully functional backend Flask API endpoint mapped to a React frontend.

**1. Launch the Backend API (Terminal 1)**
```bash
PYTHONPATH=. python api/main.py
```

**2. Launch the Frontend UI (Terminal 2)**
```bash
cd frontend
npm install  # (First time only)
npm run dev
```

Navigate your browser to the local Vite port (usually `http://localhost:5173`). Upload local media files and invoke queries freely.
