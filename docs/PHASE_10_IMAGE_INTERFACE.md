# Phase 10: Image Interface (AgentUAV V2)

This document describes the final V2 image interface which allows a user to interact with the frozen backend models (YOLOv8s+P2 and YOLO-World).

## 1. Overview
The final architecture avoids all V1 logic, and handles routing to `YOLOv8s+P2` for known objects and `YOLO-World` for unknown targets.
SAHI and Super Resolution are disabled by default. The system accepts an image and either a text query or a reference image for semantic similarity scoring.

## 2. API Endpoint
**POST** `/v2/detect`

**Description**: Main inference endpoint for the AgentUAV V2 Image Pipeline.

**Request Format (multipart/form-data):**
- `image` (File): Required. The aerial scene image (.png, .jpg, .jpeg, .webp).
- `query` (Text): The search target (e.g., "car", "pedestrian"). Required if `reference_image` is absent.
- `reference_image` (File): Optional. A template image representing the target object to search for based on visual similarity.

**Response Format (JSON):**
```json
{
  "status": "SUCCESS",
  "media_type": "IMAGE",
  "query": "car",
  "detections": [
    {
      "bbox": [100.5, 200.2, 125.3, 220.8],
      "confidence": 0.89,
      "class_label": "car",
      "source": "SPECIALIST_P2",
      "verification_status": "SATISFIED",
      "label": "car",
      "class": "car"
    }
  ],
  "annotated_image_url": "/v2/outputs/v2_out_8a4b2c1d.jpg",
  "pipeline": [
    {
      "name": "V2 Deterministic Pipeline",
      "status": "completed",
      "details": "Target 'car' is known. Routing to specialist."
    }
  ],
  "tools": [
    {"name": "Specialist (P2)", "status": "used"},
    {"name": "Open World", "status": "skipped"}
  ],
  "reasoning": "Target 'car' is known. Routing to specialist.",
  "total_time": 0.354
}
```

## 3. Model Routing
- **Known Query (e.g., "car", "pedestrian")**: Routes automatically to the `YOLOv8s+P2` Specialist detector.
- **Unknown Query (e.g., "dragon")**: Routes automatically to the `YOLO-World` Open-World detector.
- **Reference Image**: Routes to Open-World detector for broad proposal generation, then to the Verification Agent which computes `reference-image similarity` (not ReID/exact-instance).

## 4. Verification Behavior
Candidates are forwarded to the Verification Agent. For reference images, candidates are scored using CLIP embeddings against the reference image. Results are reported via `verification_status` in each detection payload (`SATISFIED`, `VIOLATED`, `UNCERTAIN`).

## 5. Error Behavior
- Missing image/query: Returns `HTTP 400 Bad Request` with structured JSON error (e.g., `{"status": "FAILED", "error": "No image uploaded"}`).
- Unsupported format: Returns `HTTP 400 Bad Request` with specific format error.
- Internal/Model Error: Returns `HTTP 500 Internal Server Error` with stringified exception message, ensuring clean failure recovery.

## 6. How to Launch
To start the backend with the new `/v2/detect` endpoint and the React Frontend:

**Terminal 1 (Backend):**
```bash
PYTHONPATH=. python api/main.py
```
*Note: This starts the Flask server on port 5005. Models will load lazily via the ModelRegistry when first invoked to save VRAM.*

**Terminal 2 (Frontend):**
```bash
cd frontend
npm install
npm run dev
```
*Note: Open `http://localhost:5173` in your browser. The frontend is fully hooked up to the `/v2/detect` route for standard Image queries.*

## 7. Manual Acceptance Tests

### TEST 1 (Known Target)
- **Action**: Upload a scene image and query `"car"`.
- **Expected Result**: Detections returned rapidly. UI shows "Specialist (P2)" tool as "used". Annotated image shows bounding boxes.

### TEST 2 (Unknown Target)
- **Action**: Upload an image and query `"boat"`.
- **Expected Result**: Detections returned (if present). UI shows "Open World" tool as "used". The reasoning states the target is unknown and routed to fallback.

### TEST 3 (Reference Image)
- **Action**: Upload a scene image, and in "Reference Mode", upload a cropped reference image.
- **Expected Result**: Candidate bounding boxes returned. Verification status displays as `UNCERTAIN` or `SATISFIED` based on similarity scoring.
