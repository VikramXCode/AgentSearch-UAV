import os
import cv2
import uuid
import time
from pathlib import Path
from flask import Blueprint, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

from v2.agents.query_agent import QueryAgentV2
from v2.agents.planning_agent import PlanningAgentV2
from v2.agents.detection_agent import DetectionAgentV2
from v2.agents.verification_agent import VerificationAgentV2
from v2.agents.tracking_agent import TrackingAgentV2
from v2.models.model_registry import ModelRegistry
from v2.models.semantic_adapter import CLIPEngineAdapter
from v2.schemas.state import AgentStateV2, MediaType, ActionType
import threading

v2_blueprint = Blueprint("v2", __name__, url_prefix="/v2")

# Initialize registry and agents globally to preserve models in memory
v2_registry = ModelRegistry()
query_agent = QueryAgentV2()
planning_agent = PlanningAgentV2()
detection_agent = DetectionAgentV2(v2_registry)
verification_agent = VerificationAgentV2(semantic_adapter=CLIPEngineAdapter())

V2_VIDEO_JOBS = {}
V2_VIDEO_JOBS_LOCK = threading.Lock()

UPLOAD_FOLDER = Path(__file__).resolve().parent / "uploads"
OUTPUT_FOLDER = Path(__file__).resolve().parent.parent / "outputs"

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

def _annotate_image(image_path: str, candidates: list) -> str:
    img = cv2.imread(image_path)
    if img is None:
        return ""
    for c in candidates:
        x1, y1, x2, y2 = map(int, c.bbox)
        # Bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Label
        text = f"{c.class_label} {c.confidence:.2f}"
        cv2.putText(img, text, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    out_name = f"v2_out_{uuid.uuid4().hex[:8]}.jpg"
    out_path = OUTPUT_FOLDER / out_name
    cv2.imwrite(str(out_path), img)
    return out_name

@v2_blueprint.route("/outputs/<filename>")
def serve_output(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

@v2_blueprint.route("/detect", methods=["POST"])
def detect():
    # Validation
    if "image" not in request.files:
        return jsonify({"status": "FAILED", "error": "No image uploaded"}), 400
    
    image_file = request.files["image"]
    query_text = request.form.get("query", "").strip()
    
    has_ref = "reference_image" in request.files and request.files["reference_image"].filename
    if not query_text and not has_ref:
        return jsonify({"status": "FAILED", "error": "Search query or reference image is required"}), 400

    ext = Path(image_file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"status": "FAILED", "error": f"Unsupported format: {ext}"}), 400

    # Save inputs
    img_id = uuid.uuid4().hex[:8]
    image_path = str(UPLOAD_FOLDER / f"{img_id}_scene{ext}")
    image_file.save(image_path)

    ref_path = None
    if has_ref:
        ref_file = request.files["reference_image"]
        r_ext = Path(ref_file.filename).suffix.lower()
        ref_path = str(UPLOAD_FOLDER / f"{img_id}_ref{r_ext}")
        ref_file.save(ref_path)

    try:
        t0 = time.perf_counter()
        
        # 1. Query
        state = AgentStateV2()
        state.media_metadata.type = MediaType.IMAGE
        state.media_metadata.path = image_path
        state.query_spec = query_agent.run(raw_query=query_text, reference_image_path=ref_path)
        
        # We loop through planner -> detect -> verify
        max_steps = 10
        step = 0
        while step < max_steps:
            action = planning_agent.run(state)
            if action == ActionType.TERMINATE:
                break
                
            state.plan.current_action = action
            state.plan.history.append(action)
                
            if action in [ActionType.DETECT_SPECIALIST, ActionType.DETECT_OPEN_WORLD]:
                state = detection_agent.run(state, image=state.media_metadata.path)
            
            elif action == ActionType.VERIFY_CANDIDATES:
                from PIL import Image
                img_pil = Image.open(state.media_metadata.path).convert("RGB")
                state.verification_results = verification_agent.run(state.query_spec, state.candidates, state.media_metadata, image=img_pil)
                
            elif action == ActionType.REDETECT:
                # Reset candidates for redetect
                state.candidates = []
                
            else:
                break
                
            step += 1

        elapsed = time.perf_counter() - t0

        # Annotate image
        out_name = _annotate_image(image_path, state.candidates)

        # Build response
        detections = []
        for c in state.candidates:
            # Check verification status for this candidate
            v_status = "UNCERTAIN"
            for v in state.verification_results:
                if v.candidate_id == c.id:
                    v_status = v.status.value
                    break
                    
            detections.append({
                "bbox": c.bbox,
                "confidence": c.confidence,
                "class_label": c.class_label,
                "source": c.source,
                "verification_status": v_status,
                # Frontend expects "label" and "class" for rendering
                "label": c.class_label,
                "class": c.class_label
            })
            
        # Simplified pipeline for frontend compatibility
        pipeline = [
            {"name": "V2 Deterministic Pipeline", "status": "completed", "details": state.plan.decision_rationale}
        ]
        
        tools = [
            {"name": "Specialist (P2)", "status": "used" if "DETECT_SPECIALIST" in state.plan.history else "skipped"},
            {"name": "Open World", "status": "used" if "DETECT_OPEN_WORLD" in state.plan.history else "skipped"}
        ]

        return jsonify({
            "status": "SUCCESS",
            "media_type": state.media_metadata.type.value,
            "query": state.query_spec.raw_query,
            "detections": detections,
            "pipeline": pipeline,
            "tools": tools,
            "annotated_image_url": f"/v2/outputs/{out_name}" if out_name else None,
            "total_time": elapsed,
            "reasoning": state.plan.decision_rationale
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "FAILED", "error": str(e)}), 500

def _process_video_job_worker(job_id: str, video_path: str, query_text: str, ref_path: str):
    try:
        t0 = time.perf_counter()
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Failed to open video")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0: fps = 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        out_name = f"v2_video_out_{uuid.uuid4().hex[:8]}.mp4"
        out_path = str(OUTPUT_FOLDER / out_name)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

        state = AgentStateV2()
        state.media_metadata.type = MediaType.VIDEO
        state.media_metadata.path = video_path
        state.query_spec = query_agent.run(raw_query=query_text, reference_image_path=ref_path)
        
        tracker = TrackingAgentV2()
        frame_idx = 0
        redetection_events = []
        temp_frame_path = str(UPLOAD_FOLDER / f"temp_{job_id}.jpg")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_idx += 1
            
            if frame_idx % 10 == 0:
                with V2_VIDEO_JOBS_LOCK:
                    if job_id in V2_VIDEO_JOBS:
                        V2_VIDEO_JOBS[job_id]["progress"] = (frame_idx / max(1, total_frames)) * 100
                        V2_VIDEO_JOBS[job_id]["current_frame"] = frame_idx
                        V2_VIDEO_JOBS[job_id]["total_frames"] = total_frames
                        V2_VIDEO_JOBS[job_id]["fps"] = fps
            
            # Reset history if we were just tracking, to let planner decide based on degradation
            if state.plan.history and state.plan.history[-1] == ActionType.TRACK:
                state.plan.history = [ActionType.TRACK]
                
            action = planning_agent.run(state)
            
            if action == ActionType.REDETECT or not state.plan.history:
                if action == ActionType.REDETECT:
                    state.plan.current_action = action
                    state.plan.history.append(action)
                    redetection_events.append({"frame": frame_idx, "reason": state.plan.decision_rationale})
                    # Re-run planner to get DETECT action
                    action = planning_agent.run(state)
                
                # We expect action to be DETECT_SPECIALIST or DETECT_OPEN_WORLD now
                cv2.imwrite(temp_frame_path, frame)
                state.media_metadata.path = temp_frame_path
                state.plan.current_action = action
                state.plan.history.append(action)
                
                state = detection_agent.run(state, image=temp_frame_path)
                
                next_action = planning_agent.run(state)
                state.plan.current_action = next_action
                state.plan.history.append(next_action) # VERIFY
                from PIL import Image
                img_pil = Image.open(temp_frame_path).convert("RGB")
                state.verification_results = verification_agent.run(state.query_spec, state.candidates, state.media_metadata, image=img_pil)
                
                track_action = planning_agent.run(state)
                state.plan.current_action = track_action
                state.plan.history.append(track_action) # TRACK
                
                verified_cands = [c for c in state.candidates if any(v.candidate_id == c.id and v.status.value in ["SATISFIED", "UNCERTAIN"] for v in state.verification_results)]
                state = tracker.run(state, new_candidates=verified_cands)
                
            else:
                # TERMINATE from planner means tracking is stable
                state = tracker.run(state, new_candidates=None)
                
            # Annotate
            for c in state.candidates:
                x1, y1, x2, y2 = map(int, c.bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                text = f"{c.class_label} [ID:{getattr(c, 'track_id', '?')}] {c.confidence:.2f}"
                cv2.putText(frame, text, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
            out.write(frame)
            
        cap.release()
        out.release()
        
        if os.path.exists(temp_frame_path):
            os.remove(temp_frame_path)

        elapsed = time.perf_counter() - t0
        
        result_payload = {
            "status": "SUCCESS",
            "media_type": "VIDEO",
            "query": query_text,
            "total_frames": total_frames,
            "fps": fps,
            "duration": total_frames / max(1, fps),
            "detections": [c.__dict__ for c in state.candidates], # last frame detections
            "tracks": [],
            "frame_results": [],
            "redetection_events": redetection_events,
            "annotated_video_url": f"/v2/outputs/{out_name}",
            "pipeline": [{"name": "V2 Video Pipeline", "status": "completed"}],
            "tools": [{"name": "Tracker", "status": "used"}],
            "total_time": elapsed,
            "reasoning": "Video processing completed."
        }
        
        with V2_VIDEO_JOBS_LOCK:
            if job_id in V2_VIDEO_JOBS:
                V2_VIDEO_JOBS[job_id].update({
                    "status": "completed",
                    "progress": 100.0,
                    "result": result_payload
                })

    except Exception as e:
        import traceback
        traceback.print_exc()
        with V2_VIDEO_JOBS_LOCK:
            if job_id in V2_VIDEO_JOBS:
                V2_VIDEO_JOBS[job_id]["status"] = "failed"
                V2_VIDEO_JOBS[job_id]["error"] = str(e)


@v2_blueprint.route("/detect-video", methods=["POST"])
def detect_video():
    if "video" not in request.files and not request.form.get("video_name"):
        return jsonify({"status": "FAILED", "error": "No video uploaded"}), 400
    
    query_text = request.form.get("query", "").strip()
    has_ref = "reference_image" in request.files and request.files["reference_image"].filename
    if not query_text and not has_ref:
        return jsonify({"status": "FAILED", "error": "Search query or reference image is required"}), 400

    ALLOWED_VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    video_path = None
    
    job_id = f"vjob_{uuid.uuid4().hex[:12]}"
    
    if "video" in request.files and request.files["video"].filename:
        video_file = request.files["video"]
        ext = Path(video_file.filename).suffix.lower()
        if ext not in ALLOWED_VIDEO_EXT:
            return jsonify({"status": "FAILED", "error": f"Unsupported format: {ext}"}), 400
        video_path = str(UPLOAD_FOLDER / f"{job_id}_scene{ext}")
        video_file.save(video_path)

    ref_path = None
    if has_ref:
        ref_file = request.files["reference_image"]
        r_ext = Path(ref_file.filename).suffix.lower()
        ref_path = str(UPLOAD_FOLDER / f"{job_id}_ref{r_ext}")
        ref_file.save(ref_path)

    with V2_VIDEO_JOBS_LOCK:
        V2_VIDEO_JOBS[job_id] = {
            "job_id": job_id,
            "status": "processing",
            "progress": 0.0,
            "current_frame": 0,
            "total_frames": 0,
            "fps": 0.0,
            "stage": "Initializing...",
            "result": None,
            "error": None,
        }

    thread = threading.Thread(
        target=_process_video_job_worker,
        args=(job_id, video_path, query_text, ref_path),
        daemon=True
    )
    thread.start()

    return jsonify({
        "status": "processing",
        "job_id": job_id
    })

@v2_blueprint.route("/video-progress/<job_id>")
def video_progress(job_id):
    with V2_VIDEO_JOBS_LOCK:
        if job_id not in V2_VIDEO_JOBS:
            return jsonify({"status": "failed", "error": "Job not found"}), 404
        return jsonify(V2_VIDEO_JOBS[job_id])
