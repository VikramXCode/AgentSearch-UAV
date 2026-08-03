from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

from pathlib import Path
from uuid import uuid4

from werkzeug.utils import secure_filename

from workflows.graph import run_pipeline


app = Flask(__name__)
CORS(app)


# =========================================================
# PATH CONFIGURATION
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

UPLOAD_FOLDER = PROJECT_ROOT / "api" / "uploads"
OUTPUT_FOLDER = PROJECT_ROOT / "outputs"

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}


# =========================================================
# HOME
# =========================================================

@app.route("/")
def home():

    return jsonify({
        "status": "success",
        "message": "AgentSearch-UAV API is running",
    })


# =========================================================
# DETECTION API
# =========================================================

@app.route("/detect", methods=["POST"])
def detect():

    # -----------------------------------------------------
    # Validate image
    # -----------------------------------------------------

    if "image" not in request.files:

        return jsonify({
            "status": "error",
            "message": "No image uploaded",
        }), 400

    image = request.files["image"]

    if not image.filename:

        return jsonify({
            "status": "error",
            "message": "Invalid image filename",
        }), 400

    # -----------------------------------------------------
    # Validate query
    # -----------------------------------------------------

    query = request.form.get("query", "").strip()

    if not query:

        return jsonify({
            "status": "error",
            "message": "Search query is required",
        }), 400

    # -----------------------------------------------------
    # Validate extension
    # -----------------------------------------------------

    extension = Path(image.filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:

        return jsonify({
            "status": "error",
            "message": "Unsupported image format",
        }), 400

    # -----------------------------------------------------
    # Save uploaded image
    # -----------------------------------------------------

    safe_name = secure_filename(image.filename)

    unique_name = f"{uuid4().hex}_{safe_name}"

    image_path = UPLOAD_FOLDER / unique_name

    image.save(image_path)

    try:

        # =================================================
        # RUN REAL MULTI-AGENT PIPELINE
        # =================================================

        state = run_pipeline(
            query=query,
            image_path=str(image_path),
        )

        # -------------------------------------------------
        # Determine final detections
        # -------------------------------------------------

        if state.strategy.enable_clip_verification:

            final_objects = state.verification.verified_objects
            confidence = state.verification.confidence_score

        else:

            final_objects = state.detection.objects_found

            if final_objects:

                confidence = sum(
                    float(obj["confidence"])
                    for obj in final_objects
                ) / len(final_objects)

            else:

                confidence = 0.0

        # -------------------------------------------------
        # Build API detections
        # -------------------------------------------------

        detections = []

        for obj in final_objects:

            detections.append({
                "label": obj["label"],
                "confidence": float(obj["confidence"]),
                "bbox": obj["bbox"],
            })

        # -------------------------------------------------
        # API response
        # -------------------------------------------------

        return jsonify({

            "status": "success",

            "mission_id": state.mission.mission_id,
            "mission_status": state.mission.status,

            "query": {
                "raw_query": state.query.raw_query,
                "target": state.query.target,
                "attributes": state.query.attributes,
                "quantity": state.query.quantity,
            },

            "strategy": {
                "detector": state.strategy.detector,
                "super_resolution": (
                    state.strategy.enable_super_resolution
                ),
                "sahi": state.strategy.enable_sahi,
                "clip_verification": (
                    state.strategy.enable_clip_verification
                ),
            },

            "objects_found": len(final_objects),

            "confidence": float(confidence),

            "detections": detections,

            "explanation": {
                "summary": state.explanation.summary,
                "reasoning": state.explanation.reasoning,
            },

            "result_image": "/result",

        })

    except Exception as exc:

        print(f"Pipeline error: {exc}")

        return jsonify({
            "status": "error",
            "message": str(exc),
        }), 500


# =========================================================
# RESULT IMAGE
# =========================================================

@app.route("/result")
def result_image():

    result_path = OUTPUT_FOLDER / "detection_result.jpg"

    if not result_path.exists():

        return jsonify({
            "status": "error",
            "message": "Result image not found",
        }), 404

    return send_file(
        result_path,
        mimetype="image/jpeg",
    )


# =========================================================
# RUN SERVER
# =========================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5001,
        debug=True,
    )