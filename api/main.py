from flask import Flask, request, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def home():
    return {
        "status": "success",
        "message": "AgentSearch-UAV API is running"
    }

@app.route("/detect", methods=["POST"])
def detect():

    if "image" not in request.files:
        return jsonify({
            "status": "error",
            "message": "No image uploaded"
        }), 400

    image = request.files["image"]
    query = request.form.get("query", "")

    image_path = os.path.join(UPLOAD_FOLDER, image.filename)
    image.save(image_path)

    return jsonify({
        "status": "success",
        "target": query,
        "objects_found": 1,
        "confidence": 99.0,
        "result_image": image_path,
        "detections": [
            {
                "object": query,
                "confidence": 99.0
            }
        ]
    })

if __name__ == "__main__":
    app.run(debug=True)