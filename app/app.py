"""
app.py
======
Flask API for the Room Style Classifier + Recommender.

Endpoints:
  POST /predict   - multipart form upload, field name "image"
                     returns predicted style, confidence, and
                     recommended Berre.ca products.
  GET  /health     - simple liveness check for Docker/host monitoring.

Run locally:
    pip install flask torch torchvision pillow --break-system-packages
    python app.py

Then:
    curl -X POST -F "image=@room.jpg" http://localhost:5000/predict
"""

from flask import Flask, request, jsonify
from flask_cors import CORS

from predict import predict_style
from catalog import get_recommendations

app = Flask(__name__)
CORS(app)  # allow the browser demo page to call this API cross-origin

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}


def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "no 'image' file field in request"}), 400

    file = request.files["image"]
    if file.filename == "" or not _allowed_file(file.filename):
        return jsonify({"error": "invalid or missing image file"}), 400

    image_bytes = file.read()

    try:
        result = predict_style(image_bytes)
    except Exception as e:
        return jsonify({"error": f"prediction failed: {e}"}), 500

    recommendations = get_recommendations(result["style"])

    return jsonify({
        "predicted_style": result["style"],
        "confidence": result["confidence"],
        "all_probabilities": result["all_probabilities"],
        "recommended_products": recommendations,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
