import io
import cv2
import numpy as np
import logging
import requests
from datetime import datetime, timezone
from flask import Blueprint, request, jsonify
from ultralytics import YOLO
from PIL import Image
import pillow_heif

pillow_heif.register_heif_opener()

# ─── Blueprint ──────────────────────────────────────────────────────
waste_classification_bp = Blueprint("waste_classification", __name__)

# ─── Load YOLO Model ─────────────────────────────────────────────────
yolo = YOLO("best.onnx")  # Update with your model path


def get_public_ip():
    try:
        return requests.get("https://api.ipify.org?format=json").json().get("ip", "Unknown")
    except Exception as e:
        logging.error(f"Error fetching IP: {e}")
        return "Unknown"


@waste_classification_bp.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files or "house_id" not in request.form:
        return jsonify({"error": "Image file and house_id are required"}), 400

    house_id = request.form["house_id"]
    upload = request.files["image"]

    raw = upload.read()
    try:
        pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG")
        buf.seek(0)

        arr = np.frombuffer(buf.read(), np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("cv2 failed to decode image")
    except Exception as e:
        logging.error(f"Failed to load/convert image: {e}")
        return jsonify({"error": "Invalid image format"}), 400

    try:
        results = yolo.predict(
            source=img,
            imgsz=768,
            conf=0.25,
            verbose=False
        )[0]

        raw_detections = []
        filtered_detections = []
        for box, conf, cls in zip(
                results.boxes.xyxy.cpu().numpy(),
                results.boxes.conf.cpu().numpy(),
                results.boxes.cls.cpu().numpy()
        ):
            det = {
                "class": results.names[int(cls)],
                "confidence": float(conf),
                "box": box.astype(int).tolist()
            }
            raw_detections.append(det)
            if conf >= 0.6:
                filtered_detections.append({
                    "class": det["class"],
                    "confidence": det["confidence"]
                })

        raw_counts = {}
        filtered_counts = {}
        for det in raw_detections:
            raw_counts[det["class"]] = raw_counts.get(det["class"], 0) + 1
        for det in filtered_detections:
            filtered_counts[det["class"]] = filtered_counts.get(det["class"], 0) + 1

        response = {
            "house_id": house_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "counts": filtered_counts,
            "inference": "segregated" if len(filtered_counts.keys()) == 1 else "Non - Segregated",
            "detections": filtered_detections
        }

        return jsonify(response), 200

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({"error": "Internal server error"}), 500
