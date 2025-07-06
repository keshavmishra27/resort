from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model

from detector import detect_objects_and_classify  # update this import path if needed

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = os.path.join("uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

model = load_model("models/garbage_tf_model.h5")  # adjust path
waste_classes = ["battery", "biological", "cardboard", "clothes", "glass", "metal", "paper", "plastic", "shoes", "trash"]

@app.route("/analyze", methods=["POST","GET"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
    file.save(path)

    result = detect_objects_and_classify(path)  # return class + count
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
