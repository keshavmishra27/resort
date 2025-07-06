from flask import Flask, request, jsonify,render_template
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename

from detector import detect_objects_and_classify

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = os.path.join("resort","backend", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    filename = secure_filename(file.filename)
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(path)

    result = detect_objects_and_classify(path)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
