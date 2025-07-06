# detector.py

import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

#resort\backend\models\garbage_tf_model.h5
# ─── CONFIG ───────────────────────────────────────────────────────────────────
MODEL_PATH    = os.path.join("resort","backend", "models", "garbage_tf_model.h5")
UPLOADS       = os.path.join("resort","backend", "uploads")
ANNOTATED_DIR = os.path.join(UPLOADS, "annotated")
IMG_SIZE      = 224

CLASS_NAMES = [
    "battery", "biological", "cardboard", "clothes",
    "glass", "metal", "paper", "plastic", "shoes", "trash"
]

# ─── MODEL LOADING ────────────────────────────────────────────────────────────
model = load_model(MODEL_PATH)

# ─── HELPERS ─────────────────────────────────────────────────────────────────
def preprocess_crop(crop: np.ndarray) -> np.ndarray:
    """Resize, normalize, and add batch dim for model.predict."""
    resized = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
    normalized = resized.astype("float32") / 255.0
    return np.expand_dims(normalized, axis=0)

# ─── MAIN FUNCTION ────────────────────────────────────────────────────────────
def detect_objects_and_classify(image_path: str) -> dict:
    """Detects objects in the image, classifies each, and returns results."""
    # Ensure annotated directory exists
    os.makedirs(ANNOTATED_DIR, exist_ok=True)

    # Load & prep image
    img = cv2.imread(image_path)
    if img is None:
        return {"error": "Could not load image."}
    original = img.copy()
    gray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur     = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge + contour detection
    edges    = cv2.Canny(blur, 50, 150)
    closed   = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    results = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w*h < 100:  # skip tiny noise
            continue

        # Crop, preprocess, classify
        crop = original[y:y+h, x:x+w]
        inp  = preprocess_crop(crop)
        preds = model.predict(inp, verbose=0)[0]
        class_id   = int(np.argmax(preds))
        confidence = float(preds[class_id] * 100)

        label = CLASS_NAMES[class_id]
        results.append({
            "label":      label,
            "confidence": round(confidence, 2),
            "box":        [int(x), int(y), int(w), int(h)]
        })

        # Draw on annotation
        cv2.rectangle(original, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(
            original, f"{label} {confidence:.1f}%",
            (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (0,0,255), 2
        )

    # Save annotated image
    fname  = os.path.basename(image_path)
    outpth = os.path.join(ANNOTATED_DIR, f"annotated_{fname}")
    cv2.imwrite(outpth, original)

    return {
        "total_objects": len(results),
        "results":       results,
        "annotated_image": outpth.replace("\\", "/")
    }
