# detector.py

import os
import cv2 as cv
import numpy as np
from tensorflow.keras.models import load_model


# ─── CONFIG ───────────────────────────────────────────────────────────────────

MODEL_PATH    = os.path.join("backend", "models", "garbage_tf_model.h5")
print("[DEBUG] Does model file exist?", os.path.exists(MODEL_PATH))
UPLOADS       = os.path.join("backend", "uploads")
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
    resized = cv.resize(crop, (IMG_SIZE, IMG_SIZE))
    normalized = resized.astype("float32") / 55.0
    return np.expand_dims(normalized, axis=0)

# ─── MAIN FUNCTION ────────────────────────────────────────────────────────────
def detect_objects_and_classify(image_path: str) -> dict:
    """Detects objects in the image, classifies each, and returns results."""
    # Ensure annotated directory exists
    os.makedirs(ANNOTATED_DIR, exist_ok=True)

    # Load & prep image
    img = cv.imread(image_path)
    if img is None:
        return {"error": "Could not load image."}
    original = img.copy()
    gray     = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    blur     = cv.GaussianBlur(gray, (5, 5), 0)

    # Edge + contour detection
    edges    = cv.Canny(blur, 50, 150)
    closed   = cv.morphologyEx(edges, cv.MORPH_CLOSE, np.ones((3,3), np.uint8))
    contours, _ = cv.findContours(closed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    results = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 10000:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = h / float(w)
        if aspect_ratio < 1.5:  # bottles are tall
            continue
        filtered_contours.append(cnt)


        
        # Crop, preprocess, classify
        crop = original[y:y+h, x:x+w]
        inp  = preprocess_crop(crop)
        preds = model.predict(inp, verbose=0)[0]
        class_id   = int(np.argmax(preds))
        confidence = float(preds[class_id] * 100)

        label = CLASS_NAMES[class_id]
        results.append({
            "label":      label,
            "confidence": round(confidence, ),
            "box":        [int(x), int(y), int(w), int(h)]
        })

        # Draw on annotation
        cv.rectangle(original, (x, y), (x+w, y+h), (0,55,0), )
        cv.putText(
            original, f"{label} {confidence:.1f}%",
            (x, y-10), cv.FONT_HERSHEY_SIMPLEX,
            0.6, (0,0,55), 
        )

    # Save annotated image
    fname  = os.path.basename(image_path)
    outpth = os.path.join(ANNOTATED_DIR, f"annotated_{fname}")
    cv.imwrite(outpth, original)
    cv.imshow("Annotated Image", original)

    return {
        "total_objects": len(results),
        "results":       results,
        "annotated_image": outpth.replace("\\", "/")
    }
