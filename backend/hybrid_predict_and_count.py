import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load your trained classification model
model = load_model(r"resort\backend\models\garbage_tf_model.h5")

# Class names in the same order as used during training
CLASS_NAMES = ["battery", "biological", "cardboard", "clothes", 
               "glass", "metal", "paper", "plastic", "shoes", "trash"]

# Load image
image_path = r"datset\test\battery\battery8.jpg"  # 🔁 Replace with actual path
img = cv2.imread(image_path)
original = img.copy()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Preprocessing: blur + Canny edge detection
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blurred, 50, 150)

# Morphological closing to join broken edges
kernel = np.ones((3, 3), np.uint8)
closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

# Find contours
contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter small objects
min_area = 100
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

print(f"Detected objects: {len(filtered_contours)}")

for i, cnt in enumerate(filtered_contours):
    x, y, w, h = cv2.boundingRect(cnt)
    
    # Crop and preprocess each object
    cropped = original[y:y+h, x:x+w]
    resized = cv2.resize(cropped, (224, 224))  # or (64, 64) depending on model input
    input_img = resized.astype('float32') / 255.0
    input_img = np.expand_dims(input_img, axis=0)  # (1, H, W, 3)

    # Predict class
    preds = model.predict(input_img, verbose=0)
    class_id = np.argmax(preds)
    label = CLASS_NAMES[class_id]

    # Draw bounding box and label
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

# Show result
cv2.imshow("Object Detection + Classification", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
