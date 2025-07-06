import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def detect_objects_and_classify(img_path):
    image = cv2.imread(img_path)

    # Count objects
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    object_count = len(contours)

    # Predict class using model
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    class_idx = np.argmax(preds[0])
    predicted_class = waste_classes[class_idx]

    return {
        "predicted_class": predicted_class,
        "object_count": object_count
    }
