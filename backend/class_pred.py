import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("backend/models/garbage_tf_model.h5")
CLASS_NAMES = ["Biodegradable","Non Biodegradable","Ewaste","Pharmaceutical and Biomedical Waste","hazardous"]

def classify_image(input_path, output_path):
    img = cv2.imread(input_path)
    h, w, _ = img.shape

    # run your model
    x = cv2.resize(img, (224,224)) / 255.0
    pred = model.predict(np.expand_dims(x,0))[0]
    idx = np.argmax(pred)
    predicted_class = CLASS_NAMES[idx]
    confidence = float(pred[idx])

    # draw label
    label = f"{predicted_class} {confidence:.2f}"
    cv2.putText(img, label, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)

    cv2.imwrite(output_path, img)
    return predicted_class, confidence
