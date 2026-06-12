import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense

# Patch to fix Keras 3 H5 loading bug where dense layer receives duplicate inputs
original_call = Dense.__call__
def patched_call(self, inputs, *args, **kwargs):
    if isinstance(inputs, (list, tuple)) and len(inputs) == 2:
        inputs = inputs[0]
    return original_call(self, inputs, *args, **kwargs)
Dense.__call__ = patched_call

model = load_model("backend/models/garbage_tf_model.h5")
CLASS_NAMES = ["Biodegradable","Non Biodegradable","Ewaste","Pharmaceutical and Biomedical Waste","hazardous"]

def classify_image(input_path, output_path):
    img = cv2.imread(input_path)
    h, w, _ = img.shape

    # run your model
    x = cv2.resize(img, (224,224)) / 255.0
    
    # Manual inference to bypass Keras 3 graph tracing bugs for this specific model
    out = tf.expand_dims(x, 0)
    for layer in model.layers:
        out = layer(out, training=False)
        if isinstance(out, (list, tuple)) and len(out) == 2:
            out = out[0]
            
    pred = out.numpy()[0]
    idx = np.argmax(pred)
    predicted_class = CLASS_NAMES[idx]
    confidence = float(pred[idx])

    # draw label
    label = f"{predicted_class} {confidence:.2f}"
    cv2.putText(img, label, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)

    cv2.imwrite(output_path, img)
    return predicted_class, confidence
