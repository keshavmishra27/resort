import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

CLASS_LABELS = [
    "battery", "biological", "cardboard", "clothes", "glass", 
    "metal", "paper", "plastic", "shoes", "trash"
]

def main():
    uploads_dir = os.getenv('UPLOADS_DIR', r'backend\static\uploads')
    model_path = os.getenv('MODEL_PATH', r'backend\models\garbage_tf_model.h5')

    if not os.path.exists(uploads_dir):
        os.makedirs(uploads_dir)
        print(f"Created uploads folder at: {uploads_dir}")
        print("Please add some images to this folder and run the script again.")
        return

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    image_files = [f for f in os.listdir(uploads_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print(f"No images found in {uploads_dir}. Please add some images to test.")
        return

    print(f"Loading model from {model_path}...\n")
    model = load_model(model_path)

    print(f"--- Predicting Uploaded Images in {uploads_dir} ---")
    
    for filename in image_files:
        img_path = os.path.join(uploads_dir, filename)
        
        # Load and preprocess the image
        img = load_img(img_path, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = img_array / 255.0  
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        predictions = model.predict(img_array, verbose=0)
        predicted_class_idx = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions)
        
        predicted_label = CLASS_LABELS[predicted_class_idx]
        
        print(f"Image: {filename}")
        print(f"  -> Predicted Class: {predicted_label} (Confidence: {confidence*100:.2f}%)\n")

if __name__ == '__main__':
    main()
