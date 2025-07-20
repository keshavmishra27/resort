import cv2
import numpy as np
from backend.class_pred import classify_image  # Your classifier

CATEGORY_SCORES = {
    "Biodegradable": 10,
    "Ewaste": 20,
    "hazardous": 30,
    "Non Biodegradable": 40,
    "Pharmaceutical and Biomedical Waste": 50
}

def universal_object_classifier(image_path, min_area=500, show_result=True, output_path=None):
    img = cv2.imread(image_path)
    if img is None:
        print("Image not found or path is incorrect.")
        return {}, 0

    original = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    k = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k, iterations=2)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, k, iterations=2)

    sure_bg = cv2.dilate(closing, k, iterations=3)
    dist = cv2.distanceTransform(closing, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.4 * dist.max(), 255, cv2.THRESH_BINARY)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(img, markers)
    original[markers == -1] = [0, 0, 255]

    label_counts = {}
    total_score = 0

    for m in range(2, markers.max() + 1):
        mask = (markers == m).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) < min_area:
            continue

        x, y, w, h = cv2.boundingRect(c)
        cropped_object = img[y:y + h, x:x + w]
        temp_path = "temp_crop.jpg"
        cv2.imwrite(temp_path, cropped_object)

        predicted_class, confidence = classify_image(temp_path)
        label_counts[predicted_class] = label_counts.get(predicted_class, 0) + 1
        total_score += CATEGORY_SCORES.get(predicted_class, 0)

        cv2.rectangle(original, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = f"{predicted_class} ({round(confidence * 100)}%)"
        cv2.putText(original, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    total_count = sum(label_counts.values())  # ✅ Moved here

    if show_result:
        cv2.imshow("Object Classification", original)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if output_path:
        cv2.imwrite(output_path, original)
        print(f"Saved annotated result to: {output_path}")

    return total_count, total_score
