import cv2
import numpy as np

# 1) Define a global score variable
score = 0

def universal_object_counter(image_path,
                             min_area=500,
                             show_result=True,
                             output_path=None):
    """
    Count objects in a single image and add that count to the global `score`.
    Returns: (count, updated_score)
    """
    global score

    # ——————————————————————————
    # your existing segmentation + watershed pipeline
    img = cv2.imread(image_path)
    if img is None:
        print("🔴 Image not found or path is incorrect.")
        return 0, score

    original = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, binary = cv2.threshold(blur, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k = np.ones((3,3), np.uint8)
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
    original[markers == -1] = [0,0,255]

    # ——————————————————————————
    # count & draw
    count = 0
    for m in range(2, markers.max() + 1):
        mask = (markers == m).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(mask,
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) < min_area:
            continue
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(original, (x,y), (x+w, y+h), (0,255,0), 2)
        count += 1

    # ——————————————————————————
    # update global score
    score += count

    print(f"Objects in this image: {count}")
    print(f"Total score so far: {score}")

    if show_result:
        cv2.imshow("Segments & Count", original)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if output_path:
        cv2.imwrite(output_path, original)
        print(f"Annotated image saved to: {output_path}")

    return count, score


if __name__ == "__main__":
    # simulate multiple uploads:
    for img_path in ["img1.jpg", "img2.jpg", "img3.jpg"]:
        cnt, total = universal_object_counter(img_path,
                                              min_area=1000,
                                              output_path=None)
        # you can now push `total` to your e‑learboard
