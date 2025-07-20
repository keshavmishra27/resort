import cv2

def detect_objects_and_save(input_path, output_path):
    img = cv2.imread(input_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # simple threshold+contours
    _, th = cv2.threshold(gray, 0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    cnts,_ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    count = 0
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        if w*h < 500: continue
        cv2.rectangle(img, (x,y),(x+w,y+h),(255,0,0),2)
        count +=1

    cv2.imwrite(output_path, img)
    return count
