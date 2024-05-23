import cv2
import imutils
from ultralytics import YOLO

img_path = r"image_path"
model_path = r"model_Path"

font = cv2.FONT_HERSHEY_SIMPLEX

model = YOLO(model_path)
img = cv2.imread(img_path)
img = imutils.resize(img, width=640) # aspect ratio

results = model(img)[0]

threshold = 0.5
for result in results.boxes.data.tolist():
    x1, x2, y1, y2, score, class_id = result
    x1, x2, y1, y2, class_id = int(x1), int(x2), int(y1), int(y2), int(class_id)
    if score > threshold:
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)

        class_name = results.names[class_id]
        score = score * 100

        text = f"{class_name}: %{score:.2f}"

        cv2.putText(img, text, (x1, y1-10), font, 0.5, (0,255,0), 1, cv2.LINE_AA)

cv2.imshow("sonuc",img)
cv2.waitKey(0)