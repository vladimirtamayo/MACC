import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import pytesseract

img1 = cv2.imread("images/caja1.jpeg")
img_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

plt.figure()
plt.imshow(img_rgb)
plt.title("Caja 1")
plt.xticks([]), plt.yticks([])
plt.show()

YOLOv8 = YOLO('yolov8n.pt')
results = YOLOv8(img_rgb)
img_inferred = results[0].plot()


plt.figure()
plt.imshow(img_inferred)
plt.title("Caja inf")
plt.xticks([]), plt.yticks([])
plt.show()

text = pytesseract.image_to_string(img_rgb)
print("Tesseract: ", text)

i = 0
for result in results:
    i += 1
    print("Result ", i, result)
