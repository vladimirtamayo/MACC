import cv2
import matplotlib.pyplot as plt
from numpy import source
from ultralytics import YOLO

cajas = cv2.VideoCapture('video/cajas1.mp4')
YOLOv8 = YOLO('yolov8n.pt')
v_size = (int(1920/2), int(1080/2))

while(True):
    [success, frame] = cajas.read()
    if success == True:
        frame = cv2.resize(frame, v_size, fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
        results = YOLOv8(source=frame)
        frame_inferred = results[0].plot()
        cv2.imshow('Fotogramas', frame_inferred)
        if cv2.waitKey(1) == ord('q'):
            break
    else:
        break
cajas.release()
cv2.destroyAllWindows()
