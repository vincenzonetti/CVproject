import cv2
from ultralytics import YOLO

model = YOLO("yolo11s.pt") 
data_path = 'data.yaml'
results = model.train(data=data_path, epochs=20, imgsz=640)