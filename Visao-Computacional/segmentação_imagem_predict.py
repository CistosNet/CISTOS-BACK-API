import cv2
from ultralytics import YOLO
import numpy as np

segmentation_model = YOLO("modelos_yolo_2/best_segmentador.pt")

file_path = r"Visao-Computacional\base_teste\img_92_jpg.rf.5e2862d76dd7376fc4df0838c1ccf34d.jpg"


results = segmentation_model.predict(source=str(file_path), save=True, imgsz=640, show_labels=False)
