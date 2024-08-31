from ultralytics.models.yolo import YOLO
from ultralytics import YOLO
from ultralytics.engine.model import Model

model = YOLO('yolov8.yaml')
model.load('yolov8.pt')

