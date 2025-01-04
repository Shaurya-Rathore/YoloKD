from ultralytics.models.yolo import YOLO
from ultralytics.engine.model import Model
model = YOLO('yolov8s.yaml')
model.info()