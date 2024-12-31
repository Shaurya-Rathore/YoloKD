from ultralytics.models.yolo import YOLO
from ultralytics.engine.model import Model
model = YOLO('yolov8m.yaml')
model.info()