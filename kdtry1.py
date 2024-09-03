from ultralytics.models.yolo import YOLO
from ultralytics import YOLO
from ultralytics.engine.model import Model

model = YOLO('yolov8m.pt')

Result_Final_model = model.train(data= 'C:\\Users\\aditya\\Downloads\\WAID-main\\WAID-main\\WAID\\data.yaml' ,epochs = 1, batch = 8, optimizer = 'auto')

