from ultralytics.models.yolo import YOLO
from ultralytics import YOLO
from ultralytics import NAS
from ultralytics.engine.model import Model
import torch
import wandb
# Check if CUDA is available and set the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize a new W&B run
wandb.login(key="833b800ff23eb3d26e6c85a8b9e1fc8bbafc9775") 
wandb.init(project="yolov8nas")
# Load the custom model configuration
model = NAS("/kaggle/input/yolonas-s-weight/yolo_nas_s.pt")
model.model.to(device)
results = model.val(data="/kaggle/input/waiddataset/WAID-main/WAID-main/WAID/data.yaml")
# Finish the W&B run
wandb.finish()
