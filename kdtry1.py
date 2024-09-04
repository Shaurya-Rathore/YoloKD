import torch
from ultralytics import YOLO
from collections import OrderedDict
import wandb

# Initialize a new W&B run
wandb.login(key="833b800ff23eb3d26e6c85a8b9e1fc8bbafc9775")
# Initialize wandb
wandb.init(project="yolov8-LDConv")

# Load the custom model configuration
model = YOLO('yolov8-LDconv.yaml')

# Load the pretrained model from the provided path
# Kaggle typically mounts the dataset under '/kaggle/input/'
pretrained_model = YOLO('/kaggle/input/yolov8m-pt/yolov8m.pt')  # Adjust the path if necessary

# Extract the state_dict (the actual model weights) from the pretrained model
pretrained_state_dict = pretrained_model['model'].state_dict()

# Get the current model's state_dict
model_state_dict = model.model.state_dict()

# Create a new state_dict that will only contain weights for common layers
new_state_dict = OrderedDict()

# Match layers and load only common weights
for layer_name in model_state_dict.keys():
    if layer_name in pretrained_state_dict and model_state_dict[layer_name].shape == pretrained_state_dict[layer_name].shape:
        new_state_dict[layer_name] = pretrained_state_dict[layer_name]
    else:
        # If a layer is not common, use the initialized weights of the current model
        new_state_dict[layer_name] = model_state_dict[layer_name]

# Load the filtered state_dict into your model
model.model.load_state_dict(new_state_dict)

# Train the model with the specified configuration and sync to W&B
Result_Final_model = model.train(
    data='/kaggle/input/waid-dataset/data.yaml',
    epochs=3,
    batch=16,
    optimizer='auto',
    project='yolov8-LDConv',
    save=True,
)

# Finish the W&B run
wandb.finish()


