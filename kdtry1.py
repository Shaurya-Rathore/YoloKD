import torch
from ultralytics import YOLO
from collections import OrderedDict

# Load the custom model configuration
model = YOLO('yolov8-LDconv.yaml')
# Load the pretrained model
pretrained_model = torch.load('yolov8m.pt')  

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

# Now the model is loaded with weights only for the layers that match the custom YAML configuration

Result_Final_model = model.train(data= 'C:\\Users\\aditya\\Downloads\\WAID-main\\WAID-main\\WAID\\data.yaml' ,epochs = 100, batch = 16, optimizer = 'auto')

