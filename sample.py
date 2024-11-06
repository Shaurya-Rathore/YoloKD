import torch
import torch.nn as nn
from model_search import YOLOv8StudentModel 
from darts_utils import process_yolov8_output

def test_network():
    # Define hyperparameters
    C = 64  # Initial number of channels
    num_classes = 6  # Number of output classes
    layers = 14  # Number of layers in the network
    steps = 4  # Number of steps per DARTS cell
    multiplier = 4  # Multiplier for channels in DARTS cells
    stem_multiplier = 3  # Multiplier for channels in the stem layer

    # Create the network (CPU only)
    model = YOLOv8StudentModel(num_classes, C=C, layers=layers, steps=steps, multiplier=multiplier, stem_multiplier=stem_multiplier)
    model.half()
    # Create a sample input (batch_size=2, input_channels=3, height=224, width=224)
    batch_size = 2
    input_channels = 3
    input_height = 32
    input_width = 32
    x = torch.randn(batch_size, input_channels, input_height, input_width,dtype=torch.half)  # CPU tensor

    # Forward pass through the model
    output = model(x)
    print(output.size())

    # Process the output
    pred_bbox, pred_cls, pred_obj = process_yolov8_output(output, num_classes=num_classes, reg_max=4)

    # Print the shapes of the predictions
    print(f"Predicted Bounding Boxes: {pred_bbox.shape}")
    print(f"Predicted Class Probabilities: {pred_cls.shape}")
    print(f"Predicted Objectness Scores: {pred_obj.shape}")

    # Calculate loss
    loss = model._loss(output, x)
    print(f"Loss: {loss.item()}")

    # Print the genotype
    genotype = model.genotype()
    print(f"Genotype: {genotype}")

if __name__ == "__main__":
    test_network()
