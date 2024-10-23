import torch
import torch.nn as nn
from model_search import YOLOv8StudentModel 
from darts_utils import process_yolov8_output

def test_network():
    # Define hyperparameters
    C = 16  # Initial number of channels
    num_classes = 6  # Number of output classes
    layers = 8  # Number of layers in the network
    #criterion = YOLOLoss  # Loss function

    # Create the network (CPU only)
    model = YOLOv8StudentModel(num_classes, C=64, layers=14, steps=4, multiplier=4, stem_multiplier=3)

    # Create a sample input (batch_size=2, input_channels=3, height=32, width=32)
    batch_size = 2
    input_channels = 3
    input_height = 32
    input_width = 32
    x = torch.randn(batch_size, input_channels, input_height, input_width)  # CPU tensor

    # Create sample labels
    labels = torch.randint(0, num_classes, (batch_size,))  # CPU tensor

    bbox_preds = model(x) 
         # Unpack the tuple returned by forward()
    #print(f"Output shape: {logits.shape}"
    
    #for tensor in bbox_preds:
        #print(tensor.shape)
        #dbox,cls = process_yolov8_output(tensor)
    shape = bbox_preds[0].shape 
    bbox_preds = torch.cat([xi.view(shape[0], num_classes + 16, -1) for xi in bbox_preds], 2)
    dbox = bbox_preds[:, : 16]
    cls = bbox_preds[:, 16 :]
    #print("bbox",bbox_preds) 
    print("dbox",dbox.size()) 
    print("cls",cls.size()) 
    #print("obj",obj.size())
    input = (dbox,cls)

    # Calculate loss
    loss = model._loss(cls, labels)
    print(f"Loss: {loss.item()}")

    # Print the genotype
    genotype = model.genotype()
    print(f"Genotype: {genotype}")

if __name__ == "__main__":
    test_network()
