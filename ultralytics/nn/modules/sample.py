import torch
import torch.nn as nn
from model_search import YOLOv8StudentModel 

def process_yolov8_output(output, num_classes=6, reg_max=4):
    """
    Process YOLOv8 output to extract bounding box predictions and class probabilities.
    
    Args:
    output (torch.Tensor): Output tensor from YOLOv8 detection head.
    num_classes (int): Number of classes.
    reg_max (int): DFL channels.
    
    Returns:
    tuple: (dbox, cls) where dbox is the bounding box predictions and cls is the class probabilities.
    """
    no = num_classes + reg_max * 4
    
    # Split the output into box and cls parts
    dbox = output[:, :reg_max * 4]
    cls = output[:, reg_max * 4:]
    
    # Reshape box to (batch_size, 4, reg_max, -1)
    #box = box.view(box.shape[0], 4, reg_max, -1).permute(0, 2, 1, 3)
    
    # Apply softmax to box predictions
    #box = box.softmax(1)
    
    # Calculate the expected value (sum of softmax * indices)
    #box = box * torch.arange(reg_max, device=box.device).float().view(1, -1, 1, 1)
    #dbox = box.sum(1)
    
    # Apply sigmoid to class probabilities
    cls = cls.sigmoid()
    
    return dbox, cls


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
    
    for tensor in bbox_preds:
        print(tensor.shape)
        dbox,cls = process_yolov8_output(tensor)

    print("bbox",bbox_preds) 
    print("dbox",dbox.size()) 
    print("cls",cls.size()) 

    # Calculate loss
    loss = model._loss(x, labels)
    print(f"Loss: {loss.item()}")

    # Print the genotype
    genotype = model.genotype()
    print(f"Genotype: {genotype}")

if __name__ == "__main__":
    test_network()
