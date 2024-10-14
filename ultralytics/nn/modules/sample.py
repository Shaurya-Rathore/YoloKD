import torch
import torch.nn as nn
from model_search import YOLOv8StudentModel 

def process_yolo_output(output_tensor):
    # Reshape the output tensor
    batch_size, _, height, width = output_tensor.shape
    # Adjust for 6 classes: 4 (bbox) + 1 (objectness) + 6 (classes) = 11
    output = output_tensor.view(batch_size, 11, height, width).permute(0, 2, 3, 1).contiguous()
    
    # Extract bounding box coordinates
    xy = output[..., :2].sigmoid()  # Center x, y
    wh = output[..., 2:4].exp()     # Width, height
    
    # Extract objectness score
    obj_score = output[..., 4].sigmoid()
    
    # Extract class probabilities
    class_probs = output[..., 5:].sigmoid()
    
    # Combine objectness score and class probabilities
    class_scores = obj_score.unsqueeze(-1) * class_probs
    
    return [xy,wh], class_scores


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

    # Forward pass
    bbox_preds,nibber= model(x)  # Unpack the tuple returned by forward()
    #print(f"Output shape: {logits.shape}")
    bbox_list = []
    cls_list = []
    for tensor in bbox_preds:
        print(tensor.shape)
        #a,b = process_yolo_output(tensor)
        #bbox_list.append(a)
        #cls_list(b)
    for tensor in nibber:
        print("size",tensor.size())
        #dbox,cls_pred =tensor.split(4, dim=1)
    print("bbox",bbox_preds) 
    #print("bbox",dbox) 
    #print("cls",cls_pred) 

    # Calculate loss
    loss = model._loss(x, labels)
    print(f"Loss: {loss.item()}")

    # Print the genotype
    genotype = model.genotype()
    print(f"Genotype: {genotype}")

if __name__ == "__main__":
    test_network()
