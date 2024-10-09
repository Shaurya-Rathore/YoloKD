import torch
import torch.nn as nn
from model_search import YOLOv8StudentModel 

def test_network():
    # Define hyperparameters
    C = 16  # Initial number of channels
    num_classes = 6  # Number of output classes
    layers = 8  # Number of layers in the network
    criterion = nn.CrossEntropyLoss()  # Loss function

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
    bbox_preds = model(x)  # Unpack the tuple returned by forward()
    #print(f"Output shape: {logits.shape}")
    print("bbox",bbox_preds) 

    # Calculate loss
    #loss = model._loss(x, labels)
    #print(f"Loss: {loss.item()}")

    # Print the genotype
    #genotype = model.genotype()
    #print(f"Genotype: {genotype}")

if __name__ == "__main__":
    test_network()
