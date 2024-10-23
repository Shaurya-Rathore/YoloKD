import torch
import torch.nn as nn
from types import SimpleNamespace
import math
import torch
import torch.nn as nn
from model_search import YOLOv8StudentModel 
from ultralytics.utils.loss import v8DetectionLoss

def create_random_training_batch(batch_size=2, img_size=32, num_classes=6, num_objects=3, device='cuda'):
    """
    Creates a random batch of training data in the format expected by YOLOv8.
    """
    # Random images
    images = torch.randn(batch_size, 3, img_size, img_size, device=device)
    
    # Random targets
    targets = []
    for batch_idx in range(batch_size):
        for _ in range(num_objects):
            # Random box: [batch_idx, class, x, y, w, h]
            x = torch.rand(1, device=device)
            y = torch.rand(1, device=device)
            w = torch.rand(1, device=device) * 0.3  # Limit size to 30% of image
            h = torch.rand(1, device=device) * 0.3
            cls = torch.randint(0, num_classes, (1,), device=device).float()
            
            box = torch.tensor([batch_idx, cls, x, y, w, h], device=device)
            targets.append(box)
    
    if targets:
        targets = torch.stack(targets)
    else:
        targets = torch.zeros((0, 6), device=device)  # Empty targets tensor
    
    # Create batch dictionary
    batch = {
        'images': images,
        'batch_idx': targets[:, 0],
        'cls': targets[:, 1],
        'bboxes': targets[:, 2:],  # x, y, w, h
    }
    
    return batch

def test_student_model():
    """
    Test the YOLOv8StudentModel with random data.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Model parameters
    num_classes = 6
    img_size = 32
    batch_size = 2
    
    # Initialize model
    model = YOLOv8StudentModel(
        num_classes=num_classes,
        C=64,
        layers=14,
        steps=4,
        multiplier=4,
        stem_multiplier=3
    ).to(device)
    
    # Add required attributes for loss function
    model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)  # Loss gains
    
    # Initialize loss function
    loss_fn = v8DetectionLoss(model)
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    
    # Test training loop
    print("\nStarting test training loop...")
    model.train()
    
    for step in range(5):  # Test 5 steps
        print(f"\nStep {step + 1}")
        
        # Create random batch
        batch = create_random_training_batch(
            batch_size=batch_size,
            img_size=img_size,
            num_classes=num_classes,
            device=device
        )
        
        # Forward pass
        optimizer.zero_grad()
        predictions = model(batch['images'])
        print(batch.shape)
        
        # Calculate loss
        try:
            loss, loss_items = loss_fn(predictions, batch)
            print(f"Loss calculated successfully:")
            print(f"- Total loss: {loss.item():.4f}")
            print(f"- Box loss: {loss_items[0].item():.4f}")
            print(f"- Class loss: {loss_items[1].item():.4f}")
            print(f"- DFL loss: {loss_items[2].item():.4f}")
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
        except Exception as e:
            print(f"Error during loss calculation: {str(e)}")
            break
            
        print(f"Step {step + 1} completed successfully")
    
    print("\nTest completed!")

if __name__ == "__main__":
    # Run the test
    test_student_model()