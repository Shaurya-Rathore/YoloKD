import torch.nn as nn
from types import SimpleNamespace
import math
import torch
import torch.nn as nn
from model_search import YOLOv8StudentModel 
from ultralytics.utils.loss import v8DetectionLoss

def create_random_training_batch(batch_size=2, max_det=300, num_classes=6, min_objects=2, max_objects=8, device='cpu'):
    """
    Creates a random batch of training data matching YOLOv8's post-NMS output format.
    
    Args:
        batch_size (int): Number of images in batch
        max_det (int): Maximum number of detections per image
        num_classes (int): Number of object classes
        min_objects (int): Minimum number of objects per image
        max_objects (int): Maximum number of objects per image
        device (str): Device to create tensors on
        
    Returns:
        tuple: (predictions, targets)
            - predictions: tensor of shape (batch_size, max_det, 6) [x, y, w, h, conf, cls]
            - targets: tensor of shape (N, 6) [batch_idx, cls, x, y, w, h] where N is total number of objects
    """
    # List to collect all targets
    all_targets = []
    
    # Create predictions tensor matching post-NMS format
    # Shape: (batch_size, max_det, 6) where 6 is [x, y, w, h, conf, cls]
    predictions = torch.zeros((batch_size, max_det, 6), device=device)
    
    for batch_idx in range(batch_size):
        # Random number of objects for this image
        num_objects = torch.randint(min_objects, max_objects + 1, (1,)).item()
        
        # Ensure we don't exceed max_det
        num_objects = min(num_objects, max_det)
        
        for obj_idx in range(num_objects):
            # Random normalized box coordinates
            x = torch.rand(1, device=device).item()
            y = torch.rand(1, device=device).item()
            w = torch.rand(1, device=device).item() * 0.3  # Limit size to 30% of image
            h = torch.rand(1, device=device).item() * 0.3
            
            # Random class and high confidence for positive samples
            cls_idx = torch.randint(0, num_classes, (1,), device=device).item()
            conf = torch.rand(1, device=device).item() * 0.5 + 0.5  # Random confidence between 0.5 and 1.0
            
            # Add ground truth target
            target = torch.tensor([batch_idx, cls_idx, x, y, w, h], device=device)
            all_targets.append(target)
            
            # Add to predictions with small random offset to simulate predicted boxes
            pred_x = min(max(x + torch.randn(1, device=device).item() * 0.1, 0), 1)
            pred_y = min(max(y + torch.randn(1, device=device).item() * 0.1, 0), 1)
            pred_w = min(max(w + torch.randn(1, device=device).item() * 0.1, 0.1), 1)
            pred_h = min(max(h + torch.randn(1, device=device).item() * 0.1, 0.1), 1)
            
            predictions[batch_idx, obj_idx] = torch.tensor(
                [pred_x, pred_y, pred_w, pred_h, conf, cls_idx],
                device=device
            )
        
        # Fill remaining detections with low confidence background predictions
        if num_objects < max_det:
            for i in range(num_objects, max_det):
                # Random coordinates
                x = torch.rand(1, device=device).item()
                y = torch.rand(1, device=device).item()
                w = torch.rand(1, device=device).item() * 0.2
                h = torch.rand(1, device=device).item() * 0.2
                
                # Low confidence and random class
                conf = torch.rand(1, device=device).item() * 0.3  # Low confidence (0-0.3)
                cls_idx = torch.randint(0, num_classes, (1,), device=device).item()
                
                predictions[batch_idx, i] = torch.tensor(
                    [x, y, w, h, conf, cls_idx],
                    device=device
                )
    
    # Stack all targets
    targets = torch.stack(all_targets) if all_targets else torch.zeros((0, 6), device=device)
    
    return predictions, targets

# Example usage and testing
if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Create random batch
    predictions, targets = create_random_training_batch(
        batch_size=2,
        max_det=100,
        num_classes=6,
        min_objects=2,
        max_objects=8,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Print shapes and sample values
    print(f"Predictions shape: {predictions.shape}")  # Should be (batch_size, max_det, 6)
    print(f"Targets shape: {targets.shape}")  # Should be (N, 6) where N is total number of objects
    
    # Print sample prediction
    print("\nSample prediction (first box):")
    print("x, y, w, h:", predictions[0, 0, :4].tolist())
    print("confidence:", predictions[0, 0, 4].item())
    print("class_index:", predictions[0, 0, 5].item())
    
    # Print sample target
    print("\nSample target (first object):")
    print("batch_idx:", targets[0, 0].item())
    print("class:", targets[0, 1].item())
    print("x, y, w, h:", targets[0, 2:].tolist())

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
        
    # Initialize loss function
    loss_fn = v8DetectionLoss(model)
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    
    # Test training loop
    print("\nStarting test training loop...")
    model.train()
    
    for step in range(3):  # Test 5 steps
        print(f"\nStep {step + 1}")
        
        try:
            # Create random batch
            batch = create_random_training_batch(batch_size=2, max_det=300, num_classes=6)
            
            # Forward pass
            optimizer.zero_grad()
            images = torch.randn(batch_size, 3, img_size, img_size, device=device)
            predictions = model(images)  # Changed from batch['images'] to batch['img']
            
            # Calculate loss
            loss, loss_items = loss_fn(predictions, batch)  # Using .call instead of direct call
            
            print(f"Loss calculated successfully:")
            print(f"- Total loss: {loss.item():.4f}")
            print(f"- Box loss: {loss_items[0].item():.4f}")
            print(f"- Class loss: {loss_items[1].item():.4f}")
            print(f"- DFL loss: {loss_items[2].item():.4f}")
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
        except Exception as e:
            print(f"Error during step {step + 1}: {str(e)}")
            import traceback
            traceback.print_exc()
            break
            
        print(f"Step {step + 1} completed successfully")
    
    print("\nTest completed!")

if __name__ == "__main__":
    # Run the test
    test_student_model()