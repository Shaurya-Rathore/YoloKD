import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from PIL import Image
import numpy as np
import glob

import albumentations as A
from albumentations.pytorch import ToTensorV2

class YOLOObjectDetectionDataset(Dataset):
    def __init__(self, img_dir, label_dir, classes, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.classes = classes

        # Get all image file paths
        self.image_paths = sorted(glob.glob(os.path.join(self.img_dir, "*.jpg")))  # Adjust extension if needed

        # Get corresponding label file paths
        self.label_paths = sorted(glob.glob(os.path.join(self.label_dir, "*.txt")))

        assert len(self.image_paths) == len(self.label_paths), "Number of images and labels must be the same."

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        
        # Load labels
        label_path = self.label_paths[idx]
        labels = np.loadtxt(label_path).reshape(-1, 5)  # [class_id, x_center, y_center, width, height]
        
        # Convert labels to torch tensor
        labels = torch.tensor(labels, dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)

        # Get batch index
        batch_idx = torch.tensor([idx] * labels.size(0), dtype=torch.int64)

        # Create target dictionary as expected by the loss function
        target = {
            "batch_idx": batch_idx,
            "cls": labels[:, 0],      # Class IDs
            "bboxes": labels[:, 1:],  # Bounding boxes [x_center, y_center, width, height]
        }

        return image, target
    
    def __len__(self):
        return len(self.image_paths)

    def get_class_name(self, class_id):
        return self.classes[class_id]

def custom_collate_fn(batch):
    # Separate batch components
    images = []
    targets = []

    for i, (image, box, label) in enumerate(batch):
        images.append(image)

        if box.numel() > 0:  # Check if there are any boxes
            # Calculate the center coordinates, width, and height for each bounding box
            x_center = (box[:, 0] + box[:, 2]) / 2.0
            y_center = (box[:, 1] + box[:, 3]) / 2.0
            width = box[:, 2] - box[:, 0]
            height = box[:, 3] - box[:, 1]
            
            # Stack these into a tensor (bbox predictions)
            bbox = torch.stack((x_center, y_center, width, height), dim=1)

             # Concatenate the class labels and box coordinates
            target = torch.cat([label.unsqueeze(1).float(), bbox], dim=1)
            # Add the batch index as the first column
            target = torch.cat([torch.full((target.shape[0], 1), i).float(), target], dim=1)
            targets.append(target)

    # Stack images along the batch dimension
    images = torch.stack(images, 0)

    # Concatenate all bbox and class prediction tensors into a single tensor
    # Concatenate all target tensors into a single tensor
    if targets:
        targets = torch.cat(targets, 0)
    else:
        # If no targets, create an empty tensor with shape [0, 6]
        targets = torch.zeros((0, 6))
    # Return the images, bbox predictions, and class predictions
    
    return images, targets

