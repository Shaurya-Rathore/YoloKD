import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from PIL import Image
import numpy as np

class YOLOObjectDetectionDataset(Dataset):
    def __init__(self, img_dir, label_dir, classes, img_size=600, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.classes = classes
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        self.transform = transform
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        label_path = os.path.join(self.label_dir, self.img_files[idx].replace('.jpg', '.txt').replace('.png', '.txt').replace('.jpeg', '.txt'))

        # Load image
        image = Image.open(img_path).convert("RGB")
        orig_width, orig_height = image.size
        image = self.transform(image)

        # Load labels
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as file:
                for line in file:
                    class_id, x_center, y_center, width, height = map(float, line.strip().split())
                    # Convert normalized coordinates to pixel coordinates
                    x_center *= orig_width
                    y_center *= orig_height
                    width *= orig_width
                    height *= orig_height
                    # Convert to [x_min, y_min, x_max, y_max] format
                    x_min = x_center - width / 2
                    y_min = y_center - height / 2
                    x_max = x_center + width / 2
                    y_max = y_center + height / 2
                    # Normalize to [0, 1] for the resized image
                    x_min /= orig_width
                    y_min /= orig_height
                    x_max /= orig_width
                    y_max /= orig_height
                    labels.append([int(class_id), x_min, y_min, x_max, y_max])

        # Convert labels to tensor
        labels = torch.tensor(labels)

        # Create target tensors
        boxes = labels[:, 1:] if len(labels) > 0 else torch.zeros((0, 4))
        labels = labels[:, 0].long() if len(labels) > 0 else torch.zeros(0, dtype=torch.int64)

        return image, boxes, labels

    def get_class_name(self, class_id):
        return self.classes[class_id]

def custom_collate_fn(batch):
    # Separate batch components
    images = []
    targets = []

    for i, (image, box,label) in enumerate(batch):
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

