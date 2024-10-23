import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from PIL import Image
import numpy as np

import albumentations as A
from albumentations.pytorch import ToTensorV2

class YOLOObjectDetectionDataset(Dataset):
    def __init__(self, img_dir, label_dir, classes, img_size=600):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.classes = classes
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        # Define transformations
        self.transform = A.Compose([
            A.Resize(img_size, img_size),
            # Add other transformations if needed
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        label_path = os.path.join(
            self.label_dir,
            self.img_files[idx].replace('.jpg', '.txt').replace('.png', '.txt').replace('.jpeg', '.txt')
        )

        # Load image
        image = np.array(Image.open(img_path).convert("RGB"))

        # Load labels
        boxes = []
        class_labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as file:
                for line in file:
                    class_id, x_center, y_center, width, height = map(float, line.strip().split())
                    boxes.append([x_center, y_center, width, height])  # Normalized coordinates
                    class_labels.append(int(class_id))

        # Apply transformations
        transformed = self.transform(image=image, bboxes=boxes, class_labels=class_labels)
        image = transformed['image']
        boxes = transformed['bboxes']
        class_labels = transformed['class_labels']

        # Convert to tensors
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(class_labels, dtype=torch.int64)

        return image, boxes, labels

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

