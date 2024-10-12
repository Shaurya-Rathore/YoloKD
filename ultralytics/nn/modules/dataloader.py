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
        labels = labels[:, 0:1].long() if len(labels) > 0 else torch.zeros(0, dtype=torch.int64)

        return image, boxes, labels

    def get_class_name(self, class_id):
        return self.classes[class_id]

def custom_collate_fn(batch):
    # batch = list(filter(lambda x: x is not None, batch))
    # return torch.utils.data.dataloader.default_collate(batch)
    images = []
    boxes = []
    labels = []
    for image, box, label in batch:
        print(image.shape)
        images.append(image)
        boxes.append(box)
        labels.append(label)
    images = torch.stack(images, 0)
    
    # Pad boxes and labels to have the same shape
    max_objects = max(box.shape[0] for box in boxes)
    padded_boxes = torch.zeros(len(boxes), max_objects, 4)
    padded_labels = torch.zeros(len(labels), max_objects, dtype=torch.long)
    
    for i, (box, label) in enumerate(zip(boxes, labels)):
        if box.numel() > 0:
            padded_boxes[i, :box.shape[0], :] = box
            padded_labels[i, :label.shape] = label
    
    return images, padded_boxes, padded_labels
