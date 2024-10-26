import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from PIL import Image
import numpy as np
import glob
from torch.nn.utils.rnn import pad_sequence
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

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

def custom_collate_fn(batch):
    images, targets = zip(*batch)  # Separate images and targets
    
    # Stack images into a single tensor (batch_size, channels, height, width)
    images = torch.stack(images)

    # Process targets
    batch_indices = []
    cls_labels = []
    bboxes = []

    for target in targets:
        batch_indices.append(target["batch_idx"])
        cls_labels.append(target["cls"])
        bboxes.append(target["bboxes"])

    # Pad batch indices, class labels, and bounding boxes to the same size
    batch_indices_padded = pad_sequence(batch_indices, batch_first=True)
    cls_labels_padded = pad_sequence(cls_labels, batch_first=True)
    bboxes_padded = pad_sequence(bboxes, batch_first=True)

    # Return images and the padded targets
    return images, {
        "batch_idx": batch_indices_padded,
        "cls": cls_labels_padded,
        "bboxes": bboxes_padded}
