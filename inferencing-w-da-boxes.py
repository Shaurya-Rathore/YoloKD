import os
import sys
import time
import glob
import numpy as np
import torch
import ultralytics.nn.modules.darts_utils
from PIL import Image
import logging
import argparse
import torch.nn as nn
import ultralytics.nn.modules.genotypes
import torch.utils
import torchvision.datasets as dset
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from ultralytics.nn.modules.dataloader import YOLOObjectDetectionDataset, custom_collate_fn
from ultralytics.nn.modules.darts_utils import YOLOLoss, process_yolov8_output
from ultralytics import YOLO
from ultralytics.utils.loss import DFLoss, BboxLoss
import wandb
import numpy as np
import yaml
from torch.autograd import Variable
from ultralytics.utils.loss import v8DetectionLoss
from torchmetrics.detection import MeanAveragePrecision

def simple_nms(boxes, scores, iou_threshold=0.5):
    # Convert to tensor if needed
    boxes = torch.tensor(boxes)
    scores = torch.tensor(scores)
    
    # Sort by descending confidence
    sorted_indices = torch.argsort(scores, descending=True)
    keep = []
    
    while sorted_indices.numel() > 0:
        # Keep the highest confidence box
        current_idx = sorted_indices[0]
        keep.append(current_idx.item())
        
        if sorted_indices.numel() == 1:
            break
            
        # Compute IoU with remaining boxes
        current_box = boxes[current_idx].unsqueeze(0)
        remaining_boxes = boxes[sorted_indices[1:]]
        
        # Calculate intersection
        x1 = torch.max(current_box[:, 0], remaining_boxes[:, 0])
        y1 = torch.max(current_box[:, 1], remaining_boxes[:, 1])
        x2 = torch.min(current_box[:, 2], remaining_boxes[:, 2])
        y2 = torch.min(current_box[:, 3], remaining_boxes[:, 3])
        
        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        
        # Calculate union
        area_current = (current_box[:, 2] - current_box[:, 0]) * (current_box[:, 3] - current_box[:, 1])
        area_remaining = (remaining_boxes[:, 2] - remaining_boxes[:, 0]) * (remaining_boxes[:, 3] - remaining_boxes[:, 1])
        union = area_current + area_remaining - intersection
        
        iou = intersection / union
        
        # Remove boxes with IoU > threshold
        mask = iou <= iou_threshold
        sorted_indices = sorted_indices[1:][mask]
    
    return keep

image_dir = r'C:\Users\Shaurya\Downloads\WAID\WAID\images\test'
label_dir = r'C:\Users\Shaurya\Downloads\WAID\WAID\labels\test'

model = YOLO('yolov8n.yaml')

model_state_dict = torch.load(r"C:\Users\Shaurya\Downloads\yolov8_softshare_waid (1).pt")
model.model.load_state_dict(model_state_dict, strict=True)
conf_threshold = 0.5

metric = MeanAveragePrecision(class_metrics=True)
count = 0
for image_path in os.listdir(image_dir):
    img = Image.open(os.path.join(image_dir, image_path)).convert("RGB")
    img_width, img_height = img.size

    results = model.predict(img, conf=0.3)
    result = results[0]

    # Load ground truth
    label_path = os.path.join(label_dir, os.path.splitext(image_path)[0] + '.txt')
    true_boxes, true_labels = [], []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f.readlines():
                class_id, x_center, y_center, width, height = map(float, line.strip().split())
                x1 = (x_center - width/2) * img_width
                y1 = (y_center - height/2) * img_height
                x2 = (x_center + width/2) * img_width
                y2 = (y_center + height/2) * img_height
                true_boxes.append([x1, y1, x2, y2])
                true_labels.append(int(class_id))

    predictions = {
        'boxes': [],
        'scores': [],
        'labels': []
    }

    replacement_map = {}

    for i in range(len(result.boxes)):
        box = result.boxes[i]
        predictions['boxes'].append(box.xyxy[0].tolist())
        predictions['scores'].append(box.conf.item())
        predictions['labels'].append(int(box.cls.item()))
    
    for i in range(len(predictions['scores'])):
        if predictions['scores'][i] < conf_threshold:
            # Get original box coordinates
            x1, y1, x2, y2 = predictions['boxes'][i]

            new_x1 = max(0, int(x1) - 50)
            new_y1 = max(0, int(y1) - 50)
            new_x2 = min(img_width, int(x2) + 50)
            new_y2 = min(img_height, int(y2) + 50)
            
            if (new_x2 <= new_x1) or (new_y2 <= new_y1):
                continue
                
            # Second pass inference
            cropped_img = img.crop((new_x1, new_y1, new_x2, new_y2))
            resized_img = cropped_img.resize((640, 640))
            new_results = model.predict(resized_img, conf=conf_threshold)
            new_result = new_results[0]

            # Process new detections
            best_conf = 0
            best_pred = None
            
            for j in range(len(new_result.boxes)):
                new_box = new_result.boxes[j]
                new_conf = new_box.conf.item()
                if new_conf > best_conf:
                    best_conf = new_conf
                    # Convert coordinates back to original space
                    nx1, ny1, nx2, ny2 = new_box.xyxy[0].tolist()
                    scale_x = (new_x2 - new_x1) / 640
                    scale_y = (new_y2 - new_y1) / 640
                    abs_x1 = new_x1 + nx1 * scale_x
                    abs_y1 = new_y1 + ny1 * scale_y
                    abs_x2 = new_x1 + nx2 * scale_x
                    abs_y2 = new_y1 + ny2 * scale_y
                    
                    best_pred = {
                        'box': [abs_x1, abs_y1, abs_x2, abs_y2],
                        'label': int(new_box.cls.item()),
                        'score': new_conf
                    }

            # Replace original prediction if better
            if best_pred and best_pred['score'] > predictions['scores'][i]:
                replacement_map[i] = best_pred

    for idx, pred in replacement_map.items():
        predictions['boxes'][idx] = pred['box']
        predictions['scores'][idx] = pred['score']
        predictions['labels'][idx] = pred['label']

    # Apply NMS to final predictions
    if predictions['boxes']:
        boxes_tensor = torch.tensor(predictions['boxes'])
        scores_tensor = torch.tensor(predictions['scores'])
        labels_tensor = torch.tensor(predictions['labels'])
        
        keep_indices = simple_nms(boxes_tensor, scores_tensor, iou_threshold=0.5)
        
        final_boxes = boxes_tensor[keep_indices].tolist()
        final_labels = labels_tensor[keep_indices].tolist()
        final_scores = scores_tensor[keep_indices].tolist()
    else:
        final_boxes = []
        final_labels = []
        final_scores = []

    preds = [{
        'boxes': torch.tensor(final_boxes) if final_boxes else torch.zeros((0, 4)),
        'scores': torch.tensor(final_scores) if final_scores else torch.zeros(0),
        'labels': torch.tensor(final_labels) if final_labels else torch.zeros(0),
    }]

    targets = [{
        'boxes': torch.tensor(true_boxes) if true_boxes else torch.zeros((0, 4)),
        'labels': torch.tensor(true_labels) if true_labels else torch.zeros(0),
    }]

    metric.update(preds, targets)
    count += 1

    # Update metrics
    metric.update(preds, targets)
    count += 1

# Calculate and print final metrics
final_metrics = metric.compute()
print(f"mAP@0.5: {final_metrics['map_50']:.4f}")
print(f"Precision: {final_metrics['map_per_class'].mean():.4f}")
print(f"Recall: {final_metrics['mar_100'].mean():.4f}")