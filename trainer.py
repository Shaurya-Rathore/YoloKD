import os
import sys
import time
import glob
import numpy as np
import torch
import ultralytics.nn.modules.darts_utils
import logging
import argparse
import torch.nn as nn
import ultralytics.nn.modules.genotypes
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from ultralytics.nn.modules.dataloader import YOLOObjectDetectionDataset, custom_collate_fn
from ultralytics.nn.modules.darts_utils import YOLOLoss, process_yolov8_output
from ultralytics import YOLO
from ultralytics.utils.loss import DFLoss, BboxLoss
import wandb
import numpy as np
import yaml
from torch.autograd import Variable
#from ultralytics.nn.modules.model import DARTSModel as Network

wandb.login(key="833b800ff23eb3d26e6c85a8b9e1fc8bbafc9775") 
wandb.init(project="yolov8")

#wandb.init(mode='disabled')

outputs_teacher = []
outputs_student = []

def forward_hook_teacher(module, input, output):
    global outputs_teacher
    outputs_teacher.append(output)

def forward_hook_student(module, input, output):
    outputs_student.append(output)

def get_shapes(obj):
    if isinstance(obj, torch.Tensor):
        return obj.shape  # Return shape of the tensor
    elif isinstance(obj, list):
        return [get_shapes(o) for o in obj]  # Recursively check lists
    elif isinstance(obj, tuple):
        return tuple(get_shapes(o) for o in obj)  # Recursively check tuples
    else:
        return None
    
class DummyYOLOStudent(nn.Module):
    def __init__(self, num_classes=6):
        super(DummyYOLOStudent, self).__init__()
        
        # Backbone (simple convolution layers instead of YOLO-like backbone)
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),  # Input image has 3 channels (RGB)
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Reduce spatial size by 2
            
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # YOLO-like detection head
        self.head = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.Conv2d(128, num_classes + 4, kernel_size=1),  # num_classes + 4 (for bbox coordinates + obj score)
        )
        
    def forward(self, x):
        # Pass through the backbone
        x = self.backbone(x)
        
        # Pass through the detection head
        x = self.head(x)
        
        # Split output into bbox, objectness, and class predictions
        # Assuming output format: [batch, num_anchors, num_classes + 5, H, W]
        # Here we simply reshape to simplify, depending on the YOLO format you're using.
        # BBox predictions: 4 coordinates per bounding box (center_x, center_y, width, height)
        # Objectness prediction: 1 score for each anchor
        # Class prediction: num_classes probabilities for each anchor
        
        return x


# Argument Parsing
parser = argparse.ArgumentParser("WAID")
parser.add_argument('--img_dir', type=str, default='/kaggle/input/ooga-dataset/ooga/ooga-main/ooga/train/', help='location of images')
parser.add_argument('--label_dir', type=str, default='/kaggle/input/ooga-dataset/ooga/ooga-main/ooga/labels/train/', help='location labels')
parser.add_argument('--val_img_dir', type=str, default='/kaggle/input/ooga-dataset/ooga/ooga-main/ooga/images/valid/', help='location of images')
parser.add_argument('--val_label_dir', type=str, default='/kaggle/input/ooga-dataset/ooga/ooga-main/ooga/labels/valid/', help='location labels')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=10, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--kd_weight', type=float, default=0.9, help='weight for knowledge distillation loss')
parser.add_argument('--ce_weight', type=float, default=0.1, help='weight for cross-entropy loss')
parser.add_argument('--temperature', type=float, default=3.0, help='temperature for distillation')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# for name, layer in teacher.named_modules():
#     print(name, layer)
# teacher = YOLO('yolov8n.yaml')
# layer_teacher = getattr(teacher.model.model, '22')
# layer_student = getattr(teacher.model.model, '22')
# layer_teacher.register_forward_hook(forward_hook_teacher)
# dummy = torch.rand(8,3,640,640)
# output1= teacher(dummy)
# # print(f'the output1 is {len(output1)}')
# # print(f'the output2 is {output2}')
# # print(f'the output3 is {output3}')
# output = outputs_teacher[0]
# output = output[0]
# print(f'the head output is {get_shapes(output[0])}')
input = torch.rand(2,3,640,640)
model = DummyYOLOStudent()
print(model(input).shape)

# YOLO Loss Class
class YOLOKDLoss(nn.Module):
    def __init__(self, lambda_bbox=5.0, lambda_obj=1.0, lambda_noobj=0.5, lambda_class=1.0, lambda_kd=0.7, temperature=3.0):
        super(YOLOKDLoss, self).__init__()
        self.lambda_bbox = lambda_bbox
        self.lambda_obj = lambda_obj
        self.lambda_noobj = lambda_noobj
        self.lambda_class = lambda_class
        self.lambda_kd = lambda_kd
        self.temperature = temperature

        self.mse = nn.MSELoss()
        self.bce = nn.BCELoss()
        self.ce = nn.CrossEntropyLoss()
        self.kldiv = nn.KLDivLoss(reduction='batchmean')

    def forward(self, student_preds, teacher_preds, targets):
        # Unpack predictions
        student_bbox, student_class, student_obj = process_yolov8_output(student_preds)
        teacher_bbox, teacher_class, teacher_obj = process_yolov8_output(teacher_preds)
        target_bbox, target_obj, target_class = targets

        # Standard YOLO losses against ground truth
        bbox_loss = self.mse(student_bbox, target_bbox)
        obj_loss = self.bce(student_obj, target_obj)
        no_obj_loss = self.bce(1 - student_obj, 1 - target_obj)
        class_loss = self.ce(student_class, target_class)

        # Knowledge distillation loss for class predictions (teacher vs student logits)
        soft_teacher_class = nn.functional.softmax(teacher_class / self.temperature, dim=-1)
        soft_student_class = nn.functional.log_softmax(student_class / self.temperature, dim=-1)
        kd_class_loss = self.kldiv(soft_student_class, soft_teacher_class) * (self.temperature ** 2)

        # Distill bounding box predictions (teacher vs student)
        kd_bbox_loss = self.mse(student_bbox, teacher_bbox)

        # Distill objectness predictions (teacher vs student)
        kd_obj_loss = self.bce(student_obj, teacher_obj)

        # Combine the losses
        total_loss = (self.lambda_bbox * bbox_loss +
                      self.lambda_obj * obj_loss +
                      self.lambda_noobj * no_obj_loss +
                      self.lambda_class * class_loss +
                      self.lambda_kd * (kd_class_loss + kd_bbox_loss + kd_obj_loss))

        return total_loss

# Main function
def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)
    
    model = DummyYOLOStudent()
    criterion = YOLOKDLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

    train_data = YOLOObjectDetectionDataset(img_dir=args.img_dir, label_dir=args.label_dir, classes=['sheep', 'cattle', 'seal', 'camelus', 'kiang', 'zebra'], transform=ultralytics.nn.modules.darts_utils._data_transforms_WAID_shaurya(args))
    train_queue = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, pin_memory=True, num_workers=2,collate_fn=custom_collate_fn)

    valid_data = YOLOObjectDetectionDataset(img_dir=args.val_img_dir, label_dir=args.val_label_dir, classes=['sheep', 'cattle', 'seal', 'camelus', 'kiang', 'zebra'], transform=ultralytics.nn.modules.darts_utils._val_data_transforms_WAID_shaurya(args))
    valid_queue = torch.utils.data.DataLoader(valid_data, batch_size=args.batch_size, pin_memory=True, num_workers=2,collate_fn=custom_collate_fn)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
    print('before epochs')
    teacher = YOLO('yolov8m.yaml')
    model_state_dict = torch.load("/kaggle/input/yolowaid/yolov8_waid.pt")
    teacher.model.load_state_dict(model_state_dict, strict=False)

    teacher.to(device)
    teacher.train(data='/kaggle/input/d/shauryasinghrathore/waiddataset/WAID-main/WAID-main/WAID/data.yaml', epochs=1, batch=8, optimizer= 'AdamW')
    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    genotype = eval("ultralytics.nn.modules.genotypes.%s" % args.arch)
    
    model = model.cuda()
    logging.info("param size = %fMB", ultralytics.nn.modules.darts_utils.count_parameters_in_MB(model))    

    for epoch in range(args.epochs):
        scheduler.step()
        logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
        print("before training")
        model.drop_path_prob = args.drop_path_prob * epoch / args.epochs
        train_acc, train_obj = train(train_queue, model, teacher, criterion, optimizer, args)
        print(f'train accuracy: {train_acc}')
        logging.info('train_acc %f', train_acc)
        valid_acc, valid_obj = infer(valid_queue, model, criterion)
        logging.info('valid_acc %f', valid_acc)
        ultralytics.nn.modules.darts_utils.save(model, os.path.join(args.save, 'weights.pt'))

# Training function
def train(train_queue, model, teacher, criterion, optimizer, args):
    objs = ultralytics.nn.modules.darts_utils.AvgrageMeter()
    top1 = ultralytics.nn.modules.darts_utils.AvgrageMeter()
    top5 = ultralytics.nn.modules.darts_utils.AvgrageMeter()

    print("training")

    model.train()

    # layer_student.register_forward_hook(forward_hook_student)

    print(f'train queue length: {len(train_queue)}')
    for step, (input, target) in enumerate(train_queue):
        input, target = input.cuda(), target.cuda()
        optimizer.zero_grad()

        with torch.no_grad():
            print('pre-predict')
            global outputs_teacher
            outputs_teacher.clear()
            _ = teacher(input)
            print(f'trying for hook {outputs_teacher}')
            output = outputs_teacher[0]
            output = output[0]

        print(f'student outputs: {model(input)}')
        student_preds = model(input)

        target_bbox = target['bbox']
        target_obj = target['obj']
        target_class = target['class']
        targets = (target_bbox, target_obj, target_class)
        
        student_bbox, student_class, student_obj = process_yolov8_output(student_preds)

        print('basics')
        loss = criterion(student_preds, output, targets)
        print('lossed')
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, prec5 = ultralytics.nn.modules.darts_utils.accuracy(student_class, target_class, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        # Optionally log the progress every few steps
        if step % args.report_freq == 0:
            logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

        outputs_teacher = []

    # Return average metrics for monitoring
    return top1.avg, objs.avg

# Validation function
def infer(valid_queue, model, criterion):
    objs = ultralytics.nn.modules.darts_utils.AvgrageMeter()
    top1 = ultralytics.nn.modules.darts_utils.AvgrageMeter()
    top5 = ultralytics.nn.modules.darts_utils.AvgrageMeter()
    model.eval()

    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            input, target = input.cuda(), target.cuda()
            logits, _ = model(input)
            loss = criterion(logits, target)
            prec1, prec5 = ultralytics.nn.modules.darts_utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            if step % args.report_freq == 0:
                logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")