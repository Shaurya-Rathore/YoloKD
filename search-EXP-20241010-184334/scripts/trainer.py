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
from ultralytics.nn.modules.dataloader import YOLOObjectDetectionDataset
from ultralytics import YOLO
from ultralytics.utils.loss import DFLoss, BboxLoss
import wandb
import numpy as np
import yaml
from torch.autograd import Variable
#from ultralytics.nn.modules.model import DARTSModel as Network

wandb.init(mode='disabled')

outputs = []
def get_shapes(obj):
    if isinstance(obj, torch.Tensor):
        return obj.shape  # Return shape of the tensor
    elif isinstance(obj, list):
        return [get_shapes(o) for o in obj]  # Recursively check lists
    elif isinstance(obj, tuple):
        return tuple(get_shapes(o) for o in obj)  # Recursively check tuples
    else:
        return None

def forward_hook(module, input, output):
    if isinstance(output, tuple):
        output = output[0]
    #outputs = module.postprocess(output, max_det=module.max_det, nc=module.nc)
    outputs.append(output)

# Argument Parsing
parser = argparse.ArgumentParser("WAID")
parser.add_argument('--img_dir', type=str, default='/kaggle/input/ooga-dataset/ooga/ooga-main/ooga/', help='location of images')
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
# Initialize the teacher model
# teacher = YOLO('yolov8-LDconv.yaml')
# #model_state_dict = torch.load("/kaggle/input/yolov8m-pt/yolov8m.pt")
# teacher.to(device)
# dummy = torch.rand(1,3,640,640)
# # for name, layer in teacher.named_modules():
# #     print(name, layer)
# layer = getattr(teacher.model.model, '22')
# # print(layer)
# hook_handle = layer.register_forward_hook(forward_hook)
# # hooks = []
# # for name, layer in teacher.model.named_modules():
# #     hooks.append(layer.register_forward_hook(forward_hook))
# #teacher.load_state_dict(torch.load('/YoloKD/yolowts.pt'))
# #img_path = "C:\\Users\\Shaurya\\Pictures\\aadhaar page 1.jpg"
# with torch.no_grad():  # No gradient computation is needed
#     output = teacher(dummy)

# # print("Final Output:", output)  # This is the model's output

# print("Captured Output from the Hook:", get_shapes(outputs))
# # Experiment setup
# try:
#     args.save = 'eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
#     ultralytics.nn.modules.darts_utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
    
#     log_format = '%(asctime)s %(message)s'
#     logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')

#     fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
#     fh.setFormatter(logging.Formatter(log_format))
#     logging.getLogger().addHandler(fh)
# except Exception as e:
#     print(f"Logging setup failed: {e}")
#     sys.exit(1)

# print('broo')
# YOLO Loss Class
class YOLOLoss(nn.Module):
    def __init__(self, lambda_bbox=5.0, lambda_obj=1.0, lambda_noobj=0.5, lambda_class=1.0):
        super(YOLOLoss, self).__init__()
        self.lambda_bbox = lambda_bbox
        self.lambda_obj = lambda_obj
        self.lambda_noobj = lambda_noobj
        self.lambda_class = lambda_class
        self.mse = nn.MSELoss()
        self.bce = nn.BCELoss()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, predictions, targets):
        pred_bbox, pred_obj, pred_class = predictions
        target_bbox, target_obj, target_class = targets
        bbox_loss = self.mse(pred_bbox, target_bbox)
        obj_loss = self.bce(pred_obj, target_obj)
        no_obj_loss = self.bce(1 - pred_obj, 1 - target_obj)
        class_loss = self.ce(pred_class, target_class)
        total_loss = (self.lambda_bbox * bbox_loss +
                      self.lambda_obj * obj_loss +
                      self.lambda_noobj * no_obj_loss +
                      self.lambda_class * class_loss)
        return total_loss

# Main function
def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)
    print('in here')
    teacher = YOLO('yolov8-LDconv.yaml')
    model_state_dict = torch.load("/kaggle/input/yolov8m-pt/yolov8m.pt")
    teacher.to(device)
    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    genotype = eval("ultralytics.nn.modules.genotypes.%s" % args.arch)
    model = YOLO('yolov8n.yaml')
    model = model.cuda()
    logging.info("param size = %fMB", ultralytics.nn.modules.darts_utils.count_parameters_in_MB(model))

    criterion = YOLOLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

    train_data = YOLOObjectDetectionDataset(img_dir=args.img_dir, label_dir=args.label_dir, classes=['sheep', 'cattle', 'seal', 'camelus', 'kiang', 'zebra'], transform=ultralytics.nn.modules.darts_utils._data_transforms_WAID)
    train_queue = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, pin_memory=True, num_workers=2)

    valid_data = YOLOObjectDetectionDataset(img_dir=args.val_img_dir, label_dir=args.val_label_dir, classes=['sheep', 'cattle', 'seal', 'camelus', 'kiang', 'zebra'], transform=ultralytics.nn.modules.darts_utils._val_data_transforms_WAID(args))
    valid_queue = torch.utils.data.DataLoader(valid_data, batch_size=args.batch_size, pin_memory=True, num_workers=2)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
    print('almost there')

    for epoch in range(args.epochs):
        scheduler.step()
        logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
        print("before training")
        model.drop_path_prob = args.drop_path_prob * epoch / args.epochs
        train_acc, train_obj = train(train_queue, model, teacher, criterion, optimizer, args)
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
    teacher.eval()
    model.train()
    print(len(train_queue))
    for step, (input, target) in enumerate(train_queue):
        input, target = input.cuda(), target.cuda()
        optimizer.zero_grad()

        with torch.no_grad():
            teacher_logits = teacher(input)

        logits, logits_aux = model(input)

        soft_targets = nn.functional.softmax(teacher_logits / args.temperature, dim=-1)
        soft_prob = nn.functional.log_softmax(logits / args.temperature, dim=-1)
        soft_targets_loss = nn.KLDivLoss(reduction='batchmean')(soft_prob, soft_targets) * (args.temperature ** 2)

        hard_loss = criterion(logits, target)
        loss = args.ce_weight * hard_loss + args.kd_weight * soft_targets_loss

        if args.auxiliary:
            loss_aux = criterion(logits_aux, target)
            loss += args.auxiliary_weight * loss_aux

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, prec5 = ultralytics.nn.modules.darts_utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
            logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

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
    print('main')
    main()
