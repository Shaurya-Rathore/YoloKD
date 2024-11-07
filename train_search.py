import os
import sys
import time
import glob
import numpy as np
import torch
import darts_utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.nn.utils.rnn import pad_sequence
import gc
from ultralytics.utils.loss import v8DetectionLoss
from darts_utils import YOLOLoss,process_yolov8_output
from torch.autograd import Variable
from model_search import YOLOv8StudentModel
from architect import Architect
from dataloader import YOLOObjectDetectionDataset,custom_collate_fn
from torch.amp import autocast


parser = argparse.ArgumentParser("WAID")
parser.add_argument('--img_dir', type=str, default='/kaggle/input/ooga-dataset/ooga/ooga-main/ooga/images/train', help='location of images')
parser.add_argument('--label_dir', type=str, default='/kaggle/input/ooga-dataset/ooga/ooga-main/ooga/labels/train', help='location labels')
parser.add_argument('--val_img_dir', type=str, default='/kaggle/input/ooga-dataset/ooga/ooga-main/ooga/images/valid', help='location of images')
parser.add_argument('--val_label_dir', type=str, default='/kaggle/input/ooga-dataset/ooga/ooga-main/ooga/labels/valid', help='location labels')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=10, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=10, help='number of training epochs')
parser.add_argument('--init_channels', type=int, default=8, help='number of initial channels')
parser.add_argument('--layers', type=int, default=14, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
args = parser.parse_args()

args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
darts_utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


WAID_CLASSES = 6

"""def custom_collate(batch):
    inputs, targets = zip(*batch)
    
    # Pad inputs to match the size of the largest tensor in the batch
    inputs_padded = pad_sequence(inputs, batch_first=True)

    # Handle dictionary-based targets and pad them if necessary
    if isinstance(targets[0], dict):
        targets_padded = {}
        for key in targets[0].keys():
            # Extract the list of tensors for this key
            tensors_for_key = [t[key] for t in targets]
            # Pad the tensors to make them the same size
            targets_padded[key] = pad_sequence(tensors_for_key, batch_first=True)
    else:
        # If targets are simple tensors, pad them directly
        targets_padded = pad_sequence(targets, batch_first=True)

    return inputs_padded, targets_padded"""

def check_gradients(model):
    has_gradients = False
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is not None:
                if param.grad.abs().sum() > 0:
                    has_gradients = True
                    #print(f"Parameter {name} has non-zero gradients")
                    break
    if not has_gradients:
        print("WARNING: No gradients are flowing!")

def check_weight_updates(model):
    # Store initial weights
    initial_weights = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            initial_weights[name] = param.data.clone()
    
    return initial_weights

def compare_weights(model, initial_weights):
    changed = False
    for name, param in model.named_parameters():
        if param.requires_grad:
            if not torch.equal(initial_weights[name], param.data):
                #print(f"Weights changed for {name}")
                changed = True
                break
    if not changed:
        print("WARNING: No weights were updated!")

def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)
  gc.collect()
  torch.cuda.empty_cache()
  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)

  model = YOLOv8StudentModel(WAID_CLASSES, args.init_channels, args.layers, steps=4, multiplier=4, stem_multiplier=3)
  model = model.cuda()
  model = model.half()
  criterion = v8DetectionLoss(model)
  #criterion = criterion.cuda()

  logging.info("param size = %fMB", darts_utils.count_parameters_in_MB(model))

  optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay)

  train_transform = darts_utils._data_transforms_WAID(args)
  classes = ['sheep','cattle','seal','camelus','kiang','zebra']
  train_data = YOLOObjectDetectionDataset(img_dir = args.img_dir,label_dir=args.label_dir,classes = classes,transform=train_transform)

  train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      pin_memory=True, num_workers=2,collate_fn=custom_collate_fn)
  logging.info('length %s', len(train_queue))
  
  val_transform = darts_utils._val_data_transforms_WAID(args)
  valid_data = YOLOObjectDetectionDataset(img_dir = args.val_img_dir,label_dir=args.val_label_dir,classes = classes,transform=val_transform)
  valid_queue = torch.utils.data.DataLoader(
      valid_data, batch_size=args.batch_size,
      pin_memory=True, num_workers=2,collate_fn=custom_collate_fn)

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)

  architect = Architect(model, args)
  scaler = torch.amp.GradScaler('cuda')  # For mixed precision training
  best_loss = float('inf')

  with autocast(device_type='cuda'):
    for epoch in range(args.epochs):
        scheduler.step()
        lr = scheduler.get_lr()[0]
        logging.info('epoch %d lr %e', epoch, lr)

        genotype = model.genotype()
        logging.info('genotype = %s', genotype)

        # print(f"alphas_normal: {F.softmax(model.alphas_normal, dim=-1)}")
        # print(f"alphas_reduce: {F.softmax(model.alphas_reduce, dim=-1)}")

        # training
        #train_acc, train_obj 
        train_total_loss, train_loss_dict,predforprint = train(
                train_queue, 
                valid_queue, 
                model, 
                architect, 
                criterion, 
                optimizer, 
                lr
            )
        logging.info('My list: %s', predforprint)
        logging.info(
                'Training - Total Loss: %.10f | Box Loss: %.10f | Class Loss: %.10f | DFL Loss: %.10f',
                train_total_loss,
                train_loss_dict['box_loss'],
                train_loss_dict['cls_loss'],
                train_loss_dict['dfl_loss']
            )

         # Validation
        valid_total_loss, valid_loss_dict = infer(valid_queue, model, criterion)
        logging.info(
                'Validating - Total Loss: %.10f | Box Loss: %.10f | Class Loss: %.10f | DFL Loss: %.10f',
                valid_total_loss,
                valid_loss_dict['box_loss'],
                valid_loss_dict['cls_loss'],
                valid_loss_dict['dfl_loss']
            )

        darts_utils.save(model, os.path.join(args.save, 'weights.pt'))


def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr):
    loss_meter = darts_utils.AvgrageMeter()
    box_loss_meter = darts_utils.AvgrageMeter()
    cls_loss_meter = darts_utils.AvgrageMeter()
    dfl_loss_meter = darts_utils.AvgrageMeter()
    pred_list = []
    for step, (input, target) in enumerate(train_queue):
        initial_weights = check_weight_updates(model)
        model.train()
        n = input.size(0)
        
        input = Variable(input, requires_grad=False).cuda()
        input = input.half()
        target = {
            "batch_idx": Variable(target["batch_idx"], requires_grad=False).cuda(),
            "cls": Variable(target["cls"], requires_grad=False).cuda(),
            "bboxes": Variable(target["bboxes"], requires_grad=False).cuda(),
        }

        # Get a random minibatch from the validation queue
        input_search, target_search = next(iter(valid_queue))
        input_search = Variable(input_search, requires_grad=False).cuda()
        target_search = {
            "batch_idx": Variable(target_search["batch_idx"], requires_grad=False).cuda(),
            "cls": Variable(target_search["cls"], requires_grad=False).cuda(),
            "bboxes": Variable(target_search["bboxes"], requires_grad=False).cuda(),
        }
        pred = model(input)
        #print("PRED: ", pred)
        pred_list.append(pred)
        #print("Length of pred_list: ", len(pred_list))
        #print(pred_list)
        pred_search = model(input_search)
        
        # Architecture step
        architect.step(pred, target, pred_search, target_search, lr, optimizer, unrolled=args.unrolled)
        # Optimization step
        optimizer.zero_grad()

        #print(f"Predictions: {pred}")
        #print(f"Target: {target}")

        total_loss, loss_items = criterion(pred, target)
        total_loss.backward()


        # Gradient clipping
        if args.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        compare_weights(model, initial_weights)

        # Update metrics
        box_loss, cls_loss, dfl_loss = loss_items
        loss_meter.update(total_loss.item(), n)
        box_loss_meter.update(box_loss.item(), n)
        cls_loss_meter.update(cls_loss.item(), n)
        dfl_loss_meter.update(dfl_loss.item(), n)

        # Logging
        if step % args.report_freq == 0:
            logging.info(
                f'Step: {step} | '
                f'Total Loss: {loss_meter.avg:.4f} | '
                f'Box Loss: {box_loss_meter.avg:.4f} | '
                f'Class Loss: {cls_loss_meter.avg:.4f} | '
                f'DFL Loss: {dfl_loss_meter.avg:.4f}'
            )

    return (
        loss_meter.avg,
        {
            'box_loss': box_loss_meter.avg,
            'cls_loss': cls_loss_meter.avg,
            'dfl_loss': dfl_loss_meter.avg
        },pred_list
    )



def infer(valid_queue, model, criterion):
  loss_meter = darts_utils.AvgrageMeter()
  box_loss_meter = darts_utils.AvgrageMeter()
  cls_loss_meter = darts_utils.AvgrageMeter()
  dfl_loss_meter = darts_utils.AvgrageMeter()
  model.eval()  
    
  with torch.no_grad():  # Disable gradient computation
      for step, (input, target) in enumerate(valid_queue):
          # Process input
          input = input.cuda()
          input = input.half()  # Convert to half precision
          
          # Process targets
          target = {
              "batch_idx": target["batch_idx"].cuda(),
              "cls": target["cls"].cuda(),
              "bboxes": target["bboxes"].cuda(),
          }
          
          # Forward pass
          with autocast(device_type='cuda', enabled=True):  # Use mixed precision
              pred = model(input)
              total_loss, loss_items = criterion(pred, target)
              
              # Unpack loss items
              box_loss, cls_loss, dfl_loss = loss_items
              
              # Update meters
              n = input.size(0)
              loss_meter.update(total_loss.item(), n)
              box_loss_meter.update(box_loss.item(), n)
              cls_loss_meter.update(cls_loss.item(), n)
              dfl_loss_meter.update(dfl_loss.item(), n)
          
          # Logging
          if step % args.report_freq == 0:
              logging.info(
                  f'Validation Step: {step} | '
                  f'Total Loss: {loss_meter.avg:.4f} | '
                  f'Box Loss: {box_loss_meter.avg:.4f} | '
                  f'Class Loss: {cls_loss_meter.avg:.4f} | '
                  f'DFL Loss: {dfl_loss_meter.avg:.4f}'
              )
  
  # Return both total loss and individual components
  return (
      loss_meter.avg,
      {
          'box_loss': box_loss_meter.avg,
          'cls_loss': cls_loss_meter.avg,
          'dfl_loss': dfl_loss_meter.avg
      }
  )


if __name__ == '__main__':
  main()