import os
import numpy as np
import torch
import shutil
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
import cv2
import torch.nn as nn


class AvgrageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0)
    res.append(correct_k.mul_(100.0/batch_size))
  return res


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img

def cv2_resize(image):
        if isinstance(image, Image.Image):
            # Convert PIL Image to numpy array
            image = np.array(image)
        elif isinstance(image, torch.Tensor):
            # Convert PyTorch tensor to numpy array
            image = image.permute(1, 2, 0).numpy()
        
        if len(image.shape) == 2:
            # If the image is grayscale, convert to RGB
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            # If the image has an alpha channel, convert to RGB
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        # Resize the image
        resized = cv2.resize(image, (640, 640), interpolation=cv2.INTER_LINEAR)
        
        # Convert back to PIL Image
        return Image.fromarray(resized)


def _data_transforms_WAID(args):
  WAID_MEAN = [0.4788, 0.4791, 0.4789]
  WAID_STD = [0.2009, 0.2009, 0.2009]

  train_transform = transforms.Compose([
    transforms.Lambda(cv2_resize),  
    # transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(WAID_MEAN, WAID_STD),
  ])
  if args.cutout:
    train_transform.transforms.append(Cutout(args.cutout_length))

  return train_transform

def _val_data_transforms_WAID(args):
  WAID_MEAN = [0.4804, 0.4807, 0.4805]
  WAID_STD = [0.1996, 0.1995, 0.1995]

  val_transform = transforms.Compose([
    transforms.Lambda(cv2_resize),    
    transforms.ToTensor(),
    transforms.Normalize(WAID_MEAN, WAID_STD),
    ])
  return val_transform


def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6


def save_checkpoint(state, is_best, save):
  filename = os.path.join(save, 'checkpoint.pth.tar')
  torch.save(state, filename)
  if is_best:
    best_filename = os.path.join(save, 'model_best.pth.tar')
    shutil.copyfile(filename, best_filename)


def save(model, model_path):
  torch.save(model.state_dict(), model_path)


def load(model, model_path):
  model.load_state_dict(torch.load(model_path))


def drop_path(x, drop_prob):
  if drop_prob > 0.:
    keep_prob = 1.-drop_prob
    mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
    x.div_(keep_prob)
    x.mul_(mask)
  return x


def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    os.mkdir(path)
  print('Experiment dir : {}'.format(path))

  if scripts_to_save is not None:
    os.mkdir(os.path.join(path, 'scripts'))
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)

class YOLOLoss(nn.Module):
    def _init_(self,lambda_bbox=5.0, lambda_obj=1.0, lambda_noobj=0.5, lambda_class=1.0):
        super(YOLOLoss, self)._init_()
        # Weights for each component of the loss
        self.lambda_bbox = lambda_bbox
        self.lambda_obj = lambda_obj
        self.lambda_noobj = lambda_noobj
        self.lambda_class = lambda_class

        # Loss functions for different components
        self.mse = nn.MSELoss()  # For bounding box regression
        self.bce = nn.BCELoss()  # For objectness prediction
        self.ce = nn.CrossEntropyLoss()  # For classification prediction

    def forward(self,predictions,targets):
        # Unpack predictions and targets
        # Assuming that 'predictions' is a tuple of (bbox, objectness, class_probs)
        # And 'targets' is the same structure
        pred_bbox, pred_obj, pred_class = predictions
        target_bbox, target_obj, target_class = targets

        # Bounding Box Loss
        bbox_loss = self.mse(pred_bbox, target_bbox)

        # Objectness Loss (whether object exists in the cell or not)
        obj_loss = self.bce(pred_obj, target_obj)

        # No-objectness Loss (penalize for false predictions of objects where none exist)
        no_obj_loss = self.bce(1 - pred_obj, 1 - target_obj)

        # Classification Loss (multi-class task)
        class_loss = self.ce(pred_class, target_class)

        # Combine losses
        total_loss = (
            self.lambda_bbox * bbox_loss +
            self.lambda_obj * obj_loss +
            self.lambda_noobj * no_obj_loss +
            self.lambda_class * class_loss
        )

        return total_loss


Test_Mean = [0.4766, 0.4769, 0.4767]
Test_Std = [0.1985, 0.1984, 0.1984]

