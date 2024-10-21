import torch
import torch.nn as nn
import torch.nn.functional as F
import unittest
import copy
import time
# from ultralytics.utils.tal import dist2bbox, make_anchors
# from ultralytics.nn.modules.conv import Conv
# from ultralytics.nn.modules.block import DFL
from operations import *
from torch.cuda.amp import autocast, GradScaler

from genotypes import PRIMITIVES
from genotypes import Genotype
from darts_utils import YOLOLoss

def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
        sy, sx = torch.meshgrid(sy, sx, indexing="ij") # if TORCH_1_10 else torch.meshgrid(sy, sx)
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)
  
def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)  # xywh bbox
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox
  
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p
  
class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))
      
class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)

class MixedOp(nn.Module):

  def __init__(self, C, stride):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    for primitive in PRIMITIVES:
      op = OPS[primitive](C, stride, False)
      if 'pool' in primitive:
        op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
      self._ops.append(op)

  def forward(self, x, weights):
    total = 0
    for w, op in zip(weights, self._ops):
      y=op(x)
      total += w * y
      print(total.shape)
    return total

class StemLayer(nn.Module):
  def __init__(self, in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1):
    super(StemLayer, self).__init__()
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
    self.bn = nn.BatchNorm2d(out_channels)
    self.relu = nn.ReLU(inplace=True)

  def forward(self, x):
    x = self.conv(x)
    x = self.bn(x)
    x = self.relu(x)
    return x

class Cell(nn.Module):

  def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
    super(Cell, self).__init__()
    self.reduction = reduction

    if reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
    self._steps = steps
    self._multiplier = multiplier

    self._ops = nn.ModuleList()
    self._bns = nn.ModuleList()
    for i in range(self._steps):
      for j in range(2+i):
        stride = 2 if reduction and j < 2 else 1
        op = MixedOp(C, stride)
        self._ops.append(op)

  def forward(self, s0, s1, weights):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]
    offset = 0
    for i in range(self._steps):
      s = sum(self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
      offset += len(states)
      states.append(s)

    final_output = torch.cat(states[-self._multiplier:], dim=1)
    
    print(f"Final concatenated state shape: {final_output.shape}")
    return final_output

class Network(nn.Module):

  def __init__(self, C, num_classes, layers, criterion, steps=4, multiplier=4, stem_multiplier=3):
    super(Network, self).__init__()
    self._C = C
    self._num_classes = num_classes
    self._layers = layers
    self._criterion = criterion
    self._steps = steps
    self._multiplier = multiplier

    C_curr = stem_multiplier*C
    self.stem = nn.Sequential(
      nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
      nn.BatchNorm2d(C_curr)
    )
 
    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
    self.cells = nn.ModuleList()
    reduction_prev = False
    for i in range(layers):
      if i in [layers//3, 2*layers//3]:
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, multiplier*C_curr

    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)

    self._initialize_alphas()

  def new(self):
    model_new = Network(self._C, self._num_classes, self._layers, self._criterion)
    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
        x.data.copy_(y.data)
    return model_new

  def forward(self, input):
    s0 = s1 = self.stem(input)
    for i, cell in enumerate(self.cells):
      if cell.reduction:
        weights = F.softmax(self.alphas_reduce, dim=-1)
      else:
        weights = F.softmax(self.alphas_normal, dim=-1)
      s0, s1 = s1, cell(s0, s1, weights)
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0),-1))
    return logits

  def _loss(self, input, target):
    logits = self(input)
    return self._criterion(logits, target) 

  def _initialize_alphas(self):
    k = sum(1 for i in range(self._steps) for n in range(2+i))
    num_ops = len(PRIMITIVES)

    self.alphas_normal = nn.Parameter(1e-3*torch.randn(k, num_ops))
    self.alphas_reduce = nn.Parameter(1e-3*torch.randn(k, num_ops))
    self._arch_parameters = [
      self.alphas_normal,
      self.alphas_reduce,
    ]

  def arch_parameters(self):
    return self._arch_parameters

  def genotype(self):

    def _parse(weights):
      gene = []
      n = 2
      start = 0
      for i in range(self._steps):
        end = start + n
        W = weights[start:end].copy()
        edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
        for j in edges:
          k_best = None
          for k in range(len(W[j])):
            if k != PRIMITIVES.index('none'):
              if k_best is None or W[j][k] > W[j][k_best]:
                k_best = k
          gene.append((PRIMITIVES[k_best], j))
        start = end
        n += 1
      return gene

    gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
    gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())

    concat = range(2+self._steps-self._multiplier, self._steps+2)
    genotype = Genotype(
      normal=gene_normal, normal_concat=concat,
      reduce=gene_reduce, reduce_concat=concat
    )
    return genotype

class DARTSBackbone(nn.Module):
  def __init__(self, C, layers, steps=4, multiplier=4, stem_multiplier=3):
    super(DARTSBackbone, self).__init__()
    
    # Assign backbone parameters
    self._C = C
    self._layers = layers
    self._steps = steps  # Set the number of steps per cell
    self._multiplier = multiplier  # Set multiplier
    
    # Stem layer to process the initial input (same as in the StemLayer)
    C_curr = stem_multiplier * C
    self.stem = nn.Sequential(
        nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
        nn.BatchNorm2d(C_curr),
        nn.ReLU(inplace=True)
    )

    # Initialize previous and current channels
    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
    self.cells = nn.ModuleList()
    reduction_prev = False
    
    self.cell6_index = 5   # Index for the 6th cell
    self.cell10_index = 9  # Index for the 10th cell
    
    for i in range(layers):
      if i in [3, 7, 11]:  # Reduction cells at 3rd, 7th, and 11th cells
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      
      cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
      self.cells.append(cell)
      
      reduction_prev = reduction
      C_prev_prev, C_prev = C_prev, multiplier * C_curr  # Update for next cell
    
    # Initialize architecture parameters (alphas)
    self._initialize_alphas()
      
  def _initialize_alphas(self):
    """Initialize architecture parameters (alphas)."""
    k = sum(1 for i in range(self._steps) for n in range(2 + i))
    num_ops = len(PRIMITIVES)

    # Alphas control the operation weights in the cells
    self.alphas_normal = nn.Parameter(1e-3 * torch.randn(k, num_ops))
    self.alphas_reduce = nn.Parameter(1e-3 * torch.randn(k, num_ops))
    
    # Register alphas as learnable parameters
    self._arch_parameters = [self.alphas_normal, self.alphas_reduce]
    
  def arch_parameters(self):
    return self._arch_parameters

  def forward(self, x):
    # Use autocast for mixed precision
    with autocast():
      s0 = s1 = self.stem(x)
      C2, C3 = None, None  # Capture outputs from the 6th and 10th cells

      for i, cell in enumerate(self.cells):
        # Apply weights based on the type of cell
        if cell.reduction:
          weights = F.softmax(self.alphas_reduce, dim=-1)
        else:
          weights = F.softmax(self.alphas_normal, dim=-1)

        s0, s1 = s1, cell(s0, s1, weights)  # Forward pass through each cell

        if i == self.cell6_index:
          C2 = s1  # Output from the 6th cell
        if i == self.cell10_index:
          C3 = s1  # Output from the 10th cell

    C4 = s1  # Final output from the backbone

    # Free unused GPU memory
    torch.cuda.empty_cache()

    return C2, C3, C4  # Return feature maps

  # Mixed precision training setup example
  # Use `torch.cuda.amp.GradScaler` to handle gradients safely during training
  def train_step(model, optimizer, data, target):
    # Initialize the scaler for mixed precision gradients
    scaler = GradScaler()

    optimizer.zero_grad()  # Clear gradients
    
    # Use autocast for forward pass
    with autocast():
      output = model(data)  # Forward pass
      loss = F.mse_loss(output[0], target)  # Example loss function

    # Scale loss and backward pass
    scaler.scale(loss).backward()

    # Optimizer step with gradient scaling
    scaler.step(optimizer)

    # Update scaler for next step
    scaler.update()

    return loss.item()


class NeckFPN(nn.Module):
    def __init__(self, in_channels):
        super(NeckFPN, self).__init__()

        # Define convolutions to adjust the channel dimensions for each feature map
        self.conv_c4 = nn.Conv2d(in_channels[2], 256, kernel_size=1)  # C4 (75x75), reduce channels to 256
        self.conv_c3 = nn.Conv2d(in_channels[1], 256, kernel_size=1)  # C3 (150x150), reduce channels to 256
        self.conv_c2 = nn.Conv2d(in_channels[0], 256, kernel_size=1)  # C2 (300x300), reduce channels to 256
        self.conv_c4 = self.conv_c4.to(torch.float16)
        self.conv_c3 = self.conv_c3.to(torch.float16)
        self.conv_c2 = self.conv_c2.to(torch.float16)
        # Final 3x3 convolutions after feature map fusion
        self.final_c2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)  # Final C2 (300x300)
        self.final_c3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)  # Final C3 (150x150)
        self.final_c4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)  # Final C4 (75x75)
        self.final_c2 = self.final_c2.to(torch.float16)
        self.final_c2 = self.final_c2.to(torch.float16)
        self.final_c2 = self.final_c2.to(torch.float16)

    def forward(self, c2, c3, c4):
        # c2: 300x300 from the 6th backbone cell (shallower, high-resolution, fewer channels)
        # c3: 150x150 from the 10th backbone cell (mid-level)
        # c4: 75x75 from the final backbone cell (deepest, lowest resolution, most channels)

        # Step 1: Adjust channels for C4 (75x75)
        c4_out = self.conv_c4(c4)  # Adjust channels for C4: (75x75 -> 256 channels)

        # Step 2: Upsample C4 (75x75 -> 150x150) and fuse with C3
        c4_upsampled = F.interpolate(c4_out, scale_factor=2, mode='nearest')  # 75x75 -> 150x150
        c4_upsampled = c4_upsampled.to(torch.float16)
        c3_fused = self.conv_c3(c3) + c4_upsampled  # Fuse C3 (150x150) and upsampled C4 (150x150)
        c3_fused = c3_fused.to(torch.float16)
        # Step 3: Upsample fused C3 (150x150 -> 300x300) and fuse with C2
        c3_upsampled = F.interpolate(c3_fused, scale_factor=2, mode='nearest')  # 150x150 -> 300x300
        c3_upsampled = c3_upsampled.to(torch.float16)
        c2_fused = self.conv_c2(c2) + c3_upsampled  # Fuse C2 (300x300) and upsampled C3 (300x300)
        c2_fused = c2_fused.to(torch.float16)

        # Step 4: Apply final 3x3 convolutions to each fused feature map
        c2_final = self.final_c2(c2_fused)  # Final output for C2 (300x300)
        c3_final = self.final_c3(c3_fused)  # Final output for C3 (150x150)
        c4_final = self.final_c4(c4_out)    # Final output for C4 (75x75)
        c2_final = c2_final.to(torch.float16)
        c3_final = c3_final.to(torch.float16)
        c4_final = c4_final.to(torch.float16)

        return c2_final, c3_final, c4_final  # Return feature maps at 300x300, 150x150, 75x75

class Detect(nn.Module):
    """YOLOv8 Detect head for detection models."""

    dynamic = False  # force grid reconstruction
    export = False  # export mode
    end2end = False  # end2end
    max_det = 300  # max_det
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=6, ch=()):
        """Initializes the YOLOv8 detection layer with specified number of classes and channels."""
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 4 # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        if isinstance(ch[0], torch.Tensor):
            ch = [tensor.shape[1] for tensor in ch]  
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        if self.end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        if self.end2end:
            return self.forward_end2end(x)

        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:  # Training path
            return x
        y = self._inference(x)
        
        return y if self.export else (y, x)

    def forward_end2end(self, x):
        """
        Performs forward pass of the v10Detect module.

        Args:
            x (tensor): Input tensor.

        Returns:
            (dict, tensor): If not in training mode, returns a dictionary containing the outputs of both one2many and one2one detections.
                           If in training mode, returns a dictionary containing the outputs of one2many and one2one detections separately.
        """
        x_detach = [xi.detach() for xi in x]
        one2one = [
            torch.cat((self.one2one_cv2[i](x_detach[i]), self.one2one_cv3[i](x_detach[i])), 1) for i in range(self.nl)
        ]
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:  # Training path
            return {"one2many": x, "one2one": one2one}

        y = self._inference(one2one)
        y = self.postprocess(y.permute(0, 2, 1), self.max_det, self.nc)
        return y if self.export else (y, {"one2many": x, "one2one": one2one})

    def _inference(self, x):
        """Decode predicted bounding boxes and class probabilities based on multiple-level feature maps."""
        # Inference path
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and self.format in {"saved_model", "pb", "tflite", "edgetpu", "tfjs"}:  # avoid TF FlexSplitV ops
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        if self.export and self.format in {"tflite", "edgetpu"}:
            # Precompute normalization factor to increase numerical stability
            # See https://github.com/ultralytics/ultralytics/issues/7371
            grid_h = shape[2]
            grid_w = shape[3]
            grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            dbox = self.decode_bboxes(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2])
        else:
            dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides
        return torch.cat((dbox, cls.sigmoid()), 1)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)
        if self.end2end:
            for a, b, s in zip(m.one2one_cv2, m.one2one_cv3, m.stride):  # from
                a[-1].bias.data[:] = 1.0  # box
                b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)

    def decode_bboxes(self, bboxes, anchors):
        """Decode bounding boxes."""
        return dist2bbox(bboxes, anchors, xywh=not self.end2end, dim=1)

    @staticmethod
    def postprocess(preds: torch.Tensor, max_det: int, nc: int = 6):
        """
        Post-processes YOLO model predictions.

        Args:
            preds (torch.Tensor): Raw predictions with shape (batch_size, num_anchors, 4 + nc) with last dimension
                format [x, y, w, h, class_probs].
            max_det (int): Maximum detections per image.
            nc (int, optional): Number of classes. Default: 80.

        Returns:
            (torch.Tensor): Processed predictions with shape (batch_size, min(max_det, num_anchors), 6) and last
                dimension format [x, y, w, h, max_class_prob, class_index].
        """
        batch_size, anchors, predictions = preds.shape  # i.e. shape(16,8400,84)
        print("anchorr",anchors)
        boxes, scores = preds.split([4, nc], dim=-1)
        index = scores.amax(dim=-1).topk(min(max_det, anchors))[1].unsqueeze(-1)
        boxes = boxes.gather(dim=1, index=index.repeat(1, 1, 4))
        scores = scores.gather(dim=1, index=index.repeat(1, 1, nc))
        scores, index = scores.flatten(1).topk(max_det)
        i = torch.arange(batch_size)[..., None]  # batch indices
        return torch.cat([boxes[i, index // nc], scores[..., None], (index % nc)[..., None].float()], dim=-1)
    

class YOLOv8StudentModel(nn.Module):
  def __init__(self, num_classes, C=64, layers=14, steps=4, multiplier=4, stem_multiplier=3):
    """
    YOLOv8 Student Model with DARTS backbone, Neck, and YOLOv8 Detection Head.
    
    Args:
        num_classes (int): Number of object classes.
        C (int): Initial number of channels for the backbone.
        layers (int): Total number of cells in the backbone (14 cells here).
        steps (int): Number of steps per DARTS cell.
        multiplier (int): Multiplier for channels in DARTS cells.
        stem_multiplier (int): Multiplier for channels in the stem layer.
    """
    super(YOLOv8StudentModel, self).__init__()
    
    # DARTS-based backbone with 14 layers (3 reduction cells)
    self.backbone = DARTSBackbone(C=C, layers=layers, steps=steps, multiplier=multiplier, stem_multiplier=stem_multiplier)
    self.arch_parameters = self.backbone.arch_parameters()
    self._multiplier = multiplier
    self._criterion = YOLOLoss()
    # Example: The channels from the backbone after feature extraction
    # Assuming C3 and C4 feature maps from backbone
    backbone_out_channels = [C*multiplier*2,C * multiplier * 4, C * multiplier * 8]  # Example for C3 and C4
    self._steps = steps

    # Neck that fuses multi-scale feature maps
    self.neck = NeckFPN(in_channels=backbone_out_channels) 
    neck_out_channels = [256, 256, 256]  # Adjust based on your NeckFPN implementation
    self.detect = Detect(nc=num_classes, ch=neck_out_channels)
    self._initialize_alphas()

  def _initialize_alphas(self):
    k = sum(1 for i in range(self._steps) for n in range(2+i))
    num_ops = len(PRIMITIVES)

    self.alphas_normal = nn.Parameter(1e-3*torch.randn(k, num_ops))
    self.alphas_reduce = nn.Parameter(1e-3*torch.randn(k, num_ops))
    self._arch_parameters = [
      self.alphas_normal,
      self.alphas_reduce,
    ]

  def _loss(self, input, target):
    logits = self(input)
    return self._criterion(logits, target) 
  
  def genotype(self):

    def _parse(weights):
      gene = []
      n = 2
      start = 0
      for i in range(self._steps):
        end = start + n
        W = weights[start:end].copy()
        edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
        for j in edges:
          k_best = None
          for k in range(len(W[j])):
            if k != PRIMITIVES.index('none'):
              if k_best is None or W[j][k] > W[j][k_best]:
                k_best = k
          gene.append((PRIMITIVES[k_best], j))
        start = end
        n += 1
      return gene

    gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
    gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())

    concat = range(2+self._steps-self._multiplier, self._steps+2)
    genotype = Genotype(
      normal=gene_normal, normal_concat=concat,
      reduce=gene_reduce, reduce_concat=concat
    )
    return genotype
     

  def forward(self, x):
    """
    Forward pass through the student model.
    
    Args:
        x (Tensor): Input image tensor (batch, channels, height, width).
    
    Returns:
        Tuple: Bounding box predictions, objectness scores, class predictions.
    """
    # Step 1: Forward pass through DARTS backbone (multi-scale feature maps)
    features = self.backbone(x)
    
    # Assuming the backbone returns two feature maps (C3 and C4)
    C2,C3, C4 = features[0], features[1],features[2]  # Take feature maps for fusion
    
    # Step 2: Pass feature maps through the neck for multi-scale fusion
    fused_features = self.neck(C2,C3,C4)
    # Step 3: Predict bounding boxes, objectness, and class scores
    x= self.detect(list(fused_features))
    #pred_bbox = x[:, :, :4]  
    #pred_obj = x[:, :, 4:5] 
    #pred_class = x[:, :, 5:]

    return x#pred_bbox, pred_obj,pred_class
  
def process_yolov8_output(output, num_classes=6, reg_max=4):
    """
    Process YOLOv8 output to extract bounding box predictions and class probabilities.
    
    Args:
    output (torch.Tensor): Output tensor from YOLOv8 detection head.
    num_classes (int): Number of classes.
    reg_max (int): DFL channels.
    
    Returns:
    tuple: (dbox, cls) where dbox is the bounding box predictions and cls is the class probabilities.
    """
    no = num_classes + reg_max * 4
    
    # Split the output into box and cls parts
    dbox = output[:, :reg_max * 4]
    cls = output[:, reg_max * 4:]
    
    # Reshape box to (batch_size, 4, reg_max, -1)
    #box = box.view(box.shape[0], 4, reg_max, -1).permute(0, 2, 1, 3)
    
    # Apply softmax to box predictions
    #box = box.softmax(1)
    
    # Calculate the expected value (sum of softmax * indices)
    #box = box * torch.arange(reg_max, device=box.device).float().view(1, -1, 1, 1)
    #dbox = box.sum(1)
    
    # Apply sigmoid to class probabilities
    cls = cls.sigmoid()
    objectness = cls.max(dim=-1).values
    return dbox, cls, objectness