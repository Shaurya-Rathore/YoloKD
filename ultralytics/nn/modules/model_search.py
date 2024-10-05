import torch
import torch.nn as nn
import torch.nn.functional as F
import unittest
import copy
from ultralytics.utils.tal import dist2bbox, make_anchors
from operations import *
from torch.autograd import Variable
from .conv import Conv
from .block import DFL
from genotypes import PRIMITIVES
from genotypes import Genotype

class MixedOp(nn.Module):

  def __init__(self, C, stride):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    self.stride = stride  # Save stride
    for primitive in PRIMITIVES:
      op = OPS[primitive](C, stride, False)
      if 'pool' in primitive:
        op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
      self._ops.append(op)

  def forward(self, x, weights):
    # Get the size of the input tensor
    input_size = x.size()[2:]

    # Apply each operation and resample the output to match the input size if necessary
    result = 0
    for idx, (w, op) in enumerate(zip(weights, self._ops)):
      print(f"Applying operation {idx} ({PRIMITIVES[idx]}) with weight {w}")
      out = op(x)
      print(f"Output shape after operation {idx}: {out.size()}")
      
      # If the operation changed the spatial size, resample it back to the input size
      if out.size()[2:] != input_size:
        print(f"Resampling output of operation {idx} ({PRIMITIVES[idx]}) from {out.size()[2:]} to {input_size}")
        out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=True)
      
      result += w * out

    return result

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
    
    print(f"s0 shape after preprocessing: {s0.shape}, s1 shape after preprocessing: {s1.shape}")

    states = [s0, s1]
    offset = 0
    for i in range(self._steps):
      s = []
      for j, h in enumerate(states):
        # Apply the operation
        print(f"Applying operation {offset+j} in step {i}")
        out = self._ops[offset+j](h, weights[offset+j])
        
        # Ensure all outputs have the same size as s0
        if out.size()[2:] != s0.size()[2:]:
          print(f"Interpolating output from {out.size()[2:]} to {s0.size()[2:]} at operation {offset+j}")
          out = F.interpolate(out, size=s0.size()[2:], mode='bilinear', align_corners=True)
        
        s.append(out)
      
      s = sum(s)
      print(f"State shape after step {i}: {s.shape}")
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

    self.alphas_normal = Variable(1e-3*torch.randn(k, num_ops), requires_grad=True)
    self.alphas_reduce = Variable(1e-3*torch.randn(k, num_ops), requires_grad=True)
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
      
    # Initialize feature map placeholders for C3 and C4
    self.C3 = None  # Refined 150x150 feature map (from after 10th cell)
    self.C4 = None  # Final 75x75 feature map (from after 14th cell)
    
    # Initialize architecture parameters (alphas)
    self._initialize_alphas()
      
  def _initialize_alphas(self):
    """Initialize architecture parameters (alphas)."""
    k = sum(1 for i in range(self._steps) for n in range(2 + i))
    num_ops = len(PRIMITIVES)
    print(f"Initializing alphas with k={k} and num_ops={num_ops}")

    # Alphas control the operation weights in the cells
    self.alphas_normal = nn.Parameter(1e-3 * torch.randn(k, num_ops))
    self.alphas_reduce = nn.Parameter(1e-3 * torch.randn(k, num_ops))
    
    # Register alphas as learnable parameters
    self._arch_parameters = [self.alphas_normal, self.alphas_reduce]
    
  def arch_parameters(self):
    return self._arch_parameters

  def forward(self, x):
    s0 = s1 = self.stem(x)
    C3, C4 = None, None
    
    print(f"Stem output shape: {s1.shape}")
    
    for i, cell in enumerate(self.cells):
      # Use softmax on alphas to get the weights for each operation in the cell
      if cell.reduction:
        weights = F.softmax(self.alphas_reduce, dim=-1)
        print(f"Cell {i + 1} is a Reduction Cell")
      else:
        weights = F.softmax(self.alphas_normal, dim=-1)
        print(f"Cell {i + 1} is a Normal Cell")
        
      print(f"Cell {i + 1} output shape: {s1.shape}")
      
      # Perform the forward pass through the cell
      s0, s1 = s1, cell(s0, s1, weights)
      
      # Capture feature maps for C3 (after the 10th cell) and C4 (final output)
      if i == 10:
        C3 = s1  # Output just before the third reduction cell (150x150)
      elif i == 13:  # After the final 14th cell
        C4 = s1  # Final refined feature map (75x75)
    
    print(f"Final output shape: {s1.shape}")
    return [C3, C4], s1  # Final feature map from the backbone
  
  
# class TestDARTSBackbone(unittest.TestCase):
#   def setUp(self):
#     # Setup the backbone with 6 layers and initial channels
#     self.backbone = DARTSBackbone(C=64, layers=6)
  
#   def test_forward_pass(self):
#     # Test the forward pass with a 600x600 image
#     input_image = torch.randn(1, 3, 600, 600)
#     try:
#       output = self.backbone(input_image)
#     except Exception as e:
#       self.fail(f"Forward pass failed with error: {e}")

#   def test_output_shape(self):
#     # Test if the output shape is as expected
#     input_image = torch.randn(1, 3, 600, 600)
#     output = self.backbone(input_image)
#     # The output shape will depend on how many downsampling operations are applied
#     # For example, after 6 layers with 2 reduction cells:
#     # The output should be downsampled twice, leading to an output of 150x150.
#     self.assertEqual(output.shape, (1, 256, 150, 150))  # Check this based on layer setup

#   def test_variable_layers(self):
#     # Test different numbers of layers
#     for num_layers in [4, 6, 8, 10]:
#       backbone = DARTSBackbone(C=64, layers=num_layers)
#       input_image = torch.randn(1, 3, 600, 600)
#       try:
#         output = backbone(input_image)
#       except Exception as e:
#         self.fail(f"Backbone with {num_layers} layers failed with error: {e}")
#       # Ensure output is a tensor
#       self.assertIsInstance(output, torch.Tensor)

# if __name__ == '__main__':
#   unittest.main()

class Neck(nn.Module):
  def __init__(self, in_channels_list, out_channels):
    """
    Initializes a dual-scale neck structure using two scales of feature maps.
    
    Args:
      in_channels_list (list): List of input channels from the backbone [C3, C4].
      out_channels (int): Number of output channels after fusion.
    """
    super(Neck, self).__init__()

    # Depthwise separable convolutions for both feature maps
    self.conv1_C3 = nn.Sequential(
      nn.Conv2d(in_channels_list[0], in_channels_list[0], kernel_size=3, padding=1, groups=in_channels_list[0]),
      nn.Conv2d(in_channels_list[0], out_channels, kernel_size=1)
    )
    self.conv1_C4 = nn.Sequential(
      nn.Conv2d(in_channels_list[1], in_channels_list[1], kernel_size=3, padding=1, groups=in_channels_list[1]),
      nn.Conv2d(in_channels_list[1], out_channels, kernel_size=1)
    )

    # Upsample C4 to match the resolution of C3
    self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    # Final convolution to fuse the scales
    self.conv_fuse = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

  def forward(self, C3, C4):
    """
    Forward pass for the dual-scale neck.
    
    Args:
      C3 (Tensor): The highest resolution feature map from the backbone.
      C4 (Tensor): The second-highest resolution feature map from the backbone.
    
    Returns:
      Tensor: The fused feature map ready for the detection head.
    """
    # Process both scales
    P3 = self.conv1_C3(C3)
    P4 = self.conv1_C4(C4)

    # Upsample P4 and add it to P3
    P4 = self.upsample(P4)
    fused = P3 + P4  # Fusion of two scales

    # Final convolution
    fused = self.conv_fuse(fused)
    return fused

class Detect(nn.Module):
  """YOLOv8 Detect head for detection models."""

  dynamic = False  # force grid reconstruction
  export = False  # export mode
  end2end = False  # end2end
  max_det = 300  # max_det
  shape = None
  anchors = torch.empty(0)  # init
  strides = torch.empty(0)  # init

  def __init__(self, nc=80, ch=()):
    """Initializes the YOLOv8 detection layer with specified number of classes and channels."""
    super().__init__()
    self.nc = nc  # number of classes
    self.nl = len(ch)  # number of detection layers
    self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
    self.no = nc + self.reg_max * 4  # number of outputs per anchor
    self.stride = torch.zeros(self.nl)  # strides computed during build
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
  def postprocess(preds: torch.Tensor, max_det: int, nc: int = 80):
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
    
    # Example: The channels from the backbone after feature extraction
    # Assuming C3 and C4 feature maps from backbone
    backbone_out_channels = [C * multiplier * 4, C * multiplier * 8]  # Example for C3 and C4

    # Neck that fuses multi-scale feature maps
    self.neck = Neck(in_channels_list=backbone_out_channels, out_channels=256)  # Assuming output to be 256 channels
    
    # Detection head for predicting bounding boxes, objectness scores, and class probabilities
    self.detect_head = Detect(num_classes=num_classes, ch=[256])  # Input channels are 256 from the neck

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
    C3, C4 = features[0], features[1]  # Take feature maps for fusion
    
    # Step 2: Pass feature maps through the neck for multi-scale fusion
    fused_features = self.neck(C3, C4)
    
    # Step 3: Predict bounding boxes, objectness, and class scores
    bbox_preds, obj_preds, cls_preds = self.detect_head(fused_features)
    
    return bbox_preds, obj_preds, cls_preds
  
class TestDetectHead(unittest.TestCase):
    def setUp(self):
        # Initialize the Detection Head with the number of object classes and input channels from the neck
        self.num_classes = 10  # Example number of object classes
        self.in_channels = 256  # Example input channels from the Neck
        self.detect_head = Detect(num_classes=self.num_classes, ch=[self.in_channels])
        
        # Example input feature map from the Neck
        self.neck_output = torch.randn(1, self.in_channels, 150, 150)  # Example Neck output (1, 256, 150, 150)

    def test_forward_pass(self):
        # Test the forward pass of the Detection Head to ensure it runs without errors
        try:
            bbox_preds, obj_preds, cls_preds = self.detect_head(self.neck_output)
        except Exception as e:
            self.fail(f"Forward pass failed with error: {e}")
        
        # Check that the outputs are tensors
        self.assertIsInstance(bbox_preds, torch.Tensor, "Bounding box predictions are not a tensor")
        self.assertIsInstance(obj_preds, torch.Tensor, "Objectness predictions are not a tensor")
        self.assertIsInstance(cls_preds, torch.Tensor, "Class predictions are not a tensor")

    def test_output_shapes(self):
        # Forward pass through the Detection Head
        bbox_preds, obj_preds, cls_preds = self.detect_head(self.neck_output)
        
        # Assuming the head outputs predictions at each pixel location (e.g., 150x150 spatial resolution)
        expected_bbox_shape = (1, 4, 150, 150)  # Bounding boxes (4 coords per pixel)
        expected_obj_shape = (1, 1, 150, 150)   # Objectness score (1 per pixel)
        expected_cls_shape = (1, self.num_classes, 150, 150)  # Class probabilities (10 classes per pixel)
        
        # Check the output shapes
        self.assertEqual(bbox_preds.shape, expected_bbox_shape, f"Expected bounding box shape {expected_bbox_shape}, but got {bbox_preds.shape}")
        self.assertEqual(obj_preds.shape, expected_obj_shape, f"Expected objectness score shape {expected_obj_shape}, but got {obj_preds.shape}")
        self.assertEqual(cls_preds.shape, expected_cls_shape, f"Expected class prediction shape {expected_cls_shape}, but got {cls_preds.shape}")

if __name__ == '__main__':
    unittest.main()