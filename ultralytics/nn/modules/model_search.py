import torch
import torch.nn as nn
import torch.nn.functional as F
import unittest
from operations import *
from torch.autograd import Variable
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
      # Use a reduction cell every few layers to downsample the spatial size
      if i in [layers // 3, 2 * layers // 3]:
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
    
    print(f"Final output shape: {s1.shape}")
    return s1  # Final feature map from the backbone
  
  
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
#     input_image = torch.randn(1, 3, 224, 224)
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

class DetectionHead(nn.Module):
  def __init__(self, num_classes, in_channels, num_anchors=3):
    """
    Initializes the detection head, which outputs bounding box coordinates, objectness scores, and class probabilities.
    
    Args:
      num_classes (int): Number of classes in the dataset.
      in_channels (int): Number of input channels from the neck.
      num_anchors (int): Number of anchor boxes per grid cell (default: 3).
    """
    super(DetectionHead, self).__init__()

    # Number of output features for bounding box regression (x, y, w, h)
    self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1)

    # Number of output features for objectness score
    self.obj_pred = nn.Conv2d(in_channels, num_anchors * 1, kernel_size=1)

    # Number of output features for class scores (num_classes)
    self.cls_pred = nn.Conv2d(in_channels, num_anchors * num_classes, kernel_size=1)

  def forward(self, x):
    """
    Forward pass for the detection head.
    
    Args:
        x (Tensor): Feature map output from the neck.
    
    Returns:
        Tuple: (bounding box predictions, objectness predictions, class predictions)
    """
    # Predict bounding boxes (x, y, w, h)
    bbox_preds = self.bbox_pred(x)

    # Predict objectness scores
    obj_preds = self.obj_pred(x)

    # Predict class probabilities
    cls_preds = self.cls_pred(x)

    return bbox_preds, obj_preds, cls_preds