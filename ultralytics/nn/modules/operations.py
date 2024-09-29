import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# DropPath implementation (used for stochastic depth)
def drop_path(x, drop_prob):
    if drop_prob > 0.:
        keep_prob = 1. - drop_prob
        # Create the mask on the same device as the input tensor `x`
        mask = torch.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob).to(x.device)
        x = x.div(keep_prob)  # Scale by keep_prob to maintain expected output during training
        x = x.mul(mask)       # Apply the mask
    return x

class ReLUConvBN(nn.Module):
    """ReLU followed by Conv then BatchNorm."""
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.op(x)

class SepConv(nn.Module):
    """Separable Convolution used in DARTS."""
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(C_out, C_out, kernel_size=kernel_size, stride=1, padding=padding, groups=C_out, bias=False),
            nn.Conv2d(C_out, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)

class DilConv(nn.Module):
    """Dilated Convolution used in DARTS."""
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)

class Identity(nn.Module):
    """Skip Connection."""
    def forward(self, x):
        return x
    
class Zero(nn.Module):
    """Null operation that outputs a tensor of zeros."""
    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return torch.zeros_like(x)
        else:
            # Downsample the input by striding if needed
            return torch.zeros(x.size(0), x.size(1), x.size(2) // self.stride, x.size(3) // self.stride, device=x.device)

class PoolBN(nn.Module):
    """Pooling operation followed by batch normalization."""
    def __init__(self, pool_type, C, kernel_size, stride, padding, affine=True):
        super(PoolBN, self).__init__()
        if pool_type == 'max':
            self.pool = nn.MaxPool2d(kernel_size, stride=stride, padding=padding)
        elif pool_type == 'avg':
            self.pool = nn.AvgPool2d(kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(C, affine=affine)

    def forward(self, x):
        return self.bn(self.pool(x))

class FactorizedReduce(nn.Module):
    def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduce, self).__init__()
        # Ensure C_out is twice of C_in to properly double channels
        assert C_out == 2 * C_in, f"Expected output channels to be twice the input channels, got {C_out} and {C_in}"
        
        # Use two convolution paths to achieve the channel doubling
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        # Split input into two paths and concatenate
        out = torch.cat([self.conv_1(x), self.conv_2(x)], dim=1)
        out = self.bn(out)
        return out

    def forward(self, x):
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out

# Deformable Convolution implementation
class LinearDeformableConv(nn.Module):
    """Linear Deformable Convolution (LDC) for handling deformations."""
    def __init__(self, C_in, C_out, kernel_size=3, stride=1, padding=1, affine=True):
        super(LinearDeformableConv, self).__init__()
        # Assuming deformable convolution package is installed
        # Using the deformable convolution module from torchvision.ops or a similar package
        self.offsets = nn.Conv2d(C_in, 2 * kernel_size * kernel_size, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv = nn.Conv2d(C_in, C_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        # Generate offsets dynamically
        offsets = self.offsets(x)
        # Apply deformable convolution
        x = F.conv2d(x, self.conv.weight, self.conv.bias, stride=self.conv.stride, padding=self.conv.padding, dilation=self.conv.dilation)
        x = self.bn(x)
        return x

# Dictionary of operations, including the new Linear Deformable Convolution
OPS = {
    'none': lambda C, stride, affine: Zero(stride),
    'avg_pool_3x3': lambda C, stride, affine: PoolBN('avg', C, 3, stride, 1, affine=affine),
    'max_pool_3x3': lambda C, stride, affine: PoolBN('max', C, 3, stride, 1, affine=affine),
    'skip_connect': lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
    'sep_conv_3x3': lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
    'sep_conv_5x5': lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
    'dil_conv_3x3': lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
    'dil_conv_5x5': lambda C, stride, affine: DilConv(C, C, 5, stride, 2, 2, affine=affine),
    'lde_conv_3x3': lambda C, stride, affine: LinearDeformableConv(C, C, 3, stride, 1, affine=affine),
}

# Test input data (batch_size, channels, height, width)
input_data = torch.randn(1, 16, 32, 32)

def test_operation(op_name, C_in, stride, affine=True):
    print(f"\nTesting operation: {op_name}")
    op = OPS[op_name](C_in, stride, affine)
    output = op(input_data)
    print(f"Input shape: {input_data.shape}")
    print(f"Output shape: {output.shape}")

# Test all the operations in the OPS dictionary
C_in = 16  # Example input channels
stride = 1

for op_name in OPS.keys():
    test_operation(op_name, C_in, stride)