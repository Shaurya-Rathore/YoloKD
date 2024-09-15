import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *

# Define the Mish activation function (since it's not part of PyTorch's core)
class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

# CBL Block: Conv-BatchNorm-LeakyReLU
class CBL(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None):
        super(CBL, self).__init__()
        # Use padding of 1 for 3x3 convolution by default
        if padding is None:
            padding = 1 if kernel_size == 3 else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.leaky_relu(x)
        return x

# CBM Block: Conv-BatchNorm-Mish
class CBM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None):
        super(CBM, self).__init__()
        # Use padding of 1 for 3x3 convolution by default
        if padding is None:
            padding = 1 if kernel_size == 3 else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.mish = Mish()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.mish(x)
        return x


# Focus block with integrated CBL
class Focus(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Focus, self).__init__()
        # Use CBL block for processing after pixel slicing
        self.cbl = CBL(in_channels * 4, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        # Pixel slicing: split the input into four sub-regions
        top_left = x[:, :, ::2, ::2]       # Even rows, even columns
        top_right = x[:, :, ::2, 1::2]     # Even rows, odd columns
        bottom_left = x[:, :, 1::2, ::2]   # Odd rows, even columns
        bottom_right = x[:, :, 1::2, 1::2] # Odd rows, odd columns

        # Concatenate along the channel dimension (increases channels by 4)
        x = torch.cat([top_left, top_right, bottom_left, bottom_right], dim=1)

        # Pass through the CBL block
        x = self.cbl(x)
        return x
    

class SPP(nn.Module):
    def __init__(self, in_channels, pool_sizes=[1, 5, 9, 13]):
        super(SPP, self).__init__()
        self.pooling_layers = nn.ModuleList([
            nn.MaxPool2d(kernel_size=size, stride=1, padding=size // 2) for size in pool_sizes
        ])

    def forward(self, x):
        pooled_features = [x]  # Start with the original input feature map
        for pool in self.pooling_layers:
            pooled_features.append(pool(x))
        x = torch.cat(pooled_features, dim=1)  # Concatenate along the channel dimension
        return x

class Cell(nn.Module):
    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()
        self.reduction = reduction

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)

        # Build the DAG (Directed Acyclic Graph)
        self._compile(C, genotype, reduction)

    def _compile(self, C, genotype, reduction):
        """
        Compile the cell by creating the list of operations (edges) and connections.
        """
        self._ops = nn.ModuleList()
        self.indices = []
        self.concat = genotype.concat
        
        # Loop through each operation and input pair (dag-like structure)
        for name, index in genotype:
            stride = 2 if reduction and index < 2 else 1
            op = OPS[name](C, stride, True)
            self._ops.append(op)
            self.indices.append(index)

    def forward(self, s0, s1, drop_prob):
        """
        Forward pass through the cell.
        """
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(len(self._ops)):
            h = states[self.indices[i]]
            h = self._ops[i](h)
            if self.training and drop_prob > 0.:
                if not isinstance(self._ops[i], Identity):
                    h = drop_path(h, drop_prob)
            states.append(h)
        
        # Concatenate along depth dimension
        return torch.cat([states[i] for i in self.concat], dim=1)


# Example Genotype structure to be used in the normal and reduction cells
class Genotype:
    """
    A placeholder class representing the search architecture results, 
    typically loaded from an external source or derived from a search algorithm.
    """
    def __init__(self, normal, normal_concat, reduce, reduce_concat):
        self.normal = normal  # List of normal operations
        self.normal_concat = normal_concat  # Concatenation indices for normal cell
        self.reduce = reduce  # List of reduction operations
        self.reduce_concat = reduce_concat  # Concatenation indices for reduction cell

# Specific Cell Implementation
class NormalCell(Cell):
    def __init__(self, C_prev_prev, C_prev, C, reduction_prev, genotype):
        super(NormalCell, self).__init__(genotype.normal, C_prev_prev, C_prev, C, reduction=False, reduction_prev=reduction_prev)

class ReductionCell(Cell):
    def __init__(self, C_prev_prev, C_prev, C, reduction_prev, genotype):
        super(ReductionCell, self).__init__(genotype.reduce, C_prev_prev, C_prev, C, reduction=True, reduction_prev=reduction_prev)
        
from torch.autograd import Variable

# Mocking a simple genotype for testing purposes
class SimpleGenotype:
    def __init__(self):
        self.normal = [("ldconv_3x3", 0), ("skip_connect", 1)]
        self.normal_concat = [2, 3]
        self.reduce = [("sep_conv_5x5", 0), ("max_pool_3x3", 1)]
        self.reduce_concat = [2, 3]

# Test function for NormalCell and ReductionCell
def test_cells():
    # Hyperparameters and sizes
    batch_size = 4
    C = 16  # Output channels
    C_prev_prev = 32  # Channels for s0 (input 0)
    C_prev = 32  # Channels for s1 (input 1)
    height = 32
    width = 32
    drop_prob = 0.2

    # Create random input tensors (from previous layers)
    s0 = torch.randn(batch_size, C_prev_prev, height, width)
    s1 = torch.randn(batch_size, C_prev, height, width)

    # Instantiate the genotype
    genotype = SimpleGenotype()

    # Instantiate and test NormalCell
    print("Testing NormalCell...")
    normal_cell = NormalCell(C_prev_prev=C_prev_prev, C_prev=C_prev, C=C, reduction_prev=False, genotype=genotype)
    output_normal = normal_cell(Variable(s0), Variable(s1), drop_prob)
    print(f"NormalCell output shape: {output_normal.shape}")
    assert output_normal.shape == (batch_size, 2 * C, height, width), "NormalCell output shape mismatch"

    # Instantiate and test ReductionCell (downsample the input by stride=2)
    print("Testing ReductionCell...")
    reduction_cell = ReductionCell(C_prev_prev=C_prev_prev, C_prev=C_prev, C=C, reduction_prev=False, genotype=genotype)
    output_reduction = reduction_cell(Variable(s0), Variable(s1), drop_prob)
    print(f"ReductionCell output shape: {output_reduction.shape}")
    assert output_reduction.shape == (batch_size, 2 * C, height // 2, width // 2), "ReductionCell output shape mismatch"

    print("All tests passed.")

# Run the tests
if __name__ == "__main__":
    test_cells()