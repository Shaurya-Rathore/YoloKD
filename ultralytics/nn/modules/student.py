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

        # Preprocessing step for s0 (previous-previous state)
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)

        # Preprocessing step for s1 (previous state)
        # Dynamically adjust the number of channels of s1 to match C
        if C_prev != C:
            self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        else:
            self.preprocess1 = nn.Identity()  # If channels match, no need to change

        # Compile operations based on the genotype
        self._compile(C, genotype, reduction)

    def _compile(self, C, genotype, reduction):
        self._ops = nn.ModuleList()
        self.indices = []

        # Set operation names and indices based on the cell type
        op_names, indices = (genotype.reduce, genotype.reduce_concat) if reduction else (genotype.normal, genotype.normal_concat)
        self.concat = indices

        for name, index in op_names:
            # Set stride to 2 for reduction operations, otherwise 1
            stride = 2 if reduction and index < 2 else 1
            op = OPS[name](C, stride, True)
            self._ops.append(op)
            self.indices.append(index)

    def forward(self, s0, s1, drop_prob):
        # Preprocess s0 and s1 to match the expected input channels
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        # Debugging output for preprocessing
        print(f"After preprocessing: s0 shape: {s0.shape}, s1 shape: {s1.shape}")

        # Ensure that both s0 and s1 have the same number of channels before proceeding
        if s0.shape[1] != s1.shape[1]:
            raise ValueError(f"Channel mismatch after preprocessing: s0 has {s0.shape[1]} channels, s1 has {s1.shape[1]} channels")

        # Track intermediate states for concatenation
        states = [s0, s1]

        # Perform operations and collect new states
        for i in range(len(self._ops)):
            h = states[self.indices[i]]
            print(f"Before operation {i} ({type(self._ops[i]).__name__}): input shape: {h.shape}")

            h = self._ops[i](h)

            # Debugging output for each operation
            print(f"After operation {i} ({type(self._ops[i]).__name__}): output shape: {h.shape}")

            # Apply drop path during training if drop_prob is set
            if self.training and drop_prob > 0.:
                if not isinstance(self._ops[i], Identity):
                    h = drop_path(h, drop_prob)

            states.append(h)

        # Concatenate selected states and return
        concatenated = torch.cat([states[i] for i in self.concat], dim=1)

        # Debugging output for concatenation
        print(f"After concatenation: concatenated shape: {concatenated.shape}")

        # Verify the channel count after concatenation to ensure correctness
        expected_channels = sum([states[i].shape[1] for i in self.concat])
        if concatenated.shape[1] != expected_channels:
            raise ValueError(f"Channel mismatch after concatenation: expected {expected_channels} channels, but got {concatenated.shape[1]} channels")

        return concatenated

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
        super(NormalCell, self).__init__(genotype, C_prev_prev, C_prev, C, reduction=False, reduction_prev=reduction_prev)

class ReductionCell(Cell):
    def __init__(self, C_prev_prev, C_prev, C, reduction_prev, genotype):
        super(ReductionCell, self).__init__(genotype, C_prev_prev, C_prev, C, reduction=True, reduction_prev=reduction_prev)

# Mocking a simple genotype for testing purposes
class SimpleGenotype:
    def __init__(self):
        # Operations for normal cell
        self.normal = [("lde_conv_3x3", 0), ("skip_connect", 1)]
        
        # Indices of intermediate nodes to concatenate for the normal cell
        self.normal_concat = [2, 3]  # Example concat indices
        
        # Operations for reduction cell
        self.reduce = [("sep_conv_5x5", 0), ("max_pool_3x3", 1)]
        
        # Indices of intermediate nodes to concatenate for the reduction cell
        self.reduce_concat = [2, 3]  # Example concat indices
        
class CombinedCellStructure(nn.Module):
    def __init__(self, C_in, num_classes, num_cells, genotype, reduction_indices, C):
        super(CombinedCellStructure, self).__init__()

        self.num_cells = num_cells
        self.reduction_indices = reduction_indices
        self.cells = nn.ModuleList()

        C_prev_prev, C_prev = C_in, C
        reduction_prev = False

        for i in range(num_cells):
            if i in reduction_indices:
                # Use a reduction cell to downsample and increase channels
                cell = ReductionCell(C_prev_prev, C_prev, C, reduction_prev, genotype)
                reduction_prev = True
                C_prev_prev, C_prev = C_prev, C  # Update channels after reduction
            else:
                # Use a normal cell
                cell = NormalCell(C_prev_prev, C_prev, C, reduction_prev, genotype)
                reduction_prev = False
                C_prev_prev, C_prev = C_prev, C

            self.cells.append(cell)

            # Double `C` after reduction cells for next normal cell
            if reduction_prev:
                C *= 2

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, x, drop_prob):
        s0 = s1 = x

        # Iterate through each cell, apply the forward pass
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, drop_prob)
            print(f"After Cell {i}: s0 channels: {s0.shape[1]}, s1 channels: {s1.shape[1]}")  # Debugging statement

        # Apply global pooling and classifier
        out = self.global_pooling(s1)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)

        return logits

import unittest
from torch.autograd import Variable

class TestCombinedCellStructure(unittest.TestCase):
    
    def setUp(self):
        # Parameters for testing
        self.batch_size = 2
        self.C_in = 16  # Input channels
        self.height, self.width = 32, 32  # Input spatial size
        self.num_classes = 10  # Number of output classes
        self.num_cells = 6  # Total number of cells (normal + reduction)
        self.reduction_indices = [2, 4]  # Reduction cells at these indices
        self.initial_C = 16  # Initial output channels after the first cell
        
        # Instantiate the SimpleGenotype (mock genotype for testing)
        self.genotype = SimpleGenotype()
        
        # Instantiate the CombinedCellStructure
        self.model = CombinedCellStructure(C_in=self.C_in, 
                                           num_classes=self.num_classes, 
                                           num_cells=self.num_cells, 
                                           genotype=self.genotype, 
                                           reduction_indices=self.reduction_indices, 
                                           C=self.initial_C)
        
    def test_forward_pass(self):
        """
        Test that the forward pass runs correctly and returns the right output shape.
        """
        # Create a random input tensor
        x = torch.randn(self.batch_size, self.C_in, self.height, self.width)
        
        # Run the model forward pass with a dropout probability
        drop_prob = 0.2
        logits = self.model(Variable(x), drop_prob)
        
        # Assert that the output has the correct shape
        self.assertEqual(logits.shape, (self.batch_size, self.num_classes), 
                         "Output shape mismatch in forward pass")
    
    def test_reduction_cells(self):
        """
        Test that reduction cells are correctly placed and that spatial dimensions are halved.
        """
        # Create a random input tensor
        x = torch.randn(self.batch_size, self.C_in, self.height, self.width)
        
        # Run the model forward pass and inspect intermediate outputs
        drop_prob = 0.0
        s0 = s1 = Variable(x)
        
        # Check cell outputs
        for i, cell in enumerate(self.model.cells):
            s0, s1 = s1, cell(s0, s1, drop_prob)
            
            if i in self.reduction_indices:
                # Assert that the spatial dimensions are halved after reduction cells
                self.assertEqual(s1.shape[2], self.height // 2, 
                                 f"Reduction cell at index {i} did not halve the height")
                self.assertEqual(s1.shape[3], self.width // 2, 
                                 f"Reduction cell at index {i} did not halve the width")
                
                # Update height and width after reduction
                self.height //= 2
                self.width //= 2
    
    def test_output_channels(self):
        """
        Test that the number of output channels is doubled after reduction cells.
        """
        # Create a random input tensor
        x = torch.randn(self.batch_size, self.C_in, self.height, self.width)
        
        # Run the model forward pass and inspect intermediate outputs
        drop_prob = 0.0
        s0 = s1 = Variable(x)
        C_prev = self.initial_C
        
        for i, cell in enumerate(self.model.cells):
            s0, s1 = s1, cell(s0, s1, drop_prob)
            
            if i in self.reduction_indices:
                # Assert that the number of channels doubled after reduction cells
                self.assertEqual(s1.shape[1], 2 * C_prev, 
                                 f"Reduction cell at index {i} did not double the channels")
                C_prev = 2 * C_prev  # Update the channel count after reduction
            else:
                self.assertEqual(s1.shape[1], C_prev, 
                                 f"Normal cell at index {i} changed the number of channels unexpectedly")
    
    def test_edge_case_small_input(self):
        """
        Test the model with smaller input dimensions to ensure it handles them correctly.
        """
        small_height, small_width = 16, 16  # Smaller spatial dimensions
        x = torch.randn(self.batch_size, self.C_in, small_height, small_width)
        
        # Run the model forward pass with a small input size
        drop_prob = 0.2
        logits = self.model(Variable(x), drop_prob)
        
        # Assert that the output has the correct shape
        self.assertEqual(logits.shape, (self.batch_size, self.num_classes), 
                         "Output shape mismatch with small input")
    
    def test_dropout_handling(self):
        """
        Test that the dropout probability is applied correctly and does not raise errors.
        """
        x = torch.randn(self.batch_size, self.C_in, self.height, self.width)
        
        # Run the model forward pass with varying dropout probabilities
        for drop_prob in [0.0, 0.2, 0.5, 0.8]:
            try:
                logits = self.model(Variable(x), drop_prob)
            except Exception as e:
                self.fail(f"Forward pass failed with drop_prob={drop_prob}: {e}")

if __name__ == "__main__":
    unittest.main()