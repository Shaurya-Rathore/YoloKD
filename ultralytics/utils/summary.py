import torch
import torch.nn as nn
from torchsummary import summary
from ..nn.modules.model_search import Detect

# Assuming Detect class is already defined, as provided in your code
class Conv(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride=1, padding=1):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class DFL(nn.Module):
    def __init__(self, reg_max):
        super(DFL, self).__init__()
        self.fc = nn.Linear(reg_max, reg_max)

    def forward(self, x):
        return self.fc(x)

# Define number of classes and channels for testing
nc = 80  # Number of classes
ch = [256, 512, 1024]  # List of channels for different detection levels

# Instantiate the Detect class
detect_model = Detect(nc=nc, ch=ch)

# Move model to the appropriate device (CPU in this case)
device = torch.device("cpu")
detect_model.to(device)

# Print summary of the model to see the number of parameters
# Use a dummy input size for each detection level
input_size = [(256, 150, 150), (512, 75, 75), (1024, 38, 38)]  # List of input feature map sizes
summary(detect_model, input_size=input_size, device=str(device))
