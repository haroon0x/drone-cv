import torch
import torch.nn as nn
class P2Head(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_classes + 5, 1)
    def forward(self, x):
        return self.conv(x) 