import torch
import torch.nn as nn
class IEMA(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.conv2 = nn.Conv2d(channels, channels, (1, 5), padding=(0, 2), groups=channels)
        self.conv3 = nn.Conv2d(channels, channels, (5, 1), padding=(2, 0), groups=channels)
        self.identity = nn.Identity()
    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.conv2(x)
        y3 = self.conv3(x)
        y4 = self.identity(x)
        return y1 + y2 + y3 + y4 