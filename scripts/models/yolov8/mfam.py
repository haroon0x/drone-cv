import torch
import torch.nn as nn
class MFAM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, 7), padding=(0, 3), groups=in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, (7, 1), padding=(3, 0), groups=in_channels)
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, padding=1, groups=in_channels)
        self.conv4 = nn.Conv2d(in_channels, out_channels, 1)
    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.conv2(x)
        y3 = self.conv3(x)
        y4 = self.conv4(x)
        return y1 + y2 + y3 + y4 + x 