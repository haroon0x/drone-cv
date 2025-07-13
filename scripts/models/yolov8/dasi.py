import torch
import torch.nn as nn
class DASI(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x1, x2, x3):
        a = self.conv(x1)
        b = self.conv(x2)
        c = self.conv(x3)
        s = self.sigmoid(a)
        out = s * b + (1 - s) * c
        return out + x1 