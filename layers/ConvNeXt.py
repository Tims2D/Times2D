import torch
import torch.nn as nn

class ConvNeXtBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvNeXtBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=in_channels),
            nn.LayerNorm([out_channels, out_channels], eps=1e-6),
            nn.Conv2d(out_channels, out_channels, kernel_size=1)
        )
        self.gelu = nn.GELU()
        self.pointwise = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.gelu(x)
        x = self.pointwise(x)
        return x
