import torch
from torch import nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += residual
        return out
    
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.resudials = nn.Sequential(
            ResidualBlock(in_channels, in_channels),
            ResidualBlock(in_channels, in_channels),
            ResidualBlock(in_channels, in_channels),
            ResidualBlock(in_channels, in_channels),
        )
        self.down_sample = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2, padding=1, bias=False)

    def forward(self, x):
        out = self.resudials(x)
        out = self.down_sample(out)
        return out
    
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=1, bias=False)
        self.resudials = nn.Sequential(
            ResidualBlock(out_channels, out_channels),
            ResidualBlock(out_channels, out_channels),
            ResidualBlock(out_channels, out_channels),
            ResidualBlock(out_channels, out_channels),
        )
        

    def forward(self, x):
        out = self.upsample(x)
        out = self.resudials(out)
        return out
    
class Drunet(nn.Module):
