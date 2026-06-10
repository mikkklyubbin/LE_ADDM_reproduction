import torch
from torch import nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)

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
            ResidualBlock(in_channels),
            ResidualBlock(in_channels),
            ResidualBlock(in_channels),
            ResidualBlock(in_channels),
        )
        self.down_sample = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2, bias=False)

    def forward(self, x):
        out1 = self.resudials(x)
        out = self.down_sample(out1)
        return out, out1
    
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pads = (0, 0)):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, bias=False, output_padding=pads)
        self.resudials = nn.Sequential(
            ResidualBlock(out_channels),
            ResidualBlock(out_channels),
            ResidualBlock(out_channels),
            ResidualBlock(out_channels),
        )
        

    def forward(self, x, skip = 0):
        out = self.upsample(x)
        out = self.resudials(out + skip)
        return out
    
class Drunet(nn.Module):
    def __init__(self, in_channels=1, channels = [64, 128, 256], shape = (380, 507)):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1, bias=False)
        self.proj2 = nn.Conv2d(channels[0], in_channels, kernel_size=3, padding=1, bias=False)
        channels = channels
        cur = shape
        pads = []
        for i in range(len(channels) - 1):
            pads.append([0,0])
            if cur[0] % 2 != 0:
                pads[-1][0] = 1
            if cur[1] % 2 != 0:
                pads[-1][1] = 1
            cur = (cur[0] // 2, cur[1] // 2)
            pads[-1] = tuple(pads[-1])

        self.enc = nn.ModuleList([EncoderBlock(channels[i], channels[i + 1]) for i in range(len(channels) - 1)])
        self.dec = nn.ModuleList([DecoderBlock(channels[i], channels[i - 1], pads[i - 1]) for i in range(len(channels) - 1, 0, -1)])
        self.midle = nn.Sequential(
            ResidualBlock(channels[-1]),
            ResidualBlock(channels[-1]),
            ResidualBlock(channels[-1]),
            ResidualBlock(channels[-1]),
        )
        self.layers = len(channels) - 1

    def forward(self, x):
        x = self.proj(x)
        enc_outs = []
        for i in range(self.layers):
            x, save = self.enc[i](x)
            enc_outs.append(save)
        x = self.midle(x)
        for i in range(self.layers):
            x = self.dec[i](x, enc_outs[self.layers - 1 - i])
        x = self.proj2(x)
        return x