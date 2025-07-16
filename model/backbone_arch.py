import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

from model import FAB, CBAM, MFFE


class FeatureBlock(nn.Module):
    def __init__(self, channels):
        super(FeatureBlock, self).__init__()
        self.fab1 = FAB(channels)
        self.fab2 = FAB(channels)
        self.cbam = CBAM(channels)  # Here CBAM is represented with CAB
        self.mffe = MFFE(channels)

    def forward(self, x):
        res = self.fab1(x)
        res = self.fab2(res)
        res = self.cbam(res)
        res = self.mffe(res)
        return x + res


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSample, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSample, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.up(x)
        return self.conv(x)


class WeightedConnect(nn.Module):
    def __init__(self, channels, height, width,requires_grad=True):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(1, channels, height, width))
        if not requires_grad:
            self.weights.requires_grad = False
        self.activation = nn.Sigmoid()
    def forward(self, x):
        return x * self.activation(self.weights)

class UBlock(nn.Module):
    def __init__(self, in_channels=3, base_channels=32,in_height=512,in_width=512,weight_connect=True):
        super(UBlock, self).__init__()
        self.head = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        self.encoder1 = nn.Sequential(FeatureBlock(base_channels),DownSample(base_channels, base_channels * 2))
        self.encoder2 = nn.Sequential(FeatureBlock(base_channels * 2),DownSample(base_channels * 2, base_channels * 4))
        self.encoder3 = nn.Sequential(FeatureBlock(base_channels * 4),DownSample(base_channels * 4, base_channels * 8))
        self.encoder4 = nn.Sequential(FeatureBlock(base_channels * 8),DownSample(base_channels * 8, base_channels * 16))
        self.bottleneck =nn.Sequential(FeatureBlock(base_channels * 16),UpSample(base_channels * 16, base_channels * 8))
        self.decoder4 = nn.Sequential( FeatureBlock(base_channels * 8),UpSample(base_channels * 8, base_channels * 4))
        self.decoder3 = nn.Sequential( FeatureBlock(base_channels * 4),UpSample(base_channels * 4, base_channels * 2))
        self.decoder2 = nn.Sequential( FeatureBlock(base_channels * 2),UpSample(base_channels * 2, base_channels * 1))
        self.decoder1 = nn.Sequential(FeatureBlock(base_channels))
        c,h,w=in_channels,in_height,in_width
        self.weight0 = WeightedConnect(c,h,w,requires_grad=weight_connect)
        self.weight1 = WeightedConnect(base_channels,h,w,requires_grad=weight_connect)
        self.weight2 = WeightedConnect(base_channels*2,h//2,w//2,requires_grad=weight_connect)
        self.weight3 = WeightedConnect(base_channels*4,h//4,w//4,requires_grad=weight_connect)
        self.weight4 = WeightedConnect(base_channels*8,h//8,w//8,requires_grad=weight_connect)

        self.tail = nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        h = self.head(x)
        e1 = self.encoder1(h)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        b = self.bottleneck(e4)
        d4 = self.decoder4(self.weight4(b))
        d3 = self.decoder3(self.weight3(d4))
        d2 = self.decoder2(self.weight2(d3))
        d1 = self.decoder1(self.weight1(d2))
        t = self.tail(d1)
        return self.weight0(t)


