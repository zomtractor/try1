import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

from model.cbam import CBAM
from model.fab import FAB


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


class UBlock(nn.Module):
    def __init__(self, in_channels=3, base_channels=32):
        super(UBlock, self).__init__()
        self.head = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        self.encoder1 = nn.Sequential(DownSample(base_channels, base_channels * 2), FeatureBlock(base_channels * 2))
        self.encoder2 = nn.Sequential(DownSample(base_channels * 2, base_channels * 4), FeatureBlock(base_channels * 4))
        self.encoder3 = nn.Sequential(DownSample(base_channels * 4, base_channels * 8), FeatureBlock(base_channels * 8))
        self.encoder4 = nn.Sequential(DownSample(base_channels * 8, base_channels * 16), FeatureBlock(base_channels * 16))

        self.decoder3 = nn.Sequential(UpSample(base_channels * 16, base_channels * 8), FeatureBlock(base_channels * 8))
        self.decoder2 = nn.Sequential(UpSample(base_channels * 8, base_channels * 4), FeatureBlock(base_channels * 4))
        self.decoder1 = nn.Sequential(UpSample(base_channels * 4, base_channels * 2), FeatureBlock(base_channels * 2))
        self.decoder0 = nn.Sequential(UpSample(base_channels * 2, base_channels), FeatureBlock(base_channels))

        self.tail = nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.head(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        d3 = self.decoder3(e4) + e3
        d2 = self.decoder2(d3) + e2
        d1 = self.decoder1(d2) + e1
        d0 = self.decoder0(d1)

        return self.tail(d0)
