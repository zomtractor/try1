import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft


class MFFE(nn.Module):  # Multi-Frequency Fusion Enhancement
    def __init__(self, channels):
        super(MFFE, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv1x1 = nn.Conv2d(channels * 2, channels, kernel_size=1)

    def forward(self, x):
        # FFT transform
        freq = torch.fft.fft2(x, norm='ortho')
        freq_real = freq.real
        freq_imag = freq.imag

        # Global Average Pooling at multiple scales
        gap_1_2 = F.adaptive_avg_pool2d(x, (x.size(2) // 2, x.size(3) // 2))
        gap_1_4 = F.adaptive_avg_pool2d(x, (x.size(2) // 4, x.size(3) // 4))
        gap_1_8 = F.adaptive_avg_pool2d(x, (x.size(2) // 8, x.size(3) // 8))

        gap_features = F.interpolate(gap_1_2, size=x.shape[2:], mode='bilinear') + \
                       F.interpolate(gap_1_4, size=x.shape[2:], mode='bilinear') + \
                       F.interpolate(gap_1_8, size=x.shape[2:], mode='bilinear')

        x_out = self.conv1(x)
        x_out = self.leaky_relu(x_out)
        x_out = self.conv2(x_out)

        # IFFT fusion
        freq_complex = torch.complex(freq_real, freq_imag)
        ifft_out = torch.fft.ifft2(freq_complex, norm='ortho').real

        cat = torch.cat([x_out, gap_features + ifft_out], dim=1)
        return self.conv1x1(cat)


class CAB(nn.Module):  # Channel Attention Block
    def __init__(self, channels, reduction=16):
        super(CAB, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.avg_pool(x)
        w = self.fc(w)
        return x * w


class FAB(nn.Module):  # Feature Attention Block
    def __init__(self, channels):
        super(FAB, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(channels)
        self.attn = CAB(channels)

    def forward(self, x):
        res = self.conv1(x)
        res = self.relu(res)
        res = self.conv2(res)
        res = self.bn(res)
        res = self.attn(res)
        return x + res


class ABTB(nn.Module):  # Axial Block (LLFormer-like)
    def __init__(self, dim):
        super(ABTB, self).__init__()
        self.norm = nn.LayerNorm([dim, 1, 1])
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        return residual + x


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class FeatureBlock(nn.Module):
    def __init__(self, channels):
        super(FeatureBlock, self).__init__()
        self.fab1 = FAB(channels)
        self.fab2 = FAB(channels)
        self.cbam = CAB(channels)  # Here CBAM is represented with CAB
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


class UFormerBlock(nn.Module):
    def __init__(self, in_channels=3, base_channels=32):
        super(UFormerBlock, self).__init__()
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
