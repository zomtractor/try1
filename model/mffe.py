import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F

class MFFE(nn.Module):  # Multi-Frequency Fusion Enhancement
    def __init__(self, channels):
        super(MFFE, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.fconv = nn.Conv2d(channels * 2, channels * 2, kernel_size=3, padding=1)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        self.relu = nn.ReLU()
        self.conv11 = nn.Conv2d(channels * 2, channels, kernel_size=1)

    def forward(self, x):
        identity = x
        print(torch.isnan(x).any())
        # === 1. 频域变换 ===
        fft = torch.fft.fft2(x)
        fft = torch.fft.fftshift(fft)
        fftreal = torch.cat((fft.real, fft.imag), dim=1)
        print(torch.isnan(fft).any())
        # === 2. clone一份用于GAP路径并标准化 ===
        gap_input = fftreal.clone()
        mean = gap_input.mean(dim=(2, 3), keepdim=True)
        std = gap_input.std(dim=(2, 3), keepdim=True).nan_to_num(0) + 1e-6  # 防止除0
        gap_input = (gap_input - mean) / std

        # === 3. 多尺度GAP + fconv 处理 ===
        gap1 = F.adaptive_avg_pool2d(gap_input, (x.size(2) // 8, x.size(3) // 8))
        gap2 = F.adaptive_avg_pool2d(gap_input, (x.size(2) // 4, x.size(3) // 4))
        gap3 = F.adaptive_avg_pool2d(gap_input, (x.size(2) // 2, x.size(3) // 2))

        fout1 = self.relu(self.fconv(gap1)) / 2
        fout2 = self.relu(self.fconv(F.interpolate(fout1, scale_factor=2, mode='bilinear') + gap2)) / 2
        fout3 = self.relu(self.fconv(F.interpolate(fout2, scale_factor=2, mode='bilinear') + gap3)) / 2

        fout = (
            F.interpolate(fout1, scale_factor=8, mode='bilinear') +
            F.interpolate(fout2, scale_factor=4, mode='bilinear') +
            F.interpolate(fout3, scale_factor=2, mode='bilinear')
        ) / 3
        print(torch.isnan(fout).any())
        # === 4. 回到复数并IFFT ===
        foutreal, foutimag = torch.chunk(fout, 2, dim=1)
        fout_complex = torch.complex(foutreal, foutimag)
        fout = torch.fft.ifft2(fout_complex).abs()

        # === 5. 空间卷积主分支 ===
        x = self.conv(x)
        x = self.leaky_relu(x)
        x = self.conv(x)
        x = x + identity

        # === 6. 融合空间与频域信息 ===
        out = torch.cat((x, fout), dim=1)
        out = self.conv11(out)
        return out



if __name__ == '__main__':
    # 示例使用
    x = torch.randn(4, 64, 256, 256)  # batch_size=4, channels=64, height=32, width=32
    print(x.shape)
    mfee = MFFE(64)
    out = mfee(x)
    print(out.shape)