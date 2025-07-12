import torch
import torch.nn as nn
import torch.nn.functional as F


class MFFE(nn.Module):  # Multi-Frequency Fusion Enhancement
    def __init__(self, channels):
        super(MFFE, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.fconv = nn.Conv2d(channels*2, channels*2, kernel_size=3, padding=1)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        self.relu = nn.ReLU()
        self.conv11 = nn.Conv2d(channels * 2, channels, kernel_size=1)

    def forward(self, x):
        identity = x

        # FFT变换
        fft = torch.fft.fft2(x)
        fft = torch.fft.fftshift(fft)
        fftreal = torch.cat((fft.real, fft.imag), dim=1)
        # 全局平均池化(GAP)
        gap1 = F.adaptive_avg_pool2d(fftreal,(x.size(2)//8,x.size(3)//8))
        gap2 = F.adaptive_avg_pool2d(fftreal,(x.size(2)//4,x.size(3)//4))
        gap3 = F.adaptive_avg_pool2d(fftreal,(x.size(2)//2,x.size(3)//2))
        fout1 = self.relu(self.fconv(gap1))/2
        fout2 = self.relu(self.fconv(F.interpolate(fout1, scale_factor=2, mode='bilinear')+gap2))/2
        fout3 = self.relu(self.fconv(F.interpolate(fout2, scale_factor=2, mode='bilinear')+gap3))/2
        fout = F.interpolate(fout1, scale_factor=8, mode='bilinear')+F.interpolate(fout2,scale_factor=4,mode='bilinear')+F.interpolate(fout3,scale_factor=2,mode='bilinear')
        fout = fout/3
        foutreal, foutimag = torch.chunk(fout, 2, dim=1)
        fout = torch.fft.ifft2(torch.complex(foutreal, foutimag)).abs()
        # 卷积路径
        x = self.conv(x)
        x = self.leaky_relu(x)
        x = self.conv(x)
        x = x + identity

        out = torch.cat((x, fout), 1)
        out = self.conv11(out)
        return out

if __name__ == '__main__':
    # 示例使用
    x = torch.randn(4, 64, 256, 256)  # batch_size=4, channels=64, height=32, width=32
    print(x.shape)
    mfee = MFFE(64)
    out = mfee(x)
    print(out.shape)