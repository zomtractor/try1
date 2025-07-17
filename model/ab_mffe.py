import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F

import cv2 as cv
from skimage.util import img_as_ubyte
from torchvision.transforms import transforms
if __name__ == '__main__':
    input = cv.imread('D:/Program_Files/stable_diffusion_launcher/outputs/txt2img-images/2023-10-29/00000-1800816867.png')
    input = cv.cvtColor(input,cv.COLOR_BGR2RGB)
    x = transforms.ToTensor()(input).unsqueeze(0)
    # x = nn.functional.batch_norm(x, running_mean=None, running_var=None, training=True, momentum=0.1, eps=1e-5)
    print(x)
    fft = torch.fft.fft2(x,norm='ortho')
    fft = torch.fft.fftshift(fft)
    # split
    real, imag = fft.real,fft.imag
    tester = torch.cat((real,imag),dim=1)
    net = nn.Sequential(
        nn.Conv2d(6, 6, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(6, 6, kernel_size=3, padding=1),
    )
    output = net(tester)
    # print(real)
    # print(imag)
    # # print max
    # print(torch.max(real))
    # print(torch.max(imag))
    # # print min
    # print(torch.min(real))
    # print(torch.min(imag))
    outr,outi = torch.chunk(output, 2, dim=1)
    re = torch.fft.ifft2(torch.fft.ifftshift(torch.complex(outr, outi)),norm='ortho')
    # restored_img = img_as_ubyte(re.real/torch.max(re.real))
    # cv2.imshow('1',restored_img)
    print(torch.max(re.imag))
    print(re.real - x)



