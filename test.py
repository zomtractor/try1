import torch

from model import UBlock

if __name__ == '__main__':
    # Example usage
    x = torch.randn(4, 3, 256,256)  # batch_size=4, channels=3, height=256, width=256
    model = UBlock(in_channels=3, base_channels=8,in_height=256,in_width=256,weight_connect=True)
    out = model(x)
    print("UBlock output shape:", out.shape)  # Should be (4, 3, 256, 256)


