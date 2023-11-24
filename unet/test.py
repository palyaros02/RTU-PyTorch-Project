import torch
from model import UNet

x = torch.randn(1, 3, 128, 128)
model = UNet()
y = model(x)
y