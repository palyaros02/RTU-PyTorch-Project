import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from .load_data import CIFAR10_loader
from .model import Net
from .train import train