import os

import torch
import torchvision
import torchvision.transforms as transforms

from . import device

class CIFAR10_loader:
    def __init__(self, batch_size=4):
        self.batch_size = batch_size
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer',
                        'dog', 'frog', 'horse', 'ship', 'truck')
        self.PATH = os.path.dirname(os.path.abspath(__file__)) + '/data'

    def load(self):
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), 
                                 (0.5, 0.5, 0.5))])

        trainset = torchvision.datasets.CIFAR10(
            root=self.PATH, train=True,
            download=True, transform=transform
            )
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=self.batch_size,
            shuffle=True, num_workers=1
            )        

        testset = torchvision.datasets.CIFAR10(
            root=self.PATH, train=False,
            download=True, transform=transform
            )
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=self.batch_size,
            shuffle=False, num_workers=1
            )
        
        return trainloader, testloader, self.classes