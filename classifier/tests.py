"""
This module provides tests for classifier.
"""
import matplotlib.pyplot as plt
import numpy as np
import torch

from . import device


def imshow(img):
    """
    Function to display an image.
    The image is normalized and displayed using matplotlib.

    Parameters:
    img (Tensor): The image tensor
    """
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def test_total_accuracy(net, testloader):
    """
    Function to test the overall accuracy of the network.

    Parameters:
    net (torch.nn.Module): The network to test
    testloader (torch.utils.data.DataLoader): DataLoader for the test data
    classes (list): List of classes
    """
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')


def test_class_accuracy(net, testloader, classes):
    """
    Function to test the accuracy of the network per class.

    Parameters:
    net (torch.nn.Module): The network to test
    testloader (torch.utils.data.DataLoader): DataLoader for the test data
    classes (list): List of classes
    """
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)

            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')