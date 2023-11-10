if __name__ == '__main__':
    from torch.optim import SGD
    from torch.nn import CrossEntropyLoss
    import torch

    import matplotlib.pyplot as plt
    import numpy as np

    from classifier import CIFAR10_loader
    from classifier import Net
    from classifier import train
    from classifier import device

    loader = CIFAR10_loader()
    trainloader, testloader, classes = loader.load()


    net = Net()
    net.to(device)


    criterion = CrossEntropyLoss()
    optimizer = SGD(net.parameters(), lr=0.001, momentum=0.9)


    net.load_state_dict(
        train(
            criterion, optimizer, net, trainloader,
            epochs=2, retrain=False, verbose=5
            ))

    def imshow(img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()


    dataiter = iter(testloader)
    images, labels = next(dataiter)
    # imshow(torchvision.utils.make_grid(images))
    print("GroundTruth: ", " ".join("%5s" % classes[labels[j]] for j in range(4)))
    _, predicted = torch.max(net(images.to(device)), 1)
    print("Predicted: ", " ".join("%5s" % classes[predicted[j]]
                                for j in range(4)))

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # again no gradients needed
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1


    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')