from classifier import (CIFAR10_loader, Net, device, test_class_accuracy,
                        test_total_accuracy, train)

# Load data
loader = CIFAR10_loader()
trainloader, testloader, classes = loader.load()

# Create model
net = Net()
net.to(device)
print(f"{device=} is used")

# Choose criterion and optimizer
from torch.nn import CrossEntropyLoss
from torch.optim import SGD

criterion = CrossEntropyLoss()
optimizer = SGD(net.parameters(), lr=0.001, momentum=0.9)

# Train model
net.load_state_dict(
    train(
        criterion, optimizer, net, trainloader,
        epochs=2, retrain=False, verbose=5
        ))

# Check model
dataiter = iter(testloader)
images, labels = next(dataiter)

# Test accuracy
test_total_accuracy(net, testloader)
test_class_accuracy(net, testloader, classes)
