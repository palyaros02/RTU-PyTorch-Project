import os
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim

from . import device

def train(
    criterion,
    optimizer: optim.Optimizer,
    model: nn.Module,
    trainloader: torch.utils.data.DataLoader,
    epochs=2,
    retrain=False,
    save_path='./classifier/data/model.pth',
    verbose=None,
    ) -> Dict[str, Any]:

    """Train the model on the trainloader data and return the model.state_dict()"""

    if not retrain:
        if os.path.exists(save_path):
            model.load_state_dict(torch.load(save_path))
            print('The model is already trained. If you want to retrain it, set retrain=True')
            return torch.load(save_path)
    elif os.path.exists(save_path):
        os.remove(save_path)

    for epoch in range(epochs):
        epoch_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if verbose and i % 2000 == 1999:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {epoch_loss / 2000:.3f}')
                epoch_loss = 0.0

    print(f'Finished Training! Saving into {save_path}')
    torch.save(model.state_dict(), save_path)

    return model.state_dict()