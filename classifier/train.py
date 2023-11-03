import torch
import torch.nn as nn
import torch.optim as optim
import os

from . import device

def train(
    criterion,
    optimizer: optim.Optimizer,
    model: nn.Module,
    trainloader: torch.utils.data.DataLoader,
    epochs=2,
    retrain=False,
    save_path=os.path.dirname(os.path.abspath(__file__)) + '/data/model.pth',
    verbose=None,
    ):
    """Train the model on the trainloader data and return the model.state_dict()"""
    print(save_path)
    if not retrain:
        if os.path.exists(save_path):
            model.load_state_dict(torch.load(save_path))
            print('The model is already trained. If you want to retrain it, set retrain=True')
            return torch.load(save_path)
        
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            pct_counter =  0
            if verbose and i % int(len(trainloader) * verbose / 100) == int(len(trainloader) * verbose / 100) - 1:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / int(len(trainloader)):.3f} {(pct_counter:=pct_counter+verbose)}%')
                running_loss = 0.0
                    
    print(f'Finished Training! Saving into {save_path}')
    torch.save(model.state_dict(), save_path)
        
    return model.state_dict()
 