import torch
from torch.utils.data.dataset import random_split
from torch import nn
from torchvision.datasets import MNIST
from model import Model, transform

data = MNIST(root='./data', train=False, download=True, transform=transform)
loader = torch.utils.data.DataLoader(data, batch_size=64, shuffle=True)

model = torch.load('model_conv.pth')

metric = nn.CrossEntropyLoss()

error = 0
with torch.no_grad():
    for vectors, labels in loader:
        error += metric(model(vectors), labels).item()
error /= len(loader)

print(f'Error: {error}')
