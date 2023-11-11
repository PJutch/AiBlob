import torch
from torch.utils.data.dataset import random_split
from torch import nn
from torch import optim
from torchvision.datasets import MNIST
from model import Model, transform


data = MNIST(root='./data', train=True, download=True, transform=transform)
train_data, test_data = random_split(data, (0.7, 0.3))

train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)

model = Model()
opt = optim.SGD(model.parameters(), lr=.3)
metric = nn.CrossEntropyLoss()


def train():
    for epoch in range(10):
        print(f'Epoch: {epoch}')
        for batch, (vectors, labels) in enumerate(train_loader):
            model.zero_grad()
            preds = model(vectors)
            error: torch.Tensor = metric(preds, labels)
            error.backward()
            opt.step()
            if batch % 100 == 0:
                print(f'Batch #{batch}, avg error {torch.mean(error)}')

        test_error = 0
        with torch.no_grad():
            for vectors, labels in test_loader:
                test_error += metric(model(vectors), labels).item()
        test_error /= len(test_loader)
        print(f'Test error: {test_error}\n')


train()

torch.save(model, 'model_conv.pth')
