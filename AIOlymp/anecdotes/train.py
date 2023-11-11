import torch

from model import Model, navec
from torch import optim
from torch import nn
from torch.utils import data


class Dataset(data.Dataset):
    def __init__(self):
        print('Loading data...')
        with open('samples4.txt') as f:
            self.samples = [torch.Tensor([int(token) for token in sample.split()]) for sample in f]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        return self.samples[item][:-1], self.samples[item][-1]


model = Model()
opt = optim.SGD(params=model.parameters(), lr=1e-5)
metric = nn.MSELoss()

train_data, test_data = data.random_split(Dataset(), (0.9, 0.1))

train_loader = data.DataLoader(train_data, batch_size=1024, shuffle=True)
test_loader = data.DataLoader(train_data, batch_size=1024, shuffle=True)


def train():
    for epoch in range(3):
        print(f'Epoch {epoch}')
        for batch, (x, goal_y) in enumerate(train_loader):
            model.zero_grad()
            y = model(x)
            loss: torch.Tensor = metric(y, model.embedding(goal_y.type(torch.int)))
            loss.backward()
            opt.step()

            if batch % 100 == 0:
                print(f'Processed batch #{batch}. Loss={loss}')

        test_loss = 0
        with torch.no_grad():
            for batch, (x, goal_y) in enumerate(test_loader):
                y = model(x)
                test_loss += metric(y, model.embedding(goal_y.type(torch.int))).item()
        print(f'Test loss={test_loss/len(test_loader)}')


train()
torch.save(model, 'model4.pth')
