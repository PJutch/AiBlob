from __future__ import print_function, division

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.io.image
from torchvision import datasets
from PIL import Image


def imshow(img):
    img = img / 2+0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def get_dataset(data_dir, data_transforms):
    image_dataset = datasets.ImageFolder(data_dir, data_transforms)
    dataset_size = len(image_dataset)
    classes = image_dataset.classes
    dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=4, num_workers=4)
    return dataloader, classes, dataset_size


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 13 * 13, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 84)
        self.fc4 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 13 * 13)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


dir = os.path.abspath(os.curdir)
data_dir = os.path.join(dir, "faces/test")

data_transforms = transforms.Compose([
    transforms.RandomResizedCrop(64),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

if __name__ == '__main__':
    PATH = os.path.join(dir, "face_model.pth")
    net = Net()
    net.load_state_dict(torch.load(PATH))

    print('filename,label')
    with torch.no_grad():
        for i in range(11582):
            if os.path.isfile(f'faces/test/unclassified/ {i + 1}.png'):
                image = Image.open(f'faces/test/unclassified/ {i + 1}.png')
                transformed = data_transforms(image)

                outputs = net(torch.reshape(transformed, (1, 3, 32, 32)))
                _, predicted = torch.max(outputs.data, 1)

                print(f'{i + 1}.png,{int(predicted)}')
