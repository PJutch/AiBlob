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
from torchvision import datasets


def get_dataset(data_dir, data_transforms ):
    image_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms['train'])

    dataset_size = len(image_dataset)
    train_size = int(0.8 * dataset_size)
    test_size = dataset_size - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(image_dataset, [train_size, test_size])

    classes = image_dataset.classes

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=4)

    return train_dataloader, test_dataloader, classes, train_size, test_size


def imshow(img):
    img = img / 2+0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 13 * 13, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 84)
        self.fc5 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 13 * 13)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc5(x)
        return x


dir = os.path.abspath(os.curdir)
data_dir = os.path.join(dir, "faces/")

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(64),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    'test': transforms.Compose([
        transforms.RandomResizedCrop(64),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
}


if __name__ == '__main__':
    trainloader, testloader, classes, train_size, test_size = get_dataset(data_dir, data_transforms)

    dataiter = iter(trainloader)
    images, labels = next(dataiter)
    # images, labels = dataiter.next()

    imshow(torchvision.utils.make_grid(images))
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

    net = Net()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    device = torch.device("cpu")
    for epoch in range(11):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 15 == 14:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 15))
                running_loss = 0.0
    print('Finished Training')

    PATH = os.path.join(dir, "face_model.pth")
    torch.save(net.state_dict(), PATH)

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)

            for printdata in list(zip(predicted, labels, outputs)):
                printclass = [classes[int(printdata[0])], classes[int(printdata[1])]]
                print('Predict class - {0}, real class - {1}, probability ({2},{3}) - {4}'.format(printclass[0],
                                                                                                  printclass[1],
                                                                                                  classes[0],
                                                                                                  classes[1],
                                                                                                  printdata[2]))

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            imshow(torchvision.utils.make_grid(images))
            # print('GroundTruth: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))
    print('Accuracy of the network on the', test_size, 'test images: %d %%' % (
            100 * correct / total))
