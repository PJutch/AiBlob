from torch import nn
from torchvision import transforms


class Model(nn.Module):
    class View(nn.Module):
        def forward(self, x):
            return x.view(-1, 5 * 12 * 12)

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 5, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            self.View(),
            nn.Linear(5 * 12 * 12, 500),
            nn.ReLU(),
            nn.Linear(500, 10),
            nn.Softmax(dim=0)
        )

    def forward(self, x):
        return self.layers.forward(x)


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.7, 0.7)
])
