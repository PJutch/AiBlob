import torch
from torch import nn
from navec import Navec
from slovnet.model.emb import NavecEmbedding

navec = Navec.load('navec_hudlit_v1_12B_500K_300d_100q.tar')


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = NavecEmbedding(navec)
        self.layers = nn.Sequential(
            nn.Linear(10 * 300, 1000),
            nn.ReLU(),
            nn.Linear(1000, 300)
        )

        with torch.no_grad():
            for param in self.layers.parameters():
                param.uniform_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x.type(torch.int))
        return self.layers(embedded.view(-1, 10 * 300))
