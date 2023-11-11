import spacy
import torch

nlp = spacy.load('en_core_web_md')

device = (
    'cuda' if torch.cuda.is_available()
    else 'mps' if torch.backends.mps.is_available()
    else 'cpu'
)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(300, 300),
            torch.nn.ReLU(),
            torch.nn.Linear(300, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)
