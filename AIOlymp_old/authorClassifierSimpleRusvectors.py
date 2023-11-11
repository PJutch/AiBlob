import spacy
import torch
import torch.utils.data
import pandas

device = (
    'cuda' if torch.cuda.is_available()
    else 'mps' if torch.backends.mps.is_available()
    else 'cpu'
)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(600, 600),
            torch.nn.ReLU(),
            torch.nn.Linear(600, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)


def parse_tensor(s):
    s = s.strip().strip('[]').strip()
    return torch.FloatTensor([float(x) for x in s.split(' ') if x])


class AuthorDataset(torch.utils.data.Dataset):
    def __init__(self):
        data = pandas.read_csv('rucode-7.0/train_vectors.csv')

        self.labels = torch.FloatTensor(data['label'] == 'ai')
        self.labels = self.labels.reshape((self.labels.size()[0], 1))

        contexts = torch.stack([parse_tensor(context) for context in data['context']])
        answers = torch.stack([parse_tensor(answer) for answer in data['answer']])
        self.features = torch.cat([contexts, answers], dim=1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        return self.features[i], self.labels[i]


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (vectors, labels) in enumerate(dataloader):
        pred = model(vectors)
        loss = loss_fn(pred, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(vectors)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0
    tp = 0
    fp = 0
    fn = 0
    tn = 0

    with torch.no_grad():
        for vectors, labels in dataloader:
            pred = model(vectors)
            test_loss += loss_fn(pred, labels).item()
            tp += ((pred > 0.5) & (labels > 0.5)).type(torch.double).sum().item()
            fp += ((pred > 0.5) & (labels <= 0.5)).type(torch.double).sum().item()
            fn += ((pred <= 0.5) & (labels > 0.5)).type(torch.double).sum().item()
            tn += ((pred <= 0.5) & (labels <= 0.5)).type(torch.double).sum().item()

    test_loss /= num_batches
    accuracy = (tp + tn) / size
    recall = tp / (tp + fn + 1e-5)
    precision = tp / (tp + fp + 1e-5)
    f1 = 2 / (1 / (recall + 1e-5) + 1 / (precision + 1e-5))
    print(f"Test Error:\n Accuracy: {(100 * accuracy):>0.1f}%, Avg loss: {test_loss:>8f} F1: {100 * f1:>0.1f} "
          f"Recall: {(100 * recall):>0.1f}% Precision: {(100 * precision):>0.1f}%\n")

    return f1


def smooth_f1(y_true, y_pred):
    tp = torch.sum(y_true * y_pred, dim=0)
    tn = torch.sum((1 - y_true) * (1 - y_pred), dim=0)
    fp = torch.sum((1 - y_true) * y_pred, dim=0)
    fn = torch.sum(y_true * (1 - y_pred), dim=0)

    p = tp / (tp + fp + 1e-5)
    r = tp / (tp + fn + 1e-5)

    f1 = 2 * p * r / (p + r + 1e-5)
    f1 = torch.where(torch.isnan(f1), torch.zeros_like(f1), f1)
    return torch.mean(f1)


def f1_loss(y_true, y_pred):
    return 1 - smooth_f1(y_true, y_pred)


if __name__ == '__main__':
    model = Model().to(device)

    print('Loading data...')
    full_data = AuthorDataset()
    print('Data loaded!')

    train_size = int(0.8 * len(full_data))
    test_size = len(full_data) - train_size
    train_data, test_data = torch.utils.data.random_split(full_data, [train_size, test_size])

    batch_size = 64
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

    loss_function = f1_loss
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    print('Training model...')
    epochs = 200
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_function, optimizer)
        test_loop(test_dataloader, model, loss_function)

    print("Model trained!")
    torch.save(model, 'authorClassifierSimpleRusvectors.pth')
