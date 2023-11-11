import torch
import torch.utils.data
import torchmetrics
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
            torch.nn.Linear(925, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 1),
            # torch.nn.ReLU()
        )

    def forward(self, x):
        return self.layers(x)


def parse_tensor(s):
    s = s.strip().strip('[]').strip()
    return torch.FloatTensor([float(x) for x in s.split(' ') if x])[:100]


def parse_bool(s):
    return torch.FloatTensor([float(bool(s))])


def get_features(data):
    text_columns = ['employer_name', 'experience_name', 'key_skills_name',
                    'specializations_profarea_name', 'professional_roles_name',
                    'lemmaized_wo_stopwords_raw_description', 'lemmaized_wo_stopwords_raw_branded_description',
                    'name_clean', 'employer_industries']
    for column in data.columns:
        data[column] = data[column].apply(parse_tensor if column in text_columns else parse_bool)
    return torch.cat([torch.stack(tuple(v for v in data[column])) for column in data.columns], dim=1)


class AuthorDataset(torch.utils.data.Dataset):
    def __init__(self):
        data = pandas.read_csv('train_vectors.csv')
        data = data.drop('Unnamed: 0', axis=1)

        self.targets = torch.FloatTensor(data['salary_mean_net'])
        self.targets = self.targets.reshape((self.targets.size()[0], 1))
        data = data.drop('salary_mean_net', axis=1)

        # mean = torch.mean(self.targets)
        # std = torch.std(self.targets)
        # print(f'{mean=} {std=}')
        # self.targets = (self.targets - mean) / std
        # self.target = torch.nn.functional.normalize(self.targets)

        self.features = get_features(data)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        return self.features[i], self.targets[i]


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
    num_batches = len(dataloader)
    test_loss = 0

    with torch.no_grad():
        for vectors, labels in dataloader:
            pred = model(vectors)
            test_loss += loss_fn(pred, labels).item()

    test_loss /= num_batches
    print(f"Test Error:Avg loss: {test_loss:>8f}\n")


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

    loss_function = torchmetrics.MeanAbsolutePercentageError()
    optimizer = torch.optim.SGD(model.parameters(), lr=150)

    print('Training model...')
    epochs = 100
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_function, optimizer)
        test_loop(test_dataloader, model, loss_function)

    print("Model trained!")
    torch.save(model, 'model_denorm9.pth')
