from sentiment_classifier import nlp, device, Model
from sentiment_dataset import SentimentDataset

import torch.utils.data
import torch.nn


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
    test_loss, correct = 0, 0

    with torch.no_grad():
        for vectors, labels in dataloader:
            pred = model(vectors)
            test_loss += loss_fn(pred, labels).item()
            correct += ((pred > 0.5) & (labels > 0.5) | (pred < 0.5) & (labels < 0.5)).type(torch.double).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


if __name__ == '__main__':
    model = Model().to(device)

    print('Loading data...')
    train_data, test_data = SentimentDataset.load()
    print('Data loaded!')

    batch_size = 64
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

    loss_function = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    print('Training model...')
    epochs = 200
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_function, optimizer)
        test_loop(test_dataloader, model, loss_function)
    print("Model trained!")

    print('Saving model...')
    torch.save(model, 'sentiment_classifier.pth')
    print('Model saved!')
