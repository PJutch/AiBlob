from sentiment_classifier import nlp, device, Model
import torch.utils.data


if __name__ == '__main__':
    print('Loading model...')
    model = torch.load('sentiment_classifier.pth').to(device)
    print('Model loaded!')

    while True:
        review = input('Enter review: ')

        vector = torch.FloatTensor(nlp(review).vector)
        pred = model(vector)

        label = 'positive' if pred > 0.5 else 'negative'
        print(f'Your review is {label} (score={pred.item():5>.3f})')
