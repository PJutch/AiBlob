from authorClassifierBow import device, Model, extract_features
import torch
import pandas

if __name__ == '__main__':
    print('Loading model...')
    model = torch.load('authorClassifierBow10.pth').to(device)

    print('Loading data...')
    data = pandas.read_csv('rucode-7.0/private_test.csv')

    print('Extracting features...')
    features = extract_features(data)

    print('Predicting labels...')
    preds = model(features)

    print('Saving result...')
    with open('rucode-7.0/bow_private_result.csv', 'w') as f:
        f.write('label\n')
        for pred in preds:
            label = 'ai' if pred > 0.5 else 'people'
            f.write(label + '\n')
