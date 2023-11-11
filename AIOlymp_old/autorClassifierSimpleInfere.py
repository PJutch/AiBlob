from authorClassifierSimple import Model, nlp, device
import torch
import pandas

if __name__ == '__main__':
    print('Loading model...')
    model = torch.load('authorClassifierSimple2.pth').to(device)

    print('Loading data...')
    data = pandas.read_csv('rucode-7.0/private_test.csv')
    contexts = torch.stack([torch.FloatTensor(context.vector) for context in nlp.pipe(data['context'])])
    answers = torch.stack([torch.FloatTensor(answer.vector) for answer in nlp.pipe(data['answer'])])
    features = torch.cat([contexts, answers], dim=1)

    print('Predicting labels')
    preds = model(features)

    print('Saving result...')
    with open('rucode-7.0/private_simple_result.csv', 'w') as f:
        f.write('label\n')
        for pred in preds:
            label = 'ai' if pred > 0.5 else 'people'
            f.write(label + '\n')
