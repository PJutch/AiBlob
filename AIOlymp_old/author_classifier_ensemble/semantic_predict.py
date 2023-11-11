from semantic import Model, nlp, device
import torch
import pandas

if __name__ == '__main__':
    data_path = input('Data path: ')
    result_path = input('Result path: ')

    print('Loading model...')
    model = torch.load('semantic_model.pth').to(device)

    print('Loading data...')
    data = pandas.read_csv(data_path)

    print('Processing data...')
    contexts = torch.stack([torch.FloatTensor(context.vector) for context in nlp.pipe(data['context'])])
    answers = torch.stack([torch.FloatTensor(answer.vector) for answer in nlp.pipe(data['answer'])])
    features = torch.cat([contexts, answers], dim=1)

    print('Predicting labels...')
    predictions = model(features)

    print('Saving result...')
    with open(result_path, 'w') as f:
        f.writelines(str(prediction.item()) + '\n' for prediction in predictions)
