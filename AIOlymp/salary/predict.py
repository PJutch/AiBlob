from train import Model, device, get_features
import torch
import pandas

if __name__ == '__main__':
    print('Loading model...')
    model = torch.load('model_denorm9.pth').to(device)

    print('Loading data...')
    data = pandas.read_csv('test_vectors.csv')
    features = get_features(data.drop(['Unnamed: 0', 'id'], axis=1))

    print('Predicting salaries...')
    preds = model(features)

    print('Saving result...')
    with open('prediction.tsv', 'w') as f:
        f.write('id\tsalary_mean_net\n')
        for i, pred in enumerate(preds):
            f.write(str(data['id'][i]) + '\t' + str(int(pred)) + '\n')
