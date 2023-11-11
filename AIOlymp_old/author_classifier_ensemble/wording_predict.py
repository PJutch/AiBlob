from wording import extract_features

from xgboost import XGBClassifier

from pandas import read_csv
from pickle import load

if __name__ == '__main__':
    data_path = input('Data path: ')
    result_path = input('Result path: ')

    print('Loading data...')
    data = read_csv(data_path)

    print('Extracting features...')
    features = extract_features(data)

    print('Loading model...')
    with open('wording_model.pth', 'rb') as f:
        model = load(f)

    print('Predicting labels...')
    predictions = model.predict_proba(features)

    print('Saving result...')
    with open(result_path, 'w') as f:
        for prediction in predictions:
            prob = (prediction[1].item() - prediction[0].item()) / 2
            f.writelines(str(prob) + '\n')
