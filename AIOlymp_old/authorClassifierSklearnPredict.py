from authorClassifierSklearn import extract_features

from xgboost import XGBClassifier

from pandas import read_csv
from pickle import load

print('Loading data...')
data = read_csv('rucode-7.0/private_test.csv')

print('Extracting features...')
features = extract_features(data)

print('Loading model...')
with open('authorClassifierXGBoost.pth', 'rb') as f:
    model = load(f)

print('Predicting labels...')
preds = model.predict(features)

print('Saving result...')
with open('rucode-7.0/private_result.csv', 'w') as f:
    f.write('label\n')
    for pred in preds:
        label = 'ai' if pred else 'people'
        f.write(label + '\n')
