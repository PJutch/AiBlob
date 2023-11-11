import numpy
from pandas import read_csv

labels = read_csv('train.csv')['label']

with open('train_prediction.csv') as f:
    prediction = numpy.asarray([line.strip() for line in f.readlines()[1:]])

tp = ((labels == 'ai') & (prediction == 'ai')).astype('float').sum()
tn = ((labels != 'ai') & (prediction != 'ai')).astype('float').sum()
fp = ((labels != 'ai') & (prediction == 'ai')).astype('float').sum()
fn = ((labels == 'ai') & (prediction != 'ai')).astype('float').sum()

precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 / (1 / recall + 1 / precision)
print(f1)
