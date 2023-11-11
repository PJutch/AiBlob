import pandas

read = pandas.read_csv('train (1).csv', sep=';', decimal=',')
read = read.sample(frac=1)

type_freq = read[:-10000]['TRN_TYPE'].value_counts()
type_freq /= type_freq.sum()
print(type_freq)

# for column in read.columns[:-1]:
#     if read.dtypes[column] == 'float64':
#         read[column] = read[column].round()

data_freq = {column: (read[:-10000][column][read['TRN_TYPE'] == 'LEGIT'].value_counts(),
                      read[:-10000][column][read['TRN_TYPE'] == 'FRAUD'].value_counts())
             for column in read.columns[:-1]}
for column in data_freq:
    data_freq[column] = (pandas.concat([data_freq[column][0], pandas.Series({'__unknown': 0})]),
                         pandas.concat([data_freq[column][1], pandas.Series({'__unknown': 0})]))

    for index in data_freq[column][0].index:
        if index == '__unknown':
            continue

        count = data_freq[column][0][index]
        if count + (data_freq[column][1][index] if index in data_freq[column][1].index else 0) <= 1:
            modified = data_freq[column][0].drop(index)
            modified['__unknown'] += 1
            data_freq[column] = modified, data_freq[column][1]

    for index in data_freq[column][1].index:
        if index == '__unknown':
            continue

        count = data_freq[column][1][index]
        if count + (data_freq[column][0][index] if index in data_freq[column][0].index else 0) <= 1:
            modified = data_freq[column][1].drop(index)
            modified['__unknown'] += 1
            data_freq[column] = data_freq[column][0], modified

    data_freq[column] = (data_freq[column][0] / data_freq[column][0].sum() * type_freq['LEGIT'] / 30,
                         data_freq[column][1] / data_freq[column][1].sum() * type_freq['FRAUD'])

tp = 0
tn = 0
fp = 0
fn = 0
for i in range(len(read) - 10000, len(read)):
    legitProb = 1
    fraudProb = 1
    for column in data_freq:
        if read[column][i] in data_freq[column][0].index and read[column][i] in data_freq[column][1].index:
            legitProb *= data_freq[column][0][read[column][i]]
            fraudProb *= data_freq[column][1][read[column][i]]
        elif read[column][i] in data_freq[column][0].index:
            legitProb *= data_freq[column][0][read[column][i]]
            fraudProb *= data_freq[column][1]['__unknown']
        elif read[column][i] in data_freq[column][1].index:
            legitProb *= data_freq[column][0]['__unknown']
            fraudProb *= data_freq[column][1][read[column][i]]
        else:
            legitProb *= data_freq[column][0]['__unknown']
            fraudProb *= data_freq[column][1]['__unknown']

    if legitProb > fraudProb:
        if read['TRN_TYPE'][i] == 'LEGIT':
            tn += 1
        else:
            fn += 1
    else:
        if read['TRN_TYPE'][i] == 'FRAUD':
            tp += 1
        else:
            fp += 1
accuracy = (tp + tn) / (tp + tn + fp + fn)
print(accuracy)
precision = (tp + 1) / (tp + fp + 1)
print(precision)
recall = (tp + 1) / (tp + tn + 1)
print(recall)
print(2 / (1 / precision + 1 / recall))
