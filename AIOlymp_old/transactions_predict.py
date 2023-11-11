import pandas

read = pandas.read_csv('train (1).csv', sep=';', decimal=',')

type_freq = read['TRN_TYPE'].value_counts()
type_freq /= type_freq.sum()

# for column in read.columns[:-1]:
#     if read.dtypes[column] == 'float64':
#         read[column] = read[column].round()

data_freq = {column: (read[column][read['TRN_TYPE'] == 'LEGIT'].value_counts(),
                      read[column][read['TRN_TYPE'] == 'FRAUD'].value_counts())
             for column in read.columns[1:-1]}

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


test = pandas.read_csv('test.csv', sep=';', decimal=',')

prediction = pandas.DataFrame()
for i in range(len(test)):
    legitProb = 1
    fraudProb = 1
    for column in data_freq:
        if test[column][i] in data_freq[column][0].index and test[column][i] in data_freq[column][1].index:
            legitProb *= data_freq[column][0][test[column][i]]
            fraudProb *= data_freq[column][1][test[column][i]]
        elif test[column][i] in data_freq[column][0].index:
            legitProb *= data_freq[column][0][test[column][i]]
            fraudProb *= data_freq[column][1]['__unknown']
        elif test[column][i] in data_freq[column][1].index:
            legitProb *= data_freq[column][0]['__unknown']
            fraudProb *= data_freq[column][1][test[column][i]]
        else:
            legitProb *= data_freq[column][0]['__unknown']
            fraudProb *= data_freq[column][1]['__unknown']

    if legitProb > fraudProb:
        prediction = pandas.concat([prediction, pandas.DataFrame(data=[{'id': test['id'][i], 'TRN_TYPE': 'LEGIT'}])])
    else:
        prediction = pandas.concat([prediction, pandas.DataFrame(data=[{'id': test['id'][i], 'TRN_TYPE': 'FRAUD'}])])

prediction.to_csv('prediction.csv', encoding='UTF-8', sep=';', index=False)
