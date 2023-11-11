from prepare_data import prepare_data

import catboost
from catboost import CatBoostRegressor

import pandas

if __name__ == '__main__':
    x = prepare_data(pandas.read_csv('test.csv'))

    regressor = CatBoostRegressor()
    regressor.load_model('model5.dump')

    prediction = regressor.predict(x)

    with open('prediction5.csv', 'w') as output:
        output.write('target_price\n')
        output.writelines([str(value) + '\n' for value in prediction])
