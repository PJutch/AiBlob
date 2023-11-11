from prepare_data import prepare_data

import catboost
from catboost import CatBoostRegressor

import pandas
import sklearn.model_selection as sk_meta
# noinspection SpellCheckingInspection
from sklearn.metrics import mean_absolute_percentage_error as mape

if __name__ == '__main__':
    data = pandas.read_csv('train.csv')
    x = prepare_data(data.drop('price_target', axis=1))
    y = data['price_target']

    cat_features = [column for column in x.columns if column.endswith('_cat')]

    x_train, x_validation, y_train, y_validation = sk_meta.train_test_split(x, y, train_size=0.75)

    train_pool = catboost.Pool(x_train, y_train, cat_features=cat_features)

    regressor = CatBoostRegressor()
    regressor.fit(train_pool, eval_set=(x_validation, y_validation), plot=True)

    feature_importance = regressor.get_feature_importance(train_pool)
    feature_names = x_train.columns
    for score, name in sorted(zip(feature_importance, feature_names), reverse=True):
        print('{}: {}'.format(name, score))

    print('error = ', mape(y_validation, regressor.predict(x_validation)))

    regressor.save_model('model5.dump')
