import pandas
from pyffm import PyFFM
from pyffm.test.data import sample_df
from sklearn.model_selection import train_test_split

data = (pandas.concat([sample_df[sample_df['click'] == 1], sample_df[sample_df['click'] == 0].sample(n=1000)])
        .sample(frac=1))
train_df, test_df = train_test_split(data)

ffm = PyFFM(model='ffm')
ffm.train(train_df)

ffm.predict(test_df)
