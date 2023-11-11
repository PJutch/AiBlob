import pandas

df = pandas.read_csv('train_courses.csv')
print(df['user_id'].unique().size, df['user_id'].size)
print(df['user_id'].value_counts())
