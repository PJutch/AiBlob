from xlearn import create_ffm

model = create_ffm()

model.setTrain('small_train.txt')
model.setValidate('small_test.txt')

model.fit({'task': 'binary', 'lr': 0.2, 'lambda': 0.002, 'metric': 'acc'}, './model.out')
