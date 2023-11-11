from xlearn import create_ffm

model = create_ffm()

model.setTest('small_test.txt')
model.setSigmoid()
model.predict('./model.out', './output.txt')
