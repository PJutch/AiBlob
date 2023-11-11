import numpy

wording_path = input('Wording prediction path: ')
semantic_path = input('Semantic prediction path: ')
result_path = input('Result path: ')

with open(wording_path) as f:
    wording = numpy.asarray([float(line.strip()) for line in f])

with open(semantic_path) as f:
    semantic = numpy.asarray([float(line.strip()) for line in f])

with open(result_path, 'w') as f:
    f.write('labels\n')
    for prediction in numpy.max([wording, semantic], axis=0):
        if prediction > 0.5:
            f.write('ai\n')
        else:
            f.write('people\n')
