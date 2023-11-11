import spacy
import numpy

nlp = spacy.load("ru_core_news_lg")


def vec_offset(frm, to):
    return nlp(to).vector - nlp(frm).vector


examples = {'мужчина': 'женщина', 'король': 'королева',
            'лис': 'лиса', 'заяц': 'зайчиха',
            'шурик': 'шура', 'валера': 'лера',
            'поэт': 'поэтесса',
            'николаич': 'николавна'}

example_transforms = numpy.asarray([vec_offset(frm, to) for frm, to in examples.items()])
transform = example_transforms.mean(axis=0)

while True:
    doc = nlp(input())
    transformed = transform + doc.vector

    ms = nlp.vocab.vectors.most_similar(transformed.reshape(1, transformed.shape[0]), n=10)
    for word in ms[0][0]:
        print(nlp.vocab.strings[word])
