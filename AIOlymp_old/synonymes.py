import spacy


def quote(string):
    return '"' + string + '"'


nlp = spacy.load("ru_core_news_lg")

doc = nlp(input())

ms = nlp.vocab.vectors.most_similar(
      doc.vector.reshape(1, doc.vector.shape[0]), n=10)

for word in ms[0][0]:
    print(f'{doc.similarity(nlp.vocab[word]):<5.3} {nlp.vocab.strings[word]}')
