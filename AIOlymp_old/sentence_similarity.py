import spacy


def quote(string):
    return '"' + string + '"'


nlp = spacy.load("ru_core_news_lg")

doc1 = nlp(input())
doc2 = nlp(input())

print(doc1.similarity(doc2))
