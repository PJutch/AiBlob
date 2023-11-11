from pandas import read_csv
import spacy
from gensim.corpora import Dictionary
from gensim.models import TfidfModel

if __name__ == '__main__':
    print('Loading train data...')
    data = read_csv('rucode-7.0/train.csv')

    print('Loading spacy model...')
    nlp = spacy.load('ru_core_news_lg')


    def parse_corpus(corpus):
        return [[token.lemma_ for token in document] for document in nlp.pipe(corpus)]


    print('Parsing data...')
    context_tokens = parse_corpus(data['context'])
    answer_tokens = parse_corpus(data['answer'])

    corpus_tokens = context_tokens + answer_tokens

    print('Fitting dictionary...')
    dictionary = Dictionary(corpus_tokens)

    print('Saving dictionary...')
    dictionary.save('rucode-7.0/dictionary')

    print('Converting corpust to BoW')
    corpus = [dictionary.doc2bow(document) for document in corpus_tokens]  # convert corpus to BoW format

    print('Fitting TF-IDF...')
    tfidfModel = TfidfModel(corpus)

    print('Saving TF-IDF...')
    tfidfModel.save('rucode-7.0/tfidf')
