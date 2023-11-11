from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from gensim.corpora import Dictionary
from gensim.matutils import corpus2dense
from gensim.models import TfidfModel
import spacy

from pandas import read_csv
from numpy import concatenate
from pickle import dump

print('Loading spacy model...')
nlp = spacy.load('ru_core_news_lg')

print('Loading dictionary...')
dictionary = Dictionary.load('dictionary')

# print('Loading TF-IDF...')
# tfidfModel = TfidfModel.load('rucode-7.0/tfidf')


def process_corpus(corpus):
    bows = (dictionary.doc2bow([token.lemma_ for token in document]) for document in nlp.pipe(corpus))
    # tfdifs = (tfidfModel[document] for document in bows)
    array = corpus2dense(bows, len(dictionary)).transpose()
    return array # / array.sum(axis=1).reshape((array.shape[0], 1)).repeat(array.shape[1], axis=1)


def extract_features(data):
    contexts = process_corpus(data['context'])
    answers = process_corpus(data['answer'])

    return concatenate([contexts, answers], axis=1)


if __name__ == '__main__':
    print('Loading data...')
    data = read_csv('rucode-7.0/train.csv')

    print('Extracting features...')
    features = extract_features(data)
    labels = data['label'] == 'ai'

    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.2)

    print('Fitting model...')
    model = XGBClassifier().fit(train_features, train_labels)

    print(f'Model score is {model.score(test_features, test_labels)}')

    print('Saving model...')
    with open('wording_model.pth', 'wb') as f:
        dump(model, f)
