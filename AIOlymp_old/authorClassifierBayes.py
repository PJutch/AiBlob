import pandas
import spacy
import math

print('Loading spacy...')
nlp = spacy.load('ru_core_news_lg')

print('Loading train data...')
data = pandas.read_csv('rucode-7.0/train.csv')

ai_answers = data[data['label'] == 'ai']['answer']
people_answers = data[data['label'] == 'people']['answer']

print('Parsing train data...')
ai_answer_words = [token.text for document in nlp.pipe(ai_answers) for token in document]
people_answer_words = [token.text for document in nlp.pipe(people_answers) for token in document]


def word_counts(words):
    counts = {}
    for word in words:
        counts.setdefault(word, 0)
        counts[word] += 1
    return counts


def word_log_probs(counts, sm=None):
    if sm is None:
        sm = sum(counts.values())

    probs = {}
    for key, value in counts.items():
        probs[key] = math.log2(value / sm)
    return probs


print('Calculating statistics...')
people_log_probs = word_log_probs(word_counts(ai_answer_words), len(ai_answer_words))
ai_log_probs = word_log_probs(word_counts(people_answer_words), len(people_answer_words))

value_counts = data['label'].value_counts()
ai_apriory_likelihood = math.log(value_counts['ai']) - math.log(value_counts['people'])


def likelihood(tokens, log_probs):
    return sum(log_probs[token.text] for token in tokens if token.text in log_probs)


print('Loading test data...')
test_data = pandas.read_csv('rucode-7.0/public_test.csv')

print('Parsing test data...')
answer_words = list(nlp.pipe(test_data['answer']))

print('Calculating labels...')
ai_likelihoods = [ai_apriory_likelihood + likelihood(answer, ai_log_probs) - likelihood(answer, people_log_probs)
                  for answer in answer_words]
labels = ['ai' if ai_likelihood > 0 else 'people' for ai_likelihood in ai_likelihoods]

print('Saving result...')
with open('rucode-7.0/bayes_result.csv', 'w') as f:
    f.write('label\n')
    for label in labels:
        f.write(label + '\n')
