import random

import numpy

from model import Model, navec
from natasha import Segmenter, NewsEmbedding, NewsMorphTagger, Doc, MorphVocab
import torch

print('Creating pipeline...')
segmenter = Segmenter()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
morph_vocab = MorphVocab()

navec_gensim = navec.as_gensim

model: Model = torch.load('model2.pth')


def process_text(text):
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    for token in doc.tokens:
        token.lemmatize(morph_vocab)
    return [navec.vocab[token.lemma] for token in doc.tokens if token.lemma in navec.vocab]


while True:
    text = input()
    print(text, end=' ')

    words = process_text(text)
    if len(words) < 10:
        words = [navec.vocab['<pad>']] * (10 - len(words)) + words

    while True:
        with torch.no_grad():
            next_vec: torch.Tensor = model(torch.Tensor(words[-10:]))[-1]

            most_similar = navec_gensim.most_similar([next_vec.numpy()], topn=5)
            word = random.choice(most_similar)[0]

            if word == '<pad>':
                print('\n')
                break

            print(word, end=' ')
            words.append(navec.vocab[word])
