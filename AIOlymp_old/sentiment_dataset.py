from sentiment_classifier import nlp

import torch.utils.data


def load_dictionary():
    phrases = {}
    with open('stanfordSentimentTreebank/dictionary.txt') as f:
        for line in f:
            phrase, phrase_id = line.split('|')
            phrases[phrase] = int(phrase_id)
    return phrases


def load_sentiment_labels():
    labels = {}
    with open('stanfordSentimentTreebank/sentiment_labels.txt') as f:
        for index, line in enumerate(f):
            if index != 0:
                phrase_id, label = line.split('|')
                labels[int(phrase_id)] = float(label)
    return labels


def load_dataset_sentences():
    sentences = []
    with open('stanfordSentimentTreebank/datasetSentences.tsv') as f:
        for index, line in enumerate(f):
            if index != 0:
                _, sentence = line.split('\t')
                sentences.append(sentence.strip())
    return sentences


def load_dataset_split():
    split = {}
    with open('stanfordSentimentTreebank/datasetSplit.csv') as f:
        for index, line in enumerate(f):
            if index != 0:
                sentence_id, label = line.split(',')
                split[int(sentence_id)] = int(label)
    return split


class SentimentDataset(torch.utils.data.Dataset):
    class Entry:
        def __init__(self, document, label):
            self.document = document
            self.label = label

    def __init__(self, dictionary, labels, sentences, documents, split, train=True):
        if train:
            split_label = 1
        else:
            split_label = 2

        self.data = []
        for i, (sentence, document) in enumerate(zip(sentences, documents)):
            if sentence in dictionary:
                phrase_id = dictionary[sentence]
                label = labels[phrase_id]

                if split[i + 1] == split_label:
                    self.data.append(self.Entry(document, torch.FloatTensor([label > 0.5])))

    @staticmethod
    def load():
        dictionary = load_dictionary()
        labels = load_sentiment_labels()
        sentences = load_dataset_sentences()
        documents = list(nlp.pipe(sentences))
        split = load_dataset_split()

        return (SentimentDataset(dictionary, labels, sentences, documents, split, train=True),
                SentimentDataset(dictionary, labels, sentences, documents, split, train=False))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i].document.vector, self.data[i].label
