from navec import Navec
from natasha import Segmenter, NewsEmbedding, NewsMorphTagger, Doc, MorphVocab

print('Reading corpus...')
with open('extract_dialogues_from_anekdots.txt', encoding='UTF-8') as f:
    corpus = f.read().split('\n\n\n\n')

print('Creating pipeline...')
segmenter = Segmenter()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
morph_vocab = MorphVocab()
navec = Navec.load('navec_hudlit_v1_12B_500K_300d_100q.tar')


def process_doc(i, doc_str):
    if i % 1000 == 0:
        print(f'Processing doc #{i}...')

    doc = Doc(doc_str)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    for token in doc.tokens:
        token.lemmatize(morph_vocab)

    stop_words = ['а', 'ведь', 'потому', 'конечно', 'же']

    # ids_len = 50
    ids = ([navec.vocab['<pad>']] * 3 + [navec.vocab[token.lemma] for token in doc.tokens
                                         if token.lemma in navec.vocab and token.lemma not in stop_words]
           + [navec.vocab['<pad>']])
    # ids = ids[:ids_len]
    #
    # pad_id = navec.vocab['<pad>']
    # while len(ids) < ids_len:
    #     ids.append(pad_id)

    return ids


ids = [process_doc(i, doc_str) for i, doc_str in enumerate(corpus)]

print('Saving ids...')
with open('ids5.txt', 'w') as f:
    for doc_ids in ids:
        for token_id in doc_ids:
            f.write(str(token_id) + ' ')
        f.write('\n')
