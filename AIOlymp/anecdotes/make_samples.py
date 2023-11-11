with open('ids5.txt') as f:
    tokens = [[token for token in doc.split()] for doc in f]

samples = []
sample_len = 11
for doc_i, doc in enumerate(tokens):
    if doc_i % 1000 == 0:
        print(f'Processing doc {doc_i}...')
    i = 0
    while i + sample_len < len(doc):
        samples.append(doc[i:i + sample_len])
        i += 1

with open('samples4.txt', 'w') as f:
    for sample in samples:
        f.write(' '.join(sample) + '\n')
