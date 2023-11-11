def remove_punctuation(w: str) -> list:
    res = []
    current = ''
    for char in w:
        if char == '/' or char == '@':
            return []

        if char.isalpha():
            if char == 'ё':
                current += 'е'
            else:
                current += char
        elif current:
            res.append(current)
            current = ''
    if current:
        res.append(current)
    return res


def strip_any_of(s: str, ends, default: int = 0):
    for end in ends:
        if s.endswith(end) and len(s) > len(end):
            return s[-len(end):]
    return s[-default:]


def stem(w: str) -> str:
    w = w[:-7]
    # w = strip_any_of(w, [], -7)
    # w = strip_any_of(w, ['ом', 'ем', 'ой', 'ей', 'ами', 'ями', 'ть', 'ем', 'им', 'ишь', 'ешь', 'ите', 'ете', 'ит', 'ет',
    #                      'ам', 'ям', 'ум', 'юм', 'ого', 'его', 'ым', 'ом', 'ыми'], -7)
    # while count_any_of(w, 'аеиоуыэюя') > 1 and w[-1] in 'аеиоуыэюя' or w and w[-1] in 'ьъ':
    while w and w[-1] in 'аеиоуыэюяьъй':
        w = w[:-1]
    return w


with open('train_text.csv', encoding='UTF-8') as f:
    lines = f.readlines()
data = [tuple(segment.strip() for segment in line.split('\t')) for line in lines[1:]]

with open('test_text.csv', encoding='UTF-8') as f:
    lines = f.readlines()
test_data = [tuple(segment.strip() for segment in line.split('\t')) for line in lines[1:]]

pos_count = 0
neg_count = 0

word_freq = {}
for sample in data:
    words = []
    for segment in sample[2].split():
        for word in remove_punctuation(segment):
            words.append(word.lower())

    for word in words:
        word = stem(word)
        if not word:
            continue

        word_freq.setdefault(word, (0, 0))
        if sample[1] == 'Positive':
            word_freq[word] = word_freq[word][0] + 1, word_freq[word][1]
            pos_count += 1
        elif sample[1] == 'Negative':
            word_freq[word] = word_freq[word][0], word_freq[word][1] + 1
            neg_count += 1

word_freq = {word: word_freq[word] for word in word_freq if word_freq[word][0] + word_freq[word][1] > 1}

lines = []

for sample in test_data:
    words = []
    for segment in sample[1].split():
        for word in remove_punctuation(segment):
            words.append(word.lower())

    pos = 1.
    neg = 1.
    for word in words:
        word = stem(word)
        if not word or word not in word_freq:
            continue

        pos *= word_freq[word][0] / pos_count
        neg *= word_freq[word][1] / neg_count

    if pos >= neg:
        lines.append(f'{sample[0]}\tPositive\n')
    else:
        lines.append(f'{sample[0]}\tNegative\n')

with open('text_submission.csv', 'w', encoding='UTF-8') as f:
    f.write('idx\tScore\n')
    f.writelines(lines)
