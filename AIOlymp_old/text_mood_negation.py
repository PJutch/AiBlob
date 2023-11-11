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
        elif current:  # and current != 'не':
            res.append(current)
            current = ''

    if current:
        res.append(current)
    return res


def split_sentences(w: str) -> list:
    res = [[]]
    current = ''
    for char in w:
        if char == '/' or char == '@':
            return []

        if char.isalpha():
            if char == 'ё':
                current += 'е'
            else:
                current += char
        elif current:  # and current != 'не':
            res[-1].append(current)
            current = ''

            if char in '.?!':
                res.append([])

    if current:
        res[-1].append(current)
    return res


def count_any_of(s, es):
    count = 0
    for e in s:
        if e in es:
            count += 1
    return count


def strip_any_of(s: str, ends, default: int = 0) -> str:
    for end in ends:
        if s.endswith(end) and len(s) > len(end):
            return s[-len(end):]
    return s[-default:]


def are_similar(s1, s2) -> bool:
    if abs(len(s1) - len(s2)) > 1:
        return False

    for i in range(min(len(s1), len(s2))):
        if s1[i] != s2[i]:
            return s1[i:] == s2[i + 1:] or s1[i + 1:] == s2[i:] or s1[i + 1:] == s2[i + 1:]

    return True


def stem(w: str) -> str:
    w = strip_any_of(w, ['ом', 'ем', 'ой', 'ей', 'ами', 'ями', 'ть', 'ем', 'им', 'ишь', 'ешь', 'ите', 'ете', 'ит', 'ет',
                         'ам', 'ям', 'ум', 'юм', 'ого', 'его', 'ым', 'ом', 'ыми'], -1)
    # while count_any_of(w, 'аеиоуыэюя') > 1 and w[-1] in 'аеиоуыэюя' or w and w[-1] in 'ьъ':
    while w and w[-1] in 'аеиоуыэюяьъй':
        w = w[:-1]
    return w


with open('train_text.csv', encoding='UTF-8') as f:
    lines = f.readlines()
data = [tuple(segment.strip() for segment in line.split('\t')) for line in lines[1:]]
train_data = data[:-500]
cross_test_data = data[-500:]

pos_count = 0
neg_count = 0

word_freq = {}
for sample in train_data:
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

statement_count = 0
negation_count = 0
neg_freq = {}
for sample in train_data:
    sentences = []
    for segment in sample[2].split():
        for sentence in split_sentences(segment):
            current = []
            for word in sentence:
                current.append(word.lower())
            sentences.append(current)

    for sentence in sentences:
        pos = 1.
        neg = 1.
        for word in sentence:
            word = stem(word)
            if not word:
                continue

            if word not in word_freq:
                continue
            #     for w in word_freq:
            #         if are_similar(word, w):
            #             word = w
            #             break
            #     else:
            #         continue

            pos *= word_freq[word][0] / pos_count
            neg *= word_freq[word][1] / neg_count

        if pos >= neg:
            negation = sample[1] == 'Positive'
        else:
            negation = sample[1] == 'Negative'

        for word in sentence:
            word = stem(word)
            if not word:
                continue

            neg_freq.setdefault(word, (0, 0))
            if negation:
                neg_freq[word] = neg_freq[word][0] + 1, neg_freq[word][1]
                statement_count += 1
            else:
                neg_freq[word] = neg_freq[word][0], neg_freq[word][1] + 1
                negation_count += 1

accurate = 0
for sample in cross_test_data:
    sentences = []
    for segment in sample[2].split():
        for sentence in split_sentences(segment):
            current = []
            for word in sentence:
                current.append(word.lower())
            sentences.append(current)

    sample_pos = 1.
    sample_neg = 1.
    for sentence in sentences:
        pos = 1.
        neg = 1.

        state = 1.
        negate = 1.
        for word in sentence:
            word = stem(word)
            if not word:
                continue

            if word not in word_freq:
                continue
            #     for w in word_freq:
            #         if are_similar(word, w):
            #             word = w
            #             break
            #     else:
            #         continue

            pos *= word_freq[word][0] / pos_count
            neg *= word_freq[word][1] / neg_count

            state *= word_freq[word][0] / statement_count
            negate *= word_freq[word][1] / negation_count

        if state >= negate:
            pos, neg = neg, pos

        sample_pos *= pos
        sample_neg *= neg

    if sample_pos > sample_neg:
        if sample[1] == 'Positive':
            accurate += 1
    else:
        if sample[1] == 'Negative':
            accurate += 1
print(accurate / len(cross_test_data))
