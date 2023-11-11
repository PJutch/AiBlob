import sys

def split_by_nonalpha(string):
    parts = []
    current = ''
    for char in string:
        if char.isalpha():
            current += char
        else:
            if current:
                parts.append(current)
            current = ''
    if current:
        parts.append(current)
    return parts


text = ''.join(sys.stdin.readlines())

words = {}
for word in split_by_nonalpha(text):
    word = word.lower()
    words.setdefault(word, 0)
    words[word] += 1

word_list = list(words.items())
word_list.sort(key=lambda record: record[0])
word_list.sort(key=lambda record: record[1], reverse=True)

for i in range(50):
    if i >= len(word_list):
        break
    print(word_list[i][0], end=' ')
