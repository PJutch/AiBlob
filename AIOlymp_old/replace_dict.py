words = {}
with open('ru.txt', encoding="utf-8") as file:
    for line in file:
        word, meaning = line.split()
        words[word] = meaning

text = input()
result = ''
translated = True
while text:
    if not text[0].isalpha():
        result += text[0]
        text = text[1:]
        continue

    best_word = ''
    best_meaning = ''
    for word, meaning in words.items():
        if text.startswith(word) and len(word) > len(best_word):
            best_word, best_meaning = word, meaning

    if not best_word:
        print('Неизвестное слово')
        translated = False
        break
    else:
        text = text[len(best_word):]
        result += best_meaning
if translated:
    print(result)
