import sys


def fix_spaces(string: str):
    new = ''
    prev_space = False
    for char in string:
        if char.isspace():
            if not prev_space:
                new += char
                prev_space = True
        else:
            new += char
            prev_space = False
    return new


def fix_punctuation(string: str):
    new = ''
    prev_punctuation = False
    prev_points = 0
    prev_question = False
    for char in string:
        if char != '.':
            if prev_points == 3:
                new = new.rstrip()
                new += '.. '
            prev_points = 0

        was_punctuation = prev_punctuation
        if char in ',.?!':
            if prev_punctuation:
                if char == '!' and prev_question:
                    new = new.rstrip()
                    new += char + ' '
                elif char == '.' and 0 < prev_points < 3:
                    prev_points += 1
            else:
                new = new.rstrip()
                new += char + ' '
                prev_punctuation = True

                if char == '.':
                    prev_points = 1
        elif char.isspace():
            if not prev_punctuation:
                new += char
        else:
            new += char
            prev_punctuation = False
        prev_question = not was_punctuation and char == '?'
    return new


def fix_case(string: str, prev_sentence):
    new = ''
    for char in string:
        if char in '.?!':
            new += char
            prev_sentence = True
        elif char.isalpha():
            new += char.title() if prev_sentence else char.lower()
            prev_sentence = False
        else:
            new += char
    return new


# lines = sys.stdin.readlines()
with open('test.txt', encoding='UTF-8') as f:
    lines = f.readlines()

rest = ''
prev_sentence = True
for line in lines:
    line = fix_spaces(line.rstrip())
    if line.startswith(' '):
        line = '   ' + line[1:]
        prev_sentence = True

    line = fix_punctuation(line)
    line = fix_case(line, prev_sentence)
    prev_sentence = line.rstrip().endswith(('.', '!', '?'))

    if line.endswith('-'):
        last_space = line.rfind(' ')
        print(line[:last_space].rstrip())
        rest = line[last_space + 1:-1]
    else:
        print((rest + line).rstrip())
        rest = ''
