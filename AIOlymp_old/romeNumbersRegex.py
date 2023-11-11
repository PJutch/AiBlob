import re

def get_value(char):
    if char == 'I':
        return 1
    if char == 'V':
        return 5
    if char == 'X':
        return 10
    if char == 'L':
        return 50
    if char == 'C':
        return 100
    if char == 'D':
        return 500
    assert char == 'M'
    return 1000


def decode(s):
    if re.match('M{0,3}(CM|CD|D?C{0,3})?(XC|XL|L?X{0,3})?(IX|IV|V?I{0,3})?$', s) is None:
        return -1

    count = 0
    result = 0
    value = 0
    for i in range(len(s)):
        v = get_value(s[i])
        if v == value:
            count += 1
        else:
            if v > value:
                result -= value * count
            else:
                result += value * count
            value = v
            count = 1
    return result + value * count


for _ in range(int(input())):
    print(decode(input()))
