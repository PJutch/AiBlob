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
    count = 0
    result = 0
    value = 0
    for i in range(len(s)):
        v = get_value(s[i])
        if v == value:
            count += 1
            if count > 3:
                return -1
            if (value == 5 or value == 50 or value == 500) and count > 1:
                return -1
        else:
            if v > value:
                if (count > 1 or (value == 5 or value == 50 or value == 500)
                        or (value != 0 and v > value * 10)):
                    return -1
                else:
                    result -= value * count
            else:
                result += value * count
            value = v
            count = 1
    return result + value * count


for _ in range(int(input())):
    print(decode(input()))
