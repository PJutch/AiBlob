def count_bracket_sequences(open_brackets=0, length=24):
    if length == 0:
        return 1 if open_brackets == 0 else 0

    res = count_bracket_sequences(open_brackets + 1, length - 1)
    if open_brackets > 0:
        res += count_bracket_sequences(open_brackets - 1, length - 1)
    return res


def factorial(n):
    res = 1
    for i in range(2, n + 1):
        res *= i
    return res


print(count_bracket_sequences() * 8 ** 4 * factorial(12) // factorial(4))
