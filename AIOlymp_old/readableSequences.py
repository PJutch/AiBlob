DIVIDER = 10 ** 9 + 7


def modPower(base, power):
    if power == 1:
        return base
    if power == 0:
        return 1

    if power % 2 == 0:
        return (modPower(base, power // 2) ** 2) % DIVIDER
    else:
        return (((modPower(base, power // 2) ** 2) % DIVIDER) * base) % DIVIDER


n = int(input())
print(modPower(3, n // 2) + modPower(3, n - n // 2))
