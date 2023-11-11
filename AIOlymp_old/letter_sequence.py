letters = 'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЫЮЯ'
vowels = 'АЕЁИОУЫЭЮЯ'

k = int(input())

current = -1
i = 1
while k:
    for _ in range(i):
        current += 1
        current %= len(letters)
        while letters[current] not in vowels:
            current += 1
            current %= len(letters)

        k -= 1
        if not k:
            break
    else:
        i += 1
        for _ in range(i):
            current += 1
            current %= len(letters)
            while letters[current] in vowels:
                current += 1
                current %= len(letters)

            k -= 1
            if not k:
                break
        i += 1
print(letters[current])
