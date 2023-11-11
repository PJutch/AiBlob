coins = [1, 5, 10, 50, 100, 200, 500, 1000, 2500]
banknotes = [500, 1000, 5000, 10000, 20000, 50000, 100000, 200000, 500000]

max1 = 0
max1i = 0
max2 = 0
max2i = 0
for i in range(int(input())):
    description = input()

    kopecks = 0
    for char in description:
        if char.isupper():
            kopecks += banknotes[ord(char) - ord('A')]
        else:
            kopecks += coins[ord(char) - ord('a')]

    if kopecks > max1:
        if max1 > max2 or max1i > max2i:
            max2 = max1
            max2i = max1i

        max1 = kopecks
        max1i = i + 1
    elif kopecks > max2:
        max2 = kopecks
        max2i = i + 1
    elif kopecks == max1:
        assert kopecks == max2
        if max1i > max2i:
            max2i = i + 1
        else:
            max1i = i + 1

print(min(max1i, max2i))
print(max(max1i, max2i))
