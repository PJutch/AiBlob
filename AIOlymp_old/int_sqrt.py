from math import sqrt

with open('input.txt') as f:
    for line in f.readlines():
        n = float(line)
        print(int(sqrt(n)))
