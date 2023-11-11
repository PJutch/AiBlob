n, m, k = [int(s) for s in input().split()]

racers = [i for i in range(n)]
interesting = 0

for _ in range(m):
    i, j = [int(s) for s in input().split()]
    i -= 1
    j -= 1

    if abs(racers[i] - i) > k:
        interesting -= 1
    if abs(racers[j] - j) > k:
        interesting -= 1

    racers[i], racers[j] = racers[j], racers[i]

    if abs(racers[i] - i) > k:
        interesting += 1
    if abs(racers[j] - j) > k:
        interesting += 1

    print(1 if interesting > 0 else 0)
