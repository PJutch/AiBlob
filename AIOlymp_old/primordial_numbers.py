primes = [2]
current = 3
while len(primes) < 70:
    for prime in primes:
        if current % prime == 0:
            break
    else:
        primes.append(current)
    current += 1
primes.insert(0, 1)

n = int(input())
maxes = []
max_number = 0
for k in range(n):
    inp = [int(i) for i in input().split(':')]

    number = 0
    for i in range(len(inp)):
        number += primes[i] * inp[- i - 1]

    if number == max_number:
        maxes.append(k + 1)
    elif number > max_number:
        max_number = number
        maxes = [k + 1]

for maximum in maxes:
    print(maximum)
