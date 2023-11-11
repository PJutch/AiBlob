n = int(input())
current = 0
for _ in range(n):
    occupied = int(input())
    while current != occupied:
        print(current)
        current += 1
    current += 1
while current < 1000000:
    print(current)
    current += 1
