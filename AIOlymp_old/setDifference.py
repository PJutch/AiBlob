n, m = (int(s) for s in input().split())
first = set(input().split())

second = []
for s in input().split():
    if s not in first:
        second.append(s)

print(len(second))
print(' '.join(second))
