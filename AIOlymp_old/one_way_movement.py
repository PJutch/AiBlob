n, m = [int(s) for s in input().split()]
towns = [[] for _ in range(n)]
for _ in range(m):
    a, b = [int(s) for s in input().split()]
    towns[a - 1].append(b - 1)

stack = [(-1, 0)]
way = [-1]
scores = [0 for _ in range(n)]
while stack:
    backtrack, town = stack.pop()

    while way[-1] != backtrack:
        way.pop()
    if town == n - 1:
        for i in range(1, len(way)):
            scores[way[i]] += len(way) - 1
        scores[n - 1] += len(way) - 1
    else:
        way.append(town)

        for destination in towns[town]:
            stack.append((town, destination))

for score in scores:
    print(score)
