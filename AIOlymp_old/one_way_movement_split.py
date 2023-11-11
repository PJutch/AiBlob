n, m = [int(s) for s in input().split()]
frm = [[] for _ in range(n)]
to = [[] for _ in range(n)]
for _ in range(m):
    a, b = [int(s) for s in input().split()]
    frm[a - 1].append(b - 1)
    to[b - 1].append(a - 1)

for i in range(n):
    ways_to = []
    ways_from = []

    way = [-1]
    stack = [(-1, i)]
    while stack:
        backtrack, town = stack.pop()

        while way[-1] != backtrack:
            way.pop()
        if town == n - 1:
            ways_from.append(len(way) - 1)
        else:
            way.append(town)

            for destination in frm[town]:
                stack.append((town, destination))

    way = [-1]
    stack = [(-1, i)]
    while stack:
        backtrack, town = stack.pop()

        while way[-1] != backtrack:
            way.pop()
        if town == 0:
            ways_to.append(len(way) - 1)
        else:
            way.append(town)

            for destination in to[town]:
                stack.append((town, destination))

    score = 0
    for way_to in ways_to:
        for way_from in ways_from:
            score += way_to + way_from
    print(score)
