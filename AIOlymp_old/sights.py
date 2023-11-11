m = int(input()) // 50
n = int(input())

sights = []
for _ in range(n):
    i, p, d = input().split(';')
    sights.append((int(i), int(p) // 50, d))

ways = [[(-1, -1) for _ in range(m + 1)] for _ in range(n + 1)]
ways[0] = [(0, -1) for _ in range(m + 1)]

for index in range(1, n + 1):
    for p in range(m + 1):
        if p >= sights[index - 1][1]:
            i1 = ways[index - 1][p][0]
            i2 = ways[index - 1][p - sights[index - 1][1]][0] + sights[index - 1][0]
            if i1 >= i2:
                ways[index][p] = (i1, 0)
            else:
                ways[index][p] = (i2, 1)
        else:
            ways[index][p] = (ways[index - 1][p][0], 0)

way = []
index = n
p = m
while index != 0:
    assert ways[index][p][1] != -1
    if ways[index][p][1] == 1:
        way.append(sights[index - 1][2])
        p -= sights[index - 1][1]
    index -= 1

way.reverse()
print(';'.join(way))
