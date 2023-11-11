n, m, a, b = [int(i) for i in input().split()]
a -= 1
b -= 1

field = []
for _ in range(n):
    field.append([int(i) for i in input().split()])

top_left = 0
for i in range(a + 1):
    for j in range(b + 1):
        top_left = max(field[i][j] + 2 * (a - i + b - j), top_left)

bottom_right = 0
for i in range(a, n):
    for j in range(b, m):
        bottom_right = max(field[i][j] + 2 * (i - a + j - b), bottom_right)

print(top_left + bottom_right + 4)
