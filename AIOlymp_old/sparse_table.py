n = int(input())

logs = [0]
for i in range(1, n):
    logs.append(logs[(i + 1) // 2 - 1] + 1)

sparse_table = [[int(i) for i in input().split()]]
for t in range(1, logs[n - 1]):
    sparse_table.append([])
    for i in range(n):
        sparse_table[t].append(max(sparse_table[t - 1][i],
                                   sparse_table[t - 1][min(i + (1 << (t - 1)), n - 1)]))

k = int(input())
for j in range(n - k + 1):
    dl, dr = [int(i) for i in input().split()]

    l = max(j - dl, 0)
    r = min(j + k + dr, n - 1)

    t = logs[r - l - 1]

    max_value = sparse_table[t][l]
    index2 = r - (1 << t)
    if index2 >= 0:
        max_value = max(sparse_table[t][index2], max_value)

    print(max_value)
