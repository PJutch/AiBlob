def replace(t, i, v):
    return t[:i] + (v, ) + t[i + 1:]


def throws(r):
    return r[1][0] + r[2][0] + r[3][0]


n = int(input())

data = {}

for _ in range(n):
    name, shot, _, result = input().split()

    if shot == 'FT':
        score = 1
    elif shot == '2pt':
        score = 2
    else:
        score = 3

    data.setdefault(name, ((0, 0), (0, 0), (0, 0)))
    if result == 'Made':
        data[name] = replace(data[name], score - 1, (data[name][score - 1][0] + 1, data[name][score - 1][1] + 1))
    else:
        data[name] = replace(data[name], score - 1, (data[name][score - 1][0], data[name][score - 1][1] + 1))

output = []
for name in data:
    output.append((name, ) + data[name])

max_made = throws(max(output, key=throws))
output = sorted((r for r in output if throws(r) == max_made), key=lambda r: r[0])

print(len(output))
for result in output:
    print(f'{result[0]} {result[2][0]}/{result[2][1]} 2pt, {result[3][0]}/{result[3][1]} 3pt, '
          f'{result[1][0]}/{result[1][1]} FT, Total: {result[1][0] + 2 * result[2][0] + 3 * result[3][0]} Points')
