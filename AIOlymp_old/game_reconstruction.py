n, k, T = [int(s) for s in input().split()]

if T < 2 * k - 1 or k == 2 and T > n * ((n + 1) // 2) + 1:
    print('impossible')
else:
    print('OK')
    winner = 'O' if T % 2 == 0 else 'X'
    field = [['.' for _ in range(n)] for _ in range(n)]
    i = 0
    j = 0
    if k == 2:
        for rest in range(T, 1, -1):
            field[i][j] = 'X' if rest % 2 == 0 else 'O'

            i += 1
            if i >= n:
                i = 0
                j += 2
        field[0][1] = winner

        for row in field:
            print(''.join(row))
        print(1, 2)
    else:
        for i in range(n):
            for j in range(n):
                field[i][j] = 'X' if (i // 2 + j + T + 1) % 2 == 0 else 'O'

        zeroes = T // 2
        crosses = T - zeroes
        for j in range(n):
            for i in range(n):
                if field[i][j] == 'X':
                    if crosses > 0:
                        crosses -= 1
                    else:
                        field[i][j] = '.'
                else:
                    if zeroes > 0:
                        zeroes -= 1
                    else:
                        field[i][j] = '.'

        i1 = 0
        j1 = 0
        for i in range(k):
            if field[i][n - 1] != winner:
                while field[i1][j1] != winner:
                    i1 += 1
                    if i1 >= n:
                        i1 = 0
                        j1 += 1
                field[i][n - 1], field[i1][j1] = field[i1][j1], field[i][n - 1]

        for row in field:
            print(''.join(row))
        print(1, n)
