mode = input()

if mode == "first":
    n, k1 = (int(i) for i in input().split())

    sum_rest = n * (n + 1) // 2
    square_sum_rest = 0
    for number in range(1, n + 1):
        square_sum_rest += number ** 2

    for current in input().split():
        sum_rest -= int(current)
        square_sum_rest -= int(current) ** 2

    print(sum_rest, square_sum_rest)
else:
    sum_rest, square_sum_rest = (int(i) for i in input().split())

    n, k2 = (int(i) for i in input().split())

    for current in input().split():
        sum_rest -= int(current)
        square_sum_rest -= int(current) ** 2

    for first in range(1, sum_rest):
        second = sum_rest - first
        if first * first + second * second == square_sum_rest:
            print(first, second)
            break
