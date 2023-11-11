a = int(input())
b = int(input())
n = int(input())

is_greater = ((b - a) // abs(b - a) + 1) // 2

start = ((a + n - 1) // n) * n * is_greater + (a // n) * n * (1 - is_greater)

numbers = []
for i in range(start, b + 1, -n):
    numbers.append(i)
numbers.reverse()

for number in numbers:
    print(number)

for i in range(start, b + 1, n):
    print(i)
