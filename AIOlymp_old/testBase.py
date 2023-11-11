base = int(input())
a = int(input(), base=base)
b = int(input(), base=base)

sm = a + b
if sm > 0:
    result = ''
else:
    result = '0'

while sm > 0:
    digit = sm % base
    if digit > 9:
        result += chr(ord('A') + digit - 10)
    else:
        result += str(digit)
    sm //= base

print(''.join(reversed(result)))
