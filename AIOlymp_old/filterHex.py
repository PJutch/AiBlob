for number in input().split():
    if number.startswith('0x'):
        number = number[2:]

    for char in number:
        if char not in '1234567890ABCDEF':
            break
    else:
        print(f'0x{number}', end=' ')
