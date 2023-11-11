def check_brackets(expression, open_brackets, close_brackets):
    bracket_stack = []
    for char in expression:
        if char in open_brackets:
            bracket_stack.append(char)
        elif char in close_brackets:
            if not bracket_stack:
                return 'Miss open bracket for bracket ' + char
            if bracket_stack[-1] == close_brackets[char]:
                bracket_stack.pop(-1)
            else:
                return f'Expected {open_brackets[bracket_stack[-1]]} but {char} found'

    if bracket_stack:
        return 'Miss close bracket for bracket ' + bracket_stack[0]
    return 'Correct'

expr = input()

n = int(input())
open_b = {}
close_b = {}

for _ in range(n):
    b = input()
    open_b[b[0]] = b[1]
    close_b[b[1]] = b[0]

print(check_brackets(expr, open_b, close_b))
