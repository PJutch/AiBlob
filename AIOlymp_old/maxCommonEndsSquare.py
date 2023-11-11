s = input()
length = 0
for i in range(len(s)):
    length = max(s.rfind(s[i]) - i + 1, length)
print(length)
