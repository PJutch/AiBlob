import math

inp = [int(i) for i in input().split()]
inp.sort(reverse=True)

p_parts = inp[:len(inp) // 2]
q = 1
for i in range(len(inp) // 2, len(inp)):
    q *= inp[i]
    for j in range(len(p_parts)):
        if i != j:
            p_parts[j] *= inp[i]
p = sum(p_parts)

gcd = math.gcd(p, q)
print(p // gcd, q // gcd)
