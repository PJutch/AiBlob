n, max_price, min_quality = map(int, input().split())

best_price = max_price + 1
best_quality = 0
best_examples = 0

for i in range(n):
    n_i, price, quality = map(int, input().split())

    if price < best_price and quality >= min_quality:
        best_price = price
        best_quality = quality
        best_examples = n_i
    elif price == best_price and quality > best_quality:
        best_quality = quality
        best_examples = n_i

print(best_examples)
