import pandas

data = pandas.read_csv('rucode-7.0/train.csv')

with open('rucode-7.0/bayes_result.csv') as f:
    results = [line.strip() for line in f]

tp = 0
fp = 0
tn = 0
fn = 0
for result, label in zip(results, data['label']):
    if result == 'ai' and label == 'ai':
        tp += 1
    elif result == 'ai' and label != 'ai':
        fp += 1
    elif result != 'ai' and label == 'ai':
        fn += 1
    elif result != 'ai' and label != 'ai':
        tn += 1

accuracy = (tp + tn) / len(results)
recall = tp / (tp + fn)
precision = tp / (tp + fp)
f = 2 / (1 / recall + 1 / precision)
print(f'total={len(results)} tp={tp:>2} fp={fp:>2} tp={fn:>2} fp={tn:>2} accuracy={accuracy * 100:.3}% '
      f'recall={recall * 100:.3}% precision={precision * 100:.3}% f={f * 100:.3}%')
