from sklearn.metrics import roc_auc_score

y_true = [0, 1, 1, 0, 0, 1, 1, 0, 0, 0]
y_score = [0.35, 0.85, 0.75, 0.25, 0.05, 0.45, 0.95, 0.65, 0.15, 0.55]

print(roc_auc_score(y_true, y_score))
