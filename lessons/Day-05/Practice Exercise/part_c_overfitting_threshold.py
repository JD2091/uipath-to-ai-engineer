from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

X, y = make_classification(n_samples=600, n_features=20, n_informative=5, weights=[0.85],
                           flip_y=0.05, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)

print("1) Overfitting: tree depth vs F1")
for depth in (1, 2, 3, 5, 8, None):
    tree = DecisionTreeClassifier(max_depth=depth, random_state=0).fit(X_tr, y_tr)
    print(f"  depth={str(depth):4}  train F1={f1_score(y_tr, tree.predict(X_tr)):.2f}  "
          f"test F1={f1_score(y_te, tree.predict(X_te)):.2f}")

print("\n2) Accuracy trap: always predict 0 ->", f"{(y_te == 0).mean():.2f} accuracy, recall 0.00")

print("\n3) Threshold trade-off (logistic regression)")
proba = LogisticRegression(max_iter=1000).fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
for t in (0.2, 0.35, 0.5, 0.65, 0.8):
    pred = (proba >= t).astype(int)
    print(f"  threshold={t:.2f}  precision={precision_score(y_te, pred, zero_division=0):.2f}  "
          f"recall={recall_score(y_te, pred):.2f}  F1={f1_score(y_te, pred):.2f}")
