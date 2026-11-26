"""Decision tree on Fragrantica (Kaggle) perfumes: will a perfume be highly rated (average 4.0+)?

PRODUCT LEVEL ONLY: no per-user data exists in this source.
Business question: which perfumes should we add to the catalog?
Model: classification tree. Baseline: the most common class.
Run from the repo root:  python tree/kaggle/run.py
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from shared.data import load_kaggle
from shared.evaluate import classification_table, confusion

HERE = Path(__file__).parent
df, top = load_kaggle()
df["high"] = (df.rating >= 4.0).astype(int)
X = [f"acc_{a}" for a in top] + ["gender_women", "gender_men", "log_count"]
tr, te = df[df.split == "train"], df[df.split == "test"]

print("=== 1. Data (REAL, product level) ===")
print(f"Fragrantica perfumes with 50+ ratings (Kaggle): {len(df):,}; one row per perfume")
print("outcome: high = average rating 4.0 or more  <- OUTCOME;  features: 20 accords, gender, log ratings")
print(f"share high: {df.high.mean():.3f}. No per-user data: cannot personalise.")

print("\n=== 2. Split ===")
print(f"random 70/30 split of perfumes (seed 0): {len(tr):,} train / {len(te):,} test")

print("\n=== 3. Fit (depth and leaf size by 5-fold CV) ===")
grid = GridSearchCV(DecisionTreeClassifier(random_state=0),
                    {"max_depth": [2, 3, 4, 5, 6], "min_samples_leaf": [50, 100, 200]},
                    cv=5, scoring="accuracy").fit(tr[X], tr.high)
tree = grid.best_estimator_
print(f"chosen: depth {tree.max_depth}, min leaf {tree.min_samples_leaf}")
small = DecisionTreeClassifier(max_depth=3, min_samples_leaf=tree.min_samples_leaf, random_state=0).fit(tr[X], tr.high)
print("depth-3 version for reading:"); print(export_text(small, feature_names=X, decimals=2))
fig, ax = plt.subplots(figsize=(11, 5))
plot_tree(small, feature_names=X, class_names=["below 4.0", "4.0+"], filled=True, impurity=False, fontsize=7, ax=ax)
fig.tight_layout(); fig.savefig(HERE / "tree.png", dpi=150)

print("\n=== 4. Test perfumes: baseline vs tree ===")
y = te.high.to_numpy(); p = tree.predict(te[X]); majority = int(tr.high.mean() >= 0.5)
print(classification_table(y, {f"baseline: most common class ({'high' if majority else 'not high'})": np.full(len(y), majority),
                               "tree": p}).to_string())
print("tree confusion matrix (test):"); print(confusion(y, p).rename(index=lambda s: s.replace("liked", "high"), columns=lambda s: s.replace("liked", "high")).to_string())
print(f"accuracy: training {(tree.predict(tr[X]) == tr.high).mean():.3f} vs test {(p == y).mean():.3f}")

print("\n=== 5. Decision: add a perfume to the catalog if the tree says 4.0+ ===")
new = {f"acc_{a}": int(a in ["amber", "vanilla", "warm spicy", "sweet", "woody"]) for a in top}
new.update(gender_women=0, gender_men=0, log_count=np.log1p(500))
pr = tree.predict_proba(pd.DataFrame([new])[X])[0, 1]
print(f"new input: unisex perfume, accords amber, vanilla, warm spicy, sweet, woody, 500 ratings -> P(4.0+) {pr:.2f}: {'add' if pr >= .5 else 'pass'}")

print("\n=== 6. Value: share of stocked perfumes that are actually 4.0+ (test perfumes) ===")
print(f"no model (stock everything): {y.mean():.3f} of {len(y):,}")
print(f"tree (stock if predicted 4.0+): {y[p == 1].mean():.3f} of {int((p == 1).sum()):,}")
print("No sales data exists for these perfumes, so no $ figure is given.")

print("\n=== 7. Sanity check: shuffled outcome ===")
sh = DecisionTreeClassifier(max_depth=tree.max_depth, min_samples_leaf=tree.min_samples_leaf, random_state=0).fit(
    tr[X], np.random.default_rng(1).permutation(tr.high.to_numpy()))
print(classification_table(y, {"shuffled-outcome tree": sh.predict(te[X])}).to_string())
