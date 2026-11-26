"""Decision tree on REAL Amazon fragrance ratings: will this person like this perfume (4-5 stars)?

Business question: should we show this perfume to this person?
Model: classification tree. Baseline: the most common class (as the brief defines it).
Run from the repo root:  python tree/amazon/run.py
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from shared.data import load_amazon, person_features, amazon_feature_names
from shared.evaluate import classification_table, confusion, top_pick_liked, value_table, print_assumptions

HERE = Path(__file__).parent
df, top = load_amazon()
df = person_features(df, top)
X = amazon_feature_names(top)
tr, te = df[df.split == "train"], df[df.split == "test"]

print("=== 1. Data (REAL) ===")
print("source: Amazon Reviews 2023 fragrance reviews about the smell, matched to Fragrantica accords.")
print(f"one row per (person, perfume): {len(df):,} ratings, {df.user_id.nunique()} people, {df.parent_asin.nunique():,} perfumes")
print("outcome: liked = rating 4 or 5  <- OUTCOME;  features: same 65 as the regression (see regression/amazon)")
print(f"share liked: {df.liked.mean():.3f}")

print("\n=== 2. Split ===")
print(f"30% of each person's ratings held out (seed 0): {len(tr):,} train / {len(te):,} test")

print("\n=== 3. Fit (depth and leaf size by 5-fold CV grouped by person, scored on balanced accuracy) ===")
grid = GridSearchCV(DecisionTreeClassifier(class_weight="balanced", random_state=0),
                    {"max_depth": [2, 3, 4, 5], "min_samples_leaf": [10, 25, 50, 100]},
                    cv=GroupKFold(5), scoring="balanced_accuracy").fit(tr[X], tr.liked, groups=tr.user_id)
tree = grid.best_estimator_
print(f"chosen: depth {tree.max_depth}, min leaf {tree.min_samples_leaf} (classes weighted so 'not liked' counts as much as 'liked')")
print(export_text(tree, feature_names=X, decimals=2))
imp = pd.Series(tree.feature_importances_, index=X).sort_values(ascending=False)
print("feature importance:"); print(imp[imp > 0].round(3).to_string())
fig, ax = plt.subplots(figsize=(11, 5))
plot_tree(tree, feature_names=X, class_names=["not liked", "liked"], filled=True, impurity=False, fontsize=7, ax=ax)
fig.tight_layout(); fig.savefig(HERE / "tree.png", dpi=150)

print("\n=== 4. Test ratings: baseline vs tree ===")
y = te.liked.to_numpy(); p = tree.predict(te[X])
majority = int(tr.liked.mean() >= 0.5)
print(classification_table(y, {"baseline: most common class (liked)": np.full(len(y), majority),
                               "tree": p}).to_string())
print("tree confusion matrix (test):"); print(confusion(y, p).to_string())
print(f"accuracy: training {(tree.predict(tr[X]) == tr.liked).mean():.3f} vs test {(p == y).mean():.3f}")
print(f"Note: the baseline is right {y.mean():.0%} of the time by always saying 'liked'. Balanced accuracy shows")
print("whether the tree can tell the two classes apart at all (0.5 = no better than a coin).")

print("\n=== 5. Decision: show a perfume if the tree says 'liked' ===")
row = te.iloc[0]
print(f"new input: person with average {row.person_avg:.2f} over {int(row.person_n)} ratings, perfume {row.brand} {row.perfume} ({row.accords})")
print(f"-> P(liked) {tree.predict_proba(te[X].iloc[[0]])[0, 1]:.2f}: {'show it' if p[0] == 1 else 'do not show'} (actual stars: {row.rating})")

print("\n=== 6. Value per month, baseline vs tree ===")
te = te.assign(prob=tree.predict_proba(te[X])[:, 1])
b, n = top_pick_liked(te.assign(s=te.fra_rating), "s")
m, _ = top_pick_liked(te, "prob")
print(f"top pick liked among each person's held-out perfumes: baseline (most popular) {b:.3f}, tree {m:.3f} ({n} people)")
print_assumptions()
print(value_table({"Baseline (popular)": b, "Tree": m}).to_string())

print("\n=== 7. Sanity check: shuffled outcome ===")
sh = DecisionTreeClassifier(class_weight="balanced", max_depth=tree.max_depth, min_samples_leaf=tree.min_samples_leaf,
                            random_state=0).fit(tr[X], np.random.default_rng(1).permutation(tr.liked.to_numpy()))
print(classification_table(y, {"shuffled-outcome tree": sh.predict(te[X])}).to_string())
