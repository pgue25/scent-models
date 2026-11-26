"""Decision tree on REAL Amazon ratings, generosity controlled:
did this person rate this perfume ABOVE their own normal?

Run from the repo root:  python tree/amazon/run_centered.py
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from shared.data import load_amazon, person_features_centered, centered_feature_names, SHRINK
from shared.evaluate import classification_table, confusion, ranking_table, value_table, print_assumptions

HERE = Path(__file__).parent
df, top = load_amazon()
df = person_features_centered(df, top)
X, X0 = centered_feature_names(top), centered_feature_names(top, scent=False)
tr, te = df[df.split == "train"], df[df.split == "test"]

print("=== 1. Data (REAL, generosity controlled) ===")
print(f"{len(df):,} ratings, {df.user_id.nunique()} people; normal = training average shrunk by {SHRINK} pseudo-ratings")
print("outcome: above_normal = stars > the person's normal  <- OUTCOME")
print(f"share above normal: train {tr.above_normal.mean():.3f}, test {te.above_normal.mean():.3f}")

print("\n=== 2. Split ===")
print(f"30% of each person's ratings held out (seed 0): {len(tr):,} train / {len(te):,} test")

print("\n=== 3. Fit (depth, leaf size by CV grouped by person) ===")
def fit(cols, weight=None):
    return GridSearchCV(DecisionTreeClassifier(random_state=0, class_weight=weight),
                        {"max_depth": [2, 3, 4, 5], "min_samples_leaf": [10, 25, 50, 100]},
                        cv=GroupKFold(5), scoring="accuracy").fit(tr[cols], tr.above_normal, groups=tr.user_id).best_estimator_
tree, tree0 = fit(X), fit(X0)
treeb = fit(X, "balanced")  # same tree, but 'at/below normal' rows count as much as 'above'
print(f"balanced-class tree: depth {treeb.max_depth}, min leaf {treeb.min_samples_leaf}")
print(export_text(treeb, feature_names=X, decimals=2))
print(f"chosen: depth {tree.max_depth}, min leaf {tree.min_samples_leaf}")
print(export_text(tree, feature_names=X, decimals=2))
imp = pd.Series(tree.feature_importances_, index=X).sort_values(ascending=False)
print("feature importance:"); print(imp[imp > 0].round(3).to_string())
fig, ax = plt.subplots(figsize=(11, 5))
plot_tree(tree, feature_names=X, class_names=["at/below normal", "above normal"], filled=True, impurity=False, fontsize=7, ax=ax)
fig.tight_layout(); fig.savefig(HERE / "tree_centered.png", dpi=150)

print("\n=== 4. Test ratings ===")
y = te.above_normal.to_numpy(); majority = int(tr.above_normal.mean() >= .5)
print(classification_table(y, {"baseline: most common class (above)": np.full(len(y), majority),
                               "tree without scent": tree0.predict(te[X0]), "tree with scent": tree.predict(te[X]),
                               "tree with scent, balanced classes": treeb.predict(te[X])}).to_string())
print("tree with scent, confusion matrix:"); print(confusion(y, tree.predict(te[X])).rename(
    index=lambda s: s.replace("not liked", "at/below").replace("liked", "above"),
    columns=lambda s: s.replace("not liked", "at/below").replace("liked", "above")).to_string())
print(f"accuracy: training {(tree.predict(tr[X]) == tr.above_normal).mean():.3f} vs test {(tree.predict(te[X]) == y).mean():.3f}")
te = te.assign(p=tree.predict_proba(te[X])[:, 1], p0=tree0.predict_proba(te[X0])[:, 1], pb=treeb.predict_proba(te[X])[:, 1])
t, ci, n = ranking_table(te, {"baseline: perfume's average rating": "fra_rating", "tree without scent": "p0",
                              "tree with scent, balanced": "pb", "tree with scent": "p"})
print("ranking each person's held-out perfumes by P(above normal):"); print(t.to_string()); print(f"people scored: {n}")
for m, (d, lo, hi) in ci.items():
    print(f"  {m}: tree with scent minus baseline {d:+.3f} (95% CI {lo:+.3f} to {hi:+.3f})")

print("\n=== 5. Decision: show a perfume if the tree says 'above normal' ===")
row = te.iloc[0]
print(f"new input: {row.brand} {row.perfume} ({row.accords}) -> P(above normal) {row.p:.2f}: "
      f"{'show' if row.p >= .5 else 'do not show'} (actual {'above' if row.above_normal else 'at/below'})")

print("\n=== 6. Value per month (top pick liked) ===")
print_assumptions()
print(value_table({"Baseline (avg rating)": t.iloc[0].top_liked, "Tree (centered)": t.iloc[-1].top_liked}).to_string())

print("\n=== 7. Sanity check: shuffled outcome ===")
sh = DecisionTreeClassifier(max_depth=tree.max_depth, min_samples_leaf=tree.min_samples_leaf, random_state=0).fit(
    tr[X], np.random.default_rng(1).permutation(tr.above_normal.to_numpy()))
print(classification_table(y, {"shuffled-outcome tree": sh.predict(te[X])}).to_string())
