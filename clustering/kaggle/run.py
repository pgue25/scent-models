"""Clustering on Fragrantica (Kaggle) perfumes: group perfumes into scent families by accords.

PRODUCT LEVEL ONLY: no per-user data. Business question: how should we organise the catalog
(and the quiz) into a few scent families? Model: k-means on accord flags. Baseline: one group.
Run from the repo root:  python clustering/kaggle/run.py
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from shared.data import load_kaggle
from shared.clustering import elbow, elbow_k, elbow_chart

HERE = Path(__file__).parent
df, top = load_kaggle()
acc = [f"acc_{a}" for a in top]
tr, te = df[df.split == "train"], df[df.split == "test"]

print("=== 1. Data (REAL, product level) ===")
print(f"Fragrantica perfumes with 50+ ratings (Kaggle): {len(df):,}; one row per perfume; features: 20 accord flags")
print("no outcome (clustering); no per-user data")

print("\n=== 2. Split ===")
print(f"random 70/30 split of perfumes (seed 0): {len(tr):,} train / {len(te):,} test")

print("\n=== 3. Elbow: train, and test recomputed from the trained centers ===")
res = elbow(tr[acc].to_numpy(float), te[acc].to_numpy(float))
print(pd.DataFrame(res, index=["train", "test"]).T.round(3).to_string())
k_tr = elbow_k({k: v[0] for k, v in res.items()}); k_te = elbow_k({k: v[1] for k, v in res.items()})
print(f"elbow: train k = {k_tr}, test k = {k_te} -> {'SAME' if k_tr == k_te else 'DIFFERENT'}")
elbow_chart(res, "Fragrantica perfumes: elbow", HERE / "elbow.png")
sample = tr[acc].sample(5000, random_state=0)
for k in (2, 3, 4, 5, 6):
    lab = KMeans(k, n_init=10, random_state=0).fit_predict(sample)
    print(f"  silhouette k={k}: {silhouette_score(sample, lab):.3f}")

K = k_tr
km = KMeans(K, n_init=10, random_state=0).fit(tr[acc])
centers = pd.DataFrame(km.cluster_centers_, columns=top)
lab = km.labels_
print(f"\n=== 4. Scent families (k = {K}) ===")
for c in range(K):
    s = centers.loc[c].sort_values(ascending=False)
    members = tr[lab == c]
    print(f"  family {c}: {len(members):,} perfumes; mostly {', '.join(s.index[:4])}; "
          f"avg rating {members.rating.mean():.2f}; women {(members.gender == 'women').mean():.0%}, men {(members.gender == 'men').mean():.0%}")
print(f"training inertia {res[K][0]:.3f} vs {res[1][0]:.3f} with one group")

print("\n=== 5. Decision: put a new perfume into a family (shelf / quiz section) ===")
new = pd.DataFrame([{f"acc_{a}": int(a in ["citrus", "aromatic", "fresh spicy", "woody", "green"]) for a in top}])
print("new input: accords citrus, aromatic, fresh spicy, woody, green -> family", int(km.predict(new[acc])[0]))

print("\n=== 6. Value ===")
te_lab = km.predict(te[acc])
same = pd.Series(te_lab).value_counts(normalize=True).sort_index()
print("test perfumes per family:", ", ".join(f"family {i}: {v:.0%}" for i, v in same.items()))
print("These families are candidates for quiz sections: one or two everyday smells per family.")
print("No sales data exists, so no $ value is given for this model.")
