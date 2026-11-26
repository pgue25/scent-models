"""Clustering on REAL Amazon ratings, generosity controlled: taste personas built from how much
each person rates each accord above or below their own (shrunk) normal.

Run from the repo root:  python clustering/amazon/run_centered.py
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from shared.data import load_amazon, SHRINK
from shared.clustering import elbow, elbow_k, elbow_chart
from shared.evaluate import ranking_table, value_table, print_assumptions

HERE = Path(__file__).parent
df, top = load_amazon()
acc = [f"acc_{a}" for a in top]
tr, te = df[df.split == "train"], df[df.split == "test"]
mu = tr.rating.mean()
g = tr.groupby("user_id").rating.agg(["sum", "count"])
normal = (g["sum"] + SHRINK * mu) / (g["count"] + SHRINK)


def profiles(rows):
    """person x accord: average (stars - normal) over their perfumes with that accord, 0 if none.
    `normal` always comes from TRAINING ratings."""
    r = rows.assign(dev=rows.rating - rows.user_id.map(normal))
    num = r[acc].mul(r.dev, axis=0).groupby(r.user_id).sum()
    cnt = r[acc].groupby(r.user_id).sum()
    return (num / cnt.replace(0, np.nan)).fillna(0.0).rename(columns=lambda c: c[4:])


P_tr = profiles(tr)
P_te = profiles(te).reindex(P_tr.index).fillna(0.0)

print("=== 1. Data (REAL, generosity controlled) ===")
print(f"one row per person: {len(P_tr)} people x 20 accord scores = average stars above/below own normal")
print(f"normal = training average shrunk toward {mu:.2f} by {SHRINK} pseudo-ratings. No outcome (clustering).")

print("\n=== 2. Split ===")
print(f"train profiles from {len(tr):,} ratings; test profiles from the held-out {len(te):,}")

print("\n=== 3. Elbow: train, and test from the trained centers ===")
res = elbow(P_tr.to_numpy(), P_te.to_numpy())
print(pd.DataFrame(res, index=["train", "test"]).T.round(3).to_string())
k_tr = elbow_k({k: v[0] for k, v in res.items()}); k_te = elbow_k({k: v[1] for k, v in res.items()})
print(f"elbow: train k = {k_tr}, test k = {k_te} -> {'SAME' if k_tr == k_te else 'DIFFERENT'}")
elbow_chart(res, "Amazon shoppers (centered): elbow", HERE / "elbow_centered.png")
for k in (2, 3, 4, 5):
    print(f"  silhouette k={k}: {silhouette_score(P_tr, KMeans(k, n_init=10, random_state=0).fit_predict(P_tr)):.3f}")
# stability: do personas come back when people are resampled?
from sklearn.metrics import adjusted_rand_score
rng = np.random.default_rng(0); base = KMeans(2, n_init=10, random_state=0).fit(P_tr)
ari = []
for _ in range(30):
    idx = rng.choice(len(P_tr), len(P_tr), replace=True)
    km_b = KMeans(2, n_init=10, random_state=0).fit(P_tr.iloc[idx])
    ari.append(adjusted_rand_score(base.labels_, km_b.predict(P_tr)))
print(f"  stability of k=2 over 30 resamples (adjusted Rand, 1 = identical): {np.mean(ari):.2f}")

K = k_tr
km = KMeans(K, n_init=10, random_state=0).fit(P_tr)
centers = pd.DataFrame(km.cluster_centers_, columns=P_tr.columns)
sizes = np.bincount(km.labels_, minlength=K)
print(f"\n=== 4. Personas (k = {K}) ===")
for c in range(K):
    s = centers.loc[c].sort_values()
    print(f"  persona {c}: {sizes[c]} people; above normal on {', '.join(s.index[::-1][:3])}; below on {', '.join(s.index[:3])}")

print("\n=== 5. Decision: assign to a persona, rank perfumes by that persona's accord scores ===")
lab = pd.Series(km.labels_, index=P_tr.index)
tt = te.assign(persona=te.user_id.map(lab))
tt["score"] = (tt[acc].to_numpy() * centers.loc[tt.persona].to_numpy()).sum(axis=1)
tt["score_tb"] = tt.score + 0.01 * tt.fra_rating
row = tt.iloc[0]
print(f"new input: person in persona {row.persona}, perfume {row.brand} {row.perfume} ({row.accords}) -> score {row.score:+.2f}")

print("\n=== 6. Ranking each person's held-out perfumes, and value ===")
t, ci, n = ranking_table(tt, {"baseline: perfume's average rating": "fra_rating", "personas": "score_tb"})
print(t.to_string()); print(f"people scored: {n}")
for m, (d, lo, hi) in ci.items():
    print(f"  {m}: personas minus baseline {d:+.3f} (95% CI {lo:+.3f} to {hi:+.3f})")
print_assumptions()
print(value_table({"Baseline (avg rating)": t.iloc[0].top_liked, "Personas (centered)": t.iloc[-1].top_liked}).to_string())
