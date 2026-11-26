"""Clustering on REAL Amazon fragrance ratings: group people into taste personas.

Business question: which group of shoppers is this person in, so we know which perfumes to show?
Each person is described by how much more or less than their own average they rate perfumes
with each accord (their taste profile). Model: k-means. Baseline: one group (everyone the same).
Run from the repo root:  python clustering/amazon/run.py
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from shared.data import load_amazon
from shared.clustering import elbow, elbow_k, elbow_chart
from shared.evaluate import top_pick_liked, value_table, print_assumptions

HERE = Path(__file__).parent
df, top = load_amazon()
acc = [f"acc_{a}" for a in top]


def profiles(rows, center):
    """person x accord: average (stars - person's TRAINING average) over their perfumes with that accord; 0 if none."""
    r = rows.assign(dev=rows.rating - rows.user_id.map(center))
    num = (r[acc].mul(r.dev, axis=0)).groupby(r.user_id).sum()
    cnt = r[acc].groupby(r.user_id).sum()
    return (num / cnt.replace(0, np.nan)).fillna(0.0).rename(columns=lambda c: c[4:])


tr, te = df[df.split == "train"], df[df.split == "test"]
avg = tr.groupby("user_id").rating.mean()
P_tr = profiles(tr, avg)
P_te = profiles(te, avg).reindex(P_tr.index).fillna(0.0)

print("=== 1. Data (REAL) ===")
print("source: Amazon Reviews 2023 fragrance reviews about the smell, matched to Fragrantica accords.")
print(f"one row per person: {len(P_tr)} people, each described by 20 accord taste scores (no outcome: clustering)")
print("taste score = average of (stars - own average) over the person's perfumes with that accord")

print("\n=== 2. Split ===")
print(f"30% of each person's ratings held out: train profiles from {len(tr):,} ratings, test profiles from {len(te):,}")

print("\n=== 3. Elbow: train, and test recomputed from the trained centers ===")
res = elbow(P_tr.to_numpy(), P_te.to_numpy())
print(pd.DataFrame(res, index=["train", "test"]).T.round(3).to_string())
k_tr = elbow_k({k: v[0] for k, v in res.items()}); k_te = elbow_k({k: v[1] for k, v in res.items()})
print(f"elbow: train k = {k_tr}, test k = {k_te} -> {'SAME' if k_tr == k_te else 'DIFFERENT'}")
elbow_chart(res, "Amazon shoppers: elbow", HERE / "elbow.png")
for k in (2, 3, 4, 5):
    lab = KMeans(k, n_init=10, random_state=0).fit_predict(P_tr)
    print(f"  silhouette k={k}: {silhouette_score(P_tr, lab):.3f}  (near 0 = groups overlap; above 0.25 = real structure)")

K = k_tr
km = KMeans(K, n_init=10, random_state=0).fit(P_tr)
centers = pd.DataFrame(km.cluster_centers_, columns=P_tr.columns)
sizes = np.bincount(km.labels_, minlength=K)
print(f"\n=== 4. Personas (k = {K}) ===")
for c in range(K):
    s = centers.loc[c].sort_values()
    print(f"  persona {c}: {sizes[c]} people; rates higher: {', '.join(s.index[::-1][:3])}; rates lower: {', '.join(s.index[:3])}")
print(f"training inertia {res[K][0]:.3f} vs {res[1][0]:.3f} with one group")

print("\n=== 5. Decision: assign a person to a persona, show perfumes with that persona's favourite accords ===")
def recommend(profile, k=3):
    c = int(km.predict(profile.to_frame().T)[0])
    cand = df.drop_duplicates("parent_asin")
    score = cand[acc].to_numpy() @ centers.loc[c].to_numpy() + 0.1 * cand.fra_rating.to_numpy()
    return c, cand.assign(score=score).nlargest(k, "score")[["brand", "perfume", "accords"]]
uid = P_tr.index[0]
c, rec = recommend(P_tr.loc[uid])
print(f"new input: a person whose strongest likes are {', '.join(P_tr.loc[uid].sort_values()[::-1].index[:3])} -> persona {c}")
print(rec.to_string(index=False))

print("\n=== 6. Value per month, baseline (one group = most popular) vs personas ===")
lab = pd.Series(km.labels_, index=P_tr.index)
tt = te.assign(persona=te.user_id.map(lab))
tt["score"] = (tt[acc].to_numpy() * centers.loc[tt.persona].to_numpy()).sum(axis=1) + 0.1 * tt.fra_rating
b, n = top_pick_liked(tt.assign(s=tt.fra_rating), "s")
m, _ = top_pick_liked(tt, "score")
print(f"top pick liked among each person's held-out perfumes: one group {b:.3f}, personas {m:.3f} ({n} people)")
print_assumptions()
print(value_table({"Baseline (one group)": b, "Personas": m}).to_string())
