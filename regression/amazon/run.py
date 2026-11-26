"""Regression on REAL Amazon fragrance ratings: predict a person's 1-5 stars for a perfume.

Business question: which perfumes should we show this person so they like what they see?
Model: ridge regression. Baseline: the average rating (as the brief defines it).
Run from the repo root:  python regression/amazon/run.py
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from shared.data import load_amazon, person_features, amazon_feature_names
from shared.evaluate import (regression_table, adj_r2, top_pick_liked, value_table, print_assumptions, bootstrap_diff)

HERE = Path(__file__).parent
df, top = load_amazon()
df = person_features(df, top)
X = amazon_feature_names(top)
tr, te = df[df.split == "train"], df[df.split == "test"]

print("=== 1. Data (REAL) ===")
print("source: Amazon Reviews 2023 (McAuley Lab), fragrance reviews 2003-2023, filtered to reviews about")
print("the smell and matched to Fragrantica accords (Kaggle). See data/README.md.")
print(f"one row per (person, perfume): {len(df):,} ratings, {df.user_id.nunique()} people, {df.parent_asin.nunique():,} perfumes")
print("outcome: rating (1-5 stars)  <- OUTCOME")
print(f"features ({len(X)}): person_avg, person_n (from the person's other training ratings), fra_rating,")
print(f"  fra_log_count (Fragrantica), acc_* (perfume has accord), taste_* (person's liking of accord vs own avg),")
print(f"  match_* = taste x accord, match_total. Accords: {', '.join(top)}")

print("\n=== 2. Split ===")
print(f"30% of each person's ratings held out (seed 0): {len(tr):,} train / {len(te):,} test")

print("\n=== 3. Fit (ridge, alpha by 5-fold CV grouped by person) ===")
alphas = np.logspace(-2, 3, 16)
cv = list(GroupKFold(5).split(tr[X], tr.rating, tr.user_id))
reg = RidgeCV(alphas=alphas, cv=cv).fit(tr[X], tr.rating)
r2_train = reg.score(tr[X], tr.rating)
print(f"alpha {reg.alpha_:.3g}; training R^2 {r2_train:.3f}, adjusted R^2 {adj_r2(r2_train, len(tr), len(X)):.3f}")
sd = tr[X].std().replace(0, 1)
eff = pd.Series(reg.coef_ * sd, index=X)
print("biggest effects (stars per 1 standard deviation of the feature):")
print(eff.reindex(eff.abs().sort_values(ascending=False).index).head(10).round(3).to_string())

print("\n=== 4. Test ratings: baseline vs model ===")
te = te.assign(pred=reg.predict(te[X]))
base_overall = np.full(len(te), tr.rating.mean())
small = ["person_avg", "person_n", "fra_rating", "fra_log_count"]
reg_small = RidgeCV(alphas=alphas, cv=cv).fit(tr[small], tr.rating)
te = te.assign(pred_small=reg_small.predict(te[small]))
print(regression_table(te.rating.to_numpy(), {
    "baseline: overall average": base_overall,
    "person's own average": te.person_avg.to_numpy(),
    "model without scent taste": te.pred_small.to_numpy(),
    "model (full)": te.pred.to_numpy()}).to_string())
err = te.assign(e_full=(te.pred.clip(1, 5) - te.rating).abs(), e_small=(te.pred_small.clip(1, 5) - te.rating).abs(),
                e_base=(base_overall - te.rating).abs()).groupby("user_id")[["e_full", "e_small", "e_base"]].mean()
print("MAE, full minus baseline, by person:           %.3f (95%% CI %.3f to %.3f)" % bootstrap_diff(err.e_full, err.e_base))
print("MAE, full minus model without scent taste:     %.3f (95%% CI %.3f to %.3f)" % bootstrap_diff(err.e_full, err.e_small))

fig, ax = plt.subplots(figsize=(5, 3.2))
names = ["Baseline\n(average)", "Person's\naverage", "Model w/o\nscent", "Model\n(full)"]
vals = [np.abs(base_overall - te.rating).mean(), (te.person_avg - te.rating).abs().mean(),
        (te.pred_small.clip(1, 5) - te.rating).abs().mean(), (te.pred.clip(1, 5) - te.rating).abs().mean()]
ax.bar(names, vals, color=["#bbb", "#bbb", "#88a", "#2a5"]); ax.set_ylabel("typical error (stars)")
ax.set_title("Test ratings: average miss, lower is better"); fig.tight_layout(); fig.savefig(HERE / "test_mae.png", dpi=150)

print("\n=== 5. Decision: show the 3 perfumes with the highest predicted rating ===")
def recommend(user_id, k=3):
    """Score every perfume in the data for a known person and return the top k they have not rated."""
    seen = set(df.loc[df.user_id == user_id, "parent_asin"])
    person = df[(df.user_id == user_id) & (df.split == "test")].iloc[0]
    cand = df.drop_duplicates("parent_asin").loc[lambda d: ~d.parent_asin.isin(seen)].copy()
    for c in ["person_avg", "person_n"] + [f"taste_{a}" for a in top]:
        cand[c] = person[c]
    for a in top:
        cand[f"match_{a}"] = cand[f"taste_{a}"] * cand[f"acc_{a}"]
    cand["match_total"] = cand[[f"match_{a}" for a in top]].sum(axis=1)
    cand["predicted stars"] = reg.predict(cand[X]).clip(1, 5).round(2)
    return cand.nlargest(k, "predicted stars")[["brand", "perfume", "accords", "predicted stars"]]
uid = te.groupby("user_id").size().idxmax()
p = df[(df.user_id == uid) & (df.split == "train")]
print(f"new input: a test person with {len(p)} known ratings (avg {p.rating.mean():.2f});")
print("  their 5-star perfumes:", "; ".join((p[p.rating == 5].perfume.head(5))))
print(recommend(uid).to_string(index=False))

print("\n=== 6. Value per month, baseline vs model ===")
print("Each test person's held-out perfumes are ranked; is the top one liked (4-5 stars)?")
b, n = top_pick_liked(te.assign(s=te.fra_rating), "s")
m, _ = top_pick_liked(te, "pred")
print(f"top pick liked: baseline (most popular by Fragrantica rating) {b:.3f}, model {m:.3f}  ({n} people with 2+ test ratings)")
print_assumptions()
print(value_table({"Baseline (popular)": b, "Regression": m}).to_string())

print("\n=== 7. Sanity check: shuffle the outcome within training rows and refit ===")
rng = np.random.default_rng(1)
shuf = RidgeCV(alphas=alphas, cv=cv).fit(tr[X], rng.permutation(tr.rating.to_numpy()))
print(regression_table(te.rating.to_numpy(), {"shuffled-outcome model": shuf.predict(te[X])}).to_string())
print("(should be no better than the baseline)")
