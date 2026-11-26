"""Regression on Fragrantica (Kaggle) perfumes: predict a perfume's AVERAGE rating from its accords.

PRODUCT LEVEL ONLY. This source has no per-user ratings, so it cannot personalise anything.
Business question: before we add a perfume to the catalog, how well will people rate it?
Model: linear regression (ridge). Baseline: the average rating.
Run from the repo root:  python regression/kaggle/run.py
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from shared.data import load_kaggle
from shared.evaluate import regression_table, adj_r2, bootstrap_diff

HERE = Path(__file__).parent
df, top = load_kaggle()
X = [f"acc_{a}" for a in top] + ["gender_women", "gender_men", "log_count"]
tr, te = df[df.split == "train"], df[df.split == "test"]

print("=== 1. Data (REAL, product level) ===")
print("source: Fragrantica perfumes via Kaggle archive (fra_perfumes.csv), scraped by the dataset author.")
print(f"one row per perfume with 50+ ratings: {len(df):,} perfumes, {df.brand.nunique():,} brands")
print("outcome: rating = Fragrantica average rating (1-5)  <- OUTCOME")
print(f"features: 20 accord flags ({', '.join(top)}), gender (women / men / unisex), log number of ratings")
print("no per-user data: this model cannot recommend for a specific person")

print("\n=== 2. Split ===")
print(f"random 70/30 split of perfumes (seed 0): {len(tr):,} train / {len(te):,} test")

print("\n=== 3. Fit ===")
reg = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(tr[X], tr.rating)
r2 = reg.score(tr[X], tr.rating)
print(f"training R^2 {r2:.3f}, adjusted R^2 {adj_r2(r2, len(tr), len(X)):.3f}")
coef = pd.Series(reg.coef_, index=X)
print("effect on average rating (stars) of having each accord / attribute:")
print(coef.sort_values(ascending=False).round(3).to_string())

print("\n=== 4. Test perfumes: baseline vs model ===")
pred = reg.predict(te[X]); base = np.full(len(te), tr.rating.mean())
print(regression_table(te.rating.to_numpy(), {"baseline: average rating": base, "regression": pred}).to_string())
print("MAE difference, model minus baseline: %.4f (95%% CI %.4f to %.4f)" %
      bootstrap_diff(np.abs(pred - te.rating), np.abs(base - te.rating)))
print(f"for scale: the spread (sd) of perfume ratings is only {df.rating.std():.2f} stars")
fig, ax = plt.subplots(figsize=(5, 3.2))
coef[[f"acc_{a}" for a in top]].sort_values().plot.barh(ax=ax, color="#2a5")
ax.set_xlabel("stars added to average rating"); ax.set_title("Which accords people rate higher")
fig.tight_layout(); fig.savefig(HERE / "accord_effects.png", dpi=150)

print("\n=== 5. Decision: stock a new perfume if its predicted rating is 4.0 or more ===")
def decide(accords, gender="women and men", n_ratings=500):
    row = {f"acc_{a}": int(a in accords) for a in top}
    row.update(gender_women=int(gender == "women"), gender_men=int(gender == "men"), log_count=np.log1p(n_ratings))
    p = float(reg.predict(pd.DataFrame([row])[X])[0])
    return round(p, 2), ("stock it" if p >= 4.0 else "pass")
new = ["citrus", "woody", "aromatic", "fresh spicy", "green"]
print(f"new input: unisex perfume with accords {new}, 500 ratings ->", decide(new))

print("\n=== 6. Value: average real rating of the perfumes each rule would stock (test perfumes) ===")
stock_model = te[pred >= 4.0]
print(f"no model (stock everything): {len(te):,} perfumes, average rating {te.rating.mean():.3f}")
print(f"model (stock if predicted 4.0+): {len(stock_model):,} perfumes, average rating {stock_model.rating.mean():.3f}")
print("No sales or revenue data exists for these perfumes, so no $ figure is given.")

print("\n=== 7. Sanity check: shuffled outcome ===")
shuf = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(tr[X], np.random.default_rng(1).permutation(tr.rating.to_numpy()))
print(regression_table(te.rating.to_numpy(), {"shuffled-outcome model": shuf.predict(te[X])}).to_string())
