"""Regression on REAL Amazon ratings, generosity controlled.

Target = stars minus the person's own "normal" (their shrunk training average), so the model
has to learn which perfumes a person rates ABOVE their usual, not how generous they are.
Scored by ranking each person's held-out perfumes. Baseline: rank by the perfume's average
rating (Fragrantica). Run from the repo root:  python regression/amazon/run_centered.py
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from shared.data import load_amazon, person_features_centered, centered_feature_names, SHRINK
from shared.evaluate import regression_table, adj_r2, ranking_table, value_table, print_assumptions

HERE = Path(__file__).parent
df, top = load_amazon()
df = person_features_centered(df, top)
X, X0 = centered_feature_names(top), centered_feature_names(top, scent=False)
tr, te = df[df.split == "train"], df[df.split == "test"]

print("=== 1. Data (REAL, generosity controlled) ===")
print(f"Amazon fragrance ratings about the smell, matched to Fragrantica: {len(df):,} ratings, {df.user_id.nunique()} people")
print(f"normal = person's training average, pulled toward the overall average by {SHRINK} pseudo-ratings")
print("outcome: target = stars - normal  <- OUTCOME  (positive = they liked it more than usual)")
print(f"features ({len(X)}): person_n, fra_rating, fra_log_count, acc_*, ctaste_* (liking of accord vs own normal), cmatch_*")
print(f"target: mean {tr.target.mean():+.2f}, sd {tr.target.std():.2f} (train)")

print("\n=== 2. Split ===")
print(f"30% of each person's ratings held out (seed 0): {len(tr):,} train / {len(te):,} test")

print("\n=== 3. Fit (ridge, alpha by CV grouped by person) ===")
cv = list(GroupKFold(5).split(tr[X], tr.target, tr.user_id)); alphas = np.logspace(-2, 3, 16)
reg = RidgeCV(alphas=alphas, cv=cv).fit(tr[X], tr.target)
reg0 = RidgeCV(alphas=alphas, cv=cv).fit(tr[X0], tr.target)
r2 = reg.score(tr[X], tr.target)
print(f"alpha {reg.alpha_:.3g}; training R^2 {r2:.3f}, adjusted R^2 {adj_r2(r2, len(tr), len(X)):.3f}")
eff = pd.Series(reg.coef_ * tr[X].std().replace(0, 1), index=X)
print("biggest effects (stars above normal per 1 sd of the feature):")
print(eff.reindex(eff.abs().sort_values(ascending=False).index).head(10).round(3).to_string())

print("\n=== 4. Test ratings ===")
te = te.assign(pred=reg.predict(te[X]), pred0=reg0.predict(te[X0]))
print("(a) predicting stars above normal:")
print(regression_table(te.target.to_numpy(), {"baseline: 0 (rate everything at normal)": np.zeros(len(te)),
                                              "model without scent": te.pred0.to_numpy(),
                                              "model with scent": te.pred.to_numpy()}).rename(columns={"MAE (stars)": "MAE"}).to_string())
print("(b) ranking each person's held-out perfumes (what the product actually does):")
t, ci, n = ranking_table(te, {"baseline: perfume's average rating": "fra_rating",
                              "model without scent": "pred0", "model with scent": "pred"})
print(t.to_string()); print(f"people scored: {n} (2+ held-out perfumes with different stars)")
for m, (d, lo, hi) in ci.items():
    print(f"  {m}: model with scent minus baseline {d:+.3f} (95% CI {lo:+.3f} to {hi:+.3f})")
t2, ci2, _ = ranking_table(te, {"model without scent": "pred0", "model with scent": "pred"})
d, lo, hi = ci2["top_is_best"]
print(f"  top_is_best: scent minus no-scent {d:+.3f} (95% CI {lo:+.3f} to {hi:+.3f})")
fig, ax = plt.subplots(figsize=(5, 3.2))
t["top_is_best"].plot.bar(ax=ax, color=["#bbb", "#88a", "#2a5"], rot=0)
ax.set_xticklabels(["Baseline\n(avg rating)", "No scent", "With scent"]); ax.set_ylabel("share of people")
ax.set_title("Top-ranked held-out perfume is their favourite"); fig.tight_layout(); fig.savefig(HERE / "centered_ranking.png", dpi=150)

print("\n=== 5. Decision: show the perfumes predicted furthest above the person's normal ===")
row = te.iloc[0]
print(f"new input: {row.brand} {row.perfume} ({row.accords}) for a person with normal {row.normal:.2f}")
print(f"-> predicted {row.pred:+.2f} stars vs their normal: {'show' if row.pred > 0 else 'do not show'} (actual {row.target:+.2f})")

print("\n=== 6. Value per month (top pick liked), baseline vs model ===")
b, m = t.loc["baseline: perfume's average rating", "top_liked"], t.loc["model with scent", "top_liked"]
print_assumptions()
print(value_table({"Baseline (avg rating)": b, "Regression (centered)": m}).to_string())

print("\n=== 7. Sanity check: shuffled outcome ===")
sh = RidgeCV(alphas=alphas, cv=cv).fit(tr[X], np.random.default_rng(1).permutation(tr.target.to_numpy()))
ts, _, _ = ranking_table(te.assign(sh=sh.predict(te[X])), {"baseline": "fra_rating", "shuffled": "sh"})
print(ts.to_string())
