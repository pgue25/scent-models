"""Scoring shared by every model, so results are comparable."""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, r2_score

# MADE-UP business assumptions (same as v1), labelled wherever they are printed.
ASSUME = {"quiz users per month": 5000, "products shown per user": 3,
          "P(purchase | shown product is liked)": 0.05, "P(purchase | not liked)": 0.005,
          "commission per purchase ($)": 8.0}


def adj_r2(r2, n, k):
    return 1 - (1 - r2) * (n - 1) / (n - k - 1)


def regression_table(y, preds):
    rows = {name: {"MAE (stars)": np.abs(np.clip(p, 1, 5) - y).mean(), "R^2": r2_score(y, p)} for name, p in preds.items()}
    return pd.DataFrame(rows).T.round(3)


def classification_table(y, preds):
    rows = {}
    for name, p in preds.items():
        rows[name] = {"accuracy": accuracy_score(y, p), "precision": precision_score(y, p, zero_division=0),
                      "recall": recall_score(y, p, zero_division=0),
                      "balanced acc.": 0.5 * (recall_score(y, p, zero_division=0) + recall_score(1 - y, 1 - p, zero_division=0))}
    return pd.DataFrame(rows).T.round(3)


def confusion(y, p):
    return pd.DataFrame(confusion_matrix(y, p, labels=[0, 1]), index=["actual: not liked", "actual: liked"],
                        columns=["pred: not liked", "pred: liked"])


def top_pick_liked(test, score_col, k=1, min_items=2):
    """Among each test person's held-out products, is the one the model ranks first liked (4-5)?
    Returns the share of people for whom it is. Ties broken at random (seeded)."""
    t = test.assign(_tie=np.random.default_rng(0).random(len(test)))
    t = t[t.groupby("user_id").user_id.transform("size") >= min_items]
    t = t.sort_values(["user_id", score_col, "_tie"], ascending=[True, False, True])
    return t.groupby("user_id").head(k).liked.mean(), t.user_id.nunique()


def value_per_month(liked_share):
    a = ASSUME
    shown = a["quiz users per month"] * a["products shown per user"]
    liked = shown * liked_share
    buys = liked * a["P(purchase | shown product is liked)"] + (shown - liked) * a["P(purchase | not liked)"]
    return {"products shown": shown, "liked products shown": round(liked, 1),
            "purchases": round(buys, 1), "commission ($)": round(buys * a["commission per purchase ($)"], 1)}


def value_table(shares):
    t = pd.DataFrame({k: value_per_month(v) for k, v in shares.items()}).T
    names = list(shares)
    t.loc[f"{names[-1]} minus {names[0]}"] = t.loc[names[-1]] - t.loc[names[0]]
    return t


def print_assumptions():
    print("  ASSUMPTIONS ARE MADE UP:")
    for k, v in ASSUME.items():
        print(f"    {k} = {v}")


def bootstrap_diff(a, b, n_boot=2000, seed=0):
    """Mean of a - b over paired units (e.g. people), with a 95% bootstrap interval."""
    d = np.asarray(a, float) - np.asarray(b, float)
    idx = np.random.default_rng(seed).integers(0, len(d), (n_boot, len(d)))
    lo, hi = np.percentile(d[idx].mean(axis=1), [2.5, 97.5])
    return d.mean(), lo, hi


def within_person_ranking(test, score_col, n_shuffles=50, seed=0):
    """Rank each person's held-out perfumes by `score_col` and compare with their real stars.
    Only people with 2+ held-out perfumes that are not all the same stars count.

    Returns per-person columns:
      top_is_best   the perfume ranked first is one of the person's highest-rated held-out ones
      top_liked     the perfume ranked first got 4-5 stars
      spearman      rank correlation between score and stars within the person
    Ties in the score are broken at random and averaged over `n_shuffles` draws, so a model
    that gives everyone the same score is scored like a coin, not like a lucky order."""
    rng = np.random.default_rng(seed)
    rows = []
    for uid, g in test.groupby("user_id"):
        if len(g) < 2 or g.rating.nunique() < 2:
            continue
        stars, sc = g.rating.to_numpy(float), g[score_col].to_numpy(float)
        best, liked, rho = [], [], []
        for _ in range(n_shuffles):
            order = np.lexsort((rng.random(len(g)), -sc))
            top = order[0]
            best.append(stars[top] == stars.max()); liked.append(stars[top] >= 4)
            ranks = np.empty(len(g)); ranks[order] = np.arange(len(g))
            rho.append(pd.Series(-ranks).corr(pd.Series(stars), method="spearman"))
        rows.append({"user_id": uid, "top_is_best": np.mean(best), "top_liked": np.mean(liked),
                     "spearman": np.nanmean(rho)})
    return pd.DataFrame(rows).set_index("user_id")


def ranking_table(test, scores):
    """scores: {name: column}. Mean of within_person_ranking per model, plus bootstrap of the
    difference between the last model and the first (the baseline)."""
    per = {name: within_person_ranking(test, col) for name, col in scores.items()}
    t = pd.DataFrame({n: p.mean() for n, p in per.items()}).T.round(3)
    names = list(scores)
    ci = {m: bootstrap_diff(per[names[-1]][m], per[names[0]][m]) for m in ("top_is_best", "top_liked", "spearman")}
    return t, ci, len(per[names[0]])
