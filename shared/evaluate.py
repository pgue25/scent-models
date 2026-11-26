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
