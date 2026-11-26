"""Loading and feature building shared by every model. Models never touch raw files."""
from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
N_ACCORDS = 20
SEED = 0


def accord_matrix(accords, top):
    """0/1 matrix: does each product list each of the `top` accords among its top 5."""
    sets = accords.fillna("").str.split(";").map(set)
    return pd.DataFrame({f"acc_{a}": sets.map(lambda s: int(a in s)) for a in top}, index=accords.index)


def top_accords(accords, n=N_ACCORDS):
    return accords.fillna("").str.split(";").explode().loc[lambda s: s != ""].value_counts().index[:n].tolist()


# ---------------------------------------------------------------- Amazon (people x products)
def load_amazon():
    """Real Amazon fragrance ratings (see data/README.md). One row per (person, product).
    `split` holds out 30% of each person's ratings as test."""
    df = pd.read_csv(DATA / "amazon_clean.csv")
    df["fra_rating"] = df.fra_rating.fillna(df.fra_rating.median())
    df["fra_log_count"] = np.log1p(df.fra_count.fillna(df.fra_count.median()))
    df["liked"] = (df.rating >= 4).astype(int)
    top = top_accords(df.drop_duplicates("parent_asin").accords)
    return pd.concat([df, accord_matrix(df.accords, top)], axis=1), top


def person_features(df, top):
    """Features describing each person, built from TRAINING ratings only.

    For a training row the row's own rating is left out (leave-one-out), so the model never
    sees the answer it is predicting. For a test row all of the person's training ratings are used.
      person_avg       average stars
      person_n         number of ratings used
      taste_<accord>   average of (stars - person_avg) over their products with that accord
      match_<accord>   taste_<accord> x the product has that accord
      match_total      sum of match_<accord>
    """
    acc = [f"acc_{a}" for a in top]
    tr = df[df.split == "train"]
    S = tr.groupby("user_id").rating.agg(["sum", "count"])
    # per person: sum of stars and count on products with each accord
    A = tr[acc].to_numpy(float)
    r = tr.rating.to_numpy(float)
    users = tr.user_id.to_numpy()
    sumA = pd.DataFrame(A * r[:, None], index=users, columns=acc).groupby(level=0).sum()
    cntA = pd.DataFrame(A, index=users, columns=acc).groupby(level=0).sum()

    out = df.copy()
    is_tr = (out.split == "train").to_numpy()
    own = np.where(is_tr, out.rating, 0.0)
    own_acc = out[acc].to_numpy(float) * is_tr[:, None]
    s = S.loc[out.user_id, "sum"].to_numpy() - own
    c = S.loc[out.user_id, "count"].to_numpy() - is_tr
    avg = s / c
    sa = sumA.loc[out.user_id].to_numpy() - own_acc * out.rating.to_numpy()[:, None]
    ca = cntA.loc[out.user_id].to_numpy() - own_acc
    taste = np.divide(sa - ca * avg[:, None], ca, out=np.zeros_like(sa), where=ca > 0)
    out["person_avg"], out["person_n"] = avg, c
    for j, a in enumerate(top):
        out[f"taste_{a}"] = taste[:, j]
        out[f"match_{a}"] = taste[:, j] * out[f"acc_{a}"]
    out["match_total"] = out[[f"match_{a}" for a in top]].sum(axis=1)
    return out


def amazon_feature_names(top):
    return (["person_avg", "person_n", "fra_rating", "fra_log_count"] + [f"acc_{a}" for a in top]
            + [f"taste_{a}" for a in top] + [f"match_{a}" for a in top] + ["match_total"])


# ---------------------------------------------------------------- Kaggle (products only)
def load_kaggle(min_ratings=50):
    """Fragrantica perfumes (product level only: no per-user data exists in this source).
    Keeps perfumes with at least `min_ratings` ratings so the average is meaningful.
    Random 70/30 split of perfumes."""
    df = pd.read_csv(DATA / "kaggle_products.csv")
    df = df[(df.rating_count >= min_ratings) & (df.accords.fillna("") != "")].reset_index(drop=True)
    top = top_accords(df.accords)
    df = pd.concat([df, accord_matrix(df.accords, top)], axis=1)
    for g in ("women", "men", "women and men"):
        df[f"gender_{g.replace(' ', '_')}"] = (df.gender == g).astype(int)
    df["log_count"] = np.log1p(df.rating_count)
    test = np.random.default_rng(SEED).random(len(df)) < 0.3
    df["split"] = np.where(test, "test", "train")
    return df, top
