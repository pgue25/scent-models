"""Elbow helpers shared by the clustering models."""
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

KS = range(1, 9)


def elbow(train, test, seed=0):
    """Average squared distance to the nearest center, for k = 1..8.
    Train: centers fit on train. Test: test points scored against those SAME trained centers."""
    out = {}
    for k in KS:
        km = KMeans(k, n_init=10, random_state=seed).fit(train)
        d_tr = km.transform(train).min(axis=1) ** 2
        d_te = km.transform(test).min(axis=1) ** 2
        out[k] = (d_tr.mean(), d_te.mean())
    return out


def elbow_k(curve):
    """k where the curve bends most: largest second difference of the inertia."""
    ks = sorted(curve); v = np.array([curve[k] for k in ks])
    second = v[:-2] - 2 * v[1:-1] + v[2:]
    return ks[1 + int(np.argmax(second))]


def elbow_chart(res, title, path):
    ks = sorted(res)
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(ks, [res[k][0] for k in ks], "o-", label="train")
    ax.plot(ks, [res[k][1] for k in ks], "s--", label="test (trained centers)")
    ax.set_xlabel("k (number of groups)"); ax.set_ylabel("avg squared distance to center")
    ax.set_title(title); ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=150)
