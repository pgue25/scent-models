# scent-models

Smell Optimizer v2 models (NYU Stern TECH-UB 24, Group 2). Three model families, each run on
each dataset where it makes sense, with shared data loading and scoring.

| | Amazon (real people x perfumes) | Kaggle / Fragrantica (perfumes only) |
|---|---|---|
| regression | predict a person's stars for a perfume | predict a perfume's average rating from its accords |
| decision tree | will this person like (4-5) this perfume | will a perfume average 4.0+ |
| clustering | taste personas of shoppers | scent families of perfumes |

Kaggle has no per-user data, so its models are about products, not people.

## Run

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python regression/amazon/run.py     # each run.py prints the 7 sections of the brief
.venv/bin/python regression/kaggle/run.py
.venv/bin/python tree/amazon/run.py
.venv/bin/python tree/kaggle/run.py
.venv/bin/python clustering/amazon/run.py
.venv/bin/python clustering/kaggle/run.py
```

Each folder keeps the script's printed output (`output.txt`) and charts. The data files are
committed; `data/build_*.py` rebuild them from the public sources (see `data/README.md`).

## Layout

```
data/            amazon_clean.csv, kaggle_products.csv, build scripts, README (sources and counts)
shared/          data.py (loading + features), evaluate.py (metrics, value), clustering.py (elbow)
regression/      amazon/run.py, kaggle/run.py
tree/            amazon/run.py, kaggle/run.py
clustering/      amazon/run.py, kaggle/run.py
docs/            v2 plan, survey draft, Amazon data findings
```

## Results so far (all on real data)

See `RESULTS.md`.
