# CLAUDE.md: scent-models

Smell Optimizer v2 models for NYU Stern TECH-UB 24 (Group 2). Owner: Paul (pgue25).
The full v2 plan, with open questions, is `docs/v2_plan.md`. Results so far: `RESULTS.md`.

## Layout rules
- One folder per model family (`regression/`, `tree/`, `clustering/`), one subfolder per
  dataset (`amazon/`, `kaggle/`, later `survey/`). Each subfolder has `run.py`, its `output.txt`
  and charts.
- Data loading and features live only in `shared/data.py`; metrics and the value calculation
  only in `shared/evaluate.py`; elbow code in `shared/clustering.py`. Model scripts import
  these and never read raw files, so every model is scored the same way.
- Data files are built by `data/build_*.py` from public sources; raw downloads go in
  `data/raw/` (git-ignored). Document every filter and its row counts in `data/README.md`.

## Every run.py prints the brief's 7 sections
1 data (source, rows, one row per what, outcome marked), 2 split, 3 fit read in business
terms, 4 test baseline vs model with the family's statistics, 5 a decision function run on a
new input, 6 value vs baseline (assumptions labelled MADE UP), 7 a shuffled-outcome sanity check.

## Evaluation rules
- Amazon: 30% of each person's ratings are test (column `split`). Person features use
  training ratings only, and leave the row's own rating out for training rows
  (`shared.data.person_features`). Never build a feature from a test rating.
- Kaggle: random 70/30 split of perfumes. No per-user data exists; do not pretend it
  personalises.
- Baselines as the brief defines them: the average (regression), the most common class
  (tree), one group (clustering). For value, the no-model product is "show the best-rated perfume".
- Report results honestly, including when a model does not beat the baseline. Give a bootstrap
  interval for small differences.
- Synthetic data, if added later, must say SYNTHETIC on the first output line, and the
  generator must not have the same form as any model (see `docs/v2_plan.md`).

## Working style
- Run a script after changing it and update its `output.txt` and `RESULTS.md`.
- Do not push to main; work on a branch and open a PR for Paul.
- The team writes the slides; code, charts and printed numbers only.
