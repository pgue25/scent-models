# CLAUDE.md: Smell Optimizer model v2

Status: DRAFT v2 (updated 2026-11-26 with Paul's answers). Lines marked `OPEN (Paul)` are still
waiting on a decision. Do not guess them; ask.

Decided: one v2 repo with a folder per model; the quiz is everyday smells and the hidden
targets are products; outcome is a 1 to 5 rating; real responses will be collected; the quiz gets redesigned
to separate people; evaluation hides 3 of each test person's 10 ratings (section 6).

## 1. What this repo is

Standalone model repo for Smell Optimizer (NYU Stern TECH-UB 24, Group 2). A new user answers
9 everyday-smell questions (1 to 6 scale, no neutral) and optionally names products they love.
The product shows them the top 3 products they are most likely to like.

Business question (same for every model): **"Which 3 products should we show this new quiz
taker so they like what they see and buy through our affiliate link?"** Every model in v2 is
judged on that question, on the same test people, in the same units, so they can be compared.

The app lives in a different repo (jl17500/pipai-group-2). Do not touch it. Integration later
is one function: smell answers in, top products out, fitted model loaded from a file.

## 2. Structure to keep (inherited from v1)

v1 lives in jl17500/smell-model. Keep its layout and add to it; do not rewrite it.

| path | role |
|---|---|
| `smell/taxonomy.py` | 10 scent nodes; quiz smell to node and product note to node maps |
| `smell/data.py` | Google Form CSV to `Responses(quiz, ratings, favorite)`; catalog loader |
| `smell/models.py` | recommenders with `fit(train, nodes)` and `predict(resp)` returning respondent x product predicted ratings |
| `smell/evaluate.py` | split by person, MAE, liked@k, paired bootstrap, CV hyperparameters |
| `smell/simulate.py` | v1 synthetic generator (keep as is, for reproducing v1) |
| `model_v1.py` | v1 standalone script (keep as is) |

New in v2:

| path | role |
|---|---|
| `smell/simulate_v2.py` | grounded synthetic generator (section 4) |
| `smell/grounding/` | scripts that turn external sources into generator parameters, plus `params.json` with a source for every number |
| `smell/models_v2.py` | `PersonaClusterV2`, `RatingRegression`, `LikeTree` (section 5) |
| `model_v2_cluster.py`, `model_v2_regression.py`, `model_v2_tree.py` | one standalone script per model, same 7 printed sections as `model_v1.py` |
| `compare_v2.py` | all models + baseline on the same split, one table and one value chart |
| `reports/v2/<model>/` | `output.txt` and charts written by each script |

Repo: Paul will create a new repo (he owns it; Claude cannot create GitHub repos).
Decided (Paul, 2026-11-26): **one new v2 repo with a folder per model**, all importing
the same `smell/data.py`, `smell/evaluate.py` and generator. Three separate repos would each
carry their own copy of the data, split and metrics, and the copies drift, so "which model
wins" stops being a fair comparison. Trial and error happens as folders or branches in that
one repo. Copy v1's `smell/` package in as the starting point (credit jl17500/smell-model).

```
smell-model-v2/
  smell/            shared: taxonomy, data, evaluate, simulate_v2, grounding/
  models/cluster/   model.py + run.py
  models/regression/
  models/tree/
  quiz/             quiz redesign analysis (section 4c)
  compare_v2.py
  reports/v2/
```
The paths in the table above map onto this layout (`model_v2_cluster.py` becomes
`models/cluster/run.py`, and so on). Do not push until Paul has created the repo and asked.

## 3. Hard rules

1. **Generator and models are independent.** Nothing under `smell/models*.py`, `model_v2_*.py`
   or `compare_v2.py` may import `simulate*.py`, `smell/grounding/`, or read `_truth/`.
   Models only see what a real Google Form export would contain. Add a test that greps for this.
2. **The generator must not have the same functional form as any model.** v1 ratings are
   `taste @ product_profile + noise`, which is exactly a linear regression on quiz x node
   interactions, so a regression would just recover the formula and look great for no reason.
   The v2 generator uses an ideal-point (distance) utility with saturation, a per-person
   leniency, and hidden product traits the notes do not capture (section 4). Write down in
   `reports/v2/README.md` which model, if any, is closest to the generator, and why.
3. **Split by person, never by rating row.** 70/30 random split of respondents, seed 0. Each
   person's ratings stay on one side. Regression and tree rows are person x product, so use a
   group split. Hyperparameters (k, ridge alpha, tree depth) are picked by CV on training
   people only.
4. **Synthetic results are a pipeline test, not validation.** Every output that comes from
   synthetic data says SYNTHETIC in its first line. The course requires validation on real
   data; see section 6.
5. **Report the ceiling.** On synthetic data, also score an oracle that uses the hidden true
   utilities. If a model gets close to the oracle, suspect leakage before celebrating.
6. **Sanity checks every run:** shuffle the target within the training set and refit; the model
   must fall back to about baseline. Print that result.
7. Reproducible: fixed seeds, `requirements.txt` pinned, each script runs in under 2 minutes
   and reproduces every number that goes on a slide.
8. The team writes the slides. Code, charts and printed numbers only; no slide text.

## 4. Data

### 4a. Inputs and answer key (unchanged from v1)

- Features: 9 everyday-smell answers (1 to 6), optional loved products (free text).
- Answer key: "Rate the products you have smelled" grid, 1 to 5, blank if not smelled.
  Liked = 4 or 5. Ratings are never an input for a new user.
- Catalog: `data/real/catalog.csv` (16 survey products). Team decision: notes stay as they are.

### 4b. Grounded synthetic generator (`smell/simulate_v2.py`)

Goal: the people are fake, but the parameters that shape them come from real sources, each
cited in `smell/grounding/params.json` (`value`, `source`, `how derived`).

Amazon reviews (checked 2026-11-26): 42,039 reviewers rated 3+ fragrances, so they carry real
person-level taste structure, not only product numbers. They can also serve as a second real
test of the see-7-predict-3 regression (ratings only, no quiz). The link from everyday smells
to products still has to come from our own survey. `OPEN (Paul)`: confirm the set. Recommended: Amazon reviews +
Arshamian, then refit on our real responses as they arrive. Candidates:

| source | what it grounds | caveat |
|---|---|---|
| Our own real responses (even 20 to 40) | everything: answer distributions, rating spread, correlations between smells | small; see section 6 |
| Arshamian et al. 2022, *Current Biology* (odor pleasantness across 9 cultures) | how much of liking is personal vs shared vs noise | single molecules, not perfumes |
| Keller & Vosshall 2016 / DREAM olfaction data (about 49 people x 476 odors, pleasantness + descriptors like sweet, fruit, flower, wood, spice, bakery) | person-level spread, and which descriptor likes go together (correlation matrix for personas) | lab odors, not products |
| Fragrantica scrapes on Kaggle (accords, average rating, vote counts) | product profiles, product quality, popularity (who has smelled what) | aggregate only, no individual people |
| Amazon Reviews 2023 (McAuley lab), fragrance subset: 55k products, 1.06M reviews, 42k reviewers with 3+ fragrances (see `amazon/FINDINGS.md`) | rating distribution, leniency, product popularity, and which scent families the same person likes together | no quiz answers; stars mix scent with price, shipping and fakes; 42% of 3+ reviewers gave all 5s |

All five are from memory and need their links and licences confirmed before use.

How the parameters feed the generator:
- Persona structure: sample tastes from a multivariate normal whose correlation matrix comes
  from the descriptor correlations, not from 5 hand-picked personas. Whether discrete personas
  exist at all then becomes something the clustering model has to discover.
- Share of liking that is personal vs shared vs noise: from Arshamian.
- Rating distribution (share of 1s to 5s), leniency spread, and the correlation between scent
  families within a person: from Amazon 3+ reviewers, tagging products by note words in their text.
- Product quality and popularity: from Amazon average ratings and review counts (Fragrantica
  vote counts as a cross-check).
- Products rated per person: from our real responses.
- Utility: `-distance(taste, product_profile)` with saturation, plus quality, plus hidden
  product traits; quiz answers are a noisy, biased read of taste (keep v1's familiarity, bias
  and straight-lining effects).

Output format is identical to a Google Form export, so `smell/data.py` loads it unchanged.

### 4c. Quiz redesign: questions that separate people (`quiz/`)

Paul's concern: some smells may be liked by the same people (vanilla with cinnamon, rain with
clean laundry), so several questions measure one thing twice, while other tastes are not
measured at all. Then no question splits people into distinct groups. Goal: a short quiz where
each question tells us something the others do not.

What makes a good question:
1. **People disagree on it.** A smell nearly everyone likes (clean laundry) tells us little.
   Keep items with high spread across people.
2. **It is not a copy of another question.** If two items correlate above about 0.7 across
   people, keep one.
3. **The set covers distinct taste directions.** Factor analysis (or PCA) of the answers shows
   the underlying directions (for example sweet/warm vs fresh/clean vs floral/fruity). Aim for
   2 items per direction: one is noisy, two are reliable, three are wasted.
4. **It predicts product ratings.** An item that splits people but never predicts what they
   rate is useless for recommendations. Score each item by how much product MAE worsens when
   it is dropped.
5. **It cancels the "likes everything" effect.** Test a few forced-choice pairs ("rain or
   vanilla?") alongside the 1 to 6 scale; pairs remove leniency by design.

Process:
- Draft a candidate pool of 15 to 20 everyday smells spanning the 10 nodes, using the smell
  studies (Keller/DREAM, Arshamian) to favour smells whose pleasantness varies most between people.
- Pilot the pool on the first real respondents (target 30+). Compute item spread, the
  item-item correlation matrix, factors, and drop-one product MAE. Print all of it to
  `reports/v2/quiz/`.
- Choose the final 9 to 10 items by those numbers, then rerun the clustering on the pilot to
  check whether clearer groups appear (silhouette, stability).
- Synthetic data cannot answer this: in the generator, item correlations are whatever we set.
  Use synthetic data only to test that the analysis code runs.

## 5. Models

All three share one interface (`fit(train, nodes)`, `predict(resp)` returning predicted
ratings or like-probabilities per respondent x product) so `evaluate.py` and the decision rule
work for every model.

Baselines, as the brief defines them:
- clustering: one group (every product's average rating), same as v1;
- regression: the average rating;
- decision tree: the most common class.
For the business comparison, every model is also compared against the v1 popularity ranking,
because that is what the product would actually do without a model.

### 5a. Clustering v2 (`PersonaClusterV2`)

Why v1's elbow disagreed (k = 3 on train, 4 on test): the curves are smooth with no sharp bend,
so the "biggest drop in slope" rule is choosing between near-ties, and 36 test people make the
test curve noisy. The quiz is part of it, but so is the v1 generator: 5 personas with wide
spread and noisy answers do not form well-separated clusters.

Changes to try, one at a time, each reported against v1:
1. Cluster on standardized raw answers (and on PCA of them) instead of the 10-node projection,
   which is a fixed linear map of 9 answers and adds no information.
2. Pick k with more than the elbow: silhouette, and stability (adjusted Rand index of
   assignments across 50 bootstrap refits). Still produce the train elbow and the test elbow
   from trained centers, because the brief asks for them.
3. Remove each person's average answer before clustering (people who rate everything high).
4. Report agreement with the hidden personas (synthetic only, diagnostics only).
If no k is stable, say so; "the data has no clear segments" is a valid finding.

### 5b. Regression (`RatingRegression`)

Outcome (decided): **a person's 1 to 5 rating of an item.** This is Paul's "formula that
guesses whether someone will like something".
- Rows: one per (person, hidden item). Features: the person's quiz answers and 7 visible
  ratings (centered on their own average), the hidden item's node weights, their
  interactions, and the person's average rating.
- Model: ridge regression, alpha by grouped CV. Show the top coefficients in plain words
  ("liking orange peel adds 0.3 stars to citrus products").
- Fit stats per the brief: adjusted R^2 on train; R^2 and MAE in stars on test, vs the average.
- Decision: rank unrated products by predicted rating, show the top 3.

### 5c. Decision tree (`LikeTree`)

Outcome: the brief requires a class for a tree, so the tree predicts **liked (4 or 5) vs not**,
derived from the same 1 to 5 rating. A regression tree on the raw 1 to 5 rating is the
fallback if Paul prefers; then report the regression statistics instead.
- Same rows and raw features as the regression (no interaction terms; the tree finds them).
- Depth and min leaf size by grouped CV; expect depth 3 to 5 with about 1,000 rows.
- Fit stats per the brief: confusion matrix, accuracy, precision, recall on test, and train vs
  test accuracy, vs predicting the most common class.
- Draw the tree (depth 3 version) for the slides; it is the most explainable of the three.
- Decision: rank by predicted probability of liking, show the top 3.
- Alternative outcome if we want a per-person tree: the person's favourite scent family.

## 6. Evaluation design: see 7, predict 3

Paul's design. Each respondent rates 10 items on 1 to 5. For a test person the model sees 7
of their ratings (plus their quiz answers) and predicts the 3 it never saw.

- **Two-level split.** First the brief's 70/30 split by person (seed 0). Training people:
  all 10 ratings are used to fit. Test people: for each one, hide 3 of their 10 ratings at
  random (seeded), give the model the other 7, score the 3. Hiding items inside people the
  model has already trained on would leak, so never do that.
- Repeat the hide-3 draw 20 times per test person and average, so one lucky draw does not
  decide the result.
- **Baseline:** for each hidden item, its average rating among training people (the brief's
  "average"). Second baseline: the person's own average of their 7 visible ratings.
- **Metrics on the hidden 3:** MAE in stars, R^2, and "is the item we would rank first
  among the 3 one they rated 4 or 5". Plus each family's brief statistics (section 7).
- With under 30 real people the brief says skip the 70/30 split; then do leave-one-person-out
  and say so.

Decided: everyday smells are the quiz input and products are the hidden targets,
because the business decision is which product to show. If all 10 are smells, the model
predicts liking of smells, and a product step is still needed on top. Also: in the app, does a
new user rate 7 items, or only answer the quiz? The decision function must take exactly what
the app will collect.
The survey draft is `survey_v2_draft.md`.

### Real responses

Real responses will be collected (decided). `OPEN (Paul)`: how many, and by when?
- Real people are the test set. Train on synthetic + real training people, test on real only,
  and report synthetic-only vs real results side by side.
- Refit the generator parameters on real responses and print how far the first guesses were off.
- Until real data arrives, every result is labelled SYNTHETIC and the write-up says the
  real-data requirement is not met yet.

## 7. What each `model_v2_*.py` prints (same 7 sections as v1)

1. Data: source, rows, period, one row per what, data dictionary with the outcome marked.
2. Split: 70/30 by person, seed, counts.
3. Fit on training rows, read in business terms.
4. Test rows: baseline vs model, with the brief's statistics for that model family, plus
   liked@1, liked@3 and MAE so the three models can be compared.
5. Decision function `recommend(answers) -> top 3`, run on the same new input as v1:
   `{'Fresh roses': 2, 'Christmas tree (Evergreen)': 5, 'Air outside after summer rain': 6,
   'Freshly baked cinnamon rolls': 3, 'Vanilla candle': 2, 'Freshly sliced ginger': 4,
   'Fresh peaches': 3, 'Clean laundry': 6, 'Orange peel': 5}`
6. Value per month vs baseline, with v1's assumptions (5,000 users, 3 shown, 5% vs 0.5%
   purchase, $8 commission) labelled MADE UP.
7. Sanity checks: shuffled-target refit, oracle ceiling (synthetic only).

`compare_v2.py` prints one table (baseline, v1, cluster v2, regression, tree) with paired
bootstrap intervals for liked@3 against v1, and one value chart.

## 8. Working style

- Build in this order: shared data + evaluation (see 7, predict 3), grounding params,
  generator, quiz analysis code, cluster v2, regression, tree, compare.
  Run each script after building it and read its output before moving on.
- Small n (about 120 people): prefer simple models, report intervals, and say when a
  difference is inside the noise.
- Never push, open a PR, or change the app repo unless asked.
