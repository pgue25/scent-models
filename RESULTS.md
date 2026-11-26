# Results (first run, 2026-11-26)

All numbers come from the `output.txt` next to each script. Data is real (Amazon reviews and
Fragrantica), not synthetic.

## Amazon: 264 people, 2,789 ratings, 30% of each person's ratings held out

| model | test result | vs baseline |
|---|---|---|
| regression (stars) | MAE 0.81 stars, R² 0.13 | baseline (average) MAE 0.89, R² 0.00. Better, 95% CI on the gain 0.06 to 0.13 stars |
| regression without any scent features | MAE 0.79, R² 0.15 | as good as the full model: **scent taste adds nothing measurable yet** |
| decision tree (liked 4-5) | accuracy 0.57, balanced accuracy 0.63 | baseline "always liked" accuracy 0.80, balanced 0.50. The tree separates the classes a little but loses on plain accuracy |
| clustering (taste personas) | train elbow k = 2, test elbow k = 7 (different); silhouette 0.20 | 2 personas: fresh / green / aromatic vs sweet / powdery / musky |
| top pick liked (value) | regression 0.81, tree 0.82, personas 0.78 | most-popular baseline 0.81. No model beats "show the best-rated perfume" yet |

What drives Amazon ratings: mostly the person's own average (how generous they are), then the
perfume's popularity. The scent-taste features rank a few accords (powdery, white floral,
fresh spicy) but the gains are within noise.

## Kaggle / Fragrantica: 24,486 perfumes with 50+ ratings (product level)

| model | test result | vs baseline |
|---|---|---|
| regression (avg rating) | MAE 0.197, R² 0.08 | baseline MAE 0.207. Small but real; ratings only spread 0.26 stars |
| decision tree (4.0+) | accuracy 0.59, balanced 0.59 | baseline 0.52. Stocking the tree's picks: 58% are 4.0+ vs 48% overall |
| clustering (scent families) | train and test elbow both k = 2; silhouette 0.12 | woody / aromatic / citrus / fresh spicy vs sweet / fruity / floral / powdery |

Accords linked to higher ratings: amber, vanilla, earthy, warm spicy. Lower: aquatic, fruity, musky.

## What this means for the quiz

Both datasets find the same main split, on different data: **fresh / green / aromatic / citrus
vs sweet / powdery / musky / floral**. That is the first thing the quiz must measure. In the
survey draft that maps to items like ocean breeze, fresh-cut grass, mint and orange peel on
one side, and vanilla, caramel, baby powder and roses on the other, plus the forced-choice
pairs that cross them.

## Honest caveats

- Amazon stars mix scent with price, gifts and delivery; the text filter removes some of this.
- 264 people is small. Differences under about 0.04 in "top pick liked" are noise.
- None of these models uses quiz answers yet. They show which scent directions matter, which
  is what the new survey has to measure; the app model needs real survey responses.

---

# Rerun with generosity controlled (2026-11-26)

Scripts: `regression/amazon/run_centered.py`, `tree/amazon/run_centered.py`,
`clustering/amazon/run_centered.py` (outputs in `output_centered.txt`).

How: each person's **normal** is their average stars on training ratings, pulled toward the
overall average (4.16) by 3 pseudo-ratings. The target is **stars minus normal**; the tree
predicts **above normal or not** (64% of training rows are above). Taste scores use the same
centered stars and leave the predicted rating out. Test ratings never feed any feature.
Scoring: rank each person's held-out perfumes and check the top one against their real stars
(154 people have 2+ held-out perfumes with different stars). Baseline: rank by the perfume's
Fragrantica average rating. Ties are broken at random and averaged.

| ranking each person's held-out perfumes | top pick is their favourite | top pick liked (4-5) | rank correlation |
|---|---|---|---|
| baseline: perfume's average rating | 0.60 | 0.73 | 0.20 |
| regression, no scent (popularity only) | **0.67** | **0.79** | **0.23** |
| regression, with scent | 0.61 | 0.74 | 0.17 |
| tree, with scent | 0.49 | 0.69 | -0.03 |
| tree, with scent, balanced classes | 0.54 | 0.72 | 0.03 |
| personas (k = 2) | 0.44 | 0.64 | -0.10 |

- **Regression:** with scent, +0.01 over baseline on "top pick is favourite" (95% CI -0.10 to
  +0.11): no gain. The version without scent does better than the one with scent (-0.06, CI
  -0.13 to 0.00). What helps is how many people rated the perfume on Fragrantica (popularity),
  not scent taste.
- **Tree:** the plain tree predicts "above normal" for everyone (accuracy 0.68 = baseline,
  balanced accuracy 0.50). Weighting the classes makes it split on floral, white floral, rose,
  woody and powdery taste: balanced accuracy 0.58, but it ranks perfumes worse than the baseline.
- **Clustering:** 2 personas again (citrus / fresh spicy / white floral vs powdery / sweet /
  vanilla), silhouette 0.22. Train elbow k = 2, test k = 3, and the personas only half come back
  when people are resampled (adjusted Rand 0.48). Ranking by persona is worse than the baseline
  (-0.16 on "top pick is favourite", CI -0.25 to -0.07).

**Bottom line:** once generosity is removed, nothing here shows that a person's past scent
taste (as measured by Fragrantica accords on Amazon purchases) predicts which perfume they will
rate higher. Popularity does. The fresh vs sweet split keeps appearing, so it is a real
difference between people, but on this data it does not turn into better picks. That is not
proof the quiz idea fails: the flaws below could hide a real effect. It does mean the case for
scent-based recommendations has to come from our own survey.

## Every flaw that weakens this experiment

1. **Small sample.** 264 people, 820 test ratings, and only 154 people can be ranked. Most
   have 2 or 3 held-out perfumes, so "top pick is favourite" is often a coin flip between two.
   Differences under about 0.10 are inside the noise.
2. **Few ratings per person.** Median 8 ratings, so a taste score per accord rests on 1 to 3
   perfumes. That noise alone could hide a real taste effect.
3. **Amazon stars are not scent ratings.** They mix in price, gifts, bottle size, delivery and
   fakes. The keyword filter removes some, not all.
4. **Keyword filter errors.** Simple word rules: "smells great, cap broke" is dropped, and a
   review about shipping that says "fresh" may stay.
5. **Matching errors.** About 1 in 10 product matches is wrong (often a flanker matched to the
   original), and only 46% of products matched, so the people left are ones who buy well-known
   brands.
6. **Selection toward picky, frequent reviewers.** Dropping all-5 raters and requiring 7+
   reviews keeps unusual people: heavy perfume buyers who vary their stars.
7. **Buyers, not quiz takers.** These people chose to buy each perfume, so most purchases
   already fit their taste. That squeezes the range of stars (58% are 5) and leaves little for
   taste to explain. Quiz takers rating perfumes they did not choose would vary more.
8. **Accords are coarse.** Top-5 Fragrantica accords, 20 kinds, as yes/no. Two "woody"
   perfumes can smell very different. Notes were not used.
9. **No everyday-smell answers.** None of this tests the quiz link (smell answers to products),
   only whether past perfume ratings predict future ones.
10. **The baseline uses outside data.** Fragrantica's average rating comes from many raters,
    while the person-level models learn from 1,969 ratings. That favours the baseline.
11. **One random split.** Results are from seed 0 only; another split could move numbers by a few points.
12. **Shrinkage choice.** The pull toward the overall average (3 pseudo-ratings) was chosen,
    not tuned.
13. **Time ignored.** Ratings span 2003 to 2023 and tastes change; the split is random, not by date.
