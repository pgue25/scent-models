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
