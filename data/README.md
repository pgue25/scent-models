# Data

Both files are built by scripts in this folder from public sources. Raw downloads go to
`data/raw/` (git-ignored).

## amazon_clean.csv (REAL people x perfumes)

Built by `build_amazon.py` from [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/)
(McAuley Lab, UCSD) plus the Fragrantica Kaggle file for accords.

| step | reviews | reviewers |
|---|---|---|
| fragrance reviews (Beauty & Personal Care + All Beauty, category "Fragrance") | 1,057,403 | 828,849 |
| talks about the smell | 669,197 | 542,650 |
| no shipping / price / damage / fake words | 503,622 | 420,250 |
| reviewers with 3+ such reviews, not all the same stars | 40,522 | 8,927 |
| product matched to Fragrantica (46% of products) | | 3,054 |
| **reviewers with 7+ matched reviews** | **2,789** | **264** |

One row per (person, perfume). Columns: `user_id`, `parent_asin` (Amazon product),
`split` (30% of each person's ratings are `test`, seed 0), `rating` (1-5, the outcome),
`brand`, `perfume`, `gender`, `fra_rating` and `fra_count` (Fragrantica average and number of
ratings), `accords` (top 5 Fragrantica accords, `;`-separated).

Stars: 58% 5, 19% 4, 12% 3, 5% 2, 5% 1. Matching was checked by hand on 30 random products
(about 27 right; errors are mostly a flanker matched to the original perfume).
Guessing accords from Amazon text was only 43% accurate, so unmatched products are dropped.

## kaggle_products.csv (REAL, perfumes only)

Built by `build_kaggle.py` from the Kaggle Fragrantica archive (`fra_perfumes.csv`, put in
`data/raw/`). 63,769 perfumes: brand, perfume, gender, average rating, number of ratings,
top-5 accords, and top/middle/base notes. **No per-user data**: models on it are product-level.
Models use perfumes with 50+ ratings (24,486). Check the Kaggle dataset's licence before
sharing this file outside the class.
