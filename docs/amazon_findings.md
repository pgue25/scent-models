# Amazon Reviews 2023: fragrance subset (checked 2026-11-26)

Source: McAuley Lab, Amazon Reviews 2023, files from mcauleylab.ucsd.edu (same data as the
Hugging Face page). Free, public, no sign-in. Covers reviews from 2003 to Sept 2023.

How it was pulled (scripts in this folder):
- Products: `meta_Beauty_and_Personal_Care` + `meta_All_Beauty`, kept items whose category
  path contains "Fragrance" (or, with no category, a perfume/cologne title). `meta_filter.py`
- Reviews: streamed `Beauty_and_Personal_Care` (23.9M reviews) and `All_Beauty` (0.7M), kept
  reviews of those products, de-duplicated on (reviewer, product). `rev_filter.py`

| | count |
|---|---|
| fragrance products | 54,992 (women's EdP 15k, women's EdT 7k, men's EdT 7k, colognes, body sprays, sets) |
| reviews | 1,057,403 |
| reviewers | 828,849 |
| reviewers with 2+ fragrances | 130,416 |
| **reviewers with 3+ fragrances** | **42,039** (182,216 reviews) |
| reviewers with 5+ | 10,504 |
| reviewers with 10+ | 1,646 |

Ratings: 70% are 5 stars, 12% are 1 star, the rest in between. Among 3+ reviewers, 42% gave
every product 5 stars. A person's average on their other products alone cuts error from 0.94
to 0.72 stars, so "how generous is this person" is a big part of any rating.

Scent words in titles/descriptions/features (products mentioning each): musk 9.8k, floral 8.6k,
amber 7.3k, jasmine 7.3k, fresh 7.2k, vanilla 6.8k, sandalwood 6.1k, rose 6.1k, bergamot 5.6k,
orange 4.9k, patchouli 4.8k, woody 4.7k, citrus 3.9k, cedar 3.5k, fruity 3.4k, lavender 3.0k,
lemon 2.8k ... 11k products list "top/heart/base notes" in the text. About two thirds of the
products rated by 3+ reviewers can be tagged with at least one scent family from text alone.

Caveats:
- A star rating mixes scent with price, shipping, fakes, bottle size and gifts. The review text
  can filter some of this (e.g. drop reviews that never mention the smell).
- Mostly perfumes and colognes. Candles are in Home & Kitchen; not pulled yet.
- No quiz answers. It grounds rating behaviour and which scent families go together for the
  same person, not the link from everyday smells to products.
- Not checked yet: whether a person's ratings of one scent family predict their ratings of
  that family elsewhere. A first quick try had a leakage bug, so no number is reported.

## Scent filter (run 2026-11-26, `scent_filter.py`, output `amazon_scent_filtered.csv`)

Keyword rules on review title + text: keep if it mentions the smell (smell, scent, notes,
sweet, lasts, longevity, projection, fades, musky, floral, vanilla, dry down, ...); drop if it
mentions shipping, delivery, packaging, price, broken, leaks, fake, counterfeit, refund,
return, seller, sprayer or cap.

| step | reviews | reviewers with 3+ |
|---|---|---|
| all fragrance reviews | 1,057,403 | 42,039 |
| mentions the smell | 669,197 | 22,442 |
| and no shipping/price/damage/fake words | 503,622 | 14,687 |
| drop reviewers who gave all 5s | 41,537 (by 3+ reviewers) | 9,240 |
| also drop anyone who gave one value to everything | **40,522** | **8,927** |

After the filter: 8,927 people, 40,522 reviews, 12,143 products; median 3 reviews per
person, mean 4.5.
- Stars: 1 = 9%, 2 = 7%, 3 = 13%, 4 = 20%, 5 = 51% (before: 70% were 5s).
- Centered on each person's own average: sd 1.10 stars; middle half from -0.67 to +0.67.
- People's own averages differ with sd 0.75; that is 30% of rating variance.
- 25% of smell-related reviews also mentioned shipping/price/damage/fakes and were dropped.

Caveats: keyword rules are rough (a review saying "smells great, cap broke" is dropped).
Dropping all-5 raters keeps the pickier people, so this sample is harsher than Amazon overall.
With a median of 3 reviews per person, see-7-predict-3 does not fit here; use see-(n-1)-
predict-1 or keep the 1,098 people with 7+ reviews (424 have 10+).

## Fragrantica match (run 2026-11-26, `match_fragrantica.py`, output `amazon_fragrantica_match.csv`)

Paul's Kaggle archive has two files: `fra_perfumes.csv` (69,948 perfumes, 1,749 brands: name,
gender, average rating, rating count, up to 10 main accords, description with top/middle/base
notes) and `fra_cleaned.csv` (24,063 of the same perfumes, notes split into columns).
**No per-user data**: only averages, so it gives product notes, not who likes what.

Matching rule: find a Fragrantica brand in the Amazon title or store, then the longest
Fragrantica perfume name (from its URL) that appears in the rest of the title.

| | products | matched |
|---|---|---|
| all Amazon fragrance products | 54,995 | 23,301 (42%) |
| products in the scent-filtered reviews | 12,143 | 5,540 (46%) |
| scent-filtered reviews | 40,522 | 50% |

Unmatched products: 3,578 have no known brand in the title (generic, dupes, body sprays),
3,025 have the brand but not the perfume name. Of matched products, 75% have a top/middle/base
note list; all have accords.

Accuracy: in a hand check of 30 random matches, about 27 were right. Typical errors: a
flanker matched to the original ("Black XS Los Angeles" to "Black XS"), or a short name
matched inside a longer title ("Tommy Tropics" to "Tommy").

After matching: **3,054 reviewers with 3+ matched, smell-focused reviews** (12,847 reviews);
264 with 7+.
